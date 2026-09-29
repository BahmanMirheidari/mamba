"""
Speaker-level dataset, collate, and training loop.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path 
from data_utils import save_oof_predictions


class SpeakerSequenceDataset(Dataset):
    """
    One sample = one speaker. All of that speaker's chunks are concatenated
    along the time dimension. Label is the speaker's label.

    audio_seq  [T_a, D_a]    concatenation of per-chunk audio embeddings
    text_seq   [T_t, D_t]    concatenation of per-chunk text embeddings
    tabular    [D_z]         speaker-mean of eGeMAPS
    label      scalar
    speaker_id str
    """

    def __init__(self, speaker_df: pd.DataFrame, cfg,
                 max_audio_T: int = 20000, max_text_T: int = 4096):
        self.df = speaker_df.reset_index(drop=True)
        self.task = cfg.task
        self.max_a = max_audio_T
        self.max_t = max_text_T

        if self.task == "classification":
            self.labels = sorted(self.df["label"].unique())
            self.label2idx = {l: i for i, l in enumerate(self.labels)}
        else:
            self.label2idx = {}

    def __len__(self):
        return len(self.df)

    def _concat_sequences(self, seqs):
        if not seqs:
            return np.zeros((1, 1), np.float32)
        arr = np.concatenate(seqs, axis=0)
        return arr.astype(np.float32)

    def __getitem__(self, i):
        row = self.df.iloc[i]

        a = self._concat_sequences(row["audio_seqs"])[:self.max_a]
        t = self._concat_sequences(row["text_seqs"])[:self.max_t]
        a_mask = np.ones(len(a), np.float32)
        t_mask = np.ones(len(t), np.float32)

        if self.task == "classification":
            label = torch.tensor(self.label2idx[row["label"]], dtype=torch.long)
        else:
            label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return {
            "audio_seq": torch.tensor(a),
            "audio_mask": torch.tensor(a_mask),
            "text_seq": torch.tensor(t),
            "text_mask": torch.tensor(t_mask),
            "tabular": torch.tensor(row["tabular"], dtype=torch.float32),
            "label": label,
            "speaker_id": row["speaker_id"],
        }


def collate_pad(batch):
    max_a = max(b["audio_seq"].size(0) for b in batch)
    max_t = max(b["text_seq"].size(0) for b in batch)
    da = batch[0]["audio_seq"].size(1)
    dt = batch[0]["text_seq"].size(1)

    out = {k: [] for k in ("audio_seq", "audio_mask", "text_seq",
                           "text_mask", "tabular", "label", "speaker_id")}
    for b in batch:
        ta = b["audio_seq"].size(0)
        a = torch.zeros(max_a, da); a[:ta] = b["audio_seq"]
        am = torch.zeros(max_a); am[:ta] = b["audio_mask"]
        tt = b["text_seq"].size(0)
        t = torch.zeros(max_t, dt); t[:tt] = b["text_seq"]
        tm = torch.zeros(max_t); tm[:tt] = b["text_mask"]
        out["audio_seq"].append(a)
        out["audio_mask"].append(am)
        out["text_seq"].append(t)
        out["text_mask"].append(tm)
        out["tabular"].append(b["tabular"])
        out["label"].append(b["label"])
        out["speaker_id"].append(b["speaker_id"])

    for k in ("audio_seq", "audio_mask", "text_seq", "text_mask",
              "tabular", "label"):
        out[k] = torch.stack(out[k])
    return out


def train_one_fold(model, train_loader, val_loader, cfg,
                   val_df: pd.DataFrame, verbose: bool = True):
    """
    Speaker-level training. Returns
        (val_probs, val_labels, val_speaker_ids, history)
    where val_probs is [n_speakers, C] (classification) or [n_speakers]
    (regression). One row per speaker, no further aggregation needed.
    """

    device = cfg.device
    task = cfg.task
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(train_loader))
    warmup = int(cfg.warmup_frac * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, 1.0 - (step - warmup) / max(1, total_steps - warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg.use_amp and device == "cuda")
    criterion = (nn.CrossEntropyLoss() if task == "classification"
                 else nn.MSELoss())

    best_val = float("inf")
    best_state = None
    best_outputs = None
    best_oof = None
    wait = 0
    history = []

    for ep in range(1, cfg.epochs + 1):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            for k in ("audio_seq", "audio_mask", "text_seq",
                      "text_mask", "tabular", "label"):
                batch[k] = batch[k].to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(
                    enabled=cfg.use_amp and device == "cuda"):
                out = model(batch["audio_seq"], batch["audio_mask"],
                            batch["text_seq"], batch["text_mask"],
                            batch["tabular"])
                logits = out[0] if isinstance(out, tuple) else out
                extras = out[1] if isinstance(out, tuple) and len(out) > 1 else {}
                if task == "classification":
                    loss = criterion(logits, batch["label"].long())
                else:
                    loss = criterion(logits.squeeze(-1), batch["label"].float())
                aux = extras.get("aux_loss") if isinstance(extras, dict) else None
                if isinstance(aux, torch.Tensor):
                    loss = loss + aux
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tr_loss += loss.item() * batch["label"].size(0)
        tr_loss /= len(train_loader.dataset)

        # ---- validate ----
        model.eval()
        va_loss = 0.0
        va_logits, va_probs, va_labels, va_speakers = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                for k in ("audio_seq", "audio_mask", "text_seq",
                          "text_mask", "tabular", "label"):
                    batch[k] = batch[k].to(device)
                with torch.cuda.amp.autocast(
                        enabled=cfg.use_amp and device == "cuda"):
                    out = model(batch["audio_seq"], batch["audio_mask"],
                                batch["text_seq"], batch["text_mask"],
                                batch["tabular"])
                    logits = out[0] if isinstance(out, tuple) else out
                    if task == "classification":
                        loss = criterion(logits, batch["label"].long())
                        probs = torch.softmax(logits, -1)
                    else:
                        loss = criterion(logits.squeeze(-1),
                                         batch["label"].float())
                        probs = logits.squeeze(-1)
                va_loss += loss.item() * batch["label"].size(0)
                va_logits.append(logits.detach().cpu().numpy())
                va_probs.append(probs.detach().cpu().numpy())
                va_labels.append(batch["label"].detach().cpu().numpy())
                va_speakers.extend(batch["speaker_id"])
        va_loss /= len(val_loader.dataset)
        va_logits = np.concatenate(va_logits, 0)
        va_probs = np.concatenate(va_probs, 0)
        va_labels = np.concatenate(va_labels, 0)

        history.append({"epoch": ep, "train_loss": tr_loss,
                        "val_loss": va_loss})
        if verbose and (ep == 1 or ep % 5 == 0):
            print(f"  ep {ep:03d}  train={tr_loss:.4f}  val={va_loss:.4f}")

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            best_outputs = (va_probs, va_labels, va_speakers)
            best_oof = (va_probs.copy(), va_logits.copy())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                if verbose:
                    print(f"  early stop at ep {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- persist speaker-level OOF ----
    if best_oof is not None and getattr(cfg, "_current_model_name", ""):
        probs_best, logits_best = best_oof
        # val_df must be aligned row-for-row with the val_loader output;
        # the loader uses shuffle=False, so the order is stable.
        vd = val_df.reset_index(drop=True)
        try:
            if task == "classification":
                path = save_oof_predictions(
                    out_dir=Path(cfg.output_dir) / "oof",
                    model_name=cfg._current_model_name,
                    fold_i=cfg._current_fold,
                    df_eval=vd, task=task,
                    probs=probs_best, logits=logits_best)
            else:
                path = save_oof_predictions(
                    out_dir=Path(cfg.output_dir) / "oof",
                    model_name=cfg._current_model_name,
                    fold_i=cfg._current_fold,
                    df_eval=vd, task=task,
                    y_pred=probs_best)
            if verbose:
                print(f"  [OOF] saved {path.name} ({len(vd)} speakers)")
        except Exception as e:
            print(f"  [OOF] save failed: {e}")

    return (*best_outputs, history)