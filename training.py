"""
Dataset, collate, and task-aware training loop.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
from data_utils import aggregate_to_unit

class MultiModalDataset(Dataset):
    def __init__(self, df, audio_emb, text_emb, tabular, cfg,
                 max_audio_T=3000, max_text_T=512):
        self.df = df.reset_index(drop=True)
        self.audio = audio_emb
        self.text = text_emb
        self.tab = tabular
        self.task = cfg.task
        self.label_col = "label"
        self.max_a = max_audio_T
        self.max_t = max_text_T

        if self.task == "classification":
            self.labels = sorted(self.df[self.label_col].unique())
            self.label2idx = {l: i for i, l in enumerate(self.labels)}
        else:
            self.label2idx = {}

    def __len__(self):
        return len(self.df)

    def _pad(self, arr, max_len):
        if arr is None or len(arr) == 0:
            return np.zeros((1, 1), dtype=np.float32), np.zeros(1)
        arr = arr[:max_len]
        mask = np.ones(len(arr), dtype=np.float32)
        return arr.astype(np.float32), mask

    def __getitem__(self, i):
        row = self.df.iloc[i]
        stem = row["file_stem"]
        a_arr, a_mask = self._pad(self.audio.get(stem), self.max_a)
        t_arr, t_mask = self._pad(self.text.get(stem), self.max_t)

        if self.task == "classification":
            label = torch.tensor(self.label2idx[row[self.label_col]],
                                 dtype=torch.long)
        else:
            label = torch.tensor(float(row[self.label_col]),
                                 dtype=torch.float32)

        return {
            "audio_seq": torch.tensor(a_arr),
            "audio_mask": torch.tensor(a_mask),
            "text_seq": torch.tensor(t_arr),
            "text_mask": torch.tensor(t_mask),
            "tabular": torch.tensor(self.tab[i], dtype=torch.float32),
            "label": label,
            "stem": stem,
        }


def collate_pad(batch):
    max_a = max(b["audio_seq"].size(0) for b in batch)
    max_t = max(b["text_seq"].size(0) for b in batch)
    da = batch[0]["audio_seq"].size(1)
    dt = batch[0]["text_seq"].size(1)

    out = {k: [] for k in ("audio_seq", "audio_mask", "text_seq",
                           "text_mask", "tabular", "label", "stem")}
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
        out["stem"].append(b["stem"])

    for k in ("audio_seq", "audio_mask", "text_seq", "text_mask",
              "tabular", "label"):
        out[k] = torch.stack(out[k])
    return out


def train_one_fold(model, train_loader, val_loader, cfg,
                   val_df: pd.DataFrame, verbose: bool = True):
    """
    Returns (agg_preds, agg_labels, unit_keys, history).
    Also writes per-fold OOF predictions to cfg.output_dir/oof/.
    """
    from pathlib import Path
    from data_utils import save_oof_predictions, aggregate_to_unit

    device = cfg.device
    task = cfg.task
    unit = cfg.resolved_aggregation_unit
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
        # ---------------- training ----------------
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
                    loss = criterion(logits.squeeze(-1),
                                     batch["label"].float())
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

        # ---------------- validation ----------------
        model.eval()
        va_loss = 0.0
        va_logits_list, va_probs_list, va_stems = [], [], []
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
                va_logits_list.append(logits.detach().cpu().numpy())
                va_probs_list.append(probs.detach().cpu().numpy())
                va_stems.extend(batch["stem"])
        va_loss /= len(val_loader.dataset)
        va_logits = np.concatenate(va_logits_list, 0)
        va_probs = np.concatenate(va_probs_list, 0)

        order = {s: i for i, s in enumerate(va_stems)}
        vd = val_df[val_df["file_stem"].isin(order)].copy()
        vd["__p"] = vd["file_stem"].map(order)
        vd = vd.sort_values("__p").reset_index(drop=True)

        # align logits/probs row-for-row with vd
        row_idx = vd["__p"].values
        aligned_logits = va_logits[row_idx]
        aligned_probs = va_probs[row_idx]

        if task == "classification":
            agg_preds, agg_labels, agg_keys = aggregate_to_unit(
                vd, aligned_probs, task, unit)
        else:
            agg_preds, agg_labels, agg_keys = aggregate_to_unit(
                vd, aligned_probs, task, unit)

        history.append({"epoch": ep, "train_loss": tr_loss,
                        "val_loss": va_loss})
        if verbose and (ep == 1 or ep % 5 == 0):
            print(f"  ep {ep:03d}  train={tr_loss:.4f}  val={va_loss:.4f}")

        # ---------------- checkpoint ----------------
        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            best_outputs = (agg_preds, agg_labels, agg_keys)
            best_oof = (vd.copy(), aligned_probs.copy(), aligned_logits.copy())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                if verbose:
                    print(f"  early stop at ep {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------------- persist OOF at best epoch ----------------
    if best_oof is not None and getattr(cfg, "_current_model_name", ""):
        vd_best, probs_best, logits_best = best_oof
        try:
            if task == "classification":
                path = save_oof_predictions(
                    out_dir=Path(cfg.output_dir) / "oof",
                    model_name=cfg._current_model_name,
                    fold_i=cfg._current_fold,
                    df_eval=vd_best,
                    task=task,
                    probs=probs_best,
                    logits=logits_best)
            else:
                path = save_oof_predictions(
                    out_dir=Path(cfg.output_dir) / "oof",
                    model_name=cfg._current_model_name,
                    fold_i=cfg._current_fold,
                    df_eval=vd_best,
                    task=task,
                    y_pred=probs_best)
            if verbose:
                print(f"  [OOF] saved {path.name} ({len(vd_best)} rows)")
        except Exception as e:
            print(f"  [OOF] save failed: {e}")

    return (*best_outputs, history) 