"""
training.py

Dataset, collate, and single-fold training for the mamba pipeline.

Compatible with pooled (D,) and sequence (T, D) embeddings returned by
feature_extractors.LazyEmbeddings. Uses torch 2.6's new torch.amp API.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from data_utils import aggregate_to_unit, save_oof_predictions


# =========================================================================
# Shape helpers
# =========================================================================
def _ensure_2d(x: np.ndarray) -> np.ndarray:
    """(D,) -> (1, D); (T, D) -> unchanged."""
    return x[None, :] if x.ndim == 1 else x


# =========================================================================
# Dataset
# =========================================================================
class MultiModalDataset(Dataset):
    """
    One sample per row of df. Emits dicts with audio/text/tabular/label.
    Auto-unsqueezes pooled vectors to (1, D) so downstream code always
    sees 2-D sequences.
    """

    def __init__(self, df, audio_emb, text_emb, tabular, cfg):
        self.df = df.reset_index(drop=True)
        self.audio_emb = audio_emb
        self.text_emb = text_emb
        self.tabular = np.asarray(tabular, dtype=np.float32)
        self.cfg = cfg
        self.task = cfg.task

        self.label2idx = None
        self.idx2label = None
        if self.task == "classification":
            labels = sorted(self.df["label"].unique().tolist())
            self.label2idx = {l: i for i, l in enumerate(labels)}
            self.idx2label = {i: l for l, i in self.label2idx.items()}

    def __len__(self):
        return len(self.df)

    def _label_for(self, row):
        if self.task == "classification":
            return torch.tensor(self.label2idx[row["label"]], dtype=torch.long)
        return torch.tensor(float(row["label"]), dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        stem = row["file_stem"]

        # Guard against stems missing from caches (fail early, clearly).
        if stem not in self.audio_emb:
            raise FileNotFoundError(
                f"audio features missing for {stem} — pre-filter df before "
                f"building the dataset")
        if stem not in self.text_emb:
            raise FileNotFoundError(
                f"text features missing for {stem} — pre-filter df before "
                f"building the dataset")

        a = np.asarray(self.audio_emb[stem], dtype=np.float32)
        a = _ensure_2d(a)
        am = np.ones(a.shape[0], dtype=np.float32)

        t = np.asarray(self.text_emb[stem], dtype=np.float32)
        t = _ensure_2d(t)
        tm = np.ones(t.shape[0], dtype=np.float32)

        return {
            "file_stem":  stem,
            "audio_seq":  torch.from_numpy(a),
            "audio_mask": torch.from_numpy(am),
            "text_seq":   torch.from_numpy(t),
            "text_mask":  torch.from_numpy(tm),
            "tabular":    torch.from_numpy(self.tabular[idx]),
            "label":      self._label_for(row),
        }


# =========================================================================
# Collate
# =========================================================================
def collate_pad(batch):
    B = len(batch)
    stems = [b["file_stem"] for b in batch]

    da = batch[0]["audio_seq"].size(1)
    dt = batch[0]["text_seq"].size(1)
    dz = batch[0]["tabular"].size(0)

    max_a = max(b["audio_seq"].size(0) for b in batch)
    max_t = max(b["text_seq"].size(0) for b in batch)

    A  = torch.zeros(B, max_a, da)
    AM = torch.zeros(B, max_a)
    T  = torch.zeros(B, max_t, dt)
    TM = torch.zeros(B, max_t)
    Z  = torch.zeros(B, dz)

    for i, b in enumerate(batch):
        a, am = b["audio_seq"], b["audio_mask"]
        t, tm = b["text_seq"], b["text_mask"]
        A[i, :a.size(0)] = a
        AM[i, :am.size(0)] = am
        T[i, :t.size(0)] = t
        TM[i, :tm.size(0)] = tm
        Z[i] = b["tabular"]

    labels = torch.stack([b["label"] for b in batch])

    return {
        "file_stem":  stems,
        "audio_seq":  A,
        "audio_mask": AM,
        "text_seq":   T,
        "text_mask":  TM,
        "tabular":    Z,
        "label":      labels,
    }


# =========================================================================
# Forward adapter
# =========================================================================
def _forward(model, batch, device):
    a  = batch["audio_seq"].to(device)
    am = batch["audio_mask"].to(device)
    t  = batch["text_seq"].to(device)
    tm = batch["text_mask"].to(device)
    z  = batch["tabular"].to(device)

    out = model(a, am, t, tm, z)

    if isinstance(out, tuple):
        logits = out[0]
        aux    = out[1] if len(out) > 1 else None
        aux2   = out[2] if len(out) > 2 else None
    else:
        logits, aux, aux2 = out, None, None

    return logits, aux, aux2


# =========================================================================
# Re-home any module buffers that ended up on the wrong device.
# Fixes: "Expected x.is_cuda() to be true" from mamba-ssm.
# =========================================================================
def _rehome_buffers(model, device):
    for name, buf in list(model.named_buffers(recurse=True)):
        if buf is not None and buf.device != device:
            # find the owning submodule and re-register
            parts = name.split(".")
            owner = model
            for p in parts[:-1]:
                owner = getattr(owner, p) if not p.isdigit() else owner[int(p)]
            owner.register_buffer(parts[-1], buf.to(device))


# =========================================================================
# Single-fold training
# =========================================================================
def train_one_fold(model, train_loader, val_loader, cfg,
                   val_df=None, verbose=True):
    device = torch.device(cfg.device)
    model = model.to(device)

    if device.type == "cuda":
        _rehome_buffers(model, device)

    if cfg.task == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=getattr(cfg, "lr", 1e-4),
        weight_decay=getattr(cfg, "weight_decay", 1e-2),
    )

    use_amp = bool(getattr(cfg, "use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "val_loss": []}

    # ---- train ----
    model.train()
    for epoch in range(int(cfg.epochs)):
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _, _ = _forward(model, batch, device)
                y = batch["label"].to(device)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=getattr(cfg, "grad_clip", 1.0))
            scaler.step(opt)
            scaler.update()

            running += float(loss.item())
            n_batches += 1

        avg = running / max(n_batches, 1)
        history["train_loss"].append(avg)
        if verbose:
            print(f"    epoch {epoch+1}/{cfg.epochs}  train_loss={avg:.4f}")

    # ---- val ----
    model.eval()
    all_stems, all_logits, all_probs, all_labels = [], [], [], []

    with torch.no_grad():
        for batch in val_loader:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _, _ = _forward(model, batch, device)
            logits = logits.float().cpu().numpy()
            all_stems.extend(batch["file_stem"])
            all_logits.append(logits)
            all_labels.extend(batch["label"].cpu().numpy().tolist())
            if cfg.task == "classification":
                p = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
                all_probs.append(p)

    logits_arr = (np.concatenate(all_logits, axis=0)
                  if all_logits else np.zeros((0, 0)))

    # ---- OOF ----
    try:
        out_dir = Path(cfg.output_dir) / "oof"
        fold_i = int(getattr(cfg, "_current_fold", 0))
        model_name = str(getattr(cfg, "_current_model_name", "model"))

        if val_df is not None:
            df_eval = (val_df.set_index("file_stem")
                       .loc[all_stems].reset_index())
        else:
            df_eval = pd.DataFrame({"file_stem": all_stems})

        if cfg.task == "classification":
            save_oof_predictions(
                out_dir=out_dir, model_name=model_name, fold_i=fold_i,
                df_eval=df_eval, task=cfg.task,
                probs=np.concatenate(all_probs, axis=0),
                logits=logits_arr)
        else:
            yp = logits_arr.squeeze(-1) if logits_arr.ndim > 1 else logits_arr
            save_oof_predictions(
                out_dir=out_dir, model_name=model_name, fold_i=fold_i,
                df_eval=df_eval, task=cfg.task, y_pred=yp)
    except Exception as e:
        print(f"    [OOF] save failed: {e}")

    # ---- aggregate to unit ----
    if cfg.task == "classification":
        per_file_pred = np.concatenate(all_probs, axis=0)
        df_pred = pd.DataFrame({"file_stem": all_stems})
        for c in range(per_file_pred.shape[1]):
            df_pred[f"prob_{c}"] = per_file_pred[:, c]
        df_pred["y_true"] = np.asarray(all_labels)
        df_pred["y_pred"] = per_file_pred.argmax(axis=1)
        if val_df is not None:
            keep = [c for c in ("speaker_id", "session_id", "question_id")
                    if c in val_df.columns]
            df_pred = df_pred.merge(
                val_df[["file_stem"] + keep], on="file_stem", how="left")
        agg = aggregate_to_unit(df_pred, cfg.resolved_aggregation_unit)
        prob_cols = [c for c in agg.columns if c.startswith("prob_")]
        agg_preds = agg[prob_cols].to_numpy(dtype=float).argmax(axis=1)
        agg_labels = agg["y_true"].to_numpy()
        agg_keys = agg.index.tolist()
    else:
        per_file_pred = (logits_arr.squeeze(-1)
                         if logits_arr.ndim > 1 else logits_arr)
        df_pred = pd.DataFrame({
            "file_stem": all_stems,
            "y_true":    np.asarray(all_labels, dtype=float),
            "y_pred":    per_file_pred,
        })
        if val_df is not None:
            keep = [c for c in ("speaker_id", "session_id", "question_id")
                    if c in val_df.columns]
            df_pred = df_pred.merge(
                val_df[["file_stem"] + keep], on="file_stem", how="left")
        agg = aggregate_to_unit(df_pred, cfg.resolved_aggregation_unit)
        agg_preds = agg["y_pred"].to_numpy(dtype=float)
        agg_labels = agg["y_true"].to_numpy(dtype=float)
        agg_keys = agg.index.tolist()

    return agg_preds, agg_labels, agg_keys, history