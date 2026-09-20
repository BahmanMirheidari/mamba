"""
training.py

Dataset, collate, and single-fold training for the mamba pipeline.

Compatible with:
  - pooled  (D,)     or sequence (T, D) embeddings returned by
    feature_extractors.LazyEmbeddings
  - classification   and regression tasks
  - torch 2.6 (new torch.amp.* API; no deprecated torch.cuda.amp.*)

Exposes:
  MultiModalDataset   torch Dataset; emits dicts with audio/text/tab/label
  collate_pad         pads variable-length sequences in a batch
  train_one_fold      full training + validation loop for a single fold
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


def _ensure_mask(m: np.ndarray, length: int) -> np.ndarray:
    """Return a (length,) float mask. If input is empty, all ones."""
    if m is None or m.size == 0:
        return np.ones(length, dtype=np.float32)
    return m.astype(np.float32)


# =========================================================================
# Dataset
# =========================================================================
class MultiModalDataset(Dataset):
    """
    One sample per row of `df`.

    Emits a dict:
        {
          'file_stem':  str,
          'audio_seq':  FloatTensor (T_a, d_a),
          'audio_mask': FloatTensor (T_a,),
          'text_seq':   FloatTensor (T_t, d_t),
          'text_mask':  FloatTensor (T_t,),
          'tabular':    FloatTensor (d_z,),
          'label':      LongTensor scalar (classification)
                        FloatTensor scalar (regression)
        }

    Embeddings come from LazyEmbeddings views (`audio_emb`, `text_emb`).
    Pooled vectors are auto-unsqueezed to (1, D) so downstream code
    uniformly sees 2-D sequences.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 audio_emb,
                 text_emb,
                 tabular: np.ndarray,
                 cfg):
        self.df = df.reset_index(drop=True)
        self.audio_emb = audio_emb
        self.text_emb = text_emb
        self.tabular = np.asarray(tabular, dtype=np.float32)
        self.cfg = cfg
        self.task = cfg.task

        # Label → integer mapping for classification.
        # Set here so the val dataset can be pointed at this same mapping
        # after construction (ds_va.label2idx = ds_tr.label2idx).
        self.label2idx: Optional[Dict] = None
        self.idx2label: Optional[Dict] = None
        if self.task == "classification":
            labels = sorted(self.df["label"].unique().tolist())
            self.label2idx = {l: i for i, l in enumerate(labels)}
            self.idx2label = {i: l for l, i in self.label2idx.items()}

    def __len__(self) -> int:
        return len(self.df)

    def _label_for(self, row) -> torch.Tensor:
        if self.task == "classification":
            y = self.label2idx[row["label"]]
            return torch.tensor(y, dtype=torch.long)
        return torch.tensor(float(row["label"]), dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        stem = row["file_stem"]

        # --- audio ---
        a = np.asarray(self.audio_emb[stem], dtype=np.float32)
        a = _ensure_2d(a)                                    # (T_a, d_a)
        a_mask = _ensure_mask(None, a.shape[0])

        # --- text ---
        t = np.asarray(self.text_emb[stem], dtype=np.float32)
        t = _ensure_2d(t)
        t_mask = _ensure_mask(None, t.shape[0])

        # --- tabular ---
        z = self.tabular[idx]

        return {
            "file_stem":  stem,
            "audio_seq":  torch.from_numpy(a),
            "audio_mask": torch.from_numpy(a_mask),
            "text_seq":   torch.from_numpy(t),
            "text_mask":  torch.from_numpy(t_mask),
            "tabular":    torch.from_numpy(z),
            "label":      self._label_for(row),
        }


# =========================================================================
# Collate — pad variable-length sequences in a batch
# =========================================================================
def collate_pad(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Pads audio_seq and text_seq to the batch max length.
    Masks are concatenated with zeros in the padded region.
    """
    B = len(batch)
    stems = [b["file_stem"] for b in batch]

    # audio dims (works for both 1-D and 2-D inputs — dataset normalizes)
    da = batch[0]["audio_seq"].size(1)
    dt = batch[0]["text_seq"].size(1)
    dz = batch[0]["tabular"].size(0)

    max_a = max(b["audio_seq"].size(0) for b in batch)
    max_t = max(b["text_seq"].size(0) for b in batch)

    A = torch.zeros(B, max_a, da)
    AM = torch.zeros(B, max_a)
    T = torch.zeros(B, max_t, dt)
    TM = torch.zeros(B, max_t)
    Z = torch.zeros(B, dz)

    for i, b in enumerate(batch):
        a = b["audio_seq"]; am = b["audio_mask"]
        t = b["text_seq"];  tm = b["text_mask"]
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
# Model forward adapter
# =========================================================================
def _forward(model, batch, device):
    """
    Call model(a, am, t, tm, z) and normalize the output.

    Models in fusion.py and novelty.py return either:
        - a tensor                       -> (logits, None, None)
        - (logits, aux, aux2) tuple      -> unpacked
    """
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
# Single-fold training
# =========================================================================
def train_one_fold(model: nn.Module,
                   train_loader,
                   val_loader,
                   cfg,
                   val_df: Optional[pd.DataFrame] = None,
                   verbose: bool = True
                   ) -> Tuple[np.ndarray, np.ndarray, List, Dict]:
    """
    Trains `model` for cfg.epochs, evaluates on val_loader, writes OOF
    predictions, and returns (agg_preds, agg_labels, agg_keys, history).
    """
    device = torch.device(cfg.device)
    model = model.to(device)

    # ---- loss ----
    if cfg.task == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    # ---- optimizer ----
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=getattr(cfg, "lr", 1e-4),
        weight_decay=getattr(cfg, "weight_decay", 1e-2),
    )

    # ---- AMP (new API: torch.amp.GradScaler / torch.amp.autocast) ----
    use_amp = bool(getattr(cfg, "use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "val_loss": []}

    # =====================================================================
    # Training
    # =====================================================================
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
                model.parameters(),
                max_norm=getattr(cfg, "grad_clip", 1.0),
            )
            scaler.step(opt)
            scaler.update()

            running += float(loss.item())
            n_batches += 1

        avg_train = running / max(n_batches, 1)
        history["train_loss"].append(avg_train)

        if verbose:
            print(f"    epoch {epoch + 1}/{cfg.epochs}  "
                  f"train_loss={avg_train:.4f}")

    # =====================================================================
    # Validation — collect per-file predictions
    # =====================================================================
    model.eval()
    all_stems: List[str] = []
    all_logits: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_labels: List = []

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

    logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 0))

    # ---- save OOF per file ----
    try:
        out_dir = Path(cfg.output_dir) / "oof"
        fold_i = int(getattr(cfg, "_current_fold", 0))
        model_name = str(getattr(cfg, "_current_model_name", "model"))

        if val_df is not None:
            # Match rows in the same order as we collected predictions
            df_eval = val_df.set_index("file_stem").loc[all_stems].reset_index()
        else:
            df_eval = pd.DataFrame({"file_stem": all_stems})

        if cfg.task == "classification":
            save_oof_predictions(
                out_dir=out_dir,
                model_name=model_name,
                fold_i=fold_i,
                df_eval=df_eval,
                task=cfg.task,
                probs=np.concatenate(all_probs, axis=0),
                logits=logits_arr,
            )
        else:
            save_oof_predictions(
                out_dir=out_dir,
                model_name=model_name,
                fold_i=fold_i,
                df_eval=df_eval,
                task=cfg.task,
                y_pred=logits_arr.squeeze(-1) if logits_arr.ndim > 1 else logits_arr,
            )
    except Exception as e:
        print(f"    [OOF] save failed: {e}")

    # ---- aggregate to unit and return ----
    if cfg.task == "classification":
        per_file_pred = np.concatenate(all_probs, axis=0)          # (N, C)
        per_file_label = np.asarray(all_labels)                    # (N,)
        # reconstruct a DataFrame with prob_* columns for aggregate_to_unit
        df_pred = pd.DataFrame({"file_stem": all_stems})
        for c in range(per_file_pred.shape[1]):
            df_pred[f"prob_{c}"] = per_file_pred[:, c]
        df_pred["y_true"] = per_file_label
        df_pred["y_pred"] = per_file_pred.argmax(axis=1)
        if val_df is not None:
            keep_cols = [c for c in ("speaker_id", "session_id", "question_id")
                         if c in val_df.columns]
            df_pred = df_pred.merge(
                val_df[["file_stem"] + keep_cols], on="file_stem", how="left")

                
        agg_preds, agg_labels, agg_keys = aggregate_to_unit(
            df_pred, per_file_pred, cfg.task, cfg.resolved_aggregation_unit)
    else:
        per_file_pred = logits_arr.squeeze(-1) if logits_arr.ndim > 1 else logits_arr
        per_file_label = np.asarray(all_labels, dtype=float)

        df_pred = pd.DataFrame({
            "file_stem": all_stems,
            "y_true":    per_file_label,
            "y_pred":    per_file_pred,
        })
        if val_df is not None:
            keep_cols = [c for c in ("speaker_id", "session_id", "question_id")
                         if c in val_df.columns]
            df_pred = df_pred.merge(
                val_df[["file_stem"] + keep_cols], on="file_stem", how="left")

        agg_preds, agg_labels, agg_keys = aggregate_to_unit(
            df_pred, per_file_pred, cfg.task, cfg.resolved_aggregation_unit)

    return agg_preds, agg_labels, agg_keys, history