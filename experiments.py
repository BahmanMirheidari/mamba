"""
Experiment registry and runner — speaker-level throughout.
"""
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data_utils import save_oof_predictions
from evaluation import compute_metrics, aggregate_fold_metrics
from fusion import (EarlyFusionBaseline, LateFusionBaseline,
                    GatedFusionBaseline, CrossAttentionBaseline)
from models import MambaSeqClassifier, train_classical
from novelty import ConfigurableMultiModal, resolve_novelty
from training import SpeakerSequenceDataset, collate_pad, train_one_fold
from pathlib import Path
import traceback

# ---------------------------------------------------------------------------
# Classical (speaker-level, using pooled features)
# ---------------------------------------------------------------------------

def run_classical_experiment(name, model_type, speaker_df, folds, cfg,
                             pooled):
    """
    pooled: {speaker_id: np.ndarray[D]}
    """ 

    fold_metrics = []

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        df_tr = speaker_df.loc[tr_idx].reset_index(drop=True)
        df_va = speaker_df.loc[va_idx].reset_index(drop=True)

        tr_keep = [s for s in df_tr["speaker_id"] if s in pooled]
        va_keep = [s for s in df_va["speaker_id"] if s in pooled]
        if not tr_keep or not va_keep:
            print(f"  fold {fold_i}: no features, skipping")
            continue

        X_tr = np.stack([pooled[s] for s in tr_keep])
        X_va = np.stack([pooled[s] for s in va_keep])
        df_tr_k = df_tr[df_tr["speaker_id"].isin(tr_keep)].reset_index(drop=True)
        df_va_k = df_va[df_va["speaker_id"].isin(va_keep)].reset_index(drop=True)

        if cfg.task == "classification":
            labels = sorted(df_tr_k["label"].unique())
            l2i = {l: i for i, l in enumerate(labels)}
            y_tr = np.array([l2i[l] for l in df_tr_k["label"]])
        else:
            y_tr = df_tr_k["label"].values.astype(float)

        _, pred = train_classical(X_tr, y_tr, X_va, model_type)
        prob_arr = np.asarray(pred, dtype=float)

        if cfg.task == "classification":
            eps = 1e-7
            p_clip = np.clip(prob_arr, eps, 1 - eps)
            logit_arr = np.log(p_clip / (1 - p_clip))
        else:
            logit_arr = None

        try:
            if cfg.task == "classification":
                save_oof_predictions(
                    out_dir=Path(cfg.output_dir) / "oof",
                    model_name=name, fold_i=fold_i,
                    df_eval=df_va_k, task=cfg.task,
                    probs=prob_arr, logits=logit_arr)
            else:
                save_oof_predictions(
                    out_dir=Path(cfg.output_dir) / "oof",
                    model_name=name, fold_i=fold_i,
                    df_eval=df_va_k, task=cfg.task,
                    y_pred=prob_arr)
        except Exception as e:
            print(f"  [OOF] save failed: {e}")

        m = compute_metrics(df_va_k["label"].values, prob_arr, cfg.task)
        m["fold"] = fold_i
        m["n_speakers"] = len(df_va_k)
        fold_metrics.append(m)
        print(f"  fold {fold_i} ({len(df_va_k)} speakers): " + ", ".join(
            f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float)))

    summary = aggregate_fold_metrics(fold_metrics)
    return fold_metrics, summary


# ---------------------------------------------------------------------------
# Sequence / novelty runner
# ---------------------------------------------------------------------------

def _wrap_mamba(d_in, cfg, use_audio: bool):
    class _W(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.use_audio = use_audio
            self.m = MambaSeqClassifier(
                d_in, d_model=cfg.mamba_d_model,
                n_layers=cfg.mamba_n_layers,
                d_state=cfg.mamba_d_state, d_conv=cfg.mamba_d_conv,
                expand=cfg.mamba_expand, n_outputs=cfg.n_outputs,
                dropout=cfg.mamba_dropout)

        def forward(self, a, am, t, tm, z):
            return (self.m(a, am) if self.use_audio
                    else self.m(t, tm)), None, None
    return _W()


def run_sequence_experiment(name, model_builder, speaker_df, folds, cfg):
    fold_metrics = []

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        cfg._current_model_name = name
        cfg._current_fold = fold_i

        df_tr = speaker_df.loc[tr_idx].reset_index(drop=True)
        df_va = speaker_df.loc[va_idx].reset_index(drop=True)

        # ---- guard: drop rows with empty sequences in this fold ----
        def _has_seqs(row):
            return len(row["audio_seqs"]) > 0 and len(row["text_seqs"]) > 0

        df_tr = df_tr[df_tr.apply(_has_seqs, axis=1)].reset_index(drop=True)
        df_va = df_va[df_va.apply(_has_seqs, axis=1)].reset_index(drop=True)

        if len(df_tr) == 0 or len(df_va) == 0:
            print(f"  fold {fold_i}: empty train or val after filtering "
                  f"(train={len(df_tr)}, val={len(df_va)}); skipping")
            continue

        # scale tabular on train only
        tab_tr = np.stack(df_tr["tabular"].values)
        scaler = StandardScaler().fit(tab_tr)
        df_tr = df_tr.copy()
        df_va = df_va.copy()
        df_tr["tabular"] = [scaler.transform(t[None, :])[0]
                            for t in df_tr["tabular"]]
        df_va["tabular"] = [scaler.transform(t[None, :])[0]
                            for t in df_va["tabular"]]

        # infer dims from the first valid row of this fold
        d_a = df_tr.iloc[0]["audio_seqs"][0].shape[1]
        d_t = df_tr.iloc[0]["text_seqs"][0].shape[1]
        d_z = df_tr.iloc[0]["tabular"].shape[0] 

        model = model_builder(d_a, d_t, d_z, cfg)

        ds_tr = SpeakerSequenceDataset(df_tr, cfg)
        ds_va = SpeakerSequenceDataset(df_va, cfg)
        if cfg.task == "classification":
            ds_va.label2idx = ds_tr.label2idx

        tr_loader = DataLoader(ds_tr, batch_size=cfg.batch_size,
                               shuffle=True, collate_fn=collate_pad)
        va_loader = DataLoader(ds_va, batch_size=cfg.batch_size,
                               shuffle=False, collate_fn=collate_pad)

        print(f"  fold {fold_i}: training on {len(df_tr)} speakers "
              f"(val={len(df_va)}) ...")
        val_probs, val_labels, val_speakers, _ = train_one_fold(
            model, tr_loader, va_loader, cfg, val_df=df_va, verbose=True)

        m = compute_metrics(val_labels, val_probs, cfg.task)
        m["fold"] = fold_i
        m["n_speakers"] = len(val_labels)
        fold_metrics.append(m)
        print(f"  fold {fold_i} ({len(val_labels)} speakers): " + ", ".join(
            f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float)))

    summary = aggregate_fold_metrics(fold_metrics)
    return fold_metrics, summary


# ---------------------------------------------------------------------------
# Experiment specs
# ---------------------------------------------------------------------------

@dataclass
class ExperimentSpec:
    name: str
    family: str
    kind: str             # classical | sequence | novelty
    description: str
    pooled_source: str = None   # egemaps | text | audio
    model_type: str = None
    use_audio: bool = None
    builder: Callable = None
    novelty_preset: str = None
    novelty_overrides: str = None


def default_experiments() -> List[ExperimentSpec]:
    return [
        ExperimentSpec(name="egemaps_xgb", family="baseline",
                       kind="classical", description="eGeMAPS + XGBoost",
                       pooled_source="egemaps", model_type="xgboost"),
        ExperimentSpec(name="egemaps_lgbm", family="baseline",
                       kind="classical", description="eGeMAPS + LightGBM",
                       pooled_source="egemaps", model_type="lightgbm"),
        ExperimentSpec(name="text_logreg", family="baseline",
                       kind="classical",
                       description="Text embeddings + Logistic Regression",
                       pooled_source="text", model_type="logreg"),
        ExperimentSpec(name="text_xgb", family="baseline",
                       kind="classical", description="Text + XGBoost",
                       pooled_source="text", model_type="xgboost"),
        ExperimentSpec(name="audio_mamba", family="baseline",
                       kind="sequence", description="Audio SSL + Mamba",
                       use_audio=True),
        ExperimentSpec(name="early_fusion", family="fusion",
                       kind="sequence", description="Pooled concat + MLP",
                       builder=lambda a, t, z, c:
                           EarlyFusionBaseline(a, t, z, c.n_outputs,
                                               d_model=c.fusion_d_model)),
        ExperimentSpec(name="late_fusion", family="fusion",
                       kind="sequence", description="Separate heads",
                       builder=lambda a, t, z, c:
                           LateFusionBaseline(a, t, z, c.n_outputs,
                                              dropout=c.fusion_dropout)),
        ExperimentSpec(name="gated_fusion", family="fusion",
                       kind="sequence", description="Softmax gate",
                       builder=lambda a, t, z, c:
                           GatedFusionBaseline(a, t, z, c.n_outputs,
                                               d_model=c.fusion_d_model)),
        ExperimentSpec(name="cross_attention", family="fusion",
                       kind="sequence", description="Audio queries text",
                       builder=lambda a, t, z, c:
                           CrossAttentionBaseline(a, t, z, c.n_outputs,
                                                  d_model=c.fusion_d_model,
                                                  n_heads=c.fusion_n_heads,
                                                  dropout=c.fusion_dropout)),
        ExperimentSpec(name="tcm_mamba", family="novelty",
                       kind="novelty",
                       description="Full Temporal Cross-Modal Mamba",
                       novelty_preset="full"),
    ]


def run_one_experiment(spec, speaker_df, folds, cfg, primary_text):
    print(f"\n{'=' * 70}\nEXPERIMENT: {spec.name} ({spec.family})\n"
          f"  {spec.description}\n{'=' * 70}")

    if spec.kind == "classical":
        # pooled feature per speaker for classical models
        pooled = {}
        for _, row in speaker_df.iterrows():
            if spec.pooled_source == "egemaps":
                pooled[row["speaker_id"]] = row["tabular"]
            elif spec.pooled_source == "text":
                pooled[row["speaker_id"]] = np.mean(
                    np.stack([t.mean(0) for t in row["text_seqs"]], 0), 0)
            elif spec.pooled_source == "audio":
                pooled[row["speaker_id"]] = np.mean(
                    np.stack([a.mean(0) for a in row["audio_seqs"]], 0), 0)
            else:
                raise ValueError(spec.pooled_source)
        fm, sm = run_classical_experiment(
            spec.name, spec.model_type, speaker_df, folds, cfg, pooled)
        return {"name": spec.name, "family": spec.family,
                "description": spec.description,
                "folds": fm, "summary": sm.to_dict("records")}

    if spec.kind == "sequence" and spec.builder is None:
        use_audio = spec.use_audio
        builder = lambda a, t, z, c: _wrap_mamba(
            a if use_audio else t, c, use_audio)
    elif spec.kind == "novelty":
        builder = lambda a, t, z, c: ConfigurableMultiModal(
            a, t, z, c.n_outputs,
            resolve_novelty(spec.novelty_preset, spec.novelty_overrides),
            d_model=c.fusion_d_model)
    else:
        builder = spec.builder

    fm, sm = run_sequence_experiment(
        spec.name, builder, speaker_df, folds, cfg)
    return {"name": spec.name, "family": spec.family,
            "description": spec.description,
            "folds": fm, "summary": sm.to_dict("records")}


def run_all_experiments(speaker_df, folds, cfg, primary_text, specs=None):
    if specs is None:
        specs = default_experiments()
    results = {}
    for spec in specs:
        try:
            results[spec.name] = run_one_experiment(
                spec, speaker_df, folds, cfg, primary_text)
        except Exception as e:
            print(f"[ERROR] {spec.name} failed: {e}")
            traceback.print_exc()
    return results