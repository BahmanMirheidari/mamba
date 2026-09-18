"""
Experiment registry and runner.

Every experiment has:
    name:       unique identifier
    family:     'baseline' | 'fusion' | 'novelty'
    build:      function(d_a, d_t, d_z, cfg) -> nn.Module
    description: str
"""
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data_utils import aggregate_to_unit, save_oof_predictions
from evaluation import compute_metrics, aggregate_fold_metrics
from fusion import (EarlyFusionBaseline, LateFusionBaseline,
                    GatedFusionBaseline, CrossAttentionBaseline)
from models import MambaSeqClassifier, train_classical
from novelty import ConfigurableMultiModal, resolve_novelty
from training import MultiModalDataset, collate_pad, train_one_fold 


# =====================================================================
# Classical runner (XGB / LGBM / LogReg on pooled features)
# =====================================================================

def run_classical_experiment(name, model_type, df, folds, cfg, pooled):
    """
    pooled: {file_stem: np.ndarray[D]}
    Writes per-fold OOF CSVs to cfg.output_dir/oof/.
    """
    from pathlib import Path

    fold_metrics = []
    unit = cfg.resolved_aggregation_unit

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        cfg._current_model_name = name
        cfg._current_fold = fold_i

        df_tr = df.loc[tr_idx].reset_index(drop=True)
        df_va = df.loc[va_idx].reset_index(drop=True)

        tr_keep = [s for s in df_tr["file_stem"] if s in pooled]
        va_keep = [s for s in df_va["file_stem"] if s in pooled]
        if not tr_keep or not va_keep:
            print(f"  fold {fold_i}: no features, skipping")
            continue

        X_tr = np.stack([pooled[s] for s in tr_keep])
        X_va = np.stack([pooled[s] for s in va_keep])
        df_tr_k = df_tr[df_tr["file_stem"].isin(tr_keep)].reset_index(drop=True)
        df_va_k = df_va[df_va["file_stem"].isin(va_keep)].reset_index(drop=True)

        if cfg.task == "classification":
            labels = sorted(df_tr_k["label"].unique())
            l2i = {l: i for i, l in enumerate(labels)}
            y_tr = np.array([l2i[l] for l in df_tr_k["label"]])
        else:
            y_tr = df_tr_k["label"].values.astype(float)

        _, pred = train_classical(X_tr, y_tr, X_va, model_type)

        # ---- derive pseudo-logits from probs ----
        prob_arr = np.asarray(pred, dtype=float)
        if cfg.task == "classification":
            eps = 1e-7
            p_clipped = np.clip(prob_arr, eps, 1 - eps)
            logit_arr = np.log(p_clipped / (1 - p_clipped))
        else:
            logit_arr = None

        # ---- persist OOF ----
        try:
            if cfg.task == "classification":
                save_oof_predictions(
                    out_dir=Path(cfg.output_dir) / "oof",
                    model_name=name,
                    fold_i=fold_i,
                    df_eval=df_va_k,
                    task=cfg.task,
                    probs=prob_arr,
                    logits=logit_arr)
            else:
                save_oof_predictions(
                    out_dir=Path(cfg.output_dir) / "oof",
                    model_name=name,
                    fold_i=fold_i,
                    df_eval=df_va_k,
                    task=cfg.task,
                    y_pred=prob_arr)
        except Exception as e:
            print(f"  [OOF] save failed: {e}")

        agg_preds, agg_labels, agg_keys = aggregate_to_unit(
            df_va_k, pred, cfg.task, unit)

        m = compute_metrics(agg_labels, agg_preds, cfg.task)
        m["fold"] = fold_i
        m["n_units"] = len(agg_keys)
        fold_metrics.append(m)
        print(f"  fold {fold_i}: " + ", ".join(
            f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float)))

    summary = aggregate_fold_metrics(fold_metrics)
    return fold_metrics, summary



# =====================================================================
# Sequence / fusion runner
# =====================================================================

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


def run_sequence_experiment(name, model_builder, df, folds, cfg,
                            audio_emb, text_emb, tabular_arr,
                            tabular_index):
    fold_metrics = []

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        cfg._current_model_name = name
        cfg._current_fold = fold_i 

        df_tr = df.loc[tr_idx].reset_index(drop=True)
        df_va = df.loc[va_idx].reset_index(drop=True)

        tab_tr = tabular_arr[[tabular_index[s] for s in df_tr["file_stem"]]]
        tab_va = tabular_arr[[tabular_index[s] for s in df_va["file_stem"]]]
        scaler = StandardScaler().fit(tab_tr)
        tab_tr = scaler.transform(tab_tr).astype(np.float32)
        tab_va = scaler.transform(tab_va).astype(np.float32)

        d_a = next(iter(audio_emb.values())).shape[1]
        d_t = next(iter(text_emb.values())).shape[1]
        d_z = tab_tr.shape[1]

        model = model_builder(d_a, d_t, d_z, cfg)

        ds_tr = MultiModalDataset(df_tr, audio_emb, text_emb, tab_tr, cfg)
        ds_va = MultiModalDataset(df_va, audio_emb, text_emb, tab_va, cfg)
        if cfg.task == "classification":
            ds_va.label2idx = ds_tr.label2idx

        tr_loader = DataLoader(ds_tr, batch_size=cfg.batch_size,
                               shuffle=True, collate_fn=collate_pad)
        va_loader = DataLoader(ds_va, batch_size=cfg.batch_size,
                               shuffle=False, collate_fn=collate_pad)

        print(f"  fold {fold_i}: training ...")
        agg_preds, agg_labels, agg_keys, _ = train_one_fold(
            model, tr_loader, va_loader, cfg, val_df=df_va, verbose=True)

        m = compute_metrics(agg_labels, agg_preds, cfg.task)
        m["fold"] = fold_i
        m["n_units"] = len(agg_keys)
        fold_metrics.append(m)
        print(f"  fold {fold_i}: " + ", ".join(
            f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float)))

    summary = aggregate_fold_metrics(fold_metrics)
    return fold_metrics, summary


# =====================================================================
# Experiment specs
# =====================================================================

@dataclass
class ExperimentSpec:
    name: str
    family: str          # baseline | fusion | novelty
    kind: str            # 'classical' | 'sequence' | 'novelty'
    description: str
    # classical
    pooled_source: str = None       # 'egemaps' | 'text' | 'audio'
    model_type: str = None          # 'xgboost' | 'lightgbm' | 'logreg'
    # sequence / novelty
    use_audio: bool = None
    builder: Callable = None
    novelty_preset: str = None
    novelty_overrides: str = None


def default_experiments() -> List[ExperimentSpec]:
    return [
        # ---- baselines ----
        ExperimentSpec(
            name="egemaps_xgb", family="baseline", kind="classical",
            description="eGeMAPS + XGBoost",
            pooled_source="egemaps", model_type="xgboost"),
        ExperimentSpec(
            name="egemaps_lgbm", family="baseline", kind="classical",
            description="eGeMAPS + LightGBM",
            pooled_source="egemaps", model_type="lightgbm"),
        ExperimentSpec(
            name="text_logreg", family="baseline", kind="classical",
            description="Text embeddings + Logistic Regression",
            pooled_source="text", model_type="logreg"),
        ExperimentSpec(
            name="text_xgb", family="baseline", kind="classical",
            description="Text embeddings + XGBoost",
            pooled_source="text", model_type="xgboost"),
        ExperimentSpec(
            name="audio_mamba", family="baseline", kind="sequence",
            description="Audio SSL + Mamba",
            use_audio=True),

        # ---- fusion baselines ----
        ExperimentSpec(
            name="early_fusion", family="fusion", kind="sequence",
            description="Pooled concat + MLP",
            builder=lambda d_a, d_t, d_z, c:
                EarlyFusionBaseline(d_a, d_t, d_z, c.n_outputs,
                                    d_model=c.fusion_d_model)),
        ExperimentSpec(
            name="late_fusion", family="fusion", kind="sequence",
            description="Separate heads + learned weights",
            builder=lambda d_a, d_t, d_z, c:
                LateFusionBaseline(d_a, d_t, d_z, c.n_outputs,
                                   dropout=c.fusion_dropout)),
        ExperimentSpec(
            name="gated_fusion", family="fusion", kind="sequence",
            description="Softmax gate over pooled modalities",
            builder=lambda d_a, d_t, d_z, c:
                GatedFusionBaseline(d_a, d_t, d_z, c.n_outputs,
                                    d_model=c.fusion_d_model)),
        ExperimentSpec(
            name="cross_attention", family="fusion", kind="sequence",
            description="Audio queries text, pooled + tabular",
            builder=lambda d_a, d_t, d_z, c:
                CrossAttentionBaseline(d_a, d_t, d_z, c.n_outputs,
                                       d_model=c.fusion_d_model,
                                       n_heads=c.fusion_n_heads,
                                       dropout=c.fusion_dropout)),

        # ---- the novel model ----
        ExperimentSpec(
            name="tcm_mamba", family="novelty", kind="novelty",
            description="Full Temporal Cross-Modal Mamba",
            novelty_preset="full"),
    ]


def run_one_experiment(spec: ExperimentSpec, df, folds, cfg,
                       egemaps_df, audio_emb, text_emb, primary_text):
    print(f"\n{'=' * 70}\nEXPERIMENT: {spec.name} ({spec.family})\n"
          f"  {spec.description}\n{'=' * 70}")

    if spec.kind == "classical":
        if spec.pooled_source == "egemaps":
            pooled = {stem: egemaps_df.loc[stem].values.astype(np.float32)
                      for stem in egemaps_df.index}
        elif spec.pooled_source == "text":
            pooled = {k: v.mean(axis=0) for k, v in text_emb.items()}
        elif spec.pooled_source == "audio":
            pooled = {k: v.mean(axis=0) for k, v in audio_emb.items()}
        else:
            raise ValueError(spec.pooled_source)
        fm, sm = run_classical_experiment(
            spec.name, spec.model_type, df, folds, cfg, pooled)
        return {"name": spec.name, "family": spec.family,
                "description": spec.description,
                "folds": fm, "summary": sm.to_dict("records")}

    # sequence / novelty
    egemaps_arr = egemaps_df.values.astype(np.float32)
    egemaps_index = {s: i for i, s in enumerate(egemaps_df.index)}

    if spec.kind == "sequence" and spec.builder is None:
        d_a = next(iter(audio_emb.values())).shape[1]
        d_t = next(iter(text_emb.values())).shape[1]
        use_audio = spec.use_audio
        builder = lambda a, t, z, c: _wrap_mamba(
            a if use_audio else t, c, use_audio)
    elif spec.kind == "novelty":
        builder = lambda d_a, d_t, d_z, c: ConfigurableMultiModal(
            d_a, d_t, d_z, c.n_outputs,
            resolve_novelty(spec.novelty_preset, spec.novelty_overrides),
            d_model=c.fusion_d_model)
    else:
        builder = spec.builder

    fm, sm = run_sequence_experiment(
        spec.name, builder, df, folds, cfg,
        audio_emb, text_emb, egemaps_arr, egemaps_index)
    return {"name": spec.name, "family": spec.family,
            "description": spec.description,
            "folds": fm, "summary": sm.to_dict("records")}


def run_all_experiments(df, folds, cfg, egemaps_df, audio_emb, text_emb,
                        primary_text, specs=None):
    if specs is None:
        specs = default_experiments()

    results = {}
    for spec in specs:
        try:
            r = run_one_experiment(spec, df, folds, cfg,
                                   egemaps_df, audio_emb, text_emb,
                                   primary_text)
            results[spec.name] = r
        except Exception as e:
            import traceback
            print(f"[ERROR] {spec.name} failed: {e}")
            traceback.print_exc()
    return results