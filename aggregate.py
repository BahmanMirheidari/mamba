"""
Aggregate per-fold OOF predictions into:
  * per_fold_metrics.csv     — one row per (model, fold)
  * mean_std_metrics.csv     — mean ± std across folds, per model
  * pooled_metrics.csv       — metrics computed once on the concatenated
                               OOF predictions across all folds

Run after the main pipeline finishes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from data_utils import aggregate_to_unit
from evaluation import compute_metrics


def load_oof_for_model(oof_dir: Path, model_name: str) -> pd.DataFrame:
    """Concatenate all <model>_fold*.csv files into one dataframe."""
    files = sorted(oof_dir.glob(f"{model_name}_fold*.csv"))
    if not files:
        return pd.DataFrame()
    parts = [pd.read_csv(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    # remove duplicate file_stem (a file can only be in one fold's val set)
    df = df.drop_duplicates(subset=["file_stem"]).reset_index(drop=True)
    return df


def pooled_metrics(oof_df: pd.DataFrame, task: str, unit: str,
                   n_classes: int) -> dict:
    if oof_df.empty:
        return {}
    if task == "classification":
        prob_cols = [c for c in oof_df.columns if c.startswith("prob_")]
        preds = oof_df[prob_cols].values.astype(float)
    else:
        preds = oof_df["y_pred"].values.astype(float)

    agg_preds, agg_labels, _ = aggregate_to_unit(
        oof_df.rename(columns={"y_true": "label"}),
        preds, task, unit)

    if task == "classification":
        # agg_preds is probabilities; label index align with original labels
        labels = sorted(oof_df["y_true"].unique())
        l2i = {l: i for i, l in enumerate(labels)}
        agg_labels_int = np.array([l2i[l] for l in agg_labels])
        # need to remap probabilities to the label order used in l2i
        return compute_metrics(agg_labels_int, agg_preds, task)
    return compute_metrics(agg_labels, agg_preds, task)


def per_fold_metrics(oof_dir: Path, model_name: str, task: str,
                     unit: str, n_classes: int) -> list:
    rows = []
    for f in sorted(oof_dir.glob(f"{model_name}_fold*.csv")):
        fold_i = int(f.stem.split("fold")[-1])
        df = pd.read_csv(f)
        if task == "classification":
            prob_cols = [c for c in df.columns if c.startswith("prob_")]
            preds = df[prob_cols].values.astype(float)
        else:
            preds = df["y_pred"].values.astype(float)

        agg_preds, agg_labels, _ = aggregate_to_unit(
            df.rename(columns={"y_true": "label"}),
            preds, task, unit)

        if task == "classification":
            labels = sorted(df["y_true"].unique())
            l2i = {l: i for i, l in enumerate(labels)}
            agg_labels_int = np.array([l2i[l] for l in agg_labels])
            m = compute_metrics(agg_labels_int, agg_preds, task)
        else:
            m = compute_metrics(agg_labels, agg_preds, task)
        m["fold"] = fold_i
        rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--wav-dir", type=str, required=True)
    ap.add_argument("--demo-csv", type=str, required=True)
    ap.add_argument("--transcriptions-csv", type=str, required=True)
    ap.add_argument("--label-col", type=str, default=None)
    ap.add_argument("--session-col", type=str, default=None)
    ap.add_argument("--task", choices=["classification", "regression"],
                    default="classification")
    ap.add_argument("--n-classes", type=int, default=2)
    ap.add_argument("--aggregation-unit",
                    choices=["auto", "question", "session", "speaker"],
                    default="auto")
    args = ap.parse_args()

    cfg = Config()
    cfg.label_col = args.label_col
    cfg.session_col = args.session_col
    cfg.task = args.task
    cfg.n_classes = args.n_classes
    cfg.aggregation_unit = args.aggregation_unit

    results_dir = Path(args.results_dir)
    oof_dir = results_dir / "oof"
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # discover models from the OOF directory
    models = sorted(set(
        f.stem.split("_fold")[0] for f in oof_dir.glob("*_fold*.csv")))
    print(f"[aggregate] found {len(models)} models: {models}")

    unit = cfg.resolved_aggregation_unit
    task = cfg.task

    per_fold_rows = []
    pooled_rows = []

    for model in models:
        print(f"\n[aggregate] {model}")
        oof_df = load_oof_for_model(oof_dir, model)
        if oof_df.empty:
            print("  no OOF files")
            continue

        # pooled
        pm = pooled_metrics(oof_df, task, unit, cfg.n_classes)
        pm["model"] = model
        pm["n_files"] = len(oof_df)
        pm["n_units"] = oof_df.groupby(["speaker_id", "session_id",
                                        "question_id"]).ngroups \
            if unit == "question" else \
            (oof_df.groupby(["speaker_id", "session_id"]).ngroups
             if unit == "session" else
             oof_df["speaker_id"].nunique())
        pooled_rows.append(pm)
        print(f"  pooled: "
              + ", ".join(f"{k}={v:.4f}" for k, v in pm.items()
                          if isinstance(v, float)))

        # per fold
        fold_metrics = per_fold_metrics(
            oof_dir, model, task, unit, cfg.n_classes)
        for fm in fold_metrics:
            fm["model"] = model
            per_fold_rows.append(fm)

    per_fold_df = pd.DataFrame(per_fold_rows)
    pooled_df = pd.DataFrame(pooled_rows)

    # mean ± std across folds
    if not per_fold_df.empty:
        metric_cols = [c for c in per_fold_df.columns
                       if c not in ("model", "fold")]
        mean_std = (per_fold_df.groupby("model")[metric_cols]
                    .agg(["mean", "std", "count"]))
        mean_std.columns = [f"{a}_{b}" for a, b in mean_std.columns]
        mean_std = mean_std.reset_index()
    else:
        mean_std = pd.DataFrame()

    per_fold_df.to_csv(tables_dir / "per_fold_metrics.csv", index=False)
    mean_std.to_csv(tables_dir / "mean_std_metrics.csv", index=False)
    pooled_df.to_csv(tables_dir / "pooled_metrics.csv", index=False)

    print(f"\n[done] wrote:")
    print(f"  {tables_dir / 'per_fold_metrics.csv'}")
    print(f"  {tables_dir / 'mean_std_metrics.csv'}")
    print(f"  {tables_dir / 'pooled_metrics.csv'}")

    # print a compact comparison
    primary = cfg.primary_metric
    if not pooled_df.empty and f"{primary}" in pooled_df.columns:
        pooled_sorted = pooled_df.sort_values(
            primary, ascending=cfg.lower_is_better)
        print(f"\nPooled metrics (primary = {primary}):")
        print(pooled_sorted[["model", primary,
                             "n_files", "n_units"]].to_string(index=False))


if __name__ == "__main__":
    main()