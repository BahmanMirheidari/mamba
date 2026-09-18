"""
aggregate.py — comprehensive per-model metrics + bootstrap CIs from OOF
predictions saved by the pipeline.

Computes:
  Classification: accuracy, balanced accuracy, kappa, MCC,
                  macro / weighted / micro P/R/F1,
                  per-class P/R/F1, sensitivity, specificity, PPV, NPV,
                  ROC-AUC (macro/weighted), PR-AUC, log-loss, Brier,
                  confusion matrix
  Regression:     RMSE, MAE, R2, explained variance, MAPE,
                  Pearson, Spearman

Reports each metric pooled across all folds and per fold, with bootstrap
CIs (default 2000 iterations; pass --n-bootstrap 10000 for the full run).

Usage
-----
python aggregate.py \
    --results-dir results \
    --task classification --n-classes 2 \
    --aggregation-unit speaker \
    --n-bootstrap 10000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             explained_variance_score, log_loss,
                             roc_auc_score)


# =========================================================================
# Fast numpy classification metrics
# =========================================================================

def _confusion(y_true, y_pred, n_classes):
    idx = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    return np.bincount(idx, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def classification_metrics(y_true, y_prob, n_classes):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = y_prob.argmax(axis=1)

    cm = _confusion(y_true, y_pred, n_classes)
    total = int(cm.sum())
    if total == 0:
        return {}

    tp = np.diag(cm).astype(float)
    fp = cm.sum(0).astype(float) - tp
    fn = cm.sum(1).astype(float) - tp
    tn = total - tp - fp - fn
    support = cm.sum(1).astype(float)

    sens = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    spec = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    ppv = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    npv = np.divide(tn, tn + fn, out=np.zeros_like(tn), where=(tn + fn) > 0)
    f1 = np.divide(2 * ppv * sens, ppv + sens,
                   out=np.zeros_like(ppv), where=(ppv + sens) > 0)

    acc = float(tp.sum() / total)
    bal_acc = float(sens.mean())

    macro_p = float(ppv.mean())
    macro_r = float(sens.mean())
    macro_f1 = float(f1.mean())

    w_p = float((ppv * support).sum() / support.sum())
    w_r = float((sens * support).sum() / support.sum())
    w_f1 = float((f1 * support).sum() / support.sum())

    denom = np.sqrt(
        (tp.sum() + fp.sum()) * (tp.sum() + fn.sum()) *
        (tn.sum() + fp.sum()) * (tn.sum() + fn.sum()))
    mcc = float((tp.sum() * tn.sum() - fp.sum() * fn.sum()) / denom) \
        if denom > 0 else 0.0

    p_o = acc
    p_e = float((cm.sum(0) * cm.sum(1)).sum() / total ** 2)
    kappa = float((p_o - p_e) / (1 - p_e)) if p_e < 1 else 0.0

    out = {
        "n": total,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "kappa": kappa,
        "mcc": mcc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": w_p,
        "weighted_recall": w_r,
        "weighted_f1": w_f1,
        "micro_precision": acc,
        "micro_recall": acc,
        "micro_f1": acc,
    }

    for c in range(n_classes):
        out[f"class{c}_sensitivity"] = float(sens[c])
        out[f"class{c}_specificity"] = float(spec[c])
        out[f"class{c}_ppv"] = float(ppv[c])
        out[f"class{c}_npv"] = float(npv[c])
        out[f"class{c}_f1"] = float(f1[c])
        out[f"class{c}_support"] = int(support[c])

    try:
        if n_classes == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            out["pr_auc"] = float(average_precision_score(y_true, y_prob[:, 1]))
        else:
            out["roc_auc"] = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"))
            out["roc_auc_weighted"] = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted"))
            prs = []
            for c in range(n_classes):
                try:
                    prs.append(average_precision_score(
                        (y_true == c).astype(int), y_prob[:, c]))
                except Exception:
                    pass
            out["pr_auc"] = float(np.mean(prs)) if prs else np.nan
    except Exception:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan

    try:
        out["log_loss"] = float(log_loss(
            y_true, y_prob, labels=list(range(n_classes))))
    except Exception:
        out["log_loss"] = np.nan

    if n_classes == 2:
        try:
            out["brier"] = float(brier_score_loss(y_true, y_prob[:, 1]))
        except Exception:
            out["brier"] = np.nan

    out["confusion_matrix"] = cm.tolist()
    return out


# =========================================================================
# Regression metrics
# =========================================================================

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n < 2:
        return {"n": n}

    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    ev = float(explained_variance_score(y_true, y_pred)) if n > 2 else np.nan

    m = y_true != 0
    mape = float(np.mean(np.abs(err[m] / y_true[m]))) * 100 if m.any() else np.nan

    if n > 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(pearsonr(y_true, y_pred)[0])
        spearman = float(spearmanr(y_true, y_pred)[0])
    else:
        pearson = spearman = np.nan

    return {
        "n": int(n),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "explained_variance": ev,
        "mape": mape,
        "pearson_r": pearson,
        "spearman_r": spearman,
    }


# =========================================================================
# Bootstrap
# =========================================================================

def bootstrap_ci(y_true, y_score, task, n_classes,
                 n_iter=2000, alpha=0.05, seed=42):
    n = len(y_true)
    rng = np.random.default_rng(seed)

    def metric_fn(yt, ys):
        if task == "classification":
            return classification_metrics(yt, ys, n_classes)
        return regression_metrics(yt, ys)

    point = metric_fn(y_true, y_score)
    keys = [k for k in point if k != "confusion_matrix"]

    samples = {k: np.full(n_iter, np.nan) for k in keys}
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if task == "classification" and len(np.unique(yt)) < 2:
            continue
        try:
            m = metric_fn(yt, ys)
        except Exception:
            continue
        for k in keys:
            if k in m and not (isinstance(m[k], float) and np.isnan(m[k])):
                samples[k][i] = m[k]

    ci = {}
    for k in keys:
        v = samples[k]
        v = v[~np.isnan(v)]
        if len(v) == 0:
            ci[k] = {"mean": np.nan, "lower": np.nan,
                     "upper": np.nan, "std": np.nan}
        else:
            ci[k] = {
                "mean": float(np.mean(v)),
                "lower": float(np.percentile(v, 100 * alpha / 2)),
                "upper": float(np.percentile(v, 100 * (1 - alpha / 2))),
                "std": float(np.std(v, ddof=1)),
            }
    return point, ci


# =========================================================================
# Loading and unit aggregation
# =========================================================================

def load_oof(oof_dir: Path, model: str) -> pd.DataFrame:
    files = sorted(oof_dir.glob(f"{model}_fold*.csv"))
    if not files:
        return pd.DataFrame()
    parts = []
    for f in files:
        d = pd.read_csv(f)
        d["__fold"] = int(f.stem.split("fold")[-1])
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def aggregate_to_unit(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    if unit == "question":
        keys = ["speaker_id", "session_id", "question_id"]
    elif unit == "session":
        keys = ["speaker_id", "session_id"]
    else:
        keys = ["speaker_id"]

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    logit_cols = [c for c in df.columns if c.startswith("logit_")]

    agg_spec = {"y_true": "first"}
    for c in prob_cols:
        agg_spec[c] = "mean"
    for c in logit_cols:
        agg_spec[c] = "mean"
    agg_spec["y_pred"] = "mean"

    grouped = df.groupby(keys, as_index=False).agg(agg_spec)

    if prob_cols:
        p = grouped[prob_cols].to_numpy(dtype=float)
        p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
        grouped[prob_cols] = p
        grouped["y_pred"] = p.argmax(axis=1)

    return grouped


# =========================================================================
# Main
# =========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--task", choices=["classification", "regression"],
                    default="classification")
    ap.add_argument("--n-classes", type=int, default=2)
    ap.add_argument("--aggregation-unit",
                    choices=["auto", "question", "session", "speaker"],
                    default="speaker")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    oof_dir = results_dir / "oof"
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not oof_dir.exists():
        raise FileNotFoundError(f"no OOF dir: {oof_dir}")

    models = sorted(set(f.stem.split("_fold")[0]
                        for f in oof_dir.glob("*_fold*.csv")))
    print(f"[aggregate] {len(models)} models: {models}")

    unit = "speaker" if args.aggregation_unit == "auto" else args.aggregation_unit

    per_fold_rows, pooled_rows, ci_rows, cm_rows = [], [], [], []

    for model in models:
        print(f"\n[aggregate] {model}")
        df = load_oof(oof_dir, model)
        if df.empty:
            continue

        # ---- per fold ----
        for fold, g in df.groupby("__fold"):
            agg = aggregate_to_unit(g, unit)
            if args.task == "classification":
                pc = [c for c in agg.columns if c.startswith("prob_")]
                yp = agg[pc].to_numpy(dtype=float)
                yt = agg["y_true"].to_numpy(dtype=int)
                m = classification_metrics(yt, yp, args.n_classes)
                cm_rows.append({
                    "model": model, "fold": int(fold),
                    "cm": json.dumps(m.pop("confusion_matrix", []))})
            else:
                yt = agg["y_true"].to_numpy(dtype=float)
                yp = agg["y_pred"].to_numpy(dtype=float)
                m = regression_metrics(yt, yp)
            m.update({"model": model, "fold": int(fold),
                      "n_units": int(len(agg))})
            per_fold_rows.append(m)

        # ---- pooled + bootstrap CI ----
        agg = aggregate_to_unit(df, unit)
        if args.task == "classification":
            pc = [c for c in agg.columns if c.startswith("prob_")]
            yp = agg[pc].to_numpy(dtype=float)
            yt = agg["y_true"].to_numpy(dtype=int)
            point, ci = bootstrap_ci(
                yt, yp, "classification", args.n_classes,
                n_iter=args.n_bootstrap, alpha=args.alpha, seed=args.seed)
            cm_rows.append({
                "model": model, "fold": -1,
                "cm": json.dumps(point.pop("confusion_matrix", []))})
        else:
            yt = agg["y_true"].to_numpy(dtype=float)
            yp = agg["y_pred"].to_numpy(dtype=float)
            point, ci = bootstrap_ci(
                yt, yp, "regression", None,
                n_iter=args.n_bootstrap, alpha=args.alpha, seed=args.seed)

        point.update({"model": model, "n_units": int(len(agg))})
        pooled_rows.append(point)

        for k, v in ci.items():
            ci_rows.append({
                "model": model, "metric": k,
                "point": point.get(k, np.nan),
                "mean": v["mean"], "lower": v["lower"],
                "upper": v["upper"], "std": v["std"]})

        # progress line
        head = ["accuracy", "macro_f1", "balanced_accuracy", "roc_auc"] \
            if args.task == "classification" else ["rmse", "r2", "mae"]
        print(f"  n_units = {len(agg)}")
        for k in head:
            if k in point:
                lo = next((r["lower"] for r in ci_rows
                           if r["model"] == model and r["metric"] == k), np.nan)
                hi = next((r["upper"] for r in ci_rows
                           if r["model"] == model and r["metric"] == k), np.nan)
                print(f"    {k:22s} = {point[k]:.4f}  "
                      f"[{lo:.4f}, {hi:.4f}]")

    # ---- save ----
    pd.DataFrame(per_fold_rows).to_csv(
        tables_dir / "per_fold_metrics.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(
        tables_dir / "pooled_metrics.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(
        tables_dir / "bootstrap_ci.csv", index=False)
    if cm_rows:
        pd.DataFrame(cm_rows).to_csv(
            tables_dir / "confusion_matrices.csv", index=False)

    # ---- mean ± std across folds ----
    pf = pd.DataFrame(per_fold_rows)
    if not pf.empty:
        num_cols = [c for c in pf.columns
                    if c not in ("model", "fold", "confusion_matrix")
                    and pd.api.types.is_numeric_dtype(pf[c])]
        ms = pf.groupby("model")[num_cols].agg(["mean", "std"]).round(4)
        ms.columns = [f"{a}_{b}" for a, b in ms.columns]
        ms.reset_index().to_csv(
            tables_dir / "mean_std_metrics.csv", index=False)

    print(f"\n[done] wrote:")
    for f in ["per_fold_metrics.csv", "pooled_metrics.csv",
              "bootstrap_ci.csv", "confusion_matrices.csv",
              "mean_std_metrics.csv"]:
        print(f"  {tables_dir / f}")


if __name__ == "__main__":
    main()