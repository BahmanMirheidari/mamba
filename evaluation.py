"""
Metrics + statistical comparisons.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel, wilcoxon
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score, cohen_kappa_score,
                             mean_absolute_error, mean_squared_error,
                             r2_score)


def compute_metrics(y_true, y_pred, task: str):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if task == "classification":
        pred = y_pred.argmax(1)
        out = {
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, pred, average="weighted")),
            "kappa": float(cohen_kappa_score(y_true, pred)),
        }
        try:
            if y_pred.shape[1] == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, y_pred[:, 1]))
            else:
                out["roc_auc"] = float(roc_auc_score(
                    y_true, y_pred, multi_class="ovr", average="macro"))
        except ValueError:
            out["roc_auc"] = np.nan
        return out

    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    out = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 2 else np.nan,
    }
    if len(y_true) > 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        out["pearson_r"] = float(pearsonr(y_true, y_pred)[0])
        out["spearman_r"] = float(pearsonr(
            y_true.argsort().argsort(),
            y_pred.argsort().argsort())[0])
    else:
        out["pearson_r"] = np.nan
        out["spearman_r"] = np.nan
    return out


def aggregate_fold_metrics(fold_metrics):
    rows = []
    for k in fold_metrics[0].keys():
        vals = np.array([fm[k] for fm in fold_metrics], dtype=float)
        vals = vals[~np.isnan(vals)]
        rows.append({
            "metric": k,
            "mean": float(vals.mean()) if len(vals) else np.nan,
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals),
        })
    return pd.DataFrame(rows)


def paired_test(a_metrics, b_metrics, metric: str):
    a = np.array([m[metric] for m in a_metrics], dtype=float)
    b = np.array([m[metric] for m in b_metrics], dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return {"n": int(len(a)), "p_ttest": np.nan,
                "p_wilcoxon": np.nan, "mean_diff": np.nan}
    t, p_t = ttest_rel(a, b)
    try:
        _, p_w = wilcoxon(a, b)
    except Exception:
        p_w = np.nan
    return {
        "n": int(len(a)),
        "t": float(t),
        "p_ttest": float(p_t),
        "p_wilcoxon": float(p_w) if not np.isnan(p_w) else np.nan,
        "mean_diff": float((a - b).mean()),
    }