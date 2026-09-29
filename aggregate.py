"""
aggregate.py — metrics + CIs from speaker-level OOF predictions.
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


def _confusion(y_true, y_pred, n_classes):
    idx = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    return np.bincount(idx, minlength=n_classes ** 2).reshape(
        n_classes, n_classes)


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
    bal = float(sens.mean())
    macro_p = float(ppv.mean())
    macro_r = float(sens.mean())
    macro_f1 = float(f1.mean())
    wp = float((ppv * support).sum() / support.sum())
    wr = float((sens * support).sum() / support.sum())
    wf1 = float((f1 * support).sum() / support.sum())

    denom = np.sqrt((tp.sum() + fp.sum()) * (tp.sum() + fn.sum()) *
                    (tn.sum() + fp.sum()) * (tn.sum() + fn.sum()))
    mcc = float((tp.sum() * tn.sum() - fp.sum() * fn.sum()) / denom) \
        if denom > 0 else 0.0

    p_o = acc
    p_e = float((cm.sum(0) * cm.sum(1)).sum() / total ** 2)
    kappa = float((p_o - p_e) / (1 - p_e)) if p_e < 1 else 0.0

    out = {
        "n_speakers": total,
        "accuracy": acc, "balanced_accuracy": bal,
        "kappa": kappa, "mcc": mcc,
        "macro_precision": macro_p, "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": wp, "weighted_recall": wr,
        "weighted_f1": wf1,
        "micro_precision": acc, "micro_recall": acc, "micro_f1": acc,
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
            out["brier"] = float(brier_score_loss(y_true, y_prob[:, 1]))
        else:
            out["roc_auc"] = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"))
            out["roc_auc_weighted"] = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted"))
            prs = [average_precision_score((y_true == c).astype(int),
                                           y_prob[:, c])
                   for c in range(n_classes)]
            out["pr_auc"] = float(np.mean(prs))
    except Exception:
        out.setdefault("roc_auc", np.nan)

    try:
        out["log_loss"] = float(log_loss(
            y_true, y_prob, labels=list(range(n_classes))))
    except Exception:
        out["log_loss"] = np.nan

    out["confusion_matrix"] = cm.tolist()
    return out


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n < 2:
        return {"n_speakers": n}
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
        pr = float(pearsonr(y_true, y_pred)[0])
        sr = float(spearmanr(y_true, y_pred)[0])
    else:
        pr = sr = np.nan
    return {"n_speakers": int(n), "rmse": rmse, "mae": mae, "r2": r2,
            "explained_variance": ev, "mape": mape,
            "pearson_r": pr, "spearman_r": sr}


def bootstrap_ci(y_true, y_score, task, n_classes,
                 n_iter=2000, alpha=0.05, seed=42):
    n = len(y_true)
    rng = np.random.default_rng(seed)

    def fn(yt, ys):
        return (classification_metrics(yt, ys, n_classes)
                if task == "classification" else regression_metrics(yt, ys))

    point = fn(y_true, y_score)
    keys = [k for k in point if k != "confusion_matrix"]
    samples = {k: np.full(n_iter, np.nan) for k in keys}
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if task == "classification" and len(np.unique(yt)) < 2:
            continue
        try:
            m = fn(yt, ys)
        except Exception:
            continue
        for k in keys:
            v = m.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                samples[k][i] = v

    ci = {}
    for k in keys:
        v = samples[k]; v = v[~np.isnan(v)]
        ci[k] = ({"mean": float(v.mean()),
                  "lower": float(np.percentile(v, 100 * alpha / 2)),
                  "upper": float(np.percentile(v, 100 * (1 - alpha / 2))),
                  "std": float(v.std(ddof=1))}
                 if len(v) else {"mean": np.nan, "lower": np.nan,
                                 "upper": np.nan, "std": np.nan})
    return point, ci


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--task", choices=["classification", "regression"],
                    default="classification")
    ap.add_argument("--n-classes", type=int, default=2)
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

    per_fold, pooled, ci_rows, cm_rows = [], [], [], []

    for model in models:
        print(f"\n[aggregate] {model}")
        df = load_oof(oof_dir, model)
        if df.empty:
            continue

        # ---- per fold (already speaker-level) ----
        for fold, g in df.groupby("__fold"):
            if args.task == "classification":
                pc = [c for c in g.columns if c.startswith("prob_")]
                yp = g[pc].to_numpy(dtype=float)
                yt = g["y_true"].to_numpy(dtype=int)
                m = classification_metrics(yt, yp, args.n_classes)
                cm_rows.append({"model": model, "fold": int(fold),
                                "cm": json.dumps(m.pop("confusion_matrix", []))})
            else:
                yt = g["y_true"].to_numpy(dtype=float)
                yp = g["y_pred"].to_numpy(dtype=float)
                m = regression_metrics(yt, yp)
            m.update({"model": model, "fold": int(fold),
                      "n_speakers": int(len(g))})
            per_fold.append(m)

        # ---- pooled (still one row per speaker, concatenated folds) ----
        if args.task == "classification":
            pc = [c for c in df.columns if c.startswith("prob_")]
            yp = df[pc].to_numpy(dtype=float)
            yt = df["y_true"].to_numpy(dtype=int)
            point, ci = bootstrap_ci(
                yt, yp, "classification", args.n_classes,
                n_iter=args.n_bootstrap, alpha=args.alpha, seed=args.seed)
            cm_rows.append({"model": model, "fold": -1,
                            "cm": json.dumps(point.pop("confusion_matrix", []))})
        else:
            yt = df["y_true"].to_numpy(dtype=float)
            yp = df["y_pred"].to_numpy(dtype=float)
            point, ci = bootstrap_ci(
                yt, yp, "regression", None,
                n_iter=args.n_bootstrap, alpha=args.alpha, seed=args.seed)

        point.update({"model": model, "n_speakers": int(len(df))})
        pooled.append(point)
        for k, v in ci.items():
            ci_rows.append({"model": model, "metric": k,
                            "point": point.get(k, np.nan),
                            "mean": v["mean"], "lower": v["lower"],
                            "upper": v["upper"], "std": v["std"]})

        head = (["accuracy", "macro_f1", "roc_auc"]
                if args.task == "classification" else ["rmse", "r2", "mae"])
        print(f"  n_speakers = {len(df)}")
        for k in head:
            if k in point:
                row = next((r for r in ci_rows
                            if r["model"] == model and r["metric"] == k), None)
                if row:
                    print(f"    {k:22s} = {point[k]:.4f}  "
                          f"[{row['lower']:.4f}, {row['upper']:.4f}]")

    pd.DataFrame(per_fold).to_csv(tables_dir / "per_fold_metrics.csv", index=False)
    pd.DataFrame(pooled).to_csv(tables_dir / "pooled_metrics.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(tables_dir / "bootstrap_ci.csv", index=False)
    if cm_rows:
        pd.DataFrame(cm_rows).to_csv(
            tables_dir / "confusion_matrices.csv", index=False)

    pf = pd.DataFrame(per_fold)
    if not pf.empty:
        num = [c for c in pf.columns
               if c not in ("model", "fold", "confusion_matrix")
               and pd.api.types.is_numeric_dtype(pf[c])]
        ms = pf.groupby("model")[num].agg(["mean", "std"]).round(4)
        ms.columns = [f"{a}_{b}" for a, b in ms.columns]
        ms.reset_index().to_csv(
            tables_dir / "mean_std_metrics.csv", index=False)

    print(f"\n[done] tables in {tables_dir}")


if __name__ == "__main__":
    main()