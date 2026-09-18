"""
Tables, figures, and statistical comparisons.
"""
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import paired_test


def _flatten_summary(summary_rows):
    return {r["metric"]: {"mean": r["mean"], "std": r["std"]}
            for r in summary_rows}


def comparison_table(results: Dict, cfg) -> pd.DataFrame:
    """
    One row per experiment with mean ± std of the primary metric plus a
    handful of secondary metrics.
    """
    primary = cfg.primary_metric
    secondaries = (["balanced_accuracy", "macro_f1", "roc_auc"]
                   if cfg.task == "classification"
                   else ["mae", "r2", "pearson_r"])

    rows = []
    for name, data in results.items():
        s = _flatten_summary(data["summary"])
        row = {
            "experiment": name,
            "family": data.get("family", "-"),
            "description": data.get("description", ""),
        }
        for m in [primary] + [x for x in secondaries if x != primary]:
            if m in s:
                row[f"{m}_mean"] = s[m]["mean"]
                row[f"{m}_std"] = s[m]["std"]
        rows.append(row)

    df = pd.DataFrame(rows)
    if f"{primary}_mean" not in df.columns:
        return df
    return df.sort_values(f"{primary}_mean",
                          ascending=cfg.lower_is_better).reset_index(drop=True)


def ablation_table(ablation_results: Dict, baseline_name: str,
                   cfg) -> pd.DataFrame:
    primary = cfg.primary_metric
    rows = []
    for label, data in ablation_results.items():
        s = _flatten_summary(data["summary"])
        if primary not in s:
            continue
        rows.append({
            "ablation": label,
            "preset": data.get("preset", "-"),
            f"{primary}_mean": s[primary]["mean"],
            f"{primary}_std": s[primary]["std"],
        })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(
        f"{primary}_mean", ascending=cfg.lower_is_better).reset_index(drop=True)

    if baseline_name in df["ablation"].values:
        base = df.loc[df["ablation"] == baseline_name,
                      f"{primary}_mean"].iloc[0]
        if cfg.lower_is_better:
            df["delta_vs_full"] = df[f"{primary}_mean"] - base
        else:
            df["delta_vs_full"] = base - df[f"{primary}_mean"]
    return df


def significance_table(results: Dict, reference: str, cfg) -> pd.DataFrame:
    """
    Paired t-test on the primary metric between each experiment and the
    reference experiment, using per-fold values.
    """
    if reference not in results:
        return pd.DataFrame()

    metric = cfg.primary_metric
    ref_folds = results[reference]["folds"]

    rows = []
    for name, data in results.items():
        if name == reference:
            continue
        test = paired_test(data["folds"], ref_folds, metric)
        rows.append({
            "experiment": name,
            "reference": reference,
            "metric": metric,
            "n_folds": test["n"],
            "mean_diff": test["mean_diff"],
            "p_ttest": test["p_ttest"],
            "p_wilcoxon": test["p_wilcoxon"],
            "significant_05": (test["p_ttest"] < 0.05
                               if not np.isnan(test["p_ttest"]) else False),
        })
    return pd.DataFrame(rows).sort_values("p_ttest")


def _save(fig, path_png, path_pdf=None, dpi=300):
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    if path_pdf is not None:
        fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bar(results: Dict, cfg, output_dir: Path):
    primary = cfg.primary_metric
    names, means, stds, colors = [], [], [], []
    family_colors = {"baseline": "#7F8C8D",
                     "fusion": "#2980B9",
                     "novelty": "#C0392B"}

    for name, data in results.items():
        s = _flatten_summary(data["summary"])
        if primary not in s:
            continue
        names.append(name)
        means.append(s[primary]["mean"])
        stds.append(s[primary]["std"])
        colors.append(family_colors.get(data.get("family", "-"), "#95A5A6"))

    if not names:
        return

    order = np.argsort(means)
    if not cfg.lower_is_better:
        order = order[::-1]
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(names))))
    ax.barh(names, means, xerr=stds, color=colors, alpha=0.85,
            edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xlabel(primary, fontsize=11, fontweight="bold")
    ax.set_title(f"Model comparison — {primary} (mean ± std across folds)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_axisbelow(True)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.005, i, f"{m:.3f}", va="center",
                fontsize=8, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, output_dir / f"comparison_{primary}.png",
          output_dir / f"comparison_{primary}.pdf")


def plot_ablation_bar(ablation_df: pd.DataFrame, cfg, output_dir: Path,
                      reference_label="full"):
    primary = cfg.primary_metric
    col = f"{primary}_mean"
    if col not in ablation_df.columns or ablation_df.empty:
        return

    df = ablation_df.sort_values(col, ascending=cfg.lower_is_better)
    ref_val = df.loc[df["ablation"] == reference_label, col]
    ref_val = ref_val.iloc[0] if len(ref_val) else df[col].iloc[0]

    colors = ["#C0392B" if v == ref_val else "#7F8C8D"
              for v in df[col].values]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(df))))
    ax.barh(df["ablation"], df[col], color=colors, alpha=0.85,
            edgecolor="black", linewidth=0.5)
    ax.axvline(ref_val, color="red", linestyle="--", linewidth=1.0,
               label=f"full model ({ref_val:.3f})")
    ax.set_xlabel(primary, fontsize=11, fontweight="bold")
    ax.set_title(f"Ablation study — {primary}",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, output_dir / f"ablation_{primary}.png",
          output_dir / f"ablation_{primary}.pdf")


def save_tables(results, ablation, significance, tables_dir: Path):
    tables_dir.mkdir(parents=True, exist_ok=True)
    if results is not None and not results.empty:
        results.to_csv(tables_dir / "comparison_table.csv", index=False)
    if ablation is not None and not ablation.empty:
        ablation.to_csv(tables_dir / "ablation_table.csv", index=False)
    if significance is not None and not significance.empty:
        significance.to_csv(tables_dir / "significance_table.csv",
                            index=False)


def save_json(results, ablation, cfg, path: Path):
    import json
    out = {
        "task": cfg.task,
        "primary_metric": cfg.primary_metric,
        "experiments": results,
        "ablations": ablation,
    }
    path.write_text(json.dumps(out, indent=2, default=str))