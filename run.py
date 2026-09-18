"""
Main CLI.

Modes
-----
  --mode experiments   Run the default experiment suite (baselines + fusion
                       + novel model).
  --mode ablation      Run the single-component ablation study on the novel
                       model.
  --mode full          Run both, plus the small grid.
  --list-experiments   Print the default experiment names and exit.
  --list-ablations     Print the ablation labels and exit.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from data_utils import build_master_table, grouped_folds
from feature_extractors import (extract_egemaps, extract_ssl_embeddings,
                                extract_text_embeddings)
from experiments import (default_experiments, run_all_experiments)
from ablations import (ABLATION_PRESETS, run_ablation_study,
                       run_ablation_grid)
from reporting import (comparison_table, ablation_table, significance_table,
                       plot_comparison_bar, plot_ablation_bar,
                       save_tables, save_json)


def main():
    ap = argparse.ArgumentParser(
        description="Leakage-free multi-modal clinical speech pipeline "
                    "with experiment and ablation runners.")
    ap.add_argument("--wav-dir", type=str, required=True)
    ap.add_argument("--demo-csv", type=str, required=True)
    ap.add_argument("--transcriptions-csv", type=str, required=True)
    ap.add_argument("--label-col", type=str, default=None)
    ap.add_argument("--speaker-col", type=str, default="speaker_id")
    ap.add_argument("--session-col", type=str, default=None)
    ap.add_argument("--transcript-file-col", type=str, default="utt_id")
    ap.add_argument("--transcript-text-col", type=str, default="transcript")
    ap.add_argument("--task", choices=["classification", "regression"],
                    default="classification")
    ap.add_argument("--n-classes", type=int, default=2)
    ap.add_argument("--aggregation-unit",
                    choices=["auto", "question", "session", "speaker"],
                    default="auto")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--ssl-model", type=str,
                    default="facebook/wav2vec2-base-960h")
    ap.add_argument("--text-models", nargs="+", default=None)
    ap.add_argument("--output-dir", type=str, default="results")
    ap.add_argument("--cache-dir", type=str, default="cache")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mode", choices=["experiments", "ablation", "full"],
                    default="experiments")
    ap.add_argument("--include-grid", action="store_true",
                    help="Include the 18-config novelty grid (slow).")
    ap.add_argument("--list-experiments", action="store_true")
    ap.add_argument("--list-ablations", action="store_true")
    args = ap.parse_args()

    # early exits
    if args.list_experiments:
        print("Default experiments:")
        for spec in default_experiments():
            print(f"  {spec.name:20s}  [{spec.family:8s}]  "
                  f"{spec.description}")
        return
    if args.list_ablations:
        print("Ablation presets:")
        for label, preset in ABLATION_PRESETS.items():
            print(f"  {label:22s}  (novelty preset '{preset}')")
        return

    cfg = Config()
    cfg.wav_dir = Path(args.wav_dir)
    cfg.demo_csv = Path(args.demo_csv)
    cfg.transcriptions_csv = Path(args.transcriptions_csv)
    cfg.label_col = args.label_col
    cfg.speaker_col = args.speaker_col
    cfg.session_col = args.session_col
    cfg.transcript_file_col = args.transcript_file_col
    cfg.transcript_text_col = args.transcript_text_col
    cfg.task = args.task
    cfg.n_classes = args.n_classes
    cfg.aggregation_unit = args.aggregation_unit
    cfg.n_folds = args.n_folds
    cfg.ssl_model_name = args.ssl_model
    if args.text_models:
        cfg.text_model_names = args.text_models
    cfg.output_dir = Path(args.output_dir)
    cfg.cache_dir = Path(args.cache_dir)
    cfg.device = args.device
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = cfg.output_dir / "tables"
    figs_dir = cfg.output_dir / "figures"

    # ---- 1. data ----
    print(f"\n{'=' * 70}\nDATA\n{'=' * 70}")
    df = build_master_table(cfg)
    folds = grouped_folds(df, cfg)
    df = df.reset_index(drop=True)

    # ---- 2. features ----
    print(f"\n{'=' * 70}\nFEATURES\n{'=' * 70}")
    print("\n[features] eGeMAPS ...")
    egemaps_df = extract_egemaps(df, cfg)

    print("\n[features] SSL audio embeddings ...")
    audio_emb = extract_ssl_embeddings(df, cfg)

    print("\n[features] text embeddings ...")
    text_emb = {}
    for mname in cfg.text_model_names:
        try:
            text_emb[mname] = extract_text_embeddings(df, mname, cfg)
        except Exception as e:
            print(f"  skipped {mname}: {e}")

    if not text_emb:
        raise RuntimeError("No text embeddings produced.")
    primary_text = list(text_emb.keys())[0]
    text_emb_primary = text_emb[primary_text]
    print(f"[features] using text encoder: {primary_text}")

    # ---- 3. experiments ----
    exp_results = {}
    if args.mode in ("experiments", "full"):
        print(f"\n{'=' * 70}\nEXPERIMENTS\n{'=' * 70}")
        exp_results = run_all_experiments(
            df, folds, cfg, egemaps_df, audio_emb, text_emb_primary,
            primary_text)

    # ---- 4. ablations ----
    ab_results = {}
    if args.mode in ("ablation", "full"):
        print(f"\n{'=' * 70}\nABLATION STUDY\n{'=' * 70}")
        ab_results = run_ablation_study(
            df, folds, cfg, egemaps_df, audio_emb, text_emb_primary)

        if args.include_grid:
            print(f"\n{'=' * 70}\nNOVELTY GRID (2×3×3)\n{'=' * 70}")
            grid = run_ablation_grid(
                df, folds, cfg, egemaps_df, audio_emb, text_emb_primary)
            for k, v in grid.items():
                ab_results[f"grid_{k}"] = v

    # ---- 5. reporting ----
    print(f"\n{'=' * 70}\nREPORTING\n{'=' * 70}")

    comp_df = comparison_table(exp_results, cfg) if exp_results else None
    ab_df = ablation_table(ab_results, "full", cfg) if ab_results else None
    sig_df = (significance_table(exp_results, "tcm_mamba", cfg)
              if exp_results and "tcm_mamba" in exp_results else None)

    if comp_df is not None and not comp_df.empty:
        print("\nComparison table:")
        print(comp_df.to_string(index=False))
    if ab_df is not None and not ab_df.empty:
        print("\nAblation table:")
        print(ab_df.to_string(index=False))
    if sig_df is not None and not sig_df.empty:
        print("\nSignificance vs tcm_mamba:")
        print(sig_df.to_string(index=False))

    figs_dir.mkdir(parents=True, exist_ok=True)
    if exp_results:
        plot_comparison_bar(exp_results, cfg, figs_dir)
    if ab_df is not None and not ab_df.empty:
        plot_ablation_bar(ab_df, cfg, figs_dir)

    save_tables(comp_df, ab_df, sig_df, tables_dir)
    save_json(exp_results, ab_results, cfg,
              cfg.output_dir / "results.json")

    print(f"\n[done] outputs in {cfg.output_dir}")


if __name__ == "__main__":
    main()