"""
Main CLI for the speaker-level multi-modal clinical speech pipeline.

Modes
-----
  --mode experiments   Run the default experiment suite.
  --mode ablation      Run the single-component ablation study.
  --mode full          Run both, plus the novelty grid (if --include-grid).
  --list-experiments   Print the default experiment names and exit.
  --list-ablations     Print the ablation labels and exit.

Pipeline
--------
1. Chunk-level master table is built from demo.csv + transcriptions.csv.
2. Features are extracted per chunk: SSL audio, text embeddings, eGeMAPS.
3. Chunks are collapsed to one row per speaker.
4. Grouped CV folds are formed over speakers.
5. Models are trained and evaluated at the speaker level.
6. OOF predictions are written per fold, one row per speaker.
"""

# --- numpy compat shim ---------------------------------------------------
import warnings
import numpy as _np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    _aliases = {
        "long": int, "ulong": int, "bool": bool, "int": int,
        "float": float, "complex": complex, "object": object,
        "str": str, "unicode": str,
        "longlong": _np.int64, "ulonglong": _np.uint64,
        "int_": _np.int64, "uint": _np.uint64,
        "ubyte": _np.uint8, "ushort": _np.uint16,
        "uintc": _np.uint32, "intc": _np.int32,
    }
    for _name, _target in _aliases.items():
        if getattr(_np, _name, None) is None:
            setattr(_np, _name, _target)
del _name, _target
# --- end shim ------------------------------------------------------------

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from data_utils import (build_master_table, build_speaker_sequences,
                        grouped_folds)
from feature_extractors import (extract_egemaps, extract_ssl_embeddings,
                                extract_text_embeddings)
from experiments import default_experiments, run_all_experiments
from ablations import (ABLATION_PRESETS, run_ablation_study,
                       run_ablation_grid)
from reporting import (comparison_table, ablation_table, significance_table,
                       plot_comparison_bar, plot_ablation_bar,
                       save_tables, save_json)


# =========================================================================
# CLI
# =========================================================================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Leakage-free multi-modal clinical speech pipeline "
                    "(speaker-level).")

    # inputs
    ap.add_argument("--wav-dir", type=str, required=True)
    ap.add_argument("--demo-csv", type=str, required=True)
    ap.add_argument("--transcriptions-csv", type=str, required=True)

    # columns
    ap.add_argument("--label-col", type=str, default=None)
    ap.add_argument("--speaker-col", type=str, default="speaker_id")
    ap.add_argument("--session-col", type=str, default=None)
    ap.add_argument("--transcript-file-col", type=str, default="utt_id")
    ap.add_argument("--transcript-text-col", type=str, default="transcript")

    # task
    ap.add_argument("--task", choices=["classification", "regression"],
                    default="classification")
    ap.add_argument("--n-classes", type=int, default=2)
    ap.add_argument("--aggregation-unit",
                    choices=["auto", "question", "session", "speaker"],
                    default="auto")

    # CV
    ap.add_argument("--n-folds", type=int, default=5)

    # encoders
    ap.add_argument("--ssl-model", type=str,
                    default="facebook/wav2vec2-base-960h")
    ap.add_argument(
        "--text-models", nargs="+",
        default=[
            "emilyalsentzer/Bio_ClinicalBERT",
            "roberta-base",
            "bert-base-uncased",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        ],
        help="Hugging Face text encoders.",
    )

    # feature-cache keys
    ap.add_argument("--max-audio-seconds", type=float, default=30.0)
    ap.add_argument("--ssl-sample-rate", type=int, default=16000)
    ap.add_argument("--text-max-length", type=int, default=512)

    # feature options
    ap.add_argument("--ssl-pool", type=lambda x: x.lower() != "false",
                    default=True)
    ap.add_argument("--ssl-half", action="store_true")
    ap.add_argument("--ssl-chunk-seconds", type=float, default=30.0)
    ap.add_argument("--text-pool", choices=["mean", "cls", "none"],
                    default="mean")
    ap.add_argument("--embed-dtype", choices=["float16", "float32"],
                    default="float16")
    ap.add_argument("--force-extract", action="store_true",
                    help="Ignore feature caches and re-extract everything.")

    # paths
    ap.add_argument("--output-dir", type=str, default="results")
    ap.add_argument("--cache-dir", type=str, default="cache")

    # training
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)

    # mode
    ap.add_argument("--mode", choices=["experiments", "ablation", "full"],
                    default="experiments")
    ap.add_argument("--include-grid", action="store_true")
    ap.add_argument("--list-experiments", action="store_true")
    ap.add_argument("--list-ablations", action="store_true")
    ap.add_argument("--diagnose", action="store_true",
                    help="Print detailed feature/speaker diagnostics "
                         "and exit before training.")
    return ap


# =========================================================================
# Config
# =========================================================================

def make_config(args) -> Config:
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
    cfg.text_model_names = list(args.text_models)

    cfg.max_audio_seconds = args.max_audio_seconds
    cfg.ssl_sample_rate = args.ssl_sample_rate
    cfg.text_max_length = args.text_max_length

    cfg.ssl_pool = args.ssl_pool
    cfg.ssl_half = args.ssl_half
    cfg.ssl_chunk_seconds = args.ssl_chunk_seconds
    cfg.text_pool = args.text_pool
    cfg.embed_dtype = args.embed_dtype
    cfg.force_extract = args.force_extract

    cfg.output_dir = Path(args.output_dir)
    cfg.cache_dir = Path(args.cache_dir)

    cfg.device = args.device
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    return cfg


# =========================================================================
# Early exits
# =========================================================================

def handle_early_exits(args) -> bool:
    if args.list_experiments:
        print("Default experiments:")
        for spec in default_experiments():
            print(f"  {spec.name:20s}  [{spec.family:8s}]  "
                  f"{spec.description}")
        return True
    if args.list_ablations:
        print("Ablation presets:")
        for label, preset in ABLATION_PRESETS.items():
            print(f"  {label:22s}  (novelty preset '{preset}')")
        return True
    return False


# =========================================================================
# Feature extraction
# =========================================================================

def extract_all_features(chunk_df, cfg):
    """
    Extract chunk-level features and collapse to one row per speaker.
    Returns (speaker_df, primary_text).
    """
    print(f"\n{'=' * 70}\nFEATURES\n{'=' * 70}")

    # ---- text ----
    print("\n[features] text embeddings ...")
    text_emb = {}
    for mname in cfg.text_model_names:
        try:
            emb = extract_text_embeddings(chunk_df, mname, cfg)
        except Exception:
            print(f"  [text] {mname} raised:")
            traceback.print_exc()
            emb = {}
        text_emb[mname] = emb
        print(f"  {mname}: {len(emb)} sequences")

    usable = {m: e for m, e in text_emb.items() if len(e) > 0}
    if not usable:
        _feature_failure_report("text", text_emb, {})
        raise RuntimeError("No text embeddings were produced.")
    primary_text = next(iter(usable))
    text_emb_primary = usable[primary_text]
    print(f"[features] primary text encoder: {primary_text}")

    # ---- SSL ----
    print("\n[features] SSL audio embeddings ...")
    try:
        audio_emb = extract_ssl_embeddings(chunk_df, cfg)
    except Exception:
        print("  [ssl] extraction raised:")
        traceback.print_exc()
        audio_emb = {}
    print(f"  {cfg.ssl_model_name}: {len(audio_emb)} sequences")

    # ---- eGeMAPS ----
    print("\n[features] eGeMAPS ...")
    try:
        egemaps_df = extract_egemaps(chunk_df, cfg)
    except Exception:
        print("  [egemaps] extraction raised:")
        traceback.print_exc()
        egemaps_df = pd.DataFrame()

    print(f"  egemaps: {len(egemaps_df)} rows")

    # ---- summary ----
    n_text = sum(len(v) for v in text_emb.values())
    n_ssl = len(audio_emb)
    n_eg = len(egemaps_df)
    print("\n[features] summary:")
    print(f"  text   : {n_text} files across {len(text_emb)} encoder(s)")
    print(f"  ssl    : {n_ssl} files")
    print(f"  egemaps: {n_eg} rows")

    if n_text == 0:
        raise RuntimeError("No text embeddings were produced.")
    if n_ssl == 0:
        raise RuntimeError("No SSL embeddings were produced.")
    if n_eg == 0:
        raise RuntimeError("No eGeMAPS features were produced.")

    # ---- quick stem-overlap check ----
    _stem_overlap_report(chunk_df, audio_emb, text_emb_primary, egemaps_df)

    # ---- collapse to speakers ----
    speaker_df = build_speaker_sequences(
        chunk_df, audio_emb, text_emb_primary, egemaps_df, cfg)
    speaker_df = speaker_df.reset_index(drop=True)

    # ---- post-collapse diagnostics ----
    _speaker_df_report(speaker_df)

    return speaker_df, primary_text


def _feature_failure_report(kind, text_emb, audio_emb):
    print(f"\n[features] {kind} FAILURE REPORT")
    for m, e in text_emb.items():
        print(f"  text encoder {m}: {len(e)} sequences")


def _stem_overlap_report(chunk_df, audio_emb, text_emb, egemaps_df):
    """Show how many of the chunk stems are found in each feature dict."""
    stems = chunk_df["file_stem"].tolist()
    n = len(stems)
    in_a = sum(1 for s in stems if s in audio_emb)
    in_t = sum(1 for s in stems if s in text_emb)
    in_e = sum(1 for s in stems if s in set(egemaps_df.index))
    print("\n[features] stem overlap with chunk_df:")
    print(f"  chunk stems total       : {n}")
    print(f"  found in audio_emb      : {in_a}")
    print(f"  found in text_emb       : {in_t}")
    print(f"  found in egemaps_df     : {in_e}")

    if in_a < n or in_t < n or in_e < n:
        sample = stems[0] if stems else "<none>"
        print(f"  sample stem             : {sample!r}")
        print(f"    in audio_emb          : {sample in audio_emb}")
        print(f"    in text_emb           : {sample in text_emb}")
        print(f"    in egemaps_df         : "
              f"{sample in set(egemaps_df.index)}")
        if audio_emb:
            k = next(iter(audio_emb))
            print(f"    audio_emb first key   : {k!r}")
        if text_emb:
            k = next(iter(text_emb))
            print(f"    text_emb first key    : {k!r}")
        if len(egemaps_df) > 0:
            k = egemaps_df.index[0]
            print(f"    egemaps first index   : {k!r}")


def _speaker_df_report(speaker_df):
    if speaker_df.empty:
        print("\n[speakers] speaker_df is empty")
        return
    n_empty_a = int((speaker_df["audio_seqs"].apply(len) == 0).sum())
    n_empty_t = int((speaker_df["text_seqs"].apply(len) == 0).sum())
    print(f"\n[speakers] {len(speaker_df)} speakers")
    print(f"  empty audio_seqs: {n_empty_a}")
    print(f"  empty text_seqs : {n_empty_t}")
    for _, row in speaker_df.head(5).iterrows():
        print(f"    {row['speaker_id']}: "
              f"n_chunks={row['n_chunks']}, "
              f"n_audio={len(row['audio_seqs'])}, "
              f"n_text={len(row['text_seqs'])}")


# =========================================================================
# Main
# =========================================================================

def main():
    ap = build_argparser()
    args = ap.parse_args()

    if handle_early_exits(args):
        return

    cfg = make_config(args)
    tables_dir = cfg.output_dir / "tables"
    figs_dir = cfg.output_dir / "figures"

    # ---- 1. data ----
    print(f"\n{'=' * 70}\nDATA\n{'=' * 70}")
    chunk_df = build_master_table(cfg)

    # ---- 2. features + speaker table ----
    speaker_df, primary_text = extract_all_features(chunk_df, cfg)

    # ---- 3. early exit for --diagnose ----
    if args.diagnose:
        print(f"\n[done] --diagnose requested; exiting before training.")
        return

    # ---- 4. folds ----
    folds = grouped_folds(speaker_df, cfg)

    # ---- 5. experiments ----
    exp_results = {}
    if args.mode in ("experiments", "full"):
        print(f"\n{'=' * 70}\nEXPERIMENTS\n{'=' * 70}")
        exp_results = run_all_experiments(
            speaker_df, folds, cfg, primary_text)

    # ---- 6. ablations ----
    ab_results = {}
    if args.mode in ("ablation", "full"):
        print(f"\n{'=' * 70}\nABLATION STUDY\n{'=' * 70}")
        ab_results = run_ablation_study(speaker_df, folds, cfg)

        if args.include_grid:
            print(f"\n{'=' * 70}\nNOVELTY GRID (2x3x3)\n{'=' * 70}")
            grid = run_ablation_grid(speaker_df, folds, cfg)
            for k, v in grid.items():
                ab_results[f"grid_{k}"] = v

    # ---- 7. reporting ----
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