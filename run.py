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
# --- numpy compat shim ---
import warnings as _warnings
import numpy as _np

with _warnings.catch_warnings():
    _warnings.simplefilter("ignore", FutureWarning)

    _aliases = {
        "long":      int,
        "ulong":     int,
        "bool":      bool,
        "int":       int,
        "float":     float,
        "complex":   complex,
        "object":    object,
        "str":       str,
        "unicode":   str,
        "longlong":  _np.int64,
        "ulonglong": _np.uint64,
        "int_":      _np.int64,
        "uint":      _np.uint64,
        "ubyte":     _np.uint8,
        "ushort":    _np.uint16,
        "uintc":     _np.uint32,
        "intc":      _np.int32,
    }
    for _name, _target in _aliases.items():
        if not hasattr(_np, _name):
            setattr(_np, _name, _target)

del _name, _target, _warnings
# --- end shim ---

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_utils import build_master_table, grouped_folds
from feature_extractors import (extract_egemaps, extract_ssl_embeddings,
                                extract_text_embeddings, LazyEmbeddings)
from experiments import default_experiments, run_all_experiments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--demo-csv", required=True)
    ap.add_argument("--transcriptions-csv", required=True)
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--speaker-col", default=None)
    ap.add_argument("--session-col", default=None)
    ap.add_argument("--transcript-file-col", default="file_stem")
    ap.add_argument("--transcript-text-col", default="transcript")
    ap.add_argument("--task", choices=["classification", "regression"],
                    default="classification")
    ap.add_argument("--n-classes", type=int, default=2)
    ap.add_argument("--aggregation-unit",
                    choices=["auto", "question", "session", "speaker"],
                    default="speaker")
    ap.add_argument("--n-folds", type=int, default=2)
    ap.add_argument("--ssl-model", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--text-models", nargs="+",
                    default=["emilyalsentzer/Bio_ClinicalBERT"])
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mode", choices=["experiments", "ablation", "full"],
                    default="experiments")
    ap.add_argument("--include-grid", action="store_true")
    ap.add_argument("--list-experiments", action="store_true")
    ap.add_argument("--list-ablations", action="store_true")
    ap.add_argument("--max-audio-seconds", type=float, default=30.0)
    ap.add_argument("--ssl-sample-rate", type=int, default=16000)
    ap.add_argument("--text-max-length", type=int, default=512)
    ap.add_argument("--ssl-pool", type=lambda x: x.lower() != "false",
                    default=True)
    ap.add_argument("--ssl-half", action="store_true")
    ap.add_argument("--ssl-chunk-seconds", type=float, default=30.0)
    ap.add_argument("--text-pool", choices=["mean", "cls", "none"],
                    default="mean")
    ap.add_argument("--embed-dtype", choices=["float16", "float32"],
                    default="float16")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Config bridge
# ---------------------------------------------------------------------------
class Config:
    pass


def build_cfg(args):
    cfg = Config()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    cfg.text_model_names = list(args.text_models)
    cfg.n_outputs = args.n_classes if args.task == "classification" else 1
    cfg.resolved_aggregation_unit = (
        "speaker" if args.aggregation_unit == "auto"
        else args.aggregation_unit)
    cfg.use_amp = True
    cfg.weight_decay = 1e-2
    cfg.grad_clip = 1.0
    # fusion defaults
    cfg.fusion_d_model = getattr(args, "fusion_d_model", 128)
    cfg.fusion_dropout = getattr(args, "fusion_dropout", 0.1)
    cfg.fusion_n_heads = getattr(args, "fusion_n_heads", 4)
    # mamba defaults
    cfg.mamba_d_model = getattr(args, "mamba_d_model", 128)
    cfg.mamba_n_layers = getattr(args, "mamba_n_layers", 4)
    cfg.mamba_d_state = getattr(args, "mamba_d_state", 16)
    cfg.mamba_d_conv = getattr(args, "mamba_d_conv", 4)
    cfg.mamba_expand = getattr(args, "mamba_expand", 2)
    cfg.mamba_dropout = getattr(args, "mamba_dropout", 0.1)
    return cfg


# ---------------------------------------------------------------------------
# Stem utilities
# ---------------------------------------------------------------------------
def _keys(obj):
    if obj is None:
        return set()
    if hasattr(obj, "keys"):
        return set(obj.keys())
    if isinstance(obj, pd.DataFrame):
        return set(obj.index)
    return set()


def filter_df_to_caches(df, folds, cfg, text_emb, audio_emb, egemaps_df):
    """Keep only rows present in every non-empty feature source."""
    sources = [*text_emb.values(), audio_emb, egemaps_df]
    valid = None
    for s in sources:
        ks = _keys(s)
        if not ks:
            continue
        valid = ks if valid is None else (valid & ks)

    if valid is None:
        raise RuntimeError("No feature source produced any stems.")

    before = len(df)
    df = df[df["file_stem"].isin(valid)].reset_index(drop=True)
    if len(df) < before:
        print(f"[features] dropped {before - len(df)} rows missing from caches")
    folds = grouped_folds(df, cfg)
    return df, folds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = build_cfg(args)

    if args.list_experiments:
        for s in default_experiments():
            print(f"{s.name:20s}  {s.family:9s}  {s.description}")
        return

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}\nCONFIG\n{'=' * 70}")
    print(f"  task    : {cfg.task}")
    print(f"  device  : {cfg.device}")
    print(f"  ssl     : {cfg.ssl_model}")
    print(f"  text    : {cfg.text_model_names}")
    print(f"  pool    : ssl={cfg.ssl_pool}  text={cfg.text_pool}")
    print(f"  dtype   : {cfg.embed_dtype}")

    # ---- 1. data ----
    print(f"\n{'=' * 70}\nDATA\n{'=' * 70}")
    df = build_master_table(cfg)
    folds = grouped_folds(df, cfg)
    df = df.reset_index(drop=True)

    # ---- 2. features: TEXT -> SSL -> eGeMAPS ----
    print(f"\n{'=' * 70}\nFEATURES\n{'=' * 70}")

    # 2a. text
    print("\n[features] text embeddings ...")
    text_emb = {}
    for mname in cfg.text_model_names:
        emb = extract_text_embeddings(df, mname, cfg)
        text_emb[mname] = emb
        print(f"  {mname}: {len(emb)} sequences")

    if text_emb and any(len(v) > 0 for v in text_emb.values()):
        primary_text = next(m for m, v in text_emb.items() if len(v) > 0)
        print(f"[features] primary text: {primary_text}")
    else:
        primary_text = None
        print("[features] WARNING: no text embeddings; "
              "continuing without text")

    # 2b. ssl
    print("\n[features] SSL audio embeddings ...")
    audio_emb = extract_ssl_embeddings(df, cfg)
    print(f"  {cfg.ssl_model}: {len(audio_emb)} sequences")

    # 2c. egemaps
    print("\n[features] eGeMAPS ...")
    egemaps_df = extract_egemaps(df, cfg)
    print(f"  egemaps: {len(egemaps_df)} rows")

    # ---- 3. cross-source sanity + df filter ----
    n_text = sum(len(v) for v in text_emb.values())
    n_ssl = len(audio_emb)
    n_eg = len(egemaps_df)
    print(f"\n[features] summary: text={n_text} ssl={n_ssl} egemaps={n_eg}")

    if n_text == 0 and n_ssl == 0 and n_eg == 0:
        raise RuntimeError("All feature extractors produced empty results.")

    df, folds = filter_df_to_caches(df, folds, cfg, text_emb, audio_emb,
                                    egemaps_df)
    print(f"[features] working set: {len(df)} rows, {len(folds)} folds")

    # ---- 4. experiments ----
    print(f"\n{'=' * 70}\nEXPERIMENTS\n{'=' * 70}")
    results = run_all_experiments(
        df, folds, cfg, egemaps_df, audio_emb, text_emb, primary_text)

    # ---- 5. save results summary ----
    out = Path(cfg.output_dir) / "results_summary.csv"
    rows = []
    for name, r in results.items():
        for f in r.get("folds", []):
            row = {"model": name, "family": r["family"]}
            row.update({k: v for k, v in f.items() if not isinstance(v, dict)})
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n[report] wrote {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)