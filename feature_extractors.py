"""
feature_extractors.py

Feature extraction for the speaker-level mamba pipeline.

Extractors, in the order they should be called:
    1. extract_text_embeddings   -> dict[stem, (T, D)]  float16
    2. extract_ssl_embeddings    -> dict[stem, (T, D)]  float16
    3. extract_egemaps           -> pd.DataFrame        functionals

Each extractor has one top-level try/except that prints a full traceback
and returns an empty result on failure, so one broken stage never aborts
the others. The caller decides what to do with missing features.

Design contract
---------------
Every embedding returned by the text and SSL extractors is a SEQUENCE
of shape (T, D). Pooling to (D,) is the caller's job — models that need
pooled input pool inside their forward pass; classical models pool in
run_one_experiment.

This contract is what makes both the classical and sequence paths work.
"""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import traceback


# =========================================================================
# Pooling helpers — used by callers, not by the extractors
# =========================================================================
def mean_pool(seq: np.ndarray) -> np.ndarray:
    """Mean over time. (T, D) -> (D,). No-op on 1-D input."""
    return seq if seq.ndim == 1 else seq.mean(axis=0)


def mean_pool_text(seq: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Mean over tokens, honoring attention mask. (T, D) + (T,) -> (D,)."""
    if seq.ndim == 1:
        return seq
    if mask is None:
        return seq.mean(axis=0)
    m = mask.astype(np.float32)[:, None]
    denom = np.clip(m.sum(), 1e-9, None)
    return (seq * m).sum(axis=0) / denom


def cls_pool_text(seq: np.ndarray) -> np.ndarray:
    """First token embedding. (T, D) -> (D,)."""
    return seq if seq.ndim == 1 else seq[0]


def _maybe_cast(arr: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "float16":
        return arr.astype(np.float16, copy=False)
    if dtype == "float32":
        return arr.astype(np.float32, copy=False)
    return arr


# =========================================================================
# 1. Text embeddings — return sequences (T, D)
# =========================================================================
@torch.no_grad()
def extract_text_embeddings(df, model_name, cfg):
    """
    Returns {file_stem: np.ndarray of shape (T, D)}.

    T = number of tokens the tokenizer produced (<= cfg.text_max_length).
    D = hidden size of the text encoder (768 for BERT-base family).

    Nothing is pooled here. Callers that need a single vector per chunk
    call mean_pool_text / cls_pool_text themselves.
    """
    from transformers import AutoTokenizer, AutoModel

    safe = model_name.replace("/", "_")
    cache = Path(cfg.cache_dir) / f"text_{safe}"
    cache.mkdir(parents=True, exist_ok=True)

    force = bool(getattr(cfg, "force_extract", False))

    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(cfg.device)
        model.eval()

        out: Dict[str, np.ndarray] = {}
        for i, row in df.iterrows():
            stem = row["file_stem"]
            npy = cache / f"{stem}.npy"

            if npy.exists() and not force:
                out[stem] = np.load(npy)
                continue

            text = row["transcript"]
            if not isinstance(text, str) or not text.strip():
                continue

            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=cfg.text_max_length, padding=False)
            enc = {k: v.to(cfg.device) for k, v in enc.items()}
            try:
                h = model(**enc).last_hidden_state.squeeze(0)   # [T, D]
            except Exception as e:
                print(f"[text] failed on {stem}: {e}")
                continue

            # ---- keep the full sequence; do NOT pool here ----
            arr = h.detach().float().cpu().numpy()              # (T, D)
            arr = _maybe_cast(arr, str(getattr(cfg, "embed_dtype",
                                               "float16")))
            np.save(npy, arr)
            out[stem] = arr

            if (i + 1) % 100 == 0:
                print(f"[text:{safe}] {i + 1}/{len(df)}")

        if out:
            sample = next(iter(out.values()))
            print(f"[text:{safe}] {len(out)} sequences cached in {cache} "
                  f"(shape={sample.shape}, dtype={sample.dtype})")
        del model
        if str(cfg.device).startswith("cuda"):
            torch.cuda.empty_cache()
        return out

    except Exception:
        print(f"[text:{safe}] FAILED for {model_name}")
        traceback.print_exc()
        return {}


# =========================================================================
# 2. SSL embeddings — return sequences (T, D)
# =========================================================================
def _load_ssl_model(model_name: str, device: str, half: bool = False):
    from transformers import Wav2Vec2Model, HubertModel, WhisperModel

    kwargs = {"torch_dtype": torch.float16} if half else {}
    name = model_name.lower()
    if "wav2vec2" in name:
        model = Wav2Vec2Model.from_pretrained(model_name, **kwargs)
    elif "hubert" in name:
        model = HubertModel.from_pretrained(model_name, **kwargs)
    elif "whisper" in name:
        model = WhisperModel.from_pretrained(model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported SSL model: {model_name}")
    model.eval()
    return model.to(device)


@torch.no_grad()
def _ssl_forward_chunked(model, wav: torch.Tensor, cfg) -> torch.Tensor:
    """
    Run an SSL model over (1, T) in overlapping chunks, return (T_out, D).
    """
    chunk_seconds = float(getattr(cfg, "ssl_chunk_seconds", 30.0))
    chunk_len = max(1, int(chunk_seconds * cfg.ssl_sample_rate))
    stride = max(1, chunk_len // 2)
    use_half = bool(getattr(cfg, "ssl_half", False))

    L = wav.shape[1]
    parts: List[torch.Tensor] = []
    pos = 0
    while pos < L:
        end = min(pos + chunk_len, L)
        chunk = wav[:, pos:end]
        if use_half:
            chunk = chunk.half()
        chunk = chunk.to(cfg.device)

        out_dict = model(chunk)
        h = (out_dict.last_hidden_state
             if hasattr(out_dict, "last_hidden_state") else out_dict[0])
        parts.append(h.squeeze(0).detach().float().cpu())

        del chunk, out_dict, h
        if str(cfg.device).startswith("cuda"):
            torch.cuda.empty_cache()

        if end == L:
            break
        pos += stride

    return torch.cat(parts, dim=0)


@torch.no_grad()
def extract_ssl_embeddings(df: pd.DataFrame, cfg) -> Dict[str, np.ndarray]:
    """
    Returns {file_stem: np.ndarray of shape (T, D)}.

    T = number of SSL frames (about 50 per second at 16 kHz for wav2vec2).
    D = hidden size (768 for wav2vec2-base, 1024 for wav2vec2-large).

    Nothing is pooled here. Callers pool when they need to.
    """
    import torchaudio

    dtype = str(getattr(cfg, "embed_dtype", "float16"))
    half = bool(getattr(cfg, "ssl_half", False))

    # Cache key: encoder + preprocessing settings that change the output.
    safe = (
        cfg.ssl_model_name.replace("/", "_")
        + f"_sr{cfg.ssl_sample_rate}"
        + f"_max{int(cfg.max_audio_seconds)}"
        + f"_{dtype}"
    )
    cache = Path(cfg.cache_dir) / f"ssl_{safe}"
    cache.mkdir(parents=True, exist_ok=True)
    force = bool(getattr(cfg, "force_extract", False))

    out: Dict[str, np.ndarray] = {}
    try:
        model = _load_ssl_model(cfg.ssl_model_name, cfg.device, half=half)

        for i, row in df.iterrows():
            stem = row["file_stem"]
            npy = cache / f"{stem}.npy"

            if npy.exists() and not force:
                out[stem] = np.load(npy)
                continue

            wav, sr = torchaudio.load(str(row["file_path"]))
            if sr != cfg.ssl_sample_rate:
                wav = torchaudio.functional.resample(
                    wav, sr, cfg.ssl_sample_rate)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)

            max_len = int(cfg.max_audio_seconds * cfg.ssl_sample_rate)
            if wav.shape[1] > max_len:
                wav = wav[:, :max_len]

            h = _ssl_forward_chunked(model, wav, cfg)   # (T, D) CPU float32
            arr = h.numpy()
            # ---- keep the full sequence; do NOT pool here ----
            arr = _maybe_cast(arr, dtype)
            np.save(npy, arr)
            out[stem] = arr

            del wav, h, arr
            if str(cfg.device).startswith("cuda"):
                torch.cuda.empty_cache()

            if (i + 1) % 100 == 0:
                print(f"[ssl:{safe}] {i + 1}/{len(df)}")

        if out:
            sample = next(iter(out.values()))
            print(f"[ssl:{safe}] {len(out)} sequences cached in {cache} "
                  f"(shape={sample.shape}, dtype={sample.dtype})")

        del model
        if str(cfg.device).startswith("cuda"):
            torch.cuda.empty_cache()
        return out

    except Exception:
        print(f"[ssl:{safe}] FAILED for {cfg.ssl_model_name}")
        traceback.print_exc()
        return {}


# =========================================================================
# 3. eGeMAPS — one row of functionals per file (tabular, unchanged)
# =========================================================================
def extract_egemaps(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    One row of eGeMAPSv02 functionals per file. Returns a DataFrame
    indexed by file_stem and aligned to df order.
    """
    cache = Path(cfg.cache_dir) / "egemaps.csv"

    try:
        import opensmile

        force = bool(getattr(cfg, "force_extract", False))
        if cache.exists() and not force:
            cached = pd.read_csv(cache, index_col=0)
            if set(df["file_stem"]).issubset(cached.index):
                print(f"[egemaps] loaded cache ({cached.shape})")
                return cached.loc[df["file_stem"]]

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        feats: Dict[str, dict] = {}
        for i, row in df.iterrows():
            try:
                f = smile.process_file(str(row["file_path"]))
                feats[row["file_stem"]] = f.iloc[0].to_dict()
            except Exception as e:
                print(f"[egemaps] failed on {row['file_stem']}: {e}")
                feats[row["file_stem"]] = {}
            if (i + 1) % 100 == 0:
                print(f"[egemaps] {i + 1}/{len(df)}")

        out = pd.DataFrame(feats).T
        out.to_csv(cache)
        return out.loc[df["file_stem"]]

    except Exception:
        print("[egemaps] FAILED")
        traceback.print_exc()
        return pd.DataFrame(index=df["file_stem"])


# =========================================================================
# Orchestrator
# =========================================================================
def extract_all_features(df: pd.DataFrame,
                         cfg,
                         text_models: Optional[List[str]] = None
                         ) -> Dict[str, object]:
    """
    Run all three extractors in order: text, ssl, egemaps.

    Returns
    -------
    dict with keys:
        'text'    -> dict[model_name, dict[stem, (T, D)]]
        'ssl'     -> dict[stem, (T, D)]
        'egemaps' -> pd.DataFrame
    """
    if text_models is None:
        text_models = getattr(cfg, "text_models", None) or [
            "emilyalsentzer/Bio_ClinicalBERT",
        ]

    results: Dict[str, object] = {
        "text": {},
        "ssl": {},
        "egemaps": pd.DataFrame(),
    }

    print("\n[features] text embeddings ...")
    for name in text_models:
        emb = extract_text_embeddings(df, name, cfg)
        results["text"][name] = emb
        print(f"  {name}: {len(emb)} sequences")

    print("\n[features] ssl embeddings ...")
    results["ssl"] = extract_ssl_embeddings(df, cfg)
    print(f"  {cfg.ssl_model_name}: {len(results['ssl'])} sequences")

    print("\n[features] egemaps ...")
    results["egemaps"] = extract_egemaps(df, cfg)
    print(f"  egemaps: {len(results['egemaps'])} rows")

    n_text = sum(len(v) for v in results["text"].values())
    n_ssl = len(results["ssl"])
    n_eg = len(results["egemaps"])

    print("\n[features] summary:")
    print(f"  text  : {n_text} files across "
          f"{len(results['text'])} model(s)")
    print(f"  ssl   : {n_ssl} files")
    print(f"  egemaps: {n_eg} files")

    if n_text == 0 and n_ssl == 0 and n_eg == 0:
        raise RuntimeError(
            "All feature extractors returned empty results. "
            "See the tracebacks above for the root cause."
        )
    if n_text == 0:
        print("[features] WARNING: no text embeddings.")
    if n_ssl == 0:
        print("[features] WARNING: no SSL embeddings.")
    if n_eg == 0:
        print("[features] WARNING: no eGeMAPS.")

    return results