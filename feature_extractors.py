"""
feature_extractors.py

Feature extraction for the mamba pipeline.

Extractors, in the order they should be called:
    1. extract_text_embeddings   -> LazyEmbeddings (pooled text vectors)
    2. extract_ssl_embeddings    -> LazyEmbeddings (pooled speech vectors)
    3. extract_egemaps           -> pd.DataFrame   (functionals)

Orchestrator:
    extract_all_features(df, cfg) -> dict with keys 'text', 'ssl', 'egemaps'

Each extractor has one top-level try/except that prints a full traceback
and returns an empty result on failure, so one broken stage never aborts
the others. The caller decides what to do with missing features.
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
import traceback


# =========================================================================
# Lazy dict-like view over a directory of .npy files
# =========================================================================
class LazyEmbeddings:
    """
    Dict-compatible view over a cache directory of .npy files.

    Supports: emb[stem], stem in emb, len(emb), iter(emb),
              .keys()/.items()/.values()/.get(), .dim(), .stack()

    Each __getitem__ hits disk. That keeps peak memory at one array
    regardless of dataset size.
    """

    def __init__(self, cache_dir: Path | str, stems: Iterable[str]):
        self.cache_dir = Path(cache_dir)
        self.stems: List[str] = list(stems)

    def _path(self, stem: str) -> Path:
        return self.cache_dir / f"{stem}.npy"

    def __getitem__(self, stem: str) -> np.ndarray:
        return np.load(self._path(stem))

    def __contains__(self, stem: str) -> bool:
        return stem in self.stems

    def __len__(self) -> int:
        return len(self.stems)

    def __iter__(self):
        return iter(self.stems)

    def __repr__(self) -> str:
        return f"LazyEmbeddings(n={len(self.stems)}, dir={self.cache_dir})"

    def keys(self):
        return self.stems

    def items(self):
        for stem in self.stems:
            yield stem, self[stem]

    def values(self):
        for stem in self.stems:
            yield self[stem]

    def get(self, stem: str, default=None):
        return self[stem] if stem in self.stems else default

    def dim(self) -> Optional[int]:
        if not self.stems:
            return None
        return self[self.stems[0]].shape[-1]

    def stack(self, dtype: str = "float32") -> np.ndarray:
        """
        Materialize all embeddings as (N, D). Only call this if the full
        matrix fits in RAM. Raises if the cached arrays are sequences
        (i.e. pooling was disabled at extraction time).
        """
        if not self.stems:
            return np.zeros((0, 0), dtype=dtype)
        first = self[self.stems[0]]
        if first.ndim != 1:
            raise ValueError(
                f"stack() expects pooled 1-D embeddings, got {first.shape}. "
                "Re-extract with pooling enabled."
            )
        out = np.empty((len(self.stems), first.shape[0]), dtype=dtype)
        for i, stem in enumerate(self.stems):
            out[i] = self[stem]
        return out


# =========================================================================
# Pooling helpers
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
    denom = np.clip(m.sum(), 1e-9, None)     # two bounds: works on numpy 1.x and 2.x
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
# 1. Text embeddings
# =========================================================================
@torch.no_grad()
def extract_text_embeddings(df: pd.DataFrame,
                            model_name: str,
                            cfg) -> LazyEmbeddings:
    """
    Extract text embeddings for every row of df using `model_name`.

    Cache layout:
        <cfg.cache_dir>/text_<model>_len<max>_<pool>_<dtype>/<stem>.npy

    Returned shape per file:
        pooled  -> (D,)
        sequence-> (T, D)   (only if cfg.text_pool == "none")
    """
    safe = (
        model_name.replace("/", "_")
        + f"_len{cfg.text_max_length}"
        + f"_{getattr(cfg, 'text_pool', 'mean')}"
        + f"_{getattr(cfg, 'embed_dtype', 'float16')}"
    )
    cache = Path(cfg.cache_dir) / f"text_{safe}"
    cache.mkdir(parents=True, exist_ok=True)
    force = bool(getattr(cfg, "force_extract", False))

    try:
        from transformers import AutoTokenizer, AutoModel

        pool_mode = str(getattr(cfg, "text_pool", "mean")).lower()
        dtype     = str(getattr(cfg, "embed_dtype", "float16"))

        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(cfg.device)
        model.eval()

        seen_stems: List[str] = []

        for i, row in df.iterrows():
            stem = row["file_stem"]
            npy  = cache / f"{stem}.npy"

            if npy.exists() and not force:
                seen_stems.append(stem)
                continue

            text = row.get("transcript")
            if not isinstance(text, str) or not text.strip():
                continue

            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=cfg.text_max_length, padding=False)
            enc = {k: v.to(cfg.device) for k, v in enc.items()}

            h = model(**enc).last_hidden_state.squeeze(0)     # (T, D)
            arr = h.detach().float().cpu().numpy()

            if pool_mode == "mean":
                mask = enc["attention_mask"].squeeze(0).cpu().numpy()
                arr = mean_pool_text(arr, mask)
            elif pool_mode == "cls":
                arr = cls_pool_text(arr)
            # "none" -> keep (T, D)

            arr = _maybe_cast(arr, dtype)
            np.save(npy, arr)

            del enc, h, arr
            if str(cfg.device).startswith("cuda"):
                torch.cuda.empty_cache()

            seen_stems.append(stem)

            if (i + 1) % 100 == 0:
                print(f"[text:{safe}] {i + 1}/{len(df)}")

        if seen_stems:
            sample = np.load(cache / f"{seen_stems[0]}.npy")
            print(f"[text:{safe}] {len(seen_stems)} sequences cached in "
                  f"{cache} (shape={sample.shape}, dtype={sample.dtype})")

        del model
        if str(cfg.device).startswith("cuda"):
            torch.cuda.empty_cache()

        # NEW — drop stems with missing .npy files
        before = len(seen_stems)
        seen_stems = [s for s in seen_stems if (cache / f"{s}.npy").exists()]
        if before != len(seen_stems):
            print(f"[text:{safe}] dropped {before - len(seen_stems)} stems "
                  f"with missing .npy files")

        return LazyEmbeddings(cache, seen_stems)

    except Exception:
        print(f"[text:{safe}] FAILED for {model_name}")
        traceback.print_exc()
        return LazyEmbeddings(cache, [])


# =========================================================================
# 2. SSL embeddings
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
    Run an SSL model over (1, T) in chunks. Bounds attention memory to
    O(chunk^2). Set cfg.ssl_chunk_seconds above the longest clip to
    disable chunking entirely. Returns CPU tensor (T_out, D).
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
def extract_ssl_embeddings(df: pd.DataFrame, cfg) -> LazyEmbeddings:
    """
    Extract SSL hidden states per file, pool, save as .npy.

    Cache layout:
        <cfg.cache_dir>/ssl_<model>_sr<sr>_max<sec>_pool<n>_<dtype>/<stem>.npy

    Returned shape per file:
        pooled  -> (D,)
        sequence-> (T, D)   (only if cfg.ssl_pool is False)
    """
    import torchaudio

    pool = bool(getattr(cfg, "ssl_pool", True))
    dtype = str(getattr(cfg, "embed_dtype", "float16"))
    half  = bool(getattr(cfg, "ssl_half", False))

    safe = (
        cfg.ssl_model_name.replace("/", "_")
        + f"_sr{cfg.ssl_sample_rate}"
        + f"_max{int(cfg.max_audio_seconds)}"
        + f"_pool{int(pool)}"
        + f"_{dtype}"
    )
    cache = Path(cfg.cache_dir) / f"ssl_{safe}"
    cache.mkdir(parents=True, exist_ok=True)
    force = bool(getattr(cfg, "force_extract", False))

    try:
        model = _load_ssl_model(cfg.ssl_model_name, cfg.device, half=half)

        seen_stems: List[str] = []

        for i, row in df.iterrows():
            stem = row["file_stem"]
            npy  = cache / f"{stem}.npy"

            if npy.exists() and not force:
                seen_stems.append(stem)
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

            h = _ssl_forward_chunked(model, wav, cfg)   # (T, D) on CPU
            arr = h.numpy()
            if pool:
                arr = mean_pool(arr)                    # (D,)
            arr = _maybe_cast(arr, dtype)
            np.save(npy, arr)

            del wav, h, arr
            if str(cfg.device).startswith("cuda"):
                torch.cuda.empty_cache()

            seen_stems.append(stem)

            if (i + 1) % 100 == 0:
                print(f"[ssl:{safe}] {i + 1}/{len(df)}")

        if seen_stems:
            sample = np.load(cache / f"{seen_stems[0]}.npy")
            print(f"[ssl:{safe}] {len(seen_stems)} sequences cached in "
                  f"{cache} (shape={sample.shape}, dtype={sample.dtype})")

        del model
        if str(cfg.device).startswith("cuda"):
            torch.cuda.empty_cache()

        # NEW — drop stems with missing .npy files
        before = len(seen_stems)
        seen_stems = [s for s in seen_stems if (cache / f"{s}.npy").exists()]
        if before != len(seen_stems):
            print(f"[ssl:{safe}] dropped {before - len(seen_stems)} stems "
                  f"with missing .npy files")

        return LazyEmbeddings(cache, seen_stems)

    except Exception:
        print(f"[ssl:{safe}] FAILED for {cfg.ssl_model_name}")
        traceback.print_exc()
        return LazyEmbeddings(cache, [])


# =========================================================================
# 3. eGeMAPS
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
# Orchestrator — text first, then SSL, then eGeMAPS
# =========================================================================
def extract_all_features(df: pd.DataFrame,
                         cfg,
                         text_models: Optional[List[str]] = None
                         ) -> Dict[str, object]:
    """
    Run all three extractors in order: text, ssl, egemaps.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain file_stem, file_path, transcript columns.
    cfg : SimpleNamespace / dataclass
        Must expose cache_dir, text_max_length, text_pool, embed_dtype,
        device, ssl_model_name, ssl_sample_rate, max_audio_seconds,
        ssl_pool, ssl_chunk_seconds, ssl_half.
    text_models : list[str] | None
        Hugging Face model IDs for text. If None, uses cfg.text_models
        if present, otherwise a safe default.

    Returns
    -------
    dict with keys:
        'text'    -> dict[model_name, LazyEmbeddings]
        'ssl'     -> LazyEmbeddings
        'egemaps' -> pd.DataFrame
    """
    if text_models is None:
        text_models = getattr(cfg, "text_models", None) or [
            "emilyalsentzer/Bio_ClinicalBERT",
        ]

    results: Dict[str, object] = {
        "text": {},
        "ssl": LazyEmbeddings(cfg.cache_dir, []),
        "egemaps": pd.DataFrame(),
    }

    # ---- 1. TEXT FIRST ----
    print("\n[features] text embeddings ...")
    for name in text_models:
        emb = extract_text_embeddings(df, name, cfg)
        results["text"][name] = emb
        print(f"  {name}: {len(emb)} sequences")

    # ---- 2. SSL ----
    print("\n[features] ssl embeddings ...")
    results["ssl"] = extract_ssl_embeddings(df, cfg)
    print(f"  {cfg.ssl_model_name}: {len(results['ssl'])} sequences")

    # ---- 3. eGeMAPS ----
    print("\n[features] egemaps ...")
    results["egemaps"] = extract_egemaps(df, cfg)
    print(f"  egemaps: {len(results['egemaps'])} rows")

    # ---- summary + sanity check ----
    n_text = sum(len(v) for v in results["text"].values())
    n_ssl  = len(results["ssl"])
    n_eg   = len(results["egemaps"])

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
        print("[features] WARNING: no text embeddings; continuing with "
              "SSL/eGeMAPS only.")
    if n_ssl == 0:
        print("[features] WARNING: no SSL embeddings; continuing with "
              "text/eGeMAPS only.")
    if n_eg == 0:
        print("[features] WARNING: no eGeMAPS; continuing with "
              "text/SSL only.")

    return results