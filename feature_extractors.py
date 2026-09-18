"""
Feature extraction: eGeMAPS, SSL, text embeddings.

Design
------
- eGeMAPS: functionals only -> one vector per file. Returned as a DataFrame.
- SSL and text: sequence embeddings from pretrained models.
    * During extraction, each file's embedding is pooled and saved to disk
      as a small .npy. Nothing is accumulated in RAM.
    * Returned as a LazyEmbeddings view: dict-compatible, loads one array
      from disk on demand. Peak RAM is independent of dataset size.
- SSL forward passes are chunked to bound self-attention memory. Set
  cfg.ssl_chunk_seconds above the longest file to disable chunking.

Storage math (base model, 768-dim, 30 s clips):
    sequence fp32  : ~4.6 MB/file  ->  60 GB for 13k files
    pooled  fp16   : ~1.5 KB/file  ->  20 MB for 13k files
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------
def mean_pool(seq: np.ndarray) -> np.ndarray:
    """Mean over the time axis. (T, D) -> (D,). No-op if already 1-D."""
    return seq if seq.ndim == 1 else seq.mean(axis=0)


def mean_pool_text(seq: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Mean over the token axis, ignoring padding if a mask is given.
    (T, D) + (T,) -> (D,).
    """
    if seq.ndim == 1:
        return seq
    if mask is None:
        return seq.mean(axis=0)
    m = mask.astype(np.float32)[:, None]                # (T, 1)
    denom = np.clip(m.sum(), 1e-9)
    return (seq * m).sum(axis=0) / denom


def cls_pool_text(seq: np.ndarray) -> np.ndarray:
    """First token (CLS) of a BERT/RoBERTa-style model. (T, D) -> (D,)."""
    return seq if seq.ndim == 1 else seq[0]


def _maybe_cast(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Cast to the target storage dtype. Keeps disk usage honest."""
    if dtype == "float16":
        return arr.astype(np.float16, copy=False)
    if dtype == "float32":
        return arr.astype(np.float32, copy=False)
    return arr


# ---------------------------------------------------------------------------
# Lazy dict-like view over a directory of .npy files
# ---------------------------------------------------------------------------
class LazyEmbeddings:
    """
    Dict-compatible view over a cache directory of .npy files.

    Supports the subset of dict operations callers typically need:
        emb[stem], stem in emb, len(emb), iter(emb), .keys/.items/.values/.get

    Each __getitem__ hits disk. That is intentional: peak memory stays at
    one array regardless of how many files are cached.
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
        """Feature dim of the first cached array, or None if empty."""
        if not self.stems:
            return None
        return self[self.stems[0]].shape[-1]

    def stack(self, dtype: str = "float32") -> np.ndarray:
        """
        Materialize all embeddings into a single (N, D) array.
        Only call this when the full matrix actually fits in memory.
        """
        if not self.stems:
            return np.zeros((0, 0), dtype=dtype)
        first = self[self.stems[0]]
        if first.ndim != 1:
            raise ValueError(
                f"stack() expects pooled 1-D embeddings, got shape {first.shape}. "
                "Set ssl_pool/text_pool to pool during extraction."
            )
        out = np.empty((len(self.stems), first.shape[0]),
                       dtype=dtype)
        for i, stem in enumerate(self.stems):
            out[i] = self[stem]
        return out


# ---------------------------------------------------------------------------
# eGeMAPS
# ---------------------------------------------------------------------------
def extract_egemaps(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    One row of eGeMAPSv02 functionals per file. Small enough to hold in RAM.
    Returned DataFrame is indexed by file_stem, aligned to df order.
    """
    import opensmile

    cache = Path(cfg.cache_dir) / "egemaps.csv"

    if cache.exists():
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


# ---------------------------------------------------------------------------
# SSL embeddings
# ---------------------------------------------------------------------------
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
    Run an SSL model over wav of shape (1, T) in chunks and concatenate
    the hidden states along time. Bounds attention memory to O(chunk^2).

    Set cfg.ssl_chunk_seconds larger than the longest file to do a single
    forward pass (recommended for short clips: less redundancy, less disk).
    Returns a CPU tensor of shape (T_out, D).
    """
    chunk_seconds = float(getattr(cfg, "ssl_chunk_seconds", 30.0))
    chunk_len = max(1, int(chunk_seconds * cfg.ssl_sample_rate))
    stride = max(1, chunk_len // 2)      # 50% overlap for continuity
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
    Extract SSL hidden states per file, pool, save as .npy, and return a
    lazy view.

    Cache path: <cfg.cache_dir>/ssl_<model>_sr<sr>_max<sec>_<pool>_<dtype>/
    Per-file shape:
        pooled  -> (D,)
        sequence-> (T, D)   (only if cfg.ssl_pool is False)
    """
    import torchaudio

    pool = bool(getattr(cfg, "ssl_pool", True))
    dtype = str(getattr(cfg, "embed_dtype", "float16"))
    half = bool(getattr(cfg, "ssl_half", False))

    safe = (
        cfg.ssl_model_name.replace("/", "_")
        + f"_sr{cfg.ssl_sample_rate}"
        + f"_max{int(cfg.max_audio_seconds)}"
        + f"_pool{int(pool)}"
        + f"_{dtype}"
    )
    cache = Path(cfg.cache_dir) / f"ssl_{safe}"
    cache.mkdir(parents=True, exist_ok=True)

    model = _load_ssl_model(cfg.ssl_model_name, cfg.device, half=half)

    seen_stems: List[str] = []
    for i, row in df.iterrows():
        stem = row["file_stem"]
        npy = cache / f"{stem}.npy"

        if npy.exists():
            seen_stems.append(stem)          # lazy view handles the read
            continue

        try:
            wav, sr = torchaudio.load(str(row["file_path"]))
        except Exception as e:
            print(f"[ssl] failed to load {stem}: {e}")
            continue

        if sr != cfg.ssl_sample_rate:
            wav = torchaudio.functional.resample(wav, sr, cfg.ssl_sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        max_len = int(cfg.max_audio_seconds * cfg.ssl_sample_rate)
        if wav.shape[1] > max_len:
            wav = wav[:, :max_len]

        try:
            h = _ssl_forward_chunked(model, wav, cfg)    # (T, D) on CPU
        except Exception as e:
            print(f"[ssl] failed on {stem}: {e}")
            del wav
            continue

        arr = h.numpy()
        if pool:
            arr = mean_pool(arr)                          # (D,)
        arr = _maybe_cast(arr, dtype)
        np.save(npy, arr)

        del wav, h, arr
        if str(cfg.device).startswith("cuda"):
            torch.cuda.empty_cache()

        seen_stems.append(stem)
        if (i + 1) % 100 == 0:
            print(f"[ssl] {i + 1}/{len(df)}")

    if seen_stems:
        sample = np.load(cache / f"{seen_stems[0]}.npy")
        print(f"[ssl] {len(seen_stems)} sequences cached in {cache} "
              f"(shape={sample.shape}, dtype={sample.dtype})")

    del model
    if str(cfg.device).startswith("cuda"):
        torch.cuda.empty_cache()

    return LazyEmbeddings(cache, seen_stems)


# ---------------------------------------------------------------------------
# Text embeddings
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_text_embeddings(df: pd.DataFrame,
                            model_name: str,
                            cfg) -> LazyEmbeddings:
    """
    Token-level hidden states per transcript, pooled, saved as .npy.
    Returns a lazy view.

    cfg.text_pool: "mean" (masked) | "cls" | "none"
    Cache: <cfg.cache_dir>/text_<model>_len<max>_<pool>_<dtype>/
    Per-file shape:
        pooled  -> (D,)
        sequence-> (T, D)   (only if cfg.text_pool == "none")
    """
    from transformers import AutoTokenizer, AutoModel

    pool_mode = str(getattr(cfg, "text_pool", "mean")).lower()
    dtype = str(getattr(cfg, "embed_dtype", "float16"))

    safe = (
        model_name.replace("/", "_")
        + f"_len{cfg.text_max_length}"
        + f"_{pool_mode}"
        + f"_{dtype}"
    )
    cache = Path(cfg.cache_dir) / f"text_{safe}"
    cache.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(cfg.device)
    model.eval()

    seen_stems: List[str] = []
    for i, row in df.iterrows():
        stem = row["file_stem"]
        npy = cache / f"{stem}.npy"

        if npy.exists():
            seen_stems.append(stem)
            continue

        text = row.get("transcript")
        if not isinstance(text, str) or not text.strip():
            continue

        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=cfg.text_max_length, padding=False)
        enc = {k: v.to(cfg.device) for k, v in enc.items()}

        try:
            h = model(**enc).last_hidden_state.squeeze(0)     # (T, D) on device
        except Exception as e:
            print(f"[text] failed on {stem}: {e}")
            del enc
            continue

        arr = h.detach().float().cpu().numpy()

        # Pool before freeing enc — we need attention_mask
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
        print(f"[text:{safe}] {len(seen_stems)} sequences cached in {cache} "
              f"(shape={sample.shape}, dtype={sample.dtype})")

    del model
    if str(cfg.device).startswith("cuda"):
        torch.cuda.empty_cache()

    return LazyEmbeddings(cache, seen_stems)