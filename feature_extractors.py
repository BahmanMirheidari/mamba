"""
Feature extraction: eGeMAPS, SSL, text embeddings.
"""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch


def extract_egemaps(df: pd.DataFrame, cfg) -> pd.DataFrame:
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
    feats = {}
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


@torch.no_grad()
def extract_ssl_embeddings(df: pd.DataFrame, cfg) -> Dict[str, np.ndarray]:
    import torchaudio
    from transformers import Wav2Vec2Model, HubertModel, WhisperModel

    safe = (cfg.ssl_model_name.replace("/", "_")
            + f"_sr{cfg.ssl_sample_rate}"
            + f"_max{int(cfg.max_audio_seconds)}")
    cache = Path(cfg.cache_dir) / f"ssl_{safe}" 
    cache.mkdir(parents=True, exist_ok=True)

    name = cfg.ssl_model_name.lower()
    if "wav2vec2" in name:
        model = Wav2Vec2Model.from_pretrained(cfg.ssl_model_name).to(cfg.device)
    elif "hubert" in name:
        model = HubertModel.from_pretrained(cfg.ssl_model_name).to(cfg.device)
    elif "whisper" in name:
        model = WhisperModel.from_pretrained(cfg.ssl_model_name).to(cfg.device)
    else:
        raise ValueError(f"Unsupported SSL model: {cfg.ssl_model_name}")
    model.eval()

    out = {}
    for i, row in df.iterrows():
        stem = row["file_stem"]
        npy = cache / f"{stem}.npy"
        if npy.exists():
            out[stem] = np.load(npy)
            continue

        wav, sr = torchaudio.load(str(row["file_path"]))
        if sr != cfg.ssl_sample_rate:
            wav = torchaudio.functional.resample(wav, sr, cfg.ssl_sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        max_len = int(cfg.max_audio_seconds * cfg.ssl_sample_rate)
        if wav.shape[1] > max_len:
            wav = wav[:, :max_len]
        wav = wav.to(cfg.device)

        try:
            out_dict = model(wav)
            h = (out_dict.last_hidden_state
                 if hasattr(out_dict, "last_hidden_state") else out_dict[0])
        except Exception as e:
            print(f"[ssl] failed on {stem}: {e}")
            continue

        arr = h.squeeze(0).cpu().numpy()
        np.save(npy, arr)
        out[stem] = arr
        if (i + 1) % 100 == 0:
            print(f"[ssl] {i + 1}/{len(df)}")

    if out:
        print(f"[ssl] {len(out)} sequences, "
              f"dim={next(iter(out.values())).shape[1]}")
    return out


@torch.no_grad()
def extract_text_embeddings(df: pd.DataFrame,
                            model_name: str,
                            cfg) -> Dict[str, np.ndarray]:
    from transformers import AutoTokenizer, AutoModel

    safe = (model_name.replace("/", "_")
            + f"_len{cfg.text_max_length}")
    cache = Path(cfg.cache_dir) / f"text_{safe}"
    cache.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(cfg.device)
    model.eval()

    out = {}
    for i, row in df.iterrows():
        stem = row["file_stem"]
        npy = cache / f"{stem}.npy"
        if npy.exists():
            out[stem] = np.load(npy)
            continue

        text = row["transcript"]
        if not isinstance(text, str) or not text.strip():
            continue

        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=512, padding=False)
        enc = {k: v.to(cfg.device) for k, v in enc.items()}
        try:
            h = model(**enc).last_hidden_state.squeeze(0)
        except Exception as e:
            print(f"[text] failed on {stem}: {e}")
            continue
        arr = h.cpu().numpy()
        np.save(npy, arr)
        out[stem] = arr
        if (i + 1) % 100 == 0:
            print(f"[text:{safe}] {i + 1}/{len(df)}")

    if out:
        print(f"[text:{safe}] {len(out)} sequences, "
              f"dim={next(iter(out.values())).shape[1]}")
    return out


def mean_pool(seq: np.ndarray) -> np.ndarray:
    return seq.mean(axis=0)