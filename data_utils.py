"""
Leakage-free data loading, chunk→speaker aggregation, grouped CV.
"""
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(stem: str, pattern: str) -> Optional[Dict[str, Optional[str]]]:
    m = re.match(pattern, stem)
    if not m:
        return None
    d = m.groupdict()
    return {k: (v if v else None) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------

def load_transcriptions_csv(path: Path, file_col: str,
                            text_col: str) -> Dict[str, str]:
    df = pd.read_csv(path)
    if file_col not in df.columns:
        raise ValueError(f"'{file_col}' not in {path}. "
                         f"Available: {list(df.columns)}")
    if text_col not in df.columns:
        raise ValueError(f"'{text_col}' not in {path}. "
                         f"Available: {list(df.columns)}")
    out = {}
    for _, r in df.iterrows():
        name = str(r[file_col]).strip()
        stem = name[:-4] if name.lower().endswith(".wav") else name
        txt = r[text_col]
        out[stem] = "" if pd.isna(txt) else str(txt).strip()
    print(f"[transcripts] loaded {len(out)} entries from {path.name}")
    return out


# ---------------------------------------------------------------------------
# WAV discovery and chunk-level master table
# ---------------------------------------------------------------------------

def _discover_wavs(cfg) -> List[Tuple[Path, Optional[str]]]:
    wav_dir = Path(cfg.wav_dir)
    if not wav_dir.exists():
        raise FileNotFoundError(f"wav-dir not found: {wav_dir}")
    wavs = sorted(wav_dir.rglob("*.wav")) if cfg.recursive_wavs \
        else sorted(wav_dir.glob("*.wav"))
    out = []
    for w in wavs:
        if cfg.recursive_wavs and w.parent != wav_dir:
            out.append((w, w.parent.name))
        else:
            out.append((w, None))
    print(f"[data] discovered {len(out)} WAV files under {wav_dir}")
    return out


def build_master_table(cfg) -> pd.DataFrame:
    """
    Per-chunk table.
    Columns: file_path, file_stem, speaker_id, session_id, question_id,
             label, transcript, [+ extra columns from demo.csv]
    """
    demo = pd.read_csv(cfg.demo_csv)
    if cfg.speaker_col not in demo.columns:
        raise ValueError(f"'{cfg.speaker_col}' missing from {cfg.demo_csv}")
    label_col = cfg.resolved_label_col
    if label_col not in demo.columns:
        raise ValueError(f"'{label_col}' missing from {cfg.demo_csv}. "
                         f"Available: {list(demo.columns)}.")

    has_session_col = bool(cfg.session_col
                           and cfg.session_col in demo.columns)

    if has_session_col:
        demo = demo.assign(
            __join_key=(demo[cfg.speaker_col].astype(str) + "||"
                        + demo[cfg.session_col].astype(str)))
        demo = demo.drop_duplicates(subset=[cfg.speaker_col, cfg.session_col])
    else:
        demo = demo.assign(__join_key=demo[cfg.speaker_col].astype(str))
        demo = demo.drop_duplicates(subset=[cfg.speaker_col])

    transcripts = load_transcriptions_csv(
        cfg.transcriptions_csv, cfg.transcript_file_col,
        cfg.transcript_text_col)

    rows, unmatched = [], []
    for wav, session_folder in _discover_wavs(cfg):
        parsed = parse_filename(wav.stem, cfg.filename_pattern)
        if parsed is None or not parsed.get("speaker"):
            unmatched.append((wav.name, "regex-miss"))
            continue
        speaker = parsed["speaker"]
        if cfg.session_from_folder and session_folder is not None:
            session_id = session_folder
        elif parsed.get("session"):
            session_id = f"{speaker}_{parsed['session']}"
        else:
            session_id = "S1"
        question = parsed.get("question") or ""
        chunk = parsed.get("chunk")
        chunk_id = int(chunk) if chunk is not None else 0

        join_key = (f"{speaker}||{session_id}" if has_session_col
                    else speaker)
        hit = demo[demo["__join_key"] == join_key]
        if hit.empty:
            unmatched.append((wav.name, f"no demo row for '{join_key}'"))
            continue
        drow = hit.iloc[0]

        rows.append({
            "file_path": wav, "file_stem": wav.stem,
            "speaker_id": speaker, "session_id": session_id,
            "question_id": question, "chunk_id": chunk_id,
            "label": drow[label_col],
            "transcript": transcripts.get(wav.stem, ""),
            **{c: drow[c] for c in demo.columns
               if c not in (cfg.speaker_col, cfg.session_col,
                            "__join_key", label_col)},
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No WAV files matched demo.csv. "
                           f"Unmatched: {unmatched[:10]}")
    if unmatched:
        print(f"[data] WARNING: {len(unmatched)} WAV files skipped")

    # ---- map labels to stable integer indices ----
    labels = sorted(df["label"].unique())
    label2idx = {l: i for i, l in enumerate(labels)}
    df["label"] = df["label"].map(label2idx).astype(int)
    df.attrs["label2idx"] = label2idx
    print(f"[data] label2idx = {label2idx}")

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / "label2idx.json").write_text(
        json.dumps({str(k): int(v) for k, v in label2idx.items()}, indent=2))

    n_spk = df["speaker_id"].nunique()
    n_ses = df.groupby(["speaker_id", "session_id"]).ngroups
    n_q = df.groupby(["speaker_id", "session_id", "question_id"]).ngroups
    cq = df.groupby(["speaker_id", "session_id", "question_id"]).size()
    print(f"[data] {len(df)} chunk files | {n_spk} speakers | "
          f"{n_ses} sessions | {n_q} questions")
    print(f"[data] chunks/question: min={cq.min()}, "
          f"median={int(cq.median())}, max={cq.max()}")
    print(f"[data] task={cfg.task}, label_col='{label_col}'")
    print(f"[data] label distribution:\n{df['label'].value_counts()}")
    return df


# ---------------------------------------------------------------------------
# Chunk → speaker aggregation for model input
# ---------------------------------------------------------------------------

def build_speaker_sequences(chunk_df: pd.DataFrame,
                            audio_emb: Dict[str, np.ndarray],
                            text_emb: Dict[str, np.ndarray],
                            egemaps_df: pd.DataFrame,
                            cfg) -> pd.DataFrame:
    """
    Returns one row per speaker with non-empty audio_seqs and text_seqs.
    Skipped speakers are logged with the reason.
    """
    egemaps_index = {s: i for i, s in enumerate(egemaps_df.index)}
    egemaps_arr = egemaps_df.values.astype(np.float32)

    rows, skipped = [], []
    for speaker, g in chunk_df.groupby("speaker_id"):
        audio_list, text_list, tab_rows = [], [], []
        for _, r in g.iterrows():
            stem = r["file_stem"]
            if stem in audio_emb:
                audio_list.append(audio_emb[stem])
            if stem in text_emb:
                text_list.append(text_emb[stem])
            if stem in egemaps_index:
                tab_rows.append(egemaps_arr[egemaps_index[stem]])

        n_chunks = len(g)
        if not audio_list or not text_list:
            skipped.append({
                "speaker_id": speaker,
                "n_chunks": n_chunks,
                "audio": len(audio_list),
                "text": len(text_list),
                "egemaps": len(tab_rows),
            })
            continue

        tab = (np.mean(np.stack(tab_rows, 0), axis=0)
               if tab_rows else np.zeros(egemaps_arr.shape[1], np.float32))
        rows.append({
            "speaker_id": speaker,
            "label": int(g["label"].iloc[0]),
            "audio_seqs": audio_list,
            "text_seqs": text_list,
            "tabular": tab.astype(np.float32),
            "n_chunks": n_chunks,
        })

    if skipped:
        print(f"[speakers] skipped {len(skipped)} speakers "
              f"missing audio or text features:")
        for s in skipped[:10]:
            print(f"    {s['speaker_id']}: n_chunks={s['n_chunks']} "
                  f"audio={s['audio']} text={s['text']} "
                  f"egemaps={s['egemaps']}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(
            "No speakers have complete audio+text features. "
            "Check that: (1) transcriptions.csv keys (utt_id) match the "
            "WAV file stems; (2) the SSL and text extractors produced "
            "non-empty outputs; (3) the caches were built from the "
            "current chunk_df (use --force-extract to rebuild).")

    print(f"[speakers] {len(out)} speakers with sequences "
          f"(chunks/speaker: min={out['n_chunks'].min()}, "
          f"median={int(out['n_chunks'].median())}, "
          f"max={out['n_chunks'].max()})")
    return out


# ---------------------------------------------------------------------------
# Grouped CV at the speaker level
# ---------------------------------------------------------------------------

def grouped_folds(df: pd.DataFrame,
                  cfg) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    df must be the *speaker* table returned by build_speaker_sequences.
    """
    unit_df = df[["speaker_id", "label"]].drop_duplicates(
        subset=["speaker_id"]).reset_index(drop=True)

    if cfg.task == "regression":
        splitter = GroupKFold(n_splits=cfg.n_folds).split(
            unit_df, groups=unit_df["speaker_id"])
    else:
        splitter = StratifiedGroupKFold(
            n_splits=cfg.n_folds, shuffle=True,
            random_state=cfg.random_state).split(
            unit_df, y=unit_df["label"].astype(str),
            groups=unit_df["speaker_id"])

    folds = []
    for tr_u, va_u in splitter:
        tr_keys = set(unit_df.iloc[tr_u]["speaker_id"])
        va_keys = set(unit_df.iloc[va_u]["speaker_id"])
        tr_idx = df.index[df["speaker_id"].isin(tr_keys)].to_numpy()
        va_idx = df.index[df["speaker_id"].isin(va_keys)].to_numpy()
        assert set(df.loc[tr_idx, "speaker_id"]).isdisjoint(
            set(df.loc[va_idx, "speaker_id"])), "speaker leakage!"
        folds.append((tr_idx, va_idx))

    for i, (tr, va) in enumerate(folds):
        print(f"[cv] fold {i}: {len(tr)} train speakers / "
              f"{len(va)} val speakers")
    return folds


# ---------------------------------------------------------------------------
# OOF persistence at speaker level
# ---------------------------------------------------------------------------

def save_oof_predictions(out_dir, model_name: str, fold_i: int,
                         df_eval: pd.DataFrame,
                         task: str,
                         probs: np.ndarray = None,
                         logits: np.ndarray = None,
                         y_pred: np.ndarray = None) -> Path:
    """
    df_eval must be the *speaker* table (one row per speaker).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}_fold{fold_i}.csv"
    df = df_eval.reset_index(drop=True)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["speaker_id", "y_true", "y_pred"]
        if task == "classification":
            C = probs.shape[1]
            header += [f"logit_{c}" for c in range(C)]
            header += [f"prob_{c}" for c in range(C)]
        w.writerow(header)

        for i, row in df.iterrows():
            if task == "classification":
                p = probs[i]
                lp = logits[i] if logits is not None else p
                pred = (int(np.argmax(p)) if y_pred is None
                        else int(y_pred[i]))
                w.writerow([row["speaker_id"], int(row["label"]), pred]
                           + list(lp) + list(p))
            else:
                w.writerow([row["speaker_id"], float(row["label"]),
                            float(y_pred[i])])
    return path