"""
Leakage-free data loading and grouped cross-validation.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import csv
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def parse_filename(stem: str, pattern: str) -> Optional[Dict[str, Optional[str]]]:
    m = re.match(pattern, stem)
    if not m:
        return None
    d = m.groupdict()
    return {k: (v if v else None) for k, v in d.items()}


def load_transcriptions_csv(path: Path,
                            file_col: str,
                            text_col: str) -> Dict[str, str]:
    df = pd.read_csv(path)
    if file_col not in df.columns:
        raise ValueError(f"'{file_col}' not in {path}. "
                         f"Available: {list(df.columns)}")
    if text_col not in df.columns:
        raise ValueError(f"'{text_col}' not in {path}. "
                         f"Available: {list(df.columns)}")

    out: Dict[str, str] = {}
    for _, r in df.iterrows():
        name = str(r[file_col]).strip()
        stem = name[:-4] if name.lower().endswith(".wav") else name
        txt = r[text_col]
        out[stem] = "" if pd.isna(txt) else str(txt).strip()
    print(f"[transcripts] loaded {len(out)} entries from {path.name}")
    return out


def _discover_wavs(cfg) -> List[Tuple[Path, Optional[str]]]:
    wav_dir = Path(cfg.wav_dir)
    if not wav_dir.exists():
        raise FileNotFoundError(f"wav-dir not found: {wav_dir}")

    if cfg.recursive_wavs:
        wavs = sorted(wav_dir.rglob("*.wav"))
    else:
        wavs = sorted(wav_dir.glob("*.wav"))

    out = []
    for w in wavs:
        if cfg.recursive_wavs and w.parent != wav_dir:
            out.append((w, w.parent.name))
        else:
            out.append((w, None))
    print(f"[data] discovered {len(out)} WAV files under {wav_dir}")
    return out


def build_master_table(cfg) -> pd.DataFrame:
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
        demo = demo.drop_duplicates(subset=[cfg.speaker_col,
                                            cfg.session_col])
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
            "file_path": wav,
            "file_stem": wav.stem,
            "speaker_id": speaker,
            "session_id": session_id,
            "question_id": question,
            "chunk_id": chunk_id,
            "label": drow[label_col],
            "transcript": transcripts.get(wav.stem, ""),
            **{c: drow[c] for c in demo.columns
               if c not in (cfg.speaker_col, cfg.session_col,
                            "__join_key", label_col)},
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No WAV files matched demo.csv. Unmatched: {unmatched[:10]}")

    if unmatched:
        print(f"[data] WARNING: {len(unmatched)} WAV files were skipped")

    # map labels to a stable integer index and remember the mapping
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

    print(f"[data] {len(df)} files | {n_spk} speakers | {n_ses} sessions "
          f"| {n_q} questions")
    print(f"[data] chunks/question: min={cq.min()}, "
          f"median={int(cq.median())}, max={cq.max()}")
    print(f"[data] task={cfg.task}, label_col='{label_col}'")
    print(f"[data] label distribution:\n{df['label'].value_counts().head(20)}")

    miss = int(df["transcript"].eq("").sum())
    if miss:
        print(f"[data] WARNING: {miss}/{len(df)} files have no transcript")

    return df


def grouped_folds(df: pd.DataFrame,
                  cfg) -> List[Tuple[np.ndarray, np.ndarray]]:
    unit_df = (df.groupby("speaker_id")
                 .agg(label=("label", lambda s: s.mode().iloc[0]))
                 .reset_index())

    if cfg.task == "regression":
        splitter = GroupKFold(n_splits=cfg.n_folds).split(
            unit_df, groups=unit_df["speaker_id"])
    else:
        splitter = StratifiedGroupKFold(
            n_splits=cfg.n_folds, shuffle=True,
            random_state=cfg.random_state).split(
            unit_df,
            y=unit_df["label"].astype(str),
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
        print(f"[cv] fold {i}: {len(tr)} train files "
              f"({df.loc[tr, 'speaker_id'].nunique()} speakers) / "
              f"{len(va)} val files "
              f"({df.loc[va, 'speaker_id'].nunique()} speakers)")
    return folds


def aggregate_to_unit(df_eval: pd.DataFrame,
                      preds: np.ndarray,
                      task: str,
                      unit: str):
    df_eval = df_eval.reset_index(drop=True)
    if unit == "question":
        keys = list(zip(df_eval["speaker_id"], df_eval["session_id"],
                        df_eval["question_id"]))
    elif unit == "session":
        keys = list(zip(df_eval["speaker_id"], df_eval["session_id"]))
    else:
        keys = list(df_eval["speaker_id"])

    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)

    agg_preds, agg_labels, agg_keys = [], [], []
    for k, idx in groups.items():
        if task == "classification":
            p = preds[idx].mean(axis=0)
            p = p / (p.sum() + 1e-12)
            agg_preds.append(p)
        else:
            agg_preds.append(float(np.mean(preds[idx])))
        agg_labels.append(df_eval.loc[idx[0], "label"])
        agg_keys.append(k)

    return np.asarray(agg_preds), np.asarray(agg_labels), agg_keys

def save_oof_predictions(out_dir, model_name: str, fold_i: int,
                         df_eval: pd.DataFrame,
                         task: str,
                         probs: np.ndarray = None,
                         logits: np.ndarray = None,
                         y_pred: np.ndarray = None) -> Path:
    """
    Persist per-file predictions for one fold.

    Classification: pass `probs` [n, C] and optionally `logits` [n, C].
                    y_pred is derived from probs if not given.
    Regression:     pass `y_pred` [n].

    Columns written:
      file_stem, speaker_id, session_id, question_id, y_true, y_pred
      (classification)  logit_0..logit_{C-1}, prob_0..prob_{C-1}
    """
    import csv
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}_fold{fold_i}.csv"

    df = df_eval.reset_index(drop=True)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["file_stem", "speaker_id", "session_id",
                  "question_id", "y_true", "y_pred"]
        if task == "classification":
            C = probs.shape[1]
            header += [f"logit_{c}" for c in range(C)]
            header += [f"prob_{c}" for c in range(C)]
        w.writerow(header)

        for i, row in df.iterrows():
            if task == "classification":
                p = probs[i]
                lp = logits[i] if logits is not None else p
                pred = int(np.argmax(p)) if y_pred is None else int(y_pred[i])
                w.writerow([row["file_stem"], row["speaker_id"],
                            row["session_id"], row["question_id"],
                            row["label"], pred]
                           + list(lp) + list(p))
            else:
                w.writerow([row["file_stem"], row["speaker_id"],
                            row["session_id"], row["question_id"],
                            row["label"], float(y_pred[i])])
    return path