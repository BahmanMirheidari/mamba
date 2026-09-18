"""
Central configuration.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    # ---- paths ----
    wav_dir: Path = Path("data/wav")
    demo_csv: Path = Path("data/demo.csv")
    transcriptions_csv: Path = Path("data/transcriptions.csv")
    cache_dir: Path = Path("cache")
    output_dir: Path = Path("results")

    # ---- task ----
    task: str = "classification"
    n_classes: int = 2
    label_col: Optional[str] = None

    # ---- demo columns ----
    speaker_col: str = "speaker_id"
    session_col: Optional[str] = None

    # ---- transcripts columns ----
    transcript_file_col: str = "utt_id"
    transcript_text_col: str = "transcript"

    # ---- filename regex (R_00013_230802_123716_Q10_0) ----
    filename_pattern: str = (
        r'^(?P<speaker>.+?)'
        r'_(?P<session>\d{6}_\d{6})'
        r'_(?P<question>Q\d+)'
        r'(?:_(?P<chunk>\d+))?'
        r'(?:_.*)?$'
    )

    # ---- flat vs recursive WAV discovery ----
    recursive_wavs: bool = False
    session_from_folder: bool = False

    # ---- aggregation unit for metric computation ----
    aggregation_unit: str = "auto"   # auto | question | session | speaker

    # ---- CV ----
    n_folds: int = 5
    random_state: int = 42

    # ---- encoders ----
    ssl_model_name: str = "facebook/wav2vec2-base-960h"
    ssl_sample_rate: int = 16000
    max_audio_seconds: float = 30.0
    text_max_length: int = 512
    text_model_names: List[str] = field(default_factory=lambda: [
        "bert-base-uncased",
        "emilyalsentzer/Bio_ClinicalBERT",
    ])

    # ---- model dims ----
    mamba_d_model: int = 256
    mamba_n_layers: int = 4
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dropout: float = 0.1

    fusion_d_model: int = 256
    fusion_n_heads: int = 4
    fusion_dropout: float = 0.1

    # ---- training ----
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 10
    grad_clip: float = 1.0
    warmup_frac: float = 0.1
    use_amp: bool = True
    device: str = "cuda"

    # ---- reporting ----
    primary_metric_classification: str = "macro_f1"
    primary_metric_regression: str = "rmse"

    # runtime state (do not set from CLI)
    _current_model_name: str = ""
    _current_fold: int = -1

    ssl_pool: bool = True
    ssl_half: bool = False
    ssl_chunk_seconds: float = 30.0
    text_pool: str = "mean"
    embed_dtype: str = "float16"

    # ---- derived ----
    @property
    def resolved_label_col(self) -> str:
        if self.label_col:
            return self.label_col
        return "label" if self.task == "classification" else "score"

    @property
    def n_outputs(self) -> int:
        return self.n_classes if self.task == "classification" else 1

    @property
    def resolved_aggregation_unit(self) -> str:
        if self.aggregation_unit != "auto":
            return self.aggregation_unit
        return "session" if self.session_col else "speaker"

    @property
    def primary_metric(self) -> str:
        return (self.primary_metric_classification
                if self.task == "classification"
                else self.primary_metric_regression)

    @property
    def lower_is_better(self) -> bool:
        return self.task == "regression"