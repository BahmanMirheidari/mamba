#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Paths and shared config
# =============================================================================
WAV_DIR="${WAV_DIR:-./data/wav}"
DEMO_CSV="${DEMO_CSV:-./data/demo.csv}"
TRANS_CSV="${TRANS_CSV:-./data/transcriptions.csv}"

SSL_MODEL="${SSL_MODEL:-facebook/wav2vec2-base-960h}"
DEVICE="${DEVICE:-cpu}"

# Speaker-level aggregation
AGG_UNIT="${AGG_UNIT:-speaker}"
SPEAKER_COL="${SPEAKER_COL:-speaker_id}"
SESSION_COL="${SESSION_COL:-}"            # keep empty for speaker-level
N_FOLDS="${N_FOLDS:-2}"

# =============================================================================
# Task: classification | regression
# =============================================================================
TASK="${TASK:-classification}"

N_CLASSES="${N_CLASSES:-2}"
LABEL_COL_CLS_FULL="${LABEL_COL_CLS_FULL:-class1}"
LABEL_COL_CLS_SMOKE="${LABEL_COL_CLS_SMOKE:-class1}"
LABEL_COL_REG_FULL="${LABEL_COL_REG_FULL:-score}"
LABEL_COL_REG_SMOKE="${LABEL_COL_REG_SMOKE:-score}"

# =============================================================================
# Output dirs
# =============================================================================
FULL_RESULTS="${FULL_RESULTS:-./results}"
FULL_CACHE="${FULL_CACHE:-./cache}"
SMOKE_RESULTS="${SMOKE_RESULTS:-./smoke_results}"
SMOKE_CACHE="${SMOKE_CACHE:-./smoke_cache}"

# =============================================================================
# Feature extraction knobs
# =============================================================================
SSL_POOL="${SSL_POOL:-true}"
SSL_HALF="${SSL_HALF:-false}"
SSL_CHUNK_SECONDS="${SSL_CHUNK_SECONDS:-60}"
TEXT_POOL="${TEXT_POOL:-mean}"
EMBED_DTYPE="${EMBED_DTYPE:-float16}"

# =============================================================================
# Post-hoc aggregation (calls aggregate.py after run.py)
# =============================================================================
AGGREGATE="${AGGREGATE:-true}"
N_BOOTSTRAP="${N_BOOTSTRAP:-2000}"
BOOTSTRAP_ALPHA="${BOOTSTRAP_ALPHA:-0.05}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"

# =============================================================================
# Mode: smoke | full
# =============================================================================
MODE="${1:-smoke}"

if [[ "$MODE" == "smoke" ]]; then
  RESULTS_DIR="$SMOKE_RESULTS"
  CACHE_DIR="$SMOKE_CACHE"
  EPOCHS="${SMOKE_EPOCHS:-1}"
  BATCH_SIZE="${SMOKE_BATCH:-1}"
  MAX_FILES="${SMOKE_MAX_FILES:-20}"
  [[ "$AGGREGATE" == "true" ]] && N_BOOTSTRAP="${SMOKE_N_BOOTSTRAP:-200}"
else
  RESULTS_DIR="$FULL_RESULTS"
  CACHE_DIR="$FULL_CACHE"
  EPOCHS="${FULL_EPOCHS:-10}"
  BATCH_SIZE="${FULL_BATCH:-4}"
  MAX_FILES="${FULL_MAX_FILES:-0}"
fi

# =============================================================================
# Pick label column based on task + mode
# =============================================================================
if [[ "$TASK" == "regression" ]]; then
  if [[ "$MODE" == "smoke" ]]; then
    LABEL_COL="$LABEL_COL_REG_SMOKE"
  else
    LABEL_COL="$LABEL_COL_REG_FULL"
  fi
  TASK_ARGS=(--task regression)
else
  if [[ "$MODE" == "smoke" ]]; then
    LABEL_COL="$LABEL_COL_CLS_SMOKE"
  else
    LABEL_COL="$LABEL_COL_CLS_FULL"
  fi
  TASK_ARGS=(--task classification --n-classes "$N_CLASSES")
fi

# =============================================================================
# Optional smoke subset
# =============================================================================
RUN_DEMO_CSV="$DEMO_CSV"
RUN_TRANS_CSV="$TRANS_CSV"
if [[ "$MAX_FILES" -gt 0 ]]; then
  RUN_DEMO_CSV="${CACHE_DIR}/demo_smoke.csv"
  RUN_TRANS_CSV="${CACHE_DIR}/trans_smoke.csv"
  mkdir -p "$CACHE_DIR"
  head -n $((MAX_FILES + 1)) "$DEMO_CSV"  > "$RUN_DEMO_CSV"
  head -n $((MAX_FILES + 1)) "$TRANS_CSV" > "$RUN_TRANS_CSV"
  echo "[run] smoke subset: $MAX_FILES files -> $RUN_DEMO_CSV"
fi

# =============================================================================
# Banner
# =============================================================================
echo "=============================================="
echo " mode          : $MODE"
echo " task          : $TASK"
echo " label col     : $LABEL_COL"
echo " epochs        : $EPOCHS"
echo " batch size    : $BATCH_SIZE"
echo " device        : $DEVICE"
echo " ssl model     : $SSL_MODEL"
echo " ssl pool      : $SSL_POOL"
echo " ssl chunk s   : $SSL_CHUNK_SECONDS"
echo " text pool     : $TEXT_POOL"
echo " embed dtype   : $EMBED_DTYPE"
echo " aggregation   : $AGG_UNIT"
echo " speaker col   : $SPEAKER_COL"
echo " session col   : ${SESSION_COL:-<none>}"
echo " aggregate     : $AGGREGATE"
echo " n bootstrap   : $N_BOOTSTRAP"
echo " cache dir     : $CACHE_DIR"
echo " results dir   : $RESULTS_DIR"
echo "=============================================="

mkdir -p "$RESULTS_DIR" "$CACHE_DIR"

# =============================================================================
# Step 1 — run.py
# =============================================================================
python run.py \
  --wav-dir "$WAV_DIR" \
  --demo-csv "$RUN_DEMO_CSV" \
  --transcriptions-csv "$RUN_TRANS_CSV" \
  "${TASK_ARGS[@]}" \
  --aggregation-unit "$AGG_UNIT" \
  --speaker-col "$SPEAKER_COL" \
  ${SESSION_COL:+--session-col "$SESSION_COL"} \
  --n-folds "$N_FOLDS" \
  --ssl-model "$SSL_MODEL" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --output-dir "$RESULTS_DIR" \
  --cache-dir "$CACHE_DIR" \
  --label-col "$LABEL_COL" \
  --ssl-pool "$SSL_POOL" \
  --ssl-half "$SSL_HALF" \
  --ssl-chunk-seconds "$SSL_CHUNK_SECONDS" \
  --text-pool "$TEXT_POOL" \
  --embed-dtype "$EMBED_DTYPE" \
  --mode experiments

# =============================================================================
# Step 2 — aggregate.py (metrics + bootstrap CIs from OOF)
# =============================================================================
if [[ "$AGGREGATE" == "true" ]]; then
  OOF_DIR="${RESULTS_DIR}/oof"
  if [[ ! -d "$OOF_DIR" ]]; then
    echo "[agg] no OOF dir at $OOF_DIR — skipping aggregate.py"
  else
    echo "[agg] running aggregate.py on $RESULTS_DIR (unit=$AGG_UNIT)"
    python aggregate.py \
      --results-dir "$RESULTS_DIR" \
      --task "$TASK" \
      --n-classes "$N_CLASSES" \
      --aggregation-unit "$AGG_UNIT" \
      --n-bootstrap "$N_BOOTSTRAP" \
      --alpha "$BOOTSTRAP_ALPHA" \
      --seed "$BOOTSTRAP_SEED"
    echo "[agg] done — see ${RESULTS_DIR}/tables/"
  fi
fi