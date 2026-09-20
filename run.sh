#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

WAV_DIR="${WAV_DIR:-./data/wav}"
DEMO_CSV="${DEMO_CSV:-./data/demo.csv}"
TRANS_CSV="${TRANS_CSV:-./data/transcriptions.csv}"

SSL_MODEL="${SSL_MODEL:-facebook/wav2vec2-base-960h}"
DEVICE="${DEVICE:-$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')}"

AGG_UNIT="${AGG_UNIT:-speaker}"
SPEAKER_COL="${SPEAKER_COL:-speaker_id}"
SESSION_COL="${SESSION_COL:-}"

N_FOLDS="${N_FOLDS:-2}"

TASK="${TASK:-classification}"
N_CLASSES="${N_CLASSES:-2}"

SCORE_LABEL="${SCORE_LABEL:-score}"

RESULTS_DIR="${RESULTS_DIR:-./results}"
CACHE_DIR="${CACHE_DIR:-./cache}"

SSL_POOL="${SSL_POOL:-true}"
SSL_HALF="${SSL_HALF:-false}"
SSL_CHUNK_SECONDS="${SSL_CHUNK_SECONDS:-30}"

TEXT_POOL="${TEXT_POOL:-mean}"
EMBED_DTYPE="${EMBED_DTYPE:-float16}"

AGGREGATE="${AGGREGATE:-true}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
BOOTSTRAP_ALPHA="${BOOTSTRAP_ALPHA:-0.05}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"
FORCE_EXTRACT="${FORCE_EXTRACT:-false}"


# =============================================================================
# Mode
# =============================================================================

MODE="${1:-smoke}"

case "$MODE" in

    smoke)
        EPOCHS="${EPOCHS:-1}"
        BATCH_SIZE="${BATCH_SIZE:-1}"
        MAX_FILES="${SMOKE_MAX_FILES:-10}"

        RESULTS_DIR="${SMOKE_RESULTS_DIR:-./smoke_results}"
        CACHE_DIR="${SMOKE_CACHE_DIR:-./smoke_cache}"

        N_BOOTSTRAP="${SMOKE_N_BOOTSTRAP:-100}"
        FORCE_EXTRACT="${FORCE_EXTRACT:-true}"
        ;;

    full)
        EPOCHS="${EPOCHS:-10}"
        BATCH_SIZE="${BATCH_SIZE:-4}"
        MAX_FILES="${FULL_MAX_FILES:-0}"
        ;;

    *)
        echo "ERROR: Unknown mode '$MODE'"
        echo
        echo "Usage:"
        echo "  ./run.sh smoke"
        echo "  ./run.sh full"
        exit 1
        ;;

esac


# =============================================================================
# Select task / label arguments
# =============================================================================

if [[ "$TASK" == "classification" ]]; then

    LABEL_COL="${LABEL_COL:-class2}"

    TASK_ARGS=(
        --task classification
        --n-classes "$N_CLASSES" 
    )

elif [[ "$TASK" == "regression" ]]; then

    LABEL_COL="${LABEL_COL:-$SCORE_LABEL}"

    TASK_ARGS=(
        --task regression
    )

else

    echo "ERROR: TASK must be 'classification' or 'regression'."
    exit 1

fi


# =============================================================================
# Check input files
# =============================================================================

if [[ ! -f "$DEMO_CSV" ]]; then
    echo "ERROR: demo CSV not found:"
    echo "  $DEMO_CSV"
    exit 1
fi

if [[ ! -f "$TRANS_CSV" ]]; then
    echo "ERROR: transcription CSV not found:"
    echo "  $TRANS_CSV"
    exit 1
fi

if [[ ! -d "$WAV_DIR" ]]; then
    echo "ERROR: WAV directory not found:"
    echo "  $WAV_DIR"
    exit 1
fi


# =============================================================================
# Environment information
# =============================================================================

echo
echo "============================================================"
echo "Environment"
echo "============================================================"

echo "Python:"
python --version

echo
echo "PyTorch:"

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY


# =============================================================================
# Create output/cache directories
# =============================================================================

mkdir -p "$RESULTS_DIR"
mkdir -p "$CACHE_DIR"


# =============================================================================
# Smoke subset
#
# Select 20 random unique speakers from demo.csv.
# Then retain ALL transcription rows belonging to those speakers.
#
# demo.csv:
#   speaker_id is the LAST column
#
# transcriptions.csv:
#   utt_id,transcript
#
# Example utt_id:
#   R_72984_241007_142302_Q6_0
#
# Speaker ID extracted:
#   R_72984
# =============================================================================

RUN_DEMO_CSV="$DEMO_CSV"
RUN_TRANS_CSV="$TRANS_CSV"

if [[ "$MODE" == "smoke" && "$MAX_FILES" -gt 0 ]]; then

    RUN_DEMO_CSV="${CACHE_DIR}/demo_smoke.csv"
    RUN_TRANS_CSV="${CACHE_DIR}/trans_smoke.csv"

    RANDOM_SPEAKERS="${CACHE_DIR}/smoke_random_speakers.txt"

    echo
    echo "============================================================"
    echo "Creating smoke subset"
    echo "============================================================"

    # -------------------------------------------------------------------------
    # Extract unique speaker IDs from the LAST column of demo.csv.
    #
    # The header is skipped.
    # CR characters are removed to handle Windows/CRLF CSV files.
    # -------------------------------------------------------------------------

    ALL_SPEAKERS="${CACHE_DIR}/all_speakers.txt"

    tail -n +2 "$DEMO_CSV" |
        awk -F',' '
            {
                gsub(/\r/, "", $NF)
                if ($NF != "") {
                    print $NF
                }
            }
        ' |
        sort -u |
        sort -V > "$ALL_SPEAKERS"


    TOTAL_SPEAKERS=$(wc -l < "$ALL_SPEAKERS" | tr -d ' ')

    echo "[smoke] Total speakers available: $TOTAL_SPEAKERS"


    # -------------------------------------------------------------------------
    # Check number of speakers
    # -------------------------------------------------------------------------

    if [[ "$TOTAL_SPEAKERS" -lt "$MAX_FILES" ]]; then

        echo
        echo "[smoke] Requested $MAX_FILES speakers."
        echo "[smoke] Only $TOTAL_SPEAKERS speakers available."
        echo "[smoke] Using all available speakers."

        MAX_FILES="$TOTAL_SPEAKERS"

    fi


    if [[ "$MAX_FILES" -le 0 ]]; then
        echo "ERROR: No speakers found in demo.csv."
        exit 1
    fi


    # -------------------------------------------------------------------------
    # Randomly select speakers
    # -------------------------------------------------------------------------

    shuf \
        -n "$MAX_FILES" \
        "$ALL_SPEAKERS" \
        > "$RANDOM_SPEAKERS"


    SELECTED_SPEAKERS=$(wc -l < "$RANDOM_SPEAKERS" | tr -d ' ')


    echo
    echo "[smoke] Selected speakers: $SELECTED_SPEAKERS"
    echo
    echo "[smoke] Speaker IDs:"
    cat "$RANDOM_SPEAKERS"


    # -------------------------------------------------------------------------
    # Create demo_smoke.csv
    #
    # Keep header + all demo rows belonging to selected speakers.
    # speaker_id is the LAST column.
    # -------------------------------------------------------------------------

    head -n 1 "$DEMO_CSV" > "$RUN_DEMO_CSV"

    awk -F',' \
        -v speaker_file="$RANDOM_SPEAKERS" '

        BEGIN {
            while ((getline speaker < speaker_file) > 0) {
                gsub(/\r/, "", speaker)
                wanted[speaker] = 1
            }
            close(speaker_file)
        }

        NR > 1 {
            speaker = $NF
            gsub(/\r/, "", speaker)

            if (speaker in wanted) {
                print
            }
        }

    ' "$DEMO_CSV" >> "$RUN_DEMO_CSV"


    # -------------------------------------------------------------------------
    # Create trans_smoke.csv
    #
    # transcriptions.csv:
    #
    #   utt_id,transcript
    #
    # Example:
    #
    #   R_72984_241007_142302_Q6_0
    #
    # Split at "_":
    #
    #   R
    #   72984
    #   241007
    #   142302
    #   Q6
    #   0
    #
    # Speaker ID = first two fields:
    #
    #   R_72984
    #
    # Keep ALL rows belonging to selected speakers.
    # -------------------------------------------------------------------------

    head -n 1 "$TRANS_CSV" > "$RUN_TRANS_CSV"

    awk -F',' \
        -v speaker_file="$RANDOM_SPEAKERS" '

        BEGIN {
            while ((getline speaker < speaker_file) > 0) {
                gsub(/\r/, "", speaker)
                wanted[speaker] = 1
            }
            close(speaker_file)
        }

        NR > 1 {

            utt_id = $1
            gsub(/\r/, "", utt_id)

            split(utt_id, parts, "_")

            speaker = parts[1] "_" parts[2]

            if (speaker in wanted) {
                print
            }
        }

    ' "$TRANS_CSV" >> "$RUN_TRANS_CSV"


    # -------------------------------------------------------------------------
    # Count rows
    # -------------------------------------------------------------------------

    DEMO_SMOKE_COUNT=$(( $(wc -l < "$RUN_DEMO_CSV") - 1 ))
    TRANS_SMOKE_COUNT=$(( $(wc -l < "$RUN_TRANS_CSV") - 1 ))


    echo
    echo "============================================================"
    echo "Smoke subset summary"
    echo "============================================================"

    echo "Speakers selected        : $SELECTED_SPEAKERS"
    echo "Demo rows                : $DEMO_SMOKE_COUNT"
    echo "Transcription rows       : $TRANS_SMOKE_COUNT"

    echo
    echo "Demo CSV:"
    echo "  $RUN_DEMO_CSV"

    echo
    echo "Transcription CSV:"
    echo "  $RUN_TRANS_CSV"

    echo
    echo "Selected speakers:"
    echo "  $RANDOM_SPEAKERS"


    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------

    if [[ "$DEMO_SMOKE_COUNT" -eq 0 ]]; then
        echo
        echo "ERROR: demo_smoke.csv contains no data rows."
        exit 1
    fi

    if [[ "$TRANS_SMOKE_COUNT" -eq 0 ]]; then
        echo
        echo "ERROR: trans_smoke.csv contains no data rows."
        exit 1
    fi


    echo
    echo "[smoke] Smoke subset created successfully."

else

    echo
    echo "[run] Using complete input CSV files."

fi

# =============================================================================
# Build run.py arguments
# =============================================================================

RUN_ARGS=(
    --wav-dir "$WAV_DIR"
    --demo-csv "$RUN_DEMO_CSV"
    --transcriptions-csv "$RUN_TRANS_CSV"

    "${TASK_ARGS[@]}"

    --aggregation-unit "$AGG_UNIT"
    --speaker-col "$SPEAKER_COL"

    --n-folds "$N_FOLDS"

    --ssl-model "$SSL_MODEL"
    --device "$DEVICE"

    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"

    --output-dir "$RESULTS_DIR"
    --cache-dir "$CACHE_DIR"

    --label-col "$LABEL_COL"

    --ssl-pool "$SSL_POOL"
    --ssl-chunk-seconds "$SSL_CHUNK_SECONDS"

    --text-pool "$TEXT_POOL"
    --embed-dtype "$EMBED_DTYPE"

    --mode experiments
)


# =============================================================================
# Optional session column
# =============================================================================

if [[ -n "$SESSION_COL" ]]; then
    RUN_ARGS+=(
        --session-col "$SESSION_COL"
    )
fi


# =============================================================================
# SSL half precision
#
# IMPORTANT:
# run.py defines --ssl-half as a boolean flag.
#
# Therefore:
#
#     SSL_HALF=true  -> --ssl-half
#     SSL_HALF=false -> nothing
#
# DO NOT pass:
#
#     --ssl-half false
# =============================================================================

if [[ "$SSL_HALF" == "true" ]]; then
    RUN_ARGS+=(--ssl-half)
fi


if [[ "$FORCE_EXTRACT" == "true" ]]; then
    RUN_ARGS+=(--force-extract)
fi


# =============================================================================
# Print configuration
# =============================================================================

echo
echo "============================================================"
echo "Experiment configuration"
echo "============================================================"

echo "Mode              : $MODE"
echo "Task              : $TASK"
echo "N classes         : $N_CLASSES"
echo "Label column      : $LABEL_COL"

echo "WAV directory     : $WAV_DIR"
echo "Demo CSV          : $RUN_DEMO_CSV"
echo "Transcriptions    : $RUN_TRANS_CSV"

echo "SSL model         : $SSL_MODEL"
echo "Device            : $DEVICE"
echo "SSL pooling       : $SSL_POOL"
echo "SSL half          : $SSL_HALF"
echo "Force extract     : $FORCE_EXTRACT"
echo "SSL chunk seconds : $SSL_CHUNK_SECONDS"

echo "Text pooling      : $TEXT_POOL"
echo "Embedding dtype   : $EMBED_DTYPE"

echo "Aggregation unit  : $AGG_UNIT"
echo "Speaker column    : $SPEAKER_COL"
echo "Session column    : ${SESSION_COL:-<none>}"

echo "N folds            : $N_FOLDS"
echo "Epochs             : $EPOCHS"
echo "Batch size         : $BATCH_SIZE"

echo "Results directory : $RESULTS_DIR"
echo "Cache directory   : $CACHE_DIR"

echo "Bootstrap samples : $N_BOOTSTRAP"


# =============================================================================
# Run experiment
# =============================================================================

echo
echo "============================================================"
echo "COMMAND: python run.py"
echo "============================================================"


printf '  %q' python run.py "${RUN_ARGS[@]}"
echo
echo

python run.py "${RUN_ARGS[@]}"


echo
echo "============================================================"
echo "run.py finished"
echo "============================================================"


# =============================================================================
# Run aggregation
# =============================================================================

if [[ "$AGGREGATE" == "true" ]]; then

    OOF_DIR="${RESULTS_DIR}/oof"

    if [[ ! -d "$OOF_DIR" ]]; then

        echo
        echo "[agg] No OOF directory found:"
        echo "      $OOF_DIR"
        echo "[agg] Skipping aggregate.py"

    else

        AGG_ARGS=(
            --results-dir "$RESULTS_DIR"
            --task "$TASK"
            --n-classes "$N_CLASSES"
            --aggregation-unit "$AGG_UNIT"
            --n-bootstrap "$N_BOOTSTRAP"
            --alpha "$BOOTSTRAP_ALPHA"
            --seed "$BOOTSTRAP_SEED"
        )


        echo
        echo "============================================================"
        echo "COMMAND: python aggregate.py"
        echo "============================================================"

        printf '  %q' python aggregate.py "${AGG_ARGS[@]}"
        echo
        echo

        python aggregate.py "${AGG_ARGS[@]}"


        echo
        echo "[agg] Aggregation finished."
        echo "[agg] Results:"
        echo "      $RESULTS_DIR/tables/"

    fi

else

    echo
    echo "[agg] Aggregation disabled."

fi


# =============================================================================
# Finished
# =============================================================================

echo
echo "============================================================"
echo "ALL DONE"
echo "============================================================"

echo "Mode    : $MODE"
echo "Results : $RESULTS_DIR"
echo "Cache   : $CACHE_DIR"
echo