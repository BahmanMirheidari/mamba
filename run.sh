#!/usr/bin/env bash
# =============================================================================
#  run.sh — driver for the clinical speech pipeline
#
#  Two modes:
#    smoke   — tiny, fast end-to-end check (a few minutes)
#    full    — real run: experiments + ablations (+ optional grid)
#
#  Usage
#    ./run.sh smoke
#    ./run.sh full <task> <n_classes> <label_col> [flags...]
#
#  Examples
#    ./run.sh smoke
#    ./run.sh full classification 2 class2
#    ./run.sh full classification 3 class1 --output-dir results_3class
#    ./run.sh full regression 1 mmse --include-grid --output-dir results_mmse
# =============================================================================

set -euo pipefail

# ---- defaults (can be overridden by env or CLI flags) -----------------------
WAV_DIR="${WAV_DIR:-data/wav}"
DEMO_CSV="${DEMO_CSV:-data/demo.csv}"
TRANS_CSV="${TRANS_CSV:-data/transcriptions.csv}"
RESULTS="${RESULTS:-results}"
CACHE="${CACHE:-cache}"
DEVICE="${DEVICE:-cuda}"
SSL_MODEL="${SSL_MODEL:-facebook/wav2vec2-base-960h}"
N_FOLDS="${N_FOLDS:-5}"
AGG_UNIT="${AGG_UNIT:-speaker}"
SESSION_COL="${SESSION_COL:-}"

# ---- colours ----------------------------------------------------------------
BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'
YELLOW=$'\033[33m'; RESET=$'\033[0m'

log()  { printf '%s[%s]%s %s\n' "$BOLD" "$(date +%H:%M:%S)" "$RESET" "$*"; }
warn() { printf '%s[%s] %s%s\n' "$YELLOW" "$(date +%H:%M:%S)" "$*" "$RESET"; }
err()  { printf '%s[%s] %s%s\n' "$RED"    "$(date +%H:%M:%S)" "$*" "$RESET"; }
ok()   { printf '%s[%s] %s%s\n' "$GREEN"  "$(date +%H:%M:%S)" "$*" "$RESET"; }

check_inputs() {
    for f in "$WAV_DIR" "$DEMO_CSV" "$TRANS_CSV"; do
        [[ -e "$f" ]] || { err "missing input: $f"; exit 2; }
    done
}

show_header() {
    printf '\n%s%s%s\n' "$BOLD" "=============================================================" "$RESET"
    printf '%s  %s%s\n' "$BOLD" "$1" "$RESET"
    printf '%s%s%s\n\n' "$BOLD" "=============================================================" "$RESET"
}

# -----------------------------------------------------------------------------
smoke() {
    check_inputs
    show_header "SMOKE TEST"

    local smoke_results="results_smoke"
    local smoke_cache="cache"
    rm -rf "$smoke_results" "$smoke_cache"

    log "outputs           : $smoke_results"
    log "cache             : $smoke_cache"
    log "folds / epochs    : 2 / 3"
    log "models            : experiments suite (no ablations)"
    echo

    if python run.py \
        --wav-dir "$WAV_DIR" \
        --demo-csv "$DEMO_CSV" \
        --transcriptions-csv "$TRANS_CSV" \
        --task classification --n-classes 2 \
        --aggregation-unit "$AGG_UNIT" \
        ${SESSION_COL:+--session-col "$SESSION_COL"} \
        --n-folds 2 \
        --ssl-model "$SSL_MODEL" \
        --device "$DEVICE" \
        --epochs 3 \
        --batch-size 8 \
        --output-dir "$smoke_results" \
        --cache-dir "$smoke_cache" \
        --label-col class2 \
        --mode experiments
    then
        ok "smoke run finished"
    else
        err "smoke run failed"; exit 1
    fi

    local missing=0
    for e in "$smoke_results/results.json" \
             "$smoke_results/tables/comparison_table.csv"; do
        if [[ -f "$e" ]]; then ok "  ✓ $e"; else err "  ✗ $e"; missing=1; fi
    done
    if [[ -d "$smoke_results/oof" ]]; then
        local n; n=$(find "$smoke_results/oof" -name '*.csv' | wc -l)
        (( n > 0 )) && ok "  ✓ OOF: $n file(s)" \
                    || warn "  ⚠ OOF exists but empty"
    else
        warn "  ⚠ no OOF directory"
    fi
    (( missing == 0 )) || { err "smoke test finished with missing artifacts"; exit 1; }
    ok "smoke test passed"
}

# -----------------------------------------------------------------------------
full() {
    check_inputs
    show_header "FULL RUN"

    local task="${1:-}"
    local n_classes="${2:-}"
    local label_col="${3:-}"
    shift 3 || true

    if [[ -z "$task" || -z "$n_classes" || -z "$label_col" ]]; then
        err "usage: ./run.sh full <classification|regression> <n_classes> <label_col> [flags...]"
        exit 2
    fi
    [[ "$task" == "classification" || "$task" == "regression" ]] || {
        err "task must be 'classification' or 'regression'"; exit 2; }

    local mode="full"
    local include_grid=0
    local extra=()
    while (( $# > 0 )); do
        case "$1" in
            --mode)             mode="$2"; shift 2 ;;
            --include-grid)     include_grid=1; shift ;;
            --epochs)           extra+=(--epochs "$2"); shift 2 ;;
            --batch-size)       extra+=(--batch-size "$2"); shift 2 ;;
            --n-folds)          extra+=(--n-folds "$2"); N_FOLDS="$2"; shift 2 ;;
            --ssl-model)        extra+=(--ssl-model "$2"); shift 2 ;;
            --device)           extra+=(--device "$2"); shift 2 ;;
            --output-dir)       RESULTS="$2"; shift 2 ;;
            --cache-dir)        CACHE="$2"; shift 2 ;;
            --aggregation-unit) AGG_UNIT="$2"; shift 2 ;;
            --session-col)      SESSION_COL="$2"; shift 2 ;;
            *)                  extra+=("$1"); shift ;;
        esac
    done

    log "task              : $task"
    log "n_classes         : $n_classes"
    log "label column      : $label_col"
    log "aggregation unit  : $AGG_UNIT"
    log "folds             : $N_FOLDS"
    log "mode              : $mode"
    log "grid              : $([[ $include_grid == 1 ]] && echo yes || echo no)"
    log "outputs           : $RESULTS"
    log "cache             : $CACHE"
    echo

    if [[ "$mode" == "full" && $include_grid == 1 ]]; then
        warn "WARNING: --mode full --include-grid is the longest run (~3-5 days)."
        warn "Ctrl-C within 10 s to abort."
        sleep 10
    fi

    local cmd=(python run.py
        --wav-dir "$WAV_DIR"
        --demo-csv "$DEMO_CSV"
        --transcriptions-csv "$TRANS_CSV"
        --task "$task"
        --n-classes "$n_classes"
        --label-col "$label_col"
        --aggregation-unit "$AGG_UNIT"
        --n-folds "$N_FOLDS"
        --ssl-model "$SSL_MODEL"
        --device "$DEVICE"
        --output-dir "$RESULTS"
        --cache-dir "$CACHE"
        --mode "$mode"
    )
    [[ -n "$SESSION_COL" ]] && cmd+=(--session-col "$SESSION_COL")
    [[ $include_grid -eq 1 && "$mode" == "full" ]] && cmd+=(--include-grid)
    (( ${#extra[@]} > 0 )) && cmd+=("${extra[@]}")

    log "running: ${cmd[*]}"
    echo
    "${cmd[@]}" || { err "run.py failed"; exit 1; }

    show_header "AGGREGATING OOF"
    local agg_cmd=(python aggregate.py
        --results-dir "$RESULTS"
        --wav-dir "$WAV_DIR"
        --demo-csv "$DEMO_CSV"
        --transcriptions-csv "$TRANS_CSV"
        --task "$task"
        --n-classes "$n_classes"
        --label-col "$label_col"
        --aggregation-unit "$AGG_UNIT"
    )
    [[ -n "$SESSION_COL" ]] && agg_cmd+=(--session-col "$SESSION_COL")

    if ! "${agg_cmd[@]}"; then
        warn "aggregate.py failed (results.json is still written)"
    else
        ok "aggregation done"
    fi

    show_header "SUMMARY"
    for f in pooled_metrics mean_std_metrics ablation_table; do
        p="$RESULTS/tables/$f.csv"
        [[ -f "$p" ]] || continue
        echo; printf '%s%s:%s\n' "$BOLD" "$f" "$RESET"
        column -t -s, "$p" | head -25
    done

    echo
    ok "all done — artifacts under $RESULTS/"
}

# -----------------------------------------------------------------------------
usage() {
    cat <<EOF
${BOLD}run.sh${RESET} — clinical speech pipeline driver

${BOLD}Usage${RESET}
  ./run.sh smoke
  ./run.sh full <task> <n_classes> <label_col> [flags...]

${BOLD}Positional arguments (full)${RESET}
  task        classification | regression
  n_classes   2 for binary, 3+ for multiclass, 1 for regression
  label_col   column name in demo.csv (e.g. class1, mmse)

${BOLD}Flags (full)${RESET}
  --include-grid         include the 18-config novelty grid (much longer)
  --mode <m>             experiments | ablation | full   (default: full)
  --n-folds <k>          number of CV folds              (default: $N_FOLDS)
  --epochs <n>           max training epochs
  --batch-size <n>       training batch size
  --ssl-model <id>       HuggingFace SSL model
  --device <dev>         cuda | cpu
  --output-dir <d>       results directory               (default: $RESULTS)
  --cache-dir <d>        feature cache directory         (default: $CACHE)
  --aggregation-unit <u> speaker | session | question | auto
  --session-col <c>      name of session column in demo.csv

${BOLD}Environment variables${RESET}
  WAV_DIR, DEMO_CSV, TRANS_CSV, RESULTS, CACHE, DEVICE,
  SSL_MODEL, N_FOLDS, AGG_UNIT, SESSION_COL

${BOLD}Examples${RESET}
  ./run.sh smoke

  # default output (results/)
  ./run.sh full classification 2 class2

  # custom output folder for this encoder
  ./run.sh full classification 2 class2 \\
      --output-dir results_roberta --cache-dir cache_roberta \\
      --ssl-model roberta-base

  # regression, 3-class variant, and full grid
  ./run.sh full regression 1 mmse --output-dir results_mmse
  ./run.sh full classification 3 class1 --include-grid \\
      --output-dir results_3class
EOF
}

main() {
    local mode="${1:-}"
    shift || true
    case "$mode" in
        smoke)          smoke ;;
        full)           full "$@" ;;
        ""|-h|--help)   usage ;;
        *)              err "unknown mode: $mode"; usage; exit 2 ;;
    esac
}

main "$@"