#!/usr/bin/env bash
set -euo pipefail

# Launch Toys4K fitting jobs across multiple GPUs.
#
# Run from the superfit repo root:
#
#   bash job_scripts/all_toys4k.sh [start_ind] [end_ind] [num_procs] [ablation] [aot_postfix] [csv_file]
#
# Common overrides:
#
#   GPUS=0,1,2,3
#   SAVE_DIR=/path/to/outputs
#   LOG_DIR=/path/to/logs
#   CONDA_ENV=sfn
#   ENV_FILE=/path/to/.env
#   FASTMODE=1
#   OVERWRITE=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CSV_FILE="${6:-${CSV_FILE:-dataset/new_testset.csv}}"
[[ "$CSV_FILE" = /* ]] || CSV_FILE="$ROOT_DIR/$CSV_FILE"
[[ -f "$CSV_FILE" ]] || { echo "Missing CSV file: $CSV_FILE" >&2; exit 1; }

START_IND="${1:-${START_IND:-0}}"
CSV_ROWS="$(awk 'NF { n++ } END { print n + 0 }' "$CSV_FILE")"
END_IND="${2:-${END_IND:-$CSV_ROWS}}"
NUM_PROCS="${3:-${NUM_PROCS:-}}"
ABLATION="${4:-${ABLATION:-0}}"
AOT_POSTFIX="${5:-${AOT_POSTFIX:-toys4k}}"

GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/outputs/toys4k}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/toys4k}"
CONDA_ENV="${CONDA_ENV:-sfn}"
ENV_FILE="${ENV_FILE:-}"
FASTMODE="${FASTMODE:-1}"
OVERWRITE="${OVERWRITE:-0}"

IFS=',' read -r -a GPU_IDS <<< "${GPUS// /}"
if [[ -z "${NUM_PROCS:-}" ]]; then
    NUM_PROCS="${#GPU_IDS[@]}"
fi

if [[ ! "$START_IND" =~ ^[0-9]+$ || ! "$END_IND" =~ ^[0-9]+$ || ! "$NUM_PROCS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: bash job_scripts/all_toys4k.sh [start_ind] [end_ind] [num_procs] [ablation] [aot_postfix] [csv_file]" >&2
    exit 1
fi

if (( END_IND <= START_IND )); then
    echo "END_IND ($END_IND) must be greater than START_IND ($START_IND)." >&2
    exit 1
fi

TOTAL=$((END_IND - START_IND))
if (( NUM_PROCS > TOTAL )); then
    NUM_PROCS="$TOTAL"
fi

mkdir -p "$SAVE_DIR" "$LOG_DIR"

if [[ -z "$ENV_FILE" ]]; then
    if [[ -f "$ROOT_DIR/.env" ]]; then
        ENV_FILE="$ROOT_DIR/.env"
    elif [[ -f "$ROOT_DIR/../.env" ]]; then
        ENV_FILE="$ROOT_DIR/../.env"
    fi
fi

echo "Toys4K fitting"
echo "  CSV:        $CSV_FILE ($CSV_ROWS rows)"
echo "  Range:      [$START_IND, $END_IND) -> $TOTAL items"
echo "  Processes:  $NUM_PROCS"
echo "  GPUs:       ${GPUS// /}"
echo "  Save dir:   $SAVE_DIR"
echo "  Logs:       $LOG_DIR"
echo

is_true() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

run_one() {
    local proc_id="$1"
    local start="$2"
    local end="$3"
    local gpu="${GPU_IDS[$((proc_id % ${#GPU_IDS[@]}))]}"
    local log_file="$LOG_DIR/toys4k_proc_${proc_id}_${AOT_POSTFIX}.out"

    echo "proc $proc_id: GPU $gpu, [$start, $end) -> $log_file"

    (
        if [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]]; then
            set +u
            set -a
            source "$ENV_FILE"
            set +a
            set -u
        fi

        if [[ -n "${CONDA_SH:-}" && -f "$CONDA_SH" ]]; then
            set +u
            source "$CONDA_SH"
            set -u
        elif ! type conda >/dev/null 2>&1 && [[ -f "$HOME/.bashrc" ]]; then
            set +u
            source "$HOME/.bashrc"
            set -u
        fi

        if ! type conda >/dev/null 2>&1; then
            echo "conda is not available. Source conda before launching or set CONDA_SH=/path/to/conda.sh." >&2
            exit 1
        fi
        conda activate "$CONDA_ENV"

        export CUDA_VISIBLE_DEVICES="$gpu"

        args=(
            scripts/testset_fit_primitives.py
            --dataset toys4k
            --file_path "$CSV_FILE"
            --save_dir "$SAVE_DIR"
            --start_ind "$start"
            --end_ind "$end"
            --ablation "$ABLATION"
            --aot_postfix "$AOT_POSTFIX"
        )

        is_true "$FASTMODE" && args+=(--fastmode)
        is_true "$OVERWRITE" && args+=(--overwrite)

        python "${args[@]}"
    ) > "$log_file" 2>&1 &
}

BASE_CHUNK=$((TOTAL / NUM_PROCS))
REMAINDER=$((TOTAL % NUM_PROCS))
CUR="$START_IND"
PIDS=()

for ((i=0; i<NUM_PROCS; i++)); do
    CHUNK="$BASE_CHUNK"
    (( i < REMAINDER )) && CHUNK=$((CHUNK + 1))
    NEXT=$((CUR + CHUNK))
    run_one "$i" "$CUR" "$NEXT"
    PIDS+=("$!")
    CUR="$NEXT"
done

echo
echo "Spawned ${#PIDS[@]} jobs. Monitor with:"
echo "  tail -F $LOG_DIR/toys4k_proc_*_${AOT_POSTFIX}.out"
echo

STATUS=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
done

if (( STATUS == 0 )); then
    echo "All Toys4K jobs completed."
else
    echo "One or more Toys4K jobs failed. Check logs in $LOG_DIR." >&2
fi

exit "$STATUS"
