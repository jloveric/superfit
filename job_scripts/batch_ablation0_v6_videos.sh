#!/bin/bash
#
# Generate opt + explode/color/compact + spiral + combined videos for every shape
# under ablation_0_v6 (one primitive_assembly.pkl per folder).
#
# Usage:
#   ./batch_ablation0_v6_videos.sh [--overwrite] [--num_procs N] [--gpu_ids IDS] [dataset_root] [folder_name ...]
#
#   --overwrite:  regenerate outputs even when MP4 files already exist
#   --num_procs:  number of local worker processes to launch (default: 1)
#   --gpu_ids:    comma-separated GPU ids to cycle across workers (default: auto-detect)
#   dataset_root: default /users/aganesh8/data/aganesh8/data/project_sf/cvpr/toys4k/ablation_0_v6
#   folder_name:  optional subset of shape folders; default = all subdirs with a pkl
#
# Camera / timing match exploring_pa.ipynb shared config (cell c080a609) and
# opt_seq_v5 (skip_iters): origin (0, -1, 0), base angles pi/8 and 3pi/4, distance 4.0,
# opt pan start pi/80 -> end pi/8 on X; 10pi/8 -> 3pi/4 on Y; 1.5 s per resfit iter.
# Renders at 1024x1024 (notebook cells often use 512 for speed).

set -euo pipefail

SCRIPT_DIR="/users/aganesh8/data/aganesh8/projects/project_sf/superfit"
ENV_FILE="/users/aganesh8/data/aganesh8/projects/project_sf/.env"
LOG_DIR="/users/aganesh8/data/aganesh8/projects/project_sf/logs/batch_ablation0_v6_videos"
PYTHON="/users/aganesh8/.conda/envs/sfn/bin/python"
GEN_VIDEO="${SCRIPT_DIR}/scripts/visualize/generate_opt_video.py"
DEFAULT_DATASET_ROOT="/users/aganesh8/data/aganesh8/data/project_sf/cvpr/toys4k/ablation_0_v6"
OVERWRITE=false
NUM_PROCS=1
GPU_IDS_ARG=""

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

POSITIONAL=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --overwrite)
      OVERWRITE=true
      ;;
    --num_procs|--n_processes|-j)
      if [ "$#" -lt 2 ]; then
        echo "Error: $1 requires an integer argument" >&2
        usage >&2
        exit 2
      fi
      NUM_PROCS="$2"
      shift
      ;;
    --gpu_ids|--gpus)
      if [ "$#" -lt 2 ]; then
        echo "Error: $1 requires a comma-separated GPU id list" >&2
        usage >&2
        exit 2
      fi
      GPU_IDS_ARG="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      POSITIONAL+=("$@")
      break
      ;;
    -*)
      echo "Error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      POSITIONAL+=("$1")
      ;;
  esac
  shift
done

if ! [[ "$NUM_PROCS" =~ ^[0-9]+$ ]] || [ "$NUM_PROCS" -lt 1 ]; then
  echo "Error: --num_procs must be a positive integer (got: $NUM_PROCS)" >&2
  exit 2
fi

if [ "${#POSITIONAL[@]}" -gt 0 ]; then
  DATASET_ROOT="${POSITIONAL[0]}"
  FOLDER_ARGS=("${POSITIONAL[@]:1}")
else
  DATASET_ROOT="$DEFAULT_DATASET_ROOT"
  FOLDER_ARGS=()
fi

# Shared render / camera flags (exploring_pa.ipynb lines 13-26).
COMMON_ARGS=(
  --render_size 1024 1024
  --aa 4
  --color_seed 0
  --origin 0 -1 0
  --angle_x 0.39269908169872414    # pi/8
  --angle_y 2.356194490192345      # 3*pi/4
  --distance 4.0
  --opt_pan_start_x 0.039269908169872415   # pi/80 (CAMERA_OPT_PAN_START_X)
  --opt_pan_start_y 3.9269908169872414  # 10*pi/8
  --opt_pan_end_x 0.39269908169872414   # pi/8
  --opt_pan_end_y 2.356194490192345     # 3*pi/4
)

OPT_ARGS=(
  "${COMMON_ARGS[@]}"
  --time_per_iter 1.0
)

mkdir -p "$LOG_DIR"

if [ ! -d "$DATASET_ROOT" ]; then
  echo "Error: dataset root not found: $DATASET_ROOT"
  exit 1
fi

# Optional folder filter from remaining args.
if [ "${#FOLDER_ARGS[@]}" -gt 0 ]; then
  FOLDERS=("${FOLDER_ARGS[@]}")
else
  mapfile -t FOLDERS < <(
    find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
  )
fi

TOTAL_FOLDERS=${#FOLDERS[@]}
if [ "$TOTAL_FOLDERS" -le 0 ]; then
  echo "Error: no folders to process under: $DATASET_ROOT" >&2
  exit 1
fi
if [ "$NUM_PROCS" -gt "$TOTAL_FOLDERS" ]; then
  NUM_PROCS="$TOTAL_FOLDERS"
fi

GPU_IDS=()

set_gpu_ids_from_csv() {
  local csv="$1"
  local item
  GPU_IDS=()
  IFS=',' read -r -a GPU_IDS <<< "$csv"
  for item in "${GPU_IDS[@]}"; do
    if [ -z "$item" ]; then
      echo "Error: empty GPU id in list: $csv" >&2
      exit 2
    fi
  done
}

detect_gpu_ids() {
  local gpu_count
  local gpu

  if [ -n "$GPU_IDS_ARG" ]; then
    set_gpu_ids_from_csv "$GPU_IDS_ARG"
    return
  fi

  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    set_gpu_ids_from_csv "$CUDA_VISIBLE_DEVICES"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count="$(nvidia-smi -L 2>/dev/null | awk 'END {print NR + 0}')"
    if [ "$gpu_count" -gt 0 ]; then
      GPU_IDS=()
      for ((gpu=0; gpu<gpu_count; gpu++)); do
        GPU_IDS+=("$gpu")
      done
      return
    fi
  fi

  # Conservative fallback for single-GPU workstations or wrappers that do not
  # expose nvidia-smi inside the shell.
  GPU_IDS=(0)
}

gpu_for_proc() {
  local proc="$1"
  local n_gpus="${#GPU_IDS[@]}"
  printf '%s\n' "${GPU_IDS[$((proc % n_gpus))]}"
}

detect_gpu_ids

setup_runtime() {
  if [ -f "$ENV_FILE" ]; then
    set +u
    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a
    set -u
  fi

  if command -v conda >/dev/null 2>&1; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate sfn
    set -u
  fi

  cd "$SCRIPT_DIR"
  export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
}

run_mode() {
  local pkl="$1"
  local save_dir="$2"
  local mode="$3"
  shift 3
  "$PYTHON" "$GEN_VIDEO" \
    --input_pkl "$pkl" \
    --mode "$mode" \
    --save_dir "$save_dir" \
    "$@"
}

output_path_for_mode() {
  local save_dir="$1"
  local asset_stem="$2"
  local mode="$3"

  case "$mode" in
    opt_seq)
      printf '%s/%s_opt_seq.mp4\n' "$save_dir" "$asset_stem"
      ;;
    explode_color_compact)
      printf '%s/%s_explode_color_compact.mp4\n' "$save_dir" "$asset_stem"
      ;;
    spiral)
      printf '%s/%s_spiral.mp4\n' "$save_dir" "$asset_stem"
      ;;
    combine)
      printf '%s/%s_opt_then_explode.mp4\n' "$save_dir" "$asset_stem"
      ;;
    *)
      echo "Error: unknown mode '$mode'" >&2
      return 1
      ;;
  esac
}

have_output() {
  local path="$1"
  [ -s "$path" ]
}

run_stage_if_needed() {
  local folder="$1"
  local pkl="$2"
  local save_dir="$3"
  local asset_stem="$4"
  local log="$5"
  local mode="$6"
  shift 6

  local out_path
  out_path="$(output_path_for_mode "$save_dir" "$asset_stem" "$mode")"

  if have_output "$out_path" && ! $OVERWRITE; then
    echo "[skip-stage] ${folder} ${mode} -> ${out_path}" | tee -a "$log"
    return 0
  fi

  if have_output "$out_path"; then
    echo "[overwrite-stage] ${folder} ${mode} -> ${out_path}" | tee -a "$log"
  else
    echo "[run-stage]  ${folder} ${mode} -> ${out_path}" | tee -a "$log"
  fi
  if run_mode "$pkl" "$save_dir" "$mode" "$@" >>"$log" 2>&1; then
    if have_output "$out_path"; then
      echo "[done-stage] ${folder} ${mode} -> ${out_path}" | tee -a "$log"
      return 0
    fi
    echo "[fail-stage] ${folder} ${mode}: command succeeded but output missing: ${out_path}" | tee -a "$log"
    return 1
  fi

  echo "[fail-stage] ${folder} ${mode}: command failed" | tee -a "$log"
  return 1
}

process_range() {
  local proc="$1"
  local start="$2"
  local end="$3"
  local n_ok=0
  local n_skip=0
  local n_fail=0

  echo "Proc ${proc}: folders [${start}, ${end})"

  for ((idx=start; idx<end; idx++)); do
    folder="${FOLDERS[$idx]}"
    shape_dir="${DATASET_ROOT%/}/${folder}"
    pkl="${shape_dir}/primitive_assembly.pkl"
    asset_stem="$(basename "${pkl%.pkl}")"
    opt_out="$(output_path_for_mode "$shape_dir" "$asset_stem" opt_seq)"
    explode_out="$(output_path_for_mode "$shape_dir" "$asset_stem" explode_color_compact)"
    spiral_out="$(output_path_for_mode "$shape_dir" "$asset_stem" spiral)"
    combine_out="$(output_path_for_mode "$shape_dir" "$asset_stem" combine)"
    had_all_outputs=false

    log="${LOG_DIR}/${folder}.log"
    echo "=== ${folder} ===" | tee "$log"

    if [ ! -f "$pkl" ]; then
      echo "[fail] ${folder}: no primitive_assembly.pkl (see ${log})" | tee -a "$log"
      n_fail=$((n_fail + 1))
      continue
    fi

    if have_output "$opt_out" && have_output "$explode_out" \
      && have_output "$spiral_out" && have_output "$combine_out"; then
      had_all_outputs=true
    fi

    if $had_all_outputs && ! $OVERWRITE; then
      echo "[skip] ${folder}: all outputs already exist -> ${combine_out}" | tee -a "$log"
      n_skip=$((n_skip + 1))
      continue
    fi

    shape_failed=false

    if ! run_stage_if_needed "$folder" "$pkl" "$shape_dir" "$asset_stem" "$log" \
      opt_seq "${OPT_ARGS[@]}"; then
      shape_failed=true
    fi

    if ! $shape_failed && ! run_stage_if_needed "$folder" "$pkl" "$shape_dir" "$asset_stem" "$log" \
      explode_color_compact "${COMMON_ARGS[@]}"; then
      shape_failed=true
    fi

    if ! $shape_failed && ! run_stage_if_needed "$folder" "$pkl" "$shape_dir" "$asset_stem" "$log" \
      spiral "${COMMON_ARGS[@]}"; then
      shape_failed=true
    fi

    if ! $shape_failed; then
      if ! have_output "$opt_out" || ! have_output "$explode_out" || ! have_output "$spiral_out"; then
        echo "[fail-stage] ${folder} combine: missing prerequisite outputs" | tee -a "$log"
        echo "             opt_seq=${opt_out} explode_color_compact=${explode_out} spiral=${spiral_out}" | tee -a "$log"
        shape_failed=true
      elif ! run_stage_if_needed "$folder" "$pkl" "$shape_dir" "$asset_stem" "$log" \
        combine "${COMMON_ARGS[@]}" --combine_segments opt_seq explode_color_compact spiral; then
        shape_failed=true
      fi
    fi

    if $shape_failed || ! have_output "$combine_out"; then
      echo "[fail] ${folder} (see ${log})" | tee -a "$log"
      n_fail=$((n_fail + 1))
      continue
    fi

    echo "[ok]   ${folder} -> ${combine_out}" | tee -a "$log"
    n_ok=$((n_ok + 1))
  done

  echo "Proc ${proc} done. ok=${n_ok} skip=${n_skip} fail=${n_fail}"

  if [ "$n_fail" -gt 0 ]; then
    return 1
  fi
}

echo "Dataset root:   $DATASET_ROOT"
echo "Folders:        $TOTAL_FOLDERS"
echo "Processes:      $NUM_PROCS"
echo "GPU ids:        ${GPU_IDS[*]}"
echo "Overwrite:      $OVERWRITE"
echo "Logs:           $LOG_DIR"

if [ "$NUM_PROCS" -eq 1 ]; then
  setup_runtime
  if ! process_range 0 0 "$TOTAL_FOLDERS"; then
    exit 1
  fi
  echo "All jobs completed."
  exit 0
fi

ITEMS_PER_PROC=$((TOTAL_FOLDERS / NUM_PROCS))
REMAINDER=$((TOTAL_FOLDERS % NUM_PROCS))
CURRENT_START=0
PIDS=()

for ((proc=0; proc<NUM_PROCS; proc++)); do
  if [ "$proc" -lt "$REMAINDER" ]; then
    CURRENT_ITEMS=$((ITEMS_PER_PROC + 1))
  else
    CURRENT_ITEMS=$ITEMS_PER_PROC
  fi

  CURRENT_END=$((CURRENT_START + CURRENT_ITEMS))

  if [ "$CURRENT_ITEMS" -le 0 ]; then
    continue
  fi

  gpu_id="$(gpu_for_proc "$proc")"
  proc_log="${LOG_DIR}/proc_${proc}_videos.out"
  echo "Proc ${proc} (GPU ${gpu_id}): folders [${CURRENT_START}, ${CURRENT_END}) -> ${proc_log}"

  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    setup_runtime
    process_range "$proc" "$CURRENT_START" "$CURRENT_END"
  ) > "$proc_log" 2>&1 &
  PIDS+=("$!")

  CURRENT_START=$CURRENT_END
done

echo "Spawned ${#PIDS[@]} background processes. Logs: ${LOG_DIR}/proc_<id>_videos.out"
echo "Monitor with: tail -F ${LOG_DIR}/proc_*_videos.out"

N_FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    N_FAIL=$((N_FAIL + 1))
  fi
done

echo "All jobs completed. failed_processes=${N_FAIL}"

if [ "$N_FAIL" -gt 0 ]; then
  exit 1
fi
