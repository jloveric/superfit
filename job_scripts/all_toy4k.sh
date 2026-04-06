#!/bin/bash

# Usage: ./all_toy4k.sh <start_ind> <end_ind> <num_gpus> [ablation]
# Example: ./all_toy4k.sh 0 100 4
# Example with ablation: ./all_toy4k.sh 0 100 4 2

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <start_ind> <end_ind> <num_gpus> [ablation]"
    echo "  start_ind: Starting index (inclusive)"
    echo "  end_ind: Ending index (exclusive)"
    echo "  num_gpus: Number of GPUs to use"
    echo "  ablation: Optional ablation number (default: 0)"
    echo "  aot_postfix: Optional AOT postfix (default: aott)"
    exit 1
fi

START_IND=$1
END_IND=$2
NUM_GPUS=$3
ABLATION=${4:-0}
AOT_POSTFIX=${5:-aott}
# Log directory configuration
LOG_DIR="/users/aganesh8/data/aganesh8/projects/project_sf/logs"
mkdir -p "$LOG_DIR"

# Calculate total items and items per GPU
TOTAL_ITEMS=$((END_IND - START_IND))
ITEMS_PER_GPU=$((TOTAL_ITEMS / NUM_GPUS))
REMAINDER=$((TOTAL_ITEMS % NUM_GPUS))

SCRIPT_DIR="/users/aganesh8/data/aganesh8/projects/project_sf/superfit"
ENV_FILE="/users/aganesh8/data/aganesh8/projects/project_sf/.env"

echo "Starting jobs: indices $START_IND to $END_IND across $NUM_GPUS GPUs"
echo "Items per GPU: $ITEMS_PER_GPU (with $REMAINDER GPUs getting 1 extra)"

# Spawn a job for each GPU
CURRENT_START=$START_IND
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    # Distribute remainder: first 'remainder' GPUs get 1 extra item
    if [ $gpu -lt $REMAINDER ]; then
        CURRENT_ITEMS=$((ITEMS_PER_GPU + 1))
    else
        CURRENT_ITEMS=$ITEMS_PER_GPU
    fi
    
    CURRENT_END=$((CURRENT_START + CURRENT_ITEMS))
    
    # Skip if no items for this GPU
    if [ $CURRENT_ITEMS -le 0 ]; then
        continue
    fi
    
    echo "GPU $gpu: Processing indices $CURRENT_START to $CURRENT_END"
    
    # Spawn the job in background
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        source ~/.bashrc
        # Load environment variables
        if [ -f "$ENV_FILE" ]; then
            set -a
            source "$ENV_FILE"
            set +a
        fi
        # Activate conda environment
        conda activate sfn
        cd "$SCRIPT_DIR"
        python scripts/testset_fit_primitives.py --start_ind $CURRENT_START --end_ind $CURRENT_END --ablation $ABLATION --fastmode --aot_postfix $AOT_POSTFIX
        # python scripts/testset_fit_partwise.py --start_ind $CURRENT_START --end_ind $CURRENT_END --ablation $ABLATION --fastmode  --aot_postfix $AOT_POSTFIX
        # python scripts/testset_fit_primitives.py --start_ind $CURRENT_START --end_ind $CURRENT_END --ablation $ABLATION --fastmode --overwrite --aot_postfix $AOT_POSTFIX --dataset partobjaverse
        # python scripts/texture_on_testset.py --start_ind $CURRENT_START --end_ind $CURRENT_END --input_path /users/aganesh8/data/aganesh8/data/project_sf/outputs/partobjaverse/ablation_2_param --save_html
    ) > "${LOG_DIR}/proc_${gpu}_${AOT_POSTFIX}.out" 2>&1 &
    
    CURRENT_START=$CURRENT_END
done

echo "All jobs spawned. Logs are in ${LOG_DIR}/"
echo "Log files: proc_0.out, proc_1.out, ..., proc_$((NUM_GPUS-1)).out"
echo "Use 'jobs' or 'ps aux | grep generate_on_testset' to monitor"

# Wait for all background jobs
wait
echo "All jobs completed."
