#!/bin/bash

# All experiment directories based on the screenshot
ALL_EXPERIMENTS=(
    "pick_place_red_mug_robot_5_human_5"
    "pick_place_red_mug_robot_5_human_15"
    "pick_place_red_mug_robot_5_human_25"
    "pick_place_red_mug_robot_5_human_35"
    "pick_place_red_mug_robot_15_human_0"
    "pick_place_red_mug_robot_15_human_15"
    "pick_place_red_mug_robot_15_human_25"
    "pick_place_red_mug_robot_15_human_35"
    "pick_place_red_mug_robot_25_human_0"
    "pick_place_red_mug_robot_25_human_15"
    "pick_place_red_mug_robot_25_human_25"
    "pick_place_red_mug_robot_25_human_35"
    "pick_place_red_mug_robot_35_human_0"
    "pick_place_red_mug_robot_35_human_15"
    "pick_place_red_mug_robot_35_human_25"
    "pick_place_red_mug_robot_35_human_35"
)

BASE_PATH="/home/haotian/Point-Policy/data"
BATCH_SIZE=4

# Function to run a single training job
run_job() {
    local gpu_id=$1
    local exp_name=$2
    local data_dir="${BASE_PATH}/${exp_name}/processed_data_pkl/expert_demos"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${exp_name} on GPU ${gpu_id}"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
        agent=point_policy \
        suite=point_policy \
        dataloader=point_policy \
        eval=false \
        suite.use_robot_points=true \
        suite.use_object_points=true \
        suite/task/franka_env=pick_place_red_mug \
        data_dir=${data_dir} \
        experiment=${exp_name}
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${exp_name} on GPU ${gpu_id}"
}

# Main loop: process experiments in batches of 6
total_experiments=${#ALL_EXPERIMENTS[@]}
batch_num=1

for ((i=0; i<total_experiments; i+=BATCH_SIZE)); do
    echo "=============================================="
    echo "Starting Batch ${batch_num} (experiments $((i+1)) to $((i+BATCH_SIZE < total_experiments ? i+BATCH_SIZE : total_experiments)))"
    echo "=============================================="
    
    # Launch up to 6 jobs: 3 on GPU 0, 3 on GPU 1
    job_count=0
    for ((j=0; j<BATCH_SIZE && i+j<total_experiments; j++)); do
        exp_name=${ALL_EXPERIMENTS[$((i+j))]}
        
        # Alternate GPUs: jobs 0,1,2 -> GPU 0; jobs 3,4,5 -> GPU 1
        if [ $j -lt 2 ]; then
            gpu_id=0
        else
            gpu_id=1
        fi
                
        run_job ${gpu_id} ${exp_name} &
        ((job_count++))
    done
    
    echo "Launched ${job_count} jobs in batch ${batch_num}, waiting for completion..."
    wait
    
    echo "=============================================="
    echo "Batch ${batch_num} completed!"
    echo "=============================================="
    echo ""
    
    ((batch_num++))
done

echo "========================================"
echo "All training jobs completed!"
echo "========================================"