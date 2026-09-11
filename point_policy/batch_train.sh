#!/bin/bash

TASK_1=pick_place_toys
TASK_2=insert_donut

# entries are "<task>:<experiment>" -- each experiment carries its own task
ALL_EXPERIMENTS=(
    # pick_place_red_mug_robot_5_human_0
    # pick_place_red_mug_robot_5_human_40
    # pick_place_red_mug_robot_5_human_70
    # pick_place_red_mug_robot_5_human_100
    # pick_place_red_mug_robot_40_human_0
    # pick_place_red_mug_robot_40_human_40
    # pick_place_red_mug_robot_40_human_70
    # pick_place_red_mug_robot_40_human_100
    # pick_place_toys_robot_200
    # pick_place_toys_robot_100
    # pick_place_toys_robot_50
    # insert_donut_robot_100_human_100
    # insert_donut_robot_100_human_200
    # insert_donut_robot_100_human_300
    # pick_place_red_mug_realrobot_40_human_0
    # pick_place_red_mug_realrobot_40_human_100
    # stack_bowls_robot_100_human_0
    # stack_bowls_robot_100_human_100
    # stack_bowls_robot_100_human_200
    # stack_bowls_robot_100_human_300
    # pick_place_toys_robot_100_sim_1
    # pick_place_toys_robot_100_sim
    # pick_toys_realrobot_100
    # pick_toys_simrobot_100_h300
    # pick_toys_realrobot_100_h300
    # pick_place_red_mug_40_simrobot_h0
    # pick_place_red_mug_40_simrobot_h40
    # pick_place_red_mug_40_simrobot_h70
    # pick_place_red_mug_40_simrobot_h100
    # insert_donut_100_simrobot_h0
    # insert_donut_100_simrobot_h100
    # insert_donut_100_simrobot_h200
    # insert_donut_100_simrobot_h300
    # pick_place_red_mug_40_simrobot_h100_2
    # insert_donut_simrobot_100_2
    # insert_donut_simrobot_100_h100_2
    # insert_donut_simrobot_100_h200_2
    # insert_donut_simrobot_100_h300_2
    # insert_donut_simrobot_100_h100_3
    # insert_donut_simrobot_100_h300_3
    # insert_donut_simrobot_100_h300_2
    # stack_bowls_simrobot_100_h0
    # stack_bowls_simrobot_100_h100
    # stack_bowls_simrobot_100_h200
    # stack_bowls_simrobot_100_h300
    # stack_bowls_simrobot_40_h100_right
    # pick_place_red_mug_simrobot_40_h100_right
    # right_mug
    # right_bowl
    # "${TASK_1}:pick_place_toys_simrobot_400"
    "${TASK_2}:insert_donut_400"
)

NUM_DEMOS=400
BASE_PATH="/data/haotian/pp_data"
LOG_DIR="./logs"
MEM_THRESHOLD=3000
JOBS_PER_GPU=1
# set to a space-separated list (e.g. "0" or "0 1") to pick GPUs explicitly and
# skip the free-GPU check -- use this to stack jobs onto an already-busy GPU
FORCE_GPUS=""
mkdir -p "${LOG_DIR}"

get_free_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' -v threshold="${MEM_THRESHOLD}" '$2 < threshold {print $1}'
}

run_job() {
    local gpu_id=$1
    local spec=$2
    local task="${spec%%:*}"
    local exp_name="${spec#*:}"
    local data_dir="${BASE_PATH}/${task}/${exp_name}/processed_data_pkl/expert_demos"
    local log_file="${LOG_DIR}/${exp_name}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${exp_name} (task=${task}) on GPU ${gpu_id} (log: ${log_file})"

    CUDA_VISIBLE_DEVICES=${gpu_id} nohup python -u train.py \
        agent=point_policy \
        suite=point_policy \
        dataloader=point_policy \
        eval=false \
        suite.use_robot_points=true \
        suite.use_object_points=true \
        suite/task/franka_env=${task} \
        data_dir=${data_dir} \
        experiment=${exp_name} \
        num_demos_per_task=${NUM_DEMOS} \
        > "${log_file}" 2>&1 &

    echo $! > "${LOG_DIR}/${exp_name}.pid"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${exp_name} launched with PID $!"
}

if [ -n "${FORCE_GPUS}" ]; then
    read -ra free_gpus <<< "${FORCE_GPUS}"
    echo "Using forced GPUs: ${free_gpus[*]} (skipping free-GPU check)"
else
    mapfile -t free_gpus < <(get_free_gpus)
fi

if [ ${#free_gpus[@]} -eq 0 ]; then
    echo "No free GPUs available. Exiting."
    exit 1
fi

echo "Free GPUs: ${free_gpus[*]}"

exp_index=0
total_experiments=${#ALL_EXPERIMENTS[@]}

for gpu_id in "${free_gpus[@]}"; do
    [ ${exp_index} -ge ${total_experiments} ] && break

    echo "--- Assigning ${JOBS_PER_GPU} jobs to GPU ${gpu_id} ---"
    for ((k=0; k<JOBS_PER_GPU; k++)); do
        [ ${exp_index} -ge ${total_experiments} ] && break
        run_job ${gpu_id} "${ALL_EXPERIMENTS[$exp_index]}"
        exp_index=$((exp_index + 1))
    done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. Launched ${exp_index}/${total_experiments} experiments."