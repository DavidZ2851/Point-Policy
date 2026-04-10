#!/bin/bash

data_dir_1="/home/haotian/Point-Policy/data/pick_place_red_mug_robot_15_human_25/processed_data_pkl/expert_demos"
bc_weight_1="/home/haotian/Point-Policy/point_policy/exp_local/2026.04.09/pick_place_red_mug_robot_15_human_25/deterministic/072233_hidden_dim_256/snapshot/100000.pt"

data_dir_2="/home/haotian/Point-Policy/data/pick_place_red_mug_robot_15_human_35/processed_data_pkl/expert_demos"
bc_weight_2="/home/haotian/Point-Policy/point_policy/exp_local/2026.04.09/pick_place_red_mug_robot_15_human_35/deterministic/072233_hidden_dim_256/snapshot/100000.pt"

echo "Starting server 1 on port 8765..."
CUDA_VISIBLE_DEVICES=0 python serve_policy.py \
    --bc_weight "$bc_weight_1" \
    --port 8765 \
    --overrides "agent=point_policy" "suite=point_policy" "dataloader=point_policy" \
    "suite.use_robot_points=true" "suite.use_object_points=true" \
    "experiment=eval_point_policy" "suite/task/franka_env=pick_place_red_mug" \
    "agent.max_episode_len=450" "data_dir=${data_dir_1}" &
PID1=$!

echo "Starting server 2 on port 8766..."
CUDA_VISIBLE_DEVICES=1 python serve_policy.py \
    --bc_weight "$bc_weight_2" \
    --port 8766 \
    --overrides "agent=point_policy" "suite=point_policy" "dataloader=point_policy" \
    "suite.use_robot_points=true" "suite.use_object_points=true" \
    "experiment=eval_point_policy" "suite/task/franka_env=pick_place_red_mug" \
    "agent.max_episode_len=450" "data_dir=${data_dir_2}" &
PID2=$!

echo "Servers running: PID1=$PID1 (port 8765), PID2=$PID2 (port 8766)"
echo "Press Ctrl+C to stop all servers"

# 捕获 Ctrl+C 信号，优雅关闭
trap "echo 'Shutting down...'; kill $PID1 $PID2 2>/dev/null; exit" SIGINT SIGTERM

wait
echo "All servers shut down."