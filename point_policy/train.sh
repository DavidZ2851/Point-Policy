!/bin/bash

DATA_DIRS=(
    "/home/haotian/Point-Policy/data/pick_place_red_mug_robot_5_human_5/processed_data_pkl/expert_demos"
    "/home/haotian/Point-Policy/data/pick_place_red_mug_robot_5_human_15/processed_data_pkl/expert_demos"
    "/home/haotian/Point-Policy/data/pick_place_red_mug_robot_5_human_25/processed_data_pkl/expert_demos"
    "/home/haotian/Point-Policy/data/pick_place_red_mug_robot_5_human_35/processed_data_pkl/expert_demos"
    "/home/haotian/Point-Policy/data/pick_place_red_mug_robot_15_human_0/processed_data_pkl/expert_demos"
    "/home/haotian/Point-Policy/data/pick_place_red_mug_robot_15_human_5/processed_data_pkl/expert_demos"
)

EXP_NAMES=(
    "pick_place_red_mug_robot_5_human_5"
    "pick_place_red_mug_robot_5_human_15"
    "pick_place_red_mug_robot_5_human_25"
    "pick_place_red_mug_robot_5_human_35"
    "pick_place_red_mug_robot_15_human_0"
    "pick_place_red_mug_robot_15_human_5"
)

CUDA_VISIBLE_DEVICES=0 python train.py \
    agent=point_policy \
    suite=point_policy \
    dataloader=point_policy \
    eval=false \
    suite.use_robot_points=true \
    suite.use_object_points=true \
    suite/task/franka_env=pick_place_red_mug \
    data_dir=${DATA_DIRS[0]} \
    experiment=${EXP_NAMES[0]} &

CUDA_VISIBLE_DEVICES=0 python train.py \
    agent=point_policy \
    suite=point_policy \
    dataloader=point_policy \
    eval=false \
    suite.use_robot_points=true \
    suite.use_object_points=true \
    suite/task/franka_env=pick_place_red_mug \
    data_dir=${DATA_DIRS[1]} \
    experiment=${EXP_NAMES[1]} &

CUDA_VISIBLE_DEVICES=1 python train.py \
    agent=point_policy \
    suite=point_policy \
    dataloader=point_policy \
    eval=false \
    suite.use_robot_points=true \
    suite.use_object_points=true \
    suite/task/franka_env=pick_place_red_mug \
    data_dir=${DATA_DIRS[2]} \
    experiment=${EXP_NAMES[2]} &

CUDA_VISIBLE_DEVICES=1 python train.py \
    agent=point_policy \
    suite=point_policy \
    dataloader=point_policy \
    eval=false \
    suite.use_robot_points=true \
    suite.use_object_points=true \
    suite/task/franka_env=pick_place_red_mug \
    data_dir=${DATA_DIRS[3]} \
    experiment=${EXP_NAMES[3]} &

CUDA_VISIBLE_DEVICES=0 python train.py \
    agent=point_policy \
    suite=point_policy \
    dataloader=point_policy \
    eval=false \
    suite.use_robot_points=true \
    suite.use_object_points=true \
    suite/task/franka_env=pick_place_red_mug \
    data_dir=${DATA_DIRS[4]} \
    experiment=${EXP_NAMES[4]} &

CUDA_VISIBLE_DEVICES=1 python train.py \
    agent=point_policy \
    suite=point_policy \
    dataloader=point_policy \
    eval=false \
    suite.use_robot_points=true \
    suite.use_object_points=true \
    suite/task/franka_env=pick_place_red_mug \
    data_dir=${DATA_DIRS[5]} \
    experiment=${EXP_NAMES[5]} &

wait
echo "All training jobs completed!"
