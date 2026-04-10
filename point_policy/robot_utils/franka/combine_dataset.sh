HUMAN_DATA="/home/haotian/Point-Policy/data/pick_place_red_mug_human_v2/processed_data_pkl/expert_demos/franka_env/pick_place_red_mug.pkl"
ROBOT_DATA="/home/haotian/Point-Policy/data/pick_place_red_mug_robot_v2/processed_data_pkl/expert_demos/franka_env/pick_place_red_mug.pkl"

# OUTPUT_DIR="/home/haotian/Point-Policy/data/pick_place_red_mug_robot_20_human_15"

# python combine_dataset.py \
#     --input ${HUMAN_DATA}:15 ${ROBOT_DATA}:20 \
#     --output-dir ${OUTPUT_DIR}

# python combine_dataset.py \
#     --input /home/haotian/Point-Policy/data/pick_place_red_mug_filter_1/processed_data_pkl/expert_demos/franka_env/pick_place_red_mug.pkl /home/haotian/Point-Policy/data/pick_place_red_mug_filter_2/processed_data_pkl/expert_demos/franka_env/pick_place_red_mug.pkl \
#     --output-dir /home/haotian/Point-Policy/data/pick_place_red_mug_human_filter


python combine_dataset.py \
    --input ${ROBOT_DATA}:35 ${HUMAN_DATA}:0 \
    --output-dir /home/haotian/Point-Policy/data/pick_place_red_mug_robot_35_human_0