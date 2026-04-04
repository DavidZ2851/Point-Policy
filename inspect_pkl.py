import pickle
from pprint import pprint

files = [
    "/home/haotian/Point-Policy/bottle_on_rack.pkl",
    "/home/haotian/Point-Policy/bowl_in_oven.pkl",
    "/home/haotian/Point-Policy/bottle_upright.pkl",
    "/home/haotian/Point-Policy/bread_on_plate.pkl",
    "/home/haotian/Point-Policy/drawer_close.pkl",
    "/home/haotian/Point-Policy/sweep_broom.pkl",
]

keys = [
    "max_cartesian",
    "min_cartesian",
    "max_gripper",
    "min_gripper",
    "max_sensor",
    "min_sensor",
]

for path in files:
    print(f"\n=== {path} ===")
    with open(path, "rb") as f:
        traj = pickle.load(f)

    print("dict_keys:", traj.keys())
    for k in keys:
        print(f"{k}:")
        pprint(traj[k])
