import pickle
import numpy as np
import imageio
import cv2

# "/home/haotian/Point-Policy/data/pick_place_red_mug_1/processed_data_pkl/expert_demos/franka_env/pick_place_red_mug.pkl" 
# "/home/haotian/Point-Policy/data/pick_place_red_mug_1/processed_data_pkl/expert_demos/franka_env/pick_place_red_mug.pkl"
# "/home/haotian/Point-Policy/data/pick_place_red_mug_robot/processed_data_pkl/expert_demos/franka_env/pick_place_red_mug.pkl"
# "/home/haotian/Point-Policy/data/pick_place_red_mug_1/processed_data_pkl/expert_demos/franka_env/pick_place_red_mug.pkl"
PKL_PATH = "/home/haotian/Point-Policy/data/pick_place_red_mug_robot/processed_data_pkl/expert_demos/franka_env/pick_place_red_mug.pkl"

with open(PKL_PATH, "rb") as f:
    traj = pickle.load(f)


o = traj["observations"][0]

n_frames = len(o["pixels1"])

POLARIS_SIZE = (1280, 720)

def draw_tracks(img, obj_2d, rob_2d):
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, POLARIS_SIZE)  # resize first
    
    for i, pt in enumerate(obj_2d):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
        cv2.putText(img, f"obj{i}", (x+3, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    for i, pt in enumerate(rob_2d):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, f"rob{i}", (x+3, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return img  # (480, 640, 3)

frames = []
for i in range(n_frames):
    img1 = draw_tracks(o["pixels1"][i].astype(np.uint8), o["object_tracks_pixels1"][i], o["robot_tracks_pixels1"][i])
    img2 = draw_tracks(o["pixels2"][i].astype(np.uint8), o["object_tracks_pixels2"][i], o["robot_tracks_pixels2"][i])
    frame = np.concatenate([img1, img2], axis=1)
    frames.append(frame)

imageio.mimwrite("trajectory.mp4", frames, fps=30, quality=8)
print("Saved trajectory.mp4")