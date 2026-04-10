import pickle
import numpy as np
import imageio
import cv2
from pathlib import Path

PKL_PATH = Path("/home/haotian/Point-Policy/data/pick_place_red_mug_human_v2/processed_data_pkl/expert_demos/franka_env/pick_place_red_mug.pkl")
OUTPUT_DIR = PKL_PATH.parent.parent.parent.parent / "vis_2d"  # Go up to pick_place_red_mug_1, then add vis_2d
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(PKL_PATH, "rb") as f:
    traj = pickle.load(f)

POLARIS_SIZE = (1280, 720)

def draw_tracks(img, obj_2d, rob_2d):
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, POLARIS_SIZE)
    
    for i, pt in enumerate(obj_2d):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
        cv2.putText(img, f"obj{i}", (x+3, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    for i, pt in enumerate(rob_2d):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, f"rob{i}", (x+3, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return img

observations = traj["observations"]

print(f"Found {len(observations)} episodes")
print(f"Saving to {OUTPUT_DIR}")

for ep_idx, o in enumerate(observations):
    n_frames = len(o["pixels1"])

    frames = []
    for i in range(n_frames):

        img1 = draw_tracks(o["pixels1"][i].astype(np.uint8), o["object_tracks_pixels1"][i], o["robot_tracks_pixels1"][i])
        img2 = draw_tracks(o["pixels2"][i].astype(np.uint8), o["object_tracks_pixels2"][i], o["robot_tracks_pixels2"][i])
        frame = np.concatenate([img1, img2], axis=1)
        frames.append(frame)

    output_path = OUTPUT_DIR / f"episode_{ep_idx:03d}.mp4"
    imageio.mimwrite(str(output_path), frames, fps=30, quality=8)
    print(f"Saved {output_path} ({n_frames} frames)")

print(f"\nDone! Saved {len(observations)} videos to {OUTPUT_DIR}")