"""
Combined pipeline: Custom robot data -> point tracking -> robot track conversion -> final pkl

Usage:
    python create_point_policy_dataset_robot.py \
        --data_dir /home/haotian/polaris/data/pick_place_red_cup_40 \
        --calib_path /home/haotian/Point-Policy/calib/calib.npy \
        --task_name pick_place_red_mug \
        --output_dir /home/haotian/Point-Policy/data/pick_place_red_mug_robot \
        --process_points
"""

import sys
sys.path.append("../../")

import yaml
import argparse
import pickle as pkl
from pathlib import Path
import cv2
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import zoom

from point_utils.points_class import PointsClass
from utils import camera2pixelkey, pixel2d_to_3d_torch, triangulate_points, ee_pose_to_robot_points, project_points

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Custom robot data -> point policy pkl pipeline")
parser.add_argument("--data_dir",       type=str, required=True, nargs="+")
parser.add_argument("--calib_path",     type=str, required=True)
parser.add_argument("--task_name",      type=str, required=True)
parser.add_argument("--num_demos",      type=int, default=None)
parser.add_argument("--process_points", action="store_true")
parser.add_argument("--output_dir",     type=str, default="./processed_data")
args = parser.parse_args()

DATA_DIRS      = [Path(d) for d in args.data_dir]
CALIB_PATH     = Path(args.calib_path)
TASK_NAME      = args.task_name
NUM_DEMOS      = args.num_demos
process_points = args.process_points
OUTPUT_DIR     = Path(args.output_dir)

camera_indices = [1, 2]  # cam0.mp4 -> cam_1, cam1.mp4 -> cam_2

# ── image settings ────────────────────────────────────────────────────────────
original_img_size = (1280, 720)
crop_h, crop_w    = (0.0, 1.0), (0.0, 1.0)
save_img_size     = (
    int(original_img_size[0] * (crop_w[1] - crop_w[0])),
    int(original_img_size[1] * (crop_h[1] - crop_h[0])),
)
save_image_size   = (256, 256)

object_labels     = ["objects"]  # no human hand for robot demos

# ── output paths ──────────────────────────────────────────────────────────────
SAVE_DATA_PATH = OUTPUT_DIR / "processed_data_pkl"
SAVE_DIR       = SAVE_DATA_PATH / "expert_demos" / "franka_env"
SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── calibration ───────────────────────────────────────────────────────────────
calibration_data = np.load(CALIB_PATH, allow_pickle=True).item()

# ── load point tracking models ────────────────────────────────────────────────
if process_points:
    with open("../../cfgs/suite/points_cfg.yaml") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    root_dir = cfg["root_dir"]
    cfg["dift_path"]            = f"{root_dir}/{cfg['dift_path']}"
    cfg["cotracker_checkpoint"] = f"{root_dir}/{cfg['cotracker_checkpoint']}"
    cfg["task_name"]            = TASK_NAME
    cfg["pixel_keys"]           = [camera2pixelkey[f"cam_{i}"] for i in camera_indices]
    cfg["object_labels"]        = object_labels
    points_class = PointsClass(**cfg)

# ── helpers ───────────────────────────────────────────────────────────────────
def load_video_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        frame = frame[
            int(h * crop_h[0]):int(h * crop_h[1]),
            int(w * crop_w[0]):int(w * crop_w[1]),
        ]
        frame = cv2.resize(frame, save_img_size)
        frames.append(frame)
    cap.release()
    return np.array(frames) if frames else None


# ── collect episodes ──────────────────────────────────────────────────────────
all_episode_dirs = []
for data_dir in DATA_DIRS:
    ep_dirs = sorted(data_dir.glob("episode_*"))
    all_episode_dirs.extend(ep_dirs)

if NUM_DEMOS is not None:
    all_episode_dirs = all_episode_dirs[:NUM_DEMOS]

print(f"Found {len(all_episode_dirs)} episodes across {len(DATA_DIRS)} data dirs.")

observations  = []
max_cartesian = np.array([ 0.24113624,  0.02650134,  0.36749125, -3.1381004 , -0.0293628 ,  0.0104839 ], dtype=np.float32)
min_cartesian = np.array([ 0.2408465 ,  0.02610677,  0.36714548, -3.1390593 , -0.03289817,  0.00859457], dtype=np.float32)
max_gripper   = np.float32(-1.0)
min_gripper   = np.float32(-1.0)

# ── main loop ─────────────────────────────────────────────────────────────────
for ep_idx, ep_dir in enumerate(all_episode_dirs):
    ep_stem = ep_dir.name

    print(f"\n[{ep_idx+1}/{len(all_episode_dirs)}] Processing {ep_stem} ...")

    observation = {}
    skip = False

    # ── load trajectory ───────────────────────────────────────────────────────
    traj_path = ep_dir / "trajectory.npz"
    if not traj_path.exists():
        print(f"  WARNING: no trajectory.npz in {ep_dir}, skipping.")
        continue
    traj_data = np.load(traj_path, allow_pickle=True)
    gripper_pcd = traj_data["gripper_pcd"]  # (T, 4, 3)
    states_ee   = traj_data["states_ee"]    # (T, 6)
    print(f"  gripper_pcd: {gripper_pcd.shape}")
    print(f"  states_ee: {states_ee.shape}")

    # ── load RGB frames ───────────────────────────────────────────────────────
    # cam0.mp4 -> pixels1 (cam_1), cam1.mp4 -> pixels2 (cam_2)
    cam_map = {
        "cam0.mp4": ("cam_1", "pixels1"),
        "cam1.mp4": ("cam_2", "pixels2"),
    }
    for cam_file, (camera_name, pixel_key) in cam_map.items():
        video_file = ep_dir / cam_file
        frames = load_video_frames(video_file)
        if frames is None:
            print(f"  WARNING: cannot open {video_file}, skipping.")
            skip = True
            break
        # trim to match trajectory length
        frames = frames[:len(gripper_pcd)]
        observation[pixel_key] = frames
        print(f"  {pixel_key}: {frames.shape}")
    if skip:
        continue

    # ── compute robot points from EE poses ────────────────────────────────────
    robot_points, gripper_states = ee_pose_to_robot_points(gripper_pcd, states_ee)
    observation["gripper_states"] = gripper_states


    # ── point tracking (object points only) ───────────────────────────────────
    if process_points:
        mark_every = 8
        save = True

        for cam_idx in camera_indices:
            if not save:
                break
            camera_name = f"cam_{cam_idx}"
            pixel_key   = camera2pixelkey[camera_name]
            frames      = [f[..., ::-1] for f in observation[pixel_key]]  # BGR -> RGB

            points_class.add_to_image_list(frames[0], pixel_key)
            for object_label in object_labels:
                points_class.find_semantic_similar_points(pixel_key, object_label)

            try:
                points_class.track_points(pixel_key, last_n_frames=mark_every, is_first_step=True)
            except Exception as e:
                print(f"  Error tracking: {e}")
                points_class.reset_episode()
                save = False
                continue

            points_class.track_points(pixel_key, last_n_frames=mark_every, one_frame=(mark_every == 1))
            points_list = [points_class.get_points_on_image(pixel_key)[0]]

            for f_idx, image in enumerate(frames[1:]):
                print(f"  Traj: {ep_idx}, Frame: {f_idx}, Cam: {pixel_key}")
                points_class.add_to_image_list(image, pixel_key)

                if (f_idx + 1) % mark_every == 0 or f_idx == len(frames) - 2:
                    to_add = mark_every - (f_idx + 1) % mark_every
                    if to_add < mark_every:
                        for _ in range(to_add):
                            points_class.add_to_image_list(image, pixel_key)
                    else:
                        to_add = 0

                    points_class.track_points(pixel_key, last_n_frames=mark_every, one_frame=(mark_every == 1))
                    points = points_class.get_points_on_image(pixel_key, last_n_frames=mark_every)
                    for j in range(mark_every - to_add):
                        points_list.append(points[j])

            observation[f"object_tracks_pixels{cam_idx}"] = torch.stack(points_list).numpy()  # (T, N, 2)
            points_class.reset_episode()

        if not save:
            continue

        # ── triangulate object points 2D -> 3D ────────────────────────────────
        for cam_idx in camera_indices:
            observation[f"object_tracks_3d_{camera2pixelkey[f'cam_{cam_idx}']}"] = []

        last_pixel_key = camera2pixelkey[f"cam_{camera_indices[-1]}"]
        for t_idx in range(len(observation[f"object_tracks_pixels{camera_indices[-1]}"])):
            P_list, pts_list = [], []
            for cam_idx in camera_indices:
                camera_name = f"cam_{cam_idx}"
                pixel_key   = camera2pixelkey[camera_name]
                extr = calibration_data[camera_name]["ext"]
                intr = calibration_data[camera_name]["int"]
                P_list.append(np.concatenate([intr, np.zeros((3, 1))], axis=1) @ extr)

                pt2d = observation[f"object_tracks_pixels{cam_idx}"][t_idx]
                point_h = pt2d[:, 1]
                h_orig_cropped = original_img_size[1] * (crop_h[1] - crop_h[0])
                point_h_orig   = ((point_h / save_img_size[1]) * h_orig_cropped + original_img_size[1] * crop_h[0]).astype(np.int32)

                point_w = pt2d[:, 0]
                w_orig_cropped = original_img_size[0] * (crop_w[1] - crop_w[0])
                point_w_orig   = ((point_w / save_img_size[0]) * w_orig_cropped + original_img_size[0] * crop_w[0]).astype(np.int32)

                pts_list.append(np.column_stack((point_w_orig, point_h_orig)))

            pts3d = triangulate_points(P_list, pts_list)
            for cam_idx in camera_indices:
                pixel_key = camera2pixelkey[f"cam_{cam_idx}"]
                observation[f"object_tracks_3d_{pixel_key}"].append(pts3d[:, :3])

        for cam_idx in camera_indices:
            pixel_key = camera2pixelkey[f"cam_{cam_idx}"]
            observation[f"object_tracks_3d_{pixel_key}"] = np.array(observation[f"object_tracks_3d_{pixel_key}"])

    # ── resize pixels and project tracks to 2D ────────────────────────────────
    for cam_idx in camera_indices:
        camera_name = f"cam_{cam_idx}"
        pixel_key   = camera2pixelkey[camera_name]

        # resize pixels
        pixels = [cv2.resize(p, save_image_size) for p in observation[pixel_key]]
        observation[pixel_key] = np.array(pixels)

        object_points_3d = observation.get(f"object_tracks_3d_{pixel_key}", np.zeros((len(robot_points), 1, 3)))

        # store 3D robot tracks
        observation[f"robot_tracks_3d_{pixel_key}"] = robot_points
        observation[f"object_tracks_3d_{pixel_key}"] = object_points_3d

        # project to 2D
        robot_2d, object_2d = project_points(calibration_data, robot_points, object_points_3d, camera_name)
        observation[f"robot_tracks_{pixel_key}"]  = robot_2d
        observation[f"object_tracks_{pixel_key}"] = object_2d

    observations.append(observation)

# ── save ──────────────────────────────────────────────────────────────────────
data = {
    "observations":  observations,
    "max_cartesian": max_cartesian.astype(np.float32) if max_cartesian is not None else None,
    "min_cartesian": min_cartesian.astype(np.float32) if min_cartesian is not None else None,
    "max_gripper":   np.float32(max_gripper) if max_gripper is not None else None,
    "min_gripper":   np.float32(min_gripper) if min_gripper is not None else None,
    "max_sensor":    None,
    "min_sensor":    None,
}

out_path = SAVE_DIR / f"{TASK_NAME}.pkl"
with open(out_path, "wb") as f:
    pkl.dump(data, f)

print(f"\nDone. Saved {len(observations)} episodes to {out_path}")