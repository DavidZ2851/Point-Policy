"""
Combined pipeline: LeRobot data_dir -> DIFT + CoTracker -> triangulation
                   -> robot track conversion -> final pkl

Reads observation.state (10,) [rot6d(6), x, y, z, gripper] directly from
parquet files — no LeRobotDataset import needed. Images are decoded from
the mp4 video files the same way as the human/robot scripts.

Usage:
    python create_point_policy_dataset_lerobot.py \
        --data_dir /path/to/lerobot/cache/org/dataset \
        --calib_path /path/to/calib.npy \
        --task_name pick_place_red_mug \
        --output_dir /path/to/output \
        --process_points
"""

import sys
sys.path.append("../../")

import yaml
import argparse
import pickle as pkl
from pathlib import Path

import torch
import cv2
import numpy as np
from scipy.ndimage import zoom

import json
import pandas as pd
from tqdm import tqdm

from point_utils.points_class import PointsClass
from utils import camera2pixelkey, pixel2d_to_3d_torch, triangulate_points, project_points
from gripper_points import extrapoints, Tshift
from pytorch3d.transforms import rotation_6d_to_matrix

# ── constants (mirror robot script) ──────────────────────────────────────────
# Finger open/closed half-widths in the gripper local Y axis (metres).
# Adjust these to match your robot's URDF / gripper_pcd implementation.
FINGER_OPEN_Y   = 0.05   # half-width when fully open
FINGER_CLOSED_Y = 0.00  # half-width when fully closed

# Gripper-state threshold: distance (m) below which fingers are considered closed.
GRIPPER_CLOSED_THRESH = 0.05

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="LeRobot data_dir -> point policy pkl pipeline",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--data_dir",       type=str, required=True, nargs="+",
                    help="One or more LeRobot dataset root dirs")
parser.add_argument("--calib_path",     type=str, required=True)
parser.add_argument("--task_name",      type=str, required=True)
parser.add_argument("--num_demos",      type=int, default=None)
parser.add_argument("--process_points", action="store_true")
parser.add_argument("--output_dir",     type=str, default="./processed_data")
parser.add_argument("--episode",        type=int, default=None,
                    help="Process a single episode index (default: all)")
args = parser.parse_args()

DATA_DIRS      = [Path(d) for d in args.data_dir]
CALIB_PATH     = Path(args.calib_path)
TASK_NAME      = args.task_name
NUM_DEMOS      = args.num_demos
process_points = args.process_points
OUTPUT_DIR     = Path(args.output_dir)

# ── LeRobot feature keys ──────────────────────────────────────────────────────
CAM_FRONT_COLOR = "observation.images.cam_azure_kinect_front.color"
CAM_LEFT_COLOR  = "observation.images.cam_azure_kinect_left.color"
STATE_KEY = "observation.right_eef_pose"
# layout: [rot6d_0..5, x, y, z, gripper]  — total 10 dims
# gripper: 0 = open, 1 = closed

CAM_COLOR_TO_PIXEL_KEY = {
    CAM_FRONT_COLOR: ("cam_1", "pixels1"),
    CAM_LEFT_COLOR:  ("cam_2", "pixels2"),
}
camera_indices = [1, 2]

# ── image settings ────────────────────────────────────────────────────────────
original_img_size = (1280, 720)
crop_h, crop_w    = (0.0, 1.0), (0.0, 1.0)
save_img_size     = (
    int(original_img_size[0] * (crop_w[1] - crop_w[0])),
    int(original_img_size[1] * (crop_h[1] - crop_h[0])),
)
save_image_size   = (256, 256)

object_labels = ["objects"]

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


# ── gripper helpers ───────────────────────────────────────────────────────────

def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d_tensor = torch.from_numpy(rot6d).float()
    mat = rotation_6d_to_matrix(rot6d_tensor)  # pytorch3d: rows convention
    return mat.numpy().swapaxes(-1, -2)        # transpose back to columns convention


def eef_pose_to_gripper_pcd(
    eef_states: np.ndarray,
    gripper_widths: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct the 4-point gripper PCD from observation.state.

    Parameters
    ----------
    eef_states     : (T, 9)  [x, y, z, rot6d_0..5 ]
    gripper_widths : (T,)    physical finger half-width (metres)

    Returns
    -------
    gripper_pcd : (T, 4, 3)
        [0] top    : EE origin offset -0.05 m along local Z
        [1] right  : +Y finger tip
        [2] left   : -Y finger tip
        [3] grasp  : EE origin (grasp centre)
    """
    T = len(eef_states)

    rot_mats = rot6d_to_matrix(eef_states[:, 3:9])  # (T, 3, 3)
    pos      = eef_states[:, :3]                  # (T, 3)

    gw    = gripper_widths[:, None]                # (T, 1)
    zeros = np.zeros((T, 1))
    neg_z = np.full((T, 1), -0.05)

    # offsets in local frame: (T, 4, 3)
    offsets = np.stack([
        np.concatenate([zeros, zeros, neg_z], axis=1),   # top
        np.concatenate([zeros,  gw,   zeros], axis=1),   # right finger
        np.concatenate([zeros, -gw,   zeros], axis=1),   # left  finger
        np.zeros((T, 3)),                                 # grasp centre
    ], axis=1)

    # rotate into world frame: offsets @ R^T  (row-vector convention)
    rot_t     = rot_mats.transpose(0, 2, 1)   # (T, 3, 3)
    pcd_world = offsets @ rot_t               # (T, 4, 3)
    pcd_world = pcd_world + pos[:, None, :]   # (T, 4, 3)

    return pcd_world


def gripper_pcd_to_robot_points(
    gripper_pcd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert (T, 4, 3) gripper PCD to 9 robot points + gripper states,
    using the same extrapoints / Tshift logic as the robot script.

    The 4 source points are:
        [0] top        [1] right finger   [2] left finger   [3] grasp centre

    Returns
    -------
    robot_points   : (T, 9, 3)
    gripper_states : (T,)   -1 = open, 1 = closed
    """
    T = len(gripper_pcd)
    robot_points_list   = []
    gripper_states_list = []

    for t in range(T):
        grasp_pos   = gripper_pcd[t, 3]          # (3,) — grasp centre
        right_pt    = gripper_pcd[t, 1]          # right finger tip
        left_pt     = gripper_pcd[t, 2]          # left  finger tip
        top_pt      = gripper_pcd[t, 0]          # top point

        # ── reconstruct orientation from the 4 points ─────────────────────
        # Z axis: from grasp centre to top point (local -Z is toward top)
        z_axis = top_pt - grasp_pos
        z_norm = np.linalg.norm(z_axis)
        if z_norm > 1e-6:
            z_axis = -z_axis / z_norm   # flip so local Z points away from top
        else:
            z_axis = np.array([0., 0., 1.])

        # Y axis: from grasp centre toward right finger
        y_axis = right_pt - grasp_pos
        y_norm = np.linalg.norm(y_axis)
        if y_norm > 1e-6:
            y_axis = y_axis / y_norm
        else:
            y_axis = np.array([0., 1., 0.])

        # X axis: orthogonal
        x_axis = np.cross(y_axis, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm > 1e-6:
            x_axis = x_axis / x_norm

        rot = np.stack([x_axis, y_axis, z_axis], axis=1)  # (3, 3) col vectors

        # ── gripper state ─────────────────────────────────────────────────
        finger_dist = np.linalg.norm(right_pt - left_pt)
        gripper_state = 1 if finger_dist < GRIPPER_CLOSED_THRESH else -1

        # ── build 4x4 transform T_g_b and apply Tshift ────────────────────
        T_g_b = np.eye(4)
        T_g_b[:3, :3] = rot
        T_g_b[:3, 3]  = grasp_pos
        T_g_b          = T_g_b @ Tshift

        # First point: Tshift-adjusted origin
        points3d = [T_g_b[:3, 3]]

        # Remaining 8 points via extrapoints (same as human script)
        for tp_idx, Tp in enumerate(extrapoints):
            Tp_use = Tp.copy()
            if gripper_state == 1 and tp_idx in [0, 1]:
                # Closed: pinch fingers inward
                Tp_use[1, 3] = 0.015 if tp_idx == 0 else -0.015
            pt = T_g_b @ Tp_use
            points3d.append(pt[:3, 3])

        robot_points_list.append(np.array(points3d))   # (9, 3)
        gripper_states_list.append(gripper_state)

    return np.array(robot_points_list), np.array(gripper_states_list)


# ── image helpers ─────────────────────────────────────────────────────────────

def resize_depth_image(depth_image: np.ndarray, new_size: tuple) -> np.ndarray:
    zoom_factors = (new_size[0] / depth_image.shape[0],
                    new_size[1] / depth_image.shape[1])
    return zoom(depth_image, zoom_factors, order=1)


# ── load parquet data (no LeRobotDataset needed) ─────────────────────────────
def load_parquet_for_dirs(data_dirs: list[Path]) -> pd.DataFrame:
    """Read all parquet chunks from one or more data_dirs and concat."""
    dfs = []
    for data_dir in data_dirs:
        parquet_files = sorted((data_dir / "data").glob("**/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {data_dir / 'data'}")
        dfs.append(pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True))
    df = pd.concat(dfs, ignore_index=True)
    # re-index episode_index globally across dirs
    return df


def load_meta(data_dir: Path) -> dict:
    meta_path = data_dir / "meta" / "info.json"
    with open(meta_path) as f:
        return json.load(f)


print("Loading parquet data...")
df = load_parquet_for_dirs(DATA_DIRS)

# Validate state key exists
if STATE_KEY not in df.columns:
    raise ValueError(f"Parquet missing column: {STATE_KEY}. Available: {list(df.columns)}")

print(f"  total frames  : {len(df)}")
print(f"  episodes      : {df['episode_index'].nunique()}")
print(f"  columns       : {[c for c in df.columns if not c.startswith('observation.image')]}")

# Build per-episode index (mirrors episode_data_index["from"/"to"])
episode_groups = df.groupby("episode_index", sort=True)
episode_indices = sorted(df["episode_index"].unique())

if args.episode is not None:
    episode_indices = [args.episode]
elif NUM_DEMOS is not None:
    episode_indices = episode_indices[:NUM_DEMOS]

print(f"Processing {len(episode_indices)} episode(s).")

# ── video helpers ─────────────────────────────────────────────────────────────
def load_video_frames_for_episode(data_dir: Path, cam_folder: str, ep_stem: str) -> np.ndarray | None:
    """Load all BGR frames from videos/chunk-000/<cam_folder>/<ep_stem>.mp4"""
    video_path = data_dir / "videos" / "chunk-000" / cam_folder / f"{ep_stem}.mp4"
    if not video_path.exists():
        # try without chunk subdir
        video_path = data_dir / "videos" / cam_folder / f"{ep_stem}.mp4"
    if not video_path.exists():
        return None
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


# Map cam feature key -> subfolder name (matches LeRobot video storage convention)
CAM_KEY_TO_FOLDER = {
    CAM_FRONT_COLOR: CAM_FRONT_COLOR,
    CAM_LEFT_COLOR:  CAM_LEFT_COLOR,
}
# e.g. "observation.images.cam_azure_kinect_front.color"
#   -> "cam_azure_kinect_front/color"

# ── main loop ─────────────────────────────────────────────────────────────────
observations  = []
max_cartesian = np.array([ 0.24113624,  0.02650134,  0.36749125, -3.1381004 , -0.0293628 ,  0.0104839 ], dtype=np.float32)
min_cartesian = np.array([ 0.2408465 ,  0.02610677,  0.36714548, -3.1390593 , -0.03289817,  0.00859457], dtype=np.float32)
max_gripper   = np.float32(-1.0)
min_gripper   = np.float32(-1.0)

for ep_idx in episode_indices:
    ep_df  = episode_groups.get_group(ep_idx).reset_index(drop=True)
    T      = len(ep_df)
    ep_stem = f"episode_{ep_idx:06d}"

    print(f"\n[ep {ep_idx}] {T} steps  ({ep_stem})")

    observation = {}
    skip = False

    # ── state: read directly from parquet ────────────────────────────────
    # observation.state column holds a list/array per row → stack to (T, 10)
    eef_poses      = []
    gripper_widths = []

    for _, row in tqdm(ep_df.iterrows(), total=T, desc=f"  State ep {ep_idx}"):

        state = np.asarray(row[STATE_KEY], dtype=np.float64)  # (10,)

        eef_poses.append(state[:9])   #  pos(3) + rot6d(6)

        # gripper: 0=open, 1=closed
        gp_norm = float(np.clip(state[9], 0.0, 1.0))
        gw = FINGER_OPEN_Y * (1 - gp_norm) + FINGER_CLOSED_Y * gp_norm
        gripper_widths.append(gw)

    eef_poses      = np.array(eef_poses)       # (T, 9)
    gripper_widths = np.array(gripper_widths)  # (T,)

    # ── RGB frames: decode from video files ──────────────────────────────
    for cam_key, (camera_name, pixel_key) in CAM_COLOR_TO_PIXEL_KEY.items():
        folder = CAM_KEY_TO_FOLDER[cam_key]
        frames = None
        for data_dir in DATA_DIRS:
            frames = load_video_frames_for_episode(data_dir, folder, ep_stem)
            if frames is not None:
                break
        if frames is None:
            print(f"  WARNING: video not found for {cam_key} ep {ep_idx}, skipping.")
            skip = True
            break
        # trim to parquet length in case video has extra frames
        frames = frames[:T]
        observation[pixel_key] = frames   # (T, H, W, 3) BGR
        print(f"  {pixel_key}: {observation[pixel_key].shape}")

    if skip:
        continue

    # ── build gripper PCD and robot points ────────────────────────────────
    g_pcd        = eef_pose_to_gripper_pcd(eef_poses, gripper_widths)  # (T,4,3)
    robot_points, gripper_states = gripper_pcd_to_robot_points(g_pcd)  # (T,9,3), (T,)
    observation["gripper_states"] = gripper_states

    print(f"  robot_points : {robot_points.shape}")
    print(f"  gripper open%: {(gripper_states == -1).mean()*100:.1f}%")

    if skip:
        continue

    # ── point tracking (object points only) ───────────────────────────────
    if process_points:
        mark_every = 8
        save = True

        for cam_idx in camera_indices:
            if not save:
                break
            camera_name = f"cam_{cam_idx}"
            pixel_key   = camera2pixelkey[camera_name]
            frames_bgr  = observation[pixel_key]
            frames_rgb  = [f[..., ::-1] for f in frames_bgr]   # BGR -> RGB

            points_class.add_to_image_list(frames_rgb[0], pixel_key)
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

            for f_idx, image in enumerate(frames_rgb[1:]):
                print(f"  Ep {ep_idx}, Frame {f_idx}, Cam {pixel_key}")
                points_class.add_to_image_list(image, pixel_key)

                if (f_idx + 1) % mark_every == 0 or f_idx == len(frames_rgb) - 2:
                    to_add = mark_every - (f_idx + 1) % mark_every
                    if to_add < mark_every:
                        for _ in range(to_add):
                            points_class.add_to_image_list(image, pixel_key)
                    else:
                        to_add = 0

                    points_class.track_points(pixel_key, last_n_frames=mark_every,
                                              one_frame=(mark_every == 1))
                    points = points_class.get_points_on_image(pixel_key, last_n_frames=mark_every)
                    for j in range(mark_every - to_add):
                        points_list.append(points[j])

            observation[f"object_tracks_{pixel_key}"] = torch.stack(points_list).numpy()  # (T, N, 2)
            points_class.reset_episode()

        if not save:
            continue

        # ── triangulate object 2D -> 3D ───────────────────────────────────
        for cam_idx in camera_indices:
            observation[f"object_tracks_3d_{camera2pixelkey[f'cam_{cam_idx}']}"] = []

        last_pixel_key = camera2pixelkey[f"cam_{camera_indices[-1]}"]
        n_track_frames = len(observation[f"object_tracks_{last_pixel_key}"])

        for t_idx in range(n_track_frames):
            P_list, pts_list = [], []
            for cam_idx in camera_indices:
                camera_name = f"cam_{cam_idx}"
                pixel_key   = camera2pixelkey[camera_name]
                extr = calibration_data[camera_name]["ext"]
                intr = calibration_data[camera_name]["int"]
                P_list.append(np.concatenate([intr, np.zeros((3, 1))], axis=1) @ extr)

                pt2d    = observation[f"object_tracks_{pixel_key}"][t_idx]
                point_h = pt2d[:, 1]
                h_orig_cropped = original_img_size[1] * (crop_h[1] - crop_h[0])
                point_h_orig   = ((point_h / save_img_size[1]) * h_orig_cropped
                                  + original_img_size[1] * crop_h[0]).astype(np.int32)

                point_w = pt2d[:, 0]
                w_orig_cropped = original_img_size[0] * (crop_w[1] - crop_w[0])
                point_w_orig   = ((point_w / save_img_size[0]) * w_orig_cropped
                                  + original_img_size[0] * crop_w[0]).astype(np.int32)

                pts_list.append(np.column_stack((point_w_orig, point_h_orig)))

            pts3d = triangulate_points(P_list, pts_list)
            for cam_idx in camera_indices:
                pixel_key = camera2pixelkey[f"cam_{cam_idx}"]
                observation[f"object_tracks_3d_{pixel_key}"].append(pts3d[:, :3])

        for cam_idx in camera_indices:
            pixel_key = camera2pixelkey[f"cam_{cam_idx}"]
            observation[f"object_tracks_3d_{pixel_key}"] = np.array(
                observation[f"object_tracks_3d_{pixel_key}"])

    # ── resize pixels and project robot + object tracks to 2D ─────────────
    for cam_idx in camera_indices:
        camera_name = f"cam_{cam_idx}"
        pixel_key   = camera2pixelkey[camera_name]

        # resize pixels
        pixels = [cv2.resize(p, save_image_size) for p in observation[pixel_key]]
        observation[pixel_key] = np.array(pixels)

        object_points_3d = observation.get(
            f"object_tracks_3d_{pixel_key}",
            np.zeros((len(robot_points), 1, 3)),
        )

        # store 3D tracks
        observation[f"robot_tracks_3d_{pixel_key}"]  = robot_points       # (T, 9, 3)
        observation[f"object_tracks_3d_{pixel_key}"] = object_points_3d

        # project to 2D
        robot_2d, object_2d = project_points(
            calibration_data, robot_points, object_points_3d, camera_name)
        observation[f"robot_tracks_{pixel_key}"]  = robot_2d
        observation[f"object_tracks_{pixel_key}"] = object_2d

    observations.append(observation)
    print(f"  Done. Total observations so far: {len(observations)}")

# ── save ──────────────────────────────────────────────────────────────────────
data = {
    "observations":  observations,
    "max_cartesian": max_cartesian,
    "min_cartesian": min_cartesian,
    "max_gripper":   max_gripper,
    "min_gripper":   min_gripper,
    "max_sensor":    None,
    "min_sensor":    None,
}

out_path = SAVE_DIR / f"{TASK_NAME}.pkl"
with open(out_path, "wb") as f:
    pkl.dump(data, f)

print(f"\nDone. Saved {len(observations)} episodes to {out_path}")