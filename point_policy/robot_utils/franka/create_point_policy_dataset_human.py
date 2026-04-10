"""
Combined pipeline: LeRobot videos -> DIFT + CoTracker -> triangulation -> robot track conversion -> final pkl

Usage:
    python create_point_policy_dataset_human.py \
        --data_dir /home/haotian/.cache/huggingface/lerobot/Kovavavvavava/pick_place_red_mug_20260327_1 \
        --calib_path /home/haotian/Point-Policy/calib/calib.npy \
        --task_name pick_place_red_mug \
        --output_dir /home/haotian/Point-Policy/data/pick_place_red_mug_1 \
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
from utils import camera2pixelkey, pixel2d_to_3d_torch, triangulate_points, rigid_transform_3D
from gripper_points import extrapoints, Tshift

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Combined LeRobot -> point policy pkl pipeline")
parser.add_argument("--data_dir", type=str, required=True, nargs="+")
parser.add_argument("--calib_path",     type=str, required=True)
parser.add_argument("--task_name",      type=str, required=True)
parser.add_argument("--num_demos",      type=int, default=None)
parser.add_argument("--process_points", action="store_true")
parser.add_argument("--use_gt_depth",   action="store_true")
parser.add_argument("--output_dir",     type=str, default="./processed_data")
args = parser.parse_args()

DATA_DIRS      = [Path(d) for d in args.data_dir]
CALIB_PATH     = Path(args.calib_path)
TASK_NAME      = args.task_name
NUM_DEMOS      = args.num_demos
process_points = args.process_points
use_gt_depth   = args.use_gt_depth
OUTPUT_DIR     = Path(args.output_dir)

ROBOT_RESET_POINT = np.array([[ 3.59703113e-01,  2.47755790e-08,  4.78416715e-01],
       [ 3.59702964e-01, -3.99999497e-02,  3.18416708e-01],
       [ 3.59702985e-01,  4.00000503e-02,  3.18416721e-01],
       [ 3.59703044e-01,  3.75298611e-08,  3.98416715e-01],
       [ 3.59703030e-01, -4.99999625e-02,  3.98416707e-01],
       [ 3.59703057e-01,  5.00000375e-02,  3.98416723e-01],
       [ 3.59703078e-01,  3.11527200e-08,  4.38416715e-01],
       [ 3.59703065e-01, -4.99999688e-02,  4.38416707e-01],
       [ 3.59703092e-01,  5.00000312e-02,  4.38416723e-01]])

# ── LeRobot paths ─────────────────────────────────────────────────────────────
CAM_FRONT_COLOR = "observation.images.cam_azure_kinect_front.color"
CAM_LEFT_COLOR  = "observation.images.cam_azure_kinect_left.color"
CAM_FRONT_DEPTH = "observation.images.cam_azure_kinect_front.transformed_depth"
CAM_LEFT_DEPTH  = "observation.images.cam_azure_kinect_left.transformed_depth"

CAM_COLOR_TO_PIXEL_KEY = {
    CAM_FRONT_COLOR: ("cam_1", "pixels1"),
    CAM_LEFT_COLOR:  ("cam_2", "pixels2"),
}
CAM_DEPTH_TO_PIXEL_KEY = {
    CAM_FRONT_DEPTH: ("cam_1", "pixels1"),
    CAM_LEFT_DEPTH:  ("cam_2", "pixels2"),
}
camera_indices = [1, 2]

# ── image settings ────────────────────────────────────────────────────────────
original_img_size = (1280, 720)
crop_h, crop_w    = (0.0, 1.0), (0.0, 1.0)
save_img_size     = (
    int(original_img_size[0] * (crop_w[1] - crop_w[0])),
    int(original_img_size[1] * (crop_h[1] - crop_h[0])),
)
save_image_size   = (256, 256)  # final size after robot track conversion

object_labels     = ["human_hand", "objects"]

# ── robot track settings ──────────────────────────────────────────────────────
num_hand_points          = 9
index_finger_indices     = [3, 4]
thumb_indices            = [7, 8]
robot_base_orientation   = R.from_rotvec([np.pi, 0, 0]).as_matrix()
index_finger_thumb_pairs = [
    (idx1, idx2) for idx1 in index_finger_indices for idx2 in thumb_indices
]

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

def prepend_initial_frames(observation, robot_points_3d, num_frames=30):
    """Prepend N initial frames with interpolated waypoints from ROBOT_POINTS to first frame."""
    for cam_idx in camera_indices:
        camera_name = f"cam_{cam_idx}"
        pixel_key = camera2pixelkey[camera_name]
        
        # Prepend first image repeated num_frames times
        observation[pixel_key] = np.concatenate([
            np.repeat(observation[pixel_key][:1], num_frames, axis=0),
            observation[pixel_key]
        ], axis=0)
        
        if use_gt_depth:
            depth_key = f"depth_{pixel_key}"
            observation[depth_key] = np.concatenate([
                np.repeat(observation[depth_key][:1], num_frames, axis=0),
                observation[depth_key]
            ], axis=0)
        
        # Interpolate robot waypoints from robot_points_3d to first frame's robot position
        first_robot_pos = observation[f"robot_tracks_3d_{pixel_key}"][0]  # (N, 3)
        
        # Generate interpolated waypoints
        waypoints_3d = []
        for i in range(num_frames):
            alpha = i / num_frames  # 0, 1/num_frames, 2/num_frames, ...
            interpolated = (1 - alpha) * robot_points_3d + alpha * first_robot_pos
            waypoints_3d.append(interpolated)
        waypoints_3d = np.array(waypoints_3d)  # (num_frames, N, 3)
        
        observation[f"robot_tracks_3d_{pixel_key}"] = np.concatenate([
            waypoints_3d,
            observation[f"robot_tracks_3d_{pixel_key}"]
        ], axis=0)
        
        # Prepend object tracks (duplicate first frame num_frames times)
        observation[f"object_tracks_3d_{pixel_key}"] = np.concatenate([
            np.repeat(observation[f"object_tracks_3d_{pixel_key}"][:1], num_frames, axis=0),
            observation[f"object_tracks_3d_{pixel_key}"]
        ], axis=0)
        
        # Project interpolated waypoints to 2D
        P = calibration_data[camera_name]["ext"]
        K = calibration_data[camera_name]["int"]
        D = calibration_data[camera_name]["dist_coeff"]
        r, t = P[:3, :3], P[:3, 3]
        rvec, _ = cv2.Rodrigues(r)
        
        waypoints_2d = []
        for pts3d in waypoints_3d:
            pts2d = cv2.projectPoints(pts3d[:, :3], rvec, t, K, D)[0].squeeze()
            waypoints_2d.append(pts2d)
        waypoints_2d = np.array(waypoints_2d)  # (num_frames, N, 2)
        
        observation[f"robot_tracks_{pixel_key}"] = np.concatenate([
            waypoints_2d,
            observation[f"robot_tracks_{pixel_key}"]
        ], axis=0)
        
        # Prepend object tracks 2D
        observation[f"object_tracks_{pixel_key}"] = np.concatenate([
            np.repeat(observation[f"object_tracks_{pixel_key}"][:1], num_frames, axis=0),
            observation[f"object_tracks_{pixel_key}"]
        ], axis=0)
    
    # Prepend gripper state and human poses
    observation["gripper_states"] = np.concatenate([
        np.repeat(observation["gripper_states"][:1], num_frames, axis=0),
        observation["gripper_states"]
    ], axis=0)
    observation["human_poses"] = np.concatenate([
        np.repeat(observation["human_poses"][:1], num_frames, axis=0),
        observation["human_poses"]
    ], axis=0)
    
    return observation

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


def load_depth_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = frame[:, :, 0]
        frame = frame.astype(np.float32) / 1000.0
        frames.append(frame)
    cap.release()
    return np.array(frames) if frames else None


def resize_depth_image(depth_image, new_size):
    zoom_factors = (new_size[0] / depth_image.shape[0], new_size[1] / depth_image.shape[1])
    return zoom(depth_image, zoom_factors, order=1)


def convert_human_to_robot_tracks(observation):
    """Convert human hand tracks to robot gripper tracks for all cameras."""
    for cam_idx in camera_indices:
        camera_name = f"cam_{cam_idx}"
        pixel_key   = camera2pixelkey[camera_name]

        # resize pixels to final save size
        pixels = observation[pixel_key]
        pixels = [cv2.resize(p, save_image_size) for p in pixels]
        observation[pixel_key] = np.array(pixels)

        if use_gt_depth:
            depth = observation[f"depth_{pixel_key}"]
            depth = [resize_depth_image(d, save_image_size) for d in depth]
            observation[f"depth_{pixel_key}"] = np.array(depth)

        human_tracks_3d = observation[f"human_tracks_3d_{pixel_key}"]
        hand_points      = human_tracks_3d[:, :num_hand_points]
        object_points    = human_tracks_3d[:, num_hand_points:]

        robot_points, gripper_states, human_poses = [], [], []

        for idx, hand_point in enumerate(hand_points):
            dists = [
                np.linalg.norm(hand_point[i1] - hand_point[i2])
                for i1, i2 in index_finger_thumb_pairs
            ]
            min_dist     = np.min(dists)
            min_dist_idx = np.argmin(dists)
            i_idx, t_idx = index_finger_thumb_pairs[min_dist_idx]
            robot_pos    = (hand_point[i_idx] + hand_point[t_idx]) / 2

            if idx == 0:
                robot_ori        = robot_base_orientation
                base_hand_points = hand_point.copy()
            else:
                rot, _   = rigid_transform_3D(base_hand_points, hand_point.copy())
                robot_ori = rot @ robot_base_orientation

            human_poses.append(np.concatenate([robot_pos, R.from_matrix(robot_ori).as_rotvec()]))

            T_g_b = np.eye(4)
            T_g_b[:3, :3] = robot_ori
            T_g_b[:3, 3]  = robot_pos
            T_g_b          = T_g_b @ Tshift

            points3d     = [T_g_b[:3, 3]]
            gripper_state = -1
            for tp_idx, Tp in enumerate(extrapoints):
                if min_dist < 0.07 and tp_idx in [0, 1]:
                    Tp = Tp.copy()
                    Tp[1, 3]      = 0.015 if tp_idx == 0 else -0.015
                    gripper_state = 1
                pt = T_g_b @ Tp
                points3d.append(pt[:3, 3])

            robot_points.append(np.array(points3d))
            gripper_states.append(gripper_state)

        observation[f"robot_tracks_3d_{pixel_key}"] = np.array(robot_points)
        observation[f"object_tracks_3d_{pixel_key}"] = np.array(object_points)
        observation["gripper_states"] = np.array(gripper_states)
        observation["human_poses"]    = np.array(human_poses)

        # project 3D -> 2D
        P = calibration_data[camera_name]["ext"]
        K = calibration_data[camera_name]["int"]
        D = calibration_data[camera_name]["dist_coeff"]
        r, t = P[:3, :3], P[:3, 3]
        rvec, _ = cv2.Rodrigues(r)

        robot_points_2d = []
        for pts3d in robot_points:
            pts2d = cv2.projectPoints(pts3d[:, :3], rvec, t, K, D)[0].squeeze()
            robot_points_2d.append(pts2d)
        observation[f"robot_tracks_{pixel_key}"] = np.array(robot_points_2d)

        object_points_2d = []
        for pts3d in object_points:
            pts2d = cv2.projectPoints(pts3d[:, :3], rvec, t, K, D)[0].squeeze()
            object_points_2d.append(pts2d)
        observation[f"object_tracks_{pixel_key}"] = np.array(object_points_2d)

    return observation


# ── collect episodes ──────────────────────────────────────────────────────────
all_episode_files = []
for data_dir in DATA_DIRS:
    video_root = data_dir / "videos" / "chunk-000"
    episode_files = sorted((video_root / CAM_FRONT_COLOR).glob("episode_*.mp4"))
    all_episode_files.extend([(video_root, ep) for ep in episode_files])

if NUM_DEMOS is not None:
    all_episode_files = all_episode_files[:NUM_DEMOS]

print(f"Found {len(all_episode_files)} episodes across {len(DATA_DIRS)} data dirs.")

observations  = []
max_cartesian = np.array([ 0.24113624,  0.02650134,  0.36749125, -3.1381004 , -0.0293628 ,  0.0104839 ], dtype=np.float32)
min_cartesian = np.array([ 0.2408465 ,  0.02610677,  0.36714548, -3.1390593 , -0.03289817,  0.00859457], dtype=np.float32)
max_gripper   = np.float32(-1.0)
min_gripper   = np.float32(-1.0)

# ── main loop ─────────────────────────────────────────────────────────────────
for ep_idx, (video_root, ep_file) in enumerate(all_episode_files):

    ep_stem = ep_file.stem
    print(f"\n[{ep_idx+1}/{len(all_episode_files)}] Processing {ep_stem} from {video_root} ...")

    observation = {}
    skip = False

    # ── load RGB ──────────────────────────────────────────────────────────────
    for cam_folder, (camera_name, pixel_key) in CAM_COLOR_TO_PIXEL_KEY.items():
        video_file = video_root / cam_folder / f"{ep_stem}.mp4"
        frames = load_video_frames(video_file)
        if frames is None:
            print(f"  WARNING: cannot open {video_file}, skipping.")
            skip = True
            break
        observation[pixel_key] = frames
        print(f"  {pixel_key}: {frames.shape}")
    if skip:
        continue

    # ── load depth ────────────────────────────────────────────────────────────
    if use_gt_depth:
        for cam_folder, (camera_name, pixel_key) in CAM_DEPTH_TO_PIXEL_KEY.items():
            depth_file = video_root / cam_folder / f"{ep_stem}.mkv"
            depth = load_depth_frames(depth_file)
            if depth is None:
                print(f"  WARNING: cannot open {depth_file}, skipping.")
                skip = True
                break
            observation[f"depth_{pixel_key}"] = depth
        if skip:
            continue

    # ── point tracking ────────────────────────────────────────────────────────
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

            if use_gt_depth:
                points_3d_list = []
                depth = observation[f"depth_{pixel_key}"][0]
                points_class.set_depth(depth, pixel_key, original_img_size, save_img_size, (crop_h, crop_w))
                points_with_depth = points_class.get_points(pixel_key)
                depths   = points_with_depth[:, :, -1]
                P = calibration_data[camera_name]["ext"]
                K = calibration_data[camera_name]["int"]
                points_3d_list.append(pixel2d_to_3d_torch(points_list[0], depths[0], K, P))

            for f_idx, image in enumerate(frames[1:]):
                print(f"  Traj: {ep_idx}, Frame: {f_idx}, Cam: {pixel_key}")
                points_class.add_to_image_list(image, pixel_key)

                if use_gt_depth:
                    depth = observation[f"depth_{pixel_key}"][f_idx]
                    points_class.set_depth(depth, pixel_key, original_img_size, save_img_size, (crop_h, crop_w))

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

                    if use_gt_depth:
                        points_with_depth = points_class.get_points(pixel_key, last_n_frames=mark_every)
                        for j in range(mark_every - to_add):
                            d        = points_with_depth[j, :, -1]
                            points_3d_list.append(pixel2d_to_3d_torch(points[j], d, K, P))

            observation[f"human_tracks_{pixel_key}"] = torch.stack(points_list).numpy()
            if use_gt_depth:
                observation[f"human_tracks_3d_{pixel_key}"] = torch.stack(points_3d_list).numpy()
            points_class.reset_episode()

        if not save:
            continue

        # ── triangulate ───────────────────────────────────────────────────────
        if not use_gt_depth:
            for cam_idx in camera_indices:
                observation[f"human_tracks_3d_{camera2pixelkey[f'cam_{cam_idx}']}"] = []

            last_pixel_key = camera2pixelkey[f"cam_{camera_indices[-1]}"]
            for t_idx in range(len(observation[f"human_tracks_{last_pixel_key}"])):
                P_list, pts_list = [], []
                for cam_idx in camera_indices:
                    camera_name = f"cam_{cam_idx}"
                    pixel_key   = camera2pixelkey[camera_name]
                    extr = calibration_data[camera_name]["ext"]
                    intr = calibration_data[camera_name]["int"]
                    P_list.append(np.concatenate([intr, np.zeros((3, 1))], axis=1) @ extr)

                    pt2d    = observation[f"human_tracks_{pixel_key}"][t_idx]
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
                    observation[f"human_tracks_3d_{pixel_key}"].append(pts3d[:, :3])

            for cam_idx in camera_indices:
                pixel_key = camera2pixelkey[f"cam_{cam_idx}"]
                observation[f"human_tracks_3d_{pixel_key}"] = np.array(observation[f"human_tracks_3d_{pixel_key}"])

    # ── convert human -> robot tracks ─────────────────────────────────────────
    if process_points:
        observation = convert_human_to_robot_tracks(observation)
        observation = prepend_initial_frames(observation, ROBOT_RESET_POINT, num_frames=60)
        observations.append(observation)
    else:
        print(f"  Skipping {ep_stem} - process_points is False")
        continue

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