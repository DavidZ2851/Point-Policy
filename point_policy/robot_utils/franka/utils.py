import torch
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

Tshift = np.array([[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, -0.127], [0, 0, 0, 1]])

extrapoints = [
    # gripper points
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.04],
            [0.0, 0.0, 1.0, 0.16],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),  # 1
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.04],
            [0.0, 0.0, 1.0, 0.16],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),  # 2
    # First horizontal line
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.08],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),  # 3
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.05],
            [0.0, 0.0, 1.0, 0.08],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),  # 4
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.05],
            [0.0, 0.0, 1.0, 0.08],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),  # 5
    # Second horizontal line
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.04],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),  # 6
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.05],
            [0.0, 0.0, 1.0, 0.04],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),  # 7
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.05],
            [0.0, 0.0, 1.0, 0.04],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),  # 8
]

OFFSET = 0.033

camera2pixelkey = {
    "cam_1": "pixels1",
    "cam_2": "pixels2",
    "cam_51": "pixels51",
}
camera_indices = [1, 2]
pixelkey2camera = {v: k for k, v in camera2pixelkey.items()}

def ee_pose_to_robot_points(gripper_pcd, states_ee):
    """
    Args:
        gripper_pcd: (T, 4, 3) gripper point cloud
            [0] top
            [1] right finger tip
            [2] left finger tip
            [3] grasp center / EE point
    """
    robot_points   = []
    gripper_states = []

    for t in range(len(gripper_pcd)):
        pt1 = gripper_pcd[t, 1]  # right finger
        pt2 = gripper_pcd[t, 2]  # left finger
        dist = np.linalg.norm(pt1 - pt2)

        gripper_state = -1 if dist > 0.05 else 1  # -1=open, 1=closed #   0.09

        ee_pos = gripper_pcd[t, 3]  # grasp center (3,)
       
        ee_wxyz = states_ee[t, 3:7]  # (w, x, y, z)
        ee_rot = R.from_quat(ee_wxyz[[1, 2, 3, 0]])  # convert to (x, y, z, w) for scipy
        ee_rot_mat = ee_rot.as_matrix()  # (3, 3)
        ee_pos = ee_pos - ee_rot_mat[:, 2] * OFFSET
        # build T_ee from EE position
        # we don't have orientation from pcd alone, use identity rotation
        T_ee = np.eye(4)
        T_ee[:3, 3] = ee_pos
        T_ee[:3, :3] = ee_rot_mat
        T_ee = T_ee @ Tshift

        points3d = [T_ee[:3, 3]]

        for tp_idx, Tp in enumerate(extrapoints):
            if gripper_state == 1 and tp_idx in [0, 1]:
                Tp = Tp.copy()
                Tp[1, 3] = 0.015 if tp_idx == 0 else -0.015
            pt = T_ee @ Tp
            points3d.append(pt[:3, 3])

        robot_points.append(np.array(points3d))
        gripper_states.append(gripper_state)

    return np.array(robot_points), np.array(gripper_states)

def robot_points_to_ee_pose(robot_points):
    """
    Convert robot points back to EE pose (position + quaternion).
    
    Args:
        robot_points: (N, 3) - robot points where:
            [0] = T_ee @ Tshift origin (shifted EE position)
            [1:] = extrapoints transformed by T_ee @ Tshift
    
    Returns:
        ee_pos: (3,) - EE position in world frame
        ee_quat: (4,) - EE orientation as quaternion (x, y, z, w)
    """
    # Reconstruct T_ee @ Tshift from robot points
    # Point 0 is the origin of T_ee @ Tshift
    origin = robot_points[0]
    
    # Use extrapoints to recover orientation
    # extrapoints[0] and extrapoints[1] give us direction vectors
    # We need at least 3 non-collinear points to recover rotation
    
    # Build rotation from point differences
    # extrapoints are relative transforms, so robot_points[i+1] = T @ extrapoints[i][:3,3]
    
    # Get vectors in current frame
    v1 = robot_points[1] - origin  # direction to first extrapoint
    v2 = robot_points[2] - origin  # direction to second extrapoint
    
    # Get reference vectors from extrapoints
    ref_v1 = extrapoints[0][:3, 3]
    ref_v2 = extrapoints[1][:3, 3]
    
    # Solve for rotation using Kabsch/rigid_transform_3D
    src_pts = np.array([np.zeros(3), ref_v1, ref_v2])
    dst_pts = np.array([np.zeros(3), v1, v2])
    
    target_rot, _ = rigid_transform_3D(src_pts, dst_pts)
    
    # Build T_target = T_ee @ Tshift
    T_target = np.eye(4)
    T_target[:3, :3] = target_rot
    T_target[:3, 3] = origin
    
    # Undo Tshift: T_ee = T_target @ inv(Tshift)
    T_ee = T_target @ np.linalg.inv(Tshift)
    
    ee_pos = T_ee[:3, 3]
    ee_rot_mat = T_ee[:3, :3]
    
    # Add back the OFFSET
    ee_pos = ee_pos + ee_rot_mat[:, 2] * OFFSET
    
    # Convert to quaternion (x, y, z, w)
    ee_quat = R.from_matrix(ee_rot_mat).as_quat()
    ee_quat_wxyz = np.array([ee_quat[3], ee_quat[0], ee_quat[1], ee_quat[2]])
    
    return ee_pos, ee_quat_wxyz


def robot_points_to_ee_pose_with_gripper(robot_points, gripper_state):
    """
    Full conversion including gripper state.
    
    Args:
        robot_points: (N, 3)
        gripper_state: float (-1=open, 1=closed) -> 0=open, 1=closed
    
    Returns:
        robot_action: (8,) - [pos(3), quat(4), gripper(1)]
    """
    ee_pos, ee_quat = robot_points_to_ee_pose(robot_points)

    gripper = np.array([0]) if gripper_state < 0 else np.array([1])  # convert back to 0=open, 1=closed
    
    return np.concatenate([ee_pos, ee_quat, gripper])


def project_points(calibration_data, robot_points, object_points, camera_name):
    """Project 3D points to 2D pixels using calibration."""
    P = calibration_data[camera_name]["ext"]
    K = calibration_data[camera_name]["int"]
    D = calibration_data[camera_name]["dist_coeff"]
    r, t    = P[:3, :3].astype(np.float64), P[:3, 3].astype(np.float64)
    K       = K.astype(np.float64)
    D       = D.astype(np.float64)
    rvec, _ = cv2.Rodrigues(r)

    robot_2d = []
    for pts3d in robot_points:
        pts2d = cv2.projectPoints(pts3d[:, :3].astype(np.float64), rvec, t, K, D)[0].reshape(-1, 2)
        robot_2d.append(pts2d)

    object_2d = []
    for pts3d in object_points:
        pts2d = cv2.projectPoints(pts3d[:, :3].reshape(-1, 3).astype(np.float64), rvec, t, K, D)[0].reshape(-1, 2)
        object_2d.append(pts2d)

    return np.array(robot_2d), np.array(object_2d)


def pixel2d_to_3d_torch(points2d, depths, intrinsic_matrix, extrinsic_matrix):
    intrinsic_matrix = torch.tensor(intrinsic_matrix).float().to(depths.device)
    extrinsic_matrix = torch.tensor(extrinsic_matrix).float().to(depths.device)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    x = (points2d[:, 0] - cx) / fx
    y = (points2d[:, 1] - cy) / fy
    points3d = torch.stack((x * depths, y * depths, depths), dim=1)  # in camera frame
    points3d = torch.cat(
        (points3d, torch.ones((len(points2d), 1)).to(depths.device)), dim=1
    )
    points3d = (torch.linalg.inv(extrinsic_matrix) @ points3d.T).T  # world frame
    return points3d[..., :3]


def pixel2d_to_3d(points2d, depths, intrinsic_matrix, extrinsic_matrix):
    points2d = np.array(points2d)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    x = (points2d[:, 0] - cx) / fx
    y = (points2d[:, 1] - cy) / fy
    points_3d = np.column_stack((x * depths, y * depths, depths))  # in camera frame
    points_3d = np.concatenate([points_3d, np.ones((len(points2d), 1))], axis=1)
    points_3d = (np.linalg.inv(extrinsic_matrix) @ points_3d.T).T  # world frame
    return points_3d[..., :3]


def pixel3d_to_2d(points3d, intrinsic_matrix, camera_projection_matrix):
    points3d = np.array(points3d)
    points3d = np.concatenate([points3d, np.ones((len(points3d), 1))], axis=1)
    points3d = (camera_projection_matrix @ points3d.T).T  # camera frame
    depth = points3d[:, 2]
    points2d = (intrinsic_matrix @ points3d.T).T
    points2d = points2d / points2d[:, 2][:, None]
    return points2d[..., :2], depth


def triangulate_points(P, points):
    """
    Triangulate a batch of points from a variable number of camera views.

    Parameters:
    P: list of 3x4 projection matrices for each camera (currently world2camera transform)
    points: list of Nx2 arrays of normalized image coordinates for each camera

    Returns:
    Nx4 array of homogeneous 3D points
    """
    num_views = len(P)
    assert num_views > 1, "At least 2 cameras are required for triangulation"

    num_points = points[0].shape[0]
    A = np.zeros((num_points, num_views * 2, 4))

    for idx in range(num_views):
        # Set up the linear system for each point
        A[:, idx * 2] = points[idx][:, 0, np.newaxis] * P[idx][2] - P[idx][0]
        A[:, idx * 2 + 1] = points[idx][:, 1, np.newaxis] * P[idx][2] - P[idx][1]

    # Solve the system using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[:, -1, :]

    # Normalize the homogeneous coordinates
    X = X / X[:, 3:]

    return X


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_cols != 3:
        raise Exception(f"matrix A is not Nx3, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_cols != 3:
        raise Exception(f"matrix B is not Nx3, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am.T @ Bm

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A.T + centroid_B.T

    return R, t


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    Converts 6D rotation representation to rotation matrix
    using Gram-Schmidt orthogonalization.

    Args:
        d6: 6D rotation representation, of shape (..., 6)

    Returns:
        Batch of rotation matrices of shape (..., 3, 3)
    """
    a1, a2 = d6[..., :3], d6[..., 3:]

    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """
    Converts rotation matrices to 6D rotation representation
    by dropping the last row.

    Args:
        matrix: Batch of rotation matrices of shape (..., 3, 3)

    Returns:
        6D rotation representation, of shape (..., 6)
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))
