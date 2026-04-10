import pickle
import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PKL_PATH = "/home/haotian/Point-Policy/data/pick_place_red_mug_human_debug/processed_data_pkl/expert_demos/franka_env/pick_place_red_mug.pkl" 

with open(PKL_PATH, "rb") as f:
    traj = pickle.load(f)

o = traj["observations"][0]
n_frames = len(o["pixels1"])

def draw_3d(obj_3d, rob_3d):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # draw XYZ axes at origin
    ax.quiver(0, 0, 0, 0.05, 0, 0, color='r', arrow_length_ratio=0.3, label='X')
    ax.quiver(0, 0, 0, 0, 0.05, 0, color='g', arrow_length_ratio=0.3, label='Y')
    ax.quiver(0, 0, 0, 0, 0, 0.05, color='b', arrow_length_ratio=0.3, label='Z')
    ax.scatter([0], [0], [0], c='black', s=50, marker='o')

    # object points (red)
    ax.scatter(obj_3d[:, 0], obj_3d[:, 1], obj_3d[:, 2], c='red', s=50, label='object')
    for i, pt in enumerate(obj_3d):
        ax.text(pt[0], pt[1], pt[2], f"o{i}", color='red', fontsize=7)

    # robot points (blue)
    ax.scatter(rob_3d[:, 0], rob_3d[:, 1], rob_3d[:, 2], c='blue', s=50, label='robot')
    for i, pt in enumerate(rob_3d):
        ax.text(pt[0], pt[1], pt[2], f"r{i}", color='blue', fontsize=7)

    # fixed axis scale
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper left', fontsize=7)
    ax.view_init(elev=20, azim=45)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape(h, w, 4)[:, :, 1:]
    plt.close(fig)
    return buf

frames = []
for i in range(n_frames):
    obj_3d = o["object_tracks_3d_pixels1"][i]  # (7, 3)
    rob_3d = o["robot_tracks_3d_pixels1"][i]   # (9, 3)
    frame = draw_3d(obj_3d, rob_3d)
    frames.append(frame)

    if i % 10 == 0:
        print(f"Frame {i}/{n_frames}")

imageio.mimwrite("trajectory_3d.mp4", frames, fps=30, quality=8)
print("Saved trajectory_3d.mp4")