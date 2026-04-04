### This script is used to transfer the extrinsics in calib.npy from cam2base to base2cam, which is needed for the new dataset format.

import numpy as np

calib = np.load("/home/haotian/Point-Policy/calib/calib.npy", allow_pickle=True).item()

for cam_id in ["cam_1", "cam_2"]:
    T_cam2base = calib[cam_id]["ext"]
    calib[cam_id]["ext"] = np.linalg.inv(T_cam2base)  # base to cam

print(calib)
np.save("calib/calib.npy", calib)