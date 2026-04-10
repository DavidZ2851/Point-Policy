import numpy as np
import json

# Load cam_calibration_aligned.json
with open("camera_calibration.json", "r") as f:
    data = json.load(f)

# Mapping: json cam_id -> calib.npy cam_name
cam_id_mapping = {
    "cam0": "cam_1",
    "cam1": "cam_2"
}

calib = {}

for json_cam_id, npy_cam_name in cam_id_mapping.items():
    intrinsics = np.array(data[json_cam_id]["intrinsic"])
    T_cam2base = np.array(data[json_cam_id]["extrinsic"])
    dist_coeff = np.array(data[json_cam_id]["distortion"])
    
    # Convert cam2base -> base2cam (invert)
    T_base2cam = np.linalg.inv(T_cam2base)
    
    calib[npy_cam_name] = {
        "int": intrinsics,
        "dist_coeff": dist_coeff,
        "ext": T_base2cam
    }
    
    print(f"\n{json_cam_id} -> {npy_cam_name}")
    print(f"  Intrinsics:\n{intrinsics}")
    print(f"  Extrinsics (base2cam):\n{T_base2cam}")
    print(f"  Distortion: {dist_coeff}")

# Save
output_path = "calib/calib.npy"
np.save(output_path, calib)
print(f"\nSaved to {output_path}")

# Verify
loaded = np.load(output_path, allow_pickle=True).item()
print("\nVerification:")
for cam_name in loaded:
    print(f"  {cam_name}: ext shape={loaded[cam_name]['ext'].shape}, int shape={loaded[cam_name]['int'].shape}")