"""
Temporally subsample a Point Policy dataset pkl file.

Every per-frame array inside each episode (pixels, tracks, gripper_states, …)
is subsampled along the time axis.  Top-level normalization stats
(max/min_cartesian, max/min_gripper) are kept unchanged.

Usage:
    python subsample_dataset.py input.pkl output.pkl
    python subsample_dataset.py input.pkl output.pkl --ratio 3
"""

import argparse
import pickle as pkl
from pathlib import Path
import numpy as np


def subsample_episode(obs: dict, ratio: int) -> dict:
    out = {}
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            out[key] = val[::ratio]
        elif isinstance(val, list):
            out[key] = val[::ratio]
        else:
            out[key] = val
    return out


def main():
    parser = argparse.ArgumentParser(description="Temporally subsample a dataset pkl file.")
    parser.add_argument("input",         type=str,              help="Input pkl file")
    parser.add_argument("output",        type=str,              help="Output pkl file")
    parser.add_argument("--ratio", "-r", type=int, default=2,   help="Keep every N-th frame (default: 2)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "rb") as f:
        data = pkl.load(f)

    observations = data["observations"]
    n_eps = len(observations)

    first_key     = next(iter(observations[0]))
    frames_before = [len(ep[first_key]) for ep in observations]

    subsampled   = [subsample_episode(ep, args.ratio) for ep in observations]
    frames_after = [len(ep[first_key]) for ep in subsampled]

    data["observations"] = subsampled

    with open(output_path, "wb") as f:
        pkl.dump(data, f)

    print(f"Episodes : {n_eps}")
    print(f"Keys per episode : {list(observations[0].keys())}")
    print(f"Frames before : {sum(frames_before)}  (per-ep: {frames_before})")
    print(f"Frames after  : {sum(frames_after)}  (per-ep: {frames_after})")
    print(f"Ratio    : {args.ratio}")
    print(f"Saved to : {output_path}")


if __name__ == "__main__":
    main()
