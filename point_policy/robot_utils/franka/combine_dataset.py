#!/usr/bin/env python3
"""
Combine multiple pkl dataset files into one with control over number of demos per dataset.

Usage:
    python combine_datasets.py --input data1.pkl:10 data2.pkl:20 data3.pkl --output-dir /path/to/output
    python combine_datasets.py --input data1.pkl:10 data2.pkl --output-dir /path/to/output --name my_dataset
"""

import argparse
import pickle as pkl
from pathlib import Path
import numpy as np
import random


def load_dataset(path: Path) -> dict:
    """Load a single pkl dataset."""
    with open(path, "rb") as f:
        return pkl.load(f)


def parse_input_spec(spec: str) -> tuple[Path, int | None]:
    """
    Parse input specification like 'path.pkl:10' or 'path.pkl'.
    Returns (path, num_demos) where num_demos is None for 'all'.
    """
    if ":" in spec:
        path_str, count_str = spec.rsplit(":", 1)
        if count_str.lower() == "all":
            return Path(path_str), None
        else:
            return Path(path_str), int(count_str)
    else:
        return Path(spec), None  # None means all


def combine_datasets(datasets: list[tuple[dict, int | None]], seed: int = 42, no_pixels: bool = False) -> dict:
    """Combine multiple datasets into one with random sampling."""
    
    PIXEL_KEYS = {'pixels1', 'pixels2'}

    random.seed(seed)
    np.random.seed(seed)
    
    # Combine observations (list of episodes)
    all_observations = []
    for data, num_demos in datasets:
        obs = data["observations"]
        if num_demos is not None and num_demos < len(obs):
            # Random sample instead of taking first N
            indices = random.sample(range(len(obs)), num_demos)
            obs = [obs[i] for i in sorted(indices)]

        if no_pixels:
            obs = [{k: v for k, v in episode.items() if k not in PIXEL_KEYS} for episode in obs]

        all_observations.extend(obs)
    
    # Combine min/max statistics by taking global min/max
    max_cartesians = [d["max_cartesian"] for d, _ in datasets if d.get("max_cartesian") is not None]
    min_cartesians = [d["min_cartesian"] for d, _ in datasets if d.get("min_cartesian") is not None]
    max_grippers = [d["max_gripper"] for d, _ in datasets if d.get("max_gripper") is not None]
    min_grippers = [d["min_gripper"] for d, _ in datasets if d.get("min_gripper") is not None]
    
    combined = {
        "observations": all_observations,
        "max_cartesian": np.max(np.stack(max_cartesians), axis=0).astype(np.float32) if max_cartesians else None,
        "min_cartesian": np.min(np.stack(min_cartesians), axis=0).astype(np.float32) if min_cartesians else None,
        "max_gripper": np.float32(max(max_grippers)) if max_grippers else None,
        "min_gripper": np.float32(min(min_grippers)) if min_grippers else None,
        "max_sensor": None,
        "min_sensor": None,
    }
    
    return combined


def print_dataset_info(data: dict, name: str = "Dataset"):
    print(f"\n{name}:")
    print(f"  Episodes: {len(data['observations'])}")
    if len(data['observations']) > 0:
        first_ep = data['observations'][0]
        print(f"  Episode keys: {list(first_ep.keys())}")
        if 'gripper_states' in first_ep:
            lengths = [obs['gripper_states'].shape[0] for obs in data['observations'] if 'gripper_states' in obs]
            if lengths:
                print(f"  Episode lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    print(f"  max_cartesian: {data['max_cartesian']}")
    print(f"  min_cartesian: {data['min_cartesian']}")
    print(f"  max_gripper: {data['max_gripper']}")
    print(f"  min_gripper: {data['min_gripper']}")

def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple pkl datasets into one",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Take 10 random demos from data1.pkl and all demos from data2.pkl
  python combine_datasets.py --input data1.pkl:10 data2.pkl --output-dir /path/to/output

  # Take 5 random demos from each with custom name
  python combine_datasets.py --input data1.pkl:5 data2.pkl:5 --output-dir /path/to/output --name combined

  # Explicitly specify 'all'
  python combine_datasets.py --input data1.pkl:10 data2.pkl:all --output-dir /path/to/output

  # With custom seed for reproducibility
  python combine_datasets.py --input data1.pkl:10 data2.pkl:20 --output-dir /path/to/output --seed 123
        """
    )
    parser.add_argument("--input", "-i", nargs="+", type=str, required=True,
                        help="Input pkl files with optional count (e.g., data.pkl:10)")
    parser.add_argument("--output-dir", "-o", type=str, required=True, 
                        help="Output directory (will create processed_data_pkl/expert_demos/franka_env/)")
    parser.add_argument("--name", "-n", type=str, default=None,
                        help="Output filename (without .pkl). Defaults to first input file's name.")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no-pixels", action="store_true",
                        help="If set, will remove pixel data from combined dataset to save space.")
    args = parser.parse_args()
    
    # Set seed early
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Parse input specifications
    input_specs = [parse_input_spec(spec) for spec in args.input]
    
    # Determine output filename
    if args.name:
        output_name = args.name
    else:
        # Use first input file's name
        output_name = input_specs[0][0].stem
    
    print(f"Combining {len(input_specs)} datasets (seed={args.seed}):")
    
    # Load all datasets
    datasets = []
    total_episodes = 0
    for path, num_demos in input_specs:
        data = load_dataset(path)
        available = len(data["observations"])
        using = min(num_demos, available) if num_demos is not None else available
        print(f"  - {path.name}: using {using}/{available} episodes (random sample)")
        total_episodes += using
        datasets.append((data, num_demos))
    
    # Combine
    print(f"\nCombining {total_episodes} total episodes...")
    combined = combine_datasets(datasets, seed=args.seed, no_pixels=args.no_pixels)
    
    
    
    # Create output directory structure
    output_dir = Path(args.output_dir) / "processed_data_pkl" / "expert_demos" / "franka_env"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save
    output_path = output_dir / f"{output_name}.pkl"
    with open(output_path, "wb") as f:
        pkl.dump(combined, f)
    
    print(f"\nSaved to {output_path}")
    # Print combined info
    print_dataset_info(combined, "Combined Dataset")


if __name__ == "__main__":
    main()