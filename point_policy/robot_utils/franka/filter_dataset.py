#!/usr/bin/env python3
"""
Select specific episodes from a pkl dataset by index.

Usage:
    python select_episodes.py --input data.pkl --episodes 0 2 5 10 --output-dir /path/to/output
    python select_episodes.py --input data.pkl --episodes 0-10 15 20-25 --output-dir /path/to/output
    python select_episodes.py --input data.pkl --episodes 0-10 --name selected_demos --output-dir /path/to/output
"""

import argparse
import pickle as pkl
from pathlib import Path
import numpy as np


def load_dataset(path: Path) -> dict:
    """Load a single pkl dataset."""
    with open(path, "rb") as f:
        return pkl.load(f)


def parse_episode_spec(specs: list[str], max_episodes: int) -> list[int]:
    """
    Parse episode specifications like '0', '2-5', '10'.
    Returns sorted list of unique episode indices.
    """
    indices = set()
    for spec in specs:
        if "-" in spec:
            start, end = spec.split("-", 1)
            start = int(start)
            end = int(end)
            indices.update(range(start, end + 1))
        else:
            indices.add(int(spec))
    
    # Filter out invalid indices
    valid_indices = sorted([i for i in indices if 0 <= i < max_episodes])
    invalid_indices = sorted([i for i in indices if i < 0 or i >= max_episodes])
    
    if invalid_indices:
        print(f"Warning: Ignoring invalid indices: {invalid_indices} (max index: {max_episodes - 1})")
    
    return valid_indices


def select_episodes(data: dict, indices: list[int]) -> dict:
    """Select specific episodes from dataset."""
    
    selected_observations = [data["observations"][i] for i in indices]
    
    return {
        "observations": selected_observations,
        "max_cartesian": data.get("max_cartesian"),
        "min_cartesian": data.get("min_cartesian"),
        "max_gripper": data.get("max_gripper"),
        "min_gripper": data.get("min_gripper"),
        "max_sensor": data.get("max_sensor"),
        "min_sensor": data.get("min_sensor"),
    }


def print_dataset_info(data: dict, name: str = "Dataset"):
    """Print dataset structure info."""
    print(f"\n{name}:")
    print(f"  Episodes: {len(data['observations'])}")
    
    if len(data['observations']) > 0:
        lengths = [obs['pixels1'].shape[0] for obs in data['observations']]
        print(f"  Episode lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Select specific episodes from a pkl dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Select specific episodes by index
  python select_episodes.py --input data.pkl --episodes 0 2 5 10 --output-dir /path/to/output

  # Select ranges of episodes
  python select_episodes.py --input data.pkl --episodes 0-10 15 20-25 --output-dir /path/to/output

  # Select with custom output name
  python select_episodes.py --input data.pkl --episodes 0-5 --name best_demos --output-dir /path/to/output
        """
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input pkl file")
    parser.add_argument("--episodes", "-e", nargs="+", type=str, required=True,
                        help="Episode indices to select (e.g., 0 2 5 or 0-10 15)")
    parser.add_argument("--output-dir", "-o", type=str, required=True,
                        help="Output directory (will create processed_data_pkl/expert_demos/franka_env/)")
    parser.add_argument("--name", "-n", type=str, default=None,
                        help="Output filename (without .pkl). Defaults to input file's name.")
    args = parser.parse_args()
    
    # Load dataset
    input_path = Path(args.input)
    print(f"Loading {input_path}...")
    data = load_dataset(input_path)
    
    num_episodes = len(data["observations"])
    print(f"  Found {num_episodes} episodes")
    
    # Parse episode indices
    indices = parse_episode_spec(args.episodes, num_episodes)
    print(f"\nSelecting {len(indices)} episodes: {indices}")
    
    # Select episodes
    selected = select_episodes(data, indices)
    
    # Print info
    print_dataset_info(selected, "Selected Dataset")
    
    # Create output directory structure
    output_dir = Path(args.output_dir) / "processed_data_pkl" / "expert_demos" / "franka_env"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename
    output_name = args.name if args.name else input_path.stem
    output_path = output_dir / f"{output_name}.pkl"
    
    # Save
    with open(output_path, "wb") as f:
        pkl.dump(selected, f)
    
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()