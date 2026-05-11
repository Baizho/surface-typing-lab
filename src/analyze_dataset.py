from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.features import extract_features


def load_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze typing session dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing session JSON files",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    files = sorted(data_dir.glob("sample_*.json"))

    if not files:
        print("No session files found.")
        return

    all_features = []

    for file in files:
        sample = load_json(file)
        features = extract_features(sample)
        all_features.append(features)

    mean_ikt = np.array([f["mean_ikt_ms"] for f in all_features], dtype=np.float64)
    std_ikt = np.array([f["std_ikt_ms"] for f in all_features], dtype=np.float64)
    durations = np.array([f["duration_ms"] for f in all_features], dtype=np.float64)
    alt_ratio = np.array(
        [f["alternating_hand_ratio"] for f in all_features],
        dtype=np.float64,
    )

    print("=== DATASET SUMMARY ===")
    print(f"Sessions: {len(files)}")
    print()

    print("Typing speed statistics:")
    print(f"Mean IKT mean: {mean_ikt.mean():.2f} ms")
    print(f"Mean IKT std:  {mean_ikt.std():.2f} ms")
    print()

    print("Session duration statistics:")
    print(f"Average duration: {durations.mean():.2f} ms")
    print(f"Min duration:     {durations.min():.2f} ms")
    print(f"Max duration:     {durations.max():.2f} ms")
    print()

    print("Hand alternation statistics:")
    print(f"Average alternation ratio: {alt_ratio.mean():.3f}")
    print()

    fastest_idx = int(np.argmin(mean_ikt))
    slowest_idx = int(np.argmax(mean_ikt))

    print("Fastest session:")
    print(f"- {files[fastest_idx].name}")
    print(f"- Mean IKT: {mean_ikt[fastest_idx]:.2f} ms")
    print()

    print("Slowest session:")
    print(f"- {files[slowest_idx].name}")
    print(f"- Mean IKT: {mean_ikt[slowest_idx]:.2f} ms")


if __name__ == "__main__":
    main()