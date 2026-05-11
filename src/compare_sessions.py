from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.features import extract_features
from src.session import load_session


FEATURE_KEYS = [
    "num_keys",
    "duration_ms",
    "mean_ikt_ms",
    "std_ikt_ms",
    "alternating_hand_ratio",
    "left_hand_presses",
    "right_hand_presses",
]


def session_to_vector(sample_path: Path) -> np.ndarray:
    session = load_session(sample_path)
    features = extract_features(session.to_dict())

    return np.array([float(features[key]) for key in FEATURE_KEYS], dtype=np.float64)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare typing sessions by feature similarity")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Path to the query session JSON file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing session JSON files",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest sessions to show",
    )
    args = parser.parse_args()

    query_path = Path(args.query)
    data_dir = Path(args.data_dir)

    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")

    sample_paths = sorted(data_dir.glob("sample_*.json"))
    if not sample_paths:
        print("No session files found.")
        return

    query_vec = session_to_vector(query_path)

    results = []
    for path in sample_paths:
        if path.resolve() == query_path.resolve():
            continue

        vec = session_to_vector(path)
        sim = cosine_similarity(query_vec, vec)
        results.append((path.name, sim))

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"Query: {query_path.name}")
    print(f"Feature keys: {FEATURE_KEYS}")
    print()

    for rank, (name, sim) in enumerate(results[: args.top_k], start=1):
        print(f"{rank}. {name}  | cosine similarity = {sim:.4f}")


if __name__ == "__main__":
    main()