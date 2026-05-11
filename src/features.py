from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev


LEFT_HAND_KEYS = set("qwertasdfgzxcvb")
RIGHT_HAND_KEYS = set("yuiophjklnm")


def hand_for_key(key: str) -> str:
    key = key.lower()

    if key in LEFT_HAND_KEYS:
        return "L"

    if key in RIGHT_HAND_KEYS:
        return "R"

    return "?"


def compute_ikt(key_times_ms: list[float]) -> list[float]:
    ikt = []

    for i in range(1, len(key_times_ms)):
        ikt.append(key_times_ms[i] - key_times_ms[i - 1])

    return ikt


def alternating_hand_ratio(hands: list[str]) -> float:
    if len(hands) < 2:
        return 0.0

    alternating = 0

    for i in range(1, len(hands)):
        if hands[i] != hands[i - 1]:
            alternating += 1

    return alternating / (len(hands) - 1)


def extract_features(sample: dict) -> dict:
    keys = sample["keys"]
    key_times_ms = sample["key_times_ms"]

    ikt = compute_ikt(key_times_ms)

    hands = [hand_for_key(k) for k in keys if k.strip()]

    features = {
        "num_keys": len(keys),
        "duration_ms": key_times_ms[-1] if key_times_ms else 0.0,
        "mean_ikt_ms": mean(ikt) if ikt else 0.0,
        "std_ikt_ms": pstdev(ikt) if len(ikt) > 1 else 0.0,
        "alternating_hand_ratio": alternating_hand_ratio(hands),
        "left_hand_presses": hands.count("L"),
        "right_hand_presses": hands.count("R"),
    }

    return features


def main() -> None:
    sample_paths = sorted(Path("data/raw").glob("sample_*.json"))

    if not sample_paths:
        print("No samples found.")
        return

    latest = sample_paths[-1]

    sample = json.loads(latest.read_text())

    features = extract_features(sample)

    print("Loaded:", latest)
    print()

    for key, value in features.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()