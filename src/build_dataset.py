from __future__ import annotations

import csv
import json
from pathlib import Path

from features import extract_features


RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/features.csv")


def main() -> None:
    sample_paths = sorted(RAW_DIR.glob("sample_*.json"))

    if not sample_paths:
        print("No raw samples found.")
        return

    rows = []

    for path in sample_paths:
        sample = json.loads(path.read_text(encoding="utf-8"))
        features = extract_features(sample)

        row = {
            "file": path.name,
            "prompt": sample.get("prompt", ""),
            "typed_text": sample.get("typed_text", ""),
            **features,
        }
        rows.append(row)

    fieldnames = list(rows[0].keys())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} samples to {OUT_PATH}")


if __name__ == "__main__":
    main()