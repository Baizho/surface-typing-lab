from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from src.features import extract_features


RAW_DIR = Path("data/raw")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    sample_paths = sorted(RAW_DIR.glob("sample_*.json"))

    if not sample_paths:
        print("No samples found.")
        return

    rows = []

    for path in sample_paths:
        sample = json.loads(path.read_text(encoding="utf-8"))
        features = extract_features(sample)

        rows.append({
            "file": path.name,
            "prompt": sample.get("prompt", ""),
            "typed_text": sample.get("typed_text", ""),
            **features,
        })

    num_samples = len(rows)
    avg_num_keys = mean(row["num_keys"] for row in rows)
    avg_duration = mean(row["duration_ms"] for row in rows)
    avg_mean_ikt = mean(row["mean_ikt_ms"] for row in rows)
    avg_alt_ratio = mean(row["alternating_hand_ratio"] for row in rows)

    report_path = REPORT_DIR / "dataset_summary.md"

    lines = []
    lines.append("# Typing Dataset Summary")
    lines.append("")
    lines.append(f"- Number of samples: {num_samples}")
    lines.append(f"- Average number of keys: {avg_num_keys:.2f}")
    lines.append(f"- Average duration (ms): {avg_duration:.2f}")
    lines.append(f"- Average IKT (ms): {avg_mean_ikt:.2f}")
    lines.append(f"- Average alternating-hand ratio: {avg_alt_ratio:.2f}")
    lines.append("")
    lines.append("## Samples")
    lines.append("")

    for row in rows:
        lines.append(f"- `{row['file']}`")
        lines.append(f"  - prompt: `{row['prompt']}`")
        lines.append(f"  - typed: `{row['typed_text']}`")
        lines.append(f"  - duration: {row['duration_ms']:.2f} ms")
        lines.append(f"  - mean IKT: {row['mean_ikt_ms']:.2f} ms")
        lines.append(f"  - alternating-hand ratio: {row['alternating_hand_ratio']:.2f}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()