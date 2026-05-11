from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from src.features import compute_ikt, hand_for_key  


RAW_DIR = Path("data/raw")
FIG_DIR = Path("data/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    sample_paths = sorted(RAW_DIR.glob("sample_*.json"))
    if not sample_paths:
        print("No samples found.")
        return

    latest = sample_paths[-1]
    sample = json.loads(latest.read_text(encoding="utf-8"))

    keys = sample["keys"]
    times = sample["key_times_ms"]
    ikt = compute_ikt(times)
    hands = [hand_for_key(k) for k in keys]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].plot(range(len(times)), times, marker="o")
    axes[0].set_title("Cumulative Keypress Time")
    axes[0].set_xlabel("Key index")
    axes[0].set_ylabel("Time (ms)")

    axes[1].bar(range(len(ikt)), ikt)
    axes[1].set_title("Inter-Keystroke Time (IKT)")
    axes[1].set_xlabel("IKT index")
    axes[1].set_ylabel("Δt (ms)")

    axes[2].step(range(len(hands)), [1 if h == "L" else 0 if h == "R" else 0.5 for h in hands], where="mid")
    axes[2].set_yticks([0, 0.5, 1])
    axes[2].set_yticklabels(["R", "?", "L"])
    axes[2].set_title("Approximate Hand Sequence")
    axes[2].set_xlabel("Key index")
    axes[2].set_ylabel("Hand")

    out_path = FIG_DIR / f"{latest.stem}.png"
    fig.suptitle(f"Typing Rhythm: {sample.get('typed_text', '')}")
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()  