"""
IKT Visualization

Generates three figures:
  1. Rhythm consistency  — all reps of one word overlaid
  2. Ambiguous pair      — would vs thing (same L-R, different IKTs)
  3. Word summary        — mean IKT per word, colored by L-R pattern

Usage:
    PYTHONPATH=. python src/analysis/visualize_ikts.py \
        --session data/raw/word_sessions/session_day1.json
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from src.analysis.vocabulary_ambiguity import get_lr_pattern


# ── data loading ──────────────────────────────────────────────────────────────

def load_session(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_ikts(session: dict, word: str) -> list:
    """Return list of IKT sequences for a given word."""
    entry = session["words"].get(word)
    if entry is None:
        return []
    return [rep["ikts"] for rep in entry["repetitions"] if rep["ikts"]]


def get_mean_ikt(ikts_list: list) -> float:
    """Overall mean IKT across all repetitions of a word."""
    all_ikts = [v for seq in ikts_list for v in seq]
    return np.mean(all_ikts) if all_ikts else 0.0


# ── plot 1: rhythm consistency ────────────────────────────────────────────────

def plot_consistency(session: dict, word: str, output_dir: str):
    """
    Overlay all repetitions of `word` on one axes.
    Shows how tightly IKTs cluster across attempts.
    """
    ikts_list = get_ikts(session, word)
    print(ikts_list)
    if not ikts_list:
        print(f"  No data for '{word}'")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    n_positions = max(len(s) for s in ikts_list)
    positions   = list(range(1, n_positions + 1))

    print(n_positions)

    # Individual repetitions — thin, transparent
    for seq in ikts_list:
        if len(seq) == n_positions:
            ax.plot(positions, seq, color="steelblue", alpha=0.25, linewidth=1)

    # Mean line — bold
    mean_seq = [
        np.mean([s[i] for s in ikts_list if len(s) > i])
        for i in range(n_positions)
    ]
    ax.plot(positions, mean_seq, color="navy", linewidth=2.5,
            marker="o", markersize=6, label="mean IKT")

    pattern = get_lr_pattern(word)
    label_pairs = [f"{a}→{b}" for a, b in zip(word, word[1:])]

    print(positions)
    print(label_pairs)
    ax.set_xticks(positions)
    ax.set_xticklabels(label_pairs, fontsize=10)
    ax.set_ylabel("Inter-Keystroke Time (ms)")
    ax.set_title(f'IKT Consistency — "{word}"  [{pattern}]'
                 f'   ({len(ikts_list)} reps)', fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = os.path.join(output_dir, f"consistency_{word}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── plot 2: ambiguous pair ────────────────────────────────────────────────────

def plot_ambiguous_pair(session: dict, word_a: str, word_b: str,
                        output_dir: str):
    """
    Compare two words that share an L-R pattern.
    Shows IKT differences that allow disambiguation.
    """
    ikts_a = get_ikts(session, word_a)
    ikts_b = get_ikts(session, word_b)

    if not ikts_a or not ikts_b:
        print(f"  Missing data for pair '{word_a}' / '{word_b}'")
        return

    pat_a = get_lr_pattern(word_a)
    pat_b = get_lr_pattern(word_b)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, word, ikts_list, color in [
        (axes[0], word_a, ikts_a, "steelblue"),
        (axes[1], word_b, ikts_b, "tomato"),
    ]:
        n_pos     = max(len(s) for s in ikts_list)
        positions = list(range(1, n_pos + 1))
        labels    = [f"{a}→{b}" for a, b in zip(word, word[1:])]

        for seq in ikts_list:
            if len(seq) == n_pos:
                ax.plot(positions, seq, color=color, alpha=0.2, linewidth=1)

        mean_seq = [
            np.mean([s[i] for s in ikts_list if len(s) > i])
            for i in range(n_pos)
        ]
        ax.plot(positions, mean_seq, color=color, linewidth=2.5,
                marker="o", markersize=6)

        pattern = get_lr_pattern(word)
        ax.set_title(f'"{word}"  [{pattern}]', fontsize=12)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("Inter-Keystroke Time (ms)")

    same_pattern = pat_a == pat_b
    status = "SAME L-R pattern → IKT must disambiguate" if same_pattern \
             else "Different L-R patterns"
    fig.suptitle(f'Ambiguous Pair: "{word_a}" vs "{word_b}"  —  {status}',
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(output_dir, f"pair_{word_a}_vs_{word_b}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── plot 3: word summary ──────────────────────────────────────────────────────

def plot_word_summary(session: dict, output_dir: str):
    """
    Bar chart of mean IKT per word.
    Words sharing an L-R pattern share a color — showing that
    color-matched words rely entirely on IKT for disambiguation.
    """
    words_in_session = list(session["words"].keys())

    # Build (word, mean_ikt, pattern) tuples
    records = []
    for word in words_in_session:
        ikts_list = get_ikts(session, word)
        if not ikts_list:
            continue
        mean = get_mean_ikt(ikts_list)
        pattern = get_lr_pattern(word)
        records.append((word, mean, pattern))

    # Sort by mean IKT descending
    records.sort(key=lambda x: x[1], reverse=True)

    # Assign colors by L-R pattern
    patterns     = list(dict.fromkeys(r[2] for r in records))
    cmap         = cm.get_cmap("tab20", len(patterns))
    pattern_color = {p: cmap(i) for i, p in enumerate(patterns)}

    words  = [r[0] for r in records]
    means  = [r[1] for r in records]
    colors = [pattern_color[r[2]] for r in records]

    fig, ax = plt.subplots(figsize=(max(14, len(words) * 0.4), 5))
    bars = ax.bar(range(len(words)), means, color=colors, edgecolor="white",
                  linewidth=0.5)

    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Mean IKT (ms)")
    ax.set_title("Mean IKT per Word  —  same color = same L-R pattern "
                 "(spatially identical, IKT must disambiguate)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    # Legend for patterns
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=pattern_color[p], label=f"[{p}]")
        for p in patterns
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=7, ncol=3, title="L-R pattern")

    fig.tight_layout()
    out = os.path.join(output_dir, "word_summary.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True,
                        help="Path to session JSON file")
    parser.add_argument("--output", default="data/figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    session = load_session(args.session)

    words_available = list(session["words"].keys())
    print(f"\n  Loaded session: {args.session}")
    print(f"  Words available: {len(words_available)}")

    # Plot 1 — consistency for a few spotlight words
    print("\n  [1/3] Consistency plots...")
    for word in ["would", "place", "home", "there", "right"]:
        if word in words_available:
            plot_consistency(session, word, args.output)

    # Plot 2 — ambiguous pairs
    print("\n  [2/3] Ambiguous pair plots...")
    pairs = [
        ("would", "thing"),   # same [LRRRL] — the key example
        ("there", "three"),   # same [LRLLL]
        ("home",  "know"),    # same [RRRL]
    ]
    for a, b in pairs:
        if a in words_available and b in words_available:
            plot_ambiguous_pair(session, a, b, args.output)

    # Plot 3 — full word summary
    print("\n  [3/3] Word summary...")
    plot_word_summary(session, args.output)

    print(f"\n  All figures saved to {args.output}/")


if __name__ == "__main__":
    main()