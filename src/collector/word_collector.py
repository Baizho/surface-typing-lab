# src/collector/word_collector.py
"""
Word-level IKT data collector.

Collects repeated typing samples of individual words,
recording inter-keystroke timings (IKTs) for each attempt.

Usage:
    python -m src.collector.word_collector --session 1
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import termios
import tty

# ── target word list ──────────────────────────────────────────────────────────
# 66 words chosen to cover different L-R patterns and lengths.
# Includes all 7 spotlight words from the ambiguity analysis.
TARGET_WORDS = [
    # 4-letter words
    "home", "that", "this", "what", "time", "some", "good", "find",
    "then", "when", "come", "back", "word", "give", "does",

    # 5-letter words
    "would", "there", "thing", "could", "place", "right", "their",
    "these", "about", "which", "think", "first", "three", "water",
    "after", "where", "while", "world", "other", "night",

    # 6-letter words
    "should", "before", "always", "around", "mother", "second",
    "follow", "little", "really", "person", "people", "summer",
    "travel", "family", "system", "during", "answer", "friend",

    # 3-letter words
    "the", "and", "for", "are", "but", "not", "you", "all",
    "can", "had", "her", "was", "one",
]

REPS_PER_WORD   = 20   # repetitions to collect per word
MIN_ACCURACY    = 0.9  # reject attempts where < 90% of keys match


# ── helpers ───────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ikts_from_times(times_ms: list) -> list:
    """Convert absolute timestamps to inter-keystroke timings."""
    if len(times_ms) < 2:
        return []
    return [round(times_ms[i] - times_ms[i - 1], 2)
            for i in range(1, len(times_ms))]


def _attempt_matches_word(typed_keys: list, word: str) -> bool:
    """Check whether the typed keys match the target word."""
    typed = "".join(k for k in typed_keys if k not in ("\n", "\r", " "))
    return typed.lower() == word.lower()


# ── single-word recording ─────────────────────────────────────────────────────

def record_one_attempt(word: str) -> dict | None:
    """
    Wait for the user to type `word` and press Enter.
    Returns a dict with keys/times, or None on typo.
    """
    import termios, tty

    keys_pressed = []
    times_ms     = []
    start_ns     = None

    # Switch terminal to raw mode — we handle echo ourselves
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)

        while True:
            ch = sys.stdin.read(1)

            # Enter
            if ch in ("\r", "\n"):
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                break

            # Backspace
            if ch in ("\x7f", "\x08"):
                if keys_pressed:
                    keys_pressed.pop()
                    times_ms.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue

            # Ctrl+C — exit cleanly
            if ch == "\x03":
                sys.stdout.write("\r\n")
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                raise KeyboardInterrupt

            now_ns = time.perf_counter_ns()
            if start_ns is None:
                start_ns = now_ns

            elapsed_ms = (now_ns - start_ns) / 1_000_000
            keys_pressed.append(ch)
            times_ms.append(round(elapsed_ms, 3))

            # Echo the character ourselves
            sys.stdout.write(ch)
            sys.stdout.flush()

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    if not keys_pressed:
        return None

    if not _attempt_matches_word(keys_pressed, word):
        return None

    return {
        "keys":         keys_pressed,
        "key_times_ms": times_ms,
        "ikts":         _ikts_from_times(times_ms),
    }


# ── session collector ─────────────────────────────────────────────────────────

def collect_session(words: list, reps: int, output_dir: str, session_id: str):
    """Collect `reps` repetitions for each word in `words`."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    total_words = len(words)

    for w_idx, word in enumerate(words):
        repetitions = []
        attempt     = 0
        rejected    = 0

        print(f"\n{'─' * 55}")
        print(f"  Word {w_idx + 1}/{total_words} :  \"{word}\"")
        print(f"  Type it {reps} times, press Enter after each.")
        print(f"{'─' * 55}")

        while len(repetitions) < reps:
            remaining = reps - len(repetitions)
            print(f"  [{len(repetitions):2}/{reps}]  type \"{word}\" : ", end="", flush=True)

            result = record_one_attempt(word)

            if result is None:
                rejected += 1
                print("  ✗ typo — try again")
                continue

            repetitions.append(result)
            attempt += 1

            # Show IKT summary for feedback
            avg_ikt = (sum(result["ikts"]) / len(result["ikts"])
                       if result["ikts"] else 0)
            print(f"  ✓  avg IKT {avg_ikt:.0f} ms")

        results[word] = {
            "word":        word,
            "repetitions": repetitions,
            "rejected":    rejected,
        }
        print(f"  Done  ({rejected} rejected attempts)")

    # Save
    filename = os.path.join(output_dir, f"session_{session_id}.json")
    payload = {
        "session_id":    session_id,
        "created_at_utc": _now_utc(),
        "reps_per_word": reps,
        "words":         results,
    }
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  Session saved → {filename}")
    return filename


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Word-level IKT collector")
    parser.add_argument("--session",  default=None,
                        help="Session ID (auto-generated if omitted)")
    parser.add_argument("--words",    nargs="+", default=None,
                        help="Override word list (space-separated)")
    parser.add_argument("--reps",     type=int, default=REPS_PER_WORD,
                        help=f"Repetitions per word (default {REPS_PER_WORD})")
    parser.add_argument("--start-at", type=int, default=0,
                        help="Skip first N words (resume a session)")
    parser.add_argument("--output",   default="data/raw/word_sessions",
                        help="Output directory")
    args = parser.parse_args()

    session_id = args.session or str(uuid.uuid4())[:8]
    words      = args.words or TARGET_WORDS
    words      = words[args.start_at:]

    print(f"\n  Surface Typing Lab — Word IKT Collector")
    print(f"  Session : {session_id}")
    print(f"  Words   : {len(words)}  ×  {args.reps} reps = "
          f"{len(words) * args.reps} total keystrokes")
    print(f"  Output  : {args.output}/")
    print(f"\n  Press Enter after each word. Backspace to correct typos.")
    print(f"  Start whenever you're ready.\n")

    collect_session(words, args.reps, args.output, session_id)
    print("\n  Session complete. Run again tomorrow to build your dataset.")


if __name__ == "__main__":
    main()