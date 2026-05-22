"""
Vocabulary Ambiguity Analysis.

Computes L-R hand patterns for each word in the CEFR A1 vocabulary
using standard QWERTY touch-typing finger assignments. Groups words
by (pattern, length) to find which words are indistinguishable
using spatial information alone.
"""

import json
import os
from collections import defaultdict

# Standard QWERTY touch typing finger assignments
# Left hand: Q W E R T A S D F G Z X C V B
# Right hand: Y U I O P H J K L N M
LEFT_KEYS  = set("qwertasdfgzxcvb")
RIGHT_KEYS = set("yuiophjklnm")


def get_hand(char: str) -> str:
    c = char.lower()
    if c in LEFT_KEYS:
        return "L"
    elif c in RIGHT_KEYS:
        return "R"
    return "?"


def get_lr_pattern(word: str) -> str:
    """
    some examples would be:
        "place" -> "RRLLL"
        "would" -> "LRRRL"
        "home"  -> "RRRL"
    """
    return "".join(get_hand(c) for c in word.lower() if c.isalpha())


def analyze(words: list) -> dict:
    groups = defaultdict(list)
    skipped = []

    for word in words:
        pattern = get_lr_pattern(word)
        if "?" in pattern or not pattern:
            skipped.append(word)
            continue
        groups[(pattern, len(pattern))].append(word)

    ambiguous = {k: v for k, v in groups.items() if len(v) > 1}
    sorted_groups = sorted(ambiguous.items(), key=lambda x: len(x[1]), reverse=True)

    total = len(words) - len(skipped)
    in_ambiguous = sum(len(v) for v in ambiguous.values())

    return {
        "total_words":             total,
        "unique_patterns":         len(groups),
        "ambiguous_patterns":      len(ambiguous),
        "words_in_ambiguous":      in_ambiguous,
        "ambiguous_fraction":      in_ambiguous / total if total else 0,
        "sorted_groups":           sorted_groups,
        "skipped":                 skipped,
    }


def print_report(results: dict, top_n: int = 20):
    r = results
    print("=" * 65)
    print("  VOCABULARY AMBIGUITY ANALYSIS  —  L-R Hand Pattern")
    print("=" * 65)
    print(f"  Words analyzed          : {r['total_words']}")
    print(f"  Unique L-R patterns     : {r['unique_patterns']}")
    print(f"  Ambiguous patterns      : {r['ambiguous_patterns']}")
    print(f"  Words in ambiguous grps : {r['words_in_ambiguous']}")
    print(f"  Fraction of vocabulary  : {r['ambiguous_fraction']:.1%}")
    print()
    print(f"  Top {top_n} most ambiguous word groups:")
    print("-" * 65)

    for i, ((pattern, length), words) in enumerate(r["sorted_groups"][:top_n]):
        label = f"{i+1:2}. [{pattern}]  len={length}  ({len(words)} words)"
        print(f"  {label}")
        print(f"      {', '.join(sorted(words))}")

    print("=" * 65)

    # Group size distribution
    size_dist = defaultdict(int)
    for _, words in r["sorted_groups"]:
        size_dist[len(words)] += 1

    print("\n  Group size distribution (ambiguous groups only):")
    for size in sorted(size_dist):
        bar = "█" * size_dist[size]
        print(f"    {size} words sharing a pattern : "
              f"{size_dist[size]:3} groups  {bar}")
    print()


def save_report(results: dict, path: str = "reports/vocabulary_ambiguity.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {
        "total_words":         results["total_words"],
        "unique_patterns":     results["unique_patterns"],
        "ambiguous_patterns":  results["ambiguous_patterns"],
        "words_in_ambiguous":  results["words_in_ambiguous"],
        "ambiguous_fraction":  results["ambiguous_fraction"],
        "top_groups": [
            {
                "pattern": pattern,
                "length":  length,
                "size":    len(words),
                "words":   sorted(words),
            }
            for (pattern, length), words in results["sorted_groups"][:50]
        ],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {path}")


if __name__ == "__main__":
    from src.data.cefr_a1 import CEFR_A1_WORDS

    results = analyze(CEFR_A1_WORDS)
    print_report(results, top_n=20)
    save_report(results)