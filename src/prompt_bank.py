from __future__ import annotations

from typing import Any

PROMPTS: list[dict[str, Any]] = [
    {"id": "p01", "category": "short", "text": "typing is fun"},
    {"id": "p02", "category": "short", "text": "hello world"},
    {"id": "p03", "category": "balanced", "text": "the quick brown fox"},
    {"id": "p04", "category": "balanced", "text": "smart watches can type"},
    {"id": "p05", "category": "balanced", "text": "surface typing is interesting"},
    {"id": "p06", "category": "balanced", "text": "research projects are fun"},
    {"id": "p07", "category": "balanced", "text": "keyboard rhythm matters"},
    {"id": "p08", "category": "punctuation", "text": "typing, sensing, and systems!"},
    {"id": "p09", "category": "punctuation", "text": "can smartwatches learn from taps?"},
    {"id": "p10", "category": "numbers", "text": "numbers 12345 change timing"},
    {"id": "p11", "category": "numbers", "text": "version 2 is better than version 1"},
    {"id": "p12", "category": "long", "text": "a small research toolkit should be useful and reproducible"},
]


def list_prompts(category: str | None = None) -> list[dict[str, Any]]:
    if category is None:
        return PROMPTS
    category = category.lower()
    return [p for p in PROMPTS if p["category"] == category]