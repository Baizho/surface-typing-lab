#!/usr/bin/env python3
from __future__ import annotations

import curses
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List


OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TypingSample:
    prompt: str
    typed_text: str
    keys: List[str]
    key_times_ms: List[float]
    created_at_utc: str


def save_sample(sample: TypingSample) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"sample_{timestamp}.json"
    path.write_text(json.dumps(asdict(sample), indent=2), encoding="utf-8")
    return path


def run_logger(stdscr: curses.window) -> TypingSample:
    curses.curs_set(1)
    stdscr.clear()
    stdscr.nodelay(False)
    stdscr.keypad(True)

    prompt = "hello world"
    typed_chars: List[str] = []
    keys: List[str] = []
    key_times_ms: List[float] = []

    stdscr.addstr(0, 0, "Surface Typing Logger")
    stdscr.addstr(2, 0, f"Type this sentence and press Enter when done:")
    stdscr.addstr(4, 0, prompt)
    stdscr.addstr(6, 0, "Typed: ")
    stdscr.refresh()

    start_time_ns = None

    while True:
        ch = stdscr.get_wch()

        if start_time_ns is None:
            start_time_ns = datetime.now(timezone.utc).timestamp() * 1_000_000_000

        now_ns = datetime.now(timezone.utc).timestamp() * 1_000_000_000
        elapsed_ms = (now_ns - start_time_ns) / 1_000_000.0

        if ch == "\n":
            break
        elif ch == "\x7f" or ch == "\b":
            if typed_chars:
                typed_chars.pop()
                stdscr.addstr(6, 0, "Typed: " + "".join(typed_chars) + " ")
                stdscr.clrtoeol()
                stdscr.refresh()
            continue
        elif isinstance(ch, str) and len(ch) == 1:
            typed_chars.append(ch)
            keys.append(ch)
            key_times_ms.append(elapsed_ms)

            stdscr.addstr(6, 0, "Typed: " + "".join(typed_chars))
            stdscr.clrtoeol()
            stdscr.refresh()

    typed_text = "".join(typed_chars)

    return TypingSample(
        prompt=prompt,
        typed_text=typed_text,
        keys=keys,
        key_times_ms=key_times_ms,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def main() -> None:
    sample = curses.wrapper(run_logger)
    path = save_sample(sample)
    print(f"Saved sample to: {path}")
    print(f"Typed text: {sample.typed_text}")
    print(f"Keystrokes recorded: {len(sample.keys)}")


if __name__ == "__main__":
    main()