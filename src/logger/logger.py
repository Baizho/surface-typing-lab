#!/usr/bin/env python3
from __future__ import annotations

import argparse
import curses
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.session import TypingSession, save_session, validate_session


OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_logger(stdscr: curses.window, prompt: str) -> TypingSession:
    curses.curs_set(1)
    stdscr.clear()
    stdscr.nodelay(False)
    stdscr.keypad(True)

    typed_chars: List[str] = []
    keys: List[str] = []
    key_times_ms: List[float] = []

    stdscr.addstr(0, 0, "Surface Typing Logger")
    stdscr.addstr(2, 0, "Type the sentence below and press Enter when done:")
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
                stdscr.addstr(6, 0, "Typed: " + "".join(typed_chars))
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

    return TypingSession(
        prompt=prompt,
        typed_text=typed_text,
        keys=keys,
        key_times_ms=key_times_ms,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        metadata={
            "source": "keyboard_logger",
            "logger_version": "1.0",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Typing timestamp logger")
    parser.add_argument(
        "--prompt",
        type=str,
        default="hello world",
        help="Sentence to type",
    )
    args = parser.parse_args()

    session = curses.wrapper(lambda stdscr: run_logger(stdscr, args.prompt))

    issues = validate_session(session)
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"- {issue}")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(ch for ch in session.prompt.lower() if ch.isalnum() or ch in ("_", "-"))[:20]
    suffix = f"_{safe_prompt}" if safe_prompt else ""
    path = OUTPUT_DIR / f"sample_{timestamp}{suffix}.json"

    save_session(session, path)

    print(f"Saved sample to: {path}")
    print(f"Typed text: {session.typed_text}")
    print(f"Keystrokes recorded: {session.num_keys}")


if __name__ == "__main__":
    main()