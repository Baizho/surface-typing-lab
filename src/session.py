from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class TypingSession:
    prompt: str
    typed_text: str
    keys: List[str]
    key_times_ms: List[float]
    created_at_utc: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_keys(self) -> int:
        return len(self.keys)

    @property
    def duration_ms(self) -> float:
        if not self.key_times_ms:
            return 0.0
        return float(self.key_times_ms[-1])

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        # Keep backward compatibility with the earlier logger format
        data.setdefault("metadata", {})
        return data


def load_session(path: str | Path) -> TypingSession:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    return TypingSession(
        prompt=data.get("prompt", ""),
        typed_text=data.get("typed_text", ""),
        keys=list(data.get("keys", [])),
        key_times_ms=list(data.get("key_times_ms", [])),
        created_at_utc=data.get("created_at_utc", ""),
        metadata=dict(data.get("metadata", {})),
    )


def save_session(session: TypingSession, path: str | Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")


def validate_session(session: TypingSession) -> list[str]:
    issues: list[str] = []

    if len(session.keys) != len(session.key_times_ms):
        issues.append("keys and key_times_ms must have the same length")

    if not session.prompt:
        issues.append("prompt is empty")

    if not session.typed_text:
        issues.append("typed_text is empty")

    for i in range(1, len(session.key_times_ms)):
        if session.key_times_ms[i] < session.key_times_ms[i - 1]:
            issues.append("key_times_ms must be non-decreasing")
            break

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a typing session")
    parser.add_argument("file", type=str, help="Path to a sample JSON file")
    args = parser.parse_args()

    session = load_session(args.file)
    issues = validate_session(session)

    print(f"File: {args.file}")
    print(f"Prompt: {session.prompt}")
    print(f"Typed: {session.typed_text}")
    print(f"Keys: {session.num_keys}")
    print(f"Duration (ms): {session.duration_ms:.2f}")

    if issues:
        print("\nValidation issues:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nValidation: OK")


if __name__ == "__main__":
    main()