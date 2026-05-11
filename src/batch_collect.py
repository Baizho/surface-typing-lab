from __future__ import annotations

import argparse
import subprocess
import sys

from src.prompt_bank import list_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a batch of typing samples")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional prompt category to collect (short, balanced, punctuation, numbers, long)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of prompts collected",
    )
    args = parser.parse_args()

    prompts = list_prompts(args.category)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    if not prompts:
        print("No prompts matched the requested category.")
        return

    print(f"Collecting {len(prompts)} prompt(s).")
    print("Each sample will open the typing logger.")
    print("Press Enter in the logger when you finish each sentence.")
    print()

    for idx, prompt in enumerate(prompts, start=1):
        print(f"[{idx}/{len(prompts)}] {prompt['id']} ({prompt['category']}): {prompt['text']}")
        input("Press Enter to start this prompt...")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.logger.logger",
                "--prompt",
                prompt["text"],
            ],
            check=True,
        )

        if idx != len(prompts):
            cont = input("Continue to the next prompt? [Enter/q]: ").strip().lower()
            if cont.startswith("q"):
                print("Stopped early.")
                break


if __name__ == "__main__":
    main()