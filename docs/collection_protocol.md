# Typing Data Collection Protocol

This repository collects typing sessions in a standardized JSON format.

## Recommended procedure
- Use the same keyboard and typing setup when possible.
- Collect multiple repetitions per prompt.
- Include a mix of short, balanced, punctuation-heavy, and numeric prompts.
- Save each session automatically in `data/raw/`.
- Record any relevant metadata later, such as device, layout, or environment.

## Batch collection
Run:

```bash
python3 -m src.batch_collect --category balanced --limit 5