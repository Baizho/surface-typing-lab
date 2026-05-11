# Surface Typing Lab

A small research toolkit for analyzing typing rhythm and surface-typing inference.

This project is a lightweight framework for collecting typing sessions, extracting timing-based features, comparing sessions, and experimenting with baseline typing-rhythm analysis. It is designed as a starting point for wearable and surface-typing research.

## What it does

- logs typing timestamps with a custom prompt
- stores sessions in a structured JSON format
- extracts inter-keystroke timing (IKT) features
- approximates left/right hand typing patterns on QWERTY
- visualizes typing rhythm
- compares sessions using simple feature similarity
- summarizes a small dataset of typing sessions
- supports batch data collection with a standardized prompt bank

## Why this exists

Typing rhythm contains useful temporal information that can be studied for:

- surface-typing research
- wearable sensing
- behavioral biometrics
- smartwatch-based input systems
- cross-modal typing inference

The project is intentionally small, reusable, and research-oriented rather than a full product.

---

# Current Features

## Data Collection

Run the logger and type a sentence:

```bash
python3 -m src.logger.logger --prompt "typing is fun"
```

This saves a JSON file into `data/raw/`.

---

## Batch Collection

Collect a standardized set of prompts:

```bash
python3 -m src.batch_collect --category balanced --limit 5
```

---

## Feature Extraction

Extract timing-based features from a saved session:

```bash
python3 src/features.py
```

Example extracted features:

- mean inter-keystroke timing
- typing duration
- left/right hand usage
- hand alternation ratio

---

## Visualization

Plot typing rhythm for the latest collected session:

```bash
python3 src/plot_sample.py
```

This generates timing visualizations in `data/figures/`.

---

## Session Comparison

Compare one typing session against others in the dataset:

```bash
python3 -m src.compare_sessions --query data/raw/<sample>.json
```

The tool computes cosine similarity across extracted timing features.

---

## Dataset Summary

Generate a summary report over all collected sessions:

```bash
python3 src/summarize_dataset.py
```

---

## Dataset-Wide Analysis

Print aggregate statistics across sessions:

```bash
python3 -m src.analyze_dataset
```

Example outputs include:

- average typing speed
- inter-keystroke timing variance
- session duration statistics
- hand alternation statistics

---

# Session Format

Each typing session is stored as JSON.

Example:

```json
{
  "prompt": "hello world",
  "typed_text": "hello world",
  "keys": ["h", "e", "l"],
  "key_times_ms": [0.0, 55.1, 120.4],
  "created_at_utc": "...",
  "metadata": {}
}
```

Main fields:

| Field | Description |
|---|---|
| prompt | Original prompt shown to the user |
| typed_text | Final typed sentence |
| keys | Ordered keystrokes |
| key_times_ms | Relative timestamps for each keypress |
| created_at_utc | UTC timestamp |
| metadata | Extra experiment metadata |

See `docs/dataset_format.md` for full details.

---

# Project Structure

```text
surface-typing-lab/
├── src/
│   ├── logger/
│   ├── features.py
│   ├── session.py
│   ├── prompt_bank.py
│   ├── batch_collect.py
│   ├── compare_sessions.py
│   ├── analyze_dataset.py
│   └── plot_sample.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── figures/
├── docs/
│   ├── collection_protocol.md
│   └── dataset_format.md
├── reports/
└── README.md
```

---

# Current Status

The repository currently supports:

- interactive typing collection
- timing feature extraction
- session comparison
- visualization
- dataset aggregation
- standardized prompt collection

The project is now structured similarly to an early-stage sensing/HCI research toolkit.

---

# Future Work

Potential future extensions include:

- smartwatch IMU integration
- accelerometer and gyroscope streaming
- cross-modal typing inference
- stronger sequence decoding baselines
- personalization experiments
- multi-user studies
- domain adaptation
- live wearable sensing pipelines

---

# Notes

This repository is intended as a compact experimental framework for studying:

- surface typing
- wearable sensing
- typing rhythm
- behavioral biometrics
- edge inference for cyber-physical systems