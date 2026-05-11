# Dataset Format

Each typing session is stored as a JSON file.

Example structure:

```json
{
  "prompt": "hello world",
  "typed_text": "hello world",
  "keys": ["h", "e", "l"],
  "key_times_ms": [0.0, 55.1, 120.4],
  "created_at_utc": "...",
  "metadata": {}
}