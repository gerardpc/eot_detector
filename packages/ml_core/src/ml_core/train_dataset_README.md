# Train dataset

Labeled pause clips cut from [`livekit/eot-bench-data`](https://huggingface.co/datasets/livekit/eot-bench-data).

WAV files store **no labels**. The label for each clip is in **`index.csv`**.

| Column | Meaning |
| --- | --- |
| `path` | Relative WAV path, e.g. `hold/en__123__span0__t800__end.wav` |
| `label` | **0 = hold**, **1 = eot** |
| `class` | `hold` or `eot` (same information as `label`) |
| `split` | Hashed `train` / `val` split by turn `id` and `split_seed` |
| `id` | Source turn id |
| `language` | Source language code |
| `kind` | Cut inside the pause: `early200`, `mid1000`, or `end` |
| `t` | Clip end time in the source turn (seconds) |
| `pause_start`, `pause_end` | Source silence span |
| `duration` | Source turn duration |

Clips also live under `hold/` and `eot/`. Prefer `index.csv` as the source of
truth. `meta.json` stores the 0/1 convention, clip counts, `split_seed`, and
`val_fraction`.

## How labels are derived

The raw dataset has `silence_spans`, not hold/eot names.

- Earlier pauses of at least 100 ms → hold (`label=0`)
- Last pause of the turn → eot (`label=1`)

Each clip is the causal window `[max(0, t − 5s), t]` at 16 kHz.

This hashed split of the public validation set is **not** an official eot-bench
score.

Regenerate with:

```bash
uv run --package eot-ml-core eot-prepare
```
