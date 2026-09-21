# Analysis

Training-curve and ROC snapshots for the pause-head experiments, used in
[`slides/`](../slides/). Curve data is copied out of gitignored
`models/pause_head/<run_id>/history.jsonl`. ROC scores are p(eot) on the
homemade val split from each run’s `best.pt`.

## Files

| Path | Role |
| --- | --- |
| `runs.json` | Per-run `steps`, `val_acc`, and `val_loss` |
| `plot_curves.py` | Smoothed matplotlib charts |
| `val_acc.png` / `val_acc.svg` | Validation accuracy vs samples |
| `val_loss.png` / `val_loss.svg` | Validation BCE loss vs samples |
| `roc.json` | Val labels and per-run p(eot) |
| `score_roc.py` | Re-score val clips (needs checkpoints) |
| `plot_roc.py` | ROC overlay |
| `roc.png` / `roc.svg` | Validation ROC |
| `lang_acc.json` | Per-language accuracy at p(eot) ≥ 0.5 |
| `plot_lang_acc.py` | Language accuracy bars |
| `lang_acc.png` / `lang_acc.svg` | Val accuracy by language (linear tail 400 ms) |

```bash
uv run python analysis/plot_curves.py
uv run python analysis/plot_roc.py
uv run python analysis/plot_lang_acc.py
```

To rebuild `roc.json` after a new run (needs `data/train_dataset/` and
`models/pause_head/`):

```bash
uv run --package turn-runtime python analysis/score_roc.py
uv run python analysis/plot_roc.py
uv run python analysis/plot_lang_acc.py
```

Faint lines on the training charts are raw evals (every 3000 samples).
Bold lines are an exponential moving average with weight 0.85, the same
idea as TensorBoard’s smoothing slider.

The ROC treats **eot** as the positive class. The x-axis is the false
EOT rate on true holds — interruptions. The y-axis is the true EOT rate.
Dots are the live cutoff `p(eot) ≥ 0.5`.

`plot_lang_acc.py` joins those scores with `language` in
`data/train_dataset/index.csv` for **Linear, tail 400 ms**. The bar chart
marks English, the median across the 14 languages, and the best and worst
languages. Threshold is 0.5.

## Runs

All heads sit on frozen Whisper-tiny. `mean-pool` averages every encoder
frame in the ≤5 s clip. `tail` averages only the last *N* ms. `ema` is a
normalized recency-weighted mean with that half-life.

| Run id | Label |
| --- | --- |
| `20260920-163007` | MLP 384→64→GELU, mean-pool 5 s |
| `20260921-082751` | Linear, mean-pool 5 s |
| `20260921-084027` | Linear, tail 400 ms |
| `20260921-084831` | Linear, EMA 400 ms |
| `20260921-085616` | Linear, tail 1 s |
| `20260921-085931` | Linear, tail 200 ms |
| `20260921-093516` | MLP 384→64→GELU, tail 400 ms |
