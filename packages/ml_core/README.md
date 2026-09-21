# ml_core

Dataset preparation and pause-head training (`eot-ml-core`).

This package downloads [`livekit/eot-bench-data`](https://huggingface.co/datasets/livekit/eot-bench-data),
cuts causal pause clips, and trains a pause head on frozen Whisper-tiny
embeddings used by `turn-runtime`. The recipe is a linear `384 → 1` head
on a **tail pool** of the last **1 s** of encoder frames. Train and live
inference share `PauseClassifier`.

A hashed 80/20 split of the public validation set is for local training only.
It is not an official eot-bench score.

## Labels

The raw dataset stores `silence_spans`, not class names.

| Label | Class | Rule |
| ---: | --- | --- |
| 0 | hold | Mid-turn pause ≥ 100 ms |
| 1 | eot | Last pause of the turn |

Each clip is `[max(0, t − 5s), t]` at 16 kHz.

| Pause type | Cut times `t` |
| --- | --- |
| hold | `start + 0.2s` if it fits, and `end` |
| eot | those, plus `start + 1s` if still before `end` |

WAV files are unlabeled. **`data/train_dataset/index.csv`** is the source of
truth (`path`, `label`, `class`, `split`, …). Clips also sit in `hold/` and
`eot/`. See `data/train_dataset/README.md` after prepare.

## Commands

```bash
uv run --package eot-ml-core eot-prepare
uv run --package eot-ml-core eot-train
uv run --package eot-ml-core tensorboard --logdir models/pause_head
```

`eot-prepare` downloads `livekit/eot-bench-data` into gitignored
`data/raw_dataset/` and writes clips + `index.csv` under
`data/train_dataset/`. It does **not** download Whisper-tiny.

`eot-train` needs those clips (`eot-prepare` first) and the encoder under
`models/whisper-tiny/`. If the encoder is missing, it downloads it. Then it
writes a new run folder and points `models/pause_head/current` at it.
Optional flags: `--head` (default `linear`), `--pool` (default `tail`),
`--pool-ms` (default 1000), `--epochs`, `--batch-size`, `--lr`,
`--weight-decay`, `--seed`, `--split-seed`, `--run-id`, `--eval-every`,
`--dataset-dir`, `--out-dir`, `--no-promote`.

`--pool tail` averages the last 1 s of encoder frames. `--no-promote`
keeps `current` on the previous run.

## Training runs

```
models/pause_head/<run_id>/
  config.json     # PauseHeadRunConfig (hyperparameters, split_seed, train_seed, metrics)
  latest.pt       # head after the last eval, or Ctrl+C
  best.pt         # lowest validation loss so far
  history.jsonl   # per-eval metrics
  tb/             # TensorBoard event files
models/pause_head/current -> <run_id>
```

The live runtime loads `current/best.pt`, then `latest.pt`.

Validation loss is computed every **3000 training samples** (override with
`--eval-every`) and again at epoch end. TensorBoard x-axis is samples seen.
`train/batch_loss` is logged every optimizer step.

Checkpoints are for inference. Optimizer state is not saved; starting train
again creates a new `run_id`. Ctrl+C after the first eval still leaves
`latest.pt` / `best.pt` on disk. Start the runtime again, or pick the run in
the Head menu, to load them.

Package internals: [`src/ml_core/README.md`](src/ml_core/README.md).

## Layout

| Path | Contents |
| --- | --- |
| `src/ml_core/prepare.py` | Download and clip extraction |
| `src/ml_core/samples.py` | Pause cut times |
| `src/ml_core/train.py` | Training loop, checkpoints, TensorBoard |
| `src/ml_core/run_config.py` | `PauseHeadRunConfig` |
