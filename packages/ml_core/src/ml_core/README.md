# `ml_core` package

Dataset clips and pause-head training. Live inference is `turn_runtime`;
this package imports `PauseClassifier` and trains only the tiny head.

## Structure

```text
ml_core/
├── prepare.py                   Download livekit/eot-bench-data and cut clips
├── samples.py                   Hold / eot cut times from silence_spans
├── train.py                     Training loop, checkpoints, TensorBoard
├── run_config.py                PauseHeadRunConfig (written to config.json)
└── train_dataset_README.md      Copied into data/train_dataset/README.md
```

## Flow

```text
eot-prepare
        │
        ├─ data/raw_dataset/          Hugging Face cache of eot-bench-data
        └─ data/train_dataset/        labeled WAVs + index.csv
                │
                ▼
        eot-train
                │
                └─ models/pause_head/<run_id>/{config.json,best.pt,latest.pt,tb/}
                         current -> <run_id>
```

Homemade 80/20 splits of the public validation set are for local training only.
They are not official eot-bench scores.
