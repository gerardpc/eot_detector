"""Pydantic config written next to each pause-head training run."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class PauseHeadRunConfig(BaseModel):
    """Hyperparameters and split metadata for one `models/pause_head/<run_id>/`."""

    run_id: str
    """Directory name under `models/pause_head/` for this training run."""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    """UTC timestamp when the run directory was created."""
    encoder_id: str = "openai/whisper-tiny"
    """Frozen encoder identifier used to produce embeddings."""
    head: str = "linear-64-gelu-linear"
    """Pause-head architecture name (`384 → 64 → GELU → 1`)."""
    dataset_id: str = "livekit/eot-bench-data"
    """Source dataset id recorded for the homemade split."""
    dataset_dir: str
    """Filesystem path to `data/train_dataset/` (clips + `index.csv`)."""
    hold_label: int = 0
    """Integer class for mid-turn hold pauses."""
    eot_label: int = 1
    """Integer class for end-of-turn pauses."""
    split_seed: int = 0
    """Seed mixed into the hashed turn-id split (0 matches an unseeded hash)."""
    val_fraction: float = 0.2
    """Fraction of turn ids assigned to the validation split."""
    train_seed: int = 0
    """Seed for embedding-cache shuffle and first-layer init."""
    epochs: int
    """Maximum number of passes over the training embeddings."""
    batch_size: int
    """Optimizer batch size over cached embeddings."""
    lr: float
    """AdamW learning rate."""
    weight_decay: float = 1e-3
    """AdamW weight decay."""
    eval_every_samples: int = 3000
    """Run validation every this many training samples, and at epoch end."""
    n_train: int = 0
    """Number of training clips in this run."""
    n_val: int = 0
    """Number of validation clips in this run."""
    pos_weight: float = 1.0
    """BCE positive-class weight (`n_neg / n_pos`) used at train time."""
    device: str = "cpu"
    """Torch device the head was trained on."""
    status: Literal["running", "completed", "interrupted"] = "running"
    """Whether training is in progress, finished, or stopped with Ctrl+C."""
    last_epoch: int = 0
    """Last epoch that produced an eval (1-based)."""
    last_samples: int = 0
    """Training samples seen at the last eval."""
    best_epoch: int | None = None
    """Epoch of the lowest validation loss, if any eval has run."""
    best_samples: int | None = None
    """Samples seen when `best.pt` was written."""
    best_val_loss: float | None = None
    """Lowest validation BCE loss so far."""
    last_train_loss: float | None = None
    """Training BCE loss at the last eval."""
    last_val_loss: float | None = None
    """Validation BCE loss at the last eval."""
    last_train_acc: float | None = None
    """Training accuracy at the last eval (`p(eot) ≥ 0.5`)."""
    last_val_acc: float | None = None
    """Validation accuracy at the last eval (`p(eot) ≥ 0.5`)."""
    tensorboard_dir: str = "tb"
    """Directory name under the run folder for TensorBoard event files."""
