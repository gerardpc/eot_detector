"""Train the frozen-encoder pause head on labeled clips."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from logging import getLogger
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from turn_runtime.classifier import DEFAULT_HEAD_DIR, DEFAULT_MODEL_DIR, HEAD_NAME, PauseClassifier
from turn_runtime.config.logging_setup import setup_logging
from turn_runtime.download import download_whisper_tiny
from turn_runtime.runtime import ENCODER_ID

from .prepare import TARGET_SR, train_dir
from .run_config import PauseHeadRunConfig

_logger = getLogger(__name__)
"""Logger for pause-head training progress."""


def load_index(path: Path) -> list[dict[str, str]]:
    """Read `index.csv` into a list of row dicts."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def new_run_id() -> str:
    """Return a UTC `YYYYMMDD-HHMMSS` run directory name."""
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def write_config(run_dir: Path, config: PauseHeadRunConfig) -> Path:
    """Write `config.json` into `run_dir` and return its path."""
    path = run_dir / "config.json"
    path.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def promote_run(run_dir: Path, root: Path) -> Path:
    """Point `root/current` at `run_dir` (relative symlink)."""
    current = root / "current"
    if current.is_symlink() or current.exists():
        current.unlink()
    current.symlink_to(run_dir.name, target_is_directory=True)
    return current


def _dataset_meta(root: Path) -> dict[str, object]:
    """Load `meta.json` from a train dataset directory, or `{}`."""
    path = root / "meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_embeddings(
    classifier: PauseClassifier,
    records: list[dict[str, str]],
    root: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode clips once; return stacked embeddings and float labels."""
    embeddings: list[torch.Tensor] = []
    labels: list[int] = []
    for record in records:
        audio, sample_rate = sf.read(root / record["path"], dtype="float32")
        if int(sample_rate) != TARGET_SR:
            raise ValueError(f"expected {TARGET_SR} Hz clips, got {sample_rate}")
        pooled = classifier.pooled_embedding(np.asarray(audio, dtype=np.float32), sample_rate)
        embeddings.append(pooled.squeeze(0).detach().cpu())
        labels.append(int(record["label"]))
    if not embeddings:
        raise ValueError("no clips found for this split")
    return torch.stack(embeddings), torch.tensor(labels, dtype=torch.float32)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of examples with `sigmoid(logit) ≥ 0.5` matching the label."""
    preds = (torch.sigmoid(logits) >= 0.5).to(labels.dtype)
    return float((preds == labels).float().mean().item())


def should_evaluate(
    samples_seen: int,
    last_eval_at: int,
    eval_every: int,
    *,
    epoch_end: bool,
) -> bool:
    """True when enough new samples have been seen, or at epoch end."""
    if samples_seen <= 0 or samples_seen == last_eval_at:
        return False
    if epoch_end:
        return True
    return samples_seen - last_eval_at >= eval_every


def _full_metrics(
    classifier: PauseClassifier,
    loss_fn: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> dict[str, float]:
    """Compute train/val BCE loss and accuracy on cached embeddings."""
    was_training = classifier.head.training
    classifier.head.eval()
    with torch.no_grad():
        train_logits = classifier.head(x_train).squeeze(-1)
        train_loss = float(loss_fn(train_logits, y_train).item())
        train_acc = _accuracy(train_logits, y_train)
        if len(x_val) == 0:
            val_loss = train_loss
            val_acc = train_acc
        else:
            val_logits = classifier.head(x_val).squeeze(-1)
            val_loss = float(loss_fn(val_logits, y_val).item())
            val_acc = _accuracy(val_logits, y_val)
    if was_training:
        classifier.head.train()
    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }


def _log_metrics(writer: SummaryWriter, step: int, metrics: dict[str, float]) -> None:
    """Write train/val scalars to TensorBoard at `step` samples."""
    writer.add_scalar("train/loss", metrics["train_loss"], step)
    writer.add_scalar("train/acc", metrics["train_acc"], step)
    writer.add_scalar("val/loss", metrics["val_loss"], step)
    writer.add_scalar("val/acc", metrics["val_acc"], step)
    writer.flush()


def _append_history(run_dir: Path, row: dict[str, object]) -> None:
    """Append one JSONL eval record to `history.jsonl`."""
    with (run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _save_checkpoint(
    classifier: PauseClassifier,
    run_dir: Path,
    root: Path,
    config: PauseHeadRunConfig,
    *,
    is_best: bool,
) -> None:
    """Write `latest.pt`, optionally `best.pt`, then refresh `current`."""
    classifier.save_head(run_dir / "latest.pt")
    if is_best:
        classifier.save_head(run_dir / "best.pt")
    write_config(run_dir, config)
    promote_run(run_dir, root)


def train(
    *,
    dataset_dir: Path | None = None,
    out_dir: Path | None = None,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    train_seed: int = 0,
    split_seed: int | None = None,
    run_id: str | None = None,
    eval_every_samples: int = 3000,
) -> Path:
    """Train the pause head, writing a versioned run under `out_dir`."""
    root = dataset_dir or train_dir()
    store = out_dir or DEFAULT_HEAD_DIR
    store.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"No clips at {index_path}. Run `eot-prepare` first "
            "(it downloads livekit/eot-bench-data and writes data/train_dataset/)."
        )
    whisper_weights = DEFAULT_MODEL_DIR / "model.safetensors"
    if not whisper_weights.is_file():
        _logger.info("Whisper-tiny missing at %s; downloading", DEFAULT_MODEL_DIR)
        download_whisper_tiny(DEFAULT_MODEL_DIR)
    records = load_index(index_path)
    train_rows = [row for row in records if row["split"] == "train"]
    val_rows = [row for row in records if row["split"] == "val"]
    if not train_rows:
        raise ValueError(f"no train clips in {root / 'index.csv'}")

    meta = _dataset_meta(root)
    resolved_split_seed = 0 if split_seed is None else split_seed
    if split_seed is None and "split_seed" in meta:
        resolved_split_seed = int(meta["split_seed"])

    run = store / (run_id or new_run_id())
    run.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(train_seed)
    np.random.seed(train_seed)

    classifier = PauseClassifier(load_trained_head=False)
    classifier.encoder.eval()
    classifier.head.train()
    device = classifier.device
    config = PauseHeadRunConfig(
        run_id=run.name,
        encoder_id=ENCODER_ID,
        head=HEAD_NAME,
        dataset_id=str(meta.get("dataset_id") or "livekit/eot-bench-data"),
        dataset_dir=str(root),
        hold_label=int(meta.get("hold_label") or 0),
        eot_label=int(meta.get("eot_label") or 1),
        split_seed=resolved_split_seed,
        val_fraction=float(meta.get("val_fraction") or 0.2),
        train_seed=train_seed,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        eval_every_samples=eval_every_samples,
        n_train=len(train_rows),
        n_val=len(val_rows),
        device=str(device),
    )
    write_config(run, config)
    _logger.info("run directory %s", run)
    tb_dir = run / config.tensorboard_dir
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("config", config.model_dump_json(indent=2), 0)
    _logger.info("tensorboard --logdir %s", store)

    best_val = float("inf")
    samples_seen = 0
    last_eval_at = 0
    try:
        x_train, y_train = _cache_embeddings(classifier, train_rows, root)
        x_val, y_val = (
            _cache_embeddings(classifier, val_rows, root)
            if val_rows
            else (torch.zeros(0, x_train.shape[1]), torch.zeros(0))
        )
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        n_pos = float(y_train.sum().item())
        n_neg = float(len(y_train) - n_pos)
        pos_weight_value = n_neg / max(n_pos, 1.0)
        pos_weight = torch.tensor([pos_weight_value], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            classifier.head.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        shuffle = torch.Generator()
        shuffle.manual_seed(train_seed)
        config.n_train = len(train_rows)
        config.n_val = len(val_rows)
        config.pos_weight = pos_weight_value
        write_config(run, config)

        def evaluate(*, epoch: int, epoch_end: bool) -> None:
            """Run validation when due, checkpoint, and log metrics."""
            nonlocal best_val, last_eval_at
            if not should_evaluate(
                samples_seen,
                last_eval_at,
                eval_every_samples,
                epoch_end=epoch_end,
            ):
                return
            metrics = _full_metrics(classifier, loss_fn, x_train, y_train, x_val, y_val)
            is_best = metrics["val_loss"] < best_val
            if is_best:
                best_val = metrics["val_loss"]
                config.best_epoch = epoch
                config.best_samples = samples_seen
                config.best_val_loss = metrics["val_loss"]
            config.last_epoch = epoch
            config.last_samples = samples_seen
            config.last_train_loss = metrics["train_loss"]
            config.last_val_loss = metrics["val_loss"]
            config.last_train_acc = metrics["train_acc"]
            config.last_val_acc = metrics["val_acc"]
            last_eval_at = samples_seen
            _save_checkpoint(classifier, run, store, config, is_best=is_best)
            _append_history(
                run,
                {
                    "epoch": epoch,
                    "samples": samples_seen,
                    "epoch_end": epoch_end,
                    "train_loss": metrics["train_loss"],
                    "train_acc": metrics["train_acc"],
                    "val_loss": metrics["val_loss"],
                    "val_acc": metrics["val_acc"],
                    "best": is_best,
                },
            )
            _log_metrics(writer, samples_seen, metrics)
            _logger.info(
                "epoch %s/%s samples=%s train_loss=%.4f train_acc=%.3f "
                "val_loss=%.4f val_acc=%.3f%s",
                epoch,
                epochs,
                samples_seen,
                metrics["train_loss"],
                metrics["train_acc"],
                metrics["val_loss"],
                metrics["val_acc"],
                " *best" if is_best else "",
            )

        for epoch in range(epochs):
            classifier.head.train()
            order = torch.randperm(len(x_train), generator=shuffle)
            for start in range(0, len(order), batch_size):
                index = order[start : start + batch_size].to(device)
                logits = classifier.head(x_train[index]).squeeze(-1)
                loss = loss_fn(logits, y_train[index])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                samples_seen += int(index.numel())
                writer.add_scalar("train/batch_loss", float(loss.item()), samples_seen)
                evaluate(epoch=epoch + 1, epoch_end=False)
            evaluate(epoch=epoch + 1, epoch_end=True)
    except KeyboardInterrupt:
        config.status = "interrupted"
        classifier.eval()
        if samples_seen > 0:
            _save_checkpoint(classifier, run, store, config, is_best=False)
            _logger.warning("interrupted after %s samples; saved %s", samples_seen, run)
        else:
            write_config(run, config)
            _logger.warning("interrupted before the first step; no weights in %s", run)
        return run
    finally:
        writer.close()

    config.status = "completed"
    write_config(run, config)
    _logger.info("saved head to %s", run)
    return run


def main() -> None:
    """CLI entrypoint for `eot-train`."""
    parser = argparse.ArgumentParser(description="Train the pause classifier head.")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0, help="Training shuffle seed.")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Recorded split seed; defaults to data/train_dataset/meta.json.",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=3000,
        help="Compute val loss every N training samples (also at epoch end).",
    )
    args = parser.parse_args()
    setup_logging()
    train(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_seed=args.seed,
        split_seed=args.split_seed,
        run_id=args.run_id,
        eval_every_samples=args.eval_every,
    )


if __name__ == "__main__":
    main()
