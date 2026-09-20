"""Tests for hashed splits, run config, and training helpers."""

from pathlib import Path

from ml_core.prepare import split_for_id
from ml_core.run_config import PauseHeadRunConfig
from ml_core.train import _log_metrics, promote_run, should_evaluate, write_config


def test_split_seed_zero_matches_unseeded_hash() -> None:
    assert split_for_id("en__1", seed=0) == split_for_id("en__1")


def test_split_is_deterministic_for_a_seed() -> None:
    assert split_for_id("en__1", seed=4) == split_for_id("en__1", seed=4)


def test_run_config_roundtrip(tmp_path: Path) -> None:
    config = PauseHeadRunConfig(
        run_id="20260101-000000",
        dataset_dir=str(tmp_path),
        epochs=3,
        batch_size=8,
        lr=1e-3,
        split_seed=7,
        train_seed=11,
    )
    run_dir = tmp_path / config.run_id
    run_dir.mkdir()
    path = write_config(run_dir, config)
    loaded = PauseHeadRunConfig.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded.split_seed == 7
    assert loaded.train_seed == 11
    assert loaded.epochs == 3
    assert loaded.status == "running"
    assert loaded.tensorboard_dir == "tb"
    assert loaded.eval_every_samples == 3000
    assert loaded.head == "linear-64-gelu-linear"
    assert loaded.weight_decay == 1e-3


def test_promote_run_symlink(tmp_path: Path) -> None:
    root = tmp_path / "pause_head"
    first = root / "a"
    second = root / "b"
    first.mkdir(parents=True)
    second.mkdir()
    promote_run(first, root)
    assert (root / "current").resolve() == first.resolve()
    promote_run(second, root)
    assert (root / "current").resolve() == second.resolve()


def test_log_metrics_writes_tensorboard_events(tmp_path: Path) -> None:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(tmp_path))
    _log_metrics(
        writer,
        3000,
        {
            "train_loss": 0.5,
            "train_acc": 0.7,
            "val_loss": 0.6,
            "val_acc": 0.65,
        },
    )
    writer.close()
    assert any(tmp_path.iterdir())


def test_should_evaluate_every_3k_and_epoch_end() -> None:
    assert not should_evaluate(0, 0, 3000, epoch_end=False)
    assert should_evaluate(3000, 0, 3000, epoch_end=False)
    assert not should_evaluate(3001, 3000, 3000, epoch_end=False)
    assert should_evaluate(6000, 3000, 3000, epoch_end=False)
    assert should_evaluate(24995, 24000, 3000, epoch_end=True)
    assert not should_evaluate(3000, 3000, 3000, epoch_end=True)
