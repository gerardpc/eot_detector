"""Tests for pause-head path resolution and listing."""

from pathlib import Path

from turn_runtime.classifier import list_pause_heads, resolve_head_weights


def test_resolve_prefers_best_in_current_run(tmp_path: Path) -> None:
    root = tmp_path / "pause_head"
    run = root / "20260101-000000"
    run.mkdir(parents=True)
    (run / "latest.pt").write_bytes(b"latest")
    (run / "best.pt").write_bytes(b"best")
    current = root / "current"
    current.symlink_to(run.name, target_is_directory=True)
    found = resolve_head_weights(root)
    assert found is not None
    assert found.name == "best.pt"
    assert found.read_bytes() == b"best"
    assert found.resolve() == (run / "best.pt").resolve()


def test_resolve_falls_back_to_latest(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "latest.pt").write_bytes(b"latest")
    assert resolve_head_weights(run) == run / "latest.pt"


def test_resolve_plain_pt_file(tmp_path: Path) -> None:
    path = tmp_path / "head.pt"
    path.write_bytes(b"weights")
    assert resolve_head_weights(path) == path


def test_list_pause_heads_includes_current_and_runs(tmp_path: Path) -> None:
    root = tmp_path / "pause_head"
    run = root / "20260101-000000"
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"best")
    (run / "config.json").write_text(
        '{"best_val_loss": 0.37, "head": "linear-64-gelu-linear"}',
        encoding="utf-8",
    )
    empty = root / "20260101-111111"
    empty.mkdir()
    current = root / "current"
    current.symlink_to(run.name, target_is_directory=True)
    heads = list_pause_heads(root)
    assert heads[0]["id"] == "current"
    assert heads[0]["label"] == "current (20260101-000000)"
    assert heads[0]["head"] == "linear-64-gelu-linear"
    assert heads[0]["pool"] == "mean"
    assert heads[0]["pool_ms"] == 1000.0
    assert heads[1]["id"] == "20260101-000000"
    assert "val 0.370" in heads[1]["label"]
    assert "linear-64-gelu-linear" in heads[1]["label"]
    assert heads[1]["head"] == "linear-64-gelu-linear"
    assert heads[1]["pool"] == "mean"
    assert all(item["id"] != "20260101-111111" for item in heads)


def test_list_pause_heads_without_store(tmp_path: Path) -> None:
    assert list_pause_heads(tmp_path / "missing") == [
        {"id": "current", "label": "current", "head": None, "pool": "mean", "pool_ms": 1000.0}
    ]
