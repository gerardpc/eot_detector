"""Tests for training-clip encoder throughput helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import soundfile as sf
from turn_runtime.stress import (
    StageResult,
    StressRun,
    default_output_dir,
    load_clip_paths,
    parse_rps_list,
    run_stress,
    write_report,
)


def _write_clip_dir(root: Path, n: int = 3) -> Path:
    """Write a tiny train-dataset layout under `root`."""
    wavs = root / "wavs"
    wavs.mkdir()
    rows = ["path,label\n"]
    for i in range(n):
        rel = f"wavs/clip_{i}.wav"
        sf.write(wavs / f"clip_{i}.wav", np.zeros(800, dtype=np.float32), 16_000)
        rows.append(f"{rel},hold\n")
    (root / "index.csv").write_text("".join(rows), encoding="utf-8")
    return root


def test_load_clip_paths_requires_index(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="eot-prepare"):
        load_clip_paths(tmp_path, limit=1, seed=0)


def test_load_clip_paths_samples_index(tmp_path: Path) -> None:
    root = _write_clip_dir(tmp_path, n=4)
    paths = load_clip_paths(root, limit=2, seed=0)
    assert len(paths) == 2
    assert all(path.is_file() for path in paths)


def test_parse_rps_list() -> None:
    assert parse_rps_list("1,2,max") == [1.0, 2.0, None]
    assert parse_rps_list("4") == [4.0]
    with pytest.raises(ValueError, match="positive"):
        parse_rps_list("0")
    with pytest.raises(ValueError, match="empty"):
        parse_rps_list(" , ")


def test_default_output_dir() -> None:
    now = datetime(2026, 9, 20, 23, 9, 0, tzinfo=ZoneInfo("Europe/Madrid"))
    path = default_output_dir(now)
    assert path.name == "2026-09-20T23-09-00"
    assert path.parent.name == "stress_tests"


def test_write_report(tmp_path: Path) -> None:
    run = StressRun(
        meta={
            "started_at": "2026-09-20T23-09-00",
            "mode": "in-process",
            "url": None,
            "device": "cpu",
            "dataset_dir": "/tmp/clips",
            "clips_preloaded": 2,
            "workers": 1,
            "seconds_per_stage": 1,
            "seed": 0,
            "rps": ["1", "max"],
        },
        stages=[
            StageResult(
                label="1",
                offered_rps=1.0,
                achieved_rps=1.0,
                workers=1,
                forwards=3,
                seconds=3.0,
                wall_ms=[10.0, 11.0, 12.0],
                encoder_ms=[9.0, 10.0, 11.0],
            ),
            StageResult(
                label="max",
                offered_rps=None,
                achieved_rps=40.0,
                workers=1,
                forwards=4,
                seconds=0.1,
                wall_ms=[8.0, 8.5, 9.0, 20.0],
                encoder_ms=[8.0, 8.5, 9.0, 20.0],
            ),
        ],
    )
    report = write_report(run, tmp_path)
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "Encoder stress 2026-09-20T23-09-00" in text
    assert "latency_distribution.png" in text
    assert (tmp_path / "latency_distribution.png").is_file()
    assert (tmp_path / "throughput.png").is_file()
    assert (tmp_path / "summary.json").is_file()
    samples = (tmp_path / "samples.csv").read_text(encoding="utf-8")
    assert "encoder_latency_ms" in samples
    assert samples.count("\n") == 8  # header + 7 samples


def test_run_stress_in_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_clip_dir(tmp_path, n=2)

    class _FakeClassifier:
        device = "cpu"

        def p_eot(self, audio: np.ndarray, sample_rate: int) -> float:
            return 0.4

    monkeypatch.setattr("turn_runtime.stress._ensure_encoder", lambda: None)
    monkeypatch.setattr("turn_runtime.stress.PauseClassifier", lambda **_kwargs: _FakeClassifier())
    run = run_stress(
        dataset_dir=root,
        seconds=0.05,
        clips=2,
        workers=1,
        seed=0,
        url=None,
        rps=[None],
    )
    assert run.meta["mode"] == "in-process"
    assert len(run.stages) == 1
    stage = run.stages[0]
    assert stage.label == "max"
    assert stage.forwards >= 1
    assert stage.achieved_rps > 0
    assert stage.stats()["encoder_latency_ms_p50"] >= 0
