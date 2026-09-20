"""Tests for clip extraction into `data/train_dataset/`."""

from pathlib import Path

import ml_core.prepare as prepare_mod
import numpy as np
import soundfile as sf
from ml_core.prepare import TARGET_SR, prepare_dataset


def test_prepare_writes_hold_and_eot_clips(tmp_path: Path, monkeypatch) -> None:
    sample_rate = 16_000
    audio = np.zeros(sample_rate * 4, dtype=np.float32)
    rows = [
        {
            "id": "turn-a",
            "language": "en",
            "duration": 4.0,
            "silence_spans": [
                {"start": 0.5, "end": 1.0},
                {"start": 2.0, "end": 3.5},
            ],
            "audio": {"array": audio, "sampling_rate": sample_rate},
        }
    ]
    monkeypatch.setattr(prepare_mod, "train_dir", lambda: tmp_path)

    out = prepare_dataset(rows)
    hold_files = list((out / "hold").glob("*.wav"))
    eot_files = list((out / "eot").glob("*.wav"))
    assert len(hold_files) == 2
    assert len(eot_files) == 3

    index = (out / "index.csv").read_text(encoding="utf-8")
    assert "label" in index
    assert ",0," in index
    assert ",1," in index
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "`index.csv`" in readme
    assert "0 = hold" in readme
    assert "1 = eot" in readme

    clip, sr = sf.read(hold_files[0], dtype="float32")
    assert sr == TARGET_SR
    assert 0 < clip.shape[0] <= TARGET_SR * 5
