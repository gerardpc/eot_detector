"""PauseClassifier tests that skip when Whisper weights are missing."""

import numpy as np
import pytest
from turn_runtime.classifier import DEFAULT_MODEL_DIR, PauseClassifier

pytest.importorskip("torch")


@pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "model.safetensors").exists(),
    reason="whisper-tiny weights are not downloaded",
)
def test_untrained_head_stays_on_hold() -> None:
    classifier = PauseClassifier(load_trained_head=False)
    audio = np.zeros(16_000, dtype=np.float32)
    assert classifier.p_eot(audio, 16_000) < 0.1


@pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "model.safetensors").exists(),
    reason="whisper-tiny weights are not downloaded",
)
def test_head_roundtrip(tmp_path) -> None:
    classifier = PauseClassifier(load_trained_head=False)
    path = tmp_path / "pause_head.pt"
    classifier.save_head(path)
    loaded = PauseClassifier(load_trained_head=False)
    assert loaded.load_head(path)
    audio = np.zeros(16_000, dtype=np.float32)
    assert loaded.p_eot(audio, 16_000) < 0.1
