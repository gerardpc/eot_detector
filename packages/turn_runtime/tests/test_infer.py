"""HTTP tests for POST /infer."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from turn_runtime.app import create_app
from turn_runtime.settings.settings import Settings


class _FakeClassifier:
    """Stand-in encoder that returns a fixed p(eot)."""

    def p_eot(self, audio: np.ndarray, sample_rate: int) -> float:
        """Ignore audio and return a constant score."""
        assert audio.size > 0
        assert sample_rate > 0
        return 0.73


def _client() -> TestClient:
    """Build a TestClient that skips Whisper load."""
    return TestClient(create_app(Settings(load_model=False)))


def _wav_bytes(n: int = 1600, sample_rate: int = 16_000) -> bytes:
    """Encode silence as a WAV body."""
    buf = io.BytesIO()
    sf.write(buf, np.zeros(n, dtype=np.float32), sample_rate, format="WAV")
    return buf.getvalue()


def test_infer_requires_classifier() -> None:
    with _client() as client:
        response = client.post("/infer", content=_wav_bytes())
        assert response.status_code == 503


def test_infer_rejects_empty_and_invalid_bodies() -> None:
    with _client() as client:
        client.app.state.classifier = _FakeClassifier()
        empty = client.post("/infer", content=b"")
        assert empty.status_code == 400
        invalid = client.post("/infer", content=b"not-a-wav")
        assert invalid.status_code == 400


def test_infer_returns_score_and_latency() -> None:
    with _client() as client:
        client.app.state.classifier = _FakeClassifier()
        response = client.post(
            "/infer",
            content=_wav_bytes(n=3200, sample_rate=16_000),
            headers={"Content-Type": "audio/wav"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["p_eot"] == 0.73
        assert body["sample_rate"] == 16_000
        assert body["samples"] == 3200
        assert isinstance(body["latency_ms"], float)
        assert body["latency_ms"] >= 0.0
        assert "X-Process-Time-Ms" in response.headers
