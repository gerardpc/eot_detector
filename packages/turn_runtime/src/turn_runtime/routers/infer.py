"""HTTP route that scores one pause clip with the shared encoder."""

from __future__ import annotations

import asyncio
import io
import time

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["infer"])
"""HTTP route for clip-level p(eot)."""


def _wav_to_mono(payload: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV body to float32 mono PCM and sample rate."""
    try:
        audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid_wav") from exc
    if audio.size == 0:
        raise HTTPException(status_code=400, detail="empty_wav")
    return np.asarray(audio[:, 0], dtype=np.float32), int(sample_rate)


@router.post("/infer")
async def infer_clip(request: Request) -> dict[str, object]:
    """Score a WAV body with the frozen encoder and pause head.

    This is the request-shaped model API: no VAD, no WebSocket. The body is
    a WAV (the same format as `data/train_dataset/` clips). `latency_ms` is
    encoder + head time including the shared infer-lock wait, not HTTP
    overhead (that is still on `X-Process-Time-Ms`).
    """
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None or not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="engine_not_ready")
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="empty_body")
    audio, sample_rate = _wav_to_mono(payload)
    lock = request.app.state.infer_lock

    def _score() -> tuple[float, float]:
        """Run p(eot) under the process infer lock and time it."""
        started = time.perf_counter()
        with lock:
            p_eot = float(classifier.p_eot(audio, sample_rate))
        latency_ms = (time.perf_counter() - started) * 1000.0
        return p_eot, latency_ms

    p_eot, latency_ms = await asyncio.to_thread(_score)
    return {
        "ok": True,
        "p_eot": p_eot,
        "latency_ms": round(latency_ms, 3),
        "sample_rate": sample_rate,
        "samples": int(audio.size),
    }
