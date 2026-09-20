"""Router exposing health and version endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from turn_runtime import __version__
from turn_runtime.runtime import (
    CONTEXT_SECONDS,
    ENCODER_ID,
    FORCE_EOT_SECONDS,
    MIN_SILENCE_SECONDS,
)

router = APIRouter(tags=["health"])
"""HTTP routes for process health and package version."""


def turn_status(connection: Any, runner: Any | None = None) -> dict[str, object]:
    """Build the turn-runtime status payload for health and WebSocket acks."""
    app = connection.app
    classifier = getattr(app.state, "classifier", None)
    if runner is not None:
        status = runner.status()
    else:
        status = {
            "state": "hold",
            "encoder_id": ENCODER_ID,
            "classifier": "whisper-tiny-head" if classifier is not None else "none",
            "head_id": "current",
            "min_silence_seconds": MIN_SILENCE_SECONDS,
            "force_eot_seconds": FORCE_EOT_SECONDS,
            "context_seconds": CONTEXT_SECONDS,
        }
    if classifier is not None:
        status["device"] = str(getattr(classifier, "device", ""))
    status["ready"] = bool(getattr(app.state, "ready", False))
    return status


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    """Return readiness only once encoder startup finished (or was skipped)."""
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="engine_not_ready")
    return {"status": "ok", "turn": turn_status(request)}


@router.get("/version")
async def api_version() -> dict[str, str]:
    """Return the installed turn-runtime package version."""
    return {"version": __version__}
