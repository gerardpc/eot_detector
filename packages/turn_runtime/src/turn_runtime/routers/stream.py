"""WebSocket route for live PCM and per-connection turn state."""

from __future__ import annotations

import asyncio
import json
from logging import getLogger

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from turn_runtime.routers.health import turn_status
from turn_runtime.runtime import TurnRunner

router = APIRouter()
"""WebSocket route for live PCM."""
_logger = getLogger(__name__)
"""Logger for disconnect and stream errors."""


def _session(websocket: WebSocket) -> TurnRunner:
    """Build a VAD session that shares the process encoder under an infer lock."""
    settings = websocket.app.state.settings
    return TurnRunner(
        classifier=websocket.app.state.classifier,
        infer_lock=websocket.app.state.infer_lock,
        head_dir=settings.pause_head_dir,
    )


@router.websocket("/ws")
async def stream_state(websocket: WebSocket) -> None:
    """Accept float32 PCM frames and return speaking / hold / eot events."""
    origin = websocket.headers.get("origin")
    allowed = list(getattr(websocket.app.state.settings, "cors_allowed_origins", []))
    if origin and "*" not in allowed and origin not in allowed:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    runner = _session(websocket)
    sample_rate = 48_000
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            text = message.get("text")
            if text:
                payload = json.loads(text)
                kind = payload.get("type")
                if kind == "start":
                    sample_rate = int(payload.get("sampleRate") or sample_rate)
                    runner.reset_stream()
                    head_id = str(payload.get("head") or "current")
                    if runner.select_head(head_id) is None:
                        await websocket.send_json({"ok": False, "error": "pause head not found"})
                        continue
                    await websocket.send_json({"ok": True, "turn": turn_status(websocket, runner)})
                elif kind == "head":
                    head_id = str(payload.get("id") or payload.get("head") or "current")
                    if runner.select_head(head_id) is None:
                        await websocket.send_json({"ok": False, "error": "pause head not found"})
                        continue
                    await websocket.send_json({"ok": True, "turn": turn_status(websocket, runner)})
                continue
            raw = message.get("bytes")
            if not raw or len(raw) % 4:
                continue
            samples = np.frombuffer(raw, dtype=np.float32).copy()
            event = await asyncio.to_thread(runner.consume_stream, samples, sample_rate)
            await websocket.send_json(
                {
                    "ok": True,
                    "state": event.state,
                    "silence_seconds": round(event.silence_seconds, 3),
                    "rms": round(event.rms, 4),
                    "p_eot": event.p_eot,
                }
            )
    except WebSocketDisconnect:
        _logger.info("WebSocket client disconnected")
