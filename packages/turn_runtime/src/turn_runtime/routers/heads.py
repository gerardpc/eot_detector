"""Router listing available pause-head runs."""

from fastapi import APIRouter, HTTPException, Request

from turn_runtime.classifier import list_pause_heads

router = APIRouter(tags=["heads"])
"""HTTP routes for listing pause-head runs."""


@router.get("/heads")
async def heads(request: Request) -> dict[str, object]:
    """List `current` plus versioned runs under the configured pause-head directory."""
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="engine_not_ready")
    settings = request.app.state.settings
    return {
        "ok": True,
        "heads": list_pause_heads(settings.pause_head_dir),
        "selected": "current",
    }
