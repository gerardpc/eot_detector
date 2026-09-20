"""Router serving the capture page and UI health."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from turn_ui import __version__

router = APIRouter(tags=["ui"])
"""HTTP routes for the capture page and UI health."""
_APP_DIR = Path(__file__).resolve().parents[1]
"""Package directory that contains `index.html`."""


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> str:
    """Serve the microphone UI with the runtime URL injected."""
    html = (_APP_DIR / "index.html").read_text(encoding="utf-8")
    return html.replace("__RUNTIME_URL__", request.app.state.settings.runtime_base_url)


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    """Return UI process liveness and the runtime origin it points at."""
    settings = request.app.state.settings
    return {
        "status": "ok",
        "runtime_url": settings.runtime_base_url,
        "version": __version__,
    }
