"""Application factory and server entrypoint for the turn UI."""

from __future__ import annotations

import time
from logging import getLogger

import uvicorn
from fastapi import FastAPI, Request

from turn_ui import __version__
from turn_ui.config.logging_setup import setup_logging
from turn_ui.routers.page import router as page_router
from turn_ui.settings.settings import Settings, get_settings

setup_logging()

_logger = getLogger(__name__)
"""Logger for UI factory and bind messages."""


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI UI application instance."""
    settings = settings or get_settings()
    _logger.info("Creating turn UI app runtime_url=%s", settings.runtime_base_url)
    app = FastAPI(
        title=settings.application_name,
        description=settings.application_description,
        version=__version__,
    )
    app.state.settings = settings

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Attach X-Process-Time-Ms to every response."""
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response

    app.include_router(page_router)
    return app


def run() -> None:
    """Start the UI with Uvicorn."""
    settings = get_settings()
    uvicorn.run(
        "turn_ui.app:create_app",
        host=settings.host,
        port=settings.port,
        reload=False,
        factory=True,
    )


if __name__ == "__main__":
    run()
