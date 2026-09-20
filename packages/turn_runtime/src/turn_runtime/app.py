"""Application factory and server entrypoint for the turn runtime API."""

from __future__ import annotations

import time
from logging import getLogger

import uvicorn
from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware

from turn_runtime import __version__
from turn_runtime.config.lifespan import lifespan
from turn_runtime.config.logging_setup import setup_logging
from turn_runtime.routers.heads import router as heads_router
from turn_runtime.routers.health import router as health_router
from turn_runtime.routers.stream import router as stream_router
from turn_runtime.settings.settings import Settings, get_settings

setup_logging()

_logger = getLogger(__name__)
"""Logger for application factory and bind messages."""


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = settings or get_settings()
    _logger.info("Creating turn runtime app")
    app = FastAPI(
        title=settings.application_name,
        description=settings.application_description,
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    if settings.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            # Browser origins allowed to call this API cross-origin.
            allow_origins=settings.cors_allowed_origins,
            # Keep credentialed browser requests off; the UI does not use cookies.
            allow_credentials=False,
            # Allow the methods used by this API plus preflight OPTIONS requests.
            allow_methods=["GET", "POST", "OPTIONS"],
            # Allow the headers browser clients send for JSON requests.
            allow_headers=["Authorization", "Content-Type"],
            # Let browser clients read the custom timing header from responses.
            expose_headers=["X-Process-Time-Ms"],
        )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Attach X-Process-Time-Ms to every response."""
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response

    app.include_router(health_router)
    app.include_router(heads_router)
    app.include_router(stream_router)
    return app


def run() -> None:
    """Start the API service with Uvicorn."""
    settings = get_settings()
    uvicorn.run(
        "turn_runtime.app:create_app",
        host=settings.host,
        port=settings.port,
        reload=False,
        factory=True,
    )


if __name__ == "__main__":
    run()
