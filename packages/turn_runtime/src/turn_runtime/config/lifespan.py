"""FastAPI lifespan hooks for encoder and pause-head initialization."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from logging import getLogger

from fastapi import FastAPI

from turn_runtime.classifier import PauseClassifier
from turn_runtime.settings.settings import Settings, get_settings

_logger = getLogger(__name__)
"""Logger for encoder load and shutdown."""


def _load_shared_classifier(settings: Settings) -> PauseClassifier:
    """Load the frozen encoder and current pause head, or fail startup."""
    weights = settings.whisper_model_dir / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(
            "WHISPER_MODEL_DIR does not contain model.safetensors: "
            f"{settings.whisper_model_dir}. Run "
            "`python -m turn_runtime.download` before serving."
        )
    classifier = PauseClassifier(
        model_dir=settings.whisper_model_dir,
        load_trained_head=False,
    )
    classifier.load_head(settings.pause_head_dir)
    return classifier


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and store shared runtime state for the API lifecycle."""
    settings = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = settings
    app.state.infer_lock = threading.Lock()
    app.state.classifier = None
    app.state.ready = False
    _logger.info(
        (
            "Resolved turn runtime settings: host=%s port=%s load_model=%s "
            "whisper_model_dir=%s pause_head_dir=%s cors_allowed_origins=%s"
        ),
        settings.host,
        settings.port,
        settings.load_model,
        settings.whisper_model_dir,
        settings.pause_head_dir,
        settings.cors_allowed_origins,
    )
    if settings.load_model:
        _logger.info("Loading Whisper-tiny encoder from %s", settings.whisper_model_dir)
        app.state.classifier = _load_shared_classifier(settings)
        _logger.info("Turn runtime encoder ready")
    else:
        _logger.info("Skipping model load (LOAD_MODEL=false)")
    app.state.ready = True
    try:
        yield
    finally:
        app.state.ready = False
        app.state.classifier = None
        _logger.info("Turn runtime shut down")
