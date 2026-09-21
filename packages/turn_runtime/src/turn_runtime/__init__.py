"""Energy VAD runtime for live end-of-turn detection on human audio."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from .classifier import (
    DEFAULT_HEAD_DIR,
    DEFAULT_MODEL_DIR,
    HEAD_LINEAR,
    HEAD_NAME,
    PauseClassifier,
    list_pause_heads,
    resolve_head_weights,
)
from .runtime import ENCODER_ID, TurnEvent, TurnRunner, runner_from_environment

try:
    __version__ = package_version("turn-runtime")
except PackageNotFoundError:
    __version__ = "0.1.0"
"""Installed `turn-runtime` version, or `0.1.0` from a source tree."""

__all__ = [
    "DEFAULT_HEAD_DIR",
    "DEFAULT_MODEL_DIR",
    "ENCODER_ID",
    "HEAD_LINEAR",
    "HEAD_NAME",
    "PauseClassifier",
    "TurnEvent",
    "TurnRunner",
    "__version__",
    "list_pause_heads",
    "resolve_head_weights",
    "runner_from_environment",
]
