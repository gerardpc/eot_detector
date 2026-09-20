"""Download the local Whisper-tiny encoder snapshot."""

from __future__ import annotations

from logging import getLogger
from pathlib import Path

_logger = getLogger(__name__)
"""Logger for Hugging Face snapshot download."""

MODEL_ID = "openai/whisper-tiny"
"""Hugging Face repo id for the frozen encoder snapshot."""
ALLOW_PATTERNS = ("config.json", "model.safetensors", "preprocessor_config.json")
"""Files copied from the Hub; decoder weights are omitted."""


def default_model_dir() -> Path:
    """Return `models/whisper-tiny/` at the repository root."""
    return Path(__file__).resolve().parents[4] / "models" / "whisper-tiny"


def download_whisper_tiny(model_dir: Path | None = None) -> Path:
    """Snapshot encoder weights from Hugging Face into `model_dir`."""
    from huggingface_hub import snapshot_download

    dest = model_dir or default_model_dir()
    dest.mkdir(parents=True, exist_ok=True)
    _logger.info("downloading %s into %s", MODEL_ID, dest)
    snapshot_download(
        MODEL_ID,
        local_dir=dest,
        allow_patterns=list(ALLOW_PATTERNS),
    )
    _logger.info("Whisper-tiny snapshot ready at %s", dest)
    return dest


if __name__ == "__main__":
    from turn_runtime.config.logging_setup import setup_logging

    setup_logging()
    download_whisper_tiny()
