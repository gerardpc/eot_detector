"""Download the local Whisper-tiny encoder snapshot."""

from __future__ import annotations

from pathlib import Path

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
    snapshot_download(
        MODEL_ID,
        local_dir=dest,
        allow_patterns=list(ALLOW_PATTERNS),
    )
    return dest


if __name__ == "__main__":
    path = download_whisper_tiny()
    print(path)
