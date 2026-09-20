"""Runtime configuration models and environment loading helpers."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from turn_runtime.classifier import DEFAULT_HEAD_DIR, DEFAULT_MODEL_DIR


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    """Ignore unknown environment variables when loading runtime settings."""
    application_name: str = "End-of-turn runtime"
    """Human-readable application name exposed by FastAPI."""
    application_description: str = (
        "End-of-turn detection on human audio: energy VAD, frozen Whisper-tiny, pause head."
    )
    """Application description exposed by FastAPI."""
    cors_allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://127.0.0.1:8765",
            "http://localhost:8765",
        ],
        validation_alias="CORS_ALLOWED_ORIGINS",
    )
    """Allowed browser origins for cross-origin frontend access."""
    host: str = Field(default="127.0.0.1", validation_alias="HOST")
    """Bind address used by the Uvicorn server."""
    port: int = Field(default=8766, validation_alias="PORT")
    """TCP port used by the API server."""
    load_model: bool = Field(default=True, validation_alias="LOAD_MODEL")
    """Load Whisper-tiny and the current pause head during lifespan startup."""
    whisper_model_dir: Path = Field(
        default=DEFAULT_MODEL_DIR,
        validation_alias="WHISPER_MODEL_DIR",
    )
    """Filesystem path to the local Whisper-tiny snapshot."""
    pause_head_dir: Path = Field(
        default=DEFAULT_HEAD_DIR,
        validation_alias="PAUSE_HEAD_DIR",
    )
    """Filesystem path to versioned pause-head runs."""

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _parse_cors_allowed_origins(
        cls,
        value: list[str] | str | None,
    ) -> list[str]:
        """Accept origin lists or comma-separated env strings."""
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [item.strip() for item in value if item.strip()]
        if not isinstance(value, str):
            raise TypeError("CORS_ALLOWED_ORIGINS must be a list or string.")
        return [item.strip() for item in value.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached runtime settings loaded from environment."""
    return Settings()
