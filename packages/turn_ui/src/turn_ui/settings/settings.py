"""Runtime configuration for the microphone UI."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """UI settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    """Ignore unknown environment variables when loading UI settings."""
    application_name: str = "End-of-turn detector"
    """Human-readable application name exposed by FastAPI."""
    application_description: str = (
        "Browser UI for live speaking / hold / eot on the human side of a voice-agent call."
    )
    """Application description exposed by FastAPI."""
    host: str = Field(default="127.0.0.1", validation_alias="HOST")
    """Bind address used by the Uvicorn server."""
    port: int = Field(default=8765, validation_alias="PORT")
    """TCP port used by the UI server."""
    runtime_url: str = Field(
        default="http://127.0.0.1:8766",
        validation_alias="RUNTIME_URL",
    )
    """Base URL of the turn runtime FastAPI (no trailing slash required)."""

    @property
    def runtime_base_url(self) -> str:
        """Runtime origin without a trailing slash."""
        return self.runtime_url.rstrip("/")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached UI settings loaded from environment."""
    return Settings()
