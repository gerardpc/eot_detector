"""Settings parsing tests for the turn runtime FastAPI."""

import os
from unittest.mock import patch

from turn_runtime.settings.settings import Settings


def test_cors_origins_parse_comma_string() -> None:
    settings = Settings(cors_allowed_origins="http://127.0.0.1:8765, http://localhost:8765")
    assert settings.cors_allowed_origins == [
        "http://127.0.0.1:8765",
        "http://localhost:8765",
    ]


def test_cors_star_origin() -> None:
    settings = Settings(cors_allowed_origins="*")
    assert settings.cors_allowed_origins == ["*"]


def test_cors_star_from_env() -> None:
    with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": "*"}, clear=False):
        settings = Settings(_env_file=None)
    assert settings.cors_allowed_origins == ["*"]


def test_default_bind() -> None:
    settings = Settings(load_model=False)
    assert settings.host == "127.0.0.1"
    assert settings.port == 8766
    assert settings.load_model is False
