"""HTTP and WebSocket tests for the turn runtime FastAPI."""

import numpy as np
from fastapi.testclient import TestClient
from turn_runtime.app import create_app
from turn_runtime.settings.settings import Settings


def _client() -> TestClient:
    """Build a TestClient that skips Whisper load."""
    return TestClient(create_app(Settings(load_model=False)))


def test_health_heads_and_ws() -> None:
    with _client() as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["turn"]["classifier"] == "none"
        assert health["turn"]["head_id"] == "current"
        assert health["turn"]["ready"] is True
        version = client.get("/version").json()
        assert "version" in version
        heads = client.get("/heads").json()
        assert heads["ok"] is True
        assert heads["heads"][0]["id"] == "current"
        assert heads["selected"] == "current"
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "start", "sampleRate": 16000, "head": "current"})
            started = websocket.receive_json()
            assert started["ok"] is True
            assert started["turn"]["head_id"] == "current"
            websocket.send_bytes(np.zeros(8000, dtype=np.float32).tobytes())
            event = websocket.receive_json()
            assert event["ok"] is True
            assert event["state"] in {"hold", "eot"}


def test_ws_rejects_unknown_head() -> None:
    with _client() as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "start", "sampleRate": 16000, "head": "no-such-run"})
            payload = websocket.receive_json()
            assert payload["ok"] is False
            websocket.send_json({"type": "start", "sampleRate": 16000, "head": "current"})
            started = websocket.receive_json()
            assert started["ok"] is True
            websocket.send_json({"type": "head", "id": "no-such-run"})
            missing = websocket.receive_json()
            assert missing["ok"] is False


def test_heads_cors() -> None:
    with _client() as client:
        response = client.get("/heads", headers={"origin": "http://127.0.0.1:8765"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:8765"
        assert "X-Process-Time-Ms" in response.headers


def test_heads_cors_star() -> None:
    client = TestClient(create_app(Settings(load_model=False, cors_allowed_origins="*")))
    with client:
        response = client.get("/heads", headers={"origin": "http://example.test:8765"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") in {
            "*",
            "http://example.test:8765",
        }
