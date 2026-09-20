"""HTTP tests for the microphone UI FastAPI."""

from fastapi.testclient import TestClient
from turn_ui.app import create_app
from turn_ui.settings.settings import Settings


def test_index_and_health() -> None:
    client = TestClient(create_app(Settings(runtime_url="http://127.0.0.1:8766/")))
    page = client.get("/")
    assert page.status_code == 200
    html = page.text
    assert "End of turn" in html
    assert "state probability" in html
    assert 'id="head"' in html
    assert 'id="runtime-url"' in html
    assert "DEFAULT_RUNTIME_URL" in html
    assert "Speaking" in html
    assert "Hold" in html
    assert "EOT" in html
    assert "DualTurn" not in html
    assert "Listen for the handoff" not in html
    assert 'id="timeline"' in html
    assert "__RUNTIME_URL__" not in html
    assert "http://127.0.0.1:8766" in html
    assert "runtimeHttp('/heads')" in html
    assert "runtimeWs('/ws')" in html
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["runtime_url"] == "http://127.0.0.1:8766"
    assert "version" in payload
    assert "X-Process-Time-Ms" in page.headers
