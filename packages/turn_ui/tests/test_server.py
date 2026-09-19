from fastapi.testclient import TestClient
from happy_robot_ui.server import app


def test_index_and_health() -> None:
    client = TestClient(app)
    assert client.get("/").status_code == 200
    payload = client.get("/api/health").json()
    assert payload["ok"] is True
    assert payload["model"]["loaded"] is False
