"""Small FastAPI server for microphone capture and live model output."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from happy_robot_turn import runner_from_environment

app = FastAPI(title="Happy Robot Turn Taking")
_APP_DIR = Path(__file__).resolve().parent
_INDEX = (_APP_DIR / "index.html").read_text(encoding="utf-8")
_RUNNER = runner_from_environment()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model": _RUNNER.status()}


@app.post("/api/model/load")
def load_model() -> dict:
    try:
        _RUNNER.load()
    except Exception as exc:  # noqa: BLE001 - expose actionable UI diagnostics.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ok": True, "model": _RUNNER.status()}


@app.post("/api/predict")
async def predict(audio: UploadFile = File(...)) -> dict:  # noqa: B008
    payload = await audio.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded audio payload is empty")
    try:
        result = _RUNNER.predict_wav_bytes(payload)
    except Exception as exc:  # noqa: BLE001 - expose actionable UI diagnostics.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "ok": True,
        "model": _RUNNER.status(),
        "last_frame": result.last_frame(),
    }


def run() -> None:
    import uvicorn

    uvicorn.run(
        "happy_robot_ui.server:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )
