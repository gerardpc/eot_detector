# `turn_runtime` package

FastAPI inference service plus the in-process VAD / Whisper / pause-head stack.

The HTTP layer is the contract for live clients. Training still imports
`PauseClassifier` directly; the file CLI does the same. Neither goes through HTTP.

## Purpose

- validate and serve live PCM over a WebSocket
- list pause-head runs on disk
- load one frozen Whisper-tiny encoder for the process
- keep VAD state per connection
- force `eot` after 3 s of silence without calling the head

## Structure

```text
turn_runtime/
├── app.py                 Application factory (`create_app`) and Uvicorn entrypoint (`run`)
├── config/
│   ├── lifespan.py        Loads encoder + current head; stores them on `app.state`
│   └── logging_setup.py   `LOG_LEVEL` console logging
├── routers/
│   ├── health.py          GET /health, GET /version
│   ├── heads.py           GET /heads
│   └── stream.py          WS /ws
├── settings/
│   └── settings.py        Pydantic settings from environment / `.env`
├── runtime.py             Energy VAD and pause-state machine
├── classifier.py          Pause head; versioned `best.pt` / `latest.pt`
├── whisper.py             Frozen encoder, log-mel, resample
├── cli.py                 File WAV CLI
└── download.py            Hugging Face snapshot into `models/whisper-tiny/`
```

## Request flow

```text
Browser (turn_ui)
        │
        ├─ GET /heads ─────────────────────────────┐
        └─ WS /ws  {start, PCM frames, head}       │
                                                   ▼
                                         turn_runtime.app
                                                   │
                         ┌─────────────────────────┼─────────────────────────┐
                         ▼                         ▼                         ▼
                   routers/heads             routers/stream            routers/health
                         │                         │
                         ▼                         ▼
                 list_pause_heads           per-socket TurnRunner
                                                   │
                                                   ├─ energy VAD (20 ms)
                                                   └─ shared PauseClassifier (locked)
```

## Lifespan and readiness

`GET /health` returns 503 until lifespan finishes. With `LOAD_MODEL=true`
(the default), missing Whisper weights fail startup. Tests pass
`Settings(load_model=False)` so the app comes up without GPU weights.

`app.state.ready` is the HTTP gate. It means the encoder path completed or was
skipped, not that a given pause head has been warmed.

## Configuration

See [`settings/settings.py`](settings/settings.py) and the repo-root
[`.env.example`](../../../../.env.example). Main knobs: `HOST`, `PORT`,
`CORS_ALLOWED_ORIGINS`, `LOAD_MODEL`, `WHISPER_MODEL_DIR`, `PAUSE_HEAD_DIR`,
`LOG_LEVEL`.
