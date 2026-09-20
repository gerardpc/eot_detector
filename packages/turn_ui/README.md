# turn_ui

Browser UI for the live human-side end-of-turn runtime (`turn-ui`).

The page captures microphone PCM and talks to `turn-runtime-serve` over
HTTP (`/heads`) and a WebSocket (`/ws`). This process does not load PyTorch.

## Run

```bash
uv run --package turn-runtime python -m turn_runtime.download
uv run --package turn-runtime turn-runtime-serve
uv run --package turn-ui turn-ui
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) and click **Start listening**.
The page defaults to `RUNTIME_URL` (`http://127.0.0.1:8766`). Override it with
that env var or the Runtime field (IP and port of `turn-runtime-serve`, including
a Docker host). The Head menu lists runs under `models/pause_head/`; `current`
is the default. Changing the menu while listening sends `{type:"head"}` on the
open socket.

Package internals: [`src/turn_ui/README.md`](src/turn_ui/README.md).

## Layout

| Path | Contents |
| --- | --- |
| `src/turn_ui/app.py` | FastAPI factory and Uvicorn entrypoint |
| `src/turn_ui/routers/page.py` | `GET /`, `GET /health` |
| `src/turn_ui/settings/` | HOST, PORT, RUNTIME_URL |
| `src/turn_ui/index.html` | Capture UI and scrolling state chart |
