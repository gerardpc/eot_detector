# `turn_runtime` Docker image

Standalone FastAPI image for live end-of-turn detection on human audio.
Whisper-tiny and the current pause-head `best.pt` are copied into the image at
build time.

## Prerequisites

From the repository root, before you build:

```bash
uv run --package turn-runtime python -m turn_runtime.download
uv run --package eot-ml-core eot-train
```

`models/whisper-tiny/model.safetensors` and
`models/pause_head/current/best.pt` must exist. The Dockerfile copies those
paths; a missing file fails the build.

## Build

Run from the repository root (do not build until you intend to):

```bash
docker build -t eot-runtime:local -f docker/turn_runtime/Dockerfile .
```

## Local run

```bash
docker run --rm -p 8766:8766 eot-runtime:local
```

The process binds `0.0.0.0:8766` inside the container. Map that to the host
with `-p 8766:8766`.

Point the UI at the published address:

```bash
RUNTIME_URL=http://127.0.0.1:8766 uv run --package turn-ui turn-ui
```

If the UI runs on another machine, use the host’s reachable IP instead of
`127.0.0.1`, for example `http://192.168.1.10:8766`. You can also type that
origin into the Model API IP field on the page without restarting the UI.

The image sets `CORS_ALLOWED_ORIGINS=*` so any browser origin can call `/heads`
and `/ws`. Override if you want a tighter list:

```bash
docker run --rm -p 8766:8766 \
  -e CORS_ALLOWED_ORIGINS=http://127.0.0.1:8765,http://localhost:8765 \
  eot-runtime:local
```

## Layout inside the image

```text
/models/whisper-tiny/          frozen encoder snapshot
/models/pause_head/current/    best.pt + config.json
```

`WHISPER_MODEL_DIR` and `PAUSE_HEAD_DIR` point at those paths. The Head menu
only lists what was baked in (`current`). Mount a different `pause_head` tree
if you want extra runs without rebuilding:

```bash
docker run --rm -p 8766:8766 \
  -v /path/to/pause_head:/models/pause_head \
  eot-runtime:local
```

## Remote run

- Publish port `8766` (or set `PORT` and map that).
- Give clients `http://<host>:<port>` as `RUNTIME_URL`.
- Keep `HOST=0.0.0.0` so the process accepts traffic from outside the container.
- CPU is enough for Whisper-tiny plus the linear pause head; no GPU flag is required.
