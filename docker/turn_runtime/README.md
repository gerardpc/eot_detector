# `turn_runtime` Docker image

Standalone FastAPI image for live end-of-turn detection on human audio.
Whisper-tiny and Linear EMA 400 ms (`20260921-084831`) are copied into
`/models/pause_head/current/` at build time.

## Prerequisites

From the repository root:

```bash
uv run --package turn-runtime python -m turn_runtime.download
```

You also need `models/pause_head/20260921-084831/{best.pt,config.json}`.
That folder is Linear + EMA 400 ms. Plain `eot-train` (default: linear +
tail 1 s) does not create it. Either keep the existing run, or:

```bash
uv run --package eot-ml-core eot-train --head linear --pool ema --pool-ms 400 --epochs 100 --no-promote
```

A missing path fails the build.

## Build

From the repository root:

```bash
docker build -t eot-runtime:local -f docker/turn_runtime/Dockerfile .
```

## Local run

```bash
docker run --rm -p 8766:8766 eot-runtime:local
```

The process binds `0.0.0.0:8766` inside the container.

Point the UI at it:

```bash
RUNTIME_URL=http://127.0.0.1:8766 uv run --package turn-ui turn-ui
```

On another machine, use the host’s reachable IP (for example
`http://192.168.1.10:8766`), or type that into the Model API IP field on
the page.

The image sets `CORS_ALLOWED_ORIGINS=*` (comma-separated origins also work).
Tighter example:

```bash
docker run --rm -p 8766:8766 \
  -e CORS_ALLOWED_ORIGINS=http://127.0.0.1:8765,http://localhost:8765 \
  eot-runtime:local
```

## Layout inside the image

```text
/models/whisper-tiny/          frozen encoder snapshot
/models/pause_head/current/    best.pt + config.json  (EMA 400 ms bake)
```

`WHISPER_MODEL_DIR` and `PAUSE_HEAD_DIR` point at those paths. The ML Model
menu only lists what was baked in (`current`). Mount a different tree for
extra runs without rebuilding:

```bash
docker run --rm -p 8766:8766 \
  -v /path/to/pause_head:/models/pause_head \
  eot-runtime:local
```

## Remote run

- Publish port `8766` (or set `PORT` and map that).
- Give clients `http://<host>:<port>` as `RUNTIME_URL`.
- Keep `HOST=0.0.0.0` so traffic from outside the container is accepted.
- CPU is enough; no GPU flag required.
