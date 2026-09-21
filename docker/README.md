# Docker

Container image for the end-of-turn runtime. The UI stays on the host (or
another process); point it at the published port with `RUNTIME_URL` or the
Model API IP field.

## Contents

- [`turn_runtime/`](turn_runtime/README.md): FastAPI runtime with Whisper-tiny
  and Linear EMA 400 ms (`20260921-084831`) baked in as `current`

## Training artifacts

Training writes `models/pause_head/<run_id>/best.pt` and can retarget
`models/pause_head/current`. The runtime image does **not** follow that
symlink. It copies:

- `models/whisper-tiny/` — frozen encoder snapshot
- `models/pause_head/20260921-084831/{best.pt,config.json}` — into
  `/models/pause_head/current/` (Linear, EMA 400 ms)

Those paths are gitignored. Before building:

```bash
uv run --package turn-runtime python -m turn_runtime.download
# need models/pause_head/20260921-084831/ already, or:
uv run --package eot-ml-core eot-train --head linear --pool ema --pool-ms 400 --epochs 100 --no-promote
```

Plain `eot-train` (tail 1 s) does not create the pinned run id.
