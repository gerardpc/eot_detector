# Docker

Container definitions for deploying the end-of-turn runtime as a standalone
image.

## Current contents

- [`turn_runtime/`](turn_runtime/README.md): FastAPI runtime with Whisper-tiny
  and the current `best.pt` pause head baked in.

The UI stays on the host (or a separate process). Point it at the published
runtime port with `RUNTIME_URL` or the Runtime field on the page.

## Relation with training artifacts

Training writes `models/pause_head/<run_id>/best.pt` and points
`models/pause_head/current` at that run. The runtime image copies:

- `models/whisper-tiny/` — frozen encoder snapshot
- `models/pause_head/current/best.pt` and `config.json` — serving head

Those paths are gitignored. Build from a machine that already ran
`python -m turn_runtime.download` and `eot-train`.
