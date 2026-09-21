# turn_runtime

Live end-of-turn runtime (`turn-runtime`).

Speech vs silence is energy VAD (20 ms frames, 100 ms minimum silence). During a
pause the runtime encodes the last ≤ 5 s with a frozen Whisper-tiny encoder,
mean-pools the last **1 s** of encoder frames, and scores `p(eot)` with a
linear `384 → 1` head. By default `p(eot) ≥ 0.5` is `eot`; otherwise `hold`.
The live WebSocket session can override that cutoff anywhere in `[0, 1]`
(`start.threshold` or `{type:"threshold", value}`). After 3 s of silence,
`eot` is forced without calling the head.

The encoder is `models/whisper-tiny/`. The head is
`models/pause_head/current/best.pt`, then `latest.pt` if `best.pt` is missing.
Without a trained run, pauses stay `hold` until the 3 s timeout.

## Serve

```bash
uv run --package turn-runtime python -m turn_runtime.download
uv run --package turn-runtime turn-runtime-serve
```

Listens on [http://127.0.0.1:8766](http://127.0.0.1:8766):

| Method | Path | Role |
| --- | --- | --- |
| `GET` | `/health` | 200 once startup finished; 503 otherwise |
| `GET` | `/version` | Installed package version |
| `GET` | `/heads` | `current` plus versioned pause-head runs |
| `POST` | `/infer` | WAV body → `p_eot` and encoder `latency_ms` |
| `WS` | `/ws` | Per-connection PCM stream |

Package internals: [`src/turn_runtime/README.md`](src/turn_runtime/README.md).
Docker: [`docker/turn_runtime`](../../docker/turn_runtime/README.md).

## File CLI

```bash
uv run --package turn-runtime turn-runtime path/to/audio.wav
```

Same VAD + head path in-process. It does not call the HTTP server.

## Encoder throughput

```bash
uv run --package turn-runtime turn-runtime-stress
uv run --package turn-runtime turn-runtime-stress --url http://127.0.0.1:8766
```

Needs `eot-prepare`. After a 2 s discarded warmup, sweeps offered clip
scores per second (`--rps`, default `1,2,4,8,max`) and writes
`data/stress_tests/<datetime>/report.md` with latency-distribution and
throughput plots. `--url` hits `POST /infer`; the default scores
in-process. Details: [`SETUP.md`](../../SETUP.md#encoder-throughput).

## Layout

| Path | Contents |
| --- | --- |
| `src/turn_runtime/app.py` | FastAPI factory and Uvicorn entrypoint |
| `src/turn_runtime/config/` | Lifespan and logging |
| `src/turn_runtime/routers/` | Health, heads, `/infer`, WebSocket |
| `src/turn_runtime/settings/` | Pydantic settings |
| `src/turn_runtime/runtime.py` | VAD and pause-state machine |
| `src/turn_runtime/whisper.py` | Frozen encoder, log-mel, resample |
| `src/turn_runtime/classifier.py` | Pause head; load/save versioned weights |
| `src/turn_runtime/cli.py` | File CLI |
| `src/turn_runtime/stress.py` | Training-clip encoder throughput |
| `src/turn_runtime/download.py` | Hugging Face snapshot into `models/whisper-tiny/` |

Training lives in [`ml_core`](../ml_core/README.md). The UI is
[`turn_ui`](../turn_ui/README.md).
