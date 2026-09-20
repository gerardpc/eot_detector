# Packages

| Package | PyPI name | Role |
| --- | --- | --- |
| [`turn_runtime/`](turn_runtime/README.md) | `turn-runtime` | Human-side EOT: energy VAD, frozen Whisper-tiny, pause head, FastAPI, WAV CLI |
| [`turn_ui/`](turn_ui/README.md) | `turn-ui` | Browser UI (talks to the runtime) |
| [`ml_core/`](ml_core/README.md) | `eot-ml-core` | Download `eot-bench-data`, cut clips, train the pause head |

FastAPI services follow the same layout: `app.py` factory, `settings/`,
`config/lifespan.py` where a model is loaded, and `routers/`. Internals:

- [`turn_runtime/src/turn_runtime/README.md`](turn_runtime/src/turn_runtime/README.md)
- [`turn_ui/src/turn_ui/README.md`](turn_ui/src/turn_ui/README.md)
- [`ml_core/src/ml_core/README.md`](ml_core/src/ml_core/README.md)
