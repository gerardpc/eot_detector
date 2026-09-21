# End-of-turn detector

Real-time end-of-turn detection for the human side of a voice-agent call.

The system is always in one of these states:

- **speaking** — the human is talking
- **hold** — quiet, but still mid-turn
- **eot** — quiet, and the turn is over

A downstream agent should start speaking on `eot` and stay silent otherwise.

| Doc | What it covers |
| --- | --- |
| [`SETUP.md`](SETUP.md) | Install, UI, train, Docker, encoder stress |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Decision path, model, data, serving |

| Package | Role |
| --- | --- |
| [`packages/turn_runtime`](packages/turn_runtime/README.md) | VAD, Whisper encoder, pause head, FastAPI, file CLI |
| [`packages/turn_ui`](packages/turn_ui/README.md) | Microphone UI (talks to the runtime) |
| [`packages/ml_core`](packages/ml_core/README.md) | Dataset clips, training, TensorBoard |
