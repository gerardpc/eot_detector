# End-of-turn detector

This repository is an implementation of a real-time end-of-turn detector 
for a human's turn in a human-voice agent conversation.

The system is always in one of the following states:

- **speaking** — the human is talking
- **hold** — the human has gone quiet, but it still sounds like a mid-turn pause
- **eot** — the human has gone quiet, and it sounds like the turn is over

A downstream agent should start its turn on `eot` and stay silent otherwise.

| Doc | What it covers |
| --- | --- |
| [`SETUP.md`](SETUP.md) | Install, run the UI, train, Docker, encoder stress |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Decision path, model, data, serving |

| Package | Role |
| --- | --- |
| [`packages/turn_runtime`](packages/turn_runtime/README.md) | VAD, Whisper encoder, pause head, FastAPI, file CLI |
| [`packages/turn_ui`](packages/turn_ui/README.md) | Microphone UI (talks to the runtime) |
| [`packages/ml_core`](packages/ml_core/README.md) | Dataset clips, training, TensorBoard |
