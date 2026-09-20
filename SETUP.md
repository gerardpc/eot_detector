# Setup

Python 3.11–3.13. From the repo root:

```bash
uv venv
source .venv/bin/activate
uv sync --all-packages
uv run pre-commit install
uv run --package turn-runtime python -m turn_runtime.download
```

That last command snapshots Whisper-tiny into gitignored
`models/whisper-tiny/`. Optional env overrides: [`.env.example`](.env.example).

`HOST` and `PORT` are per process. Do not set them in a shared `.env` if
you run both the runtime and the UI from this directory.

## Live UI

```bash
uv run --package turn-runtime turn-runtime-serve
uv run --package turn-ui turn-ui
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) and click **Start listening**.
The page defaults to the runtime at [http://127.0.0.1:8766](http://127.0.0.1:8766)
(`RUNTIME_URL`, overridable on the page). The runtime loads Whisper-tiny and
`current` at startup. The Head menu lists runs under `models/pause_head/`.

To point the UI at a container, set `RUNTIME_URL` or the Runtime field to
`http://<host>:<port>` (see [Docker](#docker)). How the two processes
relate is in [`ARCHITECTURE.md`](ARCHITECTURE.md#serving).

## File CLI

```bash
uv run --package turn-runtime turn-runtime path/to/audio.wav
```

Prints JSON with `state`, `silence_seconds`, `rms`, and `p_eot`. Same VAD +
head path in-process; it does not call HTTP.

## Data and training

```bash
uv run --package eot-ml-core eot-prepare
uv run --package eot-ml-core eot-train
uv run --package eot-ml-core tensorboard --logdir models/pause_head
```
These commands do **not** all download the same thing.

1. **Whisper-tiny** (the frozen encoder) is *not* fetched by `eot-prepare`.
   The command at the top of this file writes it to `models/whisper-tiny/`.
   If that snapshot is missing, `eot-train` downloads it before it starts.
2. **The training set** *is* fetched by `eot-prepare`: it pulls
   [`livekit/eot-bench-data`](https://huggingface.co/datasets/livekit/eot-bench-data)
   into `data/raw_dataset/` (Hugging Face cache, first run is large) and
   writes labeled clips to `data/train_dataset/`.
3. **`eot-train`** reads those clips, encodes them with Whisper-tiny, and
   trains only the MLP. It does not download the dataset again.

`eot-train` writes `models/pause_head/<run_id>/` (`best.pt`, `config.json`,
TensorBoard logs) and points `models/pause_head/current` at that run.
Start the runtime again, or pick the run in the Head menu, to load the
new weights.

Clip labels and the train/val split are described in
[`ARCHITECTURE.md`](ARCHITECTURE.md#dataset).

```mermaid
flowchart LR
  hf["livekit/eot-bench-data"] --> prepare["eot-prepare"]
  prepare --> clips["data/train_dataset/<br/>hold / eot clips + index.csv"]
  clips --> train["eot-train<br/>encoder frozen"]
  train --> run["models/pause_head/run_id/<br/>best.pt + config.json"]
  run --> current["current symlink"]
  current --> serve["turn-runtime"]
```

## Docker

A standalone image for `turn-runtime` lives in
[`docker/turn_runtime`](docker/turn_runtime/README.md). It bakes Whisper-tiny
and `models/pause_head/current/best.pt` into the image. Do not build until the
encoder snapshot and a trained `current` head exist on disk.

```bash
docker build -t eot-runtime:local -f docker/turn_runtime/Dockerfile .
docker run --rm -p 8766:8766 eot-runtime:local
```

Then run `turn-ui` as usual and set Runtime to `http://127.0.0.1:8766` (or the
machine IP if the UI is elsewhere). The image listens on `0.0.0.0:8766` and
allows any browser origin.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

`data/` and `models/` are gitignored. Package internals live in
`packages/<name>/src/<pkg>/README.md`.
