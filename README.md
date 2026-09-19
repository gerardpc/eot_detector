# Happy Robot

An audio-native turn-taking playground built around the public DualTurn model.

This repository is intentionally split into small Python packages:

- `turn_runtime`: lazy model loading, audio normalization, and inference.
- `turn_ui`: a small FastAPI browser UI with microphone capture and live output bars.
- `eot_bench`: an adapter and runner for the official LiveKit EoT benchmark harness.

The model is not downloaded during setup. The first command that loads it downloads
the checkpoint from Hugging Face. The repository keeps model and dataset caches in
the ignored root `data/` directory when using the provided commands.

## Quick start

Install the workspace tools:

```bash
uv sync --all-packages
uv run pre-commit install
```

Start the browser UI:

```bash
uv run --package happy-robot-ui happy-robot-ui
```

Open <http://127.0.0.1:8765>. Click **Start conversation**. The browser records
the microphone as the user channel and sends a silent second channel for now. This
is useful for smoke testing; production full-duplex input should provide both user
and agent channels.

Run a local file through the model after the checkpoint is available:

```bash
uv run --package happy-robot-turn happy-robot-turn path/to/stereo.wav
```

The model expects two channels. A mono file is accepted and gets a silent agent
channel so that local experiments are easy to start.

## EoT benchmark

The benchmark package uses the official `eot-bench` harness rather than copying its
metrics. Install its optional dependency and run the full benchmark configuration:

```bash
uv sync --package happy-robot-eot-bench --extra harness
uv run --package happy-robot-eot-bench --extra harness happy-robot-eot-bench predict \
  --name all \
  --split validation
```

By default, dataset/model cache and benchmark artifacts go under `data/`. Use
`--limit` for a small smoke test. The adapter maps the benchmark's mono causal
audio to DualTurn's two-channel input and scores the user's `eot_probs` at each
causal decision point.

The benchmark data remains separate from training data. Do not fine-tune on the
official validation or test split if you want an honest benchmark comparison.

## Package workflow

Each package has its own `pyproject.toml`, tests, and source layout:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

End-to-end model execution is deliberately not part of the Mac smoke-test suite;
it requires downloading a large checkpoint and is best exercised on the later CPU
or CUDA machine.

## Handoff status

This section is the authoritative handoff for the next machine or agent.

### Implemented

- Created a Git monorepo with three installable packages under `packages/`.
- Added a lazy Python wrapper around `anyreach-ai/dualturn-qwen2.5-mimi-0.5B`.
- Added mono-to-two-channel normalization: a mono input becomes user audio plus
  a silent agent channel; two-channel input is preserved.
- Added JSON-friendly access to all public DualTurn heads: `eot_probs`,
  `hold_probs`, `bot_probs`, `bc_probs`, `vad_probs`, and `fvad_probs`.
- Added a FastAPI browser UI. Its start button loads the model, captures the
  microphone, sends short WAV snapshots, and renders the latest output values.
- Added an adapter for the official `eot-bench` pointwise contract and a CLI
  bridge that delegates dataset loading, causal slicing, policy sweeps, and
  metrics to the official harness.
- Added ignored root directories `data/` and `work/` for model caches, datasets,
  benchmark outputs, and scratch artifacts.
- Added Ruff, pre-commit configuration, `uv.lock`, and Mac-safe unit tests.

### Verified on the Mac

- `uv sync --all-packages --dev` succeeds.
- The optional official `eot-bench` dependency resolves and the adapter imports
  against the real harness.
- `ruff check .` succeeds.
- `ruff format --check .` succeeds.
- The test suite passes: 4 tests.
- The FastAPI server starts on `127.0.0.1:8765`, and the browser page renders.
- No model checkpoint or benchmark dataset was downloaded.
- The model forward pass, microphone inference, full benchmark run, and GPU
  training were not run on this Mac.

### Still missing or intentionally deferred

- Run a real DualTurn forward pass after downloading the checkpoint on Ubuntu.
- Confirm the current Hugging Face checkpoint loads with the installed current
  `transformers`/PyTorch versions.
- Test the UI with a real model and real microphone input.
- Test CUDA inference and measure latency and peak memory.
- Run a small `eot-bench` smoke test before the full validation run.
- Run the full multilingual validation benchmark and inspect its reports.
- Decide whether the silent agent channel is acceptable. The benchmark adapter
  currently uses it because `eot-bench` provides mono user audio; this is not a
  substitute for full-duplex user-plus-agent audio.
- Decide whether to fine-tune DualTurn on additional data. Do not fine-tune on
  the official validation or test split if benchmark comparability matters.
- Add a proper streaming/full-duplex input path if the target system can provide
  the agent's audio channel.
- Add production concerns such as authentication, persistent sessions, model
  warm-up, structured logging, and deployment configuration only after the
  inference path is confirmed.

### Ubuntu handoff checklist

From the repository root:

```bash
uv sync --all-packages --dev
uv sync --package happy-robot-eot-bench --extra harness
```

Run the smallest model smoke test with a local stereo WAV:

```bash
DUALTURN_DEVICE=cuda \
uv run --package happy-robot-turn happy-robot-turn path/to/stereo.wav
```

Run the browser UI:

```bash
DUALTURN_DEVICE=cuda \
uv run --package happy-robot-ui happy-robot-ui
```

Run a small benchmark sample first:

```bash
DUALTURN_DEVICE=cuda \
uv run --package happy-robot-eot-bench --extra harness \
  happy-robot-eot-bench predict --name en --split validation --limit 10
```

Then run all benchmark languages:

```bash
DUALTURN_DEVICE=cuda \
uv run --package happy-robot-eot-bench --extra harness \
  happy-robot-eot-bench predict --name all --split validation
```

Keep all downloaded files and generated reports under `data/`. Before changing
the adapter or model wrapper, first record the checkpoint revision, dependency
versions, device, and the exact command used for the baseline run.
