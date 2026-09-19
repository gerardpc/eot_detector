# Benchmark adapter

This package follows the public adapter contract from
[LiveKit's `eot-bench`](https://github.com/livekit/eot-bench). It does not copy the
harness's policy sweep or metrics.

Install the optional harness dependency:

```bash
uv sync --package happy-robot-eot-bench --extra harness
```

Run the validation split for every benchmark language:

```bash
uv run --package happy-robot-eot-bench --extra harness happy-robot-eot-bench \
  predict --name all --split validation
```

The official harness creates causal audio snapshots at each silence point. The
adapter puts that mono snapshot into DualTurn's user channel and supplies a silent
agent channel, then returns the last-frame user `eot_probs` value.

Useful options:

```bash
happy-robot-eot-bench predict --name en --limit 10
happy-robot-eot-bench predict --name all --split validation --batch-size 1
```

The default cache and output location is `data/` in the repository root. The
benchmark dataset is not downloaded by package installation.

