# Turn runtime

This package is a deliberately thin wrapper around the public DualTurn checkpoint:
`anyreach-ai/dualturn-qwen2.5-mimi-0.5B`.

The wrapper does not download anything at import time. Call `DualTurnRunner.load()`
or the CLI to load the model. The model output is returned as NumPy arrays with the
keys `eot_probs`, `hold_probs`, `bot_probs`, `bc_probs`, `vad_probs`, and
`fvad_probs`.

