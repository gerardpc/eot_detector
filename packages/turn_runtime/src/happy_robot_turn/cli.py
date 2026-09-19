"""Command-line inference for a local WAV file."""

from __future__ import annotations

import argparse
import json

from .model import runner_from_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DualTurn on a WAV file.")
    parser.add_argument("audio", help="Path to a mono or stereo WAV file")
    args = parser.parse_args()

    from pathlib import Path

    runner = runner_from_environment()
    prediction = runner.predict_wav_bytes(Path(args.audio).read_bytes())
    print(json.dumps({"status": runner.status(), "last_frame": prediction.last_frame()}, indent=2))
