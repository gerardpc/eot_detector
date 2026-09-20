"""Command-line VAD pass over a local WAV file."""

from __future__ import annotations

import argparse
import json
from logging import getLogger

from turn_runtime.config.logging_setup import setup_logging
from turn_runtime.runtime import runner_from_environment

_logger = getLogger(__name__)
"""Logger for the file WAV CLI."""


def main() -> None:
    """Run energy VAD (and the pause head, if loaded) on one WAV file."""
    parser = argparse.ArgumentParser(description="Run energy VAD on a WAV file.")
    parser.add_argument("audio", help="Path to a mono or stereo WAV file")
    args = parser.parse_args()
    setup_logging()

    import soundfile as sf

    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=True)
    runner = runner_from_environment()
    event = runner.consume_stream(audio[:, 0], int(sample_rate))
    payload = {
        "status": runner.status(),
        "event": {
            "state": event.state,
            "silence_seconds": event.silence_seconds,
            "rms": event.rms,
            "p_eot": event.p_eot,
        },
    }
    _logger.info("%s", json.dumps(payload, indent=2))
