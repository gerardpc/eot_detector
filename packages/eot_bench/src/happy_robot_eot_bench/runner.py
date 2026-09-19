"""CLI bridge to the official eot-bench command."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ADAPTER = "happy_robot_eot_bench.adapter:DualTurnEotAdapter"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DualTurn through eot-bench.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict", help="Generate causal EoT predictions")
    predict.add_argument("--path", default="livekit/eot-bench-data")
    predict.add_argument("--name", default="all")
    predict.add_argument("--split", default="validation")
    predict.add_argument("--revision")
    predict.add_argument("--output-dir", default=None)
    predict.add_argument("--batch-size", type=int, default=1)
    predict.add_argument("--inference-interval", type=float, default=0.1)
    predict.add_argument("--transcript-lag", type=float, default=0.5)
    predict.add_argument("--min-silence-span", type=float, default=0.1)
    predict.add_argument("--limit", type=int)
    predict.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.command != "predict":
        parser.error(f"Unsupported command: {args.command}")

    root = _repo_root()
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HOME", str(data_dir / "huggingface"))
    output_dir = args.output_dir or str(data_dir / "eot-bench-output")

    command = [
        sys.executable,
        "-m",
        "eot_harness",
        "predict",
        "--path",
        args.path,
        "--name",
        args.name,
        "--split",
        args.split,
        "--adapter",
        ADAPTER,
        "--output-dir",
        output_dir,
        "--batch-size",
        str(args.batch_size),
        "--inference-interval",
        str(args.inference_interval),
        "--transcript-lag",
        str(args.transcript_lag),
        "--min-silence-span",
        str(args.min_silence_span),
    ]
    if args.revision:
        command.extend(["--revision", args.revision])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, cwd=root, check=True)
