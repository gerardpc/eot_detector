"""Replay training clips to measure pause-head encoder throughput.

The CLI sweeps offered request rates, records per-clip latency, and writes a
markdown report with plots under `data/stress_tests/<datetime>/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from logging import getLogger
from pathlib import Path

import numpy as np
import soundfile as sf

from turn_runtime.classifier import DEFAULT_MODEL_DIR, PauseClassifier
from turn_runtime.config.logging_setup import setup_logging
from turn_runtime.download import download_whisper_tiny

_logger = getLogger(__name__)
"""Logger for encoder download, stage progress, and the report path."""

DEFAULT_TRAIN_DIR = Path(__file__).resolve().parents[4] / "data" / "train_dataset"
"""Repository-relative labeled clips written by `eot-prepare`."""
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[4] / "data" / "stress_tests"
"""Parent directory for timestamped report folders."""
DEFAULT_SECONDS = 10.0
"""How long each offered-rate stage lasts when `--seconds` is omitted."""
DEFAULT_CLIPS = 128
"""How many training WAVs to preload when `--clips` is omitted."""
DEFAULT_WORKERS = 8
"""Max in-flight scores. The encoder lock still serializes the forward."""
DEFAULT_RPS = "1,2,4,8,max"
"""Offered rates; `max` is a closed-loop saturation stage."""


def _train_index(root: Path) -> Path:
    """Return `index.csv` under a train-dataset directory."""
    return root / "index.csv"


def load_clip_paths(root: Path, limit: int, seed: int) -> list[Path]:
    """Sample up to `limit` clip paths from `index.csv`."""
    index = _train_index(root)
    if not index.is_file():
        raise FileNotFoundError(
            f"No clips at {index}. Run `eot-prepare` first "
            "(it writes data/train_dataset/ from livekit/eot-bench-data)."
        )
    with index.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{index} has no rows")
    rng = random.Random(seed)
    rng.shuffle(rows)
    chosen = rows[: max(1, limit)]
    return [root / row["path"] for row in chosen]


def load_waveforms(paths: list[Path]) -> list[tuple[np.ndarray, int, bytes]]:
    """Read WAVs into memory: PCM, sample rate, and original file bytes."""
    loaded: list[tuple[np.ndarray, int, bytes]] = []
    for path in paths:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        pcm = np.asarray(audio[:, 0], dtype=np.float32)
        loaded.append((pcm, int(sample_rate), path.read_bytes()))
    return loaded


def parse_rps_list(raw: str) -> list[float | None]:
    """Parse `--rps`, where `max` means closed-loop saturation."""
    values: list[float | None] = []
    for part in raw.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in {"max", "saturate"}:
            values.append(None)
            continue
        offered = float(token)
        if offered <= 0:
            raise ValueError(f"offered rps must be positive, got {token!r}")
        values.append(offered)
    if not values:
        raise ValueError("rps list is empty")
    return values


def rps_label(offered_rps: float | None) -> str:
    """Stable stage name for tables and plot ticks."""
    if offered_rps is None:
        return "max"
    if offered_rps == int(offered_rps):
        return str(int(offered_rps))
    return str(offered_rps)


def _percentile(values: list[float], q: float) -> float:
    """Return the `q` percentile of `values`."""
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean, or NaN if `values` is empty."""
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _ensure_encoder() -> None:
    """Download Whisper-tiny if the local snapshot is missing."""
    if not (DEFAULT_MODEL_DIR / "model.safetensors").is_file():
        _logger.info("Whisper-tiny missing at %s; downloading", DEFAULT_MODEL_DIR)
        download_whisper_tiny(DEFAULT_MODEL_DIR)


def _score_in_process(
    classifier: PauseClassifier,
    lock: threading.Lock,
    pcm: np.ndarray,
    sample_rate: int,
) -> float:
    """Return p(eot) for one preloaded clip under `lock`."""
    with lock:
        return float(classifier.p_eot(pcm, sample_rate))


def _score_http(url: str, wav_bytes: bytes) -> dict[str, object]:
    """POST a WAV to `{url}/infer` and return the JSON body."""
    request = urllib.request.Request(
        url.rstrip("/") + "/infer",
        data=wav_bytes,
        method="POST",
        headers={"Content-Type": "audio/wav"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from /infer: {body}") from exc
    return payload


@dataclass
class StageResult:
    """Latency samples and throughput for one offered-rate stage."""

    label: str
    offered_rps: float | None
    achieved_rps: float
    workers: int
    forwards: int
    seconds: float
    wall_ms: list[float] = field(default_factory=list)
    encoder_ms: list[float] = field(default_factory=list)

    def stats(self) -> dict[str, object]:
        """JSON-safe summary without the raw sample lists."""
        return {
            "label": self.label,
            "offered_rps": self.offered_rps,
            "achieved_rps": round(self.achieved_rps, 3),
            "workers": self.workers,
            "forwards": self.forwards,
            "seconds": round(self.seconds, 3),
            "latency_ms_mean": round(_mean(self.wall_ms), 2),
            "latency_ms_p50": round(_percentile(self.wall_ms, 50), 2),
            "latency_ms_p95": round(_percentile(self.wall_ms, 95), 2),
            "latency_ms_p99": round(_percentile(self.wall_ms, 99), 2),
            "encoder_latency_ms_mean": round(_mean(self.encoder_ms), 2),
            "encoder_latency_ms_p50": round(_percentile(self.encoder_ms, 50), 2),
            "encoder_latency_ms_p95": round(_percentile(self.encoder_ms, 95), 2),
            "encoder_latency_ms_p99": round(_percentile(self.encoder_ms, 99), 2),
        }


@dataclass
class StressRun:
    """One stress session: shared clips, then one stage per offered rate."""

    meta: dict[str, object]
    stages: list[StageResult]


def run_stage(
    *,
    waveforms: list[tuple[np.ndarray, int, bytes]],
    classifier: PauseClassifier | None,
    lock: threading.Lock,
    url: str | None,
    offered_rps: float | None,
    seconds: float,
    workers: int,
) -> StageResult:
    """Run one offered-rate stage against preloaded clips."""

    def _one(index: int) -> tuple[float, float]:
        """Score one clip; return (client wall ms, encoder latency ms)."""
        pcm, sample_rate, wav_bytes = waveforms[index % len(waveforms)]
        started = time.perf_counter()
        if classifier is not None:
            _score_in_process(classifier, lock, pcm, sample_rate)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return elapsed_ms, elapsed_ms
        assert url is not None
        payload = _score_http(url, wav_bytes)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        reported = payload.get("latency_ms")
        encoder_latency = float(reported) if reported is not None else elapsed_ms
        return elapsed_ms, encoder_latency

    n_workers = max(1, workers)
    wall_ms: list[float] = []
    encoder_ms: list[float] = []
    started = time.perf_counter()
    deadline = started + seconds
    interval = None if offered_rps is None else 1.0 / offered_rps
    next_start = started
    next_index = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        pending: set[Future[tuple[float, float]]] = set()
        while True:
            now = time.perf_counter()
            if offered_rps is None:
                while len(pending) < n_workers and now < deadline:
                    pending.add(pool.submit(_one, next_index))
                    next_index += 1
            elif interval is not None:
                while now < deadline and now >= next_start and len(pending) < n_workers:
                    pending.add(pool.submit(_one, next_index))
                    next_index += 1
                    next_start += interval
                    now = time.perf_counter()
            if not pending:
                if now >= deadline:
                    break
                sleep_for = deadline - now
                if next_start > now:
                    sleep_for = min(sleep_for, next_start - now)
                time.sleep(max(sleep_for, 0.0))
                continue
            timeout = 0.05
            if offered_rps is not None and now < deadline and next_start > now:
                timeout = min(timeout, next_start - now)
            done, pending_f = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
            pending = set(pending_f)
            for future in done:
                client_ms, model_ms = future.result()
                wall_ms.append(client_ms)
                encoder_ms.append(model_ms)

    wall = max(time.perf_counter() - started, 1e-9)
    return StageResult(
        label=rps_label(offered_rps),
        offered_rps=offered_rps,
        achieved_rps=len(wall_ms) / wall,
        workers=n_workers,
        forwards=len(wall_ms),
        seconds=wall,
        wall_ms=wall_ms,
        encoder_ms=encoder_ms,
    )


def run_stress(
    *,
    dataset_dir: Path,
    seconds: float,
    clips: int,
    workers: int,
    seed: int,
    url: str | None,
    rps: list[float | None],
) -> StressRun:
    """Load training clips once, then run one stage per offered rate."""
    paths = load_clip_paths(dataset_dir, clips, seed)
    waveforms = load_waveforms(paths)
    lock = threading.Lock()
    classifier: PauseClassifier | None = None
    if url is None:
        _ensure_encoder()
        classifier = PauseClassifier(load_trained_head=True)
    stages: list[StageResult] = []
    for offered in rps:
        _logger.info(
            "starting stage offered_rps=%s workers=%s seconds=%s",
            rps_label(offered),
            max(1, workers),
            seconds,
        )
        stage = run_stage(
            waveforms=waveforms,
            classifier=classifier,
            lock=lock,
            url=url,
            offered_rps=offered,
            seconds=seconds,
            workers=workers,
        )
        _logger.info(
            "finished stage offered_rps=%s achieved_rps=%.3f forwards=%s "
            "latency_ms_p50=%.2f encoder_latency_ms_p50=%.2f",
            stage.label,
            stage.achieved_rps,
            stage.forwards,
            _percentile(stage.wall_ms, 50),
            _percentile(stage.encoder_ms, 50),
        )
        stages.append(stage)
    device = str(getattr(classifier, "device", "http")) if classifier is not None else "http"
    meta: dict[str, object] = {
        "mode": "http" if url else "in-process",
        "url": url,
        "device": device,
        "dataset_dir": str(dataset_dir),
        "clips_preloaded": len(waveforms),
        "workers": max(1, workers),
        "seconds_per_stage": seconds,
        "seed": seed,
        "rps": [rps_label(value) for value in rps],
    }
    return StressRun(meta=meta, stages=stages)


def _pyplot():
    """Import pyplot with a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_latency_distribution(stages: list[StageResult], path: Path) -> None:
    """Box plots of encoder latency, one box per offered rate."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [stage.encoder_ms for stage in stages if stage.encoder_ms]
    labels = [stage.label for stage in stages if stage.encoder_ms]
    if not data:
        ax.set_title("No latency samples")
    else:
        ax.boxplot(data, positions=list(range(1, len(data) + 1)), showfliers=True)
        ax.set_xticks(list(range(1, len(data) + 1)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Offered RPS (`max` = closed-loop saturation)")
        ax.set_ylabel("Encoder latency (ms)")
        ax.set_title("Encoder latency distribution per offered RPS")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_throughput(stages: list[StageResult], path: Path) -> None:
    """Offered vs achieved RPS, and latency percentiles vs offered RPS."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    offered = [float(stage.offered_rps) for stage in stages if stage.offered_rps is not None]
    achieved_open = [stage.achieved_rps for stage in stages if stage.offered_rps is not None]
    ax = axes[0]
    if offered:
        ax.plot(offered, achieved_open, marker="o", label="achieved")
        top = max(max(offered), max(achieved_open, default=0.0), 1.0)
        ax.plot([0, top], [0, top], linestyle="--", color="gray", label="offered = achieved")
    saturate = next((stage for stage in stages if stage.offered_rps is None), None)
    if saturate is not None:
        ax.axhline(saturate.achieved_rps, color="C1", linestyle=":", label="max achieved")
    ax.set_xlabel("Offered RPS")
    ax.set_ylabel("Achieved forwards / s")
    ax.set_title("Throughput")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    x_p50: list[float] = []
    p50: list[float] = []
    p99: list[float] = []
    for stage in stages:
        if not stage.encoder_ms:
            continue
        x_p50.append(stage.offered_rps if stage.offered_rps is not None else stage.achieved_rps)
        p50.append(_percentile(stage.encoder_ms, 50))
        p99.append(_percentile(stage.encoder_ms, 99))
    if x_p50:
        ax.plot(x_p50, p50, marker="o", label="p50")
        ax.plot(x_p50, p99, marker="o", label="p99")
    ax.set_xlabel("Offered RPS (`max` plotted at achieved)")
    ax.set_ylabel("Encoder latency (ms)")
    ax.set_title("Latency vs rate")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _markdown_table(stages: list[StageResult]) -> str:
    """Render the summary table for `report.md`."""
    header = (
        "| offered rps | achieved rps | forwards | seconds |"
        " latency p50 (ms) | latency p95 | latency p99 |"
        " encoder p50 (ms) | encoder p99 |"
    )
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, sep]
    for stage in stages:
        stats = stage.stats()
        offered = stats["label"]
        rows.append(
            f"| {offered} | {stats['achieved_rps']} | {stats['forwards']} |"
            f" {stats['seconds']} | {stats['latency_ms_p50']} |"
            f" {stats['latency_ms_p95']} | {stats['latency_ms_p99']} |"
            f" {stats['encoder_latency_ms_p50']} | {stats['encoder_latency_ms_p99']} |"
        )
    return "\n".join(rows)


def write_report(run: StressRun, output_dir: Path) -> Path:
    """Write markdown, plots, JSON, and CSV under `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution = output_dir / "latency_distribution.png"
    throughput = output_dir / "throughput.png"
    _plot_latency_distribution(run.stages, distribution)
    _plot_throughput(run.stages, throughput)
    samples_path = output_dir / "samples.csv"
    with samples_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "offered_rps", "latency_ms", "encoder_latency_ms"],
        )
        writer.writeheader()
        for stage in run.stages:
            offered = "" if stage.offered_rps is None else stage.offered_rps
            for wall, encoder in zip(stage.wall_ms, stage.encoder_ms, strict=True):
                writer.writerow(
                    {
                        "label": stage.label,
                        "offered_rps": offered,
                        "latency_ms": round(wall, 3),
                        "encoder_latency_ms": round(encoder, 3),
                    }
                )
    summary = {
        "meta": run.meta,
        "stages": [stage.stats() for stage in run.stages],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    started_at = run.meta.get("started_at", "")
    mode = run.meta.get("mode")
    device = run.meta.get("device")
    clips = run.meta.get("clips_preloaded")
    seconds = run.meta.get("seconds_per_stage")
    report = "\n".join(
        [
            f"# Encoder stress {started_at}".rstrip(),
            "",
            f"- Mode: `{mode}`",
            f"- Device: `{device}`",
            f"- URL: `{run.meta.get('url')}`",
            f"- Clips preloaded: {clips}",
            f"- Seconds per offered rate: {seconds}",
            f"- Workers (max in-flight): {run.meta.get('workers')}",
            f"- Dataset: `{run.meta.get('dataset_dir')}`",
            "",
            "Offered RPS is how many clip scores we *try* to start each second.",
            "`max` keeps `--workers` busy (closed-loop saturation). Achieved RPS is",
            "completed forwards divided by wall time. `latency_ms` is client-side",
            "(including queueing). `encoder_latency_ms` is the `/infer` `latency_ms`",
            "field, or the same as client latency for in-process runs.",
            "",
            "## Summary",
            "",
            _markdown_table(run.stages),
            "",
            "## Latency distribution per offered RPS",
            "",
            "![Encoder latency box plots](latency_distribution.png)",
            "",
            "## Throughput and percentiles",
            "",
            "![Throughput and latency vs RPS](throughput.png)",
            "",
            "Raw samples: [`samples.csv`](samples.csv). Machine-readable summary:",
            "[`summary.json`](summary.json).",
            "",
        ]
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def default_output_dir(now: datetime | None = None) -> Path:
    """Return `data/stress_tests/<YYYY-MM-DDTHH-MM-SS>`."""
    stamp = (now or datetime.now().astimezone()).strftime("%Y-%m-%dT%H-%M-%S")
    return DEFAULT_OUTPUT_ROOT / stamp


def main() -> None:
    """CLI entrypoint for `turn-runtime-stress`."""
    parser = argparse.ArgumentParser(
        description=(
            "Sweep offered RPS against training clips and write a markdown "
            "report with latency plots."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Directory with index.csv from eot-prepare.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS,
        help="Wall time per offered-rate stage.",
    )
    parser.add_argument("--clips", type=int, default=DEFAULT_CLIPS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rps",
        default=DEFAULT_RPS,
        help='Comma-separated offered rates; include "max" for saturation.',
    )
    parser.add_argument(
        "--url",
        default=None,
        help="If set, POST WAVs to {url}/infer instead of scoring in-process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report folder. Default: data/stress_tests/<datetime>.",
    )
    args = parser.parse_args()
    setup_logging()
    rps = parse_rps_list(args.rps)
    output_dir = args.output_dir or default_output_dir()
    run = run_stress(
        dataset_dir=args.dataset_dir,
        seconds=args.seconds,
        clips=args.clips,
        workers=args.workers,
        seed=args.seed,
        url=args.url,
        rps=rps,
    )
    run.meta["started_at"] = output_dir.name
    report_path = write_report(run, output_dir)
    _logger.info("wrote stress report to %s", report_path)
    for stage in run.stages:
        _logger.info("stage summary %s", json.dumps(stage.stats(), sort_keys=True))


if __name__ == "__main__":
    main()
