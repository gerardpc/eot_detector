"""Download eot-bench-data and write labeled pause clips."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .samples import CONTEXT_SECONDS, MIN_PAUSE_SECONDS, clip_window, pause_cut_times

DATASET_ID = "livekit/eot-bench-data"
"""Hugging Face dataset used to cut homemade hold/eot clips."""
TARGET_SR = 16_000
"""Sample rate of written training WAVs."""
HOLD_LABEL = 0
"""Class id for mid-turn hold pauses."""
EOT_LABEL = 1
"""Class id for end-of-turn pauses."""
DATASET_README = Path(__file__).with_name("train_dataset_README.md")
"""Template copied into `data/train_dataset/README.md` after prepare."""


def repo_root() -> Path:
    """Return the repository root (four levels above this module)."""
    return Path(__file__).resolve().parents[4]


def raw_dir() -> Path:
    """Return `data/raw_dataset/` at the repository root."""
    return repo_root() / "data" / "raw_dataset"


def train_dir() -> Path:
    """Return `data/train_dataset/` at the repository root."""
    return repo_root() / "data" / "train_dataset"


def download_raw() -> object:
    """Download the public validation split into `data/raw_dataset/`."""
    import os

    from datasets import Audio, load_dataset

    dest = raw_dir()
    dest.mkdir(parents=True, exist_ok=True)
    env = {
        "HF_HUB_CACHE": str(dest / "hub"),
        "HF_DATASETS_CACHE": str(dest / "datasets"),
    }
    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        dataset = load_dataset(
            DATASET_ID,
            "all",
            split="validation",
            cache_dir=str(dest / "datasets"),
        )
        return dataset.cast_column("audio", Audio(decode=False))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_waveform(audio_info: object) -> tuple[np.ndarray, int]:
    """Decode a datasets `audio` cell to float32 PCM and sample rate."""
    if not isinstance(audio_info, dict):
        raise TypeError(f"expected audio dict, got {type(audio_info)!r}")
    if audio_info.get("array") is not None:
        return np.asarray(audio_info["array"], dtype=np.float32), int(audio_info["sampling_rate"])
    payload = audio_info.get("bytes")
    path = audio_info.get("path")
    if payload:
        array, sample_rate = sf.read(io.BytesIO(payload), dtype="float32")
        return np.asarray(array, dtype=np.float32), int(sample_rate)
    if path:
        array, sample_rate = sf.read(path, dtype="float32")
        return np.asarray(array, dtype=np.float32), int(sample_rate)
    raise ValueError("audio column has no array, path, or bytes")


def split_for_id(turn_id: str, val_fraction: float = 0.2, seed: int = 0) -> str:
    """Hash a turn id into `'train'` or `'val'` (not an official eot-bench split)."""
    payload = turn_id if seed == 0 else f"{seed}:{turn_id}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return "val" if digest[0] < int(256 * val_fraction) else "train"


def _resample(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Downmix to mono and interpolate to `TARGET_SR`."""
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    array = array.reshape(-1)
    if sample_rate == TARGET_SR or array.size == 0:
        return array
    n = max(int(round(array.size * TARGET_SR / sample_rate)), 1)
    old = np.linspace(0.0, 1.0, array.size, endpoint=False)
    new = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(new, old, array).astype(np.float32)


def _slice(audio: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    """Slice `audio` between `start` and `end` seconds."""
    i0 = max(int(round(start * sample_rate)), 0)
    i1 = min(int(round(end * sample_rate)), audio.size)
    if i1 <= i0:
        return np.zeros(0, dtype=np.float32)
    return audio[i0:i1]


def _safe_id(turn_id: str) -> str:
    """Replace characters that are unsafe in filenames with `_`."""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in turn_id)


def prepare_dataset(dataset: object | None = None, *, split_seed: int = 0) -> Path:
    """Cut labeled clips into `data/train_dataset/` and write `index.csv`."""
    rows = dataset if dataset is not None else download_raw()
    out = train_dir()
    hold_dir = out / "hold"
    eot_dir = out / "eot"
    if hold_dir.exists():
        for path in hold_dir.glob("*.wav"):
            path.unlink()
    if eot_dir.exists():
        for path in eot_dir.glob("*.wav"):
            path.unlink()
    hold_dir.mkdir(parents=True, exist_ok=True)
    eot_dir.mkdir(parents=True, exist_ok=True)

    index_path = out / "index.csv"
    fieldnames = [
        "path",
        "label",
        "class",
        "split",
        "id",
        "language",
        "span_index",
        "t",
        "kind",
        "pause_start",
        "pause_end",
        "duration",
    ]
    records: list[dict[str, object]] = []

    for index, row in enumerate(rows, start=1):
        turn_id = str(row["id"])
        language = str(row.get("language") or "")
        audio_info = row["audio"]
        waveform, sample_rate = _load_waveform(audio_info)
        waveform = _resample(waveform, sample_rate)
        duration = float(row.get("duration") or waveform.size / TARGET_SR)
        spans = list(row.get("silence_spans") or [])
        if not spans:
            continue
        split = split_for_id(turn_id, seed=split_seed)
        last = len(spans) - 1
        for span_index, span in enumerate(spans):
            start = float(span["start"])
            end = float(span["end"])
            is_eot = span_index == last
            for t, kind in pause_cut_times(start, end, is_eot=is_eot, duration=duration):
                clip_start, clip_end = clip_window(t, duration, CONTEXT_SECONDS)
                clip = _slice(waveform, TARGET_SR, clip_start, clip_end)
                if clip.size < int(TARGET_SR * MIN_PAUSE_SECONDS):
                    continue
                label = EOT_LABEL if is_eot else HOLD_LABEL
                class_name = "eot" if is_eot else "hold"
                filename = (
                    f"{_safe_id(turn_id)}__span{span_index}__t{int(round(t * 1000))}__{kind}.wav"
                )
                relpath = f"{class_name}/{filename}"
                dest = out / relpath
                sf.write(dest, clip, TARGET_SR)
                records.append(
                    {
                        "path": relpath,
                        "label": label,
                        "class": class_name,
                        "split": split,
                        "id": turn_id,
                        "language": language,
                        "span_index": span_index,
                        "t": round(t, 4),
                        "kind": kind,
                        "pause_start": round(start, 4),
                        "pause_end": round(end, 4),
                        "duration": round(duration, 4),
                    }
                )
        if index % 100 == 0:
            print(f"processed {index} turns, {len(records)} clips")

    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "dataset_id": DATASET_ID,
                "sample_rate": TARGET_SR,
                "hold_label": HOLD_LABEL,
                "eot_label": EOT_LABEL,
                "n_clips": len(records),
                "n_hold": sum(1 for rec in records if rec["label"] == HOLD_LABEL),
                "n_eot": sum(1 for rec in records if rec["label"] == EOT_LABEL),
                "split_seed": split_seed,
                "val_fraction": 0.2,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        DATASET_README.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(f"wrote {len(records)} clips to {out}")
    return out


def main() -> None:
    """CLI entrypoint for `eot-prepare`."""
    parser = argparse.ArgumentParser(description="Download eot-bench-data and write pause clips.")
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()
    path = prepare_dataset(split_seed=args.split_seed)
    print(path)


if __name__ == "__main__":
    main()
