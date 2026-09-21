"""Score pause-head runs on the val split and write `analysis/roc.json`.

Encodes each val clip once, then pools and applies each run's `best.pt`.
Positive class is eot; a false positive is an interruption (eot | hold).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from roc_math import OPERATING_THRESHOLD, operating_point, roc_curve
from style import RUNS
from turn_runtime.classifier import (
    DEFAULT_HEAD_DIR,
    PauseClassifier,
    pool_encoder_states,
    pooling_from_run,
    resolve_head_weights,
)
from turn_runtime.whisper import log_mel_spectrogram, resample_to_16k

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATASET = REPO / "data" / "train_dataset"
TARGET_SR = 16_000
OUT_PATH = HERE / "roc.json"


def _val_rows() -> list[dict[str, str]]:
    """Read val rows from `data/train_dataset/index.csv`."""
    index_path = DATASET / "index.csv"
    with index_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "val"]
    if not rows:
        raise FileNotFoundError(f"no val clips in {index_path}")
    return rows


def _encoder_frames(
    classifier: PauseClassifier, audio: np.ndarray, sample_rate: int
) -> torch.Tensor:
    """Return CPU encoder states `(time, width)` for one clip."""
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(-1))
    if waveform.numel() == 0:
        width = int(classifier.encoder.layer_norm.normalized_shape[0])
        return torch.zeros(0, width)
    waveform = resample_to_16k(waveform.to(classifier.device), int(sample_rate))
    features = log_mel_spectrogram(waveform, classifier.mel_filters).unsqueeze(0)
    with torch.no_grad():
        hidden = classifier.encoder(features)
    return hidden.squeeze(0).detach().cpu()


def _encode_val(
    classifier: PauseClassifier, rows: list[dict[str, str]]
) -> tuple[list[torch.Tensor], np.ndarray]:
    """Encode every val clip once."""
    hiddens: list[torch.Tensor] = []
    labels: list[int] = []
    for i, row in enumerate(rows, start=1):
        audio, sample_rate = sf.read(DATASET / row["path"], dtype="float32")
        if int(sample_rate) != TARGET_SR:
            raise ValueError(f"expected {TARGET_SR} Hz clips, got {sample_rate}")
        pcm = np.asarray(audio, dtype=np.float32)
        hiddens.append(_encoder_frames(classifier, pcm, sample_rate))
        labels.append(int(row["label"]))
        if i % 1000 == 0 or i == len(rows):
            print(f"encoded {i}/{len(rows)} val clips")
    return hiddens, np.asarray(labels, dtype=np.int8)


def _pooled(
    hiddens: list[torch.Tensor],
    pool: str,
    pool_ms: float,
) -> torch.Tensor:
    """Stack pooled 384-d vectors for one pooling config."""
    vectors = [
        pool_encoder_states(hidden.unsqueeze(0), pool, pool_ms=pool_ms).squeeze(0)
        for hidden in hiddens
    ]
    return torch.stack(vectors)


def score() -> Path:
    """Write `roc.json` with val labels and per-run p(eot) from `best.pt`."""
    rows = _val_rows()
    classifier = PauseClassifier(load_trained_head=False)
    hiddens, labels = _encode_val(classifier, rows)
    pool_cache: dict[tuple[str, float], torch.Tensor] = {}
    runs_out: list[dict] = []
    for run_id, label in RUNS:
        run_dir = DEFAULT_HEAD_DIR / run_id
        weights = resolve_head_weights(run_dir)
        if weights is None:
            raise FileNotFoundError(f"no best.pt/latest.pt in {run_dir}")
        pooling = pooling_from_run(run_dir)
        if pooling is None:
            raise FileNotFoundError(f"no config.json in {run_dir}")
        pool, pool_ms = pooling
        cache_key = (pool, pool_ms)
        if cache_key not in pool_cache:
            print(f"pooling {pool} {pool_ms:g} ms")
            pool_cache[cache_key] = _pooled(hiddens, pool, pool_ms)
        if not classifier.load_head(weights):
            raise RuntimeError(f"could not load {weights}")
        with torch.no_grad():
            logits = classifier.head(pool_cache[cache_key].to(classifier.device)).squeeze(-1)
            scores = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
        _, _, auc = roc_curve(labels, scores)
        fpr_op, tpr_op = operating_point(labels, scores)
        print(
            f"{label}: AUC={auc:.4f}  "
            f"FPR@{OPERATING_THRESHOLD:g}={fpr_op:.3f}  TPR@{OPERATING_THRESHOLD:g}={tpr_op:.3f}"
        )
        runs_out.append(
            {
                "id": run_id,
                "label": label,
                "checkpoint": weights.name,
                "auc": auc,
                "fpr_at_0.5": fpr_op,
                "tpr_at_0.5": tpr_op,
                "scores": [round(float(value), 6) for value in scores],
            }
        )
    payload = {
        "n_val": int(labels.size),
        "n_hold": int((labels == 0).sum()),
        "n_eot": int((labels == 1).sum()),
        "positive_class": "eot",
        "false_positive": "interruption (eot predicted, hold true)",
        "labels": [int(value) for value in labels],
        "runs": runs_out,
    }
    OUT_PATH.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    score()
