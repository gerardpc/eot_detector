"""Frozen Whisper-tiny encoder plus a tiny hold/eot head."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .whisper import (
    ENCODER_FRAME_SECONDS,
    load_encoder,
    log_mel_spectrogram,
    resample_to_16k,
)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[4] / "models" / "whisper-tiny"
"""Repository-relative directory for the Whisper-tiny snapshot."""
DEFAULT_HEAD_DIR = Path(__file__).resolve().parents[4] / "models" / "pause_head"
"""Repository-relative directory for versioned pause-head runs."""
WEIGHT_NAMES = ("best.pt", "latest.pt")
"""Checkpoint filenames preferred in order when resolving a run."""
HEAD_HIDDEN = 64
"""Hidden width of the GELU MLP pause head."""
HEAD_MLP = "linear-64-gelu-linear"
"""Architecture tag for `384 → 64 → GELU → 1`."""
HEAD_LINEAR = "linear"
"""Architecture tag for a single `384 → 1` linear layer."""
HEAD_NAME = HEAD_MLP
"""Default pause-head architecture (the GELU MLP)."""
HEAD_ARCHITECTURES = (HEAD_MLP, HEAD_LINEAR)
"""Known `config.json` `head` values."""
POOL_MEAN = "mean"
"""Average every encoder frame in the clip."""
POOL_TAIL = "tail"
"""Average only the last `pool_ms` of encoder frames."""
POOL_EMA = "ema"
"""Normalized exponential recency weights; `pool_ms` is the half-life."""
POOL_MODES = (POOL_MEAN, POOL_TAIL, POOL_EMA)
"""Known `config.json` `pool` values."""
DEFAULT_TAIL_MS = 1000.0
"""Default tail window in milliseconds (last 1 s of encoder frames)."""


def tail_frames_for_ms(pool_ms: float) -> int:
    """How many encoder steps cover `pool_ms` milliseconds."""
    if pool_ms <= 0:
        raise ValueError(f"pool_ms must be positive, got {pool_ms}")
    return max(1, int(round((pool_ms / 1000.0) / ENCODER_FRAME_SECONDS)))


def _ema_weights(
    steps: int,
    pool_ms: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return length-`steps` weights with half-life `pool_ms`, summing to 1.

    Same recency prior as `y_t = (1-α) y_{t-1} + α x_t` with
    `α = 1 - 0.5**(1/half_life_frames)`, then normalizing so short clips
    are not shrunk.
    """
    half_life = float(tail_frames_for_ms(pool_ms))
    distance = torch.arange(steps - 1, -1, -1, device=device, dtype=dtype)
    weights = torch.pow(distance.new_tensor(0.5), distance / half_life)
    return weights / weights.sum().clamp_min(1e-8)


def pool_encoder_states(
    hidden: torch.Tensor,
    pool: str,
    *,
    pool_ms: float = DEFAULT_TAIL_MS,
) -> torch.Tensor:
    """Reduce `(batch, time, width)` encoder states to `(batch, width)`."""
    if pool not in POOL_MODES:
        raise ValueError(f"unknown pooling mode {pool!r}; expected one of {POOL_MODES}")
    if hidden.ndim != 3:
        raise ValueError(f"expected (batch, time, width), got {tuple(hidden.shape)}")
    if hidden.shape[1] == 0:
        return hidden.new_zeros(hidden.shape[0], hidden.shape[2])
    if pool == POOL_MEAN:
        return hidden.mean(dim=1)
    if pool == POOL_EMA:
        weights = _ema_weights(
            hidden.shape[1],
            pool_ms,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        return (hidden * weights.view(1, -1, 1)).sum(dim=1)
    n = min(hidden.shape[1], tail_frames_for_ms(pool_ms))
    return hidden[:, -n:, :].mean(dim=1)


def build_pause_head(width: int, architecture: str = HEAD_NAME) -> nn.Sequential:
    """Return the pause-head module for `architecture`."""
    if architecture == HEAD_LINEAR:
        return nn.Sequential(nn.Linear(width, 1))
    if architecture == HEAD_MLP:
        return nn.Sequential(
            nn.Linear(width, HEAD_HIDDEN),
            nn.GELU(),
            nn.Linear(HEAD_HIDDEN, 1),
        )
    raise ValueError(
        f"unknown pause-head architecture {architecture!r}; expected one of {HEAD_ARCHITECTURES}"
    )


def architecture_from_state_dict(state: dict[str, object]) -> str | None:
    """Infer `HEAD_MLP` or `HEAD_LINEAR` from a head `state_dict`, if possible."""
    if "2.weight" in state:
        return HEAD_MLP
    if "0.weight" in state and "2.weight" not in state:
        return HEAD_LINEAR
    return None


def architecture_from_run(path: Path) -> str | None:
    """Read `head` from `config.json` next to a weights file or run directory."""
    meta = _run_meta(path)
    head = meta.get("head")
    if head in HEAD_ARCHITECTURES:
        return str(head)
    return None


def pooling_from_run(path: Path) -> tuple[str, float] | None:
    """Read `(pool, pool_ms)` from `config.json`, or None if the file is missing."""
    config_path = path if path.is_dir() else path.parent
    if not (config_path / "config.json").is_file():
        return None
    meta = _run_meta(path)
    pool = meta.get("pool", POOL_MEAN)
    if pool not in POOL_MODES:
        pool = POOL_MEAN
    try:
        pool_ms = float(meta.get("pool_ms", DEFAULT_TAIL_MS))
    except (TypeError, ValueError):
        pool_ms = DEFAULT_TAIL_MS
    if pool_ms <= 0:
        pool_ms = DEFAULT_TAIL_MS
    return str(pool), pool_ms


def _run_meta(path: Path) -> dict[str, object]:
    """Load sibling `config.json`, or `{}`."""
    directory = path if path.is_dir() else path.parent
    config_path = directory / "config.json"
    if not config_path.is_file():
        return {}
    try:
        meta = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return meta if isinstance(meta, dict) else {}


def _weights_in(directory: Path) -> Path | None:
    """Return `best.pt` or `latest.pt` if either file exists in `directory`."""
    for name in WEIGHT_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _newest_run_weights(root: Path) -> Path | None:
    """Return weights from the most recently modified run dir under `root`."""
    runs = [path for path in root.iterdir() if path.is_dir() and path.name != "current"]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for run in runs:
        found = _weights_in(run)
        if found is not None:
            return found
    return None


def resolve_head_weights(path: Path | None = None) -> Path | None:
    """Return `best.pt` (else `latest.pt`) for a run dir, store root, or a `.pt` file."""

    target = Path(path) if path is not None else DEFAULT_HEAD_DIR
    if target.is_file():
        return target
    if not target.is_dir():
        return None
    current = target / "current"
    if current.is_file():
        return current
    if current.is_dir():
        found = _weights_in(current)
        if found is not None:
            return found
    found = _weights_in(target)
    if found is not None:
        return found
    return _newest_run_weights(target)


def _run_config(run: Path) -> dict[str, object]:
    """Load `config.json` for `run`, or `{}` if missing / invalid."""
    config_path = run / "config.json"
    if not config_path.is_file():
        return {}
    try:
        meta = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return meta if isinstance(meta, dict) else {}


def _head_meta(run: Path | None) -> dict[str, object]:
    """Architecture / pooling fields used by the UI for a pause-head run."""
    meta = _run_config(run) if run is not None else {}
    head = meta.get("head")
    if not isinstance(head, str) or not head:
        head = None
    pool = meta.get("pool", POOL_MEAN)
    if pool not in POOL_MODES:
        pool = POOL_MEAN
    try:
        pool_ms = float(meta.get("pool_ms", DEFAULT_TAIL_MS))
    except (TypeError, ValueError):
        pool_ms = DEFAULT_TAIL_MS
    if pool_ms <= 0:
        pool_ms = DEFAULT_TAIL_MS
    return {"head": head, "pool": pool, "pool_ms": pool_ms}


def list_pause_heads(root: Path | None = None) -> list[dict[str, object]]:
    """List `current` plus run directories that contain `best.pt` or `latest.pt`."""
    directory = root or DEFAULT_HEAD_DIR
    current = directory / "current"
    current_label = "current"
    current_run: Path | None = None
    if current.exists():
        resolved = current.resolve()
        if resolved.name != "current":
            current_label = f"current ({resolved.name})"
            current_run = resolved if resolved.is_dir() else None
        elif current.is_dir():
            current_run = current
    items: list[dict[str, object]] = [
        {"id": "current", "label": current_label, **_head_meta(current_run)}
    ]
    if not directory.is_dir():
        return items
    runs = [path for path in directory.iterdir() if path.is_dir() and path.name != "current"]
    runs.sort(key=lambda path: path.name, reverse=True)
    for run in runs:
        if _weights_in(run) is None:
            continue
        label = run.name
        meta = _run_config(run)
        best = meta.get("best_val_loss")
        if isinstance(best, int | float):
            label = f"{run.name}  val {float(best):.3f}"
        fields = _head_meta(run)
        arch = fields.get("head")
        if isinstance(arch, str) and arch:
            label = f"{label}  {arch}"
        pool = fields.get("pool")
        pool_ms = fields.get("pool_ms")
        if pool in {POOL_TAIL, POOL_EMA} and isinstance(pool_ms, int | float):
            label = f"{label}  {pool}-{int(pool_ms)}ms"
        elif pool == POOL_MEAN:
            label = f"{label}  mean"
        items.append({"id": run.name, "label": label, **fields})
    return items


class PauseClassifier(nn.Module):
    """Map the last ≤5 s of pause-context audio to p(eot)."""

    def __init__(
        self,
        model_dir: Path | None = None,
        device: str | None = None,
        *,
        load_trained_head: bool = True,
        head: str = HEAD_NAME,
        pool: str = POOL_MEAN,
        pool_ms: float = DEFAULT_TAIL_MS,
    ) -> None:
        """Load the frozen encoder and optionally the on-disk pause head.

        Args:
            model_dir: Whisper-tiny snapshot directory.
            device: Torch device; defaults to CUDA when available.
            load_trained_head: Load `best.pt`/`latest.pt` from the head store.
            head: Architecture to build before load (`linear-64-gelu-linear`
                or `linear`). `load_head` rebuilds this if the checkpoint
                was trained with the other one.
            pool: `mean` over the whole clip, `tail` over the last `pool_ms`
                (the serving strategy), or `ema` with half-life `pool_ms`.
            pool_ms: Window in milliseconds for `tail` / `ema` (default 1000).
        """
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder, self.mel_filters = load_encoder(model_dir or DEFAULT_MODEL_DIR, self.device)
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.eval()
        self._encoder_width = int(self.encoder.layer_norm.normalized_shape[0])
        self.head_name = head
        self._set_pool(pool, pool_ms)
        self.head = build_pause_head(self._encoder_width, head)
        self._init_head()
        self.head.to(self.device)
        self.eval()
        self._loaded_head_path: Path | None = None
        if load_trained_head:
            self.load_head()

    def _init_head(self) -> None:
        """Zero the last linear layer and set bias to -4 so p(eot) starts low."""
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, -4.0)

    def _set_head(self, architecture: str) -> None:
        """Replace `self.head` with a freshly initialized module of `architecture`."""
        if architecture == self.head_name:
            return
        self.head_name = architecture
        self.head = build_pause_head(self._encoder_width, architecture)
        self._init_head()
        self.head.to(self.device)
        self._loaded_head_path = None

    def _set_pool(self, pool: str, pool_ms: float) -> None:
        """Use `pool` / `pool_ms` for later `pooled_embedding` calls."""
        if pool not in POOL_MODES:
            raise ValueError(f"unknown pooling mode {pool!r}; expected one of {POOL_MODES}")
        self.pool_name = pool
        self.pool_ms = float(pool_ms)

    def load_head(self, path: Path | None = None) -> bool:
        """Load a head `state_dict`. Returns False if missing or incompatible."""
        head_path = resolve_head_weights(path)
        if head_path is None:
            return False
        resolved = head_path.resolve()
        if self._loaded_head_path == resolved:
            return True
        payload = torch.load(head_path, map_location=self.device, weights_only=True)
        architecture = architecture_from_run(head_path) or architecture_from_state_dict(payload)
        if architecture is not None:
            self._set_head(architecture)
        pooling = pooling_from_run(head_path)
        if pooling is not None:
            self._set_pool(*pooling)
        try:
            self.head.load_state_dict(payload)
        except RuntimeError:
            return False
        self.head.to(self.device)
        self._loaded_head_path = resolved
        return True

    def save_head(self, path: Path) -> Path:
        """Write the current head `state_dict` to `path` and return it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), path)
        return path

    def pooled_embedding(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        """Pool frozen encoder frames over the resampled waveform."""
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(-1)).to(self.device)
        if waveform.numel() == 0:
            return torch.zeros(1, self._encoder_width, device=self.device)
        waveform = resample_to_16k(waveform, int(sample_rate))
        features = log_mel_spectrogram(waveform, self.mel_filters).unsqueeze(0)
        with torch.no_grad():
            hidden = self.encoder(features)
        return pool_encoder_states(hidden, self.pool_name, pool_ms=self.pool_ms)

    def logits(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        """Return the raw hold/eot logit for `audio`."""
        return self.head(self.pooled_embedding(audio, sample_rate)).squeeze(-1)

    @torch.inference_mode()
    def p_eot(self, audio: np.ndarray, sample_rate: int) -> float:
        """Return sigmoid(logit) as p(end of turn)."""
        logit = self.logits(audio, sample_rate)
        return float(torch.sigmoid(logit).item())
