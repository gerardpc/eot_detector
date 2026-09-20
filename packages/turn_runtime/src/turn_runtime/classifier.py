"""Frozen Whisper-tiny encoder plus a tiny hold/eot head."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .whisper import load_encoder, log_mel_spectrogram, resample_to_16k

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[4] / "models" / "whisper-tiny"
"""Repository-relative directory for the Whisper-tiny snapshot."""
DEFAULT_HEAD_DIR = Path(__file__).resolve().parents[4] / "models" / "pause_head"
"""Repository-relative directory for versioned pause-head runs."""
WEIGHT_NAMES = ("best.pt", "latest.pt")
"""Checkpoint filenames preferred in order when resolving a run."""
HEAD_HIDDEN = 64
"""Hidden width of the GELU MLP pause head."""
HEAD_NAME = "linear-64-gelu-linear"
"""Architecture tag stored in training `config.json`."""


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


def list_pause_heads(root: Path | None = None) -> list[dict[str, str]]:
    """List `current` plus run directories that contain `best.pt` or `latest.pt`."""
    directory = root or DEFAULT_HEAD_DIR
    current = directory / "current"
    current_label = "current"
    if current.exists():
        resolved = current.resolve().name
        if resolved != "current":
            current_label = f"current ({resolved})"
    items = [{"id": "current", "label": current_label}]
    if not directory.is_dir():
        return items
    runs = [path for path in directory.iterdir() if path.is_dir() and path.name != "current"]
    runs.sort(key=lambda path: path.name, reverse=True)
    for run in runs:
        if _weights_in(run) is None:
            continue
        label = run.name
        config_path = run / "config.json"
        if config_path.is_file():
            try:
                meta = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
            best = meta.get("best_val_loss")
            if isinstance(best, int | float):
                label = f"{run.name}  val {float(best):.3f}"
        items.append({"id": run.name, "label": label})
    return items


class PauseClassifier(nn.Module):
    """Map the last ≤5 s of pause-context audio to p(eot)."""

    def __init__(
        self,
        model_dir: Path | None = None,
        device: str | None = None,
        *,
        load_trained_head: bool = True,
    ) -> None:
        """Load the frozen encoder and optionally the on-disk pause head.

        Args:
            model_dir: Whisper-tiny snapshot directory.
            device: Torch device; defaults to CUDA when available.
            load_trained_head: Load `best.pt`/`latest.pt` from the head store.
        """
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder, self.mel_filters = load_encoder(model_dir or DEFAULT_MODEL_DIR, self.device)
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.eval()
        width = int(self.encoder.layer_norm.normalized_shape[0])
        self.head = nn.Sequential(
            nn.Linear(width, HEAD_HIDDEN),
            nn.GELU(),
            nn.Linear(HEAD_HIDDEN, 1),
        )
        self._init_head()
        self.head.to(self.device)
        self.eval()
        self._loaded_head_path: Path | None = None
        if load_trained_head:
            self.load_head()

    def _init_head(self) -> None:
        """Zero the last linear layer and set bias to -4 so p(eot) starts low."""
        nn.init.zeros_(self.head[2].weight)
        nn.init.constant_(self.head[2].bias, -4.0)

    def load_head(self, path: Path | None = None) -> bool:
        """Load a head `state_dict`. Returns False if missing or incompatible."""
        head_path = resolve_head_weights(path)
        if head_path is None:
            return False
        resolved = head_path.resolve()
        if self._loaded_head_path == resolved:
            return True
        payload = torch.load(head_path, map_location=self.device, weights_only=True)
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
        """Mean-pool the frozen encoder over the resampled waveform."""
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(-1)).to(self.device)
        if waveform.numel() == 0:
            width = int(self.encoder.layer_norm.normalized_shape[0])
            return torch.zeros(1, width, device=self.device)
        waveform = resample_to_16k(waveform, int(sample_rate))
        features = log_mel_spectrogram(waveform, self.mel_filters).unsqueeze(0)
        with torch.no_grad():
            hidden = self.encoder(features)
        return hidden.mean(dim=1)

    def logits(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        """Return the raw hold/eot logit for `audio`."""
        return self.head(self.pooled_embedding(audio, sample_rate)).squeeze(-1)

    @torch.inference_mode()
    def p_eot(self, audio: np.ndarray, sample_rate: int) -> float:
        """Return sigmoid(logit) as p(end of turn)."""
        logit = self.logits(audio, sample_rate)
        return float(torch.sigmoid(logit).item())
