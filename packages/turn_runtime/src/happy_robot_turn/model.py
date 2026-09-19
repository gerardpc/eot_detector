"""Lazy loading and audio normalization for the public DualTurn checkpoint."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_MODEL_ID = "anyreach-ai/dualturn-qwen2.5-mimi-0.5B"
OUTPUT_NAMES = ("eot_probs", "hold_probs", "bot_probs", "bc_probs", "vad_probs", "fvad_probs")


@dataclass(frozen=True)
class TurnPrediction:
    """Model outputs for one audio window."""

    outputs: dict[str, np.ndarray]

    def last_frame(self) -> dict[str, list[float]]:
        """Return the last frame in a JSON-friendly representation."""

        return {
            name: _last_frame(values).astype(float).tolist()
            for name, values in self.outputs.items()
        }


class DualTurnRunner:
    """Load and run DualTurn only when explicitly requested.

    The model card's public API accepts a torch tensor shaped ``[channels, samples]``
    and the original sample rate. A mono input is expanded with a silent agent
    channel so local smoke tests do not need a mixer.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.device_name = _resolve_device(device)
        self.model: Any | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> DualTurnRunner:
        """Download/load the checkpoint and custom model code from Hugging Face."""

        if self.model is not None:
            return self

        from transformers import AutoModel

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.device_name == "cuda":
            kwargs["torch_dtype"] = "auto"
        model = AutoModel.from_pretrained(self.model_id, **kwargs)
        model.to(self.device_name)
        model.eval()
        self.model = model
        return self

    def predict_array(self, audio: np.ndarray, sample_rate: int) -> TurnPrediction:
        """Run inference on mono or two-channel audio."""

        self.load()
        import torch

        waveform = _as_stereo_float32(audio)
        tensor = torch.from_numpy(waveform)
        if self.device_name != "cpu":
            tensor = tensor.to(self.device_name)
        with torch.inference_mode():
            raw = self.model(tensor, sr=int(sample_rate))
        return TurnPrediction({name: _to_numpy(getattr(raw, name)) for name in OUTPUT_NAMES})

    def predict_wav_bytes(self, payload: bytes) -> TurnPrediction:
        """Decode a WAV payload and run inference."""

        import soundfile as sf

        audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
        return self.predict_array(audio, int(sample_rate))

    def status(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "device": self.device_name,
            "loaded": self.loaded,
        }


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("device='cuda' was requested but CUDA is unavailable")
        return requested

    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _as_stereo_float32(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return np.stack([array, np.zeros_like(array)], axis=0)
    if array.ndim != 2:
        raise ValueError(
            f"audio must have shape [samples] or [samples, channels], got {array.shape}"
        )
    if array.shape[1] == 1:
        return np.concatenate([array.T, np.zeros((1, array.shape[0]), dtype=np.float32)], axis=0)
    if array.shape[1] >= 2:
        return array[:, :2].T.copy()
    raise ValueError("audio contains no samples")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _last_frame(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    if array.ndim == 3:
        array = array[0]
    return array[-1]


def runner_from_environment() -> DualTurnRunner:
    """Build a runner from environment variables without loading the model."""

    return DualTurnRunner(
        model_id=os.getenv("DUALTURN_MODEL_ID", DEFAULT_MODEL_ID),
        device=os.getenv("DUALTURN_DEVICE", "auto"),
    )
