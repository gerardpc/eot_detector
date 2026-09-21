"""Energy VAD and pause-state tracking for live end-of-turn detection."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

TurnState = Literal["speaking", "hold", "eot"]
"""Live detector state: speech, mid-turn pause, or end of turn."""

FRAME_SECONDS = 0.02
"""Energy-VAD analysis hop in seconds."""
MIN_SILENCE_SECONDS = 0.1
"""Silence that must accumulate before a pause is classified."""
CONTEXT_SECONDS = 5.0
"""Maximum audio context encoded for p(eot)."""
SPEECH_ON_RMS = 0.015
"""RMS that starts a speaking state from pause."""
SPEECH_OFF_RMS = 0.008
"""RMS that keeps the speaking state from dropping to pause."""
EOT_THRESHOLD = 0.5
"""p(eot) at or above this value is labeled `eot`."""
FORCE_EOT_SECONDS = 3.0
"""Silence after which eot is forced without calling the head."""
CLASSIFY_HOP_SECONDS = 0.1
"""Minimum pause time between successive head inferences."""
ENCODER_ID = "openai/whisper-tiny"
"""Frozen encoder id reported in runner status."""


@dataclass(frozen=True)
class TurnEvent:
    """Latest turn-taking state for one streamed audio chunk."""

    state: TurnState
    """Detector state after this chunk (`speaking`, `hold`, or `eot`)."""
    silence_seconds: float
    """Consecutive silence duration in seconds."""
    rms: float
    """RMS of the last completed VAD frame."""
    p_eot: float
    """Model (or forced) probability that the pause is end of turn."""


class TurnRunner:
    """Track speaking vs pause with energy VAD, then classify long pauses.

    Speech is ``speaking``. After ``MIN_SILENCE_SECONDS`` of quiet the runner
    encodes the last ``CONTEXT_SECONDS`` of audio with a frozen Whisper-tiny
    encoder and a tiny PyTorch head. Without a trained head, pauses stay ``hold``.
    """

    def __init__(
        self,
        *,
        min_silence_seconds: float = MIN_SILENCE_SECONDS,
        speech_on_rms: float = SPEECH_ON_RMS,
        speech_off_rms: float = SPEECH_OFF_RMS,
        context_seconds: float = CONTEXT_SECONDS,
        load_classifier: bool = False,
        classifier: object | None = None,
        infer_lock: threading.Lock | None = None,
        head_dir: Path | None = None,
    ) -> None:
        """Create a runner, optionally sharing a process-level classifier.

        Args:
            min_silence_seconds: Silence that must accumulate before classifying.
            speech_on_rms: RMS that enters `speaking` from pause.
            speech_off_rms: RMS that stays in `speaking`.
            context_seconds: Audio kept for the encoder window.
            load_classifier: Construct a `PauseClassifier` on first pause.
            classifier: Shared encoder+head; skips constructing a second one.
            infer_lock: Serialize `load_head` / `p_eot` across connections.
            head_dir: Pause-head store; defaults to `models/pause_head/`.
        """
        self.min_silence_seconds = min_silence_seconds
        self.speech_on_rms = speech_on_rms
        self.speech_off_rms = speech_off_rms
        self.context_seconds = context_seconds
        self._classifier = classifier
        self._infer_lock = infer_lock
        self._load_classifier = load_classifier or classifier is not None
        self._head_dir = Path(head_dir) if head_dir is not None else None
        self.state: TurnState = "hold"
        self._pcm = np.zeros(0, dtype=np.float32)
        self._tail = np.zeros(0, dtype=np.float32)
        self._stream_sr = 0
        self._silence_seconds = 0.0
        self._last_rms = 0.0
        self._cached_p_eot = 0.0
        self._last_classify_silence = -1.0
        self._head_id = "current"
        self.eot_threshold = EOT_THRESHOLD

    def _resolved_head_dir(self) -> Path:
        """Return the pause-head store for this runner."""
        from .classifier import DEFAULT_HEAD_DIR

        return self._head_dir or DEFAULT_HEAD_DIR

    def reset_stream(self) -> None:
        """Clear PCM, VAD timers, and cached p(eot) for a new stream."""
        self.state = "hold"
        self._pcm = np.zeros(0, dtype=np.float32)
        self._tail = np.zeros(0, dtype=np.float32)
        self._stream_sr = 0
        self._silence_seconds = 0.0
        self._last_rms = 0.0
        self._cached_p_eot = 0.0
        self._last_classify_silence = -1.0

    def ensure_classifier(self) -> None:
        """Load a local `PauseClassifier` the first time a pause needs a score."""
        if self._classifier is not None or not self._load_classifier:
            return
        from .classifier import PauseClassifier

        self._classifier = PauseClassifier(load_trained_head=False)
        self._classifier.load_head(self._resolved_head_dir() / self._head_id)

    def select_head(self, run_id: str = "current") -> Path | None:
        """Select a pause-head run id. Returns the weights path, or None if missing."""
        from .classifier import resolve_head_weights

        name = (run_id or "current").strip() or "current"
        target = self._resolved_head_dir() / name
        if name != "current" and (not target.is_dir() or resolve_head_weights(target) is None):
            return None
        self._head_id = name
        self._cached_p_eot = 0.0
        self._last_classify_silence = -1.0
        if self._classifier is not None and self._infer_lock is None:
            load_head = getattr(self._classifier, "load_head", None)
            if callable(load_head):
                load_head(target)
        return resolve_head_weights(target) or target

    def set_threshold(self, value: float) -> float:
        """Clamp `value` to `[0, 1]` and use it as the live hold/eot cutoff."""
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = EOT_THRESHOLD
        if parsed != parsed:  # NaN
            parsed = EOT_THRESHOLD
        self.eot_threshold = min(1.0, max(0.0, parsed))
        return self.eot_threshold

    def consume_stream(self, samples: np.ndarray, sample_rate: int) -> TurnEvent:
        """Ingest a PCM chunk and return the latest speaking / hold / eot state."""
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self._stream_sr != sample_rate:
            self.reset_stream()
            self._stream_sr = int(sample_rate)
        if chunk.size:
            self._pcm = np.concatenate([self._pcm, chunk])
            max_len = max(int(sample_rate * self.context_seconds), 1)
            if self._pcm.size > max_len:
                self._pcm = self._pcm[-max_len:]
        self._process_frames(chunk, sample_rate)
        p_eot = self._pause_p_eot() if self.state != "speaking" else 0.0
        return TurnEvent(
            state=self.state,
            silence_seconds=self._silence_seconds,
            rms=self._last_rms,
            p_eot=p_eot,
        )

    def status(self) -> dict[str, object]:
        """Return encoder, head, and VAD configuration for health payloads."""
        return {
            "state": self.state,
            "encoder_id": ENCODER_ID,
            "classifier": "whisper-tiny-head" if self._classifier is not None else "none",
            "head_id": self._head_id,
            "min_silence_seconds": self.min_silence_seconds,
            "force_eot_seconds": FORCE_EOT_SECONDS,
            "context_seconds": self.context_seconds,
            "eot_threshold": self.eot_threshold,
        }

    def _process_frames(self, samples: np.ndarray, sample_rate: int) -> None:
        """Split PCM into `FRAME_SECONDS` hops and update VAD state."""
        audio = np.concatenate([self._tail, samples]) if samples.size else self._tail
        frame = max(int(sample_rate * FRAME_SECONDS), 1)
        usable = (audio.size // frame) * frame
        if usable:
            frames = audio[:usable].reshape(-1, frame)
            dt = frame / sample_rate
            for row in frames:
                rms = float(np.sqrt(np.mean(np.square(row))))
                self._last_rms = rms
                self._update_state(rms, dt)
        self._tail = audio[usable:]

    def _update_state(self, rms: float, dt: float) -> None:
        """Advance speaking vs pause given one frame's RMS and duration."""
        if self.state == "speaking":
            if rms >= self.speech_off_rms:
                self._silence_seconds = 0.0
                self._cached_p_eot = 0.0
                self._last_classify_silence = -1.0
                return
            self._silence_seconds += dt
            if self._silence_seconds >= self.min_silence_seconds:
                self.state = self._pause_state()
            return
        if rms >= self.speech_on_rms:
            self.state = "speaking"
            self._silence_seconds = 0.0
            self._cached_p_eot = 0.0
            self._last_classify_silence = -1.0
            return
        self._silence_seconds += dt
        self.state = self._pause_state()

    def _pause_state(self) -> TurnState:
        """Map current p(eot) onto `hold` or `eot`."""
        return "hold" if self._pause_p_eot() < self.eot_threshold else "eot"

    def _pause_p_eot(self) -> float:
        """Score p(eot) for the current pause, with hop cache and 3 s force-eot."""
        if self._silence_seconds < self.min_silence_seconds:
            return 0.0
        if self._silence_seconds >= FORCE_EOT_SECONDS:
            self._cached_p_eot = 1.0
            return 1.0
        if self._classifier is None and not self._load_classifier:
            return 0.0
        if (
            self._last_classify_silence >= 0
            and self._silence_seconds - self._last_classify_silence < CLASSIFY_HOP_SECONDS
        ):
            return self._cached_p_eot
        self.ensure_classifier()
        if self._classifier is None:
            return 0.0
        if self._infer_lock is None:
            self._cached_p_eot = self._classifier.p_eot(self._pcm, self._stream_sr)
        else:
            with self._infer_lock:
                load_head = getattr(self._classifier, "load_head", None)
                if callable(load_head):
                    load_head(self._resolved_head_dir() / self._head_id)
                self._cached_p_eot = self._classifier.p_eot(self._pcm, self._stream_sr)
        self._last_classify_silence = self._silence_seconds
        return self._cached_p_eot


def runner_from_environment() -> TurnRunner:
    """Build a runner that loads the pause classifier on first pause."""
    return TurnRunner(load_classifier=True)
