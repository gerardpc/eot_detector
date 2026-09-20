"""Pause-cut timestamps for hold/eot training clips."""

from __future__ import annotations

MIN_PAUSE_SECONDS = 0.1
"""Pauses shorter than this are not cut into training clips."""
CONTEXT_SECONDS = 5.0
"""Causal context window length used when slicing clips."""
EARLY_SECONDS = 0.2
"""Offset from pause start for the early hold/eot cut."""
EOT_MID_SECONDS = 1.0
"""Offset from pause start for the extra end-of-turn cut."""


def clip_window(
    t: float, duration: float, context_seconds: float = CONTEXT_SECONDS
) -> tuple[float, float]:
    """Return the causal `[max(0, t − context), t]` window clamped to the clip."""
    end = min(max(t, 0.0), duration)
    start = max(0.0, end - context_seconds)
    return start, end


def pause_cut_times(
    start: float,
    end: float,
    *,
    is_eot: bool,
    duration: float | None = None,
) -> list[tuple[float, str]]:
    """Return (t, kind) cut times for one pause.

    Hold: 200 ms after start if it fits, and pause end.
    EOT: 200 ms after start if it fits, 1 s after start if that is still
    before end, and pause end.
    """

    pause_end = float(end)
    if duration is not None:
        pause_end = min(pause_end, float(duration))
    pause_start = float(start)
    if pause_end - pause_start < MIN_PAUSE_SECONDS:
        return []

    cuts: list[tuple[float, str]] = []
    if pause_start + EARLY_SECONDS <= pause_end:
        cuts.append((pause_start + EARLY_SECONDS, "early200"))
    if is_eot and pause_start + EOT_MID_SECONDS < pause_end:
        cuts.append((pause_start + EOT_MID_SECONDS, "mid1000"))
    cuts.append((pause_end, "end"))
    return cuts
