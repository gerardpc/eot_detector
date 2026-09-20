"""Runtime tests for energy VAD, force-eot, and shared-head loading."""

import threading
from pathlib import Path

import numpy as np
from turn_runtime.runtime import TurnRunner


def test_loud_audio_is_speaking() -> None:
    runner = TurnRunner()
    tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16_000).astype(np.float32)
    event = runner.consume_stream(tone, 16_000)
    assert event.state == "speaking"
    assert event.silence_seconds == 0.0


def test_long_silence_is_hold() -> None:
    runner = TurnRunner()
    event = runner.consume_stream(np.zeros(8000, dtype=np.float32), 16_000)
    assert event.state == "hold"
    assert event.silence_seconds >= 0.1
    assert event.p_eot == 0.0


def test_short_dip_stays_speaking() -> None:
    runner = TurnRunner()
    speech = 0.2 * np.ones(3200, dtype=np.float32)
    dip = np.zeros(800, dtype=np.float32)  # 50 ms at 16 kHz
    more_speech = 0.2 * np.ones(3200, dtype=np.float32)
    runner.consume_stream(speech, 16_000)
    runner.consume_stream(dip, 16_000)
    event = runner.consume_stream(more_speech, 16_000)
    assert event.state == "speaking"


def test_pause_after_speech_becomes_hold() -> None:
    runner = TurnRunner()
    speech = 0.2 * np.ones(3200, dtype=np.float32)
    pause = np.zeros(3200, dtype=np.float32)  # 200 ms
    runner.consume_stream(speech, 16_000)
    event = runner.consume_stream(pause, 16_000)
    assert event.state == "hold"
    assert event.silence_seconds >= 0.1


def test_high_p_eot_marks_eot() -> None:
    runner = TurnRunner(load_classifier=True)
    runner._classifier = type("Classifier", (), {"p_eot": staticmethod(lambda *_args: 0.9)})()
    event = runner.consume_stream(np.zeros(8000, dtype=np.float32), 16_000)
    assert event.state == "eot"
    assert event.p_eot == 0.9


def test_forced_eot_after_three_seconds_skips_classifier() -> None:
    runner = TurnRunner(load_classifier=True)

    class Probe:
        calls = 0

        def p_eot(self, *_args: object) -> float:
            type(self).calls += 1
            return 0.2

    runner._classifier = Probe()
    runner.consume_stream(np.zeros(48_000, dtype=np.float32), 16_000)
    calls_before = Probe.calls
    event = runner.consume_stream(np.zeros(16_000, dtype=np.float32), 16_000)
    assert event.silence_seconds >= 3.0
    assert event.state == "eot"
    assert event.p_eot == 1.0
    assert Probe.calls == calls_before


def test_select_head_records_id_without_loading() -> None:
    runner = TurnRunner(load_classifier=True)
    found = runner.select_head("current")
    assert found is not None
    assert runner.status()["head_id"] == "current"
    assert runner.status()["classifier"] == "none"
    assert runner.select_head("no-such-run") is None
    assert runner.status()["head_id"] == "current"


def test_shared_classifier_loads_head_under_lock() -> None:
    class Probe:
        loaded = None
        calls = 0

        def load_head(self, path: object) -> bool:
            type(self).loaded = path
            return True

        def p_eot(self, *_args: object) -> float:
            type(self).calls += 1
            return 0.9

    lock = threading.Lock()
    probe = Probe()
    runner = TurnRunner(classifier=probe, infer_lock=lock)
    runner.select_head("current")
    assert Probe.loaded is None
    event = runner.consume_stream(np.zeros(8000, dtype=np.float32), 16_000)
    assert event.state == "eot"
    assert Probe.calls >= 1
    assert Probe.loaded is not None
    assert Path(str(Probe.loaded)).name == "current"


def test_forced_eot_without_classifier() -> None:
    runner = TurnRunner()
    event = runner.consume_stream(np.zeros(48_000, dtype=np.float32), 16_000)
    assert event.state == "eot"
    assert event.p_eot == 1.0


def test_speech_after_forced_eot_classifies_again() -> None:
    runner = TurnRunner(load_classifier=True)

    class Probe:
        def p_eot(self, *_args: object) -> float:
            return 0.2

    runner._classifier = Probe()
    runner.consume_stream(np.zeros(48_000, dtype=np.float32), 16_000)
    assert runner.state == "eot"
    speech = 0.2 * np.ones(3_200, dtype=np.float32)
    pause = np.zeros(3_200, dtype=np.float32)
    runner.consume_stream(speech, 16_000)
    event = runner.consume_stream(pause, 16_000)
    assert event.state == "hold"
    assert event.p_eot == 0.2
