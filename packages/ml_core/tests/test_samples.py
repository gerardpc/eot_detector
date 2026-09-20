"""Tests for hold/eot pause cut times."""

from ml_core.samples import clip_window, pause_cut_times


def test_hold_pause_uses_early_and_end() -> None:
    assert pause_cut_times(1.0, 2.0, is_eot=False) == [(1.2, "early200"), (2.0, "end")]


def test_short_hold_only_uses_end() -> None:
    assert pause_cut_times(1.0, 1.15, is_eot=False) == [(1.15, "end")]


def test_eot_pause_adds_one_second_cut() -> None:
    assert pause_cut_times(1.0, 3.0, is_eot=True) == [
        (1.2, "early200"),
        (2.0, "mid1000"),
        (3.0, "end"),
    ]


def test_eot_skips_one_second_when_it_would_duplicate_end() -> None:
    assert pause_cut_times(1.0, 2.0, is_eot=True) == [(1.2, "early200"), (2.0, "end")]


def test_pauses_under_100ms_are_skipped() -> None:
    assert pause_cut_times(1.0, 1.05, is_eot=True) == []


def test_clip_window_is_causal() -> None:
    assert clip_window(6.0, 10.0) == (1.0, 6.0)
    assert clip_window(2.0, 4.0) == (0.0, 2.0)
