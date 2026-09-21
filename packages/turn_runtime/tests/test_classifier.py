"""PauseClassifier tests that skip encoder-heavy cases when Whisper is missing."""

import json

import numpy as np
import pytest
import torch
from turn_runtime.classifier import (
    DEFAULT_MODEL_DIR,
    DEFAULT_TAIL_MS,
    HEAD_LINEAR,
    HEAD_MLP,
    POOL_EMA,
    POOL_MEAN,
    POOL_TAIL,
    PauseClassifier,
    architecture_from_state_dict,
    build_pause_head,
    pool_encoder_states,
    pooling_from_run,
    tail_frames_for_ms,
)

pytest.importorskip("torch")


def test_build_pause_head_shapes() -> None:
    mlp = build_pause_head(384, HEAD_MLP)
    linear = build_pause_head(384, HEAD_LINEAR)
    assert mlp[0].in_features == 384
    assert mlp[0].out_features == 64
    assert mlp[2].out_features == 1
    assert linear[0].in_features == 384
    assert linear[0].out_features == 1
    assert architecture_from_state_dict(mlp.state_dict()) == HEAD_MLP
    assert architecture_from_state_dict(linear.state_dict()) == HEAD_LINEAR
    with pytest.raises(ValueError, match="unknown pause-head architecture"):
        build_pause_head(384, "nope")


def test_ema_pooling_is_recency_weighted() -> None:
    ones = torch.ones(1, 17, 4)
    assert torch.allclose(pool_encoder_states(ones, POOL_EMA, pool_ms=400), torch.ones(1, 4))
    newest = torch.zeros(1, 40, 1)
    newest[:, -1, :] = 1.0
    oldest = torch.zeros(1, 40, 1)
    oldest[:, 0, :] = 1.0
    ema_new = float(pool_encoder_states(newest, POOL_EMA, pool_ms=400))
    ema_old = float(pool_encoder_states(oldest, POOL_EMA, pool_ms=400))
    mean_new = float(pool_encoder_states(newest, POOL_MEAN))
    assert ema_new > ema_old
    assert ema_new > mean_new
    assert ema_old < mean_new


def test_tail_pooling_uses_last_frames() -> None:
    assert tail_frames_for_ms(400) == 20
    assert tail_frames_for_ms(300) == 15
    assert tail_frames_for_ms(500) == 25
    assert tail_frames_for_ms(1000) == 50
    hidden = torch.zeros(1, 10, 4)
    hidden[:, -3:, :] = 1.0
    tail = pool_encoder_states(hidden, POOL_TAIL, pool_ms=60.0)
    assert torch.allclose(tail, torch.ones(1, 4))
    mean = pool_encoder_states(hidden, POOL_MEAN)
    assert torch.allclose(mean, torch.full((1, 4), 0.3))


def test_pooling_from_run_defaults_missing_pool_to_mean(tmp_path) -> None:
    run = tmp_path / "old-run"
    run.mkdir()
    (run / "config.json").write_text('{"head": "linear"}', encoding="utf-8")
    assert pooling_from_run(run) == (POOL_MEAN, DEFAULT_TAIL_MS)
    assert pooling_from_run(tmp_path / "missing") is None


@pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "model.safetensors").exists(),
    reason="whisper-tiny weights are not downloaded",
)
def test_untrained_head_stays_on_hold() -> None:
    classifier = PauseClassifier(load_trained_head=False)
    audio = np.zeros(16_000, dtype=np.float32)
    assert classifier.p_eot(audio, 16_000) < 0.1


@pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "model.safetensors").exists(),
    reason="whisper-tiny weights are not downloaded",
)
def test_head_roundtrip(tmp_path) -> None:
    classifier = PauseClassifier(load_trained_head=False)
    path = tmp_path / "pause_head.pt"
    classifier.save_head(path)
    loaded = PauseClassifier(load_trained_head=False)
    assert loaded.load_head(path)
    audio = np.zeros(16_000, dtype=np.float32)
    assert loaded.p_eot(audio, 16_000) < 0.1


@pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "model.safetensors").exists(),
    reason="whisper-tiny weights are not downloaded",
)
def test_load_head_switches_to_linear(tmp_path) -> None:
    run = tmp_path / "linear-run"
    run.mkdir()
    linear = PauseClassifier(load_trained_head=False, head=HEAD_LINEAR)
    linear.save_head(run / "best.pt")
    (run / "config.json").write_text(
        json.dumps({"head": HEAD_LINEAR}),
        encoding="utf-8",
    )
    loaded = PauseClassifier(load_trained_head=False, head=HEAD_MLP)
    assert loaded.head_name == HEAD_MLP
    assert loaded.load_head(run)
    assert loaded.head_name == HEAD_LINEAR
    assert loaded.head[0].out_features == 1
    audio = np.zeros(16_000, dtype=np.float32)
    assert loaded.p_eot(audio, 16_000) < 0.1


@pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "model.safetensors").exists(),
    reason="whisper-tiny weights are not downloaded",
)
def test_load_head_applies_tail_pool(tmp_path) -> None:
    run = tmp_path / "tail-run"
    run.mkdir()
    trained = PauseClassifier(
        load_trained_head=False,
        head=HEAD_LINEAR,
        pool=POOL_TAIL,
        pool_ms=400,
    )
    trained.save_head(run / "best.pt")
    (run / "config.json").write_text(
        json.dumps({"head": HEAD_LINEAR, "pool": POOL_TAIL, "pool_ms": 400}),
        encoding="utf-8",
    )
    loaded = PauseClassifier(load_trained_head=False, pool=POOL_MEAN)
    assert loaded.pool_name == POOL_MEAN
    assert loaded.load_head(run)
    assert loaded.pool_name == POOL_TAIL
    assert loaded.pool_ms == 400.0
