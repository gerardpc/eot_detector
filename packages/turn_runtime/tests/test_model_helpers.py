import numpy as np
from happy_robot_turn.model import _as_stereo_float32, _last_frame


def test_mono_audio_gets_silent_agent_channel() -> None:
    result = _as_stereo_float32(np.ones(4, dtype=np.float32))
    assert result.shape == (2, 4)
    np.testing.assert_array_equal(result[0], 1.0)
    np.testing.assert_array_equal(result[1], 0.0)


def test_last_frame_removes_batch_dimension() -> None:
    result = _last_frame(np.arange(12).reshape(1, 3, 4))
    np.testing.assert_array_equal(result, np.arange(8, 12))
