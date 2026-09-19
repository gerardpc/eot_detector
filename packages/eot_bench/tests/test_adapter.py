import numpy as np
from happy_robot_eot_bench.adapter import DualTurnEotAdapter


class _Prediction:
    outputs = {"eot_probs": np.array([[[0.25, 0.5], [0.75, 0.2]]], dtype=np.float32)}


def test_adapter_uses_last_user_eot_score() -> None:
    adapter = DualTurnEotAdapter.__new__(DualTurnEotAdapter)
    adapter.runner = type("Runner", (), {"predict_array": lambda *_args: _Prediction()})()
    batch = [{"audio": {"array": np.zeros(20), "sampling_rate": 16000}}]
    assert adapter.predict_batch(batch) == [0.75]
