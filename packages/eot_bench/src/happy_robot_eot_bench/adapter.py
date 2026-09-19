"""Adapter matching the official eot-bench pointwise contract."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from happy_robot_turn import DualTurnRunner


class DualTurnEotAdapter:
    """Score each causal audio snapshot with DualTurn's user EOT head."""

    display_name = "DualTurn (user channel)"
    adapter_id = "dualturn_qwen25_mimi_05b_user_channel"

    def __init__(self) -> None:
        self.runner = DualTurnRunner(
            model_id=os.getenv("DUALTURN_MODEL_ID", "anyreach-ai/dualturn-qwen2.5-mimi-0.5B"),
            device=os.getenv("DUALTURN_DEVICE", "auto"),
        )

    def supports_language(self, lang_code: str) -> bool:
        del lang_code
        return True

    def predict_batch(self, batch: list[dict[str, Any]]) -> list[float]:
        scores: list[float] = []
        for item in batch:
            audio = item.get("audio") or {}
            if "array" not in audio or "sampling_rate" not in audio:
                raise ValueError("eot-bench audio must contain array and sampling_rate")
            prediction = self.runner.predict_array(
                np.asarray(audio["array"], dtype=np.float32),
                int(audio["sampling_rate"]),
            )
            eot = np.asarray(prediction.outputs["eot_probs"])
            if eot.ndim == 3:
                eot = eot[0]
            if eot.ndim != 2 or eot.shape[1] < 1:
                raise ValueError(f"Unexpected eot_probs shape: {eot.shape}")
            scores.append(float(eot[-1, 0]))
        return scores
