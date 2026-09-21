"""ROC helpers. Positive class is eot (1); FPR is false eot on a hold."""

from __future__ import annotations

import numpy as np

OPERATING_THRESHOLD = 0.5
"""Runtime cutoff: p(eot) ≥ 0.5 is eot."""


def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return FPR, TPR, and trapezoidal AUC, sorted by descending score."""
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC needs both hold and eot labels")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    distinct = np.append(np.flatnonzero(np.diff(s_sorted)), y.size - 1)
    tpr = np.concatenate(([0.0], tps[distinct] / n_pos))
    fpr = np.concatenate(([0.0], fps[distinct] / n_neg))
    trapz = getattr(np, "trapezoid", np.trapz)
    auc = float(trapz(tpr, fpr))
    return fpr, tpr, auc


def operating_point(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = OPERATING_THRESHOLD,
) -> tuple[float, float]:
    """FPR and TPR at a probability threshold."""
    y = np.asarray(y_true, dtype=np.int8)
    pred = np.asarray(scores, dtype=np.float64) >= threshold
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    tpr = float(((pred == 1) & (y == 1)).sum() / n_pos)
    fpr = float(((pred == 1) & (y == 0)).sum() / n_neg)
    return fpr, tpr
