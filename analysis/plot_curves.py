"""Plot smoothed validation curves for slide decks.

Reads `analysis/runs.json` and writes PNG/SVG for val accuracy and val BCE loss.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "runs.json"
SMOOTH_WEIGHT = 0.85
"""TensorBoard-style exponential smoothing weight."""


def smooth_ema(values: np.ndarray, weight: float = SMOOTH_WEIGHT) -> np.ndarray:
    """Exponential moving average, same idea as TensorBoard's smoothing slider."""
    out = np.empty_like(values)
    last = float(values[0])
    for i, value in enumerate(values):
        last = weight * last + (1.0 - weight) * float(value)
        out[i] = last
    return out


def _style_axes(ax: plt.Axes, ylabel: str, ylim: tuple[float, float]) -> None:
    """Shared axis formatting for both charts."""
    ax.set_xlabel("Training samples")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 2_500_000)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000])
    ax.set_xticklabels(["0", "0.5M", "1M", "1.5M", "2M", "2.5M"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _pos: f"{y:.2f}"))
    ax.grid(True, axis="both", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_metric(
    runs: list[dict],
    key: str,
    *,
    ylabel: str,
    ylim: tuple[float, float],
    legend_loc: str,
    stem: str,
) -> tuple[Path, Path]:
    """Draw one metric and save PNG + SVG next to this script."""
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for run in runs:
        steps = np.asarray(run["steps"], dtype=np.float64)
        values = np.asarray(run[key], dtype=np.float64)
        color = COLORS[run["label"]]
        ax.plot(steps, values, color=color, alpha=0.22, linewidth=1.0, zorder=1)
        ax.plot(
            steps,
            smooth_ema(values),
            color=color,
            linewidth=2.3,
            label=run["label"],
            zorder=2,
        )
    _style_axes(ax, ylabel, ylim)
    ax.legend(frameon=False, loc=legend_loc, fontsize=9)
    fig.tight_layout()
    png = HERE / f"{stem}.png"
    svg = HERE / f"{stem}.svg"
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def plot() -> None:
    """Render validation accuracy and BCE loss charts."""
    runs = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    written = []
    written.extend(
        plot_metric(
            runs,
            "val_acc",
            ylabel="Validation accuracy",
            ylim=(0.78, 0.90),
            legend_loc="lower right",
            stem="val_acc",
        )
    )
    written.extend(
        plot_metric(
            runs,
            "val_loss",
            ylabel="Validation BCE loss",
            ylim=(0.28, 0.52),
            legend_loc="upper right",
            stem="val_loss",
        )
    )
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    plot()
