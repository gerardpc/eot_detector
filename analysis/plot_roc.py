"""Plot validation ROC curves from `analysis/roc.json`."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from roc_math import OPERATING_THRESHOLD, operating_point, roc_curve
from style import COLORS

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "roc.json"


def plot() -> tuple[Path, Path]:
    """Draw ROC curves with the p(eot)≥0.5 operating point marked."""
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    y_true = np.asarray(payload["labels"], dtype=np.int8)
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.0, label="Chance")
    for run in payload["runs"]:
        scores = np.asarray(run["scores"], dtype=np.float64)
        fpr, tpr, auc = roc_curve(y_true, scores)
        color = COLORS[run["label"]]
        ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{run['label']}  AUC {auc:.3f}")
        op_fpr, op_tpr = operating_point(y_true, scores)
        ax.scatter(
            [op_fpr],
            [op_tpr],
            color=color,
            s=28,
            zorder=3,
            edgecolors="white",
            linewidths=0.6,
        )
    ax.set_xlabel("False EOT rate  (interruptions: predicted eot | hold)")
    ax.set_ylabel("True EOT rate  (caught eot)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, axis="both", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=8,
        title=f"dots: p(eot) ≥ {OPERATING_THRESHOLD:g}",
    )
    fig.tight_layout()
    png = HERE / "roc.png"
    svg = HERE / "roc.svg"
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {svg}")
    return png, svg


if __name__ == "__main__":
    plot()
