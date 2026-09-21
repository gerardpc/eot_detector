"""Per-language val accuracy at p(eot) ≥ 0.5 for the serving head."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from roc_math import OPERATING_THRESHOLD
from style import LANG_NAMES, TARGET_RUN

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ROC_PATH = HERE / "roc.json"
INDEX_PATH = REPO / "data" / "train_dataset" / "index.csv"
OUT_JSON = HERE / "lang_acc.json"
BAR_DEFAULT = "#B0B0B0"
BAR_ENGLISH = "#4D4D4D"
BAR_BEST = "#009E73"
BAR_WORST = "#D55E00"
MEDIAN_COLOR = "#4D4D4D"


def _val_rows() -> list[dict[str, str]]:
    """Read val rows from `data/train_dataset/index.csv`."""
    with INDEX_PATH.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "val"]
    if not rows:
        raise FileNotFoundError(f"no val clips in {INDEX_PATH}")
    return rows


def _language_stats(y_true: np.ndarray, pred: np.ndarray, langs: np.ndarray) -> list[dict]:
    """Accuracy, counts, and names for each language code."""
    rows: list[dict] = []
    for code in sorted(set(langs.tolist())):
        mask = langs == code
        n = int(mask.sum())
        correct = int((pred[mask] == y_true[mask]).sum())
        rows.append(
            {
                "code": code,
                "name": LANG_NAMES.get(code, code),
                "n": n,
                "n_hold": int((y_true[mask] == 0).sum()),
                "n_eot": int((y_true[mask] == 1).sum()),
                "correct": correct,
                "acc": correct / n,
            }
        )
    rows.sort(key=lambda row: (row["acc"], row["code"]))
    return rows


def compute() -> dict:
    """Join `roc.json` scores with val languages and summarize the target run."""
    payload = json.loads(ROC_PATH.read_text(encoding="utf-8"))
    rows = _val_rows()
    y_true = np.asarray(payload["labels"], dtype=np.int8)
    index_labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    if y_true.shape != index_labels.shape or not np.array_equal(y_true, index_labels):
        raise ValueError("roc.json labels do not match data/train_dataset/index.csv val rows")
    run = next((item for item in payload["runs"] if item["label"] == TARGET_RUN), None)
    if run is None:
        raise KeyError(f"run {TARGET_RUN!r} missing from roc.json")
    scores = np.asarray(run["scores"], dtype=np.float64)
    pred = (scores >= OPERATING_THRESHOLD).astype(np.int8)
    langs = np.asarray([row["language"] for row in rows])
    languages = _language_stats(y_true, pred, langs)
    accs = np.asarray([row["acc"] for row in languages], dtype=np.float64)
    best = languages[-1]
    worst = languages[0]
    english = next(row for row in languages if row["code"] == "en")
    summary = {
        "run": TARGET_RUN,
        "run_id": run["id"],
        "checkpoint": run["checkpoint"],
        "threshold": OPERATING_THRESHOLD,
        "n_val": int(y_true.size),
        "overall_acc": float((pred == y_true).mean()),
        "median_acc": float(np.median(accs)),
        "english": english,
        "best": best,
        "worst": worst,
        "languages": languages,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    return summary


def _bar_color(row: dict, best_code: str, worst_code: str) -> str:
    """Highlight English, best, and worst; gray for the rest."""
    if row["code"] == "en":
        return BAR_ENGLISH
    if row["code"] == best_code and row["code"] != "en":
        return BAR_BEST
    if row["code"] == worst_code and row["code"] != "en":
        return BAR_WORST
    return BAR_DEFAULT


def plot(summary: dict | None = None) -> tuple[Path, Path]:
    """Horizontal accuracy bars with English, median, best, and worst marked."""
    if summary is None:
        summary = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    languages = summary["languages"]
    best_code = summary["best"]["code"]
    worst_code = summary["worst"]["code"]
    names = [f"{row['name']}" for row in languages]
    accs = [row["acc"] for row in languages]
    colors = [_bar_color(row, best_code, worst_code) for row in languages]
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    y = np.arange(len(languages))
    ax.barh(y, accs, color=colors, height=0.72, zorder=2)
    ax.axvline(
        summary["median_acc"],
        color=MEDIAN_COLOR,
        linestyle="--",
        linewidth=1.3,
        zorder=3,
    )
    for i, row in enumerate(languages):
        ax.text(
            row["acc"] + 0.002,
            i,
            f"{row['acc']:.3f}",
            va="center",
            fontsize=8,
            color="#333333",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(f"Accuracy at p(eot) ≥ {OPERATING_THRESHOLD:g}")
    ax.set_xlim(0.80, 0.95)
    ax.set_ylim(-0.7, len(languages) - 0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _pos: f"{x:.2f}"))
    ax.grid(True, axis="x", linestyle="-", linewidth=0.6, alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=BAR_ENGLISH,
            lw=8,
            label=f"English  {summary['english']['acc']:.3f}",
        ),
        plt.Line2D(
            [0],
            [0],
            color=MEDIAN_COLOR,
            linestyle="--",
            lw=1.3,
            label=f"Median  {summary['median_acc']:.3f}",
        ),
        plt.Line2D(
            [0],
            [0],
            color=BAR_BEST,
            lw=8,
            label=f"Best  {summary['best']['name']}  {summary['best']['acc']:.3f}",
        ),
        plt.Line2D(
            [0],
            [0],
            color=BAR_WORST,
            lw=8,
            label=f"Worst  {summary['worst']['name']}  {summary['worst']['acc']:.3f}",
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
    )
    fig.tight_layout()
    png = HERE / "lang_acc.png"
    svg = HERE / "lang_acc.svg"
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {svg}")
    return png, svg


if __name__ == "__main__":
    plot(compute())
