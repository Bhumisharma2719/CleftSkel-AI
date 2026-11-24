"""Generate independent, paper-ready Aariz CVM distribution figures.

Run from the repository root:
    python backend/distribution_png.py --dataset "C:\\Users\\Bhumi\\Desktop\\Aariz"

Writes cvmi_stage_counts.png and cvmi_sample_weights.png into
backend/artifacts/ by default. Counts and smoothed weights are loaded from the
current dataset and the helper used by backend/model.py.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model import DEFAULT_DATASET_DIR, _balanced_class_weights, load_aariz_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
STAGES = [1, 2, 3, 4, 5, 6]
SPLIT_ORDER = ("train", "valid", "test")
SPLIT_COLORS = {"train": "#244B68", "valid": "#2A8C82", "test": "#C18A35"}
WEIGHT_NOTE = (
    "Weights = sqrt(N / (K x class count)), clipped to [0.5, 2.5]; "
    "applied to CVM samples only"
)
NAVY = "#18324B"
SLATE = "#516A80"
INK = "#273746"


def _save_stage_counts(counts: dict[str, Counter], split_sizes: dict[str, int], output_path: Path) -> None:
    """Save a compact grouped-bar figure sized for a single IEEE column."""
    fig, ax = plt.subplots(figsize=(3.5, 2.75), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    positions = np.arange(len(STAGES))
    splits = [name for name in SPLIT_ORDER if name in counts]
    width = 0.78 / max(1, len(splits))
    for split_index, split in enumerate(splits):
        values = [counts[split].get(stage, 0) for stage in STAGES]
        offset = (split_index - (len(splits) - 1) / 2) * width
        ax.bar(
            positions + offset,
            values,
            width=width * 0.94,
            label=f"{split.title()} (n={split_sizes[split]})",
            color=SPLIT_COLORS[split],
            edgecolor="white",
            linewidth=0.7,
        )

    ax.set_title("CVM stage counts by split", fontsize=10, weight="bold", color=NAVY, pad=8)
    ax.set_xlabel("Cervical vertebral maturation stage", fontsize=8, labelpad=5)
    ax.set_ylabel("Matched labeled images", fontsize=8, labelpad=4)
    ax.set_xticks(positions, [f"CVMI-{stage}" for stage in STAGES])
    ax.tick_params(axis="both", labelsize=7, length=3)
    ax.set_ylim(0, max(max((counts[name].get(stage, 0) for stage in STAGES), default=0) for name in splits) * 1.18)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=6.8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        handlelength=1.0,
        columnspacing=0.8,
    )

    total = sum(split_sizes.values())
    split_summary = " / ".join(str(split_sizes[name]) for name in splits)
    fig.text(
        0.5, 0.015,
        f"Aariz split: train / valid / test = {split_summary} (N={total})",
        ha="center", va="bottom", fontsize=6.6, color=SLATE,
    )
    fig.tight_layout(rect=(0.04, 0.09, 0.99, 0.89), pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_sample_weights(weights: dict[int, float], output_path: Path) -> None:
    """Save the clipped square-root training multipliers as a standalone plot."""
    values = [weights[stage - 1] for stage in STAGES]
    fig, ax = plt.subplots(figsize=(3.5, 2.75), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(
        [f"CVMI-{stage}" for stage in STAGES],
        values,
        color=["#2A8C82" if value >= 1.0 else "#587D9A" for value in values],
        edgecolor="white",
        linewidth=0.7,
    )
    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        if value < 1.0:
            ax.annotate(
                f"{value:.2f}", (x, value - 0.045), ha="center", va="top",
                fontsize=7.2, color="white", fontweight="bold",
            )
        else:
            ax.annotate(
                f"{value:.2f}", (x, value), xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.2, color=INK,
            )
    ax.axhline(1.0, color="#C18A35", linewidth=1.15, linestyle="--", label="Neutral weight = 1")
    ax.set_title("Smoothed CVM sample weights", fontsize=10, weight="bold", color=NAVY, pad=8)
    ax.set_xlabel("Cervical vertebral maturation stage", fontsize=8, labelpad=5)
    ax.set_ylabel("Applied multiplier", fontsize=8, labelpad=4)
    ax.set_ylim(0, 2.85)
    ax.tick_params(axis="both", labelsize=7, length=3)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper right", handlelength=1.5)
    fig.text(0.5, 0.015, WEIGHT_NOTE, ha="center", va="bottom", fontsize=6.5, color=SLATE, wrap=True)
    fig.tight_layout(rect=(0.035, 0.105, 0.99, 0.91), pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standalone Aariz CVM count and weight figures.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR), help="Aariz dataset root.")
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for cvmi_stage_counts.png and cvmi_sample_weights.png.",
    )
    args = parser.parse_args()

    splits = load_aariz_dataset(args.dataset)
    missing_splits = [name for name in SPLIT_ORDER if name not in splits]
    if missing_splits:
        raise ValueError(f"Dataset is missing required splits: {', '.join(missing_splits)}")
    split_sizes = {name: len(splits[name]) for name in SPLIT_ORDER}
    counts = {
        split: Counter(int(record["cvm_stage"]) for record in splits[split])
        for split in SPLIT_ORDER
    }
    train_labels = [int(record["cvm_stage"]) - 1 for record in splits["train"]]
    missing_classes = sorted(set(range(6)) - set(train_labels))
    if missing_classes:
        names = ", ".join(f"CVMI-{index + 1}" for index in missing_classes)
        raise ValueError(f"Training split is missing required CVM classes: {names}")
    weights = _balanced_class_weights(train_labels)

    output_dir = Path(args.output_dir).expanduser().resolve()
    counts_path = output_dir / "cvmi_stage_counts.png"
    weights_path = output_dir / "cvmi_sample_weights.png"
    _save_stage_counts(counts, split_sizes, counts_path)
    _save_sample_weights(weights, weights_path)

    print(f"Saved stage-count figure: {counts_path}")
    print(f"Saved sample-weight figure: {weights_path}")
    print("Split counts: " + "; ".join(
        f"{split}: n={split_sizes[split]}" for split in SPLIT_ORDER
    ) + f"; total={sum(split_sizes.values())}")
    print("Stage counts: " + "; ".join(
        f"{split}: " + ", ".join(f"CVMI-{stage}={counts[split].get(stage, 0)}" for stage in STAGES)
        for split in SPLIT_ORDER
    ))
    print("Smoothed weights: " + ", ".join(
        f"CVMI-{stage}={weights[stage - 1]:.4f}" for stage in STAGES
    ))


if __name__ == "__main__":
    main()
