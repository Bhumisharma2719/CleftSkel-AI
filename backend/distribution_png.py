"""Plot actual CVM class counts and balancing weights for each Aariz split.

Run from the repository root:
    python backend/distribution_png.py

The script uses the same dataset loader and scikit-learn class-weight function
as model.py, avoiding stale hard-coded counts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model import DEFAULT_DATASET_DIR, _balanced_class_weights, load_aariz_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "distribution_polished.png"
STAGES = [1, 2, 3, 4, 5, 6]
SPLIT_COLORS = {"train": "#244B68", "valid": "#2A8C82", "test": "#C18A35"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dataset CVM stage distribution and training weights.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR), help="Aariz dataset root.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Destination PNG path.")
    args = parser.parse_args()

    splits = load_aariz_dataset(args.dataset)
    counts = {
        split: Counter(int(record["cvm_stage"]) for record in records)
        for split, records in splits.items()
    }
    train_labels = [int(record["cvm_stage"]) - 1 for record in splits["train"]]
    missing_stages = sorted(set(range(6)) - set(train_labels))
    if missing_stages:
        missing_names = ", ".join(f"CVMI-{stage + 1}" for stage in missing_stages)
        raise ValueError(f"Training split is missing required CVM classes: {missing_names}")
    class_weights = _balanced_class_weights(train_labels)
    plotted_weights = [class_weights[stage - 1] for stage in STAGES]

    fig, (count_ax, weight_ax) = plt.subplots(
        1, 2, figsize=(12.6, 5.6), dpi=300,
        gridspec_kw={"width_ratios": [1.65, 1.0]},
    )
    fig.patch.set_facecolor("white")
    stages_x = np.arange(len(STAGES))
    splits_to_plot = [name for name in ("train", "valid", "test") if name in counts]
    bar_width = 0.78 / max(1, len(splits_to_plot))

    for split_index, split in enumerate(splits_to_plot):
        values = [counts[split].get(stage, 0) for stage in STAGES]
        offset = (split_index - (len(splits_to_plot) - 1) / 2) * bar_width
        bars = count_ax.bar(
            stages_x + offset, values, width=bar_width * 0.92,
            label=f"{split.title()} (n={sum(values)})",
            color=SPLIT_COLORS[split], edgecolor="white", linewidth=0.8,
        )
        count_ax.bar_label(bars, padding=2, fontsize=7.3, color="#273746")

    count_ax.set_title("CVM stage counts by dataset split", fontsize=12, weight="bold", color="#18324B", pad=12)
    count_ax.set_xlabel("Cervical vertebral maturation stage", fontsize=9.5, labelpad=8)
    count_ax.set_ylabel("Matched labeled images", fontsize=9.5, labelpad=8)
    count_ax.set_xticks(stages_x, [f"CVMI-{stage}" for stage in STAGES])
    count_ax.tick_params(axis="both", labelsize=8.5)
    count_ax.grid(axis="y", linestyle="--", alpha=0.3)
    count_ax.set_axisbelow(True)
    count_ax.legend(frameon=False, fontsize=8, loc="upper left")

    weights_bars = weight_ax.bar(
        [f"CVMI-{stage}" for stage in STAGES], plotted_weights,
        color=["#587D9A" if value <= 1 else "#2A8C82" for value in plotted_weights],
        edgecolor="white", linewidth=0.8,
    )
    weight_ax.bar_label(weights_bars, labels=[f"{value:.2f}" for value in plotted_weights], padding=3, fontsize=8)
    weight_ax.axhline(1.0, color="#C18A35", linewidth=1.2, linestyle="--", label="Neutral weight = 1")
    weight_ax.set_title("Balanced training class weights", fontsize=12, weight="bold", color="#18324B", pad=12)
    weight_ax.set_xlabel("Cervical vertebral maturation stage", fontsize=9.5, labelpad=8)
    weight_ax.set_ylabel("Scikit-learn balanced weight", fontsize=9.5, labelpad=8)
    weight_ax.tick_params(axis="both", labelsize=8.3)
    weight_ax.grid(axis="y", linestyle="--", alpha=0.3)
    weight_ax.set_axisbelow(True)
    weight_ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("Aariz CVM Training Distribution and Imbalance Weights", fontsize=15, weight="bold", color="#18324B", y=0.99)
    fig.text(
        0.5, 0.015,
        "Counts use matched image-annotation records. Class weights are computed on the training split; focal alpha and per-example weights are derived from them.",
        ha="center", va="bottom", fontsize=7.5, color="#516A80",
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.94), w_pad=2.6)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    split_summary = "; ".join(
        f"{split}: " + ", ".join(f"CVMI-{stage}={counts[split].get(stage, 0)}" for stage in STAGES)
        for split in splits_to_plot
    )
    print(f"Saved distribution plot: {output_path}")
    print(split_summary)
    print("Training weights: " + ", ".join(f"CVMI-{stage}={class_weights[stage - 1]:.4f}" for stage in STAGES))


if __name__ == "__main__":
    main()
