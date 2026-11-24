"""Generate a publication-ready figure from saved CVM evaluation metrics.

Run from the repository root:
    python backend/evaluation_metrics_plot.py --dataset "C:\\Users\\Bhumi\\Desktop\\Aariz"

The figure is regenerated from evaluation_metrics.json and the dataset loader;
metrics are never copied into this script as constants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model import DEFAULT_DATASET_DIR, load_aariz_dataset


DEFAULT_METRICS = Path(__file__).resolve().parent / "artifacts" / "evaluation_metrics.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "artifacts" / "evaluation_metrics.png"
STAGES = [f"CVMI-{index}" for index in range(1, 7)]
NAVY = "#244B68"
TEAL = "#2A8C82"
SLATE = "#516A80"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved CleftSkel-AI validation/test metrics.")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS), help="Saved evaluation_metrics.json path.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR), help="Aariz root used for split counts.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Destination PNG path.")
    args = parser.parse_args()

    metrics_path = Path(args.metrics).expanduser().resolve()
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Evaluation metrics JSON not found: {metrics_path}")
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    cvm = saved.get("cvm", {})
    validation = cvm.get("validation")
    test = cvm.get("test")
    if not isinstance(validation, dict) or not isinstance(test, dict):
        raise ValueError("Metrics JSON must contain cvm.validation and cvm.test objects.")

    splits = load_aariz_dataset(args.dataset)
    split_counts = {name: len(records) for name, records in splits.items()}
    if not all(name in split_counts for name in ("train", "valid", "test")):
        raise ValueError(f"Expected train/valid/test splits; found {split_counts}.")
    if split_counts["valid"] != validation.get("sample_count") or split_counts["test"] != test.get("sample_count"):
        raise ValueError(
            "Evaluation sample counts do not match the current Aariz valid/test splits: "
            f"metrics={validation.get('sample_count')}/{test.get('sample_count')}, "
            f"dataset={split_counts['valid']}/{split_counts['test']}. Re-run --evaluate first."
        )

    overall_labels = ["Accuracy", "Macro F1"]
    validation_overall = [
        validation["overall_accuracy_percent"], validation["macro_f1_percent"],
    ]
    test_overall = [test["overall_accuracy_percent"], test["macro_f1_percent"]]

    val_stages = validation.get("per_stage_accuracy", {})
    test_stages = test.get("per_stage_accuracy", {})
    val_stage_values = [val_stages.get(stage, {}).get("accuracy_percent", 0.0) for stage in STAGES]
    test_stage_values = [test_stages.get(stage, {}).get("accuracy_percent", 0.0) for stage in STAGES]
    stage_supports = [
        f"{val_stages.get(stage, {}).get('support', 0)}/{test_stages.get(stage, {}).get('support', 0)}"
        for stage in STAGES
    ]

    val_age = validation.get("skeletal_age_regression", {})
    test_age = test.get("skeletal_age_regression", {})
    if val_age.get("status") != "evaluated" or test_age.get("status") != "evaluated":
        raise ValueError("Age regression metrics are absent; regenerate the saved evaluation JSON first.")
    age_labels = ["MAE", "RMSE"]
    val_age_values = [val_age["mae_years"], val_age["rmse_years"]]
    test_age_values = [test_age["mae_years"], test_age["rmse_years"]]

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.8), dpi=300)
    fig.patch.set_facecolor("white")
    positions = np.arange(2)
    width = 0.34

    for ax, labels, first, second, title, ylabel, ymax in (
        (axes[0], overall_labels, validation_overall, test_overall,
         "Overall performance", "Percent", 100),
        (axes[2], age_labels, val_age_values, test_age_values,
         "Skeletal-age proxy error", "Years", None),
    ):
        bars_a = ax.bar(positions - width / 2, first, width, label="Validation", color=NAVY)
        bars_b = ax.bar(positions + width / 2, second, width, label="Test", color=TEAL)
        age_panel = ax is axes[2]
        value_format = "%.3f" if age_panel else "%.2f"
        ax.bar_label(bars_a, fmt=value_format, padding=2, fontsize=8)
        ax.bar_label(bars_b, fmt=value_format, padding=2, fontsize=8)
        ax.set_xticks(positions, labels)
        ax.set_title(title, color=NAVY, fontsize=12, weight="bold", pad=12)
        ax.set_ylabel(ylabel)
        if ymax is not None:
            ax.set_ylim(0, ymax)
        else:
            ax.set_ylim(0, max(val_age_values + test_age_values) * 1.25)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8)

    positions = np.arange(len(STAGES))
    bars_val = axes[1].bar(positions - width / 2, val_stage_values, width, label="Validation", color=NAVY)
    bars_test = axes[1].bar(positions + width / 2, test_stage_values, width, label="Test", color=TEAL)
    axes[1].bar_label(bars_val, fmt="%.2f", padding=2, fontsize=7)
    axes[1].bar_label(bars_test, fmt="%.2f", padding=2, fontsize=7)
    axes[1].set_xticks(
        positions,
        [f"{stage}\n(n={support})" for stage, support in zip(STAGES, stage_supports)],
    )
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Per-stage accuracy", color=NAVY, fontsize=12, weight="bold", pad=12)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.tick_params(axis="both", labelsize=8)

    split_summary = (
        f"Aariz split: train {split_counts['train']} / validation {split_counts['valid']} / "
        f"test {split_counts['test']} (N={sum(split_counts.values())}); "
        f"Cohen's kappa: {validation['cohen_kappa']:.4f} validation / {test['cohen_kappa']:.4f} test"
    )
    fig.suptitle("CleftSkel-AI Preserved Baseline Evaluation", fontsize=16, weight="bold", color=NAVY, y=0.98)
    interpretation = "Age errors use synthetic stage-map proxy targets; they are not validated clinical age estimates."
    fig.text(0.5, 0.055, split_summary, ha="center", fontsize=9, color=SLATE)
    fig.text(0.5, 0.025, interpretation, ha="center", fontsize=8, color=SLATE, style="italic")
    fig.tight_layout(rect=(0.02, 0.11, 0.98, 0.91), w_pad=2.0)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved evaluation metrics figure: {output_path}")
    print(split_summary)
    print(
        "Validation/test accuracy %.2f%%/%.2f%%; macro-F1 %.2f%%/%.2f%%; "
        "age MAE %.3f/%.3f years; RMSE %.3f/%.3f years."
        % (
            validation["overall_accuracy_percent"], test["overall_accuracy_percent"],
            validation["macro_f1_percent"], test["macro_f1_percent"],
            val_age["mae_years"], test_age["mae_years"],
            val_age["rmse_years"], test_age["rmse_years"],
        )
    )


if __name__ == "__main__":
    main()
