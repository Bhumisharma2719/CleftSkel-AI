"""Render a publication-ready CleftSkel-AI pipeline diagram.

Run from the repository root with:
    python backend/create_architecture_diagram.py

The image is written to backend/artifacts/architecture.png at 300 DPI.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUTPUT = Path(__file__).parent / "artifacts" / "architecture.png"

NAVY = "#18324B"
SLATE = "#516A80"
BLUE = "#DCEAF4"
PALE = "#F3F6F9"
TEAL = "#DDF1EE"
GOLD = "#F6EBD2"
INK = "#243746"
WHITE = "#FFFFFF"


def add_box(ax, center, size, title, detail, *, face=PALE, edge=SLATE, title_size=13):
    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x, y + height * 0.16, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=NAVY, zorder=3)
    ax.text(x, y - height * 0.19, detail, ha="center", va="center",
            fontsize=9.5, color=INK, linespacing=1.25, zorder=3)
    return patch


def add_arrow(ax, start, end, *, color=SLATE, rad=0.0, linewidth=1.65):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=linewidth, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=1,
        shrinkA=4, shrinkB=4,
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(12.8, 9.2), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0.035, right=0.965, top=0.94, bottom=0.045)

    ax.text(0.5, 0.978, "CleftSkel-AI System Architecture", ha="center", va="top",
            fontsize=20, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.945, "Lateral cephalogram analysis with multi-task learning and explainability",
            ha="center", va="top", fontsize=10.5, color=SLATE)

    # Main sequential image-processing and feature-extraction pathway.
    add_box(ax, (0.5, 0.865), (0.62, 0.105),
            "Input: Lateral Cephalogram", "X-ray image  •  959 × 1198 px", face=BLUE)
    add_box(ax, (0.5, 0.705), (0.72, 0.115),
            "Preprocessing & Augmentation",
            "Resize to 224 × 224  •  RGB normalization\nTraining augmentation: rotation and zoom",
            face=PALE)
    add_box(ax, (0.5, 0.535), (0.72, 0.115),
            "MobileNetV2 Feature Extractor",
            "ImageNet transfer learning  •  fine-tuned top layers",
            face=BLUE)

    add_arrow(ax, (0.5, 0.81), (0.5, 0.765))
    add_arrow(ax, (0.5, 0.645), (0.5, 0.595))

    # Parallel task-specific heads.
    add_box(ax, (0.285, 0.345), (0.40, 0.135),
            "Branch A — CVM Classification",
            "Softmax probabilities over\nCVMI-1 · CVMI-2 · CVMI-3 · CVMI-4 · CVMI-5 · CVMI-6",
            face=TEAL, edge="#3E8B83", title_size=12)
    add_box(ax, (0.715, 0.345), (0.40, 0.135),
            "Branch B — Skeletal Age Regression",
            "Continuous skeletal-age estimate\n(years; proxy target when only CVM labels exist)",
            face=GOLD, edge="#B88A3B", title_size=12)

    add_arrow(ax, (0.43, 0.475), (0.32, 0.415), rad=0.05)
    add_arrow(ax, (0.57, 0.475), (0.68, 0.415), rad=-0.05)

    # Explainability then clinical dossier output.
    add_box(ax, (0.285, 0.115), (0.40, 0.13),
            "Explainability — Grad-CAM",
            "Class-activation heatmap overlay\nfor visual review of model attention",
            face=PALE, edge=SLATE, title_size=12)
    add_box(ax, (0.715, 0.115), (0.40, 0.13),
            "Clinical Output — ReportLab PDF Dossier",
            "CVM stage + confidence  •  skeletal-age estimate\nX-ray / heatmap previews  •  timestamp and disclaimer",
            face=BLUE, edge=NAVY, title_size=11.5)

    add_arrow(ax, (0.285, 0.275), (0.285, 0.185), color="#3E8B83")
    add_arrow(ax, (0.375, 0.30), (0.62, 0.18), color=SLATE, rad=-0.10)
    add_arrow(ax, (0.715, 0.275), (0.715, 0.185), color="#B88A3B")
    add_arrow(ax, (0.49, 0.115), (0.51, 0.115), color=NAVY)

    ax.text(0.5, 0.027,
            "Research prototype — outputs require validation by a qualified clinician.",
            ha="center", va="center", fontsize=8.5, color=SLATE, style="italic")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Saved architecture diagram: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
