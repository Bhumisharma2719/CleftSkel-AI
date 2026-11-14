"""Render a publication-ready diagram of the configured CleftSkel-AI model.

Run from the repository root:
    python backend/create_architecture_diagram.py

Writes backend/artifacts/architecture.png at 300 DPI.
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


def add_box(ax, center, size, title, detail, *, face=PALE, edge=SLATE, title_size=12):
    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.005,rounding_size=0.018",
        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x, y + height * 0.23, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=NAVY, zorder=3)
    ax.text(x, y - height * 0.13, detail, ha="center", va="center",
            fontsize=8.4, color=INK, linespacing=1.25, zorder=3)


def add_arrow(ax, start, end, *, color=SLATE, rad=0.0, linewidth=1.55):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13,
        linewidth=linewidth, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=1,
        shrinkA=4, shrinkB=4,
    ))


def main():
    fig, ax = plt.subplots(figsize=(12.8, 10.0), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0.035, right=0.965, top=0.94, bottom=0.04)

    ax.text(0.5, 0.978, "CleftSkel-AI System Architecture", ha="center", va="top",
            fontsize=19, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.946, "Multi-task CVM stage and skeletal-age proxy analysis",
            ha="center", va="top", fontsize=10, color=SLATE)

    # Sequential image processing and shared feature extraction.
    add_box(ax, (0.5, 0.87), (0.66, 0.105),
            "Input: Lateral Cephalogram X-ray",
            "Native resolution varies by source image",
            face=BLUE)
    add_box(ax, (0.5, 0.715), (0.80, 0.155),
            "Preprocessing and Training Augmentation",
            "Resize: 224 x 224 RGB; MobileNetV2 preprocessing\nGeneral: +/-18 deg; shifts 0.12; zoom 0.20; contrast 0.88-1.12\nCVMI-3: +/-20 deg; shifts 0.15; zoom 0.22; contrast 0.78-1.22",
            face=PALE)
    add_box(ax, (0.5, 0.535), (0.78, 0.13),
            "MobileNetV2 Shared Feature Extractor",
            "Phase 1: frozen-backbone warm-up; 10 epochs at Adam 1e-3\nPhase 2: top 30 layers selected; batch normalization frozen\nAdam 1e-5 with CosineDecay; batch size 16; maximum 100 epochs",
            face=BLUE)
    add_arrow(ax, (0.5, 0.8175), (0.5, 0.7925))
    add_arrow(ax, (0.5, 0.6375), (0.5, 0.60))

    # Parallel task outputs.
    add_box(ax, (0.275, 0.345), (0.42, 0.15),
            "Branch A: CVM Classification",
            "6-class softmax: CVMI-1 through CVMI-6\nBalanced sparse focal loss (gamma=1.5)\nCapped class alpha and softened sample weights",
            face=TEAL, edge="#3E8B83", title_size=11.5)
    add_box(ax, (0.725, 0.345), (0.42, 0.15),
            "Branch B: Skeletal-Age Regression",
            "Parallel linear output in years\nDefault training targets are stage proxies:\n9.0, 10.5, 12.0, 13.5, 15.0, 16.0",
            face=GOLD, edge="#B88A3B", title_size=11.5)
    add_arrow(ax, (0.43, 0.47), (0.32, 0.42), rad=0.04)
    add_arrow(ax, (0.57, 0.47), (0.68, 0.42), rad=-0.04)

    # XAI and report outputs.
    add_box(ax, (0.275, 0.155), (0.42, 0.12),
            "Explainability: Grad-CAM",
            "Coarse activation overlay for visual review",
            face=PALE, edge=SLATE, title_size=11.5)
    add_box(ax, (0.725, 0.155), (0.42, 0.12),
            "Clinical Research PDF Dossier",
            "Stage and confidence; age and proxy-source note; image/heatmap\nOptional CVA result; metadata; timestamp; clinician disclaimer",
            face=BLUE, edge=NAVY, title_size=11.5)
    add_arrow(ax, (0.275, 0.27), (0.275, 0.22), color="#3E8B83")
    add_arrow(ax, (0.37, 0.275), (0.63, 0.215), color=SLATE, rad=-0.08)
    add_arrow(ax, (0.725, 0.27), (0.725, 0.22), color="#B88A3B")
    add_arrow(ax, (0.485, 0.155), (0.515, 0.155), color=NAVY)

    ax.text(
        0.5, 0.065,
        "Best validation-loss checkpoint; early stopping patience 20; "
        "warm-up ReduceLROnPlateau factor 0.2, patience 5; cosine decay during fine-tuning.",
        ha="center", va="center", fontsize=8.2, color=SLATE,
    )
    ax.text(0.5, 0.028,
            "Research prototype. Stage-mapped ages are synthetic proxies; clinician validation is required.",
            ha="center", va="center", fontsize=8.1, color=SLATE, style="italic")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Saved architecture diagram: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
