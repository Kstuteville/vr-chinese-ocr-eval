"""
perturbation_analysis.py
────────────────────────
Visual analysis of all perturbation types on real CASIA samples.

What this script does:
  1. Loads a small number of real images from the CASIA-HWDB2-line dataset.
  2. Applies every perturbation type (one at a time) to each image.
  3. Displays a grid: rows = perturbation type, columns = sample images.
     The first column is always the original (clean) image.

This serves two purposes:
  - Validate that each perturbation looks realistic for the Meta Quest / 
    whiteboard scenario.
  - Generate a figure ready to drop into the project presentation.

Usage (in your notebook):
  %run perturbation_analysis.py
  # or import and call show_perturbation_grid()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import the perturbation pipeline
# Adjust the import path if running from inside /notebooks
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perturbations.pipeline import perturbate_data, PERTURBATION_TYPES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_object_array(items):
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def _ensure_uint8_rgb(img):
    """Convert any PIL Image or numpy array to a uint8 RGB numpy array."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:                          # grayscale → RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:                      # RGBA → RGB
        arr = arr[:, :, :3]
    return arr


def _apply_single_perturbation(images, perturbation_name, random_state=42):
    """
    Apply one specific perturbation type to a list of images.

    Parameters
    ----------
    images : list of np.ndarray (uint8, H×W×3)
    perturbation_name : str  — must be in PERTURBATION_TYPES
    random_state : int

    Returns
    -------
    list of np.ndarray (uint8, H×W×3)
    """
    # Build a plan that applies the same single perturbation to every image
    plan = np.array(
        [(perturbation_name,)] * len(images),
        dtype=object
    )

    dummy_labels = _to_object_array([""] * len(images))
    X_arr = _to_object_array(images)

    X_out, _, _ = perturbate_data(
        X_arr,
        dummy_labels,
        random_state=random_state,
        show_progress=False,
        perturbation_plan=plan,
    )
    return list(X_out)


# ── Main visualization ────────────────────────────────────────────────────────

def show_perturbation_grid(
    images_raw=None,
    labels_raw=None,
    n_samples=5,
    random_state=42,
    save_path=None,
):
    """
    Display a grid showing every perturbation applied to real CASIA images.

    Layout
    ──────
    Rows  : one per perturbation type  (+ 1 header row = "Original")
    Cols  : one per sample image

    The first row is always the original clean image.
    Each subsequent row shows that image after one specific perturbation.

    Parameters
    ----------
    images_raw : sequence or None
        Raw image collection (e.g. list/array of PIL Images or ndarrays).
        If None, falls back to Hugging Face CASIA dataset loading.
    labels_raw : sequence or None
        Raw labels corresponding to images_raw.
        If None, blank labels are used.
    n_samples : int
        Number of CASIA images to use as columns (default 5).
        Keep low (≤8) for readable display.
    random_state : int
        Seed for reproducibility.
    save_path : str or None
        If given, also saves the figure to this path (e.g. "analysis.png").
    """
    if images_raw is None:
        from datasets import load_dataset

        print("Loading CASIA dataset (this may take a moment the first time)...")
        ds = load_dataset("Teklia/CASIA-HWDB2-line")
        images_raw = ds["train"]["image"]
        labels_raw = ds["train"]["text"]

    images_raw = list(images_raw)
    if labels_raw is None:
        labels_raw = [""] * len(images_raw)
    else:
        labels_raw = list(labels_raw)

    if len(images_raw) == 0:
        raise ValueError("images_raw must contain at least one image.")
    if len(labels_raw) != len(images_raw):
        raise ValueError("labels_raw must have the same length as images_raw.")

    rng = np.random.default_rng(random_state)
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be >= 1.")
    if n_samples > len(images_raw):
        n_samples = len(images_raw)
    idx = rng.choice(len(images_raw), size=n_samples, replace=False)

    # Convert to uint8 RGB numpy arrays
    originals = [_ensure_uint8_rgb(images_raw[i]) for i in idx]
    sample_labels = [labels_raw[i] for i in idx]

    n_perturbations = len(PERTURBATION_TYPES)
    n_rows = n_perturbations + 1   # +1 for the "Original" row
    n_cols = n_samples

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.2, n_rows * 2.0),
    )

    # ── Row 0: Original images ────────────────────────────────────────────────
    for col, (img, label) in enumerate(zip(originals, sample_labels)):
        ax = axes[0, col]
        ax.imshow(img)
        ax.set_title(f'"{label}"', fontsize=7, pad=3)
        ax.axis("off")

    # Left-side label for the original row
    axes[0, 0].set_ylabel("Original", fontsize=9, fontweight="bold", labelpad=6)

    # ── Rows 1..N: One row per perturbation ───────────────────────────────────
    for row_idx, pert_name in enumerate(PERTURBATION_TYPES, start=1):
        print(f"  Applying: {pert_name} ...")

        perturbed = _apply_single_perturbation(
            originals, pert_name, random_state=random_state
        )

        for col, img in enumerate(perturbed):
            ax = axes[row_idx, col]
            ax.imshow(img)
            ax.axis("off")

        # Row label on the left
        axes[row_idx, 0].set_ylabel(
            pert_name.replace("_", "\n"),
            fontsize=8,
            fontweight="bold",
            labelpad=6,
            rotation=0,
            ha="right",
            va="center",
        )

    # ── Annotations ───────────────────────────────────────────────────────────
    # Color-code the two groups of perturbations in the row labels
    original_patch = mpatches.Patch(color="steelblue",  label="Original perturbations (Kezia)")
    new_patch      = mpatches.Patch(color="darkorange", label="New perturbations (Quest-specific)")

    # Highlight new perturbations with a colored background on their y-label
    new_perturbations = {"perspective_warp", "motion_blur", "white_balance_shift", "low_resolution"}
    for row_idx, pert_name in enumerate(PERTURBATION_TYPES, start=1):
        color = "darkorange" if pert_name in new_perturbations else "steelblue"
        axes[row_idx, 0].yaxis.label.set_color(color)

    fig.legend(
        handles=[original_patch, new_patch],
        loc="lower center",
        ncol=2,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "Perturbation Analysis — CASIA-HWDB2-line samples\n"
        "Simulating Meta Quest whiteboard capture conditions",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")

    plt.show()
    print("\nDone.")


# ── Run directly if executed as a script ─────────────────────────────────────
if __name__ == "__main__":
    show_perturbation_grid(n_samples=5, save_path="perturbation_analysis.png")
