"""
accuracy_curves.py
──────────────────
Visualization functions for model evaluation results.

All functions accept a results_dict structured as:
    {
        "model_name": evaluate(predictions, y_test, perturbation_types),
        ...
    }
where evaluate() is from evaluation/metrics.py.

Functions:
  - plot_accuracy_by_perturbation() : grouped bar chart, main paper figure
  - plot_robustness_gap_heatmap()   : heatmap of accuracy drop per model × perturbation
  - plot_robustness_gap_bars()      : grouped bar chart of accuracy drop (easier to read)
  - plot_clean_comparison()         : simple bar chart comparing clean accuracy across models
  - plot_f1_by_perturbation()       : same as accuracy chart but for F1 score
  - plot_overall_metrics()          : accuracy / precision / recall / F1 side by side
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from evaluation.metrics import robustness_gap


# Fixed color per model — consistent across every figure
_MODEL_COLOR_MAP = {
    "ANCHOR":       "#4C72B0",
    "PaddleOCR":    "#DD8452",
    "EasyOCR":      "#55A868",
    "CnOCR":        "#C44E52",
    "Apple Vision": "#9B59B6",
}
_FALLBACK_PALETTE = ["#6B7FB5", "#B0A0C0", "#A8C4A2", "#D4A0A0", "#C0A8D0"]


def _color(name, index=0):
    return _MODEL_COLOR_MAP.get(name, _FALLBACK_PALETTE[index % len(_FALLBACK_PALETTE)])


def _get_perturbation_keys(results_dict):
    """Extract ordered perturbation keys from any model's results."""
    for result in results_dict.values():
        if "by_perturbation" in result:
            keys = list(result["by_perturbation"].keys())
            clean = [k for k in keys if k == "clean"]
            rest  = sorted([k for k in keys if k != "clean"])
            return clean + rest
    return []


def plot_accuracy_by_perturbation(results_dict, metric="exact_match", save_path=None):
    """
    Grouped bar chart: accuracy per perturbation type, one group per perturbation,
    one bar per model. This is the main figure for the paper.
    """
    model_names  = list(results_dict.keys())
    pert_keys    = _get_perturbation_keys(results_dict)

    if not pert_keys:
        raise ValueError("results_dict must contain 'by_perturbation' keys from evaluate().")

    n_models = len(model_names)
    n_groups = len(pert_keys)
    x        = np.arange(n_groups)
    width    = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(12, n_groups * 1.4), 6))

    for i, name in enumerate(model_names):
        values = []
        for key in pert_keys:
            bp = results_dict[name].get("by_perturbation", {})
            values.append(bp.get(key, {}).get(metric, 0.0))
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name, color=_color(name, i), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [k.replace("_", "\n") for k in pert_keys],
        fontsize=9,
    )
    ax.set_xlabel("Perturbation type", fontsize=11)

    if metric == "exact_match":
        ax.set_ylabel("Accuracy (exact match)", fontsize=11)
        ax.set_title("Top-1 Accuracy by Perturbation Type", fontsize=13, fontweight="bold")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1.05)
    else:
        ax.set_ylabel("Mean CER (lower = better)", fontsize=11)
        ax.set_title("Character Error Rate by Perturbation Type", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)

    ax.legend(title="Model", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_robustness_gap_heatmap(results_dict, save_path=None):
    """
    Heatmap: rows = models, columns = perturbation types,
    cell = exact_match_drop from clean baseline.
    """
    model_names = list(results_dict.keys())
    pert_keys   = _get_perturbation_keys(results_dict)
    pert_keys   = [k for k in pert_keys if k != "clean"]

    matrix = np.zeros((len(model_names), len(pert_keys)))

    for i, name in enumerate(model_names):
        bp = results_dict[name].get("by_perturbation", {})
        if not bp or "clean" not in bp:
            continue
        gaps = robustness_gap(bp)
        for j, key in enumerate(pert_keys):
            matrix[i, j] = gaps.get(key, {}).get("exact_match_drop", 0.0)

    fig, ax = plt.subplots(figsize=(max(10, len(pert_keys) * 1.1), len(model_names) * 1.2 + 1.5))

    im = ax.imshow(matrix, cmap="Reds", aspect="auto", vmin=0, vmax=max(0.01, matrix.max()))

    ax.set_xticks(range(len(pert_keys)))
    ax.set_xticklabels([k.replace("_", "\n") for k in pert_keys], fontsize=9)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)

    for i in range(len(model_names)):
        for j in range(len(pert_keys)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Accuracy drop from clean baseline")
    ax.set_title(
        "Robustness Gap Heatmap\n(higher = model degrades more under that perturbation)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_robustness_gap_bars(results_dict, save_path=None):
    """
    Grouped bar chart of accuracy drop from clean baseline per perturbation type.

    Same data as the heatmap but easier to compare magnitudes across models.
    Positive values = model got worse. Negative = model actually improved.
    """
    model_names = list(results_dict.keys())
    pert_keys   = _get_perturbation_keys(results_dict)
    pert_keys   = [k for k in pert_keys if k != "clean"]

    n_models = len(model_names)
    n_groups = len(pert_keys)
    x        = np.arange(n_groups)
    width    = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(12, n_groups * 1.4), 6))

    for i, name in enumerate(model_names):
        bp = results_dict[name].get("by_perturbation", {})
        if not bp or "clean" not in bp:
            continue
        gaps = robustness_gap(bp)
        values = [gaps.get(k, {}).get("exact_match_drop", 0.0) for k in pert_keys]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name, color=_color(name, i), alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n") for k in pert_keys], fontsize=9)
    ax.set_xlabel("Perturbation type", fontsize=11)
    ax.set_ylabel("Accuracy drop from clean baseline", fontsize=11)
    ax.set_title(
        "Robustness Gap by Perturbation Type\n(positive = model degrades, negative = model improves)",
        fontsize=13, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.legend(title="Model", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_clean_comparison(results_dict, save_path=None):
    """
    Horizontal bar chart comparing clean accuracy across all models.
    """
    model_names = list(results_dict.keys())
    clean_acc   = []

    for name in model_names:
        bp = results_dict[name].get("by_perturbation", {})
        if bp and "clean" in bp:
            clean_acc.append(bp["clean"]["exact_match"])
        else:
            clean_acc.append(results_dict[name].get("exact_match", 0.0))

    order       = np.argsort(clean_acc)[::-1]
    model_names = [model_names[i] for i in order]
    clean_acc   = [clean_acc[i] for i in order]
    colors      = [_color(name, i) for i, name in enumerate(model_names)]

    fig, ax = plt.subplots(figsize=(7, max(3, len(model_names) * 0.9)))
    bars = ax.barh(model_names, clean_acc, color=colors, alpha=0.85)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Top-1 Accuracy (clean images)", fontsize=11)
    ax.set_title("Clean Baseline Accuracy — All Models", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, clean_acc):
        ax.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_overall_metrics(results_dict, save_path=None):
    """
    Grouped bar chart showing Accuracy, Precision, Recall, F1 side by side
    per model, with CER on a secondary y-axis.
    """
    model_names = list(results_dict.keys())
    metrics     = ["exact_match", "precision", "recall", "f1"]
    labels      = ["Accuracy", "Precision", "Recall", "F1"]

    n_models  = len(model_names)
    n_metrics = len(metrics)
    x         = np.arange(n_models)
    width     = 0.18

    fig, ax1 = plt.subplots(figsize=(max(10, n_models * 2), 6))

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = [results_dict[m].get(metric, 0.0) for m in model_names]
        offset = (i - n_metrics / 2 + 0.5) * width
        ax1.bar(x + offset, values, width, label=label, alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=11)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Overall Model Performance — Accuracy, Precision, Recall, F1",
                  fontsize=12, fontweight="bold")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    cer_values = [results_dict[m].get("mean_cer", 0.0) for m in model_names]
    ax2.plot(x, cer_values, color="black", marker="D", linewidth=2,
             markersize=7, label="CER (lower = better)", zorder=5)
    ax2.set_ylabel("Character Error Rate (CER)", fontsize=11)
    ax2.set_ylim(0, 1.2)
    ax2.legend(fontsize=9, loc="upper left")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_f1_by_perturbation(results_dict, save_path=None):
    """
    Grouped bar chart of F1 score per perturbation type, one bar per model.
    """
    model_names = list(results_dict.keys())
    pert_keys   = _get_perturbation_keys(results_dict)

    n_models = len(model_names)
    n_groups = len(pert_keys)
    x        = np.arange(n_groups)
    width    = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(12, n_groups * 1.4), 6))

    for i, name in enumerate(model_names):
        values = []
        for key in pert_keys:
            bp = results_dict[name].get("by_perturbation", {})
            values.append(bp.get(key, {}).get("f1", 0.0))
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name, color=_color(name, i), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n") for k in pert_keys], fontsize=9)
    ax.set_xlabel("Perturbation type", fontsize=11)
    ax.set_ylabel("F1 Score (macro)", fontsize=11)
    ax.set_title("F1 Score by Perturbation Type", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()
