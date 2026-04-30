"""
failure_analysis.py
───────────────────
Tools for understanding where and why each model fails.

Functions:
  - show_failures()         : display a grid of misclassified images
  - confusion_summary()     : top-N most confused character pairs
  - per_class_accuracy()    : which characters each model struggles with most
  - compare_model_failures(): which images ALL models fail on vs only one
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def show_failures(
    X_test,
    y_test,
    predictions,
    model_name,
    perturbation_types=None,
    n=12,
    random_state=42,
):
    """
    Display a grid of images the model got wrong.

    Parameters
    ----------
    X_test : np.ndarray of shape (n_samples,)
        Object array of images.
    y_test : np.ndarray of shape (n_samples,)
        Ground truth labels.
    predictions : np.ndarray of shape (n_samples,)
        Model predictions.
    model_name : str
        Label for the plot title.
    perturbation_types : np.ndarray or None
        Per-sample perturbation label. If provided, shown in subtitle.
    n : int
        Max number of failure examples to show.
    random_state : int
        Seed for reproducible sampling.
    """
    preds = np.asarray(predictions, dtype=object)
    gt    = np.asarray(y_test, dtype=object)
    X     = np.asarray(X_test, dtype=object)

    wrong_idx = np.where(preds != gt)[0]

    if len(wrong_idx) == 0:
        print(f"[{model_name}] No failures found!")
        return

    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(wrong_idx, size=min(n, len(wrong_idx)), replace=False)

    ncols = 4
    nrows = int(np.ceil(len(sample_idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = np.array(axes).reshape(-1)

    for i, idx in enumerate(sample_idx):
        ax = axes[i]
        ax.imshow(X[idx])
        ax.axis("off")
        pert = ""
        if perturbation_types is not None:
            p = perturbation_types[idx]
            pert = f"\npert: {p}" if p is not None else "\nclean"
        ax.set_title(
            f"GT: {gt[idx]}  Pred: {preds[idx]}{pert}",
            fontsize=8,
            color="red",
        )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{model_name} — failure examples", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def confusion_summary(y_test, predictions, model_name, top_n=10):
    """
    Show the top-N most confused character pairs.

    A confused pair is (ground_truth, prediction) where they differ.
    This tells you whether the model has systematic confusions
    (e.g. always confusing 人 with 入) vs random errors.

    Parameters
    ----------
    y_test : array-like of str
    predictions : array-like of str
    model_name : str
    top_n : int
        Number of most common confusion pairs to show.

    Returns
    -------
    list of ((gt, pred), count) sorted by count descending
    """
    preds = np.asarray(predictions, dtype=object)
    gt    = np.asarray(y_test, dtype=object)

    wrong_mask = preds != gt
    confused_pairs = list(zip(gt[wrong_mask], preds[wrong_mask]))
    counts = Counter(confused_pairs).most_common(top_n)

    print(f"\n{model_name} — top {top_n} confused pairs (ground truth → prediction):")
    print(f"{'GT':>6}  {'Pred':>6}  {'Count':>6}")
    print("-" * 24)
    for (g, p), c in counts:
        print(f"{g:>6}  {p:>6}  {c:>6}")

    return counts


def per_class_accuracy(y_test, predictions, model_name, worst_n=15):
    """
    Show which characters the model struggles with most.

    Computes per-character accuracy and prints the worst_n characters
    ranked by accuracy ascending (worst first).

    Parameters
    ----------
    y_test : array-like of str
    predictions : array-like of str
    model_name : str
    worst_n : int
        How many worst-performing characters to show.

    Returns
    -------
    dict mapping character → accuracy float
    """
    preds = np.asarray(predictions, dtype=object)
    gt    = np.asarray(y_test, dtype=object)

    chars = np.unique(gt)
    acc_by_char = {}
    for char in chars:
        mask = gt == char
        acc_by_char[char] = float(np.mean(preds[mask] == gt[mask]))

    ranked = sorted(acc_by_char.items(), key=lambda x: x[1])

    print(f"\n{model_name} — {worst_n} hardest characters:")
    print(f"{'Char':>6}  {'Accuracy':>10}  {'n':>6}")
    print("-" * 28)
    for char, acc in ranked[:worst_n]:
        n = int(np.sum(gt == char))
        print(f"{char:>6}  {acc:>10.1%}  {n:>6}")

    return acc_by_char


def compare_model_failures(y_test, predictions_dict):
    """
    Categorize test samples by how many models got them wrong.

    Useful for identifying:
    - Images ALL models fail on → likely a hard/ambiguous character
    - Images only ONE model fails on → that model's specific weakness

    Parameters
    ----------
    y_test : array-like of str
        Ground truth labels.
    predictions_dict : dict of str → array-like of str
        Keys are model names, values are prediction arrays.
        e.g. {"cnocr": preds_cnocr, "anchor": preds_anchor, ...}

    Returns
    -------
    dict with keys:
        'all_wrong'  : indices where every model failed
        'all_correct': indices where every model succeeded
        'mixed'      : dict mapping model_name → indices only that model got wrong
    """
    gt = np.asarray(y_test, dtype=object)
    n  = len(gt)

    wrong_sets = {}
    for name, preds in predictions_dict.items():
        p = np.asarray(preds, dtype=object)
        wrong_sets[name] = set(np.where(p != gt)[0])

    all_wrong   = set.intersection(*wrong_sets.values())
    all_correct = set(range(n)) - set.union(*wrong_sets.values())

    print(f"\nModel failure overlap ({n} total samples):")
    print(f"  All models wrong:   {len(all_wrong):>5}  ({len(all_wrong)/n:.1%})")
    print(f"  All models correct: {len(all_correct):>5}  ({len(all_correct)/n:.1%})")

    mixed = {}
    for name, wrong in wrong_sets.items():
        others_wrong = set.union(*[v for k, v in wrong_sets.items() if k != name])
        only_this_model = wrong - others_wrong
        mixed[name] = np.array(sorted(only_this_model))
        print(f"  Only {name} wrong:     {len(only_this_model):>5}  ({len(only_this_model)/n:.1%})")

    return {
        "all_wrong":   np.array(sorted(all_wrong)),
        "all_correct": np.array(sorted(all_correct)),
        "mixed":       mixed,
    }
