"""
metrics.py
──────────────────────────────────────────────────────────────────────────────
Evaluation metrics

Models evaluated:
  - ANCHOR, DINOv2  →  classifier models, output a single character
  - TrOCR, CnOCR    →  sequence models, output a full text line

Core metrics (work for all four models):
  1. Character Error Rate (CER)
       edit_distance(pred, gt) / len(gt), capped at 1.0. Lower is better.
       Standard OCR metric — handles single-char and multi-char outputs equally.

  2. Exact Match (EM)
       Fraction of samples where pred == gt exactly. Stricter than CER.

Optional metrics:
  3. Robustness Gap  →  robustness_gap()
       Per-perturbation drop in EM and increase in CER relative to clean.
       Shows which perturbation types hurt each model the most.

  4. Macro CER  →  macro_mean_cer()
       Unweighted average CER across perturbation buckets.
       Use when buckets are imbalanced and you don't want large buckets
       to dominate the overall score.

Usage:
  results = evaluate(predictions, y_test, perturbation_types=perturbation_type_test)
  gaps    = robustness_gap(results["by_perturbation"])
  macro   = macro_mean_cer(results["by_perturbation"])
"""

import numpy as np


def _edit_distance(a, b):
    """Levenshtein distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j - 1], prev[j], dp[j - 1])
    return dp[n]


def cer(pred, gt):
    """Character Error Rate for a single sample, capped at 1.0."""
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return min(_edit_distance(pred, gt) / len(gt), 1.0)


def mean_cer(predictions, y_test):
    """Mean CER across all samples."""
    return float(np.mean([cer(p, g) for p, g in zip(predictions, y_test)]))


def exact_match(predictions, y_test):
    """Fraction of predictions that exactly match ground truth."""
    return float(np.mean([p == g for p, g in zip(predictions, y_test)]))


def evaluate(predictions, y_test, perturbation_types=None):
    """
    Compute overall and per-perturbation metrics.

    Parameters
    ----------
    predictions : array-like of str
        Model output strings, one per sample.
    y_test : array-like of str
        Ground truth strings.
    perturbation_types : array-like or None
        Per-sample perturbation label (None = clean). When provided,
        results will include a 'by_perturbation' breakdown.

    Returns
    -------
    dict with keys:
        'exact_match'     : float
        'mean_cer'        : float
        'by_perturbation' : dict (only when perturbation_types is provided)
            keyed by perturbation label or 'clean', each value is
            {'exact_match': float, 'mean_cer': float, 'n': int}
    """
    results = {
        "exact_match": exact_match(predictions, y_test),
        "mean_cer": mean_cer(predictions, y_test),
    }

    if perturbation_types is not None:
        preds = np.asarray(predictions, dtype=object)
        gt = np.asarray(y_test, dtype=object)
        pt = np.asarray(perturbation_types, dtype=object)

        by_perturbation = {}
        # 'clean' first, then alphabetical perturbation names
        unique = sorted(set(pt), key=lambda x: ("" if x is None else x))
        for label in unique:
            mask = pt == label
            key = "clean" if label is None else label
            by_perturbation[key] = {
                "exact_match": exact_match(preds[mask], gt[mask]),
                "mean_cer": mean_cer(preds[mask], gt[mask]),
                "n": int(mask.sum()),
            }

        results["by_perturbation"] = by_perturbation

    return results


def robustness_gap(by_perturbation):
    """
    Compute performance drop from clean to each perturbation type.

    Parameters
    ----------
    by_perturbation : dict
        The 'by_perturbation' value from evaluate(). Must include a 'clean' key.

    Returns
    -------
    dict keyed by perturbation name, each value is:
        'exact_match_drop' : float  (positive = model got worse)
        'cer_increase'     : float  (positive = model got worse)
        'n'                : int
    """
    if "clean" not in by_perturbation:
        raise ValueError("by_perturbation must include a 'clean' key")

    clean = by_perturbation["clean"]
    return {
        key: {
            "exact_match_drop": clean["exact_match"] - stats["exact_match"],
            "cer_increase": stats["mean_cer"] - clean["mean_cer"],
            "n": stats["n"],
        }
        for key, stats in by_perturbation.items()
        if key != "clean"
    }


def macro_mean_cer(by_perturbation):
    """
    Unweighted average CER across all perturbation groups (including clean).

    Unlike overall mean_cer (sample-weighted), this treats every perturbation
    type equally regardless of bucket size.
    """
    return float(np.mean([s["mean_cer"] for s in by_perturbation.values()]))
