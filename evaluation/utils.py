import numpy as np
import matplotlib.pyplot as plt

from evaluation.metrics import evaluate, robustness_gap, macro_mean_cer, cer, mean_cer, exact_match


def plot_loss(history):
    """
    Plot the training and validation loss and accuracy.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="val")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(history.history["accuracy"], label="train")
    ax[1].plot(history.history["val_accuracy"], label="val")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()


def print_results_table(results_dict):
    """
    Print a summary table comparing all models side by side.

    Parameters
    ----------
    results_dict : dict
        {model_name: evaluate() output} for each model.

    Example output:
        Model       Accuracy    CER     Precision  Recall   F1
        ──────────────────────────────────────────────────────
        cnocr        85.2%    0.148      83.1%     82.4%   82.7%
        anchor       91.3%    0.087      90.5%     89.8%   90.1%
        ...
    """
    header = f"{'Model':<12} {'Accuracy':>10} {'CER':>8} {'Precision':>11} {'Recall':>8} {'F1':>8}"
    print(header)
    print("─" * len(header))
    for name, result in results_dict.items():
        print(
            f"{name:<12}"
            f" {result['exact_match']:>10.1%}"
            f" {result['mean_cer']:>8.3f}"
            f" {result.get('precision', 0):>11.1%}"
            f" {result.get('recall', 0):>8.1%}"
            f" {result.get('f1', 0):>8.1%}"
        )


def print_perturbation_table(results_dict):
    """
    Print per-perturbation accuracy for all models in a single table.

    Rows = perturbation types, Columns = models (exact match accuracy).

    Parameters
    ----------
    results_dict : dict
        {model_name: evaluate() output} for each model.
    """
    model_names = list(results_dict.keys())

    # Collect all perturbation keys
    all_keys = set()
    for result in results_dict.values():
        all_keys.update(result.get("by_perturbation", {}).keys())
    clean = ["clean"] if "clean" in all_keys else []
    pert_keys = clean + sorted(k for k in all_keys if k != "clean")

    col_w = 10
    header = f"{'Perturbation':<22}" + "".join(f"{m:>{col_w}}" for m in model_names)
    print(header)
    print("─" * len(header))

    for key in pert_keys:
        row = f"{key:<22}"
        for name in model_names:
            bp = results_dict[name].get("by_perturbation", {})
            val = bp.get(key, {}).get("exact_match", float("nan"))
            row += f"{val:>{col_w}.1%}" if not np.isnan(val) else f"{'N/A':>{col_w}}"
        print(row)


def print_robustness_table(results_dict):
    """
    Print robustness gap (accuracy drop from clean) for all models.

    Rows = perturbation types, Columns = models.
    Positive value = model got worse vs clean.

    Parameters
    ----------
    results_dict : dict
        {model_name: evaluate() output} for each model.
    """
    model_names = list(results_dict.keys())

    all_keys = set()
    for result in results_dict.values():
        bp = result.get("by_perturbation", {})
        all_keys.update(k for k in bp if k != "clean")
    pert_keys = sorted(all_keys)

    col_w = 10
    header = f"{'Perturbation':<22}" + "".join(f"{m:>{col_w}}" for m in model_names)
    print("\nRobustness Gap (accuracy drop from clean — higher = more fragile)")
    print(header)
    print("─" * len(header))

    for key in pert_keys:
        row = f"{key:<22}"
        for name in model_names:
            bp = results_dict[name].get("by_perturbation", {})
            if "clean" not in bp or key not in bp:
                row += f"{'N/A':>{col_w}}"
                continue
            drop = bp["clean"]["exact_match"] - bp[key]["exact_match"]
            row += f"{drop:>{col_w}.1%}"
        print(row)
