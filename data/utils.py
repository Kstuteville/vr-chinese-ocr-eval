import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt

from perturbations.pipeline import (
    perturbate_data,
    _build_balanced_perturbation_plan_for_buckets,
    select_perturbation_buckets,
    required_perturb_count,
)


def _to_object_array(items):
    """Build a 1D object array without NumPy trying to broadcast nested shapes."""
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def load_data(
    clean_count=1500,
    perturb_count=None,
    test_size=0.2,
    random_state=124,
    return_perturbation_type=True,
    show_progress=True,
    balanced_perturbations=True,
    min_per_perturb_bucket=60,
    include_combinations=False,
    perturb_bucket_count=None,
):
    """Load CASIA, build disjoint clean/perturb pools, perturb, and split."""
    selected_buckets = select_perturbation_buckets(
        include_combinations=include_combinations,
        bucket_count=perturb_bucket_count,
        random_state=random_state,
    )

    if perturb_count is None:
        perturb_count = required_perturb_count(
            min_per_bucket=min_per_perturb_bucket,
            include_combinations=include_combinations,
            bucket_count=perturb_bucket_count,
        )

    ds = load_dataset("Teklia/CASIA-HWDB2-line")
    images = _to_object_array(ds["train"]["image"])
    labels = _to_object_array(ds["train"]["text"])

    if clean_count + perturb_count > len(images):
        raise ValueError(
            "clean_count + perturb_count must be <= available samples "
            f"({len(images)})."
        )

    rng = np.random.default_rng(random_state)
    clean_idx = rng.choice(len(images), size=clean_count, replace=False)
    X_clean = images[clean_idx]
    y_clean = labels[clean_idx]

    remaining_idx = np.setdiff1d(np.arange(len(images)), clean_idx, assume_unique=False)
    perturb_idx = rng.choice(remaining_idx, size=perturb_count, replace=False)
    X_perturb_src = images[perturb_idx]
    y_perturb_src = labels[perturb_idx]

    perturbation_plan = None
    if balanced_perturbations:
        perturbation_plan = _build_balanced_perturbation_plan_for_buckets(
            perturb_count,
            selected_buckets,
            rng,
        )

    X_perturbed, y_perturbed, pert_types = perturbate_data(
        X_perturb_src,
        y_perturb_src,
        random_state=random_state,
        show_progress=show_progress,
        perturbation_plan=perturbation_plan,
        available_buckets=selected_buckets,
    )

    X_all = np.concatenate([_to_object_array(X_clean), _to_object_array(X_perturbed)])
    y_all = np.concatenate([_to_object_array(y_clean), _to_object_array(y_perturbed)])
    clean_types = _to_object_array([None] * len(X_clean))
    perturbation_types = np.concatenate([clean_types, _to_object_array(pert_types)])

    shuffle_idx = rng.permutation(len(X_all))
    X_all = X_all[shuffle_idx]
    y_all = y_all[shuffle_idx]
    perturbation_types = perturbation_types[shuffle_idx]

    n_total = len(X_all)
    n_train = int(round((1 - test_size) * n_total))
    n_train = max(1, min(n_total - 1, n_train))

    X_train = X_all[:n_train]
    X_test = X_all[n_train:]
    y_train = y_all[:n_train]
    y_test = y_all[n_train:]
    perturbation_type_train = perturbation_types[:n_train]
    perturbation_type_test = perturbation_types[n_train:]

    if return_perturbation_type:
        return (
            X_train,
            X_test,
            y_train,
            y_test,
            perturbation_type_train,
            perturbation_type_test,
        )

    return X_train, X_test, y_train, y_test

def split_data(X, y, train_size=0.8, val_size=0.1):
    """
    Splits the input data into training and validation according
    to the given percentages, using ALL samples exactly once.

    Rules:
    - n_train = round(train_size * n)
    - n_val = n - n_train  (remainder goes to val)
    

    Parameters
    ----------
    X : np.ndarray
        The input features to be split.
    y : np.ndarray
        The input labels to be split.
    train_size : float, optional
        The proportion of data to be used for training, by default 0.9.
    val_size : float, optional
        The proportion of data to be used for validation, by default 0.1.

    Returns
    -------
    X_train : np.ndarray
        The input features to be used for training.
    X_val : np.ndarray
        The input features to be used for validation.
    y_train : np.ndarray
        The input labels to be used for training.
    y_val : np.ndarray
        The input labels to be used for validation.
    """
    # Calculate the size of training and validation sets given the percentages
    n_train = round(train_size * len(X)) # or X.shape[0] (number_of_samples, number_of_features)
    x_train = X[:n_train]
    x_val = X[n_train:]
    y_train = y[:n_train]
    y_val = y[n_train:]
    return x_train, x_val, y_train, y_val


def explore_data(X_train, y_train, y_test, y_val):
    """
    Plots label-frequency distributions in training, validation, and test sets.

    Parameters
    ----------
    X_train : np.ndarray
        A numpy array containing the features of the training set.
    y_train : np.ndarray
        A numpy array containing the labels of the training set.
    y_test : np.ndarray
        A numpy array containing the labels of the test set.
    y_val : np.ndarray
        A numpy array containing the labels of the validation set.

    Returns
    -------
    None
    """

    def _top_label_counts(y, top_n=20):
        values, counts = np.unique(np.array(y, dtype=object), return_counts=True)
        order = np.argsort(counts)[::-1]
        values = values[order][:top_n]
        counts = counts[order][:top_n]
        labels = [str(v) if v is not None else "clean" for v in values]
        return labels, counts

    # Plot train/val/test label frequencies side-by-side.
    fig, ax = plt.subplots(1, 3, figsize=(14, 5))

    train_labels, train_counts = _top_label_counts(y_train)
    ax[0].bar(range(len(train_labels)), train_counts)
    ax[0].set_xticks(range(len(train_labels)))
    ax[0].set_xticklabels(train_labels, rotation=75, ha='right', fontsize=8)
    ax[0].set_title('Training labels (top 20)')

    val_labels, val_counts = _top_label_counts(y_val)
    ax[1].bar(range(len(val_labels)), val_counts)
    ax[1].set_xticks(range(len(val_labels)))
    ax[1].set_xticklabels(val_labels, rotation=75, ha='right', fontsize=8)
    ax[1].set_title('Validation labels (top 20)')

    test_labels, test_counts = _top_label_counts(y_test)
    ax[2].bar(range(len(test_labels)), test_counts)
    ax[2].set_xticks(range(len(test_labels)))
    ax[2].set_xticklabels(test_labels, rotation=75, ha='right', fontsize=8)
    ax[2].set_title('Test labels (top 20)')

    plt.tight_layout()
    plt.show()
