"""
ANCHOR inference module (stub).

ANCHOR is a VGG-based CNN built specifically for handwritten Chinese character
recognition, reported at 97% accuracy on ICDAR 2013. It is included as our
specialist CNN baseline to compare against transformer models (TrOCR, DINOv2)
under perturbation.

TODO: ANCHOR is a research model without a standard pip package.
Steps to complete this module:
  1. Find the ANCHOR GitHub repo and download pretrained weights
  2. Clone or copy the model architecture definition into this repo
  3. Load the weights below and implement _predict_single()
  4. Verify output format matches the other inference modules (predicted string)

Reference: ANCHOR paper — https://arxiv.org/abs/1904.01375 (verify correct paper
with team before implementing)
"""

import numpy as np


def _to_object_array(items):
    """Build a 1D object array without NumPy broadcasting nested shapes."""
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def run_anchor(X_test, checkpoint_path=None, show_progress=True):
    """
    Run ANCHOR inference on a test set and return predictions.

    Parameters
    ----------
    X_test : np.ndarray of shape (n_samples,)
        Object array of images (uint8, H x W x 3).
    checkpoint_path : str or None
        Path to the ANCHOR pretrained weights file.
        TODO: update once weights are located.
    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    predictions : np.ndarray of shape (n_samples,)
        Object array of predicted strings, one per image.
        Returns empty strings until checkpoint is loaded.
    """
    # TODO: load ANCHOR model architecture and weights
    # model = AnchorModel()
    # model.load_state_dict(torch.load(checkpoint_path))
    # model.eval()

    if checkpoint_path is None:
        print("[ANCHOR] WARNING: No checkpoint provided. Returning empty predictions.")
        print("[ANCHOR] See module docstring for setup instructions.")
        empty = [""] * len(X_test)
        return _to_object_array(empty)

    # TODO: implement inference loop once model is loaded
    # predictions = []
    # for img in X_test:
    #     predicted = _predict_single(model, img)
    #     predictions.append(predicted)
    # return _to_object_array(predictions)

    raise NotImplementedError(
        "ANCHOR inference not yet implemented. "
        "See module docstring for setup instructions."
    )