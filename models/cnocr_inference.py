"""
CnOCR inference module.

CnOCR is a PyTorch-based Chinese OCR library with 20+ pretrained models.
It receives an image and returns a list of detected text fragments, each
with a `text` field and a `score` (confidence). Since our CASIA samples
are full text lines, we concatenate all fragments in order to reconstruct
the complete predicted line before comparing against the ground truth.

Reference: https://cnocr.readthedocs.io
"""

import numpy as np
from cnocr import CnOcr


def _to_object_array(items):
    """Build a 1D object array without NumPy broadcasting nested shapes."""
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def _predict_single(ocr, img):
    """
    Run CnOCR on a single image and return the predicted text.

    CnOCR may return multiple fragments per image (e.g. if it detects
    multiple text regions). We concatenate all fragments in order to
    reconstruct the full predicted line. This is important because the
    ground truth labels are full lines, not individual characters.

    Parameters
    ----------
    ocr : CnOcr
        Loaded CnOCR model instance.
    img : np.ndarray (uint8, H x W x 3)
        Input image.

    Returns
    -------
    str
        Predicted text (all fragments concatenated).
    """
    try:
        results = ocr.ocr(img)
        # Each result is a dict with keys 'text' and 'score'
        # Concatenate all fragments in detection order
        predicted = "".join([r["text"] for r in results])
        return predicted
    except Exception:
        # Return empty string if inference fails on a sample
        return ""


def run_cnocr(X_test, model_name="densenet_lite_136-gru", show_progress=True):
    """
    Run CnOCR inference on a test set and return predictions.

    Parameters
    ----------
    X_test : np.ndarray of shape (n_samples,)
        Object array of images (uint8, H x W x 3). Variable sizes are fine.
    model_name : str
        CnOCR pretrained model name. Default is 'densenet_lite_136-gru',
        a lightweight model that balances speed and accuracy.
        See https://cnocr.readthedocs.io for all available models.
    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    predictions : np.ndarray of shape (n_samples,)
        Object array of predicted strings, one per image.
        Empty string if inference failed on a sample.
    """
    ocr = CnOcr(rec_model_name=model_name, det_model_name=None)

    predictions = []

    iterator = X_test
    if show_progress:
        from tqdm.auto import tqdm
        iterator = tqdm(X_test, desc="CnOCR inference")

    for img in iterator:
        predicted = _predict_single(ocr, img)
        predictions.append(predicted)

    return _to_object_array(predictions)
