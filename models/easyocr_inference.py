"""
EasyOCR inference module.

EasyOCR (JaidedAI) uses a CRAFT text detector + CRNN recognizer pipeline.
It is architecturally distinct from PaddleOCR and provides a useful
comparison point as a second production-grade Chinese OCR system.

We disable the detector (detail=0, paragraph=False) since images are
already single cropped characters.

Reference: https://github.com/JaidedAI/EasyOCR
"""

import numpy as np


def _to_object_array(items):
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def run_easyocr(X_test, lang=["ch_sim", "en"], show_progress=True):
    """
    Run EasyOCR inference on a test set and return predictions.

    Parameters
    ----------
    X_test : np.ndarray of shape (n_samples,)
        Object array of images (uint8, H x W x 3).
    lang : list of str
        Languages to load. ['ch_sim', 'en'] for simplified Chinese.
    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    predictions : np.ndarray of shape (n_samples,)
        Object array of predicted strings, one per image.
    """
    import easyocr

    reader = easyocr.Reader(lang, gpu=False, verbose=False)

    predictions = []
    iterator = X_test
    if show_progress:
        from tqdm.auto import tqdm
        iterator = tqdm(X_test, desc="EasyOCR inference")

    for img in iterator:
        try:
            results = reader.readtext(img, detail=0, paragraph=False)
            text = "".join(results) if results else ""
            predictions.append(text)
        except Exception:
            predictions.append("")

    return _to_object_array(predictions)
