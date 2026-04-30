"""
PaddleOCR inference module.

PaddleOCR is Baidu's production-grade OCR system, state of the art for
Chinese character recognition. Unlike CnOCR and TrOCR, it includes both
a detection and recognition pipeline optimized for Chinese text.

We disable detection (use_det=False) since our images are already
single cropped characters — no need to locate text regions.

Reference: https://github.com/PaddlePaddle/PaddleOCR
"""

import numpy as np
from PIL import Image


def _to_object_array(items):
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def run_paddleocr(X_test, lang="ch", show_progress=True):
    """
    Run PaddleOCR inference on a test set and return predictions.

    Parameters
    ----------
    X_test : np.ndarray of shape (n_samples,)
        Object array of images (uint8, H x W x 3).
    lang : str
        Language. 'ch' for Chinese (default).
    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    predictions : np.ndarray of shape (n_samples,)
        Object array of predicted strings, one per image.
    """
    from paddleocr import PaddleOCR

    # use_det=False: images are already cropped single characters
    # use_cls=False: no text orientation classification needed
    # PaddleOCR 3.x API: disable detection/orientation, recognition only
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=lang,
    )

    predictions = []
    iterator = X_test
    if show_progress:
        from tqdm.auto import tqdm
        iterator = tqdm(X_test, desc="PaddleOCR inference")

    for img in iterator:
        try:
            result = ocr.ocr(img)
            # PaddleOCR 3.x returns a list of dicts with 'rec_texts' key
            if result and isinstance(result[0], dict) and result[0].get('rec_texts'):
                text = "".join(result[0]['rec_texts'])
            else:
                text = ""
            predictions.append(text)
        except Exception:
            predictions.append("")

    return _to_object_array(predictions)
