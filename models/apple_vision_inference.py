"""
Apple Vision OCR inference module.

Uses macOS Vision framework (VNRecognizeTextRequest) for on-device
text recognition — no network, no GPU, runs entirely via the OS.

We use the accurate recognition level with Simplified Chinese (zh-Hans).
Since Vision is designed for natural-scene multi-word text, we upscale
images to 128×128 before passing them in, which gives the recognizer
enough resolution to work on isolated single characters.

Requires: pyobjc-framework-Vision  (macOS only)
  pip install pyobjc-framework-Vision

Reference: https://developer.apple.com/documentation/vision/vnrecognizetextrequest
"""

import io
import numpy as np
from PIL import Image


def _to_object_array(items):
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def _img_to_nsdata(img_array):
    """Convert a numpy image array to NSData (PNG bytes) for Vision."""
    from Foundation import NSData

    pil_img = Image.fromarray(img_array)
    # Ensure RGB — Vision framework expects colour input
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    # Upscale: Vision's text recognizer performs better on larger images
    if max(pil_img.size) < 128:
        pil_img = pil_img.resize((128, 128), Image.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    return NSData.dataWithBytes_length_(png_bytes, len(png_bytes))


def _build_request():
    """Create a configured VNRecognizeTextRequest for Simplified Chinese."""
    import Vision

    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setRecognitionLanguages_(["zh-Hans"])
    req.setUsesLanguageCorrection_(False)
    return req


def _recognize_single(img_array):
    """
    Run Vision OCR on one image. Returns the first recognised character,
    or an empty string if nothing was detected.
    """
    import Vision

    ns_data = _img_to_nsdata(img_array)
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, {})
    request = _build_request()

    try:
        handler.performRequests_error_([request], None)
    except Exception:
        return ""

    results = request.results()
    if not results or len(results) == 0:
        return ""

    candidates = results[0].topCandidates_(1)
    if not candidates or len(candidates) == 0:
        return ""

    text = str(candidates[0].string()).strip()
    # Vision may return multiple characters; we only score the first one
    # since the ground truth is always a single character
    return text[0] if text else ""


def run_apple_vision(X_test, show_progress=True):
    """
    Run Apple Vision OCR on a test set and return predictions.

    Parameters
    ----------
    X_test : np.ndarray of shape (n_samples,)
        Object array of images (uint8, H × W or H × W × 3).
    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    predictions : np.ndarray of shape (n_samples,)
        Object array of predicted label strings, one per image.
        Empty string if Vision returned no result for a sample.

    Raises
    ------
    ImportError
        If pyobjc-framework-Vision is not installed.
    """
    try:
        import Vision  # noqa: F401
        from Foundation import NSData  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pyobjc-framework-Vision is required for Apple Vision OCR.\n"
            "Install it with: pip install pyobjc-framework-Vision"
        ) from e

    predictions = []
    iterator = X_test
    if show_progress:
        from tqdm.auto import tqdm
        iterator = tqdm(X_test, desc="Apple Vision OCR")

    for img in iterator:
        try:
            pred = _recognize_single(img)
        except Exception:
            pred = ""
        predictions.append(pred)

    return _to_object_array(predictions)
