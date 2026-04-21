"""
TrOCR inference module.

TrOCR is a Microsoft transformer model (Vision Encoder + Text Decoder) fine-tuned
on Traditional Chinese handwritten text. It is the closest pretrained model
to our whiteboard capture conditions among the four models we evaluate.

The model receives a PIL image and returns a predicted Chinese character directly,
making it easy to compare against CASIA ground truth labels.

Reference: https://huggingface.co/ZihCiLin/trocr-traditional-chinese-baseline
"""

import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def _to_object_array(items):
    """Build a 1D object array without NumPy broadcasting nested shapes."""
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def _predict_single(processor, model, img):
    """
    Run TrOCR on a single image and return the predicted text.

    Parameters
    ----------
    processor : TrOCRProcessor
        Loaded TrOCR processor instance.
    model : VisionEncoderDecoderModel
        Loaded TrOCR model instance.
    img : np.ndarray (uint8, H x W x 3)
        Input image.

    Returns
    -------
    str
        Predicted text string.
    """
    try:
        # TrOCR expects a PIL RGB image
        pil_img = Image.fromarray(img).convert("RGB")
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        predicted = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return predicted
    except Exception:
        # Return empty string if inference fails on a sample
        return ""


def run_trocr(X_test, model_name="microsoft/trocr-base-handwritten", show_progress=True):
    """    Run TrOCR inference on a test set and return predictions.

    Parameters
    ----------
    X_test : np.ndarray of shape (n_samples,)
        Object array of images (uint8, H x W x 3). Variable sizes are fine.
    model_name : str
        HuggingFace model name. Default is 'ZihCiLin/trocr-traditional-chinese-baseline',
        fine-tuned on Traditional Chinese handwritten characters.
        Alternative: 'microsoft/trocr-base-handwritten' for English handwriting baseline.
        show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    predictions : np.ndarray of shape (n_samples,)
        Object array of predicted strings, one per image.
        Empty string if inference failed on a sample.
    """
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.eval()

    predictions = []

    iterator = X_test
    if show_progress:
        from tqdm.auto import tqdm
        iterator = tqdm(X_test, desc="TrOCR inference")

    for img in iterator:
        predicted = _predict_single(processor, model, img)
        predictions.append(predicted)

    return _to_object_array(predictions)