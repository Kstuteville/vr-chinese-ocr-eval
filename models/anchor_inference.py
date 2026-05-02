"""
ANCHOR inference module.

ANCHOR (ANgzhou Chinese Handwriting Optical Recognition) is a VGG-like CNN
built specifically for handwritten Chinese character recognition, achieving
97.68% top-1 accuracy on ICDAR 2013 competition data.

This module loads the pretrained ANCHOR model and runs inference on images,
returning predicted Chinese character strings to match the interface of the
other inference modules.

Key preprocessing notes:
  - ANCHOR expects GRAYSCALE images resized to 64x64 pixels
  - Input shape is (64, 64, 1) — NOT RGB
  - Labels are loaded from data/labels.txt in the ANCHOR repo

Reference: https://github.com/angzhou/anchor
"""

import numpy as np
from PIL import Image
from pathlib import Path


# Default paths — assumes anchor repo cloned to home directory
_DEFAULT_WEIGHTS = str(Path.home() / "anchor" / "data" / "weights08.h5")
_DEFAULT_LABELS = str(Path.home() / "anchor" / "data" / "labels.txt")

# ANCHOR was trained on 64x64 grayscale images
_IMG_SIZE = 96
_NUM_CLASSES = 3755


def _to_object_array(items):
    """Build a 1D object array without NumPy broadcasting nested shapes."""
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def _load_labels(labels_path):
    """
    Load the character label list from labels.txt.

    Parameters
    ----------
    labels_path : str
        Path to labels.txt from the ANCHOR repo.

    Returns
    -------
    labels : list of str
        List of 3755 Chinese characters, index matches model output class.
    """
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def _load_anchor_model(weights_path):
    """
    Build and load the ANCHOR model with pretrained weights.

    Parameters
    ----------
    weights_path : str
        Path to weights08.h5 from the ANCHOR repo.

    Returns
    -------
    model : keras.Model
        Loaded ANCHOR model in inference mode.
    """
    # Import here to avoid hard dependency at module load time
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Flatten, Dense, ZeroPadding2D,
        Conv2D, Activation, MaxPooling2D, BatchNormalization
    )

    # LeakyReLU moved in newer Keras versions
    try:
        from tensorflow.keras.layers import LeakyReLU
    except ImportError:
        from tensorflow.keras.layers.advanced_activations import LeakyReLU

    def relu():
        return LeakyReLU(alpha=0.01)

    def conv_unit(input_tensor, nb_filters, mp=False):
        x = ZeroPadding2D()(input_tensor)
        x = Conv2D(nb_filters, (3, 3))(x)
        x = relu()(x)
        x = BatchNormalization(axis=3, momentum=0.66)(x)
        if mp:
            x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        return x

    def out_block(input_tensor, nb_classes):
        x = Flatten()(input_tensor)
        x = Dense(1024)(x)
        x = relu()(x)
        x = BatchNormalization(momentum=0.66)(x)
        x = Dense(256)(x)
        x = relu()(x)
        x = BatchNormalization(momentum=0.66)(x)
        x = Dense(nb_classes)(x)
        x = Activation('softmax')(x)
        return x

    # Build model architecture (must match exactly what was trained)
    inputs = Input(shape=(_IMG_SIZE, _IMG_SIZE, 1))
    x = ZeroPadding2D()(inputs)
    x = Conv2D(64, (3, 3), strides=(2, 2))(x)
    x = relu()(x)
    x = BatchNormalization(momentum=0.66)(x)
    x = conv_unit(x, 128)
    x = conv_unit(x, 128, mp=True)
    x = conv_unit(x, 256)
    x = conv_unit(x, 256, mp=True)
    x = conv_unit(x, 384)
    x = conv_unit(x, 384)
    x = conv_unit(x, 384, mp=True)
    x = conv_unit(x, 512)
    x = conv_unit(x, 512)
    x = conv_unit(x, 512, mp=True)
    x = out_block(x, _NUM_CLASSES)

    model = Model(inputs=inputs, outputs=x)
    model.load_weights(weights_path)
    return model


def _preprocess(img):
    """
    Preprocess a single image for ANCHOR inference.

    ANCHOR expects 64x64 grayscale images normalized to [0, 1].

    Parameters
    ----------
    img : np.ndarray (uint8, H x W x 3)
        Input RGB image.

    Returns
    -------
    np.ndarray of shape (1, 64, 64, 1)
        Preprocessed image ready for model input.
    """
    pil_img = Image.fromarray(img).convert("L")  # RGB -> grayscale
    pil_img = pil_img.resize((_IMG_SIZE, _IMG_SIZE), Image.LANCZOS)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = 1.0 - arr  # ANCHOR trained on white strokes on black background
    return arr.reshape(1, _IMG_SIZE, _IMG_SIZE, 1)


def _predict_single(model, labels, img):
    """
    Run ANCHOR on a single image and return the predicted character.

    Parameters
    ----------
    model : keras.Model
        Loaded ANCHOR model.
    labels : list of str
        Character label list from labels.txt.
    img : np.ndarray (uint8, H x W x 3)
        Input image.

    Returns
    -------
    str
        Predicted Chinese character string.
    """
    try:
        x = _preprocess(img)
        preds = model.predict(x, verbose=0)
        idx = np.argmax(preds[0])
        return labels[idx]
    except Exception:
        return ""


def run_anchor(
    X_test,
    weights_path=_DEFAULT_WEIGHTS,
    labels_path=_DEFAULT_LABELS,
    show_progress=True,
):
    """
    Run ANCHOR inference on a test set and return predictions.

    Parameters
    ----------
    X_test : np.ndarray of shape (n_samples,)
        Object array of images (uint8, H x W x 3).
    weights_path : str
        Path to weights08.h5. Defaults to ~/anchor/data/weights08.h5.
    labels_path : str
        Path to labels.txt. Defaults to ~/anchor/data/labels.txt.
    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    predictions : np.ndarray of shape (n_samples,)
        Object array of predicted Chinese character strings, one per image.
        Empty string if inference failed on a sample.
    """
    if not Path(weights_path).exists():
        print(f"[ANCHOR] WARNING: Weights not found at {weights_path}")
        print("[ANCHOR] Clone https://github.com/angzhou/anchor to ~/anchor")
        return _to_object_array([""] * len(X_test))

    print("[ANCHOR] Loading model and weights...")
    labels = _load_labels(labels_path)
    model = _load_anchor_model(weights_path)
    print("[ANCHOR] Model loaded successfully!")

    predictions = []

    iterator = X_test
    if show_progress:
        from tqdm.auto import tqdm
        iterator = tqdm(X_test, desc="ANCHOR inference")

    for img in iterator:
        predicted = _predict_single(model, labels, img)
        predictions.append(predicted)

    return _to_object_array(predictions)