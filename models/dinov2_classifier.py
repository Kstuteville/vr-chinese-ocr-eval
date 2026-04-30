"""
DINOv2 classifier module.

DINOv2 is Meta's self-supervised vision transformer trained on 1.7B images.
Unlike the other three models, DINOv2 is a feature extractor — it produces
rich image embeddings but does not output text directly. A lightweight
linear classification head is trained on top to map embeddings to Chinese
character labels.

This module follows the same interface as the other inference modules:
call run_dinov2(X_train, y_train, X_test) to train and predict in one step.

Reference: https://github.com/facebookresearch/dinov2
"""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier


# Standard ImageNet normalization used by DINOv2
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _to_object_array(items):
    """Build a 1D object array without NumPy broadcasting nested shapes."""
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def _load_dinov2(model_name="dinov2_vitb14"):
    """
    Load DINOv2 from torch.hub.

    Parameters
    ----------
    model_name : str
        DINOv2 variant. Options (small to large):
        'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'
        Default is 'dinov2_vitb14' (base) — good balance of speed and quality.

    Returns
    -------
    model : torch.nn.Module
        DINOv2 feature extractor in eval mode.
    """
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    return model


def _extract_embeddings(X, model, show_progress=True):
    """
    Extract DINOv2 embeddings for a set of images.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples,)
        Object array of images (uint8, H x W x 3).
    model : torch.nn.Module
        Loaded DINOv2 model.
    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    embeddings : np.ndarray of shape (n_samples, embedding_dim)
        Float32 embedding vectors. embedding_dim is 768 for vitb14.
    """
    embeddings = []

    iterator = X
    if show_progress:
        from tqdm.auto import tqdm
        iterator = tqdm(X, desc="DINOv2 extracting embeddings")

    with torch.no_grad():
        for img in iterator:
            pil_img = Image.fromarray(img).convert("RGB")
            tensor = _TRANSFORM(pil_img).unsqueeze(0)  # (1, 3, 224, 224)
            embedding = model(tensor).squeeze(0).numpy()  # (embedding_dim,)
            embeddings.append(embedding)

    return np.stack(embeddings).astype(np.float32)


def run_dinov2(X_train, y_train, X_test, model_name="dinov2_vitb14", show_progress=True):
    """
    Run DINOv2 inference on a test set.

    Trains a lightweight linear classification head on X_train/y_train first,
    then predicts on X_test. This is required because DINOv2 is a feature
    extractor and does not output text directly.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_samples,)
        Object array of training images (uint8, H x W x 3).
    y_train : np.ndarray of shape (n_samples,)
        Ground truth label strings for training images.
    X_test : np.ndarray of shape (n_samples,)
        Object array of test images (uint8, H x W x 3).
    model_name : str
        DINOv2 variant to use. Default is 'dinov2_vitb14'.
    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    predictions : np.ndarray of shape (n_samples,)
        Object array of predicted label strings, one per image.
        Empty string if inference failed on a sample.
    """
    # Load DINOv2 once and reuse for both train and test
    model = _load_dinov2(model_name)

    # Extract embeddings for train and test
    print("[DINOv2] Extracting training embeddings...")
    embeddings_train = _extract_embeddings(X_train, model, show_progress=show_progress)

    print("[DINOv2] Extracting test embeddings...")
    embeddings_test = _extract_embeddings(X_test, model, show_progress=show_progress)

    # kNN classifier — works far better than a linear head when training data
    # is sparse (≈1 sample per class). DINOv2 embeddings are rich enough that
    # nearest-neighbor in embedding space gives meaningful results even with
    # very few examples per character class.
    print("[DINOv2] Fitting kNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    knn.fit(embeddings_train, y_train)

    predictions = knn.predict(embeddings_test)
    return _to_object_array(list(predictions))