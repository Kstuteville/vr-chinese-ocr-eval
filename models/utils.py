"""
Models utility facade.

Re-exports inference functions from each model module so notebooks
can use a single consistent import surface.

Usage
-----
from models import utils as model_utils

predictions = model_utils.run_cnocr(X_test)
predictions = model_utils.run_trocr(X_test)
predictions = model_utils.run_anchor(X_test)
predictions = model_utils.run_dinov2(X_train, y_train, X_test)
"""

from models.cnocr_inference import run_cnocr
from models.trocr_inference import run_trocr
from models.dinov2_classifier import run_dinov2
from models.anchor_inference import run_anchor

__all__ = [
    "run_cnocr",
    "run_trocr",
    "run_dinov2",
    "run_anchor",
]