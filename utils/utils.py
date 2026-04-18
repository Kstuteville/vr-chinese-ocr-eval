"""Facade utilities for notebooks.

Importing `utils` provides a stable surface that re-exports commonly used
helpers from `data`, `evaluation`, and `models` modules.
"""

from data import utils as data_utils
from evaluation import utils as evaluation_utils
from models import utils as model_utils

# Data utilities
load_data = data_utils.load_data
split_data = data_utils.split_data
explore_data = data_utils.explore_data

# Evaluation utilities
evaluate_model = evaluation_utils.evaluate_model
plot_loss = evaluation_utils.plot_loss

# Model utilities
# build_baseline = model_utils.build_baseline
# train_model = model_utils.train_model

__all__ = [
	"load_data",
	"split_data",
	"explore_data",
	"evaluate_model",
	"plot_loss",
	# "build_baseline",
	# "train_model",
	# "data_utils",
	# "evaluation_utils",
	# "model_utils",
]







