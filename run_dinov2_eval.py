"""
Run DINOv2 inference and save predictions to outputs/preds_dinov2.npy

Usage:
    python run_dinov2_eval.py
"""

import numpy as np
from pathlib import Path
from models.dinov2_classifier import run_dinov2

print("Loading data...")
X_train = np.load("notebooks/data/X_train.npy", allow_pickle=True)
y_train = np.load("notebooks/data/y_train.npy", allow_pickle=True)
X_test  = np.load("notebooks/data/X_test.npy",  allow_pickle=True)
y_test  = np.load("notebooks/data/y_test.npy",  allow_pickle=True)
print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

print("\nRunning DINOv2...")
preds = run_dinov2(X_train, y_train, X_test)

Path("outputs").mkdir(exist_ok=True)
np.save("outputs/preds_dinov2.npy", preds)
print(f"\nSaved to outputs/preds_dinov2.npy")

# Quick sanity check
from evaluation.metrics import evaluate
results = evaluate(preds, y_test)
print(f"\nDINOv2 results:")
print(f"  Accuracy  : {results['exact_match']:.1%}")
print(f"  CER       : {results['mean_cer']:.3f}")
print(f"  Precision : {results['precision']:.1%}")
print(f"  Recall    : {results['recall']:.1%}")
print(f"  F1        : {results['f1']:.1%}")
