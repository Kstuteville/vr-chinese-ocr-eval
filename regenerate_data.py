"""
Regenerates the shared .npy files from the Kaggle single-character dataset.

Run this once after the Kaggle dataset finishes downloading:
    python regenerate_data.py
"""

import numpy as np
from pathlib import Path
from data.utils import load_data

print("Loading data from Kaggle CASIA dataset...")
X_train, X_test, y_train, y_test, pert_type_train, pert_type_test = load_data(
    clean_count=4500,
    min_per_perturb_bucket=60,
)

print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# Filter to Chinese characters only — punctuation labels like 'colon', '!' are
# not relevant to the VR handwriting app and would pollute evaluation results.
def _is_chinese(label):
    return any(0x4E00 <= ord(c) <= 0x9FFF for c in str(label))

train_mask = [_is_chinese(y) for y in y_train]
test_mask  = [_is_chinese(y) for y in y_test]

X_train      = X_train[train_mask]
y_train      = y_train[train_mask]
pert_type_train = pert_type_train[train_mask]
X_test       = X_test[test_mask]
y_test       = y_test[test_mask]
pert_type_test  = pert_type_test[test_mask]

print(f"After filtering to Chinese only — Train: {len(X_train)} | Test: {len(X_test)}")

Path("notebooks/data").mkdir(parents=True, exist_ok=True)

np.save("notebooks/data/X_train.npy", X_train)
np.save("notebooks/data/X_test.npy",  X_test)
np.save("notebooks/data/y_train.npy", y_train)
np.save("notebooks/data/y_test.npy",  y_test)
np.save("notebooks/data/pert_types.npy", pert_type_test)

print("\nSample labels (should be Chinese characters):")
print(y_test[:8])

# Sanity check — most labels should be CJK (some will be punctuation like 'colon')
cjk_count = sum(
    1 for label in y_test[:20]
    if any(0x4E00 <= ord(c) <= 0x9FFF for c in str(label))
)
if cjk_count >= 10:
    print(f"\nLabels look correct — {cjk_count}/20 are Chinese characters (rest are punctuation).")
else:
    print(f"\nWARNING: Only {cjk_count}/20 labels are Chinese characters. Something is wrong.")
