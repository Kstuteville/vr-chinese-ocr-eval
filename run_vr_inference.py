"""
Standalone script to run all 5 model inferences on vr_test_images/
and save prediction caches to outputs/vr_preds_*.npy

Run from the repo root with:
    conda run -n vr-ocr-eval python run_vr_inference.py
"""

import sys, os
import numpy as np
from pathlib import Path
from PIL import Image

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "outputs"
VR_DIR  = ROOT / "vr_test_images"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))

MAX_SIDE = 640  # VR screenshots are ~2890x1682 — resize for inference speed

# ── Load images ──────────────────────────────────────────────────────────────
print("Loading VR images...")
images, labels = [], []

for char_dir in sorted(VR_DIR.iterdir()):
    if not char_dir.is_dir():
        continue
    char = char_dir.name
    for img_path in sorted(char_dir.glob("*.png")):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        images.append(np.array(img, dtype=np.uint8))
        labels.append(char)

X_vr = np.empty(len(images), dtype=object)
for i, img in enumerate(images):
    X_vr[i] = img
y_vr = np.array(labels, dtype=object)
print(f"  {len(y_vr)} images loaded, size {images[0].shape}")


def run_or_load(cache_path, run_fn, name):
    if cache_path.exists():
        preds = np.load(cache_path, allow_pickle=True)
        print(f"[{name}] Loaded cache ({len(preds)} samples)")
        return preds
    print(f"[{name}] Running inference...")
    preds = run_fn(X_vr, show_progress=True)
    np.save(cache_path, preds)
    print(f"[{name}] Saved to {cache_path}")
    return preds


# ── ANCHOR ───────────────────────────────────────────────────────────────────
from models.anchor_inference import run_anchor
run_or_load(OUT_DIR / "vr_preds_anchor.npy", run_anchor, "ANCHOR")

# ── PaddleOCR ─────────────────────────────────────────────────────────────────
from models.paddleocr_inference import run_paddleocr
run_or_load(OUT_DIR / "vr_preds_paddleocr.npy", run_paddleocr, "PaddleOCR")

# ── EasyOCR ───────────────────────────────────────────────────────────────────
from models.easyocr_inference import run_easyocr
run_or_load(OUT_DIR / "vr_preds_easyocr.npy", run_easyocr, "EasyOCR")

# ── CnOCR ─────────────────────────────────────────────────────────────────────
from models.cnocr_inference import run_cnocr
run_or_load(OUT_DIR / "vr_preds_cnocr.npy", run_cnocr, "CnOCR")

# ── Apple Vision ──────────────────────────────────────────────────────────────
try:
    from models.apple_vision_inference import run_apple_vision
    run_or_load(OUT_DIR / "vr_preds_applevision.npy", run_apple_vision, "Apple Vision")
except Exception as e:
    print(f"[Apple Vision] Skipped: {e}")

print("\nAll inferences complete. Caches saved to outputs/vr_preds_*.npy")
