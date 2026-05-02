# VR Chinese OCR Evaluation

## What is this?

We built this project to answer a practical question: which OCR model should you actually use in a VR app that reads handwritten Chinese characters off a whiteboard? For our Deep Learning for Media class (MPATE-GE 2039 / DM-GY 9103), we evaluated four OCR models on handwritten Chinese characters under nine simulated VR distortions — things like motion blur from head movement, radial glare from the headset lens, and low resolution from the Quest display.

The short answer: ANCHOR, a model from 2013 built specifically for handwritten Chinese, outperforms all three modern production OCR systems under the conditions that actually matter in a headset. Modern OCR tools are optimized for printed text and documents, not isolated handwritten characters on a VR whiteboard — and it shows.

## The Data

We used the [CASIA-HWDB dataset](https://www.kaggle.com/datasets/pascalbliem/handwritten-chinese-character-hanzi-datasets) from Kaggle — a large collection of offline handwritten Chinese characters from hundreds of writers.

We sampled 4,500 clean images and applied nine VR-inspired perturbations to 4,500 additional images (~500 per perturbation type). After filtering to Chinese-only labels, our test set has ~818 clean samples and ~88–107 samples per perturbation type — large enough to draw reliable conclusions.

One thing we ran into early: the Kaggle zip file decodes UTF-8 folder names as CP437, turning Chinese characters into garbled strings like `Θàí` instead of `睾`. Everything looked fine until we checked the actual labels — a good reminder to always sanity-check your ground truth before running a single model.

## The Perturbations

All nine perturbations simulate conditions specific to VR headset use:

| Category | Perturbations |
|---|---|
| Lighting | Radial glare, light streak, contrast variation, white balance shift |
| Capture quality | Gaussian noise, motion blur, low resolution |
| Geometric | Perspective warp |
| Physical | Occlusion |

## The Models

| Model | Type | Clean Accuracy |
|---|---|---|
| **ANCHOR** | VGG-like CNN, trained on CASIA handwriting | 58.1% |
| **PaddleOCR** | Baidu's production OCR system | 40.5% |
| **EasyOCR** | Open-source OCR (CRAFT + CRNN) | 26.9% |
| **CnOCR** | Lightweight Chinese OCR library | 15.2% |

We also tried TrOCR (Microsoft) and DINOv2 (Meta). TrOCR had no working Chinese handwriting checkpoint available. DINOv2 achieved <1% accuracy — not because it's a bad model, but because it needs task-specific fine-tuning to distinguish thousands of Chinese character classes. Off-the-shelf ViT features don't transfer here without a lot more per-class training data than we had.

## Results

ANCHOR is the strongest choice for VR handwriting recognition across almost every condition:

| Perturbation | ANCHOR | PaddleOCR | EasyOCR | CnOCR |
|---|---|---|---|---|
| Clean | 58.1% | 40.5% | 26.9% | 15.2% |
| Motion blur | 27.2% | 7.8% | 1.9% | 1.9% |
| Low resolution | 47.9% | 36.5% | 12.5% | 5.2% |
| Perspective warp | 68.9% | 42.2% | 34.4% | 11.1% |
| Radial glare | 62.2% | 44.9% | 28.6% | 19.4% |
| Gaussian noise | 26.0% | 41.0% | 23.0% | 18.0% |

A few things worth calling out:

- **Motion blur is the hardest perturbation across the board.** Every model degrades significantly — this is the most important thing to improve for VR.
- **Gaussian noise specifically kills ANCHOR** (drops 32 points) while barely affecting PaddleOCR (drops 0.5%). This is ANCHOR's clear weakness and an interesting contrast.
- **ANCHOR barely degrades under occlusion** — it learned to recognize partial strokes from training on real handwriting, which tends to have gaps and incomplete characters.
- **Perspective warp barely hurts anyone** — slightly tilted characters are apparently not that hard to read.

## Code Structure

```
vr-chinese-ocr-eval/
├── data/
│   └── utils.py                  # data loading, CP437 label fix, splitting
├── models/
│   ├── anchor_inference.py       # ANCHOR (CNN for handwritten Chinese)
│   ├── paddleocr_inference.py    # PaddleOCR 3.x
│   ├── easyocr_inference.py      # EasyOCR
│   └── cnocr_inference.py        # CnOCR
├── perturbations/
│   └── pipeline.py               # all 9 VR perturbation implementations
├── evaluation/
│   ├── metrics.py                # accuracy, CER, precision, recall, F1
│   ├── accuracy_curves.py        # paper figures (bar charts, heatmaps)
│   ├── failure_analysis.py       # misclassification grids, confusion pairs
│   └── utils.py                  # formatted table printers
├── notebooks/
│   ├── workspace.ipynb           # main notebook — runs full evaluation
│   └── data/                     # pre-generated .npy files (tracked via Git LFS)
├── outputs/                      # model predictions + figures (gitignored, regenerate locally)
└── regenerate_data.py            # rebuilds notebooks/data/ from scratch
```

## Setup

```bash
pip install -r requirements.txt
pip install paddleocr easyocr cnocr tensorflow
```

The `.npy` data files are already in `notebooks/data/` (tracked via Git LFS). If you need to regenerate them:

```bash
python regenerate_data.py
```

To run a model and evaluate:

```python
import numpy as np
from models.anchor_inference import run_anchor
from evaluation.metrics import evaluate

X_test     = np.load("notebooks/data/X_test.npy", allow_pickle=True)
y_test     = np.load("notebooks/data/y_test.npy", allow_pickle=True)
pert_types = np.load("notebooks/data/pert_types.npy", allow_pickle=True)

preds   = run_anchor(X_test)
results = evaluate(preds, y_test, pert_types)
```

## Who Did What

- **Kaylie** — data pipeline, label encoding fix, evaluation infrastructure, PaddleOCR integration, EasyOCR integration, ANCHOR debugging and pixel inversion fix, full model evaluation, figures
- **Jasmine** — [insert work here]
- **Kezia** — [insert work here]
- **Eros** — [insert work here]
- **Lia** — [insert work here]
