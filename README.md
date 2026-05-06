# VR Chinese OCR Evaluation

## Goals

We built this project to answer a practical question: **which OCR model should you actually use in a VR app that reads handwritten Chinese characters off a whiteboard?**

For our Deep Learning for Media class (MPATE-GE 2039 / DM-GY 9103), we set up a systematic evaluation of five OCR models under nine VR-specific distortions — conditions like motion blur from head movement, radial glare from the headset lens, and low resolution from the Quest passthrough camera. The aim was to understand not just which model is most accurate in ideal conditions, but which one degrades the least when the environment fights back.

Our secondary goal was to study the **representation space** of these models: where do they fail, which character pairs get confused with each other, and do different perturbations break different models in different ways?

The short answer: on the synthetic perturbed test set, ANCHOR — a model from 2013 built specifically for handwritten Chinese — outperforms every modern production OCR system. But on real frames captured from the Meta Quest 3S, the ranking inverts: ANCHOR drops to 0% and CnOCR jumps to first. ANCHOR's recognizer might still be the strongest of the five — but it requires a much heavier preprocessing pipeline to be usable in production. Synthetic perturbations underestimate the real domain gap.

---

## The Data

We used the [CASIA-HWDB dataset](https://www.kaggle.com/datasets/pascalbliem/handwritten-chinese-character-hanzi-datasets) from Kaggle — a large collection of offline handwritten Chinese characters written by hundreds of different writers, covering 3,755 character classes.

We sampled **4,500 clean images** and generated **4,500 perturbed images** (~500 per perturbation type). After filtering to Chinese-only labels, our test set contains ~818 clean samples and ~88–107 samples per perturbation type.

One thing we caught early: the Kaggle zip file decodes UTF-8 folder names as CP437, turning Chinese characters into garbled strings like `Θàí` instead of `睾`. Everything looked fine until we checked the actual labels. Always sanity-check your ground truth before running any model.

---

## The Perturbations

All nine perturbations simulate conditions specific to VR headset use, grouped by cause:

| Category | Perturbations |
|---|---|
| Lighting | Radial glare, light streak, contrast variation, white balance shift |
| Capture quality | Gaussian noise, motion blur, low resolution |
| Geometric | Perspective warp (camera tilt via homography) |
| Physical | Occlusion (rectangular blockout simulating obstructions) |

Implementations are in [`perturbations/pipeline.py`](perturbations/pipeline.py).

---

## The Models

| Model | Type | Clean Accuracy |
|---|---|---|
| **ANCHOR** | VGG-like CNN trained on CASIA handwriting | 58.1% |
| **PaddleOCR** | Baidu's production OCR system | 40.5% |
| **Apple Vision** | macOS on-device OCR (`VNRecognizeTextRequest`, zh-Hans) | 39.7% |
| **EasyOCR** | Open-source OCR (CRAFT + CRNN) | 26.9% |
| **CnOCR** | Lightweight Chinese OCR library | 15.2% |

We also evaluated:
- **CLIP** (OpenAI) — used as a feature extractor with 1-NN classification. Results were too low to include in the main comparison; the embedding space was not expressive enough for fine-grained character discrimination without fine-tuning.
- **DINOv2** (Meta) — same approach as CLIP, achieved <1% accuracy. Off-the-shelf ViT features don't transfer to 3,755-class character recognition without task-specific training data.
- **TrOCR** (Microsoft) — no working Chinese handwriting checkpoint was available.

---

## Results

### Overall accuracy across all conditions

| Model | Accuracy | CER | F1 |
|---|---|---|---|
| ANCHOR | 54.3% | 0.457 | 37.9% |
| PaddleOCR | 38.3% | 0.617 | 30.0% |
| Apple Vision | 36.8% | 0.632 | 31.6% |
| EasyOCR | 24.9% | 0.751 | 20.1% |
| CnOCR | 13.6% | 0.864 | 8.2% |

### Accuracy by perturbation type

| Perturbation | ANCHOR | PaddleOCR | Apple Vision | EasyOCR | CnOCR |
|---|---|---|---|---|---|
| Clean | 58.1% | 40.5% | 39.7% | 26.9% | 15.2% |
| Contrast variation | 53.3% | 46.7% | 43.9% | 29.0% | 14.0% |
| Gaussian noise | 26.0% | 41.0% | 42.0% | 23.0% | 18.0% |
| Light streak | 63.6% | 39.8% | 33.0% | 22.7% | 11.4% |
| Low resolution | 47.9% | 36.5% | 21.9% | 12.5% | 5.2% |
| Motion blur | 27.2% | 7.8% | 8.7% | 1.9% | 1.9% |
| Occlusion | 58.2% | 31.6% | 30.6% | 24.5% | 17.3% |
| Perspective warp | 68.9% | 42.2% | 40.0% | 34.4% | 11.1% |
| Radial glare | 62.2% | 44.9% | 41.8% | 28.6% | 19.4% |
| White balance shift | 54.0% | 37.0% | 45.0% | 31.0% | 11.0% |

### Key findings

- **Motion blur is the hardest perturbation across the board.** Every model collapses — ANCHOR drops 31 points, PaddleOCR and Apple Vision drop ~32 points. This is the most critical problem to solve for real VR use.
- **Gaussian noise specifically kills ANCHOR** (−32 points) while barely affecting PaddleOCR (+0.5%) and Apple Vision (+2.3%). ANCHOR's training distribution doesn't include this noise pattern; the others appear more robust to it.
- **Apple Vision is competitive with PaddleOCR** on clean images and beats it on contrast, noise, and white balance — impressive for a general-purpose on-device system with no Chinese handwriting specialization.
- **Perspective warp barely hurts anyone** — slightly tilted characters are not difficult for any of these models.
- **ANCHOR handles occlusion well** — it learned to recognize partial strokes from training on real handwriting.

All figures are saved to [`outputs/`](outputs/) and can be regenerated by running [`notebooks/analysis_final.ipynb`](notebooks/analysis_final.ipynb).

---

## Real VR Capture Test

After the synthetic evaluation, we tested all five models on **151 real frames** captured directly from the Meta Quest 3S app (across 14 characters). These are images that have already been preprocessed by the WebXR app's pipeline — ArUco-based whiteboard detection, perspective correction, and cropping to the whiteboard region. The captured frames live in [`vr_test_images/`](vr_test_images/), the inference script that runs all five models on them is [`run_vr_inference.py`](run_vr_inference.py), and the full per-character analysis is in [`notebooks/analysis_vr.ipynb`](notebooks/analysis_vr.ipynb).

The rankings flipped completely:

| Model | Synthetic Clean | Real VR | Δ |
|---|---|---|---|
| ANCHOR | 58.1% | 0.0% | −58.1 |
| PaddleOCR | 40.5% | 34.4% | −6.1 |
| EasyOCR | 26.9% | 1.3% | −25.6 |
| **CnOCR** | 15.2% | **59.6%** | **+44.4** |
| Apple Vision | 39.7% | 23.2% | −16.5 |

ANCHOR collapses to 0% because it was trained on CASIA images where the character fills almost the entire frame. In our real-VR snapshots the character is small relative to the whiteboard panel around it, and ANCHOR's recognizer ends up classifying the whiteboard region itself — every prediction comes back as 日 or 口, characters whose shape matches the panel.

CnOCR and PaddleOCR include a detection stage that locates the character before recognizing it, which explains why they generalize from synthetic to real captures while ANCHOR doesn't. It also helps that the two perturbations that hurt these models most in the synthetic test — motion blur and Gaussian noise — don't appear in the real-VR captures: the WebXR app captures static snapshots after ArUco marker lock (avoiding motion blur by construction), and the Quest 3S sensor doesn't produce Gaussian-style noise under normal lighting.

**Takeaway:** a model that can read a character isn't enough. In production, the pipeline architecture is just as important as the model itself — it's what gives the model the input it was designed for. Even the best recognizer won't deliver without the right pipeline around it.

---

## Code Structure

```
vr-chinese-ocr-eval/
├── data/
│   └── utils.py                      # data loading, CP437 label fix, train/test split
├── models/
│   ├── anchor_inference.py           # ANCHOR (CNN for handwritten Chinese)
│   ├── paddleocr_inference.py        # PaddleOCR 3.x (detection disabled)
│   ├── easyocr_inference.py          # EasyOCR (CRAFT + CRNN)
│   ├── cnocr_inference.py            # CnOCR (lightweight Chinese OCR)
│   ├── apple_vision_inference.py     # Apple Vision (macOS VNRecognizeTextRequest)
│   ├── clip_inference.py             # CLIP image encoder + 1-NN (not in main comparison)
│   └── dinov2_classifier.py          # DINOv2 feature extractor + 1-NN (not in main comparison)
├── perturbations/
│   ├── pipeline.py                   # all 9 VR perturbation implementations
│   └── perturbation_analysis.py      # visual analysis of perturbation effects
├── evaluation/
│   ├── metrics.py                    # accuracy, CER, precision, recall, F1, robustness gap
│   ├── accuracy_curves.py            # bar charts, heatmaps, robustness gap plots
│   ├── failure_analysis.py           # misclassification grids, confusion pairs
│   └── utils.py                      # formatted table printers for notebooks
├── notebooks/
│   ├── analysis_final.ipynb          # main analysis notebook — all 5 models, all figures
│   ├── analysis_vr.ipynb             # real-VR evaluation — per-character results across all models
│   ├── workspace.ipynb               # exploratory notebook
│   └── data/                         # pre-generated .npy files (tracked via Git LFS)
│       ├── X_test.npy / X_train.npy
│       ├── y_test.npy / y_train.npy
│       └── pert_types.npy
├── vr_test_images/                   # 151 real captures from Meta Quest 3S, organized by character
├── outputs/                          # generated predictions + figures (gitignored for .npy)
├── run_vr_inference.py               # runs all 5 models on the real-VR image set
└── regenerate_data.py                # rebuilds notebooks/data/ from scratch
```

---

## Setup

```bash
pip install -r requirements.txt
pip install paddleocr easyocr cnocr tensorflow
# macOS only — for Apple Vision model:
pip install pyobjc-framework-Vision
```

The `.npy` data files are in `notebooks/data/` (tracked via Git LFS). If you need to regenerate them from the raw Kaggle dataset:

```bash
python regenerate_data.py
```

To run the full analysis, open and execute [`notebooks/analysis_final.ipynb`](notebooks/analysis_final.ipynb). All figures will be saved to `outputs/`.

To run a single model programmatically:

```python
import numpy as np
from models.anchor_inference import run_anchor
from evaluation.metrics import evaluate

X_test     = np.load("notebooks/data/X_test.npy",     allow_pickle=True)
y_test     = np.load("notebooks/data/y_test.npy",     allow_pickle=True)
pert_types = np.load("notebooks/data/pert_types.npy", allow_pickle=True)

preds   = run_anchor(X_test)
results = evaluate(preds, y_test, pert_types)
```

---

## Who Did What

**Eros Carrasco** — Integrated CnOCR and Apple Vision into the evaluation pipeline (models/cnocr_inference.py, models/apple_vision_inference.py). Evaluated CLIP as a feature extractor and dropped it from the main comparison after confirming it didn't add signal. Implemented the nine VR perturbation types in perturbations/pipeline.py, based on real conditions encountered with Meta Quest passthrough cameras. Helped with generating and comparing results across all models on the synthetic test set.

Built a WebXR application that runs in the Meta Quest 3S browser: a real-time handwriting reader with an image preprocessing pipeline that uses ArUco markers to locate the whiteboard region, applies perspective correction, and produces a cropped snapshot of the whiteboard area for the model to read.

Used this app to collect the real-VR evaluation set (vr_test_images/): 151 preprocessed snapshots captured directly from the Meta Quest 3S across 14 characters. Wrote the inference script that runs all five models on this set (run_vr_inference.py) and the analysis notebook for the per-character results (notebooks/analysis_vr.ipynb). Surfaced the central finding that the synthetic-to-real gap inverts the model ranking — ANCHOR drops from 58.1% to 0%, CnOCR jumps from 15.2% to 59.6% — and that pipeline architecture (detection + recognition) matters more in production than recognition accuracy alone.

**Kaylie Stuteville** — Built evaluation pipeline on top of the initial metrics foundation written by Jasmine. Wrote all visualization functions for paper figures including grouped bar charts, robustness heatmap, F1 breakdown, and overall metrics figure with CER secondary axis (evaluation/accuracy_curves.py). Wrote all failure analysis functions including misclassification grids, top confused character pairs, and cross-model failure comparison (evaluation/failure_analysis.py). Extended metrics with per-perturbation breakdowns and formatted result tables (evaluation/metrics.py, evaluation/utils.py). git log --oneline --author="Kaylie" 

Discovered and fixed the critical ANCHOR pixel inversion bug — ANCHOR expects white strokes on black background and was scoring 0.5% without this fix. After the fix it jumped to 58.1%, making it the strongest model in the evaluation. Added ANCHOR, EasyOCR, and PaddleOCR as working models (models/anchor_inference.py, models/easyocr_inference.py, models/paddleocr_inference.py).
Rebalanced the test set to ~100 samples per perturbation for fair comparison and fixed a perspective warp crash for small images (regenerate_data.py, perturbations/pipeline.py).
Contributed to the main notebook. Managed Git LFS setup for large data files.


**Kezia Widjaja** — Establish the `workspace.ipynb` with `Setup` for `utils`, `Load Data`, `Perturbation`. Set the main utils file on `/utils` to load the rest of the utils methods for `load_data`, `split_data`, `explore_data`, `evaluate_model` (unused), `plot_loss` (unused). Establish the `utils.py` for `/evaluation`, `/models`, and `/data`. Defining the initial perturbation list and create function to combine diffferent perturbations (dropped due to complexity). Updating `README.md` with instruction to run the notebook and the perturbations list. Writing hypothesis and analysis of the model evaluation. Most of the commits are from [my fork](https://github.com/ekkezia/vr-chinese-ocr-eval) at `d6b3312fb5e671cd17657b5adf1a7cc617dc966d`.

**Jasmine Zhang** — Sourced the CASIA-HWDB dataset on Chinese National Library of Pattern recognitio and Kaggle, making it available as the primary data source for the evaluation.

Authored the core metric logic in `evaluation/metrics.py` — exact match, CER via Levenshtein edit distance, per-perturbation breakdowns, robustness gap, and macro CER for handling imbalanced perturbation buckets. Implemented the visualization functions in `evaluation/accuracy_curves.py` (grouped bar charts, robustness gap heatmaps, F1 charts for paper figures) and `evaluation/failure_analysis.py` (misclassification grids, top confused character pairs, per-class accuracy breakdowns, and cross-model failure comparison) — these were then integrated into the final notebook by Kaylie.

Evaluated why DINOv2 and TrOCR were ultimately dropped from the main comparison — DINOv2's off-the-shelf ViT features failed to transfer to 3,755-class character recognition without fine-tuning, and no working Chinese handwriting checkpoint existed for TrOCR. Refactored the DINOv2 classifier to replace the custom linear head with a proper `KNeighborsClassifier` for 1-NN evaluation.

**Lia Cociorva** — Researching models to evaluate and which perturbations to evaluate them on. Using VR as our use case, we landed on 9 perturbations which are supported by academic research relevant to video and image perturbation for VR. 
