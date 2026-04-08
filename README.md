# vr-chinese-ocr-eval

## How to setup
source .venv/bin/activate
pip install -r requirements.txt

## Python Version
3.11.15

## Where to work
Modify /utils/utils.py, /data/utils.py, /evaluation/utils.py, /models/utils.py, and /perturbations/pipeline.py
Run on /notebooks/workspace.ipynb

Feel free to add more modular py files to work on your own stuff

## Perturbation Methods
Perturbations are implemented in `/perturbations/pipeline.py` and applied on a sampled subset of clean data.

Current perturbation transforms:
- Radial glare spot
- Directional light streak
- Gaussian noise
- Contrast / lighting variation
- Occlusion block

## Data Pipeline (Current)
1. Load CASIA-HWDB2-line from Hugging Face.
2. Sample `clean_count` clean images from the train split.
3. Build a disjoint perturbation source pool from indices outside clean indices.
4. Sample `perturb_count` examples from that outside pool.
5. Apply perturbations according to a sampled or balanced bucket plan.
6. Merge clean + perturbed data, shuffle, and split into train/test.

## Unified Utils Facade
Use `import utils as u` in notebooks.

`/utils/utils.py` is a facade that re-exports helpers from:
- `/data/utils.py` (data loading, splitting, exploration)
- `/evaluation/utils.py` (evaluation and plotting)
- `/models/utils.py` (model build/train helpers)

This keeps notebook code clean while preserving modular code organization.

## load_data Notes
`load_data` is implemented in `/data/utils.py` and exposed through `u.load_data(...)`.
Useful parameters:
- `perturb_count=None`: auto-calculate count from bucket settings.
- `min_per_perturb_bucket`: minimum examples per selected bucket when auto mode is used.
- `include_combinations`: if `True`, use combination buckets from the perturbation power set.
- `perturb_bucket_count`: choose only a subset of buckets (useful when total budget is small).
- `balanced_perturbations=True`: distribute perturbation plan near-evenly across selected buckets.
- `show_progress=True`: enable tqdm progress while applying perturbations.
- `return_perturbation_type=True`: return perturbation metadata arrays.

## Return Values
When `return_perturbation_type=True`, `load_data` returns:
- `X_train, X_test, y_train, y_test, perturbation_type_train, perturbation_type_test`

When `return_perturbation_type=False`, it returns:
- `X_train, X_test, y_train, y_test`

`perturbation_type_*` values are:
- `None` for clean samples
- a perturbation label string (single or combined), e.g.:
	- `gaussian_noise`
	- `radial_glare+occlusion`

## Bucket Planning Helpers
Helpers in `/perturbations/pipeline.py`:
- `perturbation_buckets(include_combinations=False)`
- `select_perturbation_buckets(include_combinations=False, bucket_count=None, random_state=124)`
- `required_perturb_count(min_per_bucket=50, include_combinations=False, bucket_count=None)`

Examples:
- Single-type buckets only: `required_perturb_count(50, include_combinations=False)` -> `250`
- Full power-set buckets (non-empty combinations): `required_perturb_count(50, include_combinations=True)` -> `1550`

## Notebook Usage
Recommended setup cell:

```python
import sys
import importlib
from pathlib import Path

repo_root = Path.cwd().resolve().parent
if str(repo_root) not in sys.path:
	sys.path.insert(0, str(repo_root))

import utils as u
importlib.reload(u)
```

If you use metadata, unpack 6 outputs:

```python
X_train, X_test, y_train, y_test, perturbation_type_train, perturbation_type_test = u.load_data(
		show_progress=True,
		perturb_count=None,
		include_combinations=True,
		min_per_perturb_bucket=50,
		return_perturbation_type=True,
)
```

If you only need 4 outputs, set:

```python
X_train, X_test, y_train, y_test = u.load_data(return_perturbation_type=False)
```