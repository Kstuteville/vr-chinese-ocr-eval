import numpy as np
from itertools import combinations
from PIL import Image
from PIL import ImageEnhance
from tqdm.auto import tqdm

PERTURBATION_TYPES = (
	"radial_glare",
	"light_streak",
	"gaussian_noise",
	"contrast_variation",
	"occlusion",
)


def _normalize_bucket(bucket):
	if isinstance(bucket, str):
		return (bucket,)
	return tuple(bucket)


def _build_balanced_perturbation_plan_for_buckets(n_samples, buckets, rng):
	n_buckets = len(buckets)
	base = n_samples // n_buckets
	remainder = n_samples % n_buckets

	plan = []
	for i, bucket in enumerate(buckets):
		count = base + (1 if i < remainder else 0)
		plan.extend([bucket] * count)

	plan = np.array(plan, dtype=object)
	rng.shuffle(plan)
	return plan


def perturbation_buckets(include_combinations=False):
	"""Return perturbation buckets derived from PERTURBATION_TYPES."""
	types = list(PERTURBATION_TYPES)
	if not include_combinations:
		return [(p_type,) for p_type in types]

	buckets = []
	for r in range(1, len(types) + 1):
		buckets.extend(list(combinations(types, r)))
	return buckets


def select_perturbation_buckets(
	include_combinations=False,
	bucket_count=None,
	random_state=124,
):
	"""Select a subset of perturbation buckets.

	If bucket_count is None, all buckets are returned.
	"""
	buckets = perturbation_buckets(include_combinations=include_combinations)
	if bucket_count is None or bucket_count >= len(buckets):
		return buckets

	if bucket_count <= 0:
		raise ValueError("bucket_count must be >= 1.")

	rng = np.random.default_rng(random_state)
	idx = rng.choice(len(buckets), size=bucket_count, replace=False)
	return [buckets[i] for i in idx]


def required_perturb_count(
	min_per_bucket=50,
	include_combinations=False,
	bucket_count=None,
):
	"""Return required perturb_count based on desired minimum samples per bucket."""
	n_buckets = len(
		select_perturbation_buckets(
			include_combinations=include_combinations,
			bucket_count=bucket_count,
		)
	)
	return int(min_per_bucket) * n_buckets


def _to_object_array(items):
	"""Build a 1D object array without broadcasting nested image shapes."""
	arr = np.empty(len(items), dtype=object)
	for i, item in enumerate(items):
		arr[i] = item
	return arr


def perturbate_data(
	X_train,
	y_train,
	random_state=124,
	show_progress=True,
	perturbation_plan=None,
	available_buckets=None,
):
	"""Apply perturbations and return per-sample perturbation labels."""
	rng = np.random.default_rng(random_state)

	def _ensure_uint8_rgb(img):
		arr = np.asarray(img)
		if arr.dtype != np.uint8:
			arr = np.clip(arr, 0, 255).astype(np.uint8)
		if arr.ndim == 2:
			arr = np.stack([arr, arr, arr], axis=-1)
		return arr

	def _radial_glare(arr):
		h, w, _ = arr.shape
		cy = rng.integers(0, h)
		cx = rng.integers(0, w)
		yy, xx = np.ogrid[:h, :w]
		dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
		radius = max(1.0, rng.uniform(0.15, 0.35) * min(h, w))
		mask = np.clip(1.0 - (dist / radius), 0.0, 1.0)
		boost = rng.uniform(60.0, 120.0)
		glare = arr.astype(np.float32) + (mask[..., None] * boost)
		return np.clip(glare, 0, 255).astype(np.uint8)

	def _light_streak(arr):
		h, w, _ = arr.shape
		streak = np.zeros((h, w), dtype=np.float32)
		x0 = rng.integers(0, w)
		slope = rng.uniform(-0.6, 0.6)
		thickness = int(rng.integers(2, 7))
		for y in range(h):
			x = int(x0 + slope * (y - h / 2))
			x1 = max(0, x - thickness)
			x2 = min(w, x + thickness)
			if x1 < x2:
				streak[y, x1:x2] = 1.0
		boost = rng.uniform(35.0, 95.0)
		out = arr.astype(np.float32) + boost * streak[..., None]
		return np.clip(out, 0, 255).astype(np.uint8)

	def _gaussian_noise(arr):
		sigma = rng.uniform(8.0, 22.0)
		noise = rng.normal(0, sigma, size=arr.shape)
		out = arr.astype(np.float32) + noise
		return np.clip(out, 0, 255).astype(np.uint8)

	def _contrast_variation(arr):
		pil = Image.fromarray(np.asarray(arr).astype(np.uint8))
		pil = ImageEnhance.Contrast(pil)
		return np.asarray(pil.enhance(float(rng.uniform(0.55, 1.55))))

	def _occlusion(arr):
		h, w, _ = arr.shape
		img = np.array(arr, dtype=np.uint8)
		y1 = int(rng.integers(0, max(1, h - 10)))
		x1 = int(rng.integers(0, max(1, w - 10)))
		block_h = int(rng.integers(max(8, h // 10), max(9, h // 4)))
		block_w = int(rng.integers(max(8, w // 10), max(9, w // 4)))
		y2 = min(h, y1 + block_h)
		x2 = min(w, x1 + block_w)
		fill = int(rng.integers(0, 256))
		img[y1:y2, x1:x2, :] = fill
		return img

	perturbations = {
		"radial_glare": _radial_glare,
		"light_streak": _light_streak,
		"gaussian_noise": _gaussian_noise,
		"contrast_variation": _contrast_variation,
		"occlusion": _occlusion,
	}

	if perturbation_plan is not None:
		if len(perturbation_plan) != len(X_train):
			raise ValueError("perturbation_plan length must match X_train length.")
		for bucket in perturbation_plan:
			invalid = set(_normalize_bucket(bucket)) - set(PERTURBATION_TYPES)
			if invalid:
				raise ValueError(f"Unknown perturbation types in plan: {sorted(invalid)}")

	if available_buckets is not None:
		available_buckets = [_normalize_bucket(b) for b in available_buckets]
		for bucket in available_buckets:
			invalid = set(bucket) - set(PERTURBATION_TYPES)
			if invalid:
				raise ValueError(f"Unknown perturbation types in available_buckets: {sorted(invalid)}")
		if len(available_buckets) == 0:
			raise ValueError("available_buckets must not be empty.")

	X_augmented = []
	y_augmented = []
	perturbation_types = []
	iterator = zip(X_train, y_train)
	if show_progress:
		iterator = tqdm(iterator, total=len(X_train), desc="Applying perturbations")
	for i, (img, label) in enumerate(iterator):
		arr = _ensure_uint8_rgb(img)
		if perturbation_plan is None:
			if available_buckets is None:
				bucket = (PERTURBATION_TYPES[int(rng.integers(0, len(PERTURBATION_TYPES)))],)
			else:
				bucket = available_buckets[int(rng.integers(0, len(available_buckets)))]
		else:
			bucket = _normalize_bucket(perturbation_plan[i])

		out = arr
		for name in bucket:
			op = perturbations[name]
			out = op(out)
		X_augmented.append(out)
		y_augmented.append(label)
		perturbation_types.append("+".join(bucket))

	return (
		_to_object_array(X_augmented),
		_to_object_array(y_augmented),
		_to_object_array(perturbation_types),
	)


def augment_data(X_train, y_train, n_perturb=300, random_state=124, show_progress=True):
	"""Sample a subset to perturb, merge with clean samples, shuffle, and track type."""
	if len(X_train) != len(y_train):
		raise ValueError("X_train and y_train must have the same length.")

	if n_perturb < 0:
		raise ValueError("n_perturb must be non-negative.")

	n_perturb = min(n_perturb, len(X_train))
	rng = np.random.default_rng(random_state)

	perturb_idx = rng.choice(len(X_train), size=n_perturb, replace=False)
	X_subset = X_train[perturb_idx]
	y_subset = y_train[perturb_idx]

	X_perturbed, y_perturbed, pert_types = perturbate_data(
		X_subset,
		y_subset,
		random_state=random_state,
		show_progress=show_progress,
	)

	X_combined = np.concatenate([_to_object_array(X_train), _to_object_array(X_perturbed)])
	y_combined = np.concatenate([_to_object_array(y_train), _to_object_array(y_perturbed)])
	clean_types = _to_object_array([None] * len(X_train))
	type_combined = np.concatenate([clean_types, _to_object_array(pert_types)])

	shuffle_idx = rng.permutation(len(X_combined))
	X_combined = X_combined[shuffle_idx]
	y_combined = y_combined[shuffle_idx]
	type_combined = type_combined[shuffle_idx]

	return X_combined, y_combined, type_combined
