import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torchvision.transforms import Compose
except ImportError:
    class Compose:
        def __init__(self, transforms):
            self.transforms = list(transforms)

        def __call__(self, sample):
            for transform in self.transforms:
                sample = transform(sample)
            return sample

from finetune_stf.dataset.transform import NormalizeImage, PrepareForNet, Resize


DEFAULT_STF_ROOT = "/home/caq/6666_raw/seeingthroughfog"
REQUIRED_COLUMNS = ("filename_stem", "lut_preview", "lidar_proj_left")
STF_TRAIN_TARGET_MODES = ("gt_sparse", "dav2_pseudo", "da3_pseudo_sparse_metric")
STF_PSEUDO_TRAIN_TARGET_MODES = ("dav2_pseudo", "da3_pseudo_sparse_metric")
STF_DA3_DEPTH_UNITS_VALUE = "affine_invariant_depth_from_da3mono"
STF_DA3_MIN_ALIGN_POINTS = 128


def _resolve_data_path(root, path_str):
    path = Path(path_str.strip()).expanduser()
    if path.is_absolute():
        return path
    return (Path(root).expanduser().resolve() / path).resolve()


def _load_manifest_rows(manifest_path, stf_root):
    rows = []
    with Path(manifest_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(
                f"{manifest_path} is missing required STF columns: {', '.join(missing)}"
            )

        for row in reader:
            rows.append(
                {
                    "sample_name": row["filename_stem"],
                    "image_path": _resolve_data_path(stf_root, row["lut_preview"]),
                    "depth_path": _resolve_data_path(stf_root, row["lidar_proj_left"]),
                }
            )
    return rows


def _load_depth_npz(path):
    with np.load(path, allow_pickle=False) as data:
        if "arr_0" not in data.files:
            raise KeyError(f"{path} does not contain arr_0")
        return np.array(data["arr_0"], dtype=np.float32, copy=True)


def _resolve_manifest_data_path(path_str):
    return Path(path_str.strip()).expanduser().resolve()


def _load_pseudo_manifest_rows(manifest_path, split_names, target_kind="dav2_pseudo"):
    split_names = set(split_names)
    rows = []
    with Path(manifest_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"sample_name", "split", "rgb_path", "sparse_depth_path", "pseudo_depth_npy"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"{manifest_path} is missing required STF pseudo columns: {', '.join(missing)}"
            )
        for row in reader:
            if row["split"] not in split_names:
                continue
            rows.append(
                {
                    "sample_name": row["sample_name"],
                    "image_path": _resolve_manifest_data_path(row["rgb_path"]),
                    "depth_path": _resolve_manifest_data_path(row["pseudo_depth_npy"]),
                    "sparse_depth_path": _resolve_manifest_data_path(row["sparse_depth_path"]),
                    "split": row["split"],
                    "target_kind": target_kind,
                }
            )
    return rows


def _load_manifest_metadata(manifest_path):
    manifest_dir = Path(manifest_path).expanduser().resolve().parent
    metadata = []
    for name in ("run_config.json", "run_summary.json"):
        path = manifest_dir / name
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse STF pseudo metadata {path}: {exc}") from exc
        metadata.append((path, payload))
    return metadata


def _depth_units_value(payload):
    units = payload.get("depth_value_units")
    if isinstance(units, dict):
        value = units.get("value")
        if value is not None:
            return str(value)
    return None


def validate_stf_pseudo_manifest_for_target_mode(manifest_path, target_mode):
    if target_mode not in STF_PSEUDO_TRAIN_TARGET_MODES:
        return

    metadata = _load_manifest_metadata(manifest_path)
    unit_records = [
        (path, value)
        for path, payload in metadata
        for value in [_depth_units_value(payload)]
        if value is not None
    ]

    if target_mode == "dav2_pseudo":
        for path, value in unit_records:
            if value == STF_DA3_DEPTH_UNITS_VALUE:
                raise ValueError(
                    f"{manifest_path} is DA3 affine-invariant depth according to {path}; "
                    "use stf_train_target_mode='da3_pseudo_sparse_metric' instead of 'dav2_pseudo'."
                )
        return

    if target_mode == "da3_pseudo_sparse_metric":
        for path, value in unit_records:
            if value != STF_DA3_DEPTH_UNITS_VALUE:
                raise ValueError(
                    f"{manifest_path} metadata {path} declares depth_value_units.value={value!r}; "
                    f"expected {STF_DA3_DEPTH_UNITS_VALUE!r} for da3_pseudo_sparse_metric."
                )


def _solve_affine_scale_shift(x, y):
    if x.size < STF_DA3_MIN_ALIGN_POINTS:
        return None
    A = np.stack([x, np.ones_like(x)], axis=1)
    try:
        scale, shift = np.linalg.lstsq(A, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    scale = float(scale)
    shift = float(shift)
    if not np.isfinite(scale) or not np.isfinite(shift) or scale <= 1e-6:
        return None
    return scale, shift


def _fit_da3_to_sparse_metric(da3_depth, sparse_depth, min_depth, max_depth):
    valid = (
        np.isfinite(da3_depth)
        & (da3_depth > 0)
        & np.isfinite(sparse_depth)
        & (sparse_depth >= min_depth)
        & (sparse_depth <= max_depth)
    )
    valid_count = int(valid.sum())
    if valid_count < STF_DA3_MIN_ALIGN_POINTS:
        return None

    x = da3_depth[valid].astype(np.float64, copy=False)
    y = sparse_depth[valid].astype(np.float64, copy=False)

    first_fit = _solve_affine_scale_shift(x, y)
    if first_fit is None:
        return None
    scale, shift = first_fit

    residual = y - (scale * x + shift)
    residual_med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - residual_med)))
    if not np.isfinite(mad):
        return None
    inlier_threshold = max(3.0 * 1.4826 * mad, 2.0)
    inliers = np.abs(residual - residual_med) <= inlier_threshold
    inlier_count = int(inliers.sum())
    if inlier_count < STF_DA3_MIN_ALIGN_POINTS:
        return None

    refit = _solve_affine_scale_shift(x[inliers], y[inliers])
    if refit is None:
        return None
    scale, shift = refit
    return {
        "scale": scale,
        "shift": shift,
        "valid_points": valid_count,
        "inlier_points": inlier_count,
        "inlier_threshold": float(inlier_threshold),
    }


def build_da3_sparse_metric_target(da3_depth_path, sparse_depth_path, min_depth, max_depth):
    da3_depth = np.load(da3_depth_path).astype(np.float32, copy=False)
    if da3_depth.ndim != 2:
        raise RuntimeError(f"Unexpected STF DA3 pseudo target shape for {da3_depth_path}: {da3_depth.shape}")

    sparse_depth = _load_depth_npz(sparse_depth_path)
    if sparse_depth.ndim != 2:
        raise RuntimeError(f"Unexpected STF sparse target shape for {sparse_depth_path}: {sparse_depth.shape}")

    if da3_depth.shape != sparse_depth.shape:
        target_h, target_w = sparse_depth.shape
        da3_depth = cv2.resize(da3_depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    fit = _fit_da3_to_sparse_metric(da3_depth, sparse_depth, min_depth, max_depth)
    if fit is not None:
        metric_depth = fit["scale"] * da3_depth + fit["shift"]
        valid_mask = (
            np.isfinite(metric_depth)
            & (metric_depth >= min_depth)
            & (metric_depth <= max_depth)
        )
        if int(valid_mask.sum()) >= STF_DA3_MIN_ALIGN_POINTS:
            metric_depth = np.where(valid_mask, metric_depth, 0.0).astype(np.float32, copy=False)
            return metric_depth, valid_mask, {"target_source": "dense_aligned", **fit}

    valid_mask = (
        np.isfinite(sparse_depth)
        & (sparse_depth >= min_depth)
        & (sparse_depth <= max_depth)
    )
    sparse_depth = np.where(valid_mask, sparse_depth, 0.0).astype(np.float32, copy=False)
    return sparse_depth, valid_mask, {"target_source": "sparse_fallback"}


class STF(Dataset):
    def __init__(
        self,
        split,
        *,
        stf_root=DEFAULT_STF_ROOT,
        size=(512, 960),
        min_depth=1.0,
        max_depth=80.0,
        merge_test_into_train=True,
        stf_train_target_mode="gt_sparse",
        stf_pseudo_manifest=None,
    ):
        self.split = split
        self.mode = "train" if split == "train" else "val"
        self.stf_root = Path(stf_root).expanduser().resolve()
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.size = tuple(size)
        self.stf_train_target_mode = str(stf_train_target_mode)
        self.stf_pseudo_manifest = (
            Path(stf_pseudo_manifest).expanduser().resolve()
            if stf_pseudo_manifest is not None
            else None
        )

        if self.stf_train_target_mode not in STF_TRAIN_TARGET_MODES:
            raise ValueError(f"Unsupported STF train target mode: {self.stf_train_target_mode}")

        manifest_dir = self.stf_root / "manifests"
        if split == "train" and self.stf_train_target_mode in STF_PSEUDO_TRAIN_TARGET_MODES:
            if self.stf_pseudo_manifest is None or not self.stf_pseudo_manifest.is_file():
                raise FileNotFoundError(f"Missing STF pseudo manifest: {self.stf_pseudo_manifest}")
            validate_stf_pseudo_manifest_for_target_mode(
                self.stf_pseudo_manifest,
                self.stf_train_target_mode,
            )
            pseudo_splits = ("train", "test") if merge_test_into_train else ("train",)
            self.rows = _load_pseudo_manifest_rows(
                self.stf_pseudo_manifest,
                pseudo_splits,
                target_kind=self.stf_train_target_mode,
            )
        elif split == "train" and merge_test_into_train:
            manifest_paths = [
                manifest_dir / "stf_raw_depth_v1_train.csv",
                manifest_dir / "stf_raw_depth_v1_test.csv",
            ]
            self.rows = []
            for manifest_path in manifest_paths:
                if not manifest_path.is_file():
                    raise FileNotFoundError(f"Missing STF manifest: {manifest_path}")
                self.rows.extend(_load_manifest_rows(manifest_path, self.stf_root))
        else:
            manifest_paths = [manifest_dir / f"stf_raw_depth_v1_{split}.csv"]
            self.rows = []
            for manifest_path in manifest_paths:
                if not manifest_path.is_file():
                    raise FileNotFoundError(f"Missing STF manifest: {manifest_path}")
                self.rows.extend(_load_manifest_rows(manifest_path, self.stf_root))

        if not self.rows:
            raise ValueError(f"No STF samples found for split={split}")

        height, width = self.size
        self.transform = Compose(
            [
                Resize(
                    width=width,
                    height=height,
                    resize_target=self.mode == "train",
                    keep_aspect_ratio=False,
                    ensure_multiple_of=1,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def _resize_dense_target_and_mask(self, depth, valid_mask):
        target_h, target_w = self.size
        depth_resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(
            valid_mask.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        valid = (mask_resized > 0.5) & np.isfinite(depth_resized) & (depth_resized > 0)
        depth_resized = np.where(valid, depth_resized, 0.0).astype(np.float32, copy=False)
        return depth_resized, valid

    def _resize_depth_and_mask(self, depth, valid_mask):
        target_h, target_w = self.size
        depth_resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        mask_resized = cv2.resize(
            valid_mask.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        valid = (mask_resized > 0.5) & np.isfinite(depth_resized) & (depth_resized > 0)
        depth_resized = np.where(valid, depth_resized, 0.0).astype(np.float32, copy=False)
        return depth_resized, valid

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = row["image_path"]
        depth_path = row["depth_path"]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read STF image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        target_kind = row.get("target_kind", "gt_sparse")
        if target_kind == "dav2_pseudo":
            depth = np.load(depth_path).astype(np.float32, copy=False)
            if depth.ndim != 2:
                raise RuntimeError(f"Unexpected STF pseudo target shape for {depth_path}: {depth.shape}")
            valid_mask = np.isfinite(depth) & (depth > 0)
            depth = np.where(valid_mask, depth, 0.0).astype(np.float32, copy=False)
            if tuple(depth.shape[:2]) != self.size:
                depth, valid_mask = self._resize_dense_target_and_mask(depth, valid_mask)
        elif target_kind == "da3_pseudo_sparse_metric":
            depth, valid_mask, target_meta = build_da3_sparse_metric_target(
                depth_path,
                row["sparse_depth_path"],
                self.min_depth,
                self.max_depth,
            )
            if tuple(depth.shape[:2]) != self.size:
                if target_meta["target_source"] == "sparse_fallback":
                    depth, valid_mask = self._resize_depth_and_mask(depth, valid_mask)
                else:
                    depth, valid_mask = self._resize_dense_target_and_mask(depth, valid_mask)
        else:
            depth = _load_depth_npz(depth_path)
            valid_mask = np.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)
            depth = np.where(valid_mask, depth, 0.0).astype(np.float32, copy=False)
            target_meta = {"target_source": "sparse_gt"}

        sample = self.transform({"image": image, "depth": depth, "mask": valid_mask.astype(np.float32)})
        mask = sample.pop("mask")

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        sample["valid_mask"] = torch.from_numpy(mask > 0.5)
        sample["image_path"] = str(image_path)
        sample["depth_path"] = str(depth_path)
        sample["sample_name"] = row["sample_name"]
        sample["split"] = row.get("split", self.split)
        sample["rgb_path"] = str(image_path)
        sample["rgb_src_path"] = str(image_path)
        sample["rgb_eval_path"] = str(image_path)
        sample["target_space"] = "inverse_relative" if target_kind == "dav2_pseudo" else "metric_depth"
        sample["target_kind"] = target_kind
        sample["target_source"] = target_meta["target_source"] if target_kind != "dav2_pseudo" else "dense_pseudo"
        if "sparse_depth_path" in row:
            sample["sparse_depth_path"] = str(row["sparse_depth_path"])
            sample["pseudo_depth_path"] = str(depth_path)
        return sample
