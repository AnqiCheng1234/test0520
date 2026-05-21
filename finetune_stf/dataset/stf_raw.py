import csv
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

from finetune_stf.dataset.raw_utils import (
    DEFAULT_RAW_NPZ_ROOT,
    STF_RAW_DECODE_MODES,
    bayer_to_3ch,
    decode_stf_raw_4ch,
    load_rectified_bayer_npz,
    normalize_raw,
    normalize_raw_4ch,
)
from finetune_stf.dataset.stf import (
    DEFAULT_STF_ROOT,
    REQUIRED_COLUMNS,
    STF_PSEUDO_TRAIN_TARGET_MODES,
    STF_TRAIN_TARGET_MODES,
    build_da3_sparse_metric_target,
    _load_depth_npz,
    _resolve_data_path,
    validate_stf_pseudo_manifest_for_target_mode,
)
from finetune_stf.dataset.transform import NormalizeImage, PrepareForNet, Resize


STF_RAW_NATIVE_HW = (512, 960)
DEFAULT_STF_PSEUDO_MANIFEST = (
    "/mnt/drive/3333_raw/seeing_through_fog/"
    "pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/"
    "stf_rgb_lut_manifest_6216.csv"
)
STF_FAST_EVAL_BACKENDS = ("proxy", "sparse")


def _load_manifest_rows_raw(manifest_path, stf_root, raw_npz_root):
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
            sample_name = row["filename_stem"]
            rows.append(
                {
                    "sample_name": sample_name,
                    "image_path": (raw_npz_root / f"{sample_name}.npz").resolve(),
                    "depth_path": _resolve_data_path(stf_root, row["lidar_proj_left"]),
                    "lut_preview": _resolve_data_path(stf_root, row["lut_preview"]),
                    "split": row.get("official_split", ""),
                    "target_kind": "gt_sparse",
                }
            )
    return rows


def _resolve_manifest_data_path(path_str):
    return Path(path_str.strip()).expanduser().resolve()


def _load_pseudo_manifest_rows(manifest_path, raw_npz_root, split_names, target_kind="dav2_pseudo"):
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
            sample_name = row["sample_name"]
            rows.append(
                {
                    "sample_name": sample_name,
                    "image_path": (raw_npz_root / f"{sample_name}.npz").resolve(),
                    "depth_path": _resolve_manifest_data_path(row["pseudo_depth_npy"]),
                    "sparse_depth_path": _resolve_manifest_data_path(row["sparse_depth_path"]),
                    "lut_preview": _resolve_manifest_data_path(row["rgb_path"]),
                    "split": row["split"],
                    "target_kind": target_kind,
                }
            )
    return rows


def _validate_raw_root_decode_mode(raw_npz_root, decode_mode):
    root_str = str(Path(raw_npz_root).expanduser())
    is_legacy_root = "cam_stereo_left_bayer_rect" in root_str
    is_canonical_root = "canonical" in root_str.lower()
    if is_legacy_root and decode_mode == "canonical_decomp16":
        raise ValueError(
            "The legacy STF rectified NPZ root stores channels as [B,G,G,R]; "
            "use legacy_companded or legacy_online_decomp16."
        )
    if is_canonical_root and decode_mode != "canonical_decomp16":
        raise ValueError(
            "A canonical STF RAW root should be used with --stf-raw-decode-mode canonical_decomp16."
        )


class STF_RAW(Dataset):
    def __init__(
        self,
        split,
        *,
        stf_root=DEFAULT_STF_ROOT,
        raw_npz_root=DEFAULT_RAW_NPZ_ROOT,
        size=(512, 960),
        min_depth=1.0,
        max_depth=80.0,
        merge_test_into_train=True,
        norm_mode="companded",
        channel_mode="rgb_avg_g",
        use_imagenet_norm=True,
        input_mode="raw_naive",
        stf_raw_decode_mode="legacy_companded",
        stf_train_target_mode="gt_sparse",
        stf_pseudo_manifest=DEFAULT_STF_PSEUDO_MANIFEST,
        depth_mode="fast",
        fast_eval_backend="sparse",
    ):
        self.split = split
        self.mode = "train" if split == "train" else "val"
        self.stf_root = Path(stf_root).expanduser().resolve()
        self.raw_npz_root = Path(raw_npz_root).expanduser().resolve()
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.size = tuple(size)
        self.norm_mode = norm_mode
        self.channel_mode = channel_mode
        self.use_imagenet_norm = bool(use_imagenet_norm)
        self.input_mode = input_mode
        self.stf_raw_decode_mode = str(stf_raw_decode_mode)
        self.stf_train_target_mode = str(stf_train_target_mode)
        self.stf_pseudo_manifest = Path(stf_pseudo_manifest).expanduser().resolve()
        self.depth_mode = str(depth_mode)
        self.fast_eval_backend = str(fast_eval_backend)

        if self.stf_raw_decode_mode not in STF_RAW_DECODE_MODES:
            raise ValueError(f"Unsupported STF RAW decode mode: {self.stf_raw_decode_mode}")
        if self.stf_train_target_mode not in STF_TRAIN_TARGET_MODES:
            raise ValueError(f"Unsupported STF train target mode: {self.stf_train_target_mode}")
        if self.fast_eval_backend not in STF_FAST_EVAL_BACKENDS:
            raise ValueError(f"Unsupported STF fast_eval_backend: {self.fast_eval_backend}")
        if self.stf_raw_decode_mode != "legacy_companded" and self.norm_mode != "passthrough":
            raise ValueError(
                f"{self.stf_raw_decode_mode} already returns [0,1] decompanded RAW; "
                "use norm_mode='passthrough' to avoid a second normalization."
            )
        _validate_raw_root_decode_mode(self.raw_npz_root, self.stf_raw_decode_mode)

        manifest_dir = self.stf_root / "manifests"
        if split == "train" and self.stf_train_target_mode in STF_PSEUDO_TRAIN_TARGET_MODES:
            if not self.stf_pseudo_manifest.is_file():
                raise FileNotFoundError(f"Missing STF pseudo manifest: {self.stf_pseudo_manifest}")
            validate_stf_pseudo_manifest_for_target_mode(
                self.stf_pseudo_manifest,
                self.stf_train_target_mode,
            )
            pseudo_splits = ("train", "test") if merge_test_into_train else ("train",)
            self.rows = _load_pseudo_manifest_rows(
                self.stf_pseudo_manifest,
                self.raw_npz_root,
                pseudo_splits,
                target_kind=self.stf_train_target_mode,
            )
        else:
            if split == "train" and merge_test_into_train:
                manifest_paths = [
                    manifest_dir / "stf_raw_depth_v1_train.csv",
                    manifest_dir / "stf_raw_depth_v1_test.csv",
                ]
            else:
                manifest_paths = [manifest_dir / f"stf_raw_depth_v1_{split}.csv"]

            self.rows = []
            for manifest_path in manifest_paths:
                if not manifest_path.is_file():
                    raise FileNotFoundError(f"Missing STF manifest: {manifest_path}")
                self.rows.extend(
                    _load_manifest_rows_raw(manifest_path, self.stf_root, self.raw_npz_root)
                )

        if not self.rows:
            raise ValueError(f"No STF RAW samples found for split={split}")

        transforms = []
        # raw_ram: no ImageNet norm in dataset -- model handles it internally
        if self.input_mode != "raw_ram" and self.use_imagenet_norm:
            transforms.append(
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        transforms.append(PrepareForNet())
        self.transform = Compose(transforms)

    def _resize_depth_and_mask(self, depth, valid_mask):
        target_h, target_w = self.size
        depth_resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        mask_resized = cv2.resize(
            valid_mask.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        return depth_resized.astype(np.float32, copy=False), mask_resized > 0.5

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

    def __len__(self):
        return len(self.rows)

    def build_sample(self, idx, *, include_geometry=False):
        row = self.rows[idx]
        image_path = row["image_path"]
        depth_path = row["depth_path"]

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing STF RAW NPZ: {image_path}")

        bayer_rect = load_rectified_bayer_npz(image_path)
        if tuple(bayer_rect.shape[:2]) != self.size:
            raise ValueError(
                f"Expected STF packed Bayer with spatial size {self.size}, got {tuple(bayer_rect.shape[:2])}"
            )
        bayer_rect = decode_stf_raw_4ch(bayer_rect, decode_mode=self.stf_raw_decode_mode)
        if self.input_mode == "raw_ram":
            # Return 4-channel packed Bayer normalized to [0, 1]
            image = normalize_raw_4ch(bayer_rect, norm_mode=self.norm_mode)
        else:
            image = bayer_to_3ch(bayer_rect, channel_mode=self.channel_mode)
            image = normalize_raw(image, norm_mode=self.norm_mode)

        target_kind = row.get("target_kind", "gt_sparse")
        if target_kind == "dav2_pseudo":
            depth = np.load(depth_path).astype(np.float32, copy=False)
            if depth.ndim != 2:
                raise RuntimeError(f"Unexpected STF pseudo target shape for {depth_path}: {depth.shape}")
            valid_mask = np.isfinite(depth) & (depth > 0)
            depth = np.where(valid_mask, depth, 0.0).astype(np.float32, copy=False)
            if tuple(depth.shape[:2]) != self.size:
                depth, valid_mask = self._resize_dense_target_and_mask(depth, valid_mask)
            target_meta = {"target_source": "dense_pseudo"}
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
            if self.mode == "train":
                depth, valid_mask = self._resize_depth_and_mask(depth, valid_mask)
            target_meta = {"target_source": "sparse_gt"}

        sample = self.transform(
            {"image": image, "depth": depth, "mask": valid_mask.astype(np.float32)}
        )
        mask = sample.pop("mask")

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        sample["valid_mask"] = torch.from_numpy(mask > 0.5)
        sample["image_path"] = str(image_path)
        sample["raw_path"] = str(image_path)
        sample["depth_path"] = str(depth_path)
        sample["sample_name"] = row["sample_name"]
        sample["split"] = row.get("split", self.split)
        sample["lut_preview"] = str(row["lut_preview"])
        sample["rgb_path"] = str(row["lut_preview"])
        sample["rgb_src_path"] = str(row["lut_preview"])
        sample["rgb_eval_path"] = str(row["lut_preview"])
        sample["target_space"] = "inverse_relative" if target_kind == "dav2_pseudo" else "metric_depth"
        sample["target_kind"] = target_kind
        sample["target_source"] = target_meta["target_source"]
        sample["stf_raw_decode_mode"] = self.stf_raw_decode_mode
        sample["norm_mode"] = self.norm_mode
        if "sparse_depth_path" in row:
            sample["sparse_depth_path"] = str(row["sparse_depth_path"])
            sample["pseudo_depth_path"] = str(depth_path)
        if self.input_mode == "raw_ram":
            sample["raw"] = sample["image"]
        if self.mode != "train":
            sample["depth_mode"] = self.depth_mode
            sample["fast_eval_backend"] = self.fast_eval_backend
        return sample

    def __getitem__(self, idx):
        return self.build_sample(idx)
