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
    bayer_to_3ch,
    load_rectified_bayer_npz,
    normalize_raw,
    normalize_raw_4ch,
)
from finetune_stf.dataset.stf import (
    DEFAULT_STF_ROOT,
    REQUIRED_COLUMNS,
    _load_depth_npz,
    _resolve_data_path,
)
from finetune_stf.dataset.transform import NormalizeImage, PrepareForNet, Resize


STF_RAW_NATIVE_HW = (512, 960)


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
                }
            )
    return rows


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

        manifest_dir = self.stf_root / "manifests"
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

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
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
        if self.input_mode == "raw_ram":
            # Return 4-channel packed Bayer normalized to [0, 1]
            image = normalize_raw_4ch(bayer_rect, norm_mode=self.norm_mode)
        else:
            image = bayer_to_3ch(bayer_rect, channel_mode=self.channel_mode)
            image = normalize_raw(image, norm_mode=self.norm_mode)

        depth = _load_depth_npz(depth_path)
        valid_mask = np.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)
        depth = np.where(valid_mask, depth, 0.0).astype(np.float32, copy=False)
        if self.mode == "train":
            depth, valid_mask = self._resize_depth_and_mask(depth, valid_mask)

        sample = self.transform(
            {"image": image, "depth": depth, "mask": valid_mask.astype(np.float32)}
        )
        mask = sample.pop("mask")

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        sample["valid_mask"] = torch.from_numpy(mask > 0.5)
        sample["image_path"] = str(image_path)
        sample["depth_path"] = str(depth_path)
        sample["sample_name"] = row["sample_name"]
        sample["target_space"] = "metric_depth"
        return sample
