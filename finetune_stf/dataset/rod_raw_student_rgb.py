"""ROD-night dataset: RAW24 online student RGB paired with cached pseudo labels."""

from __future__ import annotations

import csv
import random
from pathlib import Path

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

from finetune_stf.dataset.lod_raw import _apply_crop, _sample_crop_box
from finetune_stf.dataset.rod_raw_rgb import (
    DEGREEN_GAINS,
    ROD_NATIVE_HW,
    STUDENT_GAMMA,
    STUDENT_WHITE_PERCENTILE,
    render_pipeline,
    unpack_raw24,
)
from finetune_stf.dataset.transform import NormalizeImage, PrepareForNet


DEFAULT_ROD_ROOT = "/home/caq/6666_raw/0000_dataset/ROD"
DEFAULT_ROD_NIGHT_TEACHER_MANIFEST = (
    "/home/caq/6666_raw/0000_dataset/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/"
    "rod_night_dav2_rel_manifest.csv"
)
ROD_REQUIRED_COLUMNS = (
    "sample_id",
    "split",
    "raw24_path",
    "pseudo_depth_path",
    "label_space",
    "height",
    "width",
)


def _resolve_data_path(root: Path, path_str: str) -> Path:
    path = Path(path_str.strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _load_rod_manifest_rows(manifest_path: str | Path, rod_root: str | Path, split: str) -> list[dict[str, object]]:
    root = Path(rod_root).expanduser().resolve()
    wanted_split = str(split)
    rows: list[dict[str, object]] = []
    with Path(manifest_path).expanduser().open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = [name for name in ROD_REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(f"{manifest_path} is missing required ROD columns: {', '.join(missing)}")

        for row in reader:
            row_split = row["split"].strip()
            if row_split != wanted_split:
                continue
            sample_id = row["sample_id"].strip()
            if not sample_id.startswith("night-"):
                raise ValueError(f"ROD night manifest contains unexpected sample_id={sample_id!r}")
            label_space = row["label_space"].strip()
            if label_space != "inverse_relative":
                raise ValueError(f"ROD sample {sample_id} has unsupported label_space={label_space!r}")
            height = int(row["height"])
            width = int(row["width"])
            if (height, width) != ROD_NATIVE_HW:
                raise ValueError(f"ROD sample {sample_id} has label size {(height, width)}, expected {ROD_NATIVE_HW}")
            raw_path = _resolve_data_path(root, row["raw24_path"])
            target_path = _resolve_data_path(root, row["pseudo_depth_path"])
            if "LOD" in str(raw_path) or "LOD" in str(target_path):
                raise ValueError(f"ROD manifest must not point to LOD paths: {sample_id}")
            rows.append(
                {
                    "split": row_split,
                    "sample_id": sample_id,
                    "raw_path": raw_path,
                    "rggb_path": _resolve_data_path(root, row["rggb_path"]) if row.get("rggb_path", "").strip() else None,
                    "target_path": target_path,
                    "label_space": label_space,
                }
            )
    return rows


class RODRawStudentRGB(Dataset):
    def __init__(
        self,
        *,
        rod_root: str | Path = DEFAULT_ROD_ROOT,
        manifest_path: str | Path = DEFAULT_ROD_NIGHT_TEACHER_MANIFEST,
        split: str = "00Train",
        size: tuple[int, int] = (512, 960),
        mode: str = "train",
        raw_source: str = "raw24",
        rgb_pipeline: str = "student_dark_degreen_v1",
        label_space: str = "inverse_relative",
        crop_mode: str = "random",
        student_white_percentile: float = STUDENT_WHITE_PERCENTILE,
        student_gamma: float = STUDENT_GAMMA,
        student_channel_gains: tuple[float, float, float] = tuple(float(v) for v in DEGREEN_GAINS.tolist()),
    ):
        if raw_source != "raw24":
            raise ValueError(f"ROD v1 supports raw_source='raw24', got {raw_source!r}")
        if rgb_pipeline != "student_dark_degreen_v1":
            raise ValueError(f"ROD v1 supports rgb_pipeline='student_dark_degreen_v1', got {rgb_pipeline!r}")
        if label_space != "inverse_relative":
            raise ValueError(f"ROD v1 supports label_space='inverse_relative', got {label_space!r}")
        if abs(float(student_white_percentile) - STUDENT_WHITE_PERCENTILE) > 1e-6:
            raise ValueError("student_white_percentile must match student_dark_degreen_v1")
        if abs(float(student_gamma) - STUDENT_GAMMA) > 1e-6:
            raise ValueError("student_gamma must match student_dark_degreen_v1")
        if not np.allclose(np.asarray(student_channel_gains, dtype=np.float32), DEGREEN_GAINS):
            raise ValueError("student_channel_gains must match student_dark_degreen_v1")
        if crop_mode not in {"center", "random"}:
            raise ValueError("crop_mode must be one of {'center', 'random'}")

        self.rod_root = Path(rod_root).expanduser().resolve()
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.split = str(split)
        self.size = tuple(int(v) for v in size)
        self.mode = str(mode)
        self.raw_source = raw_source
        self.rgb_pipeline = rgb_pipeline
        self.label_space = label_space
        self.crop_mode = "center" if self.mode == "val" else crop_mode
        self.rows = _load_rod_manifest_rows(self.manifest_path, self.rod_root, self.split)
        if not self.rows:
            raise ValueError(f"No ROD samples found for split={self.split} in {self.manifest_path}")

        self.transform = Compose(
            [
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.build_sample(idx)

    def build_sample(self, idx, *, rng=random, include_geometry=False):
        row = self.rows[idx]
        raw_path = row["raw_path"]
        target_path = row["target_path"]
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing ROD raw24 file: {raw_path}")
        if not target_path.is_file():
            raise FileNotFoundError(f"Missing ROD pseudo depth file: {target_path}")

        raw = unpack_raw24(raw_path)
        image = render_pipeline(raw, self.rgb_pipeline).astype(np.float32) / 255.0
        target = np.load(target_path).astype(np.float32, copy=False)
        if target.shape != ROD_NATIVE_HW:
            raise ValueError(f"Expected ROD target shape {ROD_NATIVE_HW}, got {tuple(target.shape)}")
        if image.shape[:2] != ROD_NATIVE_HW:
            raise ValueError(f"Expected ROD image shape {ROD_NATIVE_HW}, got {tuple(image.shape[:2])}")

        crop_box = _sample_crop_box(image.shape[0], image.shape[1], self.size, self.crop_mode, rng)
        image = _apply_crop(image, crop_box)
        target = _apply_crop(target, crop_box)

        valid_mask = np.isfinite(target) & (target > 0)
        target = np.where(valid_mask, target, 0.0).astype(np.float32, copy=False)

        sample = self.transform({"image": image, "depth": target, "mask": valid_mask.astype(np.float32)})
        mask = sample.pop("mask")
        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        sample["valid_mask"] = torch.from_numpy(mask > 0.5)
        sample["sample_id"] = row["sample_id"]
        sample["sample_name"] = row["sample_id"]
        sample["dataset"] = "rod"
        sample["split"] = row["split"]
        sample["target_space"] = self.label_space
        sample["raw_path"] = str(raw_path)
        sample["image_path"] = str(raw_path)
        sample["depth_path"] = str(target_path)
        sample["pseudo_depth_path"] = str(target_path)
        sample["rgb_pipeline"] = self.rgb_pipeline
        if row.get("rggb_path") is not None:
            sample["rggb_path"] = str(row["rggb_path"])
        if include_geometry:
            sample["geometry_params"] = {
                "original_hw": [int(ROD_NATIVE_HW[0]), int(ROD_NATIVE_HW[1])],
                "crop_box": [int(v) for v in crop_box],
                "crop_box_format": "h_start_w_start_h_w",
                "hflip_applied": False,
            }
        return sample


__all__ = [
    "DEFAULT_ROD_ROOT",
    "DEFAULT_ROD_NIGHT_TEACHER_MANIFEST",
    "RODRawStudentRGB",
]
