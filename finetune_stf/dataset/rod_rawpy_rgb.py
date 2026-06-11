"""ROD-night dataset: RAW24 online RawPy default RGB paired with pseudo labels."""

from __future__ import annotations

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
from finetune_stf.dataset.rod_raw_rgb import ROD_NATIVE_HW
from finetune_stf.dataset.rod_raw_student_rgb import (
    DEFAULT_ROD_NIGHT_TEACHER_MANIFEST,
    DEFAULT_ROD_ROOT,
    _load_rod_manifest_rows,
)
from finetune_stf.dataset.rod_rawpy_render import (
    RAWPY_DNG_PROFILE,
    RAWPY_POSTPROCESS_PROFILE,
    RAWPY_RGB_PIPELINE,
    render_rawpy_default_rgb_from_raw24,
    validate_rawpy_profiles,
)
from finetune_stf.dataset.transform import NormalizeImage, PrepareForNet


DEFAULT_ROD_RAWPY_TEMP_ROOT = "/tmp/rod_rawpy_dng_tmp"


class RODRawRawPyRGB(Dataset):
    def __init__(
        self,
        *,
        rod_root: str | Path = DEFAULT_ROD_ROOT,
        manifest_path: str | Path = DEFAULT_ROD_NIGHT_TEACHER_MANIFEST,
        split: str = "00Train",
        size: tuple[int, int] = (512, 960),
        mode: str = "train",
        raw_source: str = "raw24",
        rgb_pipeline: str = RAWPY_RGB_PIPELINE,
        label_space: str = "inverse_relative",
        crop_mode: str = "random",
        rawpy_dng_profile: str = RAWPY_DNG_PROFILE,
        rawpy_postprocess_profile: str = RAWPY_POSTPROCESS_PROFILE,
        rawpy_temp_root: str | Path = DEFAULT_ROD_RAWPY_TEMP_ROOT,
    ):
        if raw_source != "raw24":
            raise ValueError(f"ROD RawPy RGB supports raw_source='raw24', got {raw_source!r}")
        if label_space != "inverse_relative":
            raise ValueError(f"ROD RawPy RGB supports label_space='inverse_relative', got {label_space!r}")
        validate_rawpy_profiles(
            rgb_pipeline=rgb_pipeline,
            dng_profile=rawpy_dng_profile,
            postprocess_profile=rawpy_postprocess_profile,
        )
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
        self.rawpy_dng_profile = rawpy_dng_profile
        self.rawpy_postprocess_profile = rawpy_postprocess_profile
        self.rawpy_temp_root = Path(rawpy_temp_root).expanduser()
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

        image_u8 = render_rawpy_default_rgb_from_raw24(raw_path, temp_root=self.rawpy_temp_root)
        image = image_u8.astype(np.float32) / 255.0
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
        sample["dataset_input_mode"] = "raw24_rawpy_rgb"
        sample["rawpy_dng_profile"] = self.rawpy_dng_profile
        sample["rawpy_postprocess_profile"] = self.rawpy_postprocess_profile
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
    "DEFAULT_ROD_RAWPY_TEMP_ROOT",
    "RODRawRawPyRGB",
]
