"""True LOD four-directory datasets paired with RGB_normal DAv2 pseudo labels."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

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

from finetune_stf.dataset.lod_raw import _apply_crop, _sample_crop_box
from finetune_stf.dataset.lod_aug import (
    LODAugConfig,
    apply_geometric,
    apply_photometric_raw,
    apply_photometric_rgb,
)
from finetune_stf.dataset.transform import NormalizeImage, PrepareForNet


DEFAULT_LOD_TRUE_ROOT = "/home/caq/6666_raw/0000_dataset/LOD"
DEFAULT_LOD_TRUE_MANIFEST = (
    "/home/caq/6666_raw/0000_dataset/LOD/"
    "pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest.csv"
)
LOD_TRUE_NATIVE_HW = (800, 1200)
LOD_TRUE_REQUIRED_COLUMNS = (
    "pair_id",
    "split",
    "normal_id",
    "dark_id",
    "rgb_normal_path",
    "rgb_dark_path",
    "raw_normal_path",
    "raw_dark_path",
    "pseudo_depth_path",
    "label_space",
    "height",
    "width",
    "teacher_source",
)
LOD_RAW_RGB16_NORM_MODE = "uint16_div_65535"
LOD_RAW_RGB16_STORAGE_FORMAT = "raw_rgb16_png_3ch"


def _resolve_data_path(root: Path, path_str: str | Path) -> Path:
    path = Path(str(path_str).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _load_lod_true_manifest_rows(
    manifest_path: str | Path,
    lod_root: str | Path,
    split: str,
) -> list[dict[str, Any]]:
    root = Path(lod_root).expanduser().resolve()
    wanted_split = str(split)
    rows: list[dict[str, Any]] = []
    with Path(manifest_path).expanduser().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [name for name in LOD_TRUE_REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(f"{manifest_path} missing required true-LOD columns: {', '.join(missing)}")

        for row in reader:
            row_split = row["split"].strip()
            if row_split != wanted_split:
                continue
            label_space = row["label_space"].strip()
            if label_space != "inverse_relative":
                raise ValueError(f"LOD sample {row['pair_id']} has unsupported label_space={label_space!r}")
            height, width = int(row["height"]), int(row["width"])
            if (height, width) != LOD_TRUE_NATIVE_HW:
                raise ValueError(
                    f"LOD sample {row['pair_id']} has label size {(height, width)}, expected {LOD_TRUE_NATIVE_HW}"
                )
            normal_id = int(row["normal_id"])
            dark_id = int(row["dark_id"])
            if dark_id != normal_id + 1:
                raise ValueError(f"LOD sample {row['pair_id']} violates dark_id=normal_id+1")
            rows.append(
                {
                    "pair_id": row["pair_id"].strip(),
                    "split": row_split,
                    "normal_id": normal_id,
                    "dark_id": dark_id,
                    "rgb_normal_path": _resolve_data_path(root, row["rgb_normal_path"]),
                    "rgb_dark_path": _resolve_data_path(root, row["rgb_dark_path"]),
                    "raw_normal_path": _resolve_data_path(root, row["raw_normal_path"]),
                    "raw_dark_path": _resolve_data_path(root, row["raw_dark_path"]),
                    "target_path": _resolve_data_path(root, row["pseudo_depth_path"]),
                    "label_space": label_space,
                    "teacher_source": row["teacher_source"].strip(),
                }
            )
    return rows


def _load_rgb_u8(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV failed to read RGB image: {path}")
    if image_bgr.shape != (LOD_TRUE_NATIVE_HW[0], LOD_TRUE_NATIVE_HW[1], 3):
        raise ValueError(f"Expected LOD RGB shape {(*LOD_TRUE_NATIVE_HW, 3)}, got {tuple(image_bgr.shape)}: {path}")
    if image_bgr.dtype != np.uint8:
        raise ValueError(f"Expected LOD RGB dtype uint8, got {image_bgr.dtype}: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _load_raw_rgb16_png(path: Path, *, norm_mode: str) -> np.ndarray:
    if norm_mode != LOD_RAW_RGB16_NORM_MODE:
        raise ValueError(f"Unsupported true-LOD raw norm mode: {norm_mode!r}")
    raw_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw_bgr is None:
        raise ValueError(f"OpenCV failed to read RAW PNG: {path}")
    if raw_bgr.shape != (LOD_TRUE_NATIVE_HW[0], LOD_TRUE_NATIVE_HW[1], 3):
        raise ValueError(f"Expected LOD RAW shape {(*LOD_TRUE_NATIVE_HW, 3)}, got {tuple(raw_bgr.shape)}: {path}")
    if raw_bgr.dtype != np.uint16:
        raise ValueError(f"Expected LOD RAW dtype uint16, got {raw_bgr.dtype}: {path}")
    raw_rgb = raw_bgr[..., ::-1].astype(np.float32) / 65535.0
    if not np.isfinite(raw_rgb).all():
        raise ValueError(f"LOD RAW contains non-finite values after normalization: {path}")
    return raw_rgb


def _chw_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.transpose(np.ascontiguousarray(array), (2, 0, 1)).astype(np.float32, copy=False))


class _LODTrueBase(Dataset):
    dataset_name = "lod"
    model_input_tensor = "image"

    def __init__(
        self,
        *,
        lod_root: str | Path = DEFAULT_LOD_TRUE_ROOT,
        manifest_path: str | Path = DEFAULT_LOD_TRUE_MANIFEST,
        split: str = "00Train",
        size: tuple[int, int] = (512, 960),
        mode: str = "train",
        label_space: str = "inverse_relative",
        crop_mode: str = "random",
        aug_config: LODAugConfig | None = None,
    ):
        if label_space != "inverse_relative":
            raise ValueError(f"LOD true supports label_space='inverse_relative', got {label_space!r}")
        if crop_mode not in {"center", "random"}:
            raise ValueError("crop_mode must be one of {'center', 'random'}")
        self.lod_root = Path(lod_root).expanduser().resolve()
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.split = str(split)
        self.size = tuple(int(v) for v in size)
        self.mode = str(mode)
        self.label_space = str(label_space)
        self.crop_mode = "center" if self.mode == "val" else crop_mode
        self.aug_config = aug_config if self.mode == "train" and aug_config is not None and aug_config.enabled else None
        self.aug_epoch = 0
        self.rows = _load_lod_true_manifest_rows(self.manifest_path, self.lod_root, self.split)
        if not self.rows:
            raise ValueError(f"No true-LOD samples found for split={self.split} in {self.manifest_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.build_sample(idx)

    def set_epoch(self, epoch: int) -> None:
        self.aug_epoch = int(epoch)

    def _load_target(self, row: dict[str, Any]) -> np.ndarray:
        target_path = row["target_path"]
        if not target_path.is_file():
            raise FileNotFoundError(f"Missing LOD pseudo depth file: {target_path}")
        target = np.load(target_path).astype(np.float32, copy=False)
        if target.shape != LOD_TRUE_NATIVE_HW:
            raise ValueError(f"Expected LOD target shape {LOD_TRUE_NATIVE_HW}, got {tuple(target.shape)}")
        return target

    def _base_sample(
        self,
        row: dict[str, Any],
        target: np.ndarray,
        valid_mask: np.ndarray,
        *,
        include_geometry: bool,
        crop_box: tuple[int, int, int, int],
        geometry_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sample = {
            "depth": torch.from_numpy(np.ascontiguousarray(target).astype(np.float32, copy=False)),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid_mask).astype(bool)),
            "sample_id": row["pair_id"],
            "sample_name": row["pair_id"],
            "dataset": self.dataset_name,
            "split": row["split"],
            "target_space": self.label_space,
            "target_source": row["teacher_source"],
            "teacher_source": row["teacher_source"],
            "rgb_normal_path": str(row["rgb_normal_path"]),
            "rgb_dark_path": str(row["rgb_dark_path"]),
            "raw_normal_path": str(row["raw_normal_path"]),
            "raw_dark_path": str(row["raw_dark_path"]),
            "depth_path": str(row["target_path"]),
            "pseudo_depth_path": str(row["target_path"]),
            "normal_id": int(row["normal_id"]),
            "dark_id": int(row["dark_id"]),
            "model_input_tensor": self.model_input_tensor,
        }
        if include_geometry:
            if geometry_params is None:
                geometry_params = {
                    "original_hw": [int(LOD_TRUE_NATIVE_HW[0]), int(LOD_TRUE_NATIVE_HW[1])],
                    "crop_box": [int(v) for v in crop_box],
                    "crop_box_format": "h_start_w_start_h_w",
                    "hflip_applied": False,
                }
            sample["geometry_params"] = geometry_params
        return sample


class LODTrueRGBDark(_LODTrueBase):
    """Student input is LOD RGB_Dark JPG; target is RGB_normal DAv2-L pseudo label."""

    model_input_tensor = "image"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transform = Compose(
            [
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def build_sample(self, idx, *, rng=random, include_geometry=False):
        row = self.rows[idx]
        if not row["rgb_dark_path"].is_file():
            raise FileNotFoundError(f"Missing LOD RGB_Dark file: {row['rgb_dark_path']}")
        image = _load_rgb_u8(row["rgb_dark_path"])
        target = self._load_target(row)
        if image.shape[:2] != target.shape:
            raise ValueError(
                f"LOD RGB_Dark shape {image.shape[:2]} does not match pseudo label {target.shape}: {row['pair_id']}"
            )
        geometry_params = None
        if self.aug_config is not None:
            valid_mask = np.isfinite(target) & (target > 0)
            geom_rng = self.aug_config.rng(epoch=self.aug_epoch, sample_id=row["pair_id"], stream="geom")
            image, target, valid_mask, geometry_params = apply_geometric(
                image,
                target,
                valid_mask,
                rng=geom_rng,
                cfg=self.aug_config,
                size=self.size,
                crop_mode=self.crop_mode,
            )
            photo_rng = self.aug_config.rng(epoch=self.aug_epoch, sample_id=row["pair_id"], stream="photo_rgb")
            image = apply_photometric_rgb(image, rng=photo_rng, cfg=self.aug_config)
            crop_box = tuple(int(v) for v in geometry_params["crop_box"])
        else:
            crop_box = _sample_crop_box(image.shape[0], image.shape[1], self.size, self.crop_mode, rng)
            image = _apply_crop(image, crop_box)
            target = _apply_crop(target, crop_box)
            valid_mask = np.isfinite(target) & (target > 0)
            target = np.where(valid_mask, target, 0.0).astype(np.float32, copy=False)

        sample = self.transform({"image": image, "depth": target, "mask": valid_mask.astype(np.float32)})
        mask = sample.pop("mask")
        base = self._base_sample(
            row,
            sample["depth"],
            mask > 0.5,
            include_geometry=include_geometry,
            crop_box=crop_box,
            geometry_params=geometry_params,
        )
        base["image"] = torch.from_numpy(sample["image"])
        base["image_path"] = str(row["rgb_dark_path"])
        base["student_input_path"] = str(row["rgb_dark_path"])
        base["input_domain"] = "rgb"
        return base


class LODTrueRawDarkRGB16(_LODTrueBase):
    """Student input is LOD RAW_Dark uint16 3-channel PNG read as model RGB."""

    model_input_tensor = "raw"

    def __init__(
        self,
        *,
        raw_storage_format: str = LOD_RAW_RGB16_STORAGE_FORMAT,
        lod_raw_norm_mode: str = LOD_RAW_RGB16_NORM_MODE,
        **kwargs,
    ):
        if raw_storage_format != LOD_RAW_RGB16_STORAGE_FORMAT:
            raise ValueError(
                f"LOD true RAW requires raw_storage_format={LOD_RAW_RGB16_STORAGE_FORMAT!r}, "
                f"got {raw_storage_format!r}"
            )
        if lod_raw_norm_mode != LOD_RAW_RGB16_NORM_MODE:
            raise ValueError(
                f"LOD true RAW requires lod_raw_norm_mode={LOD_RAW_RGB16_NORM_MODE!r}, got {lod_raw_norm_mode!r}"
            )
        super().__init__(**kwargs)
        self.raw_storage_format = raw_storage_format
        self.lod_raw_norm_mode = lod_raw_norm_mode

    def build_sample(self, idx, *, rng=random, include_geometry=False):
        row = self.rows[idx]
        if not row["raw_dark_path"].is_file():
            raise FileNotFoundError(f"Missing LOD RAW_Dark file: {row['raw_dark_path']}")
        raw = _load_raw_rgb16_png(row["raw_dark_path"], norm_mode=self.lod_raw_norm_mode)
        target = self._load_target(row)
        if raw.shape[:2] != target.shape:
            raise ValueError(
                f"LOD RAW_Dark shape {raw.shape[:2]} does not match pseudo label {target.shape}: {row['pair_id']}"
            )
        geometry_params = None
        if self.aug_config is not None:
            valid_mask = np.isfinite(target) & (target > 0)
            geom_rng = self.aug_config.rng(epoch=self.aug_epoch, sample_id=row["pair_id"], stream="geom")
            raw, target, valid_mask, geometry_params = apply_geometric(
                raw,
                target,
                valid_mask,
                rng=geom_rng,
                cfg=self.aug_config,
                size=self.size,
                crop_mode=self.crop_mode,
            )
            photo_rng = self.aug_config.rng(epoch=self.aug_epoch, sample_id=row["pair_id"], stream="photo_raw")
            raw = apply_photometric_raw(raw, rng=photo_rng, cfg=self.aug_config)
            crop_box = tuple(int(v) for v in geometry_params["crop_box"])
        else:
            crop_box = _sample_crop_box(raw.shape[0], raw.shape[1], self.size, self.crop_mode, rng)
            raw = _apply_crop(raw, crop_box)
            target = _apply_crop(target, crop_box)
            valid_mask = np.isfinite(target) & (target > 0)
            target = np.where(valid_mask, target, 0.0).astype(np.float32, copy=False)
        image = raw.astype(np.float32, copy=False)

        base = self._base_sample(
            row,
            target,
            valid_mask,
            include_geometry=include_geometry,
            crop_box=crop_box,
            geometry_params=geometry_params,
        )
        base["raw"] = _chw_tensor(raw)
        base["image"] = _chw_tensor(image)
        base["rgb_preview"] = _chw_tensor(image)
        base["raw_path"] = str(row["raw_dark_path"])
        base["image_path"] = str(row["raw_dark_path"])
        base["student_input_path"] = str(row["raw_dark_path"])
        base["input_domain"] = "raw3"
        base["raw_storage_format"] = self.raw_storage_format
        base["lod_raw_norm_mode"] = self.lod_raw_norm_mode
        base["raw_channel_order"] = "RGB"
        return base


__all__ = [
    "DEFAULT_LOD_TRUE_MANIFEST",
    "DEFAULT_LOD_TRUE_ROOT",
    "LOD_RAW_RGB16_NORM_MODE",
    "LOD_RAW_RGB16_STORAGE_FORMAT",
    "LOD_TRUE_NATIVE_HW",
    "LODTrueRGBDark",
    "LODTrueRawDarkRGB16",
]
