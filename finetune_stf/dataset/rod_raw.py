"""ROD-night dataset: packed Bayer RAW4 paired with cached pseudo labels."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from finetune_stf.dataset.lod_raw import _apply_crop, _sample_crop_box
from finetune_stf.dataset.rod_raw_rgb import ROD_NATIVE_HW, unpack_raw24
from finetune_stf.dataset.rod_raw_student_rgb import (
    DEFAULT_ROD_NIGHT_TEACHER_MANIFEST,
    DEFAULT_ROD_ROOT,
    _load_rod_manifest_rows,
)


PACKED_BAYER_SOURCE = "rggb_cache_prefer_raw24_fallback"


def pack_raw24_to_rggb(raw: np.ndarray) -> np.ndarray:
    """Pack true RGGB RAW24 plane to [R, Gr, Gb, B] at native ROD size."""

    expected_hw = (ROD_NATIVE_HW[0] * 2, ROD_NATIVE_HW[1] * 2)
    if tuple(raw.shape) != expected_hw:
        raise ValueError(f"Expected unpacked ROD RAW shape {expected_hw}, got {tuple(raw.shape)}")
    return np.stack(
        (
            raw[0::2, 0::2],
            raw[0::2, 1::2],
            raw[1::2, 0::2],
            raw[1::2, 1::2],
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def packed_bayer_to_base_rgb_np(raw4: np.ndarray) -> np.ndarray:
    """Preview RGB: [R, (Gr+Gb)/2, B] from packed Bayer [R, Gr, Gb, B]."""

    if raw4.ndim != 3 or raw4.shape[-1] != 4:
        raise ValueError(f"Expected packed Bayer raw4 HWC, got {tuple(raw4.shape)}")
    return np.stack(
        (
            raw4[..., 0],
            0.5 * (raw4[..., 1] + raw4[..., 2]),
            raw4[..., 3],
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def _load_rggb_or_raw24(row: dict[str, object], *, allow_raw24_fallback: bool) -> tuple[np.ndarray, str]:
    rggb_path = row.get("rggb_path")
    if isinstance(rggb_path, Path) and rggb_path.is_file():
        raw4 = np.load(rggb_path, mmap_mode="r")
        source = "rggb_cache"
    else:
        if not allow_raw24_fallback:
            raise FileNotFoundError(f"Missing ROD rggb cache and fallback is disabled: {rggb_path}")
        raw_path = row["raw_path"]
        if not isinstance(raw_path, Path) or not raw_path.is_file():
            raise FileNotFoundError(f"Missing ROD raw24 fallback file: {raw_path}")
        raw4 = pack_raw24_to_rggb(unpack_raw24(raw_path))
        source = "raw24_fallback"

    if tuple(raw4.shape) != (*ROD_NATIVE_HW, 4):
        raise ValueError(f"Expected ROD packed Bayer shape {(*ROD_NATIVE_HW, 4)}, got {tuple(raw4.shape)}")
    if not np.issubdtype(raw4.dtype, np.floating):
        raise ValueError(f"Expected ROD packed Bayer float data, got dtype={raw4.dtype}")
    return raw4, source


def _chw_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.transpose(np.ascontiguousarray(array), (2, 0, 1)).astype(np.float32, copy=False))


class RODRaw(Dataset):
    def __init__(
        self,
        *,
        rod_root: str | Path = DEFAULT_ROD_ROOT,
        manifest_path: str | Path = DEFAULT_ROD_NIGHT_TEACHER_MANIFEST,
        split: str = "00Train",
        size: tuple[int, int] = (512, 960),
        mode: str = "train",
        raw_source: str = "raw24",
        label_space: str = "inverse_relative",
        crop_mode: str = "random",
        allow_raw24_fallback: bool = True,
    ):
        if raw_source != "raw24":
            raise ValueError(f"ROD raw v1 supports raw_source='raw24', got {raw_source!r}")
        if label_space != "inverse_relative":
            raise ValueError(f"ROD raw v1 supports label_space='inverse_relative', got {label_space!r}")
        if crop_mode not in {"center", "random"}:
            raise ValueError("crop_mode must be one of {'center', 'random'}")

        self.rod_root = Path(rod_root).expanduser().resolve()
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.split = str(split)
        self.size = tuple(int(v) for v in size)
        self.mode = str(mode)
        self.raw_source = raw_source
        self.label_space = label_space
        self.crop_mode = "center" if self.mode == "val" else crop_mode
        self.allow_raw24_fallback = bool(allow_raw24_fallback)
        self.packed_bayer_source = PACKED_BAYER_SOURCE
        self.rows = _load_rod_manifest_rows(self.manifest_path, self.rod_root, self.split)
        if not self.rows:
            raise ValueError(f"No ROD raw samples found for split={self.split} in {self.manifest_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.build_sample(idx)

    def build_sample(self, idx, *, rng=random, include_geometry=False):
        row = self.rows[idx]
        target_path = row["target_path"]
        if not isinstance(target_path, Path) or not target_path.is_file():
            raise FileNotFoundError(f"Missing ROD pseudo depth file: {target_path}")

        raw4, loaded_from = _load_rggb_or_raw24(row, allow_raw24_fallback=self.allow_raw24_fallback)
        target = np.load(target_path, mmap_mode="r").astype(np.float32, copy=False)
        if target.shape != ROD_NATIVE_HW:
            raise ValueError(f"Expected ROD target shape {ROD_NATIVE_HW}, got {tuple(target.shape)}")

        crop_box = _sample_crop_box(raw4.shape[0], raw4.shape[1], self.size, self.crop_mode, rng)
        raw4 = _apply_crop(np.asarray(raw4, dtype=np.float32), crop_box)
        target = _apply_crop(target, crop_box)

        if not np.all(np.isfinite(raw4)):
            raise ValueError(f"ROD raw4 contains non-finite values: sample={row['sample_id']}")
        if float(np.min(raw4)) < -1e-6 or float(np.max(raw4)) > 1.0 + 1e-6:
            raise ValueError(f"ROD raw4 expected in [0,1]: sample={row['sample_id']}")

        image = packed_bayer_to_base_rgb_np(raw4)
        valid_mask = np.isfinite(target) & (target > 0)
        target = np.where(valid_mask, target, 0.0).astype(np.float32, copy=False)

        sample = {
            "raw": _chw_tensor(raw4),
            "image": _chw_tensor(image),
            "rgb_preview": _chw_tensor(image),
            "depth": torch.from_numpy(np.ascontiguousarray(target).astype(np.float32, copy=False)),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid_mask).astype(bool)),
            "sample_id": row["sample_id"],
            "sample_name": row["sample_id"],
            "dataset": "rod",
            "split": row["split"],
            "target_space": self.label_space,
            "raw_path": str(row["raw_path"]),
            "image_path": str(row["raw_path"]),
            "depth_path": str(target_path),
            "pseudo_depth_path": str(target_path),
            "packed_bayer_source": self.packed_bayer_source,
            "packed_bayer_loaded_from": loaded_from,
            "model_input_tensor": "raw",
        }
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


class RODRawRGB3(RODRaw):
    """ROD RAW view for 3-channel RamCore3 models: raw=[R,(Gr+Gb)/2,B]."""

    def build_sample(self, idx, *, rng=random, include_geometry=False):
        sample = super().build_sample(idx, rng=rng, include_geometry=include_geometry)
        sample["raw"] = sample["image"]
        sample["raw_rgb3_source"] = "packed_bayer_to_base_rgb_np"
        sample["raw_rgb3_channel_order"] = "R_Gavg_B"
        sample["dataset_input_mode"] = "raw24_base_rgb3"
        return sample


__all__ = [
    "PACKED_BAYER_SOURCE",
    "RODRaw",
    "RODRawRGB3",
    "pack_raw24_to_rggb",
    "packed_bayer_to_base_rgb_np",
]
