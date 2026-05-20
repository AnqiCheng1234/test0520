from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from .vkitti2_raw import (
    VKITTI2Raw,
    _crop_hwew,
    _imagenet_normalize_rgb_tensor,
    _numpy_to_torch,
    _rgb_preview_tensor,
)


DEFAULT_HYPERSIM_PROCESSED_BASE = Path("/mnt/drive/1111_new_works/hypersim_marigold_processed/hypersim")
SPLIT_DIR_NAMES = {
    "train": "hypersim_processed_train",
    "val": "hypersim_processed_val",
    "test": "hypersim_processed_test",
}


class HypersimProcessedRaw(VKITTI2Raw):
    """HyperSim Marigold-processed RGB-D wrapper with online pseudo-RAW generation.

    The processed filename lists store paths relative to their split directory,
    e.g. ``ai_001_001/rgb_cam_00_fr0000.png``.  The split directory, not the
    common ``hypersim`` parent, is the sample root.
    """

    def __init__(
        self,
        filelist_path: str | Path | None = None,
        *,
        processed_base: str | Path = DEFAULT_HYPERSIM_PROCESSED_BASE,
        split: str = "train",
        split_root: str | Path | None = None,
        metadata_path: str | Path | None = None,
        mode: str = "train",
        size: Tuple[int, int] = (512, 960),
        min_depth: float = 0.1,
        max_depth: float = 80.0,
        randomize_unprocessing: bool = True,
        unprocessing_preset: str = "sensor_linear_dual",
        unprocessing_mix_weights: object | None = None,
        hflip_prob: float | None = None,
    ) -> None:
        split_key = str(split)
        if split_key not in SPLIT_DIR_NAMES:
            raise ValueError(f"Unsupported HyperSim processed split {split!r}; expected one of {sorted(SPLIT_DIR_NAMES)}")

        self.processed_base = Path(processed_base).expanduser().resolve()
        self.split = split_key
        self.split_root = (
            Path(split_root).expanduser().resolve()
            if split_root is not None
            else self.processed_base / SPLIT_DIR_NAMES[split_key]
        )
        default_filelist = self.split_root / f"filename_list_{split_key}.txt"
        default_metadata = self.split_root / f"filename_meta_{split_key}.csv"
        self.metadata_path = (
            Path(metadata_path).expanduser().resolve()
            if metadata_path is not None
            else default_metadata
        )

        super().__init__(
            filelist_path=filelist_path or default_filelist,
            mode=mode,
            size=size,
            min_depth=min_depth,
            max_depth=max_depth,
            randomize_unprocessing=randomize_unprocessing,
            unprocessing_preset=unprocessing_preset,
            unprocessing_mix_weights=unprocessing_mix_weights,
            hflip_prob=hflip_prob,
        )

    def _resolve_split_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.split_root / path

    def build_sample(
        self,
        idx: int,
        *,
        py_rng: Optional[random.Random] = None,
        torch_generator: Optional[torch.Generator] = None,
        include_geometry: bool = False,
    ) -> Dict[str, Any]:
        img_path_str, depth_path_str = self.filelist[idx].split()[:2]
        img_path = self._resolve_split_path(img_path_str)
        depth_path = self._resolve_split_path(depth_path_str)

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read HyperSim processed image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        original_hw = [int(image.shape[0]), int(image.shape[1])]

        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise ValueError(f"Failed to read HyperSim processed depth: {depth_path}")
        depth = depth_raw.astype(np.float32) / 1000.0

        valid_mask = np.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)
        depth = np.where(valid_mask, depth, 0.0).astype(np.float32, copy=False)

        image, depth, valid_mask = self._resize_to_cover(
            image,
            depth,
            valid_mask,
        )

        geometry_params: Dict[str, Any] = {
            "original_hw": original_hw,
            "resized_hw": [int(image.shape[0]), int(image.shape[1])],
            "crop_box": [0, 0, int(image.shape[0]), int(image.shape[1])],
            "crop_box_format": "h_start_w_start_h_end_w_end",
            "hflip_applied": False,
        }

        if self.mode == "train":
            image, depth, valid_mask, crop_box = self._random_crop(image, depth, valid_mask, rng=py_rng)
            image, depth, valid_mask, hflip_applied = self._random_horizontal_flip(
                image,
                depth,
                valid_mask,
                rng=py_rng,
            )
            geometry_params["crop_box"] = [int(v) for v in crop_box]
            geometry_params["hflip_applied"] = bool(hflip_applied)
        else:
            image, depth, valid_mask, crop_box = self._center_crop(image, depth, valid_mask)
            geometry_params["crop_box"] = [int(v) for v in crop_box]

        image_tensor = _numpy_to_torch(
            np.transpose(np.ascontiguousarray(image), (2, 0, 1)).astype(np.float32, copy=False)
        )
        selected_sub_preset_name = self._select_unprocessing_sub_preset(
            py_rng=py_rng,
            torch_generator=torch_generator,
        )
        unprocessing_transform = self._unprocessing_transforms[selected_sub_preset_name]
        raw_tensor, isp_params = unprocessing_transform(image_tensor, generator=torch_generator)
        isp_params = self._augment_isp_params(
            isp_params,
            selected_sub_preset_name=selected_sub_preset_name,
        )
        isp_params["hypersim_format"] = "marigold_processed"
        isp_params["hypersim_split"] = self.split

        depth_sensor = cv2.resize(depth, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
        valid_mask_sensor = cv2.resize(
            valid_mask.astype(np.float32),
            (self.size[1], self.size[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0.5

        sample = {
            "raw": raw_tensor.float(),
            "depth": _numpy_to_torch(depth_sensor.astype(np.float32, copy=False)),
            "valid_mask": _numpy_to_torch(valid_mask_sensor.astype(np.uint8)).bool(),
            "isp_params": isp_params,
            "image_path": str(img_path),
            "depth_path": str(depth_path),
            "sample_name": f"{img_path.parent.name}_{img_path.stem}",
            "target_space": "metric_depth",
        }
        if include_geometry:
            sample["geometry_params"] = geometry_params
            rgb_preview = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
            sample["rgb_preview"] = _numpy_to_torch(
                np.transpose(np.ascontiguousarray(rgb_preview), (2, 0, 1)).astype(np.float32, copy=False)
            )
        return sample

    def _resize_to_cover(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_h, target_w = self.fullres_size
        height, width = image.shape[:2]
        scale = max(float(target_h) / float(height), float(target_w) / float(width))
        resized_h = int(round(height * scale))
        resized_w = int(round(width * scale))
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
        valid_mask = cv2.resize(
            valid_mask.astype(np.float32),
            (resized_w, resized_h),
            interpolation=cv2.INTER_NEAREST,
        ) > 0.5
        return image, depth.astype(np.float32, copy=False), valid_mask

    def _resize_image_to_cover(self, image: np.ndarray) -> np.ndarray:
        target_h, target_w = self.fullres_size
        height, width = image.shape[:2]
        scale = max(float(target_h) / float(height), float(target_w) / float(width))
        resized_h = int(round(height * scale))
        resized_w = int(round(width * scale))
        return cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)

    def build_rgb_baseline_input(
        self,
        idx: int,
        geometry: Dict[str, Any],
        *,
        target_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path_str, _ = self.filelist[idx].split()[:2]
        img_path = self._resolve_split_path(img_path_str)

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read HyperSim processed image for RGB baseline: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = self._resize_image_to_cover(image)
        crop_box = geometry.get("crop_box")
        if crop_box is None:
            raise ValueError(f"Missing HyperSim geometry crop_box for idx={idx}")
        image = _crop_hwew(image, tuple(crop_box))
        if bool(geometry.get("hflip_applied", False)):
            image = np.ascontiguousarray(image[:, ::-1])
        return _imagenet_normalize_rgb_tensor(image, target_hw), _rgb_preview_tensor(image, target_hw)
