from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_TRAIN_LIST = (
    Path(__file__).resolve().parents[3]
    / "finetune_stf"
    / "dataset"
    / "splits"
    / "vkitti2"
    / "train.txt"
)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CONTROL_RAW_STORAGE_CHOICES = ("not_applicable",)
CONTROL_FULLRES_EVEN_POLICY_CHOICES = ("crop_bottom_to_even",)
CONTROL_RGB_INPUT_SPACE_CHOICES = ("halfres_2x2_area",)
CONTROL_DEPTH_TARGET_SPACE_CHOICES = ("halfres_2x2_valid_mean",)


def validate_vkitti_halfres_rgb_depth_semantics(
    *,
    raw_storage_format: str,
    fullres_even_policy: str,
    rgb_input_space: str,
    depth_target_space: str,
) -> None:
    expected = {
        "raw_storage_format": "not_applicable",
        "fullres_even_policy": "crop_bottom_to_even",
        "rgb_input_space": "halfres_2x2_area",
        "depth_target_space": "halfres_2x2_valid_mean",
    }
    actual = {
        "raw_storage_format": raw_storage_format,
        "fullres_even_policy": fullres_even_policy,
        "rgb_input_space": rgb_input_space,
        "depth_target_space": depth_target_space,
    }
    for key, expected_value in expected.items():
        if actual[key] != expected_value:
            raise ValueError(f"{key} must be {expected_value!r} for VKITTI2HalfresRGBDepth, got {actual[key]!r}")


def _numpy_dtype_to_torch(dtype: np.dtype) -> torch.dtype:
    if dtype == np.float32:
        return torch.float32
    if dtype == np.float64:
        return torch.float64
    if dtype == np.uint8:
        return torch.uint8
    if dtype == np.int64:
        return torch.int64
    if dtype == np.int32:
        return torch.int32
    raise TypeError(f"Unsupported numpy dtype for tensor conversion: {dtype}")


def _numpy_to_torch(array: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(array)
    torch_dtype = _numpy_dtype_to_torch(array.dtype)
    return torch.frombuffer(bytearray(array.tobytes()), dtype=torch_dtype).view(*array.shape)


def _crop_hwew(array: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
    h_start, w_start, h_end, w_end = [int(v) for v in crop_box]
    return np.ascontiguousarray(array[h_start:h_end, w_start:w_end, ...])


def _rgb_preview_tensor_from_array(image_rgb: np.ndarray) -> torch.Tensor:
    image_rgb = np.clip(image_rgb, 0.0, 1.0)
    return _numpy_to_torch(
        np.transpose(np.ascontiguousarray(image_rgb), (2, 0, 1)).astype(np.float32, copy=False)
    )


def _imagenet_normalize_rgb_tensor_from_array(image_rgb: np.ndarray) -> torch.Tensor:
    image_rgb = np.clip(image_rgb, 0.0, 1.0)
    normalized = (image_rgb.astype(np.float32, copy=False) - IMAGENET_MEAN) / IMAGENET_STD
    return _numpy_to_torch(np.transpose(np.ascontiguousarray(normalized), (2, 0, 1)).astype(np.float32, copy=False))


def _downsample_rgb_2x2_area_from_even_fullres(image_rgb: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"Expected even fullres RGB shape, got {(height, width)}")
    blocks = np.ascontiguousarray(image_rgb).reshape(height // 2, 2, width // 2, 2, image_rgb.shape[2])
    return blocks.mean(axis=(1, 3)).astype(np.float32, copy=False)


def _downsample_depth_valid_mean_2x2(
    depth: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape[:2]
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"Expected even fullres depth shape, got {(height, width)}")
    depth_blocks = np.ascontiguousarray(depth).reshape(height // 2, 2, width // 2, 2)
    valid_blocks = np.ascontiguousarray(valid_mask).reshape(height // 2, 2, width // 2, 2)
    counts = valid_blocks.sum(axis=(1, 3)).astype(np.float32, copy=False)
    sums = (depth_blocks * valid_blocks.astype(np.float32, copy=False)).sum(axis=(1, 3))
    valid_half = counts > 0.0
    depth_half = np.zeros((height // 2, width // 2), dtype=np.float32)
    depth_half[valid_half] = sums[valid_half] / counts[valid_half]
    return depth_half, valid_half


class VKITTI2HalfresRGBDepth(Dataset):
    """VKITTI2 RGB/depth control dataset with fixed even-fullres halfres geometry."""

    def __init__(
        self,
        filelist_path: str | Path | None = None,
        *,
        mode: str = "train",
        size: Tuple[int, int] = (187, 621),
        min_depth: float = 1.0,
        max_depth: float = 80.0,
        hflip_prob: float | None = None,
        include_geometry: bool = False,
        raw_storage_format: str = "not_applicable",
        fullres_even_policy: str = "crop_bottom_to_even",
        rgb_input_space: str = "halfres_2x2_area",
        depth_target_space: str = "halfres_2x2_valid_mean",
    ) -> None:
        self.mode = str(mode)
        if self.mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.size = (int(size[0]), int(size[1]))
        self.filelist_path = Path(filelist_path or DEFAULT_TRAIN_LIST).expanduser().resolve()
        self.include_geometry = bool(include_geometry)
        self.raw_storage_format = str(raw_storage_format)
        self.fullres_even_policy = str(fullres_even_policy)
        self.rgb_input_space = str(rgb_input_space)
        self.depth_target_space = str(depth_target_space)
        validate_vkitti_halfres_rgb_depth_semantics(
            raw_storage_format=self.raw_storage_format,
            fullres_even_policy=self.fullres_even_policy,
            rgb_input_space=self.rgb_input_space,
            depth_target_space=self.depth_target_space,
        )

        if not self.filelist_path.is_file():
            raise FileNotFoundError(f"Missing VKITTI2 split file: {self.filelist_path}")

        with self.filelist_path.open("r", encoding="utf-8") as f:
            self.filelist = [line.strip() for line in f if line.strip()]
        if not self.filelist:
            raise ValueError(f"No VKITTI2 samples found in {self.filelist_path}")

        self.hflip_prob = 0.5 if hflip_prob is None and self.mode == "train" else float(hflip_prob or 0.0)
        if not (0.0 <= self.hflip_prob <= 1.0):
            raise ValueError(f"hflip_prob must be in [0, 1], got {self.hflip_prob}")

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.build_sample(idx, include_geometry=self.include_geometry)

    def describe_geometry(self) -> Dict[str, Any]:
        img_path_str, _ = self.filelist[0].split()
        image = cv2.imread(str(Path(img_path_str)), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read VKITTI2 image for geometry description: {img_path_str}")
        geometry = self._build_even_fullres_geometry(image.shape[:2])
        return {
            "source_original_hw": list(geometry["source_original_hw"]),
            "original_hw": list(geometry["original_hw"]),
            "even_fullres_hw": list(geometry["even_fullres_hw"]),
            "cropped_bottom_rows": int(geometry["cropped_bottom_rows"]),
            "crop_box": list(geometry["crop_box"]),
            "crop_box_format": geometry["crop_box_format"],
            "crop_box_semantics": geometry["crop_box_semantics"],
            "packed_hw": "not_applicable",
            "raw_storage_format": self.raw_storage_format,
            "input_hw": [int(self.size[0]), int(self.size[1])],
            "rgb_input_space": self.rgb_input_space,
            "depth_target_space": self.depth_target_space,
            "fullres_even_policy": self.fullres_even_policy,
        }

    def build_sample(
        self,
        idx: int,
        *,
        py_rng: Optional[random.Random] = None,
        include_geometry: bool = False,
    ) -> Dict[str, Any]:
        img_path_str, depth_path_str = self.filelist[idx].split()
        img_path = Path(img_path_str)
        depth_path = Path(depth_path_str)

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read VKITTI2 image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        original_hw = [int(image.shape[0]), int(image.shape[1])]

        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Failed to read VKITTI2 depth: {depth_path}")
        depth = depth.astype(np.float32) / 100.0
        valid_mask = np.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)
        depth = np.where(valid_mask, depth, 0.0).astype(np.float32, copy=False)

        image, depth, valid_mask, geometry_params = self._crop_bottom_to_even_fullres(
            image,
            depth,
            valid_mask,
            original_hw=original_hw,
        )

        hflip_applied = False
        if self.mode == "train":
            image, depth, valid_mask, hflip_applied = self._random_horizontal_flip(
                image,
                depth,
                valid_mask,
                rng=py_rng,
            )
        geometry_params["hflip_applied"] = bool(hflip_applied)

        rgb_half = _downsample_rgb_2x2_area_from_even_fullres(image)
        depth_half, valid_half = _downsample_depth_valid_mean_2x2(depth, valid_mask)
        if tuple(rgb_half.shape[:2]) != self.size:
            raise ValueError(f"Halfres RGB shape mismatch for {img_path}: got={rgb_half.shape[:2]} expected={self.size}")
        if tuple(depth_half.shape) != self.size or tuple(valid_half.shape) != self.size:
            raise ValueError(
                "Halfres depth/valid shape mismatch: "
                f"depth={depth_half.shape} valid={valid_half.shape} expected={self.size}"
            )

        sample = {
            "image": _imagenet_normalize_rgb_tensor_from_array(rgb_half),
            "depth": _numpy_to_torch(depth_half.astype(np.float32, copy=False)),
            "valid_mask": _numpy_to_torch(valid_half.astype(np.uint8)).bool(),
            "rgb_preview": _rgb_preview_tensor_from_array(rgb_half),
            "image_path": str(img_path),
            "depth_path": str(depth_path),
            "sample_name": f"{img_path.parent.name}_{img_path.stem}",
            "target_space": "metric_depth",
        }
        if include_geometry:
            sample["geometry_params"] = geometry_params
        return sample

    def _build_even_fullres_geometry(self, original_hw: Tuple[int, int] | list[int]) -> Dict[str, Any]:
        original_h, original_w = int(original_hw[0]), int(original_hw[1])
        if original_w % 2 != 0:
            raise ValueError(f"VKITTI halfres control requires an even source width; got {(original_h, original_w)}")
        even_h = original_h - (original_h % 2)
        even_w = original_w
        if even_h <= 0:
            raise ValueError(f"Invalid source height for bottom crop: original_hw={(original_h, original_w)}")
        expected_even_hw = (int(self.size[0]) * 2, int(self.size[1]) * 2)
        if (even_h, even_w) != expected_even_hw:
            raise ValueError(
                "VKITTI halfres control size must match bottom-cropped even fullres: "
                f"original_hw={(original_h, original_w)} even_fullres={(even_h, even_w)} "
                f"expected_even_fullres={expected_even_hw} input_size={self.size}"
            )
        cropped_bottom_rows = original_h - even_h
        return {
            "original_hw": [original_h, original_w],
            "source_original_hw": [original_h, original_w],
            "even_fullres_hw": [even_h, even_w],
            "fullres_even_policy": self.fullres_even_policy,
            "cropped_bottom_rows": int(cropped_bottom_rows),
            "crop_box": [0, 0, even_h, even_w],
            "crop_box_format": "h_start_w_start_h_end_w_end",
            "crop_box_semantics": "source_fullres_to_even_fullres",
            "packed_hw": "not_applicable",
            "raw_storage_format": self.raw_storage_format,
            "rgb_input_space": self.rgb_input_space,
            "depth_target_space": self.depth_target_space,
            "hflip_applied": False,
        }

    def _crop_bottom_to_even_fullres(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        valid_mask: np.ndarray,
        *,
        original_hw: list[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        geometry = self._build_even_fullres_geometry(original_hw)
        crop_box = tuple(int(v) for v in geometry["crop_box"])
        return (
            _crop_hwew(image, crop_box),
            _crop_hwew(depth, crop_box),
            _crop_hwew(valid_mask, crop_box),
            geometry,
        )

    def _random_horizontal_flip(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        valid_mask: np.ndarray,
        *,
        rng: Optional[random.Random] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        rng = rng or random
        if self.hflip_prob <= 0.0 or rng.random() >= self.hflip_prob:
            return image, depth, valid_mask, False
        return (
            np.ascontiguousarray(image[:, ::-1]),
            np.ascontiguousarray(depth[:, ::-1]),
            np.ascontiguousarray(valid_mask[:, ::-1]),
            True,
        )
