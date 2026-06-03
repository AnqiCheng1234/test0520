from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from foundation.engine.transforms import (
    NOT_APPLICABLE,
    build_unprocessing_transform_from_resolved_config,
    raw_adapter_summary_from_config,
    resolve_unprocessing_config,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

LED_GEOMETRY_MODE_CHOICES = ("resize_fullres_756x1344_then_halfres_378x672",)
LED_RAW_STORAGE_FORMAT_CHOICES = ("synthetic_packed_bayer_4ch_halfres",)
LED_RGB_INPUT_SPACE_CHOICES = ("resize_area_756x1344_then_2x2_area",)
LED_DEPTH_TARGET_SPACE_CHOICES = ("resize_nearest_756x1344_then_2x2_valid_mean",)
LED_DEPTH_LABEL_CHOICES = ("distance_to_image_plane",)
LED_DEPTH_UNIT_CHOICES = ("meter",)

LED_SOURCE_HW = (1080, 1920)
LED_RESIZED_FULLRES_HW = (756, 1344)
LED_INPUT_HW = (378, 672)


def default_led_hb_raw_adapter_config() -> dict[str, Any]:
    return {
        "unprocessing_method": "raw_adapter_style",
        "vkitti_unprocessing_preset": NOT_APPLICABLE,
        "vkitti_unprocessing_mix_weights": None,
        "randomize_unprocessing": False,
        "raw_adapter_backend": "analytic",
        "raw_adapter_cfa_pattern": "RGGB",
        "raw_adapter_packed_channel_order": "R_Gr_Gb_B",
        "raw_adapter_rgb_transfer": "srgb_piecewise",
        "raw_adapter_inverse_tone": "global_0p15",
        "raw_adapter_ccm": "identity",
        "raw_adapter_red_gain_range": [1.9, 2.4],
        "raw_adapter_blue_gain_range": [1.5, 1.9],
        "raw_adapter_fixed_red_gain": 2.15,
        "raw_adapter_fixed_blue_gain": 1.70,
        "raw_adapter_fixed_light_scale": 1.0,
        "raw_adapter_dark_light_scale_range": [0.05, 0.4],
        "raw_adapter_over_light_scale_range": [1.5, 2.5],
        "raw_adapter_shot_noise": 0.001,
        "raw_adapter_read_noise": 0.0005,
        "raw_adapter_noise_mean_mode": "zero",
        "raw_adapter_black_level": 0.0,
        "raw_adapter_white_level": 1.0,
        "raw_adapter_random_seed_policy": "dataloader_generator",
        "raw_adapter_external_raw_rgb_root": NOT_APPLICABLE,
        "raw_adapter_external_key": NOT_APPLICABLE,
        "raw_adapter_external_cache_space": NOT_APPLICABLE,
        "raw_adapter_variant_policy": "normal",
        "raw_adapter_variant_weights": "normal=1.0,dark=0.0,over=0.0",
    }


def _check_equal(name: str, actual: Any, expected: Any) -> None:
    if str(actual) != str(expected):
        raise ValueError(f"{name} must be {expected!r}, got {actual!r}")


def _check_float_equal(name: str, actual: Any, expected: float, *, atol: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"{name} must be {expected!r}, got {actual!r}")


def _float_pair(value: Any, *, name: str) -> tuple[float, float]:
    if isinstance(value, str):
        parts = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError(f"{name} must contain two values, got {value!r}")
    low, high = float(parts[0]), float(parts[1])
    if not (math.isfinite(low) and math.isfinite(high) and low <= high):
        raise ValueError(f"{name} must satisfy finite low <= high, got {(low, high)}")
    return low, high


def _check_not_applicable(kwargs: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in kwargs and str(kwargs[key]) != NOT_APPLICABLE:
            raise ValueError(f"{key} must be {NOT_APPLICABLE!r}, got {kwargs[key]!r}")


def _geometry_mode(kwargs: Mapping[str, Any]) -> str:
    return str(kwargs.get("led_geometry_mode", kwargs.get("dataset_geometry_mode", "")))


def _validate_common_led_semantics(
    *,
    dataset_name: str,
    illumination: str,
    led_geometry_mode: str,
    input_height: int,
    input_width: int,
    depth_label: str,
    depth_unit: str,
    min_depth: float,
    max_depth: float,
) -> None:
    _check_equal("dataset_name", dataset_name, "led_hb")
    _check_equal("illumination", illumination, "HB")
    _check_equal("led_geometry_mode", led_geometry_mode, LED_GEOMETRY_MODE_CHOICES[0])
    if (int(input_height), int(input_width)) != LED_INPUT_HW:
        raise ValueError(f"LED-HB input size must be {LED_INPUT_HW}, got {(input_height, input_width)}")
    _check_equal("depth_label", depth_label, "distance_to_image_plane")
    _check_equal("depth_unit", depth_unit, "meter")
    _check_float_equal("min_depth", min_depth, 1.0)
    if not any(math.isclose(float(max_depth), value, rel_tol=0.0, abs_tol=1e-9) for value in (200.0, 80.0)):
        raise ValueError(f"max_depth must be 200.0 or 80.0 for LED-HB, got {max_depth!r}")


def validate_led_hb_rgb_depth_semantics(
    *,
    dataset_name: str = "led_hb",
    illumination: str = "HB",
    led_geometry_mode: str | None = None,
    dataset_geometry_mode: str | None = None,
    input_height: int = 378,
    input_width: int = 672,
    depth_label: str = "distance_to_image_plane",
    depth_unit: str = "meter",
    min_depth: float = 1.0,
    max_depth: float = 200.0,
    input_domain: str = "rgb",
    model_input_tensor: str = "image",
    raw_storage_format: str = NOT_APPLICABLE,
    unprocessing_method: str = NOT_APPLICABLE,
    **kwargs: Any,
) -> None:
    _validate_common_led_semantics(
        dataset_name=dataset_name,
        illumination=illumination,
        led_geometry_mode=str(led_geometry_mode or dataset_geometry_mode or ""),
        input_height=input_height,
        input_width=input_width,
        depth_label=depth_label,
        depth_unit=depth_unit,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    _check_equal("input_domain", input_domain, "rgb")
    _check_equal("model_input_tensor", model_input_tensor, "image")
    _check_equal("raw_storage_format", raw_storage_format, NOT_APPLICABLE)
    _check_equal("unprocessing_method", unprocessing_method, NOT_APPLICABLE)
    _check_not_applicable(
        kwargs,
        tuple(key for key in kwargs if key.startswith("raw_adapter_") or key == "vkitti_unprocessing_preset"),
    )


def validate_led_hb_raw_semantics(
    *,
    dataset_name: str = "led_hb",
    illumination: str = "HB",
    led_geometry_mode: str | None = None,
    dataset_geometry_mode: str | None = None,
    input_height: int = 378,
    input_width: int = 672,
    depth_label: str = "distance_to_image_plane",
    depth_unit: str = "meter",
    min_depth: float = 1.0,
    max_depth: float = 200.0,
    input_domain: str = "raw4",
    model_input_tensor: str = "raw",
    raw_storage_format: str = "synthetic_packed_bayer_4ch_halfres",
    unprocessing_method: str = "raw_adapter_style",
    raw_adapter_backend: str = "analytic",
    randomize_unprocessing: bool = False,
    raw_adapter_fixed_light_scale: float = 1.0,
    raw_adapter_variant_policy: str = "normal",
    raw_adapter_dark_light_scale_range: Any = (0.05, 0.4),
    raw_adapter_over_light_scale_range: Any = (1.5, 2.5),
    **_kwargs: Any,
) -> None:
    _validate_common_led_semantics(
        dataset_name=dataset_name,
        illumination=illumination,
        led_geometry_mode=str(led_geometry_mode or dataset_geometry_mode or ""),
        input_height=input_height,
        input_width=input_width,
        depth_label=depth_label,
        depth_unit=depth_unit,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    _check_equal("input_domain", input_domain, "raw4")
    _check_equal("model_input_tensor", model_input_tensor, "raw")
    _check_equal("raw_storage_format", raw_storage_format, "synthetic_packed_bayer_4ch_halfres")
    _check_equal("unprocessing_method", unprocessing_method, "raw_adapter_style")
    _check_equal("raw_adapter_backend", raw_adapter_backend, "analytic")
    if bool(randomize_unprocessing):
        raise ValueError("randomize_unprocessing must be false for LED-HB RA0 RAW")
    variant = str(raw_adapter_variant_policy)
    fixed_light_scale = float(raw_adapter_fixed_light_scale)
    if variant == "normal":
        _check_float_equal("raw_adapter_fixed_light_scale", fixed_light_scale, 1.0)
    elif variant == "dark":
        low, high = _float_pair(raw_adapter_dark_light_scale_range, name="raw_adapter_dark_light_scale_range")
        if not (low <= fixed_light_scale <= high):
            raise ValueError(
                "raw_adapter_fixed_light_scale must fall inside raw_adapter_dark_light_scale_range "
                f"when raw_adapter_variant_policy='dark', got {fixed_light_scale} not in {(low, high)}"
            )
    elif variant == "over":
        low, high = _float_pair(raw_adapter_over_light_scale_range, name="raw_adapter_over_light_scale_range")
        if not (low <= fixed_light_scale <= high):
            raise ValueError(
                "raw_adapter_fixed_light_scale must fall inside raw_adapter_over_light_scale_range "
                f"when raw_adapter_variant_policy='over', got {fixed_light_scale} not in {(low, high)}"
            )
    else:
        raise ValueError("raw_adapter_variant_policy must be one of 'normal', 'dark', or 'over' for LED-HB RA0 RAW")


def _numpy_to_torch(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array))


def _rgb_preview_tensor_from_array(image_rgb: np.ndarray) -> torch.Tensor:
    image_rgb = np.clip(image_rgb, 0.0, 1.0).astype(np.float32, copy=False)
    return _numpy_to_torch(np.transpose(image_rgb, (2, 0, 1))).float()


def _imagenet_normalize_rgb_tensor_from_array(image_rgb: np.ndarray) -> torch.Tensor:
    image_rgb = np.clip(image_rgb, 0.0, 1.0)
    normalized = (image_rgb.astype(np.float32, copy=False) - IMAGENET_MEAN) / IMAGENET_STD
    return _numpy_to_torch(np.transpose(normalized, (2, 0, 1))).float()


def _downsample_rgb_2x2_area_from_even_fullres(image_rgb: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"Expected even fullres RGB shape, got {(height, width)}")
    blocks = np.ascontiguousarray(image_rgb).reshape(height // 2, 2, width // 2, 2, image_rgb.shape[2])
    return blocks.mean(axis=(1, 3)).astype(np.float32, copy=False)


def _downsample_depth_valid_mean_2x2(depth: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def _read_filelist(path: Path) -> list[tuple[Path, Path, Path, Path]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing LED-HB filelist: {path}")
    rows: list[tuple[Path, Path, Path, Path]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"{path}:{line_no}: expected 4 columns, got {len(parts)}")
            rows.append(tuple(Path(item).expanduser().resolve() for item in parts))  # type: ignore[arg-type]
    if not rows:
        raise ValueError(f"No LED-HB samples found in {path}")
    return rows


def _sample_name(img_path: Path) -> str:
    map_name = img_path.parent.parent.name
    split_name = img_path.parent.parent.parent.name
    return f"led_hb_{split_name}_{map_name}_{img_path.stem}"


class LEDHBHalfresRGBDepth(Dataset):
    """LED-HB RGB/depth dataset with fixed aspect-preserving 756x1344 -> 378x672 geometry."""

    def __init__(
        self,
        filelist_path: str | Path,
        *,
        mode: str = "train",
        size: Tuple[int, int] = LED_INPUT_HW,
        min_depth: float = 1.0,
        max_depth: float = 200.0,
        hflip_prob: float | None = None,
        include_geometry: bool = False,
        dataset_name: str = "led_hb",
        illumination: str = "HB",
        led_geometry_mode: str = LED_GEOMETRY_MODE_CHOICES[0],
        raw_storage_format: str = NOT_APPLICABLE,
        rgb_input_space: str = LED_RGB_INPUT_SPACE_CHOICES[0],
        depth_target_space: str = LED_DEPTH_TARGET_SPACE_CHOICES[0],
        depth_label: str = LED_DEPTH_LABEL_CHOICES[0],
        depth_unit: str = LED_DEPTH_UNIT_CHOICES[0],
        _skip_rgb_semantic_validation: bool = False,
    ) -> None:
        self.mode = str(mode)
        if self.mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")
        self.size = (int(size[0]), int(size[1]))
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.filelist_path = Path(filelist_path).expanduser().resolve()
        self.include_geometry = bool(include_geometry)
        self.dataset_name = str(dataset_name)
        self.illumination = str(illumination)
        self.led_geometry_mode = str(led_geometry_mode)
        self.raw_storage_format = str(raw_storage_format)
        self.rgb_input_space = str(rgb_input_space)
        self.depth_target_space = str(depth_target_space)
        self.depth_label = str(depth_label)
        self.depth_unit = str(depth_unit)
        if _skip_rgb_semantic_validation:
            _validate_common_led_semantics(
                dataset_name=self.dataset_name,
                illumination=self.illumination,
                led_geometry_mode=self.led_geometry_mode,
                input_height=self.size[0],
                input_width=self.size[1],
                depth_label=self.depth_label,
                depth_unit=self.depth_unit,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
            )
        else:
            validate_led_hb_rgb_depth_semantics(
                dataset_name=self.dataset_name,
                illumination=self.illumination,
                led_geometry_mode=self.led_geometry_mode,
                input_height=self.size[0],
                input_width=self.size[1],
                depth_label=self.depth_label,
                depth_unit=self.depth_unit,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                raw_storage_format=self.raw_storage_format,
            )
        if self.rgb_input_space != LED_RGB_INPUT_SPACE_CHOICES[0]:
            raise ValueError(f"Unsupported rgb_input_space={self.rgb_input_space!r}")
        if self.depth_target_space != LED_DEPTH_TARGET_SPACE_CHOICES[0]:
            raise ValueError(f"Unsupported depth_target_space={self.depth_target_space!r}")
        self.filelist = _read_filelist(self.filelist_path)
        self.hflip_prob = 0.5 if hflip_prob is None and self.mode == "train" else float(hflip_prob or 0.0)
        if not (0.0 <= self.hflip_prob <= 1.0):
            raise ValueError(f"hflip_prob must be in [0,1], got {self.hflip_prob}")

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.build_sample(idx, include_geometry=self.include_geometry)

    def describe_geometry(self) -> dict[str, Any]:
        first_image = cv2.imread(str(self.filelist[0][0]), cv2.IMREAD_COLOR)
        if first_image is None:
            raise ValueError(f"Failed to read LED-HB image for geometry: {self.filelist[0][0]}")
        source_h, source_w = int(first_image.shape[0]), int(first_image.shape[1])
        scale_h = float(LED_RESIZED_FULLRES_HW[0]) / float(source_h)
        scale_w = float(LED_RESIZED_FULLRES_HW[1]) / float(source_w)
        if abs(scale_h - scale_w) >= 1e-6:
            raise ValueError(f"LED-HB resize must be isotropic, got scale_h={scale_h} scale_w={scale_w}")
        return {
            "dataset_name": self.dataset_name,
            "illumination": self.illumination,
            "source_original_hw": [source_h, source_w],
            "resized_fullres_hw": list(LED_RESIZED_FULLRES_HW),
            "input_hw": [int(self.size[0]), int(self.size[1])],
            "resize_scale_h": scale_h,
            "resize_scale_w": scale_w,
            "isotropic_resize": True,
            "raw_storage_format": self.raw_storage_format,
            "led_geometry_mode": self.led_geometry_mode,
            "rgb_input_space": self.rgb_input_space,
            "depth_target_space": self.depth_target_space,
            "depth_label": self.depth_label,
            "depth_unit": self.depth_unit,
        }

    def _read_rgb_depth(self, idx: int) -> tuple[Path, Path, Path, Path, np.ndarray, np.ndarray, dict[str, Any]]:
        img_path, depth_path, camera_path, transforms_path = self.filelist[idx]
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Failed to read LED-HB RGB: {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Failed to read LED-HB EXR depth: {depth_path}")
        depth_channels_used = "single"
        if depth.ndim == 3:
            depth = depth[..., 0]
            depth_channels_used = "first_of_multi_channel"
        depth = depth.astype(np.float32, copy=False)
        metadata = {
            "depth_channels_used": depth_channels_used,
            "source_original_hw": [int(image.shape[0]), int(image.shape[1])],
        }
        return img_path, depth_path, camera_path, transforms_path, image, depth, metadata

    def _resize_to_led_fullres(
        self,
        image: np.ndarray,
        depth: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        source_h, source_w = image.shape[:2]
        if (source_h, source_w) != LED_SOURCE_HW:
            raise ValueError(f"LED-HB source shape must be {LED_SOURCE_HW}, got {(source_h, source_w)}")
        scale_h = float(LED_RESIZED_FULLRES_HW[0]) / float(source_h)
        scale_w = float(LED_RESIZED_FULLRES_HW[1]) / float(source_w)
        if abs(scale_h - scale_w) >= 1e-6:
            raise ValueError(f"LED-HB resize must be isotropic, got scale_h={scale_h} scale_w={scale_w}")
        valid = np.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)
        depth_clean = np.where(valid, depth, 0.0).astype(np.float32, copy=False)
        image = cv2.resize(image, (LED_RESIZED_FULLRES_HW[1], LED_RESIZED_FULLRES_HW[0]), interpolation=cv2.INTER_AREA)
        depth_resized = cv2.resize(depth_clean, (LED_RESIZED_FULLRES_HW[1], LED_RESIZED_FULLRES_HW[0]), interpolation=cv2.INTER_NEAREST)
        valid_resized = (
            cv2.resize(valid.astype(np.uint8), (LED_RESIZED_FULLRES_HW[1], LED_RESIZED_FULLRES_HW[0]), interpolation=cv2.INTER_NEAREST)
            > 0
        )
        geometry = {
            "source_original_hw": [source_h, source_w],
            "resized_fullres_hw": list(LED_RESIZED_FULLRES_HW),
            "input_hw": list(LED_INPUT_HW),
            "resize_scale_h": scale_h,
            "resize_scale_w": scale_w,
            "isotropic_resize": True,
            "hflip_applied": False,
        }
        return image.astype(np.float32, copy=False), depth_resized.astype(np.float32, copy=False), valid_resized, geometry

    def _maybe_flip(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        valid_mask: np.ndarray,
        *,
        rng: Optional[random.Random],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        rng = rng or random
        if self.mode != "train" or self.hflip_prob <= 0.0 or rng.random() >= self.hflip_prob:
            return image, depth, valid_mask, False
        return (
            np.ascontiguousarray(image[:, ::-1]),
            np.ascontiguousarray(depth[:, ::-1]),
            np.ascontiguousarray(valid_mask[:, ::-1]),
            True,
        )

    def build_sample(
        self,
        idx: int,
        *,
        py_rng: Optional[random.Random] = None,
        include_geometry: bool = False,
    ) -> dict[str, Any]:
        img_path, depth_path, camera_path, transforms_path, image, depth, metadata = self._read_rgb_depth(idx)
        image, depth, valid_mask, geometry = self._resize_to_led_fullres(image, depth)
        image, depth, valid_mask, hflip_applied = self._maybe_flip(image, depth, valid_mask, rng=py_rng)
        geometry.update(metadata)
        geometry["hflip_applied"] = bool(hflip_applied)

        rgb_half = _downsample_rgb_2x2_area_from_even_fullres(image)
        depth_half, valid_half = _downsample_depth_valid_mean_2x2(depth, valid_mask)
        if tuple(rgb_half.shape[:2]) != self.size:
            raise ValueError(f"LED-HB RGB shape mismatch: got={rgb_half.shape[:2]} expected={self.size}")
        if tuple(depth_half.shape) != self.size or tuple(valid_half.shape) != self.size:
            raise ValueError(f"LED-HB depth shape mismatch: depth={depth_half.shape} valid={valid_half.shape} expected={self.size}")

        sample: dict[str, Any] = {
            "image": _imagenet_normalize_rgb_tensor_from_array(rgb_half),
            "rgb_preview": _rgb_preview_tensor_from_array(rgb_half),
            "depth": _numpy_to_torch(depth_half.astype(np.float32, copy=False)).float(),
            "valid_mask": _numpy_to_torch(valid_half.astype(np.uint8)).bool(),
            "image_path": str(img_path),
            "depth_path": str(depth_path),
            "camera_params_path": str(camera_path),
            "transforms_path": str(transforms_path),
            "sample_name": _sample_name(img_path),
            "target_space": "metric_depth",
            "dataset_index": int(idx),
            "geometry_params": geometry,
        }
        if not include_geometry:
            sample["geometry_params"] = geometry
        return sample


class LEDHBRaw(LEDHBHalfresRGBDepth):
    """LED-HB synthetic packed Bayer RAW-like dataset built from resized fullres RGB."""

    def __init__(
        self,
        filelist_path: str | Path,
        *,
        mode: str = "train",
        size: Tuple[int, int] = LED_INPUT_HW,
        min_depth: float = 1.0,
        max_depth: float = 200.0,
        randomize_unprocessing: bool = False,
        unprocessing_method: str = "raw_adapter_style",
        unprocessing_config: Mapping[str, Any] | None = None,
        hflip_prob: float | None = None,
        include_rgb_input: bool = True,
        include_rgb_preview: bool = False,
        include_geometry: bool = False,
        dataset_name: str = "led_hb",
        illumination: str = "HB",
        led_geometry_mode: str = LED_GEOMETRY_MODE_CHOICES[0],
        raw_storage_format: str = "synthetic_packed_bayer_4ch_halfres",
        rgb_input_space: str = LED_RGB_INPUT_SPACE_CHOICES[0],
        depth_target_space: str = LED_DEPTH_TARGET_SPACE_CHOICES[0],
        depth_label: str = LED_DEPTH_LABEL_CHOICES[0],
        depth_unit: str = LED_DEPTH_UNIT_CHOICES[0],
    ) -> None:
        super().__init__(
            filelist_path=filelist_path,
            mode=mode,
            size=size,
            min_depth=min_depth,
            max_depth=max_depth,
            hflip_prob=hflip_prob,
            include_geometry=include_geometry,
            dataset_name=dataset_name,
            illumination=illumination,
            led_geometry_mode=led_geometry_mode,
            raw_storage_format=raw_storage_format,
            rgb_input_space=rgb_input_space,
            depth_target_space=depth_target_space,
            depth_label=depth_label,
            depth_unit=depth_unit,
            _skip_rgb_semantic_validation=True,
        )
        self.include_rgb_input = bool(include_rgb_input)
        self.include_rgb_preview = bool(include_rgb_preview)
        if unprocessing_config is None:
            source = {"unprocessing_method": unprocessing_method, "randomize_unprocessing": bool(randomize_unprocessing)}
        else:
            source = dict(unprocessing_config)
        self.resolved_unprocessing_config = resolve_unprocessing_config(source)
        validate_led_hb_raw_semantics(
            dataset_name=self.dataset_name,
            illumination=self.illumination,
            led_geometry_mode=self.led_geometry_mode,
            input_height=self.size[0],
            input_width=self.size[1],
            depth_label=self.depth_label,
            depth_unit=self.depth_unit,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            input_domain="raw4",
            model_input_tensor="raw",
            raw_storage_format=self.raw_storage_format,
            unprocessing_method=self.resolved_unprocessing_config["unprocessing_method"],
            raw_adapter_backend=self.resolved_unprocessing_config["raw_adapter_backend"],
            randomize_unprocessing=self.resolved_unprocessing_config["randomize_unprocessing"],
            raw_adapter_fixed_light_scale=self.resolved_unprocessing_config["raw_adapter_fixed_light_scale"],
            raw_adapter_variant_policy=self.resolved_unprocessing_config["raw_adapter_variant_policy"],
            raw_adapter_dark_light_scale_range=self.resolved_unprocessing_config["raw_adapter_dark_light_scale_range"],
            raw_adapter_over_light_scale_range=self.resolved_unprocessing_config["raw_adapter_over_light_scale_range"],
        )
        split = "train" if self.mode == "train" else "led_hb_val"
        self.unprocessing, self.raw_adapter_unprocessing_summary = build_unprocessing_transform_from_resolved_config(
            self.resolved_unprocessing_config,
            split=split,
        )

    def describe_geometry(self) -> dict[str, Any]:
        payload = super().describe_geometry()
        payload["raw_storage_format"] = self.raw_storage_format
        payload["packed_hw"] = [int(self.size[0]), int(self.size[1])]
        return payload

    def describe_unprocessing(self) -> dict[str, Any]:
        return dict(raw_adapter_summary_from_config(self.resolved_unprocessing_config))

    def build_sample(
        self,
        idx: int,
        *,
        py_rng: Optional[random.Random] = None,
        torch_generator: Optional[torch.Generator] = None,
        include_geometry: bool = False,
        include_rgb_input: Optional[bool] = None,
        include_rgb_preview: Optional[bool] = None,
    ) -> dict[str, Any]:
        include_rgb_input = self.include_rgb_input if include_rgb_input is None else bool(include_rgb_input)
        include_rgb_preview = self.include_rgb_preview if include_rgb_preview is None else bool(include_rgb_preview)
        img_path, depth_path, camera_path, transforms_path, image, depth, metadata = self._read_rgb_depth(idx)
        image, depth, valid_mask, geometry = self._resize_to_led_fullres(image, depth)
        image, depth, valid_mask, hflip_applied = self._maybe_flip(image, depth, valid_mask, rng=py_rng)
        geometry.update(metadata)
        geometry["hflip_applied"] = bool(hflip_applied)

        image_tensor = _numpy_to_torch(np.transpose(np.ascontiguousarray(image), (2, 0, 1)).astype(np.float32, copy=False)).float()
        raw_tensor, isp_params = self.unprocessing(image_tensor, generator=torch_generator)
        raw_tensor = raw_tensor.float()
        if tuple(raw_tensor.shape) != (4, int(self.size[0]), int(self.size[1])):
            raise ValueError(f"LED-HB RAW shape mismatch: got={tuple(raw_tensor.shape)} expected={(4, *self.size)}")
        isp_params = dict(isp_params)
        isp_params["hflip_applied"] = bool(hflip_applied)

        rgb_half = _downsample_rgb_2x2_area_from_even_fullres(image)
        depth_half, valid_half = _downsample_depth_valid_mean_2x2(depth, valid_mask)
        sample: dict[str, Any] = {
            "raw": raw_tensor,
            "depth": _numpy_to_torch(depth_half.astype(np.float32, copy=False)).float(),
            "valid_mask": _numpy_to_torch(valid_half.astype(np.uint8)).bool(),
            "isp_params": isp_params,
            "image_path": str(img_path),
            "depth_path": str(depth_path),
            "camera_params_path": str(camera_path),
            "transforms_path": str(transforms_path),
            "sample_name": _sample_name(img_path),
            "target_space": "metric_depth",
            "dataset_index": int(idx),
            "geometry_params": geometry,
        }
        if include_rgb_input:
            sample["image"] = _imagenet_normalize_rgb_tensor_from_array(rgb_half)
        if include_rgb_preview:
            sample["rgb_preview"] = _rgb_preview_tensor_from_array(rgb_half)
        if not include_geometry:
            sample["geometry_params"] = geometry
        return sample

    def build_rgb_baseline_input(
        self,
        idx: int,
        geometry: Mapping[str, Any],
        *,
        target_hw: Tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del geometry
        img_path, _depth_path, _camera_path, _transforms_path, image, depth, _metadata = self._read_rgb_depth(idx)
        del img_path, depth
        image, _depth, _valid, _geometry = self._resize_to_led_fullres(image, np.ones(LED_SOURCE_HW, dtype=np.float32))
        rgb_half = _downsample_rgb_2x2_area_from_even_fullres(image)
        if tuple(rgb_half.shape[:2]) != tuple(int(v) for v in target_hw):
            raise ValueError(f"LED-HB RGB baseline shape mismatch: got={rgb_half.shape[:2]} target={target_hw}")
        return _imagenet_normalize_rgb_tensor_from_array(rgb_half), _rgb_preview_tensor_from_array(rgb_half)
