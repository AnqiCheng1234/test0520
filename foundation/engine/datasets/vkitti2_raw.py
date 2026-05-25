from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from foundation.engine.transforms import (
    NOT_APPLICABLE,
    build_unprocessing_transform_from_preset,
    build_unprocessing_transform_from_resolved_config,
    get_unprocessing_preset,
    raw_adapter_summary_from_config,
    resolve_unprocessing_mix_weights,
    resolve_unprocessing_config,
)


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
RAW_STORAGE_FORMAT_CHOICES = (
    "synthetic_packed_bayer_4ch",
    "synthetic_packed_bayer_4ch_halfres",
)
FULLRES_EVEN_POLICY_CHOICES = ("not_applicable", "crop_bottom_to_even")
RGB_INPUT_SPACE_CHOICES = ("not_applicable", "halfres_2x2_area")
DEPTH_TARGET_SPACE_CHOICES = ("not_applicable", "halfres_2x2_valid_mean")


def validate_vkitti_raw_semantics(
    *,
    raw_storage_format: str,
    fullres_even_policy: str,
    rgb_input_space: str,
    depth_target_space: str,
) -> None:
    if raw_storage_format not in RAW_STORAGE_FORMAT_CHOICES:
        raise ValueError(f"Unsupported raw_storage_format: {raw_storage_format!r}")
    if fullres_even_policy not in FULLRES_EVEN_POLICY_CHOICES:
        raise ValueError(f"Unsupported fullres_even_policy: {fullres_even_policy!r}")
    if rgb_input_space not in RGB_INPUT_SPACE_CHOICES:
        raise ValueError(f"Unsupported rgb_input_space: {rgb_input_space!r}")
    if depth_target_space not in DEPTH_TARGET_SPACE_CHOICES:
        raise ValueError(f"Unsupported depth_target_space: {depth_target_space!r}")

    expected_by_format = {
        "synthetic_packed_bayer_4ch": {
            "fullres_even_policy": "not_applicable",
            "rgb_input_space": "not_applicable",
            "depth_target_space": "not_applicable",
        },
        "synthetic_packed_bayer_4ch_halfres": {
            "fullres_even_policy": "crop_bottom_to_even",
            "rgb_input_space": "halfres_2x2_area",
            "depth_target_space": "halfres_2x2_valid_mean",
        },
    }
    expected = expected_by_format[raw_storage_format]
    actual = {
        "fullres_even_policy": fullres_even_policy,
        "rgb_input_space": rgb_input_space,
        "depth_target_space": depth_target_space,
    }
    for key, expected_value in expected.items():
        if actual[key] != expected_value:
            raise ValueError(
                f"{key} must be {expected_value!r} when "
                f"raw_storage_format={raw_storage_format!r}, got {actual[key]!r}"
            )


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
    tensor = torch.frombuffer(bytearray(array.tobytes()), dtype=torch_dtype).view(*array.shape)
    return tensor


def _resize_rgb_short_edge(image: np.ndarray, *, short_edge: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = float(short_edge) / float(min(height, width))
    resized_h = int(round(height * scale))
    resized_w = int(round(width * scale))
    return cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)


def _crop_hwew(array: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
    h_start, w_start, h_end, w_end = [int(v) for v in crop_box]
    return np.ascontiguousarray(array[h_start:h_end, w_start:w_end, ...])


def _rgb_preview_tensor(image_rgb: np.ndarray, target_hw: Tuple[int, int]) -> torch.Tensor:
    resized = cv2.resize(image_rgb, (int(target_hw[1]), int(target_hw[0])), interpolation=cv2.INTER_AREA)
    resized = np.clip(resized, 0.0, 1.0)
    return _rgb_preview_tensor_from_array(resized)


def _imagenet_normalize_rgb_tensor(image_rgb: np.ndarray, target_hw: Tuple[int, int]) -> torch.Tensor:
    resized = cv2.resize(image_rgb, (int(target_hw[1]), int(target_hw[0])), interpolation=cv2.INTER_AREA)
    resized = np.clip(resized, 0.0, 1.0)
    return _imagenet_normalize_rgb_tensor_from_array(resized)


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


class VKITTI2Raw(Dataset):
    """VKITTI2 dataset wrapper that generates pseudo-RAW packed Bayer on the fly."""

    def __init__(
        self,
        filelist_path: str | Path | None = None,
        *,
        mode: str = "train",
        size: Tuple[int, int] = (512, 960),
        min_depth: float = 1.0,
        max_depth: float = 80.0,
        randomize_unprocessing: bool = True,
        unprocessing_method: str = "old_brooks_preset",
        unprocessing_preset: str = "sensor_linear_dual",
        unprocessing_mix_weights: object | None = None,
        unprocessing_config: Mapping[str, Any] | None = None,
        hflip_prob: float | None = None,
        include_rgb_input: bool = False,
        include_rgb_preview: bool = False,
        include_geometry: bool = False,
        raw_storage_format: str = "synthetic_packed_bayer_4ch",
        fullres_even_policy: str = "not_applicable",
        rgb_input_space: str = "not_applicable",
        depth_target_space: str = "not_applicable",
    ) -> None:
        self.mode = mode
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.size = tuple(size)
        self.fullres_size = (self.size[0] * 2, self.size[1] * 2)
        self.filelist_path = Path(filelist_path or DEFAULT_TRAIN_LIST).expanduser().resolve()
        self.include_rgb_input = bool(include_rgb_input)
        self.include_rgb_preview = bool(include_rgb_preview)
        self.include_geometry = bool(include_geometry)
        self.raw_storage_format = str(raw_storage_format)
        self.fullres_even_policy = str(fullres_even_policy)
        self.rgb_input_space = str(rgb_input_space)
        self.depth_target_space = str(depth_target_space)
        validate_vkitti_raw_semantics(
            raw_storage_format=self.raw_storage_format,
            fullres_even_policy=self.fullres_even_policy,
            rgb_input_space=self.rgb_input_space,
            depth_target_space=self.depth_target_space,
        )

        if not self.filelist_path.is_file():
            raise FileNotFoundError(
                "Missing VKITTI2 split file. Generate it with scripts/generate_vkitti2_split.py "
                f"or pass filelist_path explicitly. Expected: {self.filelist_path}"
            )

        with self.filelist_path.open("r", encoding="utf-8") as f:
            self.filelist = [line.strip() for line in f if line.strip()]

        if not self.filelist:
            raise ValueError(f"No VKITTI2 samples found in {self.filelist_path}")

        if unprocessing_config is None:
            unprocessing_source: Dict[str, Any] = {
                "unprocessing_method": unprocessing_method,
                "randomize_unprocessing": bool(randomize_unprocessing),
                "vkitti_unprocessing_preset": unprocessing_preset,
                "vkitti_unprocessing_mix_weights": unprocessing_mix_weights,
            }
        else:
            unprocessing_source = dict(unprocessing_config)
        self.resolved_unprocessing_config = resolve_unprocessing_config(unprocessing_source)
        self.unprocessing_method = str(self.resolved_unprocessing_config["unprocessing_method"])
        self.randomize_unprocessing = bool(self.resolved_unprocessing_config["randomize_unprocessing"])

        if self.unprocessing_method == "old_brooks_preset":
            self.unprocessing_preset = str(self.resolved_unprocessing_config["vkitti_unprocessing_preset"])
            self.unprocessing_preset_spec = get_unprocessing_preset(self.unprocessing_preset)
            self.unprocessing_preset_version = str(self.unprocessing_preset_spec["preset_version"])
            self.unprocessing_preset_hash = str(self.unprocessing_preset_spec["preset_hash"])
            self.unprocessing_profile_group = str(self.unprocessing_preset_spec["isp_profile_group"])

            kind = str(self.unprocessing_preset_spec["kind"])
            mix_weights = self.resolved_unprocessing_config.get("vkitti_unprocessing_mix_weights")
            if kind == "single":
                self.unprocessing_mix_weights = resolve_unprocessing_mix_weights(
                    self.unprocessing_preset,
                    mix_weights,
                )
                self.unprocessing_sub_presets = tuple(self.unprocessing_mix_weights.keys())
                self.unprocessing_default_sub_preset = self.unprocessing_preset
            elif kind == "dual":
                self.unprocessing_mix_weights = resolve_unprocessing_mix_weights(
                    self.unprocessing_preset,
                    mix_weights,
                )
                self.unprocessing_sub_presets = tuple(self.unprocessing_mix_weights.keys())
                self.unprocessing_default_sub_preset = str(self.unprocessing_preset_spec["default_sub_preset"])
                if self.unprocessing_default_sub_preset not in self.unprocessing_sub_presets:
                    raise ValueError(
                        f"default_sub_preset '{self.unprocessing_default_sub_preset}' is not in "
                        f"sub presets {self.unprocessing_sub_presets}"
                    )
            else:
                raise ValueError(f"Unsupported unprocessing preset kind: {kind}")

            self._unprocessing_transforms = {
                name: build_unprocessing_transform_from_preset(
                    name,
                    randomize=self.randomize_unprocessing,
                )
                for name in self.unprocessing_sub_presets
            }
            self._sub_preset_hashes = {
                name: str(get_unprocessing_preset(name)["preset_hash"])
                for name in self.unprocessing_sub_presets
            }
            self.unprocessing = self._unprocessing_transforms[self.unprocessing_default_sub_preset]
            self.raw_adapter_unprocessing_summary = raw_adapter_summary_from_config(self.resolved_unprocessing_config)
        elif self.unprocessing_method == "raw_adapter_style":
            self.unprocessing_preset = NOT_APPLICABLE
            self.unprocessing_preset_spec = {"kind": NOT_APPLICABLE}
            self.unprocessing_preset_version = NOT_APPLICABLE
            self.unprocessing_preset_hash = str(self.resolved_unprocessing_config["raw_adapter_config_hash"])
            self.unprocessing_profile_group = "raw_adapter_style"
            self.unprocessing_mix_weights = {"raw_adapter_style": 1.0}
            self.unprocessing_sub_presets = ("raw_adapter_style",)
            self.unprocessing_default_sub_preset = "raw_adapter_style"
            self._sub_preset_hashes = {"raw_adapter_style": self.unprocessing_preset_hash}
            split = "train" if self.mode == "train" else "vkitti_val"
            transform, summary = build_unprocessing_transform_from_resolved_config(
                self.resolved_unprocessing_config,
                split=split,
            )
            self._unprocessing_transforms = {"raw_adapter_style": transform}
            self.unprocessing = transform
            self.raw_adapter_unprocessing_summary = summary
        else:
            raise ValueError(f"Unsupported unprocessing_method: {self.unprocessing_method!r}")

        self.hflip_prob = 0.5 if hflip_prob is None and self.mode == "train" else float(hflip_prob or 0.0)

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.build_sample(idx, include_geometry=self.include_geometry)

    def _summarize_unprocessing_transform(self, transform: Any) -> Dict[str, Any]:
        xyz_to_cam_override = getattr(transform, "xyz_to_cam_override", None)
        if xyz_to_cam_override is not None:
            xyz_to_cam_override = [
                [float(value) for value in row]
                for row in xyz_to_cam_override.detach().cpu().tolist()
            ]
        return {
            "randomize": bool(transform.randomize),
            "randomize_ccm": bool(getattr(transform, "randomize_ccm", True)),
            "xyz_to_cam_override": xyz_to_cam_override,
            "red_gain_range": list(transform.red_gain_range),
            "blue_gain_range": list(transform.blue_gain_range),
            "black_level_range": list(transform.black_level_range),
            "shot_log_gain_range": list(transform.shot_log_gain_range),
            "read_noise_std_range": list(transform.read_noise_std_range),
            "exposure_gain_range": list(transform.exposure_gain_range),
            "cfa_patterns": list(transform.cfa_patterns),
            "eps": float(transform.eps),
            "canonical_params": dict(transform.canonical_params.__dict__),
        }

    def describe_unprocessing(self) -> Dict[str, Any]:
        if self.unprocessing_method == "raw_adapter_style":
            return dict(self.raw_adapter_unprocessing_summary)

        active_unprocessing = self.unprocessing
        payload = {
            "unprocessing_method": "old_brooks_preset",
            "unprocessing_preset": self.unprocessing_preset,
            "unprocessing_kind": str(self.unprocessing_preset_spec["kind"]),
            "preset_version": self.unprocessing_preset_version,
            "preset_hash": self.unprocessing_preset_hash,
            "isp_profile_group": self.unprocessing_profile_group,
            "sub_presets": list(self.unprocessing_sub_presets),
            "default_sub_preset": self.unprocessing_default_sub_preset,
            "mix_weights": {
                name: float(self.unprocessing_mix_weights[name])
                for name in self.unprocessing_sub_presets
            },
            "active_transform": self._summarize_unprocessing_transform(active_unprocessing),
            "sub_preset_transforms": {
                name: self._summarize_unprocessing_transform(self._unprocessing_transforms[name])
                for name in self.unprocessing_sub_presets
            },
        }
        payload.update(self.raw_adapter_unprocessing_summary)
        return payload

    def describe_geometry(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "raw_storage_format": self.raw_storage_format,
            "fullres_even_policy": self.fullres_even_policy,
            "rgb_input_space": self.rgb_input_space,
            "depth_target_space": self.depth_target_space,
            "input_hw": [int(self.size[0]), int(self.size[1])],
            "packed_hw": [int(self.size[0]), int(self.size[1])],
        }
        if self.raw_storage_format != "synthetic_packed_bayer_4ch_halfres":
            payload.update(
                {
                    "source_original_hw": "n/a",
                    "original_hw": "n/a",
                    "even_fullres_hw": "n/a",
                    "cropped_bottom_rows": "n/a",
                }
            )
            return payload

        img_path_str, _ = self.filelist[0].split()
        image = cv2.imread(str(Path(img_path_str)), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read VKITTI2 image for geometry description: {img_path_str}")
        geometry = self._build_even_fullres_geometry(image.shape[:2])
        payload.update(
            {
                "source_original_hw": list(geometry["original_hw"]),
                "original_hw": list(geometry["original_hw"]),
                "even_fullres_hw": list(geometry["even_fullres_hw"]),
                "cropped_bottom_rows": int(geometry["cropped_bottom_rows"]),
                "crop_box": list(geometry["crop_box"]),
                "crop_box_format": geometry["crop_box_format"],
                "crop_box_semantics": geometry["crop_box_semantics"],
            }
        )
        return payload

    def build_sample(
        self,
        idx: int,
        *,
        py_rng: Optional[random.Random] = None,
        torch_generator: Optional[torch.Generator] = None,
        include_geometry: bool = False,
        include_rgb_input: Optional[bool] = None,
        include_rgb_preview: Optional[bool] = None,
    ) -> Dict[str, Any]:
        include_rgb_input = self.include_rgb_input if include_rgb_input is None else bool(include_rgb_input)
        include_rgb_preview = self.include_rgb_preview if include_rgb_preview is None else bool(include_rgb_preview)

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

        if self.raw_storage_format == "synthetic_packed_bayer_4ch_halfres":
            return self._build_halfres_sample(
                idx=idx,
                img_path=img_path,
                depth_path=depth_path,
                image=image,
                depth=depth,
                valid_mask=valid_mask,
                original_hw=original_hw,
                py_rng=py_rng,
                torch_generator=torch_generator,
                include_geometry=include_geometry,
                include_rgb_input=include_rgb_input,
                include_rgb_preview=include_rgb_preview,
            )

        image, depth, valid_mask = self._resize_short_edge(
            image,
            depth,
            valid_mask,
            short_edge=self.fullres_size[0],
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

        image_tensor = _numpy_to_torch(np.transpose(np.ascontiguousarray(image), (2, 0, 1)).astype(np.float32, copy=False))
        selected_sub_preset_name = self._select_unprocessing_sub_preset(py_rng=py_rng, torch_generator=torch_generator)
        unprocessing_transform = self._unprocessing_transforms[selected_sub_preset_name]
        raw_tensor, isp_params = unprocessing_transform(image_tensor, generator=torch_generator)
        isp_params = self._augment_isp_params(
            isp_params,
            selected_sub_preset_name=selected_sub_preset_name,
        )
        isp_params["hflip_applied"] = bool(geometry_params["hflip_applied"])
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
        if include_rgb_input:
            sample["image"] = _imagenet_normalize_rgb_tensor(image, self.size)
        if include_rgb_preview:
            sample["rgb_preview"] = _rgb_preview_tensor(image, self.size)
        return sample

    def _build_halfres_sample(
        self,
        *,
        idx: int,
        img_path: Path,
        depth_path: Path,
        image: np.ndarray,
        depth: np.ndarray,
        valid_mask: np.ndarray,
        original_hw: list[int],
        py_rng: Optional[random.Random],
        torch_generator: Optional[torch.Generator],
        include_geometry: bool,
        include_rgb_input: bool,
        include_rgb_preview: bool,
    ) -> Dict[str, Any]:
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

        even_h, even_w = image.shape[:2]
        if even_h % 2 != 0 or even_w % 2 != 0:
            raise ValueError(f"Expected even fullres shape before RAW packing, got {(even_h, even_w)}")
        expected_even_hw = (int(self.size[0]) * 2, int(self.size[1]) * 2)
        if (even_h, even_w) != expected_even_hw:
            raise ValueError(
                "VKITTI halfres geometry mismatch: "
                f"even_fullres={(even_h, even_w)} expected={expected_even_hw} "
                f"from input size={self.size} image={img_path}"
            )

        image_tensor = _numpy_to_torch(
            np.transpose(np.ascontiguousarray(image), (2, 0, 1)).astype(np.float32, copy=False)
        )
        selected_sub_preset_name = self._select_unprocessing_sub_preset(
            py_rng=py_rng,
            torch_generator=torch_generator,
        )
        unprocessing_transform = self._unprocessing_transforms[selected_sub_preset_name]
        raw_tensor, isp_params = unprocessing_transform(image_tensor, generator=torch_generator)
        raw_tensor = raw_tensor.float()
        expected_raw_shape = (4, int(self.size[0]), int(self.size[1]))
        if tuple(raw_tensor.shape) != expected_raw_shape:
            raise ValueError(
                f"Packed RAW shape mismatch for {img_path}: got={tuple(raw_tensor.shape)} "
                f"expected={expected_raw_shape}"
            )
        isp_params = self._augment_isp_params(
            isp_params,
            selected_sub_preset_name=selected_sub_preset_name,
        )
        isp_params["hflip_applied"] = bool(geometry_params["hflip_applied"])

        rgb_half = _downsample_rgb_2x2_area_from_even_fullres(image)
        depth_sensor, valid_mask_sensor = _downsample_depth_valid_mean_2x2(depth, valid_mask)
        if tuple(rgb_half.shape[:2]) != self.size:
            raise ValueError(f"Halfres RGB shape mismatch: got={rgb_half.shape[:2]} expected={self.size}")
        if tuple(depth_sensor.shape) != self.size or tuple(valid_mask_sensor.shape) != self.size:
            raise ValueError(
                "Halfres depth/valid shape mismatch: "
                f"depth={depth_sensor.shape} valid={valid_mask_sensor.shape} expected={self.size}"
            )

        sample = {
            "raw": raw_tensor,
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
        if include_rgb_input:
            sample["image"] = _imagenet_normalize_rgb_tensor_from_array(rgb_half)
        if include_rgb_preview:
            sample["rgb_preview"] = _rgb_preview_tensor_from_array(rgb_half)
        return sample

    def build_rgb_baseline_input(
        self,
        idx: int,
        geometry: Dict[str, Any],
        *,
        target_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path_str, _ = self.filelist[idx].split()
        img_path = Path(img_path_str)

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read VKITTI2 image for RGB baseline: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.raw_storage_format == "synthetic_packed_bayer_4ch_halfres":
            crop_box = geometry.get("crop_box")
            if crop_box is None:
                raise ValueError(f"Missing VKITTI2 halfres geometry crop_box for idx={idx}")
            if geometry.get("crop_box_semantics") not in (None, "source_fullres_to_even_fullres"):
                raise ValueError(
                    "Unexpected VKITTI halfres crop_box semantics: "
                    f"{geometry.get('crop_box_semantics')!r}"
                )
            image = _crop_hwew(image, tuple(crop_box))
            expected_even_hw = (int(target_hw[0]) * 2, int(target_hw[1]) * 2)
            if tuple(image.shape[:2]) != expected_even_hw:
                raise ValueError(
                    "VKITTI halfres RGB baseline geometry mismatch: "
                    f"cropped={tuple(image.shape[:2])} expected_even={expected_even_hw} idx={idx}"
                )
            if bool(geometry.get("hflip_applied", False)):
                image = np.ascontiguousarray(image[:, ::-1])
            rgb_half = _downsample_rgb_2x2_area_from_even_fullres(image)
            if tuple(rgb_half.shape[:2]) != tuple(int(v) for v in target_hw):
                raise ValueError(
                    f"VKITTI halfres RGB baseline output mismatch: got={rgb_half.shape[:2]} target={target_hw}"
                )
            return _imagenet_normalize_rgb_tensor_from_array(rgb_half), _rgb_preview_tensor_from_array(rgb_half)

        image = _resize_rgb_short_edge(image, short_edge=self.fullres_size[0])
        crop_box = geometry.get("crop_box")
        if crop_box is None:
            raise ValueError(f"Missing VKITTI2 geometry crop_box for idx={idx}")
        image = _crop_hwew(image, tuple(crop_box))
        if bool(geometry.get("hflip_applied", False)):
            image = np.ascontiguousarray(image[:, ::-1])
        return _imagenet_normalize_rgb_tensor(image, target_hw), _rgb_preview_tensor(image, target_hw)

    def _build_even_fullres_geometry(self, original_hw: Tuple[int, int] | list[int]) -> Dict[str, Any]:
        original_h, original_w = int(original_hw[0]), int(original_hw[1])
        if original_w % 2 != 0:
            raise ValueError(
                "VKITTI halfres packed RAW requires an even source width; "
                f"got original_hw={(original_h, original_w)}"
            )
        even_h = original_h - (original_h % 2)
        even_w = original_w
        if even_h <= 0:
            raise ValueError(f"Invalid source height for bottom crop: original_hw={(original_h, original_w)}")
        cropped_bottom_rows = original_h - even_h
        expected_even_hw = (int(self.size[0]) * 2, int(self.size[1]) * 2)
        if (even_h, even_w) != expected_even_hw:
            raise ValueError(
                "VKITTI halfres packed RAW size must match source bottom-cropped even fullres: "
                f"original_hw={(original_h, original_w)} even_fullres={(even_h, even_w)} "
                f"expected_even_fullres={expected_even_hw} input_size={self.size}"
            )
        return {
            "original_hw": [original_h, original_w],
            "source_original_hw": [original_h, original_w],
            "even_fullres_hw": [even_h, even_w],
            "fullres_even_policy": self.fullres_even_policy,
            "cropped_bottom_rows": int(cropped_bottom_rows),
            "crop_box": [0, 0, even_h, even_w],
            "crop_box_format": "h_start_w_start_h_end_w_end",
            "crop_box_semantics": "source_fullres_to_even_fullres",
            "packed_hw": [int(self.size[0]), int(self.size[1])],
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

    def _select_unprocessing_sub_preset(
        self,
        *,
        py_rng: Optional[random.Random],
        torch_generator: Optional[torch.Generator],
    ) -> str:
        if len(self.unprocessing_sub_presets) == 1:
            return self.unprocessing_sub_presets[0]
        if not self.randomize_unprocessing:
            return self.unprocessing_default_sub_preset

        if py_rng is not None:
            pivot = py_rng.random()
        elif torch_generator is not None:
            pivot = float(torch.rand((), generator=torch_generator).item())
        else:
            pivot = random.random()

        cumulative = 0.0
        for preset_name in self.unprocessing_sub_presets:
            cumulative += float(self.unprocessing_mix_weights[preset_name])
            if pivot <= cumulative:
                return preset_name
        return self.unprocessing_sub_presets[-1]

    def _augment_isp_params(
        self,
        isp_params: Dict[str, Any],
        *,
        selected_sub_preset_name: str,
    ) -> Dict[str, Any]:
        if self.unprocessing_method == "raw_adapter_style":
            return dict(isp_params)

        metadata = dict(isp_params)
        metadata["isp_profile_name"] = self.unprocessing_preset
        metadata["isp_profile_group"] = self.unprocessing_profile_group
        metadata["selected_sub_preset_name"] = selected_sub_preset_name
        metadata["preset_version"] = self.unprocessing_preset_version
        metadata["preset_hash"] = self.unprocessing_preset_hash
        metadata["preset_mix_weights"] = [
            {
                "name": name,
                "weight": float(self.unprocessing_mix_weights[name]),
            }
            for name in self.unprocessing_sub_presets
        ]
        metadata["selected_sub_preset_hash"] = self._sub_preset_hashes[selected_sub_preset_name]
        if metadata.get("xyz_to_cam_override") is None:
            metadata["xyz_to_cam_override"] = torch.empty(0, dtype=torch.float32)
        else:
            metadata["xyz_to_cam_override"] = torch.as_tensor(
                metadata["xyz_to_cam_override"],
                dtype=torch.float32,
            )
        return metadata

    def _resize_short_edge(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        valid_mask: np.ndarray,
        *,
        short_edge: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        scale = float(short_edge) / float(min(height, width))
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

    def _random_crop(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        valid_mask: np.ndarray,
        *,
        rng: Optional[random.Random] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        target_h, target_w = self.fullres_size
        height, width = image.shape[:2]
        if height < target_h or width < target_w:
            raise ValueError(
                f"Resize should guarantee minimum size >= {self.fullres_size}, but got {(height, width)}"
            )
        if height == target_h and width == target_w:
            return image, depth, valid_mask, (0, 0, target_h, target_w)

        rng = rng or random
        h_start = rng.randint(0, height - target_h)
        w_start = rng.randint(0, width - target_w)
        h_end = h_start + target_h
        w_end = w_start + target_w

        return (
            image[h_start:h_end, w_start:w_end],
            depth[h_start:h_end, w_start:w_end],
            valid_mask[h_start:h_end, w_start:w_end],
            (h_start, w_start, h_end, w_end),
        )

    def _center_crop(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        target_h, target_w = self.fullres_size
        height, width = image.shape[:2]
        if height < target_h or width < target_w:
            raise ValueError(
                f"Resize should guarantee minimum size >= {self.fullres_size}, but got {(height, width)}"
            )
        h_start = max((height - target_h) // 2, 0)
        w_start = max((width - target_w) // 2, 0)
        h_end = h_start + target_h
        w_end = w_start + target_w
        return (
            image[h_start:h_end, w_start:w_end],
            depth[h_start:h_end, w_start:w_end],
            valid_mask[h_start:h_end, w_start:w_end],
            (h_start, w_start, h_end, w_end),
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
