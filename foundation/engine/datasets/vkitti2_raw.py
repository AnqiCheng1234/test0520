from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from foundation.engine.transforms import (
    build_unprocessing_transform_from_preset,
    get_unprocessing_preset,
    resolve_unprocessing_mix_weights,
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
    return _numpy_to_torch(np.transpose(np.ascontiguousarray(resized), (2, 0, 1)).astype(np.float32, copy=False))


def _imagenet_normalize_rgb_tensor(image_rgb: np.ndarray, target_hw: Tuple[int, int]) -> torch.Tensor:
    resized = cv2.resize(image_rgb, (int(target_hw[1]), int(target_hw[0])), interpolation=cv2.INTER_AREA)
    normalized = (resized.astype(np.float32, copy=False) - IMAGENET_MEAN) / IMAGENET_STD
    return _numpy_to_torch(np.transpose(np.ascontiguousarray(normalized), (2, 0, 1)).astype(np.float32, copy=False))


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
        unprocessing_preset: str = "sensor_linear_dual",
        unprocessing_mix_weights: object | None = None,
        hflip_prob: float | None = None,
    ) -> None:
        self.mode = mode
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.size = tuple(size)
        self.fullres_size = (self.size[0] * 2, self.size[1] * 2)
        self.filelist_path = Path(filelist_path or DEFAULT_TRAIN_LIST).expanduser().resolve()

        if not self.filelist_path.is_file():
            raise FileNotFoundError(
                "Missing VKITTI2 split file. Generate it with scripts/generate_vkitti2_split.py "
                f"or pass filelist_path explicitly. Expected: {self.filelist_path}"
            )

        with self.filelist_path.open("r", encoding="utf-8") as f:
            self.filelist = [line.strip() for line in f if line.strip()]

        if not self.filelist:
            raise ValueError(f"No VKITTI2 samples found in {self.filelist_path}")

        self.randomize_unprocessing = bool(randomize_unprocessing)
        self.unprocessing_preset = str(unprocessing_preset)
        self.unprocessing_preset_spec = get_unprocessing_preset(self.unprocessing_preset)
        self.unprocessing_preset_version = str(self.unprocessing_preset_spec["preset_version"])
        self.unprocessing_preset_hash = str(self.unprocessing_preset_spec["preset_hash"])
        self.unprocessing_profile_group = str(self.unprocessing_preset_spec["isp_profile_group"])

        kind = str(self.unprocessing_preset_spec["kind"])
        if kind == "single":
            self.unprocessing_mix_weights = resolve_unprocessing_mix_weights(
                self.unprocessing_preset,
                unprocessing_mix_weights,
            )
            self.unprocessing_sub_presets = tuple(self.unprocessing_mix_weights.keys())
            self.unprocessing_default_sub_preset = self.unprocessing_preset
        elif kind == "dual":
            self.unprocessing_mix_weights = resolve_unprocessing_mix_weights(
                self.unprocessing_preset,
                unprocessing_mix_weights,
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

        self.hflip_prob = 0.5 if hflip_prob is None and self.mode == "train" else float(hflip_prob or 0.0)

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.build_sample(idx)

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
        active_unprocessing = self.unprocessing
        return {
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

    def build_sample(
        self,
        idx: int,
        *,
        py_rng: Optional[random.Random] = None,
        torch_generator: Optional[torch.Generator] = None,
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
            sample["rgb_preview"] = _rgb_preview_tensor(image, self.size)
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
        image = _resize_rgb_short_edge(image, short_edge=self.fullres_size[0])
        crop_box = geometry.get("crop_box")
        if crop_box is None:
            raise ValueError(f"Missing VKITTI2 geometry crop_box for idx={idx}")
        image = _crop_hwew(image, tuple(crop_box))
        if bool(geometry.get("hflip_applied", False)):
            image = np.ascontiguousarray(image[:, ::-1])
        return _imagenet_normalize_rgb_tensor(image, target_hw), _rgb_preview_tensor(image, target_hw)

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
