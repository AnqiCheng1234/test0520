#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from depth_anything_v2.dpt import DepthAnythingV2
from foundation.engine.datasets import validate_vkitti_raw_semantics
from foundation.engine.models import build_raw_residual_dav2_model
from foundation.engine.transforms import (
    NOT_APPLICABLE,
    assert_unprocessing_summary_matches_config,
    build_unprocessing_transform_from_resolved_config,
    resolve_unprocessing_config,
)
from foundation.tools._viz_distribution import (
    DEFAULT_RAW_COLORS,
    DEFAULT_RGB_COLORS,
    draw_distribution_tile,
    summarize_channels,
)
from foundation.tools.make_vkitti_raw_residual_qual_panels import (
    choose_depth_range,
    clip_metric_depth_for_eval,
    colorize_depth,
    colorize_error,
    colorize_gate,
    colorize_improvement,
    colorize_signed,
    draw_tile,
    image_from_array,
    load_font,
)


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog", "d1", "d2", "d3")
TARGET_EVEN_HW = (374, 1242)
TARGET_MODEL_HW = (187, 621)
GEOMETRY_POLICY = "canonical_even_pad_crop"
MIN_VALID_PIXELS = 128


class KittiGeometryError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline KITTI eval for VKITTI-trained RAW residual DAv2.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kitti-val-split", required=True)
    parser.add_argument("--kitti-base", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-panels", type=int, default=10)
    parser.add_argument("--panel-selection", choices=["uniform"], default="uniform")
    parser.add_argument("--sample-indices", default=None, help="Comma-separated original KITTI dataset indices.")
    parser.add_argument("--error-max-abs-rel", type=float, default=0.75)
    parser.add_argument("--depth-pmin", type=float, default=1.0)
    parser.add_argument("--depth-pmax", type=float, default=99.0)
    parser.add_argument("--hist-bins", type=int, default=128)
    parser.add_argument("--tile-width", type=int, default=414)
    parser.add_argument("--tile-height", type=int, default=125)
    parser.add_argument("--header-height", type=int, default=30)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_raw_residual_kitti")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("[%(asctime)s][%(levelname)8s] %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(output_dir / "eval.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def resolve_model_state(ckpt_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    return ckpt_obj


def load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_run_config(config: dict[str, Any]) -> None:
    expected = {
        "input_domain": "raw4",
        "front_end": "raw_to_base_rgb_ram3",
        "model_input_tensor": "raw",
        "raw_storage_format": "synthetic_packed_bayer_4ch_halfres",
        "fullres_even_policy": "crop_bottom_to_even",
        "rgb_input_space": "halfres_2x2_area",
        "depth_target_space": "halfres_2x2_valid_mean",
        "input_height": 187,
        "input_width": 621,
        "min_depth": 1.0,
        "max_depth": 80.0,
    }
    for key, value in expected.items():
        actual = config.get(key)
        if isinstance(value, float):
            if not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"Config {key} must be {value!r}, got {actual!r}")
        else:
            if actual != value:
                raise ValueError(f"Config {key} must be {value!r}, got {actual!r}")
    validate_vkitti_raw_semantics(
        raw_storage_format=str(config["raw_storage_format"]),
        fullres_even_policy=str(config["fullres_even_policy"]),
        rgb_input_space=str(config["rgb_input_space"]),
        depth_target_space=str(config["depth_target_space"]),
    )
    method = str(config.get("unprocessing_method", "old_brooks_preset"))
    config_for_unprocessing = dict(config)
    config_for_unprocessing["unprocessing_method"] = method
    resolved = resolve_unprocessing_config(config_for_unprocessing)
    if method == "old_brooks_preset":
        if str(resolved["vkitti_unprocessing_preset"]) != "sensor_linear_dual":
            raise ValueError(
                "old_brooks_preset KITTI eval currently requires "
                "vkitti_unprocessing_preset='sensor_linear_dual', got "
                f"{resolved['vkitti_unprocessing_preset']!r}"
            )
        return

    if method != "raw_adapter_style":
        raise ValueError(f"Unsupported unprocessing_method in run config: {method!r}")
    if resolved["raw_adapter_backend"] != "analytic":
        raise ValueError("Phase A KITTI eval only supports raw_adapter_backend='analytic'")
    if bool(resolved["randomize_unprocessing"]):
        raise ValueError("Phase A raw_adapter_style KITTI eval requires randomize_unprocessing=false")
    if resolved["raw_adapter_random_seed_policy"] != "dataloader_generator":
        raise ValueError("raw_adapter_random_seed_policy must be 'dataloader_generator'")
    if resolved["raw_adapter_cfa_pattern"] != "RGGB":
        raise ValueError("raw_adapter_cfa_pattern must be 'RGGB'")
    if resolved["raw_adapter_packed_channel_order"] != "R_Gr_Gb_B":
        raise ValueError("raw_adapter_packed_channel_order must be 'R_Gr_Gb_B'")
    if resolved["raw_adapter_rgb_transfer"] != "srgb_piecewise":
        raise ValueError("raw_adapter_rgb_transfer must be 'srgb_piecewise'")
    if resolved["raw_adapter_inverse_tone"] not in ("none", "global_0p15"):
        raise ValueError("raw_adapter_inverse_tone must be 'none' or 'global_0p15'")
    if resolved["raw_adapter_noise_mean_mode"] not in ("zero", "rawadapter_text"):
        raise ValueError("raw_adapter_noise_mean_mode must be 'zero' or 'rawadapter_text'")
    if resolved["raw_adapter_variant_policy"] not in ("normal", "dark", "over"):
        raise ValueError("fixed Phase A raw_adapter_style eval requires variant_policy normal/dark/over")
    if resolved["noise_model"] != "none" or bool(resolved["noise_realization_applied"]):
        raise ValueError("fixed Phase A raw_adapter_style eval requires noise_model=none and no noise realization")


def remap_kitti_path(path_str: str, kitti_base: Path) -> Path:
    path = path_str.strip()
    raw_prefix = "/mnt/bn/liheyang/Kitti/raw_data/"
    depth_prefix = "/mnt/bn/liheyang/Kitti/data_depth_annotated/"
    if path.startswith(raw_prefix):
        rel_path = Path(path[len(raw_prefix) :])
        if rel_path.suffix.lower() == ".png":
            rel_path = rel_path.with_suffix(".jpg")
        return kitti_base / rel_path
    if path.startswith(depth_prefix):
        return kitti_base / "annotated_depth" / Path(path[len(depth_prefix) :])
    return Path(path)


def numpy_to_torch(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array))


def imagenet_normalize_rgb_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    image_rgb = np.clip(image_rgb.astype(np.float32, copy=False), 0.0, 1.0)
    normalized = (image_rgb - IMAGENET_MEAN) / IMAGENET_STD
    return numpy_to_torch(np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False))


def rgb_preview_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    image_rgb = np.clip(image_rgb.astype(np.float32, copy=False), 0.0, 1.0)
    return numpy_to_torch(np.transpose(image_rgb, (2, 0, 1)).astype(np.float32, copy=False))


def downsample_rgb_2x2_area(image_rgb: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if h % 2 != 0 or w % 2 != 0:
        raise KittiGeometryError(f"Expected even fullres RGB shape, got {(h, w)}")
    return image_rgb.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3)).astype(np.float32, copy=False)


def downsample_depth_valid_mean_2x2(depth: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape[:2]
    if h % 2 != 0 or w % 2 != 0:
        raise KittiGeometryError(f"Expected even fullres depth shape, got {(h, w)}")
    depth_blocks = np.ascontiguousarray(depth).reshape(h // 2, 2, w // 2, 2)
    valid_blocks = np.ascontiguousarray(valid_mask).reshape(h // 2, 2, w // 2, 2)
    counts = valid_blocks.sum(axis=(1, 3)).astype(np.float32, copy=False)
    sums = (depth_blocks * valid_blocks.astype(np.float32, copy=False)).sum(axis=(1, 3))
    valid_half = counts > 0.0
    depth_half = np.zeros((h // 2, w // 2), dtype=np.float32)
    depth_half[valid_half] = sums[valid_half] / counts[valid_half]
    return depth_half, valid_half


def sample_name_from_image_path(image_path: Path) -> str:
    if image_path.parent.name == "data" and image_path.parent.parent.name.startswith("image_"):
        camera = image_path.parent.parent.name
        drive = image_path.parent.parent.parent.name
        return f"{drive}_{camera}_{image_path.stem}"
    return image_path.stem


def center_crop_pad_params(source_hw: tuple[int, int], target_hw: tuple[int, int]) -> dict[str, int]:
    source_h, source_w = int(source_hw[0]), int(source_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if source_h <= 0 or source_w <= 0:
        raise KittiGeometryError(f"Invalid source shape for canonicalization: {source_hw}")
    h_crop = max(source_h - target_h, 0)
    w_crop = max(source_w - target_w, 0)
    top_crop = h_crop // 2
    bottom_crop = h_crop - top_crop
    left_crop = w_crop // 2
    right_crop = w_crop - left_crop
    cropped_h = source_h - top_crop - bottom_crop
    cropped_w = source_w - left_crop - right_crop
    h_pad = max(target_h - cropped_h, 0)
    w_pad = max(target_w - cropped_w, 0)
    top_pad = h_pad // 2
    bottom_pad = h_pad - top_pad
    left_pad = w_pad // 2
    right_pad = w_pad - left_pad
    return {
        "top_pad": int(top_pad),
        "bottom_pad": int(bottom_pad),
        "left_pad": int(left_pad),
        "right_pad": int(right_pad),
        "top_crop": int(top_crop),
        "bottom_crop": int(bottom_crop),
        "left_crop": int(left_crop),
        "right_crop": int(right_crop),
    }


def apply_canonical_even_pad_crop(
    image: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    *,
    target_hw: tuple[int, int] = TARGET_EVEN_HW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    source_hw = tuple(int(v) for v in image.shape[:2])
    params = center_crop_pad_params(source_hw, target_hw)
    h, w = source_hw
    top_crop = params["top_crop"]
    bottom_crop = params["bottom_crop"]
    left_crop = params["left_crop"]
    right_crop = params["right_crop"]
    h_end = h - bottom_crop
    w_end = w - right_crop
    image = np.ascontiguousarray(image[top_crop:h_end, left_crop:w_end, :])
    depth = np.ascontiguousarray(depth[top_crop:h_end, left_crop:w_end])
    valid_mask = np.ascontiguousarray(valid_mask[top_crop:h_end, left_crop:w_end])

    pad_hw = (
        (params["top_pad"], params["bottom_pad"]),
        (params["left_pad"], params["right_pad"]),
    )
    if any(v > 0 for key, v in params.items() if key.endswith("_pad")):
        image = np.pad(image, (*pad_hw, (0, 0)), mode="edge")
        depth = np.pad(depth, pad_hw, mode="constant", constant_values=0.0)
        valid_mask = np.pad(valid_mask, pad_hw, mode="constant", constant_values=False)

    if tuple(image.shape[:2]) != tuple(target_hw):
        raise KittiGeometryError(
            f"Canonical KITTI shape mismatch: got={tuple(image.shape[:2])} target={target_hw}"
        )
    return image, depth.astype(np.float32, copy=False), valid_mask.astype(bool, copy=False), params


class KittiHalfresRawDataset(Dataset):
    def __init__(
        self,
        *,
        filelist_path: str | Path,
        kitti_base: str | Path,
        min_depth: float,
        max_depth: float,
        unprocessing_config: Mapping[str, Any] | None = None,
        unprocessing_preset: str | None = None,
    ) -> None:
        self.filelist_path = Path(filelist_path).expanduser().resolve()
        self.kitti_base = Path(kitti_base).expanduser().resolve()
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        if unprocessing_config is None:
            if unprocessing_preset is None:
                raise ValueError("KittiHalfresRawDataset requires unprocessing_config or unprocessing_preset")
            unprocessing_config = {
                "unprocessing_method": "old_brooks_preset",
                "randomize_unprocessing": False,
                "vkitti_unprocessing_preset": str(unprocessing_preset),
                "vkitti_unprocessing_mix_weights": None,
            }
        self.resolved_unprocessing_config = resolve_unprocessing_config(dict(unprocessing_config))
        self.unprocessing, self.unprocessing_summary = build_unprocessing_transform_from_resolved_config(
            self.resolved_unprocessing_config,
            split="kitti_val",
        )
        self.unprocessing_preset = str(
            self.resolved_unprocessing_config.get("vkitti_unprocessing_preset", NOT_APPLICABLE)
        )

        if not self.filelist_path.is_file():
            raise FileNotFoundError(f"Missing KITTI split file: {self.filelist_path}")
        if not self.kitti_base.is_dir():
            raise FileNotFoundError(f"Missing KITTI base directory: {self.kitti_base}")

        self.rows: list[dict[str, Any]] = []
        with self.filelist_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                image_path_str, depth_path_str = line.split()
                image_path = remap_kitti_path(image_path_str, self.kitti_base)
                depth_path = remap_kitti_path(depth_path_str, self.kitti_base)
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing KITTI image on split row {line_idx}: {image_path}")
                if not depth_path.is_file():
                    raise FileNotFoundError(f"Missing KITTI depth on split row {line_idx}: {depth_path}")
                self.rows.append(
                    {
                        "image_path": image_path,
                        "depth_path": depth_path,
                        "raw_image_path": image_path_str,
                        "raw_depth_path": depth_path_str,
                    }
                )
        if not self.rows:
            raise ValueError(f"No KITTI samples found in {self.filelist_path}")

        self.source_shape_counts = self._scan_source_shape_counts()

    def __len__(self) -> int:
        return len(self.rows)

    def _scan_source_shape_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for row in self.rows:
            image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to read KITTI image while scanning shapes: {row['image_path']}")
            h, w = image.shape[:2]
            counts[f"{int(h)}x{int(w)}"] += 1
        return dict(sorted(counts.items()))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        try:
            sample = self.build_sample(idx)
            sample["status"] = "ok"
            return sample
        except FileNotFoundError as exc:
            return self._error_sample(idx, "skipped_io_error", exc)
        except KittiGeometryError as exc:
            return self._error_sample(idx, "skipped_geometry_error", exc)
        except (OSError, ValueError, cv2.error) as exc:
            return self._error_sample(idx, "skipped_io_error", exc)

    def _error_sample(self, idx: int, status: str, exc: BaseException) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "dataset_index": int(idx),
            "sample_name": sample_name_from_image_path(row["image_path"]),
            "image_path": str(row["image_path"]),
            "depth_path": str(row["depth_path"]),
            "status": status,
            "error": str(exc),
        }

    def build_sample(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image_path = row["image_path"]
        depth_path = row["depth_path"]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read KITTI image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        source_hw = [int(image.shape[0]), int(image.shape[1])]

        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"Failed to read KITTI depth: {depth_path}")
        depth = depth_raw.astype(np.float32) / 256.0
        if tuple(depth.shape[:2]) != tuple(image.shape[:2]):
            raise KittiGeometryError(
                f"KITTI RGB/depth shape mismatch for idx={idx}: image={image.shape[:2]} depth={depth.shape[:2]}"
            )

        valid_mask = np.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)
        depth = np.where(valid_mask, depth, 0.0).astype(np.float32, copy=False)

        cropped_bottom_rows = 0
        if image.shape[0] % 2 != 0:
            image = np.ascontiguousarray(image[:-1, :, :])
            depth = np.ascontiguousarray(depth[:-1, :])
            valid_mask = np.ascontiguousarray(valid_mask[:-1, :])
            cropped_bottom_rows = 1
        after_even_policy_hw = [int(image.shape[0]), int(image.shape[1])]

        image, depth, valid_mask, pad_crop = apply_canonical_even_pad_crop(image, depth, valid_mask)
        canonical_even_hw = [int(image.shape[0]), int(image.shape[1])]
        if tuple(canonical_even_hw) != TARGET_EVEN_HW:
            raise KittiGeometryError(f"Expected canonical KITTI HW {TARGET_EVEN_HW}, got {canonical_even_hw}")

        image_tensor = numpy_to_torch(
            np.transpose(np.ascontiguousarray(image), (2, 0, 1)).astype(np.float32, copy=False)
        )
        raw_tensor, isp_params = self.unprocessing(image_tensor)
        raw_tensor = raw_tensor.float()
        if tuple(raw_tensor.shape) != (4, *TARGET_MODEL_HW):
            raise KittiGeometryError(f"Packed RAW shape mismatch: got={tuple(raw_tensor.shape)}")

        rgb_half = downsample_rgb_2x2_area(image)
        depth_half, valid_half = downsample_depth_valid_mean_2x2(depth, valid_mask)
        if tuple(rgb_half.shape[:2]) != TARGET_MODEL_HW:
            raise KittiGeometryError(f"Halfres RGB shape mismatch: got={rgb_half.shape[:2]}")
        if tuple(depth_half.shape) != TARGET_MODEL_HW or tuple(valid_half.shape) != TARGET_MODEL_HW:
            raise KittiGeometryError(
                f"Halfres depth/mask shape mismatch: depth={depth_half.shape} valid={valid_half.shape}"
            )

        geometry = {
            "source_hw": source_hw,
            "after_even_policy_hw": after_even_policy_hw,
            "canonical_even_hw": canonical_even_hw,
            "target_even_fullres_hw": list(TARGET_EVEN_HW),
            "raw_hw": list(TARGET_MODEL_HW),
            "raw_shape": [4, int(TARGET_MODEL_HW[0]), int(TARGET_MODEL_HW[1])],
            "geometry_policy": GEOMETRY_POLICY,
            "fullres_even_policy": "crop_bottom_to_even",
            "cropped_bottom_rows": int(cropped_bottom_rows),
            "pad_crop": pad_crop,
        }
        return {
            "dataset_index": int(idx),
            "raw": raw_tensor,
            "image": imagenet_normalize_rgb_tensor(rgb_half),
            "rgb_preview": rgb_preview_tensor(rgb_half),
            "depth": numpy_to_torch(depth_half.astype(np.float32, copy=False)),
            "valid_mask": numpy_to_torch(valid_half.astype(np.uint8)).bool(),
            "image_path": str(image_path),
            "depth_path": str(depth_path),
            "sample_name": sample_name_from_image_path(image_path),
            "geometry_params": geometry,
            "isp_params": isp_params,
            "target_space": "metric_depth",
        }

    def describe_geometry(self) -> dict[str, Any]:
        return {
            "name": GEOMETRY_POLICY,
            "target_even_fullres_hw": list(TARGET_EVEN_HW),
            "target_model_hw": list(TARGET_MODEL_HW),
            "fullres_even_policy": "crop_bottom_to_even",
            "canonicalization": "center pad/crop to fixed even fullres canvas; RGB edge pad; depth zero pad; valid false pad",
            "source_shape_counts": self.source_shape_counts,
        }

    def describe_unprocessing(self) -> dict[str, Any]:
        return dict(self.unprocessing_summary)


def build_model(config: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    base_model = DepthAnythingV2(**MODEL_CONFIGS[str(config["encoder"])])
    model = build_raw_residual_dav2_model(
        base_model,
        residual_feature_source=str(config["residual_feature_source"]),
        residual_alpha=float(config["residual_alpha"]),
        d0_sign=int(config["d0_sign"]),
        sensor_hw=(int(config["input_height"]), int(config["input_width"])),
        backbone_hw=None,
    )
    ckpt_obj = torch.load(str(checkpoint), map_location="cpu")
    state = strip_module_prefix(resolve_model_state(ckpt_obj))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def parse_indices(arg: str | None, dataset_len: int) -> list[int] | None:
    if not arg:
        return None
    indices = [int(x.strip()) for x in arg.split(",") if x.strip()]
    out: list[int] = []
    for idx in indices:
        if idx < 0 or idx >= dataset_len:
            raise IndexError(f"Sample index {idx} out of range for KITTI dataset length {dataset_len}")
        if idx not in out:
            out.append(idx)
    return out


def select_uniform_indices(ok_rows: list[dict[str, Any]], max_panels: int) -> list[int]:
    if max_panels <= 0:
        return []
    if len(ok_rows) < max_panels:
        raise RuntimeError(
            f"Cannot select {max_panels} uniform panels from only {len(ok_rows)} ok KITTI samples."
        )
    if max_panels == 1:
        positions = [0]
    else:
        positions = [int(x) for x in np.linspace(0, len(ok_rows) - 1, int(max_panels))]
    selected: list[int] = []
    for pos in positions:
        idx = int(ok_rows[pos]["dataset_index"])
        if idx not in selected:
            selected.append(idx)
    if len(selected) != max_panels:
        for row in ok_rows:
            idx = int(row["dataset_index"])
            if idx not in selected:
                selected.append(idx)
            if len(selected) == max_panels:
                break
    return selected


def filter_metrics(metrics: dict[str, Any] | None) -> dict[str, float | None]:
    if metrics is None:
        return {key: None for key in METRIC_KEYS}
    return {key: to_jsonable(metrics.get(key)) for key in METRIC_KEYS}


def mean_finite(values: list[Any]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    return float(np.mean(finite))


def average_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    return {key: mean_finite([row.get(key) for row in rows]) for key in METRIC_KEYS}


def make_base_row(sample: dict[str, Any]) -> dict[str, Any]:
    geometry = sample.get("geometry_params", {})
    return {
        "dataset_index": int(sample["dataset_index"]),
        "sample_name": str(sample["sample_name"]),
        "image_path": str(sample["image_path"]),
        "depth_path": str(sample["depth_path"]),
        "status": str(sample.get("status", "ok")),
        "source_hw": geometry.get("source_hw"),
        "after_even_policy_hw": geometry.get("after_even_policy_hw"),
        "canonical_even_hw": geometry.get("canonical_even_hw"),
        "raw_hw": geometry.get("raw_hw"),
        "raw_shape": geometry.get("raw_shape"),
        "geometry_policy": geometry.get("geometry_policy", GEOMETRY_POLICY),
        "pad_crop": geometry.get("pad_crop"),
        "valid_pixels": None,
        "final": filter_metrics(None),
        "D0": filter_metrics(None),
        "diagnostics": {
            "mean_gate": None,
            "max_gate": None,
            "mean_abs_delta": None,
            "mean_abs_gate_delta": None,
        },
    }


def packed_raw_preview(raw_chw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_chw, dtype=np.float32)
    rgb = np.stack([raw[0], 0.5 * (raw[1] + raw[2]), raw[3]], axis=-1)
    finite = rgb[np.isfinite(rgb)]
    if finite.size:
        lo = float(np.percentile(finite, 1.0))
        hi = float(np.percentile(finite, 99.0))
    else:
        lo, hi = 0.0, 1.0
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32, copy=False)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def add_horizontal_colorbar(
    tile: Image.Image,
    *,
    colors: np.ndarray,
    labels: tuple[str, str, str],
    font: ImageFont.ImageFont,
) -> Image.Image:
    out = tile.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    bar_left = 42
    bar_right = width - 42
    bar_top = height - 20
    bar_bottom = height - 13
    draw.rectangle([0, bar_top - 4, width, height], fill=(16, 16, 16))

    gradient = np.asarray(colors, dtype=np.uint8)
    if gradient.ndim != 2 or gradient.shape[1] != 3:
        raise ValueError(f"Expected colorbar gradient shape (N,3), got {gradient.shape}")
    gradient_img = Image.fromarray(gradient.reshape(1, gradient.shape[0], 3), mode="RGB").resize(
        (bar_right - bar_left, bar_bottom - bar_top + 1),
        Image.Resampling.BILINEAR,
    )
    out.paste(gradient_img, (bar_left, bar_top))
    draw.rectangle([bar_left, bar_top, bar_right, bar_bottom], outline=(238, 238, 238), width=1)

    y_text = bar_bottom + 1
    left_label, mid_label, right_label = labels
    draw.text((bar_left, y_text), left_label, fill=(245, 245, 245), font=font)
    draw.text(
        ((bar_left + bar_right - _text_width(draw, mid_label, font)) // 2, y_text),
        mid_label,
        fill=(245, 245, 245),
        font=font,
    )
    draw.text(
        (bar_right - _text_width(draw, right_label, font), y_text),
        right_label,
        fill=(245, 245, 245),
        font=font,
    )
    return out


def colormap_gradient(cmap_name: str, *, width: int = 256) -> np.ndarray:
    values = np.linspace(0.0, 1.0, int(width), dtype=np.float32)
    return (colormaps[cmap_name](values)[..., :3] * 255.0).round().astype(np.uint8)


def improvement_gradient(*, vlim: float, width: int = 256) -> np.ndarray:
    values = np.linspace(-float(vlim), float(vlim), int(width), dtype=np.float32)
    pos = np.clip(values / max(float(vlim), 1e-6), 0.0, 1.0)
    neg = np.clip(-values / max(float(vlim), 1e-6), 0.0, 1.0)
    rgb = np.zeros((int(width), 3), dtype=np.uint8)
    rgb[:, 1] = (pos * 255.0).round().astype(np.uint8)
    rgb[:, 0] = (neg * 255.0).round().astype(np.uint8)
    rgb[:, 2] = (neg * 90.0).round().astype(np.uint8)
    return rgb


def metric_label(value: float, *, unit: str = "", precision: int = 2, show_sign: bool = False) -> str:
    sign = "+" if show_sign else ""
    return f"{float(value):{sign}.{precision}f}{unit}"


def collect_eval_for_sample(
    *,
    sample: dict[str, Any],
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    args: argparse.Namespace,
    collect_panel: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    row = make_base_row(sample)
    if sample.get("status") != "ok":
        row["error"] = sample.get("error")
        return row, None

    image = sample["image"].unsqueeze(0).to(device, non_blocking=True).float()
    raw = sample["raw"].unsqueeze(0).to(device, non_blocking=True).float()
    depth_t = sample["depth"].unsqueeze(0).to(device, non_blocking=True).float()
    valid_t = sample["valid_mask"].unsqueeze(0).to(device, non_blocking=True).bool()
    valid_t = valid_t & (depth_t >= float(config["min_depth"])) & (depth_t <= float(config["max_depth"]))
    valid_pixels = int(valid_t[0].sum().item())
    row["valid_pixels"] = valid_pixels
    if valid_pixels < MIN_VALID_PIXELS:
        row["status"] = "skipped_invalid_pixels"
        row["error"] = f"valid_pixels={valid_pixels} < {MIN_VALID_PIXELS}"
        return row, None

    try:
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=amp_enabled and device.type == "cuda",
        ):
            out = model({"image": image, "raw": raw, "valid_mask": valid_t})

        depth_np = depth_t[0].detach().cpu().numpy().astype(np.float32)
        valid_np = valid_t[0].detach().cpu().numpy().astype(bool)
        pred_np = out["pred"][0].float().detach().cpu().numpy().astype(np.float32)
        d0_np = (float(config["d0_sign"]) * out["D0"][0].float()).detach().cpu().numpy().astype(np.float32)
        aligned_final, _ = affine_align_disp(depth_np, pred_np, valid_np)
        aligned_d0, _ = affine_align_disp(depth_np, d0_np, valid_np)
        metrics_final = compute_metrics(
            depth_np,
            aligned_final,
            valid_np,
            min_depth=float(config["min_depth"]),
            max_depth=float(config["max_depth"]),
        )
        metrics_d0 = compute_metrics(
            depth_np,
            aligned_d0,
            valid_np,
            min_depth=float(config["min_depth"]),
            max_depth=float(config["max_depth"]),
        )
        if metrics_final is None or metrics_d0 is None:
            row["status"] = "skipped_metric_failure"
            row["error"] = "compute_metrics returned None"
            return row, None

        gate = out["gate"].float()
        delta = out["delta"].float()
        gate_delta = gate * delta
        row["status"] = "ok"
        row["final"] = filter_metrics(metrics_final)
        row["D0"] = filter_metrics(metrics_d0)
        row["diagnostics"] = {
            "mean_gate": float(gate[valid_t].mean().detach().item()),
            "max_gate": float(gate[valid_t].max().detach().item()),
            "mean_abs_delta": float(delta[valid_t].abs().mean().detach().item()),
            "mean_abs_gate_delta": float(gate_delta[valid_t].abs().mean().detach().item()),
        }

        if not collect_panel:
            return row, None

        aligned_final = aligned_final.astype(np.float32)
        aligned_d0 = aligned_d0.astype(np.float32)
        aligned_final_eval = clip_metric_depth_for_eval(
            aligned_final,
            min_depth=float(config["min_depth"]),
            max_depth=float(config["max_depth"]),
        )
        aligned_d0_eval = clip_metric_depth_for_eval(
            aligned_d0,
            min_depth=float(config["min_depth"]),
            max_depth=float(config["max_depth"]),
        )
        error_valid_final = valid_np & np.isfinite(aligned_final_eval) & (aligned_final_eval > 0.0) & (depth_np > 0.0)
        error_valid_d0 = valid_np & np.isfinite(aligned_d0_eval) & (aligned_d0_eval > 0.0) & (depth_np > 0.0)
        eval_valid = error_valid_final & error_valid_d0
        err_final = np.zeros_like(depth_np, dtype=np.float32)
        err_d0 = np.zeros_like(depth_np, dtype=np.float32)
        err_final[error_valid_final] = (
            np.abs(aligned_final_eval[error_valid_final] - depth_np[error_valid_final])
            / np.clip(depth_np[error_valid_final], 1e-6, None)
        )
        err_d0[error_valid_d0] = (
            np.abs(aligned_d0_eval[error_valid_d0] - depth_np[error_valid_d0])
            / np.clip(depth_np[error_valid_d0], 1e-6, None)
        )
        depth_vmin, depth_vmax = choose_depth_range(
            depth_np,
            valid_np,
            min_depth=float(config["min_depth"]),
            max_depth=float(config["max_depth"]),
            pmin=float(args.depth_pmin),
            pmax=float(args.depth_pmax),
        )
        raw_np = sample["raw"].detach().cpu().numpy().astype(np.float32)
        rgb_np = sample["rgb_preview"].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        panel_record = {
            "dataset_index": int(sample["dataset_index"]),
            "sample_name": str(sample["sample_name"]),
            "image_path": str(sample["image_path"]),
            "depth_path": str(sample["depth_path"]),
            "rgb": rgb_np,
            "raw": raw_np,
            "raw_preview": packed_raw_preview(raw_np),
            "rgb_stats": summarize_channels(rgb_np, channels=("R", "G", "B"), channel_axis=2),
            "raw_stats": summarize_channels(raw_np, channels=("R", "Gr", "Gb", "B"), channel_axis=0),
            "depth": depth_np,
            "gt_valid": valid_np,
            "valid": eval_valid,
            "aligned_d0": aligned_d0,
            "aligned_final": aligned_final,
            "err_d0": err_d0,
            "err_final": err_final,
            "gate": gate[0].detach().cpu().numpy().astype(np.float32),
            "delta": delta[0].detach().cpu().numpy().astype(np.float32),
            "gate_delta": gate_delta[0].detach().cpu().numpy().astype(np.float32),
            "depth_vmin": float(depth_vmin),
            "depth_vmax": float(depth_vmax),
            "final": row["final"],
            "D0": row["D0"],
        }
        return row, panel_record
    except Exception as exc:  # noqa: BLE001 - status rows must survive per-sample failures.
        row["status"] = "skipped_metric_failure"
        row["error"] = str(exc)
        return row, None


def make_panel(record: dict[str, Any], args: argparse.Namespace, residual_vlim: float, epoch: int) -> tuple[Image.Image, dict[str, Any]]:
    tile_w = int(args.tile_width)
    tile_h = int(args.tile_height)
    header_h = int(args.header_height)
    font = load_font(12)
    small_font = load_font(10)
    bar_font = load_font(9)
    canvas = Image.new("RGB", (tile_w * 4, (tile_h + header_h) * 3), (0, 0, 0))
    gt_valid = record["gt_valid"]
    display_valid = np.ones_like(gt_valid, dtype=bool)
    depth_range = f"{record['depth_vmin']:.2f}..{record['depth_vmax']:.2f}m"
    error_range = f"0..{float(args.error_max_abs_rel):.2f} absrel"
    improve_range = f"+green +/-{float(args.error_max_abs_rel):.2f}"
    residual_range = f"+/-{float(residual_vlim):.3f}"
    d0_depth_metric = f"{depth_range} absrel={record['D0']['abs_rel']:.3f} delta1={record['D0']['d1']:.3f}"
    final_depth_metric = (
        f"{depth_range} absrel={record['final']['abs_rel']:.3f} delta1={record['final']['d1']:.3f}"
    )
    depth_mid = 0.5 * (float(record["depth_vmin"]) + float(record["depth_vmax"]))
    error_mid = 0.5 * float(args.error_max_abs_rel)
    depth_bar = colormap_gradient("Spectral_r")
    error_bar = colormap_gradient("magma")
    residual_bar = colormap_gradient("coolwarm")
    gate_bar = colormap_gradient("viridis")
    improve_bar = improvement_gradient(vlim=float(args.error_max_abs_rel))
    depth_labels = (
        metric_label(record["depth_vmin"], unit="m", precision=1),
        metric_label(depth_mid, unit="m", precision=1),
        metric_label(record["depth_vmax"], unit="m", precision=1),
    )
    error_labels = (
        metric_label(0.0, precision=2),
        metric_label(error_mid, precision=2),
        metric_label(args.error_max_abs_rel, precision=2),
    )
    residual_labels = (
        metric_label(-residual_vlim, precision=2, show_sign=True),
        metric_label(0.0, precision=2),
        metric_label(residual_vlim, precision=2, show_sign=True),
    )
    gate_labels = (metric_label(0.0, precision=1), metric_label(0.5, precision=1), metric_label(1.0, precision=1))
    improve_labels = (
        metric_label(-args.error_max_abs_rel, precision=2, show_sign=True),
        metric_label(0.0, precision=2),
        metric_label(args.error_max_abs_rel, precision=2, show_sign=True),
    )

    rgb_hist, rgb_dist_meta = draw_distribution_tile(
        record["rgb"],
        channels=("R", "G", "B"),
        colors=DEFAULT_RGB_COLORS,
        channel_axis=2,
        bins=int(args.hist_bins),
        width=tile_w,
        height=tile_h,
        font=font,
        small_font=small_font,
    )
    raw_hist, raw_dist_meta = draw_distribution_tile(
        record["raw"],
        channels=("R", "Gr", "Gb", "B"),
        colors=DEFAULT_RAW_COLORS,
        channel_axis=0,
        bins=int(args.hist_bins),
        width=tile_w,
        height=tile_h,
        font=font,
        small_font=small_font,
    )

    dav2_depth_tile = add_horizontal_colorbar(
        image_from_array(
            colorize_depth(record["aligned_d0"], display_valid, vmin=record["depth_vmin"], vmax=record["depth_vmax"]),
            tile_width=tile_w,
            tile_height=tile_h,
        ),
        colors=depth_bar,
        labels=depth_labels,
        font=bar_font,
    )
    ours_depth_tile = add_horizontal_colorbar(
        image_from_array(
            colorize_depth(record["aligned_final"], display_valid, vmin=record["depth_vmin"], vmax=record["depth_vmax"]),
            tile_width=tile_w,
            tile_height=tile_h,
        ),
        colors=depth_bar,
        labels=depth_labels,
        font=bar_font,
    )
    residual_tile = add_horizontal_colorbar(
        image_from_array(
            colorize_signed(record["gate_delta"], display_valid, vlim=residual_vlim, cmap_name="coolwarm"),
            tile_width=tile_w,
            tile_height=tile_h,
        ),
        colors=residual_bar,
        labels=residual_labels,
        font=bar_font,
    )
    gate_tile = add_horizontal_colorbar(
        image_from_array(colorize_gate(record["gate"], display_valid), tile_width=tile_w, tile_height=tile_h),
        colors=gate_bar,
        labels=gate_labels,
        font=bar_font,
    )
    dav2_error_tile = add_horizontal_colorbar(
        image_from_array(
            colorize_error(record["err_d0"], display_valid, vmax=float(args.error_max_abs_rel)),
            tile_width=tile_w,
            tile_height=tile_h,
        ),
        colors=error_bar,
        labels=error_labels,
        font=bar_font,
    )
    ours_error_tile = add_horizontal_colorbar(
        image_from_array(
            colorize_error(record["err_final"], display_valid, vmax=float(args.error_max_abs_rel)),
            tile_width=tile_w,
            tile_height=tile_h,
        ),
        colors=error_bar,
        labels=error_labels,
        font=bar_font,
    )
    improve_tile = add_horizontal_colorbar(
        image_from_array(
            colorize_improvement(
                record["err_d0"] - record["err_final"],
                display_valid,
                vlim=float(args.error_max_abs_rel),
            ),
            tile_width=tile_w,
            tile_height=tile_h,
        ),
        colors=improve_bar,
        labels=improve_labels,
        font=bar_font,
    )
    gt_depth_tile = add_horizontal_colorbar(
        image_from_array(
            colorize_depth(record["depth"], gt_valid, vmin=record["depth_vmin"], vmax=record["depth_vmax"]),
            tile_width=tile_w,
            tile_height=tile_h,
        ),
        colors=depth_bar,
        labels=depth_labels,
        font=bar_font,
    )

    tiles: list[tuple[str, str, Image.Image]] = [
        (
            "RGB input",
            "",
            image_from_array(np.clip(record["rgb"] * 255.0, 0.0, 255.0).round().astype(np.uint8), tile_width=tile_w, tile_height=tile_h),
        ),
        (
            "Pseudo-RAW preview",
            "p1..p99 display stretch",
            image_from_array(np.clip(record["raw_preview"] * 255.0, 0.0, 255.0).round().astype(np.uint8), tile_width=tile_w, tile_height=tile_h),
        ),
        ("RGB distribution", "min/p50/p99/max", rgb_hist),
        ("RAW distribution", "min/p50/p99/max", raw_hist),
        ("DAV2-S depth", d0_depth_metric, dav2_depth_tile),
        (f"Ours epoch{epoch:02d}", final_depth_metric, ours_depth_tile),
        ("Residual gate*delta", residual_range, residual_tile),
        ("Gate", "0..1", gate_tile),
        ("DAV2 error", error_range, dav2_error_tile),
        ("Ours error", error_range, ours_error_tile),
        ("Err improve +green", improve_range, improve_tile),
        ("GT depth", depth_range, gt_depth_tile),
    ]

    for i, (title, subtitle, tile) in enumerate(tiles):
        draw_tile(
            canvas,
            col=i % 4,
            row=i // 4,
            tile=tile,
            title=title,
            subtitle=subtitle,
            tile_width=tile_w,
            tile_height=tile_h,
            header_height=header_h,
            font=font,
            small_font=small_font,
        )

    return canvas, {
        "rgb_distribution": rgb_dist_meta,
        "raw_distribution": raw_dist_meta,
    }


def panel_manifest_record(
    record: dict[str, Any],
    *,
    panel_path: Path,
    order: int,
    residual_vlim: float,
    distribution_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "order": int(order),
        "dataset_index": int(record["dataset_index"]),
        "sample_name": str(record["sample_name"]),
        "image_path": str(record["image_path"]),
        "depth_path": str(record["depth_path"]),
        "panel_path": str(panel_path),
        "final": record["final"],
        "D0": record["D0"],
        "raw_stats": record["raw_stats"],
        "rgb_stats": record["rgb_stats"],
        "visualization": {
            "depth_cmap": "Spectral_r",
            "depth_range_scope": "per_panel_shared_by_gt_valid_percentiles",
            "depth_vmin": float(record["depth_vmin"]),
            "depth_vmax": float(record["depth_vmax"]),
            "depth_percentiles": [float(args.depth_pmin), float(args.depth_pmax)],
            "depth_tiles_share_range": ["GT depth", "DAV2-S depth", "Ours epoch"],
            "depth_colorbar": {
                "shown": True,
                "ticks": ["vmin", "midpoint", "vmax"],
                "unit": "meters",
            },
            "error_cmap": "magma",
            "error_vmin": 0.0,
            "error_vmax_abs_rel": float(args.error_max_abs_rel),
            "error_tiles_share_range": ["DAV2 error", "Ours error"],
            "error_colorbar": {
                "shown": True,
                "ticks": [0.0, 0.5 * float(args.error_max_abs_rel), float(args.error_max_abs_rel)],
            },
            "residual_cmap": "coolwarm",
            "residual_range_scope": "global_selected_samples_symmetric_p99_abs",
            "residual_vmin": -float(residual_vlim),
            "residual_vmax": float(residual_vlim),
            "residual_colorbar": {
                "shown": True,
                "ticks": [-float(residual_vlim), 0.0, float(residual_vlim)],
            },
            "gate_cmap": "viridis",
            "gate_vmin": 0.0,
            "gate_vmax": 1.0,
            "gate_colorbar": {
                "shown": True,
                "ticks": [0.0, 0.5, 1.0],
            },
            "improvement_colormap": "black_zero_green_positive_red_negative",
            "improvement_vmin_abs_rel": -float(args.error_max_abs_rel),
            "improvement_vmax_abs_rel": float(args.error_max_abs_rel),
            "improvement_colorbar": {
                "shown": True,
                "ticks": [-float(args.error_max_abs_rel), 0.0, float(args.error_max_abs_rel)],
            },
            "rgb_distribution": {
                "channels": ["R", "G", "B"],
                "x_range": [0.0, 1.0],
                "bins": int(args.hist_bins),
                "normalization": "per_channel_max",
                "stats": ["min", "p50", "p99", "max"],
                "channel_stats": distribution_meta["rgb_distribution"]["stats"],
            },
            "raw_distribution": {
                "channels": ["R", "Gr", "Gb", "B"],
                "x_range": [0.0, 1.0],
                "bins": int(args.hist_bins),
                "normalization": "per_channel_max",
                "stats": ["min", "p50", "p99", "max"],
                "channel_stats": distribution_meta["raw_distribution"]["stats"],
            },
        },
    }


def infer_epoch(checkpoint: Path) -> int:
    stem = checkpoint.stem
    if stem.startswith("epoch_"):
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return -1
    return -1


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(max(seconds, 0.0) + 0.5), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def write_per_sample_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
    handle.flush()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    logger = setup_logger(output_dir)

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided.")
    if args.max_panels < 0:
        raise ValueError("--max-panels must be non-negative.")

    config = load_run_config(run_dir)
    validate_run_config(config)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is not available.")

    logger.info("run_dir=%s", run_dir)
    logger.info("checkpoint=%s", checkpoint)
    logger.info("kitti_val_split=%s", Path(args.kitti_val_split).expanduser().resolve())
    logger.info("kitti_base=%s", Path(args.kitti_base).expanduser().resolve())
    logger.info("device=%s", device)
    logger.info(
        "semantic config: raw_storage_format=%s fullres_even_policy=%s rgb_input_space=%s "
        "depth_target_space=%s input_hw=(%s,%s) min_depth=%s max_depth=%s unprocessing_method=%s",
        config["raw_storage_format"],
        config["fullres_even_policy"],
        config["rgb_input_space"],
        config["depth_target_space"],
        config["input_height"],
        config["input_width"],
        config["min_depth"],
        config["max_depth"],
        config.get("unprocessing_method", "old_brooks_preset"),
    )

    dataset = KittiHalfresRawDataset(
        filelist_path=args.kitti_val_split,
        kitti_base=args.kitti_base,
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
        unprocessing_config=config,
    )
    assert_unprocessing_summary_matches_config(
        dataset.describe_unprocessing(),
        config,
        context="offline KITTI dataset vs training config",
    )
    if len(dataset) != 652:
        raise RuntimeError(f"Expected KITTI val length 652, got {len(dataset)}")
    logger.info("KITTI dataset length=%d", len(dataset))
    logger.info("KITTI source shape counts=%s", dataset.source_shape_counts)
    logger.info("KITTI unprocessing policy=%s", dataset.describe_unprocessing())

    model = build_model(config, checkpoint, device)
    amp_enabled = bool(config.get("amp", False)) and device.type == "cuda"
    amp_dtype = torch.float16 if str(config.get("amp_dtype", "bf16")) == "fp16" else torch.bfloat16
    epoch = infer_epoch(checkpoint)

    max_visit = len(dataset) if args.max_samples is None else min(int(args.max_samples), len(dataset))
    explicit_panel_indices = parse_indices(args.sample_indices, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=lambda batch: batch[0],
        persistent_workers=int(args.num_workers) > 0,
    )

    per_sample_path = output_dir / "per_sample.jsonl"
    rows: list[dict[str, Any]] = []
    ok_metric_rows: list[dict[str, Any]] = []
    start = time.time()
    logger.info("evaluation start max_visit=%d num_workers=%d", max_visit, int(args.num_workers))
    with per_sample_path.open("w", encoding="utf-8") as f:
        for visited, sample in enumerate(loader):
            if visited >= max_visit:
                break
            row, _ = collect_eval_for_sample(
                sample=sample,
                model=model,
                config=config,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                args=args,
                collect_panel=False,
            )
            rows.append(row)
            if row["status"] == "ok":
                ok_metric_rows.append(row)
            write_per_sample_row(f, row)
            if (visited + 1) % 50 == 0 or visited + 1 == max_visit:
                elapsed = time.time() - start
                logger.info(
                    "processed=%d/%d ok=%d elapsed=%s",
                    visited + 1,
                    max_visit,
                    len(ok_metric_rows),
                    format_seconds(elapsed),
                )

    if not ok_metric_rows:
        raise RuntimeError("KITTI offline eval produced zero ok samples.")

    if explicit_panel_indices is None:
        selected_panel_indices = select_uniform_indices(ok_metric_rows, int(args.max_panels))
        skipped_panel_records: list[dict[str, Any]] = []
    else:
        selected_panel_indices = explicit_panel_indices[: int(args.max_panels) if args.max_panels else len(explicit_panel_indices)]
        skipped_panel_records = []

    panel_records: list[dict[str, Any]] = []
    for idx in selected_panel_indices:
        sample = dataset[idx]
        row, panel_record = collect_eval_for_sample(
            sample=sample,
            model=model,
            config=config,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            args=args,
            collect_panel=True,
        )
        if row["status"] != "ok" or panel_record is None:
            skipped_panel_records.append(
                {
                    "dataset_index": int(idx),
                    "sample_name": row.get("sample_name"),
                    "status": row["status"],
                    "error": row.get("error"),
                }
            )
            continue
        panel_records.append(panel_record)

    if explicit_panel_indices is None and len(panel_records) != int(args.max_panels):
        raise RuntimeError(f"Expected {args.max_panels} uniform panels, collected {len(panel_records)}")

    residual_values = []
    for record in panel_records:
        values = np.abs(record["gate_delta"][np.isfinite(record["gate_delta"])])
        if values.size:
            residual_values.append(values.astype(np.float32))
    residual_vlim = float(np.percentile(np.concatenate(residual_values), 99.0)) if residual_values else 1.0
    residual_vlim = max(residual_vlim, 1e-6)

    panels_dir = output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    panel_manifest_records = []
    for order, record in enumerate(panel_records, start=1):
        safe_name = str(record["sample_name"]).replace("/", "_")
        panel_path = panels_dir / f"{order:02d}_kitti_{safe_name}_epoch{epoch:02d}_panel.jpg"
        panel, distribution_meta = make_panel(record, args, residual_vlim, epoch)
        panel.save(panel_path, quality=95)
        panel_manifest_records.append(
            panel_manifest_record(
                record,
                panel_path=panel_path,
                order=order,
                residual_vlim=residual_vlim,
                distribution_meta=distribution_meta,
                args=args,
            )
        )
        logger.info("wrote panel %s", panel_path)

    final_metrics = [row["final"] for row in ok_metric_rows]
    d0_metrics = [row["D0"] for row in ok_metric_rows]
    overall_final = average_metrics(final_metrics)
    overall_d0 = average_metrics(d0_metrics)
    delta = {
        "final_abs_rel_minus_D0_abs_rel": (
            None
            if overall_final["abs_rel"] is None or overall_d0["abs_rel"] is None
            else overall_final["abs_rel"] - overall_d0["abs_rel"]
        ),
        "final_d1_minus_D0_d1": (
            None
            if overall_final["d1"] is None or overall_d0["d1"] is None
            else overall_final["d1"] - overall_d0["d1"]
        ),
    }
    status_counts = Counter(str(row["status"]) for row in rows)
    elapsed_seconds = time.time() - start
    metrics_payload = {
        "dataset": "kitti_val_halfres_raw",
        "dataset_samples": int(len(dataset)),
        "visited_samples": int(len(rows)),
        "samples": int(len(ok_metric_rows)),
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
        "note": (
            "KITTI val is evaluated with min_depth=1.0 and canonical_even_pad_crop to match the "
            "VKITTI-trained fixed 187x621 RAW residual model; scores are not directly comparable "
            "to KITTI public benchmark settings."
        ),
        "geometry_policy": dataset.describe_geometry(),
        "validate_run_config_branch": str(config.get("unprocessing_method", "old_brooks_preset")),
        "unprocessing_policy_source": "training_config",
        "unprocessing_policy": dataset.describe_unprocessing(),
        "status_counts": dict(status_counts),
        "overall": {
            "final": overall_final,
            "D0": overall_d0,
            "delta": delta,
        },
        "elapsed_seconds": float(elapsed_seconds),
        "seconds_per_visited_sample": float(elapsed_seconds / max(len(rows), 1)),
        "panel_selection": {
            "mode": "explicit" if explicit_panel_indices is not None else args.panel_selection,
            "selected_dataset_indices": [int(record["dataset_index"]) for record in panel_records],
            "requested_sample_indices": explicit_panel_indices,
            "max_panels": int(args.max_panels),
        },
    }
    save_json(output_dir / "metrics.json", metrics_payload)

    panel_manifest = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "output_dir": str(panels_dir),
        "epoch": int(epoch),
        "selected_indices": [int(record["dataset_index"]) for record in panel_records],
        "skipped_panel_records": skipped_panel_records,
        "panel_layout": [
            "RGB input",
            "Pseudo-RAW preview",
            "RGB distribution",
            "RAW distribution",
            "DAV2-S depth",
            "Ours epoch09",
            "Residual gate*delta",
            "Gate",
            "DAV2 error",
            "Ours error",
            "Err improve +green",
            "GT depth",
        ],
        "visualization_defaults": {
            "rgb_distribution": {
                "channels": ["R", "G", "B"],
                "x_range": [0.0, 1.0],
                "bins": int(args.hist_bins),
                "normalization": "per_channel_max",
                "stats": ["min", "p50", "p99", "max"],
            },
            "raw_distribution": {
                "channels": ["R", "Gr", "Gb", "B"],
                "x_range": [0.0, 1.0],
                "bins": int(args.hist_bins),
                "normalization": "per_channel_max",
                "stats": ["min", "p50", "p99", "max"],
            },
            "depth_cmap": "Spectral_r",
            "depth_tiles_share_range": ["DAV2-S depth", "Ours epoch09", "GT depth"],
            "depth_colorbar": {"shown": True, "ticks": ["vmin", "midpoint", "vmax"], "unit": "meters"},
            "error_cmap": "magma",
            "error_tiles_share_range": ["DAV2 error", "Ours error"],
            "residual_cmap": "coolwarm",
            "residual_colorbar": {"shown": True, "ticks": ["-p99_abs", "0", "+p99_abs"]},
            "gate_cmap": "viridis",
            "gate_colorbar": {"shown": True, "ticks": [0.0, 0.5, 1.0]},
            "improvement_colormap": "black_zero_green_positive_red_negative",
            "improvement_colorbar": {
                "shown": True,
                "ticks": [-float(args.error_max_abs_rel), 0.0, float(args.error_max_abs_rel)],
            },
            "error_vmax_abs_rel": float(args.error_max_abs_rel),
            "residual_vmax_abs_p99": float(residual_vlim),
        },
        "records": panel_manifest_records,
    }
    save_json(panels_dir / "manifest.json", panel_manifest)
    logger.info(
        "done samples=%d ok=%d final_abs_rel=%.6f D0_abs_rel=%.6f delta_abs_rel=%+.6f final_d1=%.6f D0_d1=%.6f elapsed=%s",
        len(rows),
        len(ok_metric_rows),
        float(overall_final["abs_rel"]),
        float(overall_d0["abs_rel"]),
        float(delta["final_abs_rel_minus_D0_abs_rel"]),
        float(overall_final["d1"]),
        float(overall_d0["d1"]),
        format_seconds(elapsed_seconds),
    )


if __name__ == "__main__":
    main()
