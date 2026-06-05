#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from anqi_eval.raw_audit_common import (
    build_image_edge_band,
    build_labeled_grid,
    colorize_depth,
    colorize_error,
    colorize_scalar,
    denorm_rgb_tensor,
    mask_to_rgb,
    raw_tensor_to_preview,
)
from depth_anything_v2.dpt import DepthAnythingV2
from finetune_stf.dataset.raw_storage import get_raw_storage_spec
from finetune_stf.dataset.raw_utils import (
    DEFAULT_RAW_NPZ_ROOT,
    RECTIFIED_BAYER_KEY,
    decode_stf_raw_by_storage_format,
    load_rectified_bayer_npz,
)
from finetune_stf.dataset.stf import DEFAULT_STF_ROOT, _load_depth_npz
from finetune_stf.models.spatial_adapter import CenterPadCropAdapter
from finetune_stf.util.loss import build_training_target, robust_normalize_target_per_sample
from foundation.engine.transforms import packed_bayer_to_base_rgb
from foundation.tools.residual_training_common import (
    METRIC_KEYS,
    average_dicts,
    resolve_model_state,
    save_json,
    strip_module_prefix,
)
from foundation.tools.train_led_hb_incremental_residual import (
    MODEL_CONFIGS,
    add_dataset_raw_donor_if_needed,
    build_model as build_nseries_model,
    feature_ablation_active,
    feature_ablation_mode,
    forward_incremental_model,
    load_incremental_checkpoint,
)
from foundation.tools.train_led_hb_residual_control import build_dav2_residual_control_model


DEFAULT_D0_CKPT = "/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "plans/0531_led_night/stf_val"
DEFAULT_MODEL_HW = (512, 960)
RAW_FORMAT = "legacy_bggR_decomp16"
RAW_STORAGE_CHANNEL_ORDER = ("B", "Gr", "Gb", "R")
RAW_MODEL_CHANNEL_ORDER = ("R", "Gr", "Gb", "B")
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
MIN_ALIGN_POINTS = 128


RUNS = {
    "D0": {
        "kind": "d0",
        "run_dir": str(DEFAULT_OUTPUT_ROOT / "d0_reference"),
        "checkpoint": DEFAULT_D0_CKPT,
    },
    "C2": {
        "kind": "c2",
        "run_dir": str(PROJECT_ROOT / "finetune_stf/exp/0531_2341_led_hb_c2_d0only_residual_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e20"),
        "checkpoint": "/mnt/drive/3333_raw/0000_exp_ckpt/0531_2341_led_hb_c2_d0only_residual_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e20/best_abs_rel.pth",
    },
    "N5": {
        "kind": "nseries",
        "run_dir": str(PROJECT_ROOT / "finetune_stf/exp/0601_0355_led_hb_n5_d1_lp0p5_q0p3_lfl0p0_rftnot_applicable_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10"),
        "checkpoint": "/mnt/drive/3333_raw/0000_exp_ckpt/0601_0355_led_hb_n5_d1_lp0p5_q0p3_lfl0p0_rftnot_applicable_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10/best_abs_rel.pth",
    },
    "N3": {
        "kind": "nseries",
        "run_dir": str(PROJECT_ROOT / "finetune_stf/exp/0601_0550_led_hb_n3_rgb_lp0p5_q0p3_lfl0p0_rftnot_applicable_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10"),
        "checkpoint": "/mnt/drive/3333_raw/0000_exp_ckpt/0601_0550_led_hb_n3_rgb_lp0p5_q0p3_lfl0p0_rftnot_applicable_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10/best_abs_rel.pth",
    },
    "N2": {
        "kind": "nseries",
        "run_dir": str(PROJECT_ROOT / "finetune_stf/exp/0601_0751_led_hb_n2_x3_lp0p8_q0p3_lfl0p0_rfttrue_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10"),
        "checkpoint": "/mnt/drive/3333_raw/0000_exp_ckpt/0601_0751_led_hb_n2_x3_lp0p8_q0p3_lfl0p0_rfttrue_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10/best_abs_rel.pth",
    },
    "N7": {
        "kind": "nseries",
        "run_dir": str(PROJECT_ROOT / "finetune_stf/exp/0601_1005_led_hb_n7_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10"),
        "checkpoint": "/mnt/drive/3333_raw/0000_exp_ckpt/0601_1005_led_hb_n7_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10/best_abs_rel.pth",
    },
}

STF_META_KEYS = (
    "official_split",
    "is_adverse_weather",
    "daytime",
    "fog",
    "precipitation",
    "road_state",
    "infrastructure",
    "tunnel",
    "cfa_pattern",
    "raw_representation",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def manifest_path(stf_root: Path, split: str, manifest: str | None = None) -> Path:
    if manifest:
        return Path(manifest).expanduser().resolve()
    return stf_root / "manifests" / f"stf_raw_depth_v1_{split}.csv"


def resolve_data_path(stf_root: Path, value: str) -> Path:
    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        return path
    return (stf_root / path).resolve()


def load_manifest_rows(stf_root: Path, split: str, manifest: str | None = None) -> list[dict[str, Any]]:
    path = manifest_path(stf_root, split, manifest)
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"filename_stem", "lut_preview", "lidar_proj_left"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row in reader:
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"No STF rows found in {path}")
    return rows


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    return (rgb.astype(np.float32, copy=False) - IMAGE_MEAN) / IMAGE_STD


def resize_rgb(rgb: np.ndarray, hw: tuple[int, int], interpolation: int) -> np.ndarray:
    h, w = hw
    if tuple(rgb.shape[:2]) == (h, w):
        return rgb.astype(np.float32, copy=False)
    return cv2.resize(rgb, (w, h), interpolation=interpolation).astype(np.float32, copy=False)


def resize_sparse_depth(depth: np.ndarray, valid: np.ndarray, hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = hw
    if tuple(depth.shape[:2]) == (h, w):
        return depth.astype(np.float32, copy=False), valid.astype(bool, copy=False)
    depth_r = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32, copy=False)
    valid_r = cv2.resize(valid.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
    depth_r = np.where(valid_r, depth_r, 0.0).astype(np.float32, copy=False)
    return depth_r, valid_r


def read_lut_preview(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read STF lut_preview: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def tensor_chw(array_hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.transpose(array_hwc, (2, 0, 1))).astype(np.float32))


def raw_path_for(raw_root: Path, sample_name: str) -> Path:
    return (raw_root / f"{sample_name}.npz").resolve()


def load_decoded_raw(raw_path: Path, raw_storage_format: str) -> tuple[np.ndarray, np.ndarray]:
    storage = load_rectified_bayer_npz(raw_path, key=RECTIFIED_BAYER_KEY)
    decoded = decode_stf_raw_by_storage_format(storage, raw_storage_format)
    return storage, decoded.astype(np.float32, copy=False)


class STFLEDEvalDataset(Dataset):
    def __init__(
        self,
        *,
        stf_root: str | Path = DEFAULT_STF_ROOT,
        raw_npz_root: str | Path = DEFAULT_RAW_NPZ_ROOT,
        split: str = "val",
        manifest: str | None = None,
        model_input_hw: tuple[int, int] = DEFAULT_MODEL_HW,
        min_depth: float = 1.0,
        max_depth: float = 80.0,
        include_raw: bool = False,
        raw_storage_format: str = RAW_FORMAT,
        image_resize: str = "area",
        max_samples: int | None = None,
        sample_indices: list[int] | None = None,
        model_valid_mask_source: str = "sparse_downsampled",
    ) -> None:
        self.stf_root = Path(stf_root).expanduser().resolve()
        self.raw_npz_root = Path(raw_npz_root).expanduser().resolve()
        self.split = str(split)
        self.model_input_hw = tuple(int(v) for v in model_input_hw)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.include_raw = bool(include_raw)
        self.raw_storage_format = str(raw_storage_format)
        self.model_valid_mask_source = str(model_valid_mask_source)
        if self.raw_storage_format != RAW_FORMAT:
            raise ValueError(f"Formal STF RAW eval requires raw_storage_format={RAW_FORMAT!r}")
        if self.model_valid_mask_source not in ("sparse_downsampled", "all_ones"):
            raise ValueError("model_valid_mask_source must be sparse_downsampled or all_ones")
        if image_resize == "area":
            self.image_interpolation = cv2.INTER_AREA
        elif image_resize == "cubic":
            self.image_interpolation = cv2.INTER_CUBIC
        else:
            raise ValueError("--image-resize must be area or cubic")
        self.image_resize = str(image_resize)
        rows = load_manifest_rows(self.stf_root, self.split, manifest)
        if sample_indices:
            selected_rows = []
            selected_indices = []
            for index in sample_indices:
                index = int(index)
                if index < 0 or index >= len(rows):
                    raise IndexError(f"sample index {index} is outside STF split length {len(rows)}")
                selected_rows.append(rows[index])
                selected_indices.append(index)
            rows = selected_rows
            self.row_indices = selected_indices
        else:
            self.row_indices = list(range(len(rows)))
        self.rows = rows
        if max_samples is not None:
            self.rows = self.rows[: int(max_samples)]
            self.row_indices = self.row_indices[: int(max_samples)]

    def __len__(self) -> int:
        return len(self.rows)

    def describe_geometry(self) -> dict[str, Any]:
        return {
            "dataset": "STF",
            "split": self.split,
            "model_input_hw": list(self.model_input_hw),
            "image_source": "lut_preview",
            "image_native_hw": [1024, 1920],
            "image_resize": f"2x_{self.image_resize}_resize_to_model_input_hw",
            "raw_packed_hw": [512, 960],
            "raw_storage_format": self.raw_storage_format if self.include_raw else "not_applicable",
            "raw_image_hw_contract": "image.HW == raw.HW == model_input_hw" if self.include_raw else "image.HW == model_input_hw",
            "eval_depth_source": "sparse_lidar_full_resolution",
            "metric_protocol": "resize_prediction_to_sparse_gt_hw_then_per_image_affine_disp",
            "model_valid_mask_source": self.model_valid_mask_source,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        source_index = int(self.row_indices[idx])
        sample_name = row["filename_stem"]
        image_path = resolve_data_path(self.stf_root, row["lut_preview"])
        depth_path = resolve_data_path(self.stf_root, row["lidar_proj_left"])
        rgb_full = read_lut_preview(image_path)
        rgb_model = resize_rgb(rgb_full, self.model_input_hw, self.image_interpolation)
        image_norm = normalize_rgb(rgb_model)

        eval_depth = _load_depth_npz(depth_path)
        eval_valid = np.isfinite(eval_depth) & (eval_depth >= self.min_depth) & (eval_depth <= self.max_depth)
        model_depth, model_valid = resize_sparse_depth(eval_depth, eval_valid, self.model_input_hw)
        if self.model_valid_mask_source == "all_ones":
            model_valid = np.ones(self.model_input_hw, dtype=bool)

        sample: dict[str, Any] = {
            "image": tensor_chw(image_norm),
            "rgb_preview": tensor_chw(rgb_model),
            "depth": torch.from_numpy(model_depth.astype(np.float32, copy=False)),
            "valid_mask": torch.from_numpy(model_valid.astype(bool, copy=False)),
            "eval_depth": torch.from_numpy(eval_depth.astype(np.float32, copy=False)),
            "eval_valid_mask": torch.from_numpy(eval_valid.astype(bool, copy=False)),
            "sample_name": sample_name,
            "dataset_index": source_index,
            "image_path": str(image_path),
            "depth_path": str(depth_path),
            "raw_path": str(raw_path_for(self.raw_npz_root, sample_name)),
            "raw_storage_format": self.raw_storage_format if self.include_raw else "not_applicable",
        }
        for key in STF_META_KEYS:
            sample[key] = str(row.get(key, ""))

        if self.include_raw:
            raw_path = raw_path_for(self.raw_npz_root, sample_name)
            if not raw_path.is_file():
                raise FileNotFoundError(f"Missing STF RAW NPZ: {raw_path}")
            storage, raw = load_decoded_raw(raw_path, self.raw_storage_format)
            if tuple(raw.shape[:2]) != self.model_input_hw:
                raise RuntimeError(
                    f"Decoded RAW HW {tuple(raw.shape[:2])} does not match model_input_hw={self.model_input_hw}"
                )
            if tuple(raw.shape[:2]) != tuple(rgb_model.shape[:2]):
                raise RuntimeError(f"image/raw HW mismatch: image={rgb_model.shape[:2]} raw={raw.shape[:2]}")
            sample["raw"] = tensor_chw(raw)
            sample["raw_storage_shape"] = list(storage.shape)
            sample["raw_storage_dtype"] = str(storage.dtype)
            sample["raw_storage_min"] = float(np.min(storage))
            sample["raw_storage_max"] = float(np.max(storage))
            sample["raw_decoded_min"] = float(np.min(raw))
            sample["raw_decoded_max"] = float(np.max(raw))
        return sample


def collate_eval(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("STF eval currently uses batch_size=1 to keep full-resolution sparse GT explicit.")
    return batch[0]


def build_loader(dataset: Dataset) -> DataLoader:
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)


def load_dav2(encoder: str, checkpoint: Path) -> DepthAnythingV2:
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(strip_module_prefix(resolve_model_state(ckpt)), strict=True)
    return model


def load_c2_model(config: dict[str, Any], checkpoint: Path, model_hw: tuple[int, int], device: torch.device) -> torch.nn.Module:
    base = DepthAnythingV2(**MODEL_CONFIGS[str(config.get("encoder", "vits"))])
    model = build_dav2_residual_control_model(
        base,
        residual_feature_source=str(config.get("residual_feature_source", "d0")),
        residual_alpha=float(config.get("residual_alpha", 0.5)),
        d0_sign=int(config.get("d0_sign", 1)),
        sensor_hw=model_hw,
        backbone_hw=None,
    )
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(strip_module_prefix(resolve_model_state(ckpt)), strict=True)
    return model.to(device).eval()


def nseries_args(config: dict[str, Any], model_hw: tuple[int, int], max_depth: float, feature_mode: str) -> argparse.Namespace:
    cfg = dict(config)
    cfg["input_height"] = int(model_hw[0])
    cfg["input_width"] = int(model_hw[1])
    cfg["min_depth"] = 1.0
    cfg["max_depth"] = float(max_depth)
    cfg["eval_only"] = True
    cfg["eval_feature_ablation_mode"] = str(feature_mode)
    cfg["feature_ablation_scope"] = str(cfg.get("feature_ablation_scope", "both"))
    cfg["feature_ablation_key"] = str(cfg.get("feature_ablation_key", "x3"))
    cfg["feature_ablation_seed"] = int(cfg.get("feature_ablation_seed", 42))
    cfg["feature_ablation_donor_offset"] = int(cfg.get("feature_ablation_donor_offset", 1))
    cfg["train_feature_ablation_mode"] = "true"
    cfg["bs"] = 1
    cfg["amp"] = bool(cfg.get("amp", True))
    return argparse.Namespace(**cfg)


def load_nseries_model(
    config: dict[str, Any],
    checkpoint: Path,
    model_hw: tuple[int, int],
    max_depth: float,
    feature_mode: str,
    device: torch.device,
) -> tuple[torch.nn.Module, argparse.Namespace]:
    args = nseries_args(config, model_hw, max_depth, feature_mode)
    model = build_nseries_model(args)
    load_incremental_checkpoint(model, checkpoint, strict=True)
    return model.to(device).eval(), args


def resize_prediction_to_eval(pred: np.ndarray, eval_hw: tuple[int, int]) -> np.ndarray:
    if tuple(pred.shape[:2]) == tuple(eval_hw):
        return pred.astype(np.float32, copy=False)
    h, w = eval_hw
    return cv2.resize(pred.astype(np.float32, copy=False), (w, h), interpolation=cv2.INTER_LINEAR)


def align_and_metrics(
    *,
    gt: np.ndarray,
    pred_disp: np.ndarray,
    valid: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray | None, dict[str, Any] | None, dict[str, float] | None]:
    if int(valid.sum()) < MIN_ALIGN_POINTS:
        return None, None, None
    aligned, stats = affine_align_disp(gt, pred_disp, valid)
    metrics = compute_metrics(gt, aligned, valid, min_depth=min_depth, max_depth=max_depth)
    if metrics is None:
        return aligned, stats, None
    return aligned, stats, {key: float(metrics[key]) for key in metrics}


def region_abs_rel(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float | None:
    valid = mask & np.isfinite(pred) & (pred > 0) & np.isfinite(gt) & (gt > 0)
    if int(valid.sum()) < 10:
        return None
    return float(np.mean(np.abs(pred[valid] - gt[valid]) / np.clip(gt[valid], 1e-6, None)))


def top_fraction_mask(values: np.ndarray, valid: np.ndarray, fraction: float) -> np.ndarray:
    out = np.zeros_like(valid, dtype=bool)
    vals = values[valid & np.isfinite(values)]
    if vals.size < 10:
        return out
    threshold = float(np.quantile(vals, 1.0 - float(fraction)))
    return valid & np.isfinite(values) & (values >= threshold)


def build_region_masks(
    *,
    gt: np.ndarray,
    valid: np.ndarray,
    rgb_full: np.ndarray,
    aligned_d0: np.ndarray | None,
    aligned_d1: np.ndarray | None,
    min_depth: float,
    max_depth: float,
) -> dict[str, np.ndarray]:
    luma = 0.2126 * rgb_full[..., 0] + 0.7152 * rgb_full[..., 1] + 0.0722 * rgb_full[..., 2]
    finite_luma = luma[np.isfinite(luma)]
    dark_threshold = float(np.quantile(finite_luma, 0.20)) if finite_luma.size else 0.0
    edge = build_image_edge_band((rgb_full * 255.0).astype(np.uint8), low=80, high=160, dilate=5)
    masks = {
        "near_1_20": valid & (gt >= max(min_depth, 1.0)) & (gt <= 20.0),
        "mid_20_50": valid & (gt > 20.0) & (gt <= 50.0),
        "far_50_80": valid & (gt > 50.0) & (gt <= max_depth),
        "dark_q20": valid & (luma <= dark_threshold),
        "saturated": valid & (np.max(rgb_full, axis=-1) >= 0.95),
        "image_edge_band": valid & edge,
    }
    if aligned_d0 is not None:
        d0_err = np.zeros_like(gt, dtype=np.float32)
        d0_valid = valid & np.isfinite(aligned_d0) & (aligned_d0 > 0) & (gt > 0)
        d0_err[d0_valid] = np.abs(aligned_d0[d0_valid] - gt[d0_valid]) / np.clip(gt[d0_valid], 1e-6, None)
        masks["d0_high_error"] = top_fraction_mask(d0_err, d0_valid, 0.20)
    if aligned_d1 is not None:
        d1_err = np.zeros_like(gt, dtype=np.float32)
        d1_valid = valid & np.isfinite(aligned_d1) & (aligned_d1 > 0) & (gt > 0)
        d1_err[d1_valid] = np.abs(aligned_d1[d1_valid] - gt[d1_valid]) / np.clip(gt[d1_valid], 1e-6, None)
        masks["d1_high_error"] = top_fraction_mask(d1_err, d1_valid, 0.20)
    return masks


def per_region_metrics(gt: np.ndarray, aligned: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, mask in masks.items():
        out[name] = {
            "abs_rel": region_abs_rel(gt, aligned, mask),
            "point_count": int(mask.sum()),
            "has_region": bool(mask.sum() > 0),
        }
    return out


def aggregate_region(rows: list[dict[str, Any]], output_names: list[str]) -> dict[str, Any]:
    region_names = sorted({name for row in rows for out in output_names for name in row.get("region", {}).get(out, {})})
    result: dict[str, Any] = {}
    for out in output_names:
        out_result = {}
        for region in region_names:
            vals = [
                row["region"][out][region]["abs_rel"]
                for row in rows
                if row.get("status") == "ok"
                and out in row.get("region", {})
                and region in row["region"][out]
                and row["region"][out][region]["abs_rel"] is not None
            ]
            point_count = sum(
                int(row["region"][out][region]["point_count"])
                for row in rows
                if out in row.get("region", {}) and region in row["region"][out]
            )
            image_count = sum(
                1
                for row in rows
                if out in row.get("region", {})
                and region in row["region"][out]
                and bool(row["region"][out][region]["has_region"])
            )
            out_result[region] = {
                "abs_rel": float(np.mean(vals)) if vals else None,
                "point_count": int(point_count),
                "image_count_with_region": int(image_count),
            }
        result[out] = out_result
    return result


def aggregate_scene(rows: list[dict[str, Any]], output_names: list[str]) -> dict[str, Any]:
    keys = ("daytime", "is_adverse_weather", "fog", "precipitation", "road_state", "tunnel")
    result: dict[str, Any] = {}
    for key in keys:
        by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("status") == "ok":
                by_value[str(row.get("meta", {}).get(key, "unknown") or "unknown")].append(row)
        result[key] = {}
        for value, items in sorted(by_value.items()):
            item_summary = {"num_images": len(items)}
            for out in output_names:
                metrics = [row[out] for row in items if out in row]
                item_summary[out] = average_dicts(metrics, METRIC_KEYS) if metrics else {}
                item_summary[f"{out}_point_count"] = int(sum(row[out].get("valid_eval_pixels", 0) for row in items if out in row))
            result[key][value] = item_summary
    return result


def subtract_metric_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float | None]:
    return {key: None if key not in a or key not in b else float(a[key]) - float(b[key]) for key in METRIC_KEYS}


def output_names_for_method(method: str) -> list[str]:
    if method == "D0":
        return ["D0"]
    if method == "C2":
        return ["final", "D0"]
    return ["final", "D1", "D0"]


def summarize_rows(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    outputs = output_names_for_method(method)
    overall = {out: average_dicts([row[out] for row in ok if out in row], METRIC_KEYS) for out in outputs}
    for out in outputs:
        overall[out]["avg_valid_eval_pixels"] = float(np.mean([row[out].get("valid_eval_pixels", 0) for row in ok if out in row])) if ok else 0.0
        overall[out]["total_valid_eval_pixels"] = int(sum(row[out].get("valid_eval_pixels", 0) for row in ok if out in row))
    if method == "C2":
        overall["delta_final_minus_D0"] = subtract_metric_dicts(overall["final"], overall["D0"])
    elif method not in ("D0",):
        overall["delta_final_minus_D1"] = subtract_metric_dicts(overall["final"], overall["D1"])
        overall["delta_D1_minus_D0"] = subtract_metric_dicts(overall["D1"], overall["D0"])
    return {
        "num_images": len(ok),
        "num_skipped_images": len(rows) - len(ok),
        "overall": overall,
        "region": aggregate_region(ok, outputs),
        "scene": aggregate_scene(ok, outputs),
    }


def cv2_colorize_disp(disp: np.ndarray) -> np.ndarray:
    valid = np.isfinite(disp)
    arr = np.zeros_like(disp, dtype=np.float32)
    if valid.any():
        lo, hi = np.percentile(disp[valid], [1, 99])
        if hi <= lo:
            hi = lo + 1e-6
        arr = np.clip((disp - lo) / (hi - lo), 0.0, 1.0)
    return cv2.cvtColor(cv2.applyColorMap((arr * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def dilate_sparse_display(rgb: np.ndarray, valid_mask: np.ndarray, *, kernel_size: int = 9) -> np.ndarray:
    """Enlarge sparse colored points for visualization only; metrics remain unchanged."""
    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    valid = np.asarray(valid_mask, dtype=bool)
    if kernel_size <= 1 or not valid.any():
        return rgb_u8
    kernel = np.ones((int(kernel_size), int(kernel_size)), np.uint8)
    dilated_rgb = cv2.dilate(rgb_u8, kernel, iterations=1)
    dilated_mask = cv2.dilate(valid.astype(np.uint8), kernel, iterations=1) > 0
    out = np.zeros_like(rgb_u8)
    out[dilated_mask] = dilated_rgb[dilated_mask]
    return out


def save_review_panel(
    *,
    path: Path,
    sample: dict[str, Any],
    outputs: dict[str, np.ndarray],
    aligned: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    raw_tensor: torch.Tensor | None,
    raw_storage: np.ndarray | None,
    edge_mask: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> None:
    rgb_model = denorm_rgb_tensor(sample["image"])
    gt = sample["eval_depth"].numpy().astype(np.float32)
    valid = sample["eval_valid_mask"].numpy().astype(bool)
    vmin, vmax = min_depth, min(max_depth, 80.0)
    panels: list[tuple[str, np.ndarray]] = [
        ("lut_preview model RGB", (rgb_model * 255.0).round().astype(np.uint8)),
        ("sparse lidar mask (display dilated)", mask_to_rgb(valid, dilate=9)),
        ("image edge diagnostic", mask_to_rgb(edge_mask, dilate=1)),
        (
            "sparse lidar GT (display dilated)",
            dilate_sparse_display(colorize_depth(gt, valid, vmin=vmin, vmax=vmax), valid, kernel_size=9),
        ),
    ]
    if raw_storage is not None:
        for idx, name in enumerate(("storage c0", "storage c1", "storage c2", "storage c3")):
            panels.append((name, colorize_scalar(raw_storage[..., idx], vmax_pct=99.5)))
    if raw_tensor is not None:
        raw_preview = raw_tensor_to_preview(raw_tensor)
        panels.append(("decoded raw base_rgb", raw_preview))
    for name, pred in outputs.items():
        panels.append((f"{name} disp", cv2_colorize_disp(pred)))
        if name in aligned:
            metric = metrics.get(name, {}).get("abs_rel")
            label = f"{name} aligned abs_rel={metric:.3f}" if metric is not None else f"{name} aligned"
            panels.append((label, colorize_depth(aligned[name], np.isfinite(aligned[name]) & (aligned[name] > 0), vmin=vmin, vmax=vmax)))
            err = np.zeros_like(gt, dtype=np.float32)
            ok = valid & np.isfinite(aligned[name]) & (aligned[name] > 0) & (gt > 0)
            err[ok] = np.abs(aligned[name][ok] - gt[ok]) / np.clip(gt[ok], 1e-6, None)
            panels.append((
                f"{name} sparse rel error (display dilated)",
                dilate_sparse_display(colorize_error(err, ok, vmax=0.75), ok, kernel_size=9),
            ))
    footer = f"{sample['sample_name']} | {sample.get('raw_storage_format', 'not_applicable')}"
    grid = build_labeled_grid(panels, cols=3, tile_hw=(256, 480), footer=footer)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def evaluate_method(args: argparse.Namespace) -> None:
    method = str(args.method).upper()
    if method not in RUNS:
        raise ValueError(f"Unsupported method {method!r}; expected one of {sorted(RUNS)}")
    run = RUNS[method]
    run_dir = Path(args.run_dir or run["run_dir"]).expanduser().resolve()
    checkpoint = Path(args.checkpoint or run["checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")
    model_hw = (int(args.input_height), int(args.input_width))
    include_raw = method in ("N2", "N7")
    dataset = STFLEDEvalDataset(
        stf_root=args.stf_root,
        raw_npz_root=args.raw_npz_root,
        split=args.split,
        manifest=args.manifest_path,
        model_input_hw=model_hw,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        include_raw=include_raw,
        raw_storage_format=args.raw_storage_format,
        image_resize=args.image_resize,
        max_samples=args.max_samples,
        sample_indices=args.sample_indices,
        model_valid_mask_source=args.model_valid_mask_source,
    )
    loader = build_loader(dataset)

    config: dict[str, Any]
    if method == "D0":
        config = {"encoder": args.encoder, "d0_sign": args.d0_sign}
        model = load_dav2(args.encoder, checkpoint).to(device).eval()
        adapter = CenterPadCropAdapter(sensor_hw=model_hw, backbone_hw=None).to(device)
        model_args = None
    else:
        config = load_json(run_dir / "config.json")
        if run["kind"] == "c2":
            model = load_c2_model(config, checkpoint, model_hw, device)
            model_args = None
        else:
            model, model_args = load_nseries_model(
                config,
                checkpoint,
                model_hw,
                args.max_depth,
                args.feature_ablation_mode,
                device,
            )

    rows: list[dict[str, Any]] = []
    review_dir = Path(args.review_dir).expanduser().resolve() if args.review_dir else None
    review_indices = {int(v) for v in args.review_indices}
    start = time.time()
    with torch.no_grad():
        for sample in loader:
            idx = int(sample["dataset_index"])
            image = sample["image"].unsqueeze(0).to(device).float()
            model_valid = sample["valid_mask"].unsqueeze(0).to(device).bool()
            gt = sample["eval_depth"].numpy().astype(np.float32)
            valid = sample["eval_valid_mask"].numpy().astype(bool)
            row: dict[str, Any] = {
                "dataset_index": idx,
                "sample_name": sample["sample_name"],
                "status": "ok",
                "meta": {key: sample.get(key, "") for key in STF_META_KEYS},
                "image_path": sample["image_path"],
                "depth_path": sample["depth_path"],
                "raw_path": sample["raw_path"],
            }
            if int(valid.sum()) < MIN_ALIGN_POINTS:
                row["status"] = "skipped_too_few_sparse_points"
                rows.append(row)
                continue

            outputs: dict[str, np.ndarray] = {}
            raw_tensor = None
            raw_storage = None
            if method == "D0":
                pred = int(args.d0_sign) * adapter.crop_depth(model(adapter.pad_rgb(image)))[0]
                outputs["D0"] = pred.float().detach().cpu().numpy().astype(np.float32)
            elif method == "C2":
                out = model({"image": image, "valid_mask": model_valid})
                outputs["final"] = out["pred"][0].float().detach().cpu().numpy().astype(np.float32)
                outputs["D0"] = (float(config.get("d0_sign", 1)) * out["D0"][0].float()).detach().cpu().numpy().astype(np.float32)
            else:
                raw = sample.get("raw")
                if raw is not None:
                    raw_tensor = raw
                    raw = raw.unsqueeze(0).to(device).float()
                model_batch = {"image": image, "valid_mask": model_valid}
                if raw is not None:
                    model_batch["raw"] = raw
                add_dataset_raw_donor_if_needed(
                    model_batch=model_batch,
                    dataset=dataset,
                    sample_indices=[idx],
                    args=model_args,
                    device=device,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(args.amp and device.type == "cuda")):
                    out = forward_incremental_model(model, model_batch, model_args, phase="eval")
                outputs["final"] = out["pred"][0].float().detach().cpu().numpy().astype(np.float32)
                outputs["D1"] = out["D1_norm"][0].float().detach().cpu().numpy().astype(np.float32)
                outputs["D0"] = (float(config.get("d0_sign", 1)) * out["D0"][0].float()).detach().cpu().numpy().astype(np.float32)
                row["feature_ablation_mode"] = feature_ablation_mode(model_args, phase="eval")
                row["feature_ablation_scope"] = str(getattr(model_args, "feature_ablation_scope", "both"))
                row["feature_ablation_key"] = str(getattr(model_args, "feature_ablation_key", "x3"))
                row["feature_ablation_applied"] = bool(feature_ablation_active(model_args, phase="eval"))
                if "feature_ablation_donor_indices" in model_batch:
                    row["feature_ablation_donor_indices"] = list(model_batch["feature_ablation_donor_indices"])
                if method in ("N2", "N7"):
                    if tuple(image.shape[-2:]) != tuple(raw.shape[-2:]):
                        raise RuntimeError(f"image/raw HW mismatch for {sample['sample_name']}")
                    row["raw_decoded_min"] = float(sample["raw_decoded_min"])
                    row["raw_decoded_max"] = float(sample["raw_decoded_max"])
                    raw_storage, _ = load_decoded_raw(Path(sample["raw_path"]), args.raw_storage_format)

            aligned_outputs: dict[str, np.ndarray] = {}
            metrics_outputs: dict[str, dict[str, float]] = {}
            for name, pred_model_hw in outputs.items():
                pred_eval = resize_prediction_to_eval(pred_model_hw, gt.shape[:2])
                outputs[name] = pred_eval
                aligned, align_stats, metrics = align_and_metrics(
                    gt=gt,
                    pred_disp=pred_eval,
                    valid=valid,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                )
                row[f"{name}_align_stats"] = align_stats
                if metrics is None or aligned is None:
                    row["status"] = "skipped_metric_failure"
                    break
                row[name] = metrics
                aligned_outputs[name] = aligned
                metrics_outputs[name] = metrics
            if row["status"] == "ok":
                rgb_full = read_lut_preview(Path(sample["image_path"]))
                aligned_d1_for_regions = aligned_outputs.get("D1")
                if aligned_d1_for_regions is None and method == "C2":
                    aligned_d1_for_regions = aligned_outputs.get("final")
                masks = build_region_masks(
                    gt=gt,
                    valid=valid,
                    rgb_full=rgb_full,
                    aligned_d0=aligned_outputs.get("D0"),
                    aligned_d1=aligned_d1_for_regions,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                )
                row["region"] = {
                    name: per_region_metrics(gt, aligned, masks)
                    for name, aligned in aligned_outputs.items()
                }
                if review_dir is not None and idx in review_indices:
                    panel_label = method
                    if method in ("N2", "N7"):
                        panel_label = f"{method}_{args.feature_ablation_mode}"
                    edge_full = build_image_edge_band((rgb_full * 255.0).astype(np.uint8), low=80, high=160, dilate=5)
                    save_review_panel(
                        path=review_dir / f"{idx:05d}_{sample['sample_name']}_{panel_label}.png",
                        sample=sample,
                        outputs=outputs,
                        aligned=aligned_outputs,
                        metrics=metrics_outputs,
                        raw_tensor=raw_tensor,
                        raw_storage=raw_storage,
                        edge_mask=edge_full,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                    )
            rows.append(row)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(method, rows)
    raw_spec = get_raw_storage_spec(args.raw_storage_format)
    resolved_config = {
        "method": method,
        "run_kind": run["kind"],
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "stf_root": str(Path(args.stf_root).expanduser().resolve()),
        "raw_npz_root": str(Path(args.raw_npz_root).expanduser().resolve()),
        "split": args.split,
        "manifest_path": str(manifest_path(Path(args.stf_root).expanduser().resolve(), args.split, args.manifest_path)),
        "model_input_hw": list(model_hw),
        "image_source": "lut_preview",
        "image_resize": args.image_resize,
        "model_valid_mask_source": args.model_valid_mask_source,
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "eval_protocol": "sparse_lidar_per_image_affine_disp_resize_prediction_to_gt_hw",
        "precision": "amp_bf16" if bool(args.amp and device.type == "cuda") else "fp32",
        "raw_storage_format": args.raw_storage_format if include_raw else "not_applicable",
        "raw_storage_channel_order": list(raw_spec.storage_channel_order) if include_raw else ["not_applicable"],
        "raw_model_channel_order": list(raw_spec.model_channel_order) if include_raw else ["not_applicable"],
        "raw_decompand": raw_spec.decompand if include_raw else "not_applicable",
        "raw_post_decode_norm": raw_spec.post_decode_norm if include_raw else "not_applicable",
        "feature_ablation_mode": args.feature_ablation_mode if include_raw else "not_applicable",
        "dataset_geometry": dataset.describe_geometry(),
    }
    summary.update(
        {
            "method": method,
            "elapsed_seconds": float(time.time() - start),
            "resolved_config": resolved_config,
        }
    )
    save_json(output_dir / "metrics.json", summary)
    save_json(output_dir / "resolved_config.json", resolved_config)
    save_json(output_dir / "region_metrics.json", summary["region"])
    with (output_dir / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[eval] wrote {output_dir / 'metrics.json'}")


def corrcoef_safe(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    if mask is not None:
        a = a[mask]
        b = b[mask]
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 10:
        return None
    a = a[finite]
    b = b[finite]
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def quantiles(arr: np.ndarray) -> dict[str, float]:
    vals = np.asarray(arr, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {}
    return {f"q{q:g}": float(np.quantile(vals, q)) for q in (0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0)}


def sample_raw_audit(
    row: dict[str, Any],
    *,
    stf_root: Path,
    raw_npz_root: Path,
    output_dir: Path,
    raw_storage_format: str,
) -> dict[str, Any]:
    sample_name = row["filename_stem"]
    image_path = resolve_data_path(stf_root, row["lut_preview"])
    raw_path = raw_path_for(raw_npz_root, sample_name)
    rgb_full = read_lut_preview(image_path)
    rgb_raw_hw = resize_rgb(rgb_full, DEFAULT_MODEL_HW, cv2.INTER_AREA)
    storage, decoded = load_decoded_raw(raw_path, raw_storage_format)
    raw_storage_f = storage.astype(np.float32) / 3967.0
    chroma = rgb_raw_hw[..., 0] - rgb_raw_hw[..., 2]
    high_chroma = np.abs(chroma) >= float(np.quantile(np.abs(chroma), 0.80))
    storage_c0_minus_c3 = raw_storage_f[..., 0] - raw_storage_f[..., 3]
    storage_c3_minus_c0 = raw_storage_f[..., 3] - raw_storage_f[..., 0]
    gr_gb_corr = corrcoef_safe(raw_storage_f[..., 1], raw_storage_f[..., 2])
    record = {
        "sample_name": sample_name,
        "image_path": str(image_path),
        "raw_path": str(raw_path),
        "raw_key": RECTIFIED_BAYER_KEY,
        "npz_shape": list(storage.shape),
        "npz_dtype": str(storage.dtype),
        "npz_min": int(storage.min()),
        "npz_max": int(storage.max()),
        "daytime": row.get("daytime"),
        "fog": row.get("fog"),
        "precipitation": row.get("precipitation"),
        "road_state": row.get("road_state"),
        "is_adverse_weather": row.get("is_adverse_weather"),
        "chroma_abs_p80_threshold": float(np.quantile(np.abs(chroma), 0.80)),
        "corr_storage_c0_minus_c3_vs_preview_R_minus_B": corrcoef_safe(storage_c0_minus_c3, chroma),
        "corr_storage_c3_minus_c0_vs_preview_R_minus_B": corrcoef_safe(storage_c3_minus_c0, chroma),
        "corr_storage_c0_minus_c3_vs_preview_R_minus_B_high_chroma": corrcoef_safe(storage_c0_minus_c3, chroma, high_chroma),
        "corr_storage_c3_minus_c0_vs_preview_R_minus_B_high_chroma": corrcoef_safe(storage_c3_minus_c0, chroma, high_chroma),
        "corr_storage_Gr_Gb": gr_gb_corr,
        "storage_channel_quantiles_companded_0_1": {
            name: quantiles(raw_storage_f[..., idx])
            for idx, name in enumerate(("c0_storage", "c1_storage", "c2_storage", "c3_storage"))
        },
        "decoded_channel_quantiles_0_1": {
            name: quantiles(decoded[..., idx])
            for idx, name in enumerate(RAW_MODEL_CHANNEL_ORDER)
        },
    }
    raw_preview = raw_tensor_to_preview(torch.from_numpy(np.transpose(decoded, (2, 0, 1))))
    panels: list[tuple[str, np.ndarray]] = [
        ("lut_preview area 512x960", (rgb_raw_hw * 255.0).round().astype(np.uint8)),
        ("decoded base_rgb [R,G,B]", raw_preview),
    ]
    for idx, name in enumerate(("storage c0", "storage c1", "storage c2", "storage c3")):
        panels.append((name, colorize_scalar(raw_storage_f[..., idx], vmax_pct=99.5)))
    for idx, name in enumerate(RAW_MODEL_CHANNEL_ORDER):
        panels.append((f"decoded {name}", colorize_scalar(decoded[..., idx], vmax_pct=99.5)))
    panel_path = output_dir / "panels" / f"{sample_name}_raw_audit.png"
    grid = build_labeled_grid(panels, cols=3, tile_hw=(256, 480), footer=f"{sample_name} | {raw_storage_format}")
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(panel_path)
    record["panel_path"] = str(panel_path)
    return record


def row_chroma_score(row: dict[str, Any], stf_root: Path) -> float:
    try:
        rgb = read_lut_preview(resolve_data_path(stf_root, row["lut_preview"]))
    except Exception:
        return -1.0
    small = cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA)
    return float(np.quantile(np.abs(small[..., 0] - small[..., 2]), 0.95))


def select_audit_rows(rows: list[dict[str, Any]], stf_root: Path, count: int) -> list[dict[str, Any]]:
    stride = max(len(rows) // min(max(count * 8, count), len(rows)), 1)
    candidates = rows[::stride]
    scored = [(row_chroma_score(row, stf_root), idx, row) for idx, row in enumerate(candidates)]
    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[dict[str, Any]] = []
    seen = set()
    for preference in ("day", "twilight", "night"):
        for _, _, row in scored:
            if len(selected) >= count:
                break
            if row.get("filename_stem") in seen:
                continue
            if str(row.get("daytime", "")).lower() == preference:
                selected.append(row)
                seen.add(row["filename_stem"])
        if len(selected) >= count:
            break
    for _, _, row in scored:
        if len(selected) >= count:
            break
        if row.get("filename_stem") not in seen:
            selected.append(row)
            seen.add(row["filename_stem"])
    return selected


def aggregate_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "corr_storage_c0_minus_c3_vs_preview_R_minus_B",
        "corr_storage_c3_minus_c0_vs_preview_R_minus_B",
        "corr_storage_c0_minus_c3_vs_preview_R_minus_B_high_chroma",
        "corr_storage_c3_minus_c0_vs_preview_R_minus_B_high_chroma",
        "corr_storage_Gr_Gb",
    )
    out = {}
    for key in keys:
        vals = np.asarray([record[key] for record in records if record.get(key) is not None], dtype=np.float64)
        out[key] = {
            "count": int(vals.size),
            "mean": float(vals.mean()) if vals.size else None,
            "median": float(np.median(vals)) if vals.size else None,
            "min": float(vals.min()) if vals.size else None,
            "max": float(vals.max()) if vals.size else None,
        }
    return out


def raw_distribution_summary(records: list[dict[str, Any]], led_summary_path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stf_decoded_raw_channel_quantiles_0_1": {},
        "led_synthetic_raw_reference": None,
        "note": "Distribution comparison is diagnostic only; STF is eval-only and these stats are not mixed into LED training configs.",
    }
    channels = {name: [] for name in RAW_MODEL_CHANNEL_ORDER}
    for record in records:
        for name in RAW_MODEL_CHANNEL_ORDER:
            q = record["decoded_channel_quantiles_0_1"][name]
            if q:
                channels[name].append(q)
    for name, qs in channels.items():
        if not qs:
            continue
        result["stf_decoded_raw_channel_quantiles_0_1"][name] = {
            key: float(np.mean([item[key] for item in qs if key in item]))
            for key in sorted({key for item in qs for key in item})
        }
    if led_summary_path is not None and led_summary_path.is_file():
        led = load_json(led_summary_path)
        result["led_synthetic_raw_reference"] = {
            "path": str(led_summary_path),
            "channel_quantiles": led.get("channel_quantiles"),
            "raw_all_quantiles": led.get("raw_all_quantiles"),
            "num_stats_samples": led.get("num_stats_samples"),
        }
    return result


def run_raw_audit(args: argparse.Namespace) -> None:
    stf_root = Path(args.stf_root).expanduser().resolve()
    raw_root = Path(args.raw_npz_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.raw_storage_format != RAW_FORMAT:
        raise ValueError(f"Stage A formal audit requires raw_storage_format={RAW_FORMAT!r}")
    rows = load_manifest_rows(stf_root, args.split, args.manifest_path)
    selected = select_audit_rows(rows, stf_root, int(args.audit_samples))
    records = [
        sample_raw_audit(
            row,
            stf_root=stf_root,
            raw_npz_root=raw_root,
            output_dir=output_dir,
            raw_storage_format=args.raw_storage_format,
        )
        for row in selected
    ]
    raw_spec = get_raw_storage_spec(args.raw_storage_format)
    led_summary_path = Path(args.led_raw_summary).expanduser().resolve() if args.led_raw_summary else None
    payload = {
        "stage": "A_raw_audit",
        "created_at_unix": time.time(),
        "stf_root": str(stf_root),
        "raw_npz_root": str(raw_root),
        "manifest_path": str(manifest_path(stf_root, args.split, args.manifest_path)),
        "raw_key": RECTIFIED_BAYER_KEY,
        "raw_storage_format": args.raw_storage_format,
        "raw_storage_channel_order": list(raw_spec.storage_channel_order),
        "raw_model_channel_order": list(raw_spec.model_channel_order),
        "raw_decompand": raw_spec.decompand,
        "raw_post_decode_norm": raw_spec.post_decode_norm,
        "channel_reorder": list(raw_spec.channel_reorder),
        "decision": {
            "formal_n2_n7_main_table_allowed_format": RAW_FORMAT,
            "identity_reading_allowed": False,
            "legacy_companded_default_allowed": False,
            "r_b_evidence_rule": "Use chroma difference corr(c0-c3, preview_R-preview_B); negative c0-c3 correlation supports storage [B,Gr,Gb,R].",
        },
        "audit_sample_count": len(records),
        "audit_samples": records,
        "aggregate": aggregate_audit(records),
        "raw_distribution_led_vs_stf": raw_distribution_summary(records, led_summary_path),
        "known_conflict": {
            "readme_or_pack_comment_claim": "[R, Gr, Gb, B] identity / GBRG comments",
            "current_npz_pixel_evidence": "storage behaves as [B, Gr, Gb, R], model input uses reorder (3,1,2,0)",
            "action": "Do not change formal eval format without repeating Stage A if raw root changes.",
        },
    }
    save_json(output_dir / "raw_audit.json", payload)
    save_json(output_dir / "raw_distribution_led_vs_stf.json", payload["raw_distribution_led_vs_stf"])
    print(f"[raw-audit] wrote {output_dir / 'raw_audit.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STF sparse eval for LED-HB D0/C2/N-series checkpoints.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--stf-root", default=DEFAULT_STF_ROOT)
    common.add_argument("--raw-npz-root", default=DEFAULT_RAW_NPZ_ROOT)
    common.add_argument("--split", default="val")
    common.add_argument("--manifest-path", default=None)
    common.add_argument("--raw-storage-format", default=RAW_FORMAT, choices=[RAW_FORMAT])
    common.add_argument("--input-height", type=int, default=DEFAULT_MODEL_HW[0])
    common.add_argument("--input-width", type=int, default=DEFAULT_MODEL_HW[1])
    common.add_argument("--min-depth", type=float, default=1.0)
    common.add_argument("--max-depth", type=float, default=80.0)
    common.add_argument("--image-resize", default="area", choices=["area", "cubic"])

    audit = sub.add_parser("raw-audit", parents=[common])
    audit.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "raw_audit"))
    audit.add_argument("--audit-samples", type=int, default=20)
    audit.add_argument(
        "--led-raw-summary",
        default=str(DEFAULT_OUTPUT_ROOT.parent / "led_hb_invisp_unprocessing_onepage_linear_0601_0020/led_hb_invisp_unprocessing_summary.json"),
    )

    eval_p = sub.add_parser("eval", parents=[common])
    eval_p.add_argument("--method", required=True, choices=sorted(RUNS))
    eval_p.add_argument("--run-dir", default=None)
    eval_p.add_argument("--checkpoint", default=None)
    eval_p.add_argument("--output-dir", required=True)
    eval_p.add_argument("--max-samples", type=int, default=None)
    eval_p.add_argument("--sample-indices", type=int, nargs="*", default=None)
    eval_p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    eval_p.add_argument("--encoder", default="vits", choices=sorted(MODEL_CONFIGS))
    eval_p.add_argument("--d0-sign", type=int, default=1, choices=[-1, 1])
    eval_p.add_argument("--amp", action="store_true", default=True)
    eval_p.add_argument("--no-amp", action="store_false", dest="amp")
    eval_p.add_argument("--feature-ablation-mode", default="true", choices=["true", "none", "zero", "mean", "shuffle"])
    eval_p.add_argument("--model-valid-mask-source", default="sparse_downsampled", choices=["sparse_downsampled", "all_ones"])
    eval_p.add_argument("--review-dir", default=None)
    eval_p.add_argument("--review-indices", type=int, nargs="*", default=[])

    args = parser.parse_args()
    if args.command == "raw-audit" and args.audit_samples <= 0:
        raise ValueError("--audit-samples must be positive")
    if args.command == "eval" and args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "raw-audit":
        run_raw_audit(args)
    elif args.command == "eval":
        evaluate_method(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
