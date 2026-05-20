#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.raw_audit_common import (
    DEVICE,
    MODEL_CONFIGS,
    build_labeled_grid,
    build_raw_student_model,
    build_rgb_reference_model,
    denorm_rgb_tensor,
    extract_dav2_patch_features,
    extract_rgb_teacher_patch_features,
    load_experiment_config,
    prepare_raw_student_vit_input,
    raw_tensor_to_preview,
    resize_rgb,
    resolve_checkpoint,
    write_json,
)
from finetune_stf.dataset.lod_raw import (
    DEFAULT_LOD_DAY_MANIFEST,
    DEFAULT_LOD_NIGHT_MANIFEST,
    DEFAULT_LOD_ROOT,
)
from finetune_stf.dataset.raw_utils import normalize_raw_4ch
from finetune_stf.dataset.robotcar import RobotCarValRGB, RobotCarValRaw
from foundation.engine.datasets.vkitti2_raw import DEFAULT_TRAIN_LIST, VKITTI2Raw


SPLIT_CHOICES = ("vkitti2", "lod_day", "lod_night", "robotcar_day", "robotcar_night")
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
STATUS_ORDER = {"pass": 0, "warning": 1, "fail": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E2' RAW/RGB geometry alignment sanity check in image space and DAv2 token space."
    )
    parser.add_argument("--exp-dir", required=True, type=Path, help="Experiment directory containing config.json.")
    parser.add_argument("--checkpoint", default="last", help='Checkpoint alias: "last", "current", "best", or a .pth path.')
    parser.add_argument("--splits", nargs="+", default=list(SPLIT_CHOICES), choices=SPLIT_CHOICES)
    parser.add_argument("--samples-per-split", type=int, default=5)
    parser.add_argument("--sample-plan", type=Path, default=None, help="Optional JSON mapping split names to indices/sample names.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Optional DINOv2 block indices to tap.")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic sample seed. Defaults to config seed.")
    parser.add_argument("--panel-width", type=int, default=420)
    parser.add_argument("--input-pass-threshold", type=float, default=0.5)
    parser.add_argument("--input-fail-threshold", type=float, default=1.5)
    parser.add_argument("--token-pass-threshold", type=float, default=0.5)
    parser.add_argument("--token-fail-threshold", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "__" for ch in str(text))


def parse_sample_plan(path: Path | None) -> dict[str, list[Any]]:
    if path is None:
        return {}
    with path.expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    out: dict[str, list[Any]] = {}
    for split, value in payload.items():
        if isinstance(value, dict):
            value = value.get("indices", value.get("samples"))
        if not isinstance(value, list):
            raise ValueError(f"Sample plan entry for {split!r} must be a list or object with indices/samples")
        out[str(split)] = value
    return out


def status_for_shift(value: float, pass_threshold: float, fail_threshold: float) -> str:
    if not np.isfinite(value):
        return "warning"
    if float(value) < float(pass_threshold):
        return "pass"
    if float(value) < float(fail_threshold):
        return "warning"
    return "fail"


def worst_status(statuses: list[str]) -> str:
    if not statuses:
        return "warning"
    return max(statuses, key=lambda item: STATUS_ORDER.get(item, 1))


def status_counts(statuses: list[str]) -> dict[str, int]:
    return {status: int(sum(item == status for item in statuses)) for status in ("pass", "warning", "fail")}


def resolve_layers(cfg: dict[str, Any], args: argparse.Namespace) -> list[int]:
    if args.layers:
        return [int(layer) for layer in args.layers]
    if cfg.get("bridge_layers"):
        return [int(layer) for layer in cfg["bridge_layers"]]
    encoder = cfg.get("encoder", "vitl")
    return list(MODEL_CONFIGS[encoder].get("intermediate_layer_idx", [])) or {
        "vits": [2, 5, 8, 11],
        "vitb": [2, 5, 8, 11],
        "vitl": [4, 11, 17, 23],
        "vitg": [9, 19, 29, 39],
    }[encoder]


def rgb_uint8_to_norm_tensor(rgb_u8: np.ndarray) -> torch.Tensor:
    rgb = np.asarray(rgb_u8, dtype=np.float32) / 255.0
    norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(np.transpose(norm, (2, 0, 1)).astype(np.float32, copy=False)))


def rgb_float_to_norm_tensor(rgb: np.ndarray) -> torch.Tensor:
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(np.transpose(norm, (2, 0, 1)).astype(np.float32, copy=False)))


def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def center_crop_np(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_h, target_w = size
    height, width = array.shape[:2]
    if height < target_h or width < target_w:
        raise ValueError(f"Requested crop {size} exceeds input shape {(height, width)}")
    top = max((height - target_h) // 2, 0)
    left = max((width - target_w) // 2, 0)
    return np.ascontiguousarray(array[top : top + target_h, left : left + target_w, ...])


def resolve_data_path(root: str | Path, path_str: str) -> Path:
    path = Path(str(path_str).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(root).expanduser().resolve() / path).resolve()


def load_lod_rows(manifest_path: str | Path, lod_root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(manifest_path).expanduser().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "sample_name", "rgb_path", "rggb_path"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{manifest_path} is missing required LOD columns: {', '.join(missing)}")
        for row in reader:
            rows.append(
                {
                    "split": row["split"].strip(),
                    "sample_name": row["sample_name"].strip(),
                    "rgb_path": resolve_data_path(lod_root, row["rgb_path"]),
                    "raw_path": resolve_data_path(lod_root, row["rggb_path"]),
                }
            )
    if not rows:
        raise ValueError(f"No LOD rows found: {manifest_path}")
    return rows


def dataset_sample_name(state: dict[str, Any], idx: int) -> str:
    split = state["split"]
    if split == "vkitti2":
        img_path = Path(state["dataset"].filelist[int(idx)].split()[0])
        return f"{img_path.parent.name}/{img_path.stem}"
    if split in {"lod_day", "lod_night"}:
        row = state["rows"][int(idx)]
        return f"{row['split']}/{row['sample_name']}"
    row = state["rgb_dataset"].rows[int(idx)]
    return f"{row['scene']}/{row['sample_name']}"


def select_indices(state: dict[str, Any], sample_plan: dict[str, list[Any]], samples_per_split: int) -> list[int]:
    split = state["split"]
    length = int(state["length"])
    if split not in sample_plan:
        count = max(1, min(int(samples_per_split), length))
        return np.linspace(0, length - 1, count, dtype=int).tolist()

    name_to_idx = {dataset_sample_name(state, idx): idx for idx in range(length)}
    stem_to_idx = {name.split("/")[-1]: idx for name, idx in name_to_idx.items()}
    indices: list[int] = []
    for item in sample_plan[split]:
        if isinstance(item, int):
            idx = int(item)
        elif isinstance(item, str) and item.isdigit():
            idx = int(item)
        elif isinstance(item, str) and item in name_to_idx:
            idx = int(name_to_idx[item])
        elif isinstance(item, str) and item in stem_to_idx:
            idx = int(stem_to_idx[item])
        else:
            raise ValueError(f"Could not resolve sample {item!r} for split {split}")
        if idx < 0 or idx >= length:
            raise ValueError(f"Sample index {idx} out of range for split {split} length {length}")
        indices.append(idx)
    return sorted(dict.fromkeys(indices))


def resolve_raw_input_mode(input_type: str) -> str:
    if str(input_type) in {
        "raw_packed",
        "raw_ram",
        "raw_ram_rgb",
        "raw_ram_residual",
        "raw_ram_feature_adapter",
        "raw_ram_bridge_feature_adapter",
        "raw_ram_bridge_feature_adapter_lora",
        "raw_ram_bridge",
        "raw_ram_bridge_lora",
        "raw_ram_rgb_bridge",
        "raw_ram_rgb_bridge_lora",
    }:
        return "raw_ram"
    return "raw_naive"


def build_split_state(split: str, cfg: dict[str, Any]) -> dict[str, Any]:
    input_hw = (int(cfg.get("input_height", 644)), int(cfg.get("input_width", 1008)))
    if split == "vkitti2":
        dataset = VKITTI2Raw(
            filelist_path=cfg.get("vkitti_train_list", str(DEFAULT_TRAIN_LIST)),
            mode="train",
            size=input_hw,
            min_depth=float(cfg.get("min_depth", 1.0)),
            max_depth=float(cfg.get("max_depth", 80.0)),
            randomize_unprocessing=bool(cfg.get("vkitti_randomize_unprocessing", True)),
            unprocessing_preset=cfg.get("vkitti_unprocessing_preset", "sensor_linear_dual"),
            unprocessing_mix_weights=cfg.get("vkitti_unprocessing_mix_weights"),
            hflip_prob=float(cfg.get("vkitti_hflip_prob", 0.5)),
        )
        return {"split": split, "kind": "vkitti2", "dataset": dataset, "length": len(dataset)}

    if split in {"lod_day", "lod_night"}:
        manifest = cfg.get("lod_day_manifest") if split == "lod_day" else cfg.get("lod_night_manifest")
        if not manifest:
            manifest = DEFAULT_LOD_DAY_MANIFEST if split == "lod_day" else DEFAULT_LOD_NIGHT_MANIFEST
        rows = load_lod_rows(manifest, cfg.get("lod_root", DEFAULT_LOD_ROOT))
        return {
            "split": split,
            "kind": "lod",
            "rows": rows,
            "length": len(rows),
            "manifest": str(Path(manifest).expanduser().resolve()),
        }

    is_night = split == "robotcar_night"
    common = dict(
        robotcar_root=cfg.get(
            "robotcar_night_root" if is_night else "robotcar_root",
            "/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_subset100",
        ),
        depth_mode="fast",
        fast_eval_backend=cfg.get("robotcar_night_fast_eval_backend" if is_night else "robotcar_fast_eval_backend", "sparse"),
        min_depth=float(cfg.get("robotcar_night_min_depth" if is_night else "robotcar_min_depth", 0.1)),
        max_depth=float(cfg.get("robotcar_night_max_depth" if is_night else "robotcar_max_depth", 50.0)),
    )
    if is_night:
        common["manifest_name"] = cfg.get("robotcar_night_manifest_name", "robotcar_raw_depth_v1_val_balanced250_scene_interleaved.csv")
    rgb_dataset = RobotCarValRGB(**common)
    raw_dataset = RobotCarValRaw(
        **common,
        norm_mode=cfg.get("robotcar_night_norm_mode" if is_night else "robotcar_norm_mode", "sensor_linear"),
        channel_mode=cfg.get("channel_mode", "rgb_avg_g"),
        use_imagenet_norm=bool(cfg.get("use_imagenet_norm", True)),
        input_mode=resolve_raw_input_mode(cfg.get("input_type", "raw_ram_bridge")),
    )
    if len(rgb_dataset) != len(raw_dataset):
        raise RuntimeError(f"{split}: RGB/RAW dataset length mismatch: {len(rgb_dataset)} vs {len(raw_dataset)}")
    return {
        "split": split,
        "kind": "robotcar",
        "rgb_dataset": rgb_dataset,
        "raw_dataset": raw_dataset,
        "length": len(rgb_dataset),
    }


def replay_vkitti_rgb_to_sensor(dataset: VKITTI2Raw, sample: dict[str, Any]) -> np.ndarray:
    image = read_rgb(sample["image_path"]).astype(np.float32) / 255.0
    dummy_depth = np.zeros(image.shape[:2], dtype=np.float32)
    dummy_valid = np.ones(image.shape[:2], dtype=bool)
    image, _, _ = dataset._resize_short_edge(image, dummy_depth, dummy_valid, short_edge=dataset.fullres_size[0])
    top, left, bottom, right = [int(v) for v in sample["geometry_params"]["crop_box"]]
    image = image[top:bottom, left:right]
    if bool(sample["geometry_params"].get("hflip_applied", False)):
        image = np.ascontiguousarray(image[:, ::-1])
    sensor_h, sensor_w = dataset.size
    image = cv2.resize(image, (sensor_w, sensor_h), interpolation=cv2.INTER_AREA)
    return np.clip(image, 0.0, 1.0)


def load_pair(state: dict[str, Any], idx: int, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    split = state["split"]
    input_hw = (int(cfg.get("input_height", 644)), int(cfg.get("input_width", 1008)))
    if split == "vkitti2":
        dataset: VKITTI2Raw = state["dataset"]
        py_rng = random.Random(int(seed) + int(idx))
        torch_generator = torch.Generator().manual_seed(int(seed) + int(idx))
        sample = dataset.build_sample(
            int(idx),
            py_rng=py_rng,
            torch_generator=torch_generator,
            include_geometry=True,
        )
        raw_tensor = sample["raw"].float()
        rgb_float = replay_vkitti_rgb_to_sensor(dataset, sample)
        rgb_u8 = (rgb_float * 255.0).round().astype(np.uint8)
        return {
            "split": split,
            "dataset_index": int(idx),
            "sample_name": dataset_sample_name(state, idx),
            "raw_tensor": raw_tensor,
            "rgb_tensor": rgb_float_to_norm_tensor(rgb_float),
            "raw_vis": raw_tensor_to_preview(raw_tensor),
            "rgb_vis": rgb_u8,
            "metadata": {
                "source": "VKITTI2 synthetic RAW from paired RGB",
                "image_path": sample["image_path"],
                "geometry_params": sample["geometry_params"],
                "isp_params": {key: str(value) for key, value in sample.get("isp_params", {}).items()},
            },
        }

    if split in {"lod_day", "lod_night"}:
        row = state["rows"][int(idx)]
        raw = np.load(row["raw_path"]).astype(np.float32, copy=False)
        if raw.ndim != 3 or raw.shape[-1] != 4:
            raise ValueError(f"Expected LOD raw HWC4, got {raw.shape} for {row['raw_path']}")
        raw = normalize_raw_4ch(raw, norm_mode=cfg.get("norm_mode", "sensor_linear"))
        rgb_u8 = read_rgb(row["rgb_path"])
        if tuple(rgb_u8.shape[:2]) != tuple(raw.shape[:2]):
            rgb_u8 = cv2.resize(rgb_u8, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_AREA)
        raw = center_crop_np(raw, input_hw)
        rgb_u8 = center_crop_np(rgb_u8, input_hw)
        raw_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(raw, (2, 0, 1)).astype(np.float32, copy=False)))
        return {
            "split": split,
            "dataset_index": int(idx),
            "sample_name": dataset_sample_name(state, idx),
            "raw_tensor": raw_tensor,
            "rgb_tensor": rgb_uint8_to_norm_tensor(rgb_u8),
            "raw_vis": raw_tensor_to_preview(raw_tensor),
            "rgb_vis": rgb_u8,
            "metadata": {
                "source": "LOD real RAW and paired SDR RGB center crop",
                "manifest": state.get("manifest"),
                "rgb_path": str(row["rgb_path"]),
                "raw_path": str(row["raw_path"]),
            },
        }

    rgb_sample = state["rgb_dataset"][int(idx)]
    raw_sample = state["raw_dataset"][int(idx)]
    raw_tensor = raw_sample.get("raw", raw_sample["image"]).float()
    rgb_u8 = (denorm_rgb_tensor(rgb_sample["image"]) * 255.0).round().astype(np.uint8)
    raw_hw = tuple(int(v) for v in raw_tensor.shape[-2:])
    if tuple(rgb_u8.shape[:2]) != raw_hw:
        rgb_u8 = resize_rgb(rgb_u8, raw_hw, interpolation=cv2.INTER_AREA)
        rgb_tensor = rgb_uint8_to_norm_tensor(rgb_u8)
    else:
        rgb_tensor = rgb_sample["image"].float()
    return {
        "split": split,
        "dataset_index": int(idx),
        "sample_name": dataset_sample_name(state, idx),
        "raw_tensor": raw_tensor,
        "rgb_tensor": rgb_tensor,
        "raw_vis": raw_tensor_to_preview(raw_tensor),
        "rgb_vis": rgb_u8,
        "metadata": {
            "source": "RobotCar real paired RAW/RGB eval geometry",
            "rgb_path": rgb_sample["image_path"],
            "raw_path": raw_sample["image_path"],
        },
    }


def make_tile_hw(sample_hw: tuple[int, int], panel_width: int) -> tuple[int, int]:
    h, w = sample_hw
    width = max(64, int(panel_width))
    height = max(64, int(round(h * width / max(w, 1))))
    return height, width


def rgb_to_luma(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_u8, dtype=np.float32) / 255.0
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32, copy=False)


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)


def canny_edges(gray: np.ndarray) -> np.ndarray:
    gray_u8 = np.clip(np.asarray(gray, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    return (cv2.Canny(gray_u8, 80, 160) > 0).astype(np.float32)


def phase_correlate(ref_map: np.ndarray, moving_map: np.ndarray) -> dict[str, float]:
    ref = np.asarray(ref_map, dtype=np.float32)
    moving = np.asarray(moving_map, dtype=np.float32)
    if ref.shape != moving.shape:
        moving = cv2.resize(moving, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)
    finite = np.isfinite(ref) & np.isfinite(moving)
    if int(finite.sum()) < 16:
        return {"dx": float("nan"), "dy": float("nan"), "shift_norm": float("nan"), "response": float("nan")}
    ref = np.where(finite, ref, float(np.nanmean(ref[finite]))).astype(np.float32, copy=False)
    moving = np.where(finite, moving, float(np.nanmean(moving[finite]))).astype(np.float32, copy=False)
    ref = ref - float(ref.mean())
    moving = moving - float(moving.mean())
    ref_std = float(ref.std())
    moving_std = float(moving.std())
    if ref_std < 1e-6 or moving_std < 1e-6:
        return {"dx": float("nan"), "dy": float("nan"), "shift_norm": float("nan"), "response": float("nan")}
    ref = np.ascontiguousarray(ref / ref_std)
    moving = np.ascontiguousarray(moving / moving_std)
    window = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(ref, moving, window)
    return {
        "dx": float(dx),
        "dy": float(dy),
        "shift_norm": float((dx * dx + dy * dy) ** 0.5),
        "response": float(response),
    }


def phase_correlate_best_sign(ref_map: np.ndarray, moving_map: np.ndarray) -> dict[str, float]:
    direct = phase_correlate(ref_map, moving_map)
    flipped = phase_correlate(ref_map, -np.asarray(moving_map, dtype=np.float32))
    direct_score = direct["response"] if np.isfinite(direct["response"]) else -np.inf
    flipped_score = flipped["response"] if np.isfinite(flipped["response"]) else -np.inf
    if flipped_score > direct_score:
        return {**flipped, "sign": -1.0}
    return {**direct, "sign": 1.0}


def normalize_01(values: np.ndarray, *, symmetric: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    if symmetric:
        vmax = float(np.percentile(np.abs(arr[finite]), 99.0))
        vmax = max(vmax, 1e-6)
        return np.clip((arr + vmax) / (2.0 * vmax), 0.0, 1.0)
    vmin = float(np.percentile(arr[finite], 1.0))
    vmax = float(np.percentile(arr[finite], 99.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)


def colorize_map(values: np.ndarray, *, cmap: int = cv2.COLORMAP_VIRIDIS, symmetric: bool = False) -> np.ndarray:
    scaled = normalize_01(values, symmetric=symmetric)
    rgb = cv2.applyColorMap((scaled * 255.0).round().astype(np.uint8), cmap)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def colorize_cosine(values: np.ndarray) -> np.ndarray:
    scaled = np.clip((np.asarray(values, dtype=np.float32) + 1.0) * 0.5, 0.0, 1.0)
    rgb = cv2.applyColorMap((scaled * 255.0).round().astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def checkerboard(a: np.ndarray, b: np.ndarray, *, tile: int = 32) -> np.ndarray:
    if a.shape[:2] != b.shape[:2]:
        b = resize_rgb(b, a.shape[:2], interpolation=cv2.INTER_AREA)
    h, w = a.shape[:2]
    yy, xx = np.indices((h, w))
    mask = ((yy // int(tile)) + (xx // int(tile))) % 2 == 0
    out = np.asarray(a).copy()
    out[~mask] = b[~mask]
    return out.astype(np.uint8, copy=False)


def edge_overlay(raw_gray: np.ndarray, raw_edges: np.ndarray, rgb_edges: np.ndarray) -> np.ndarray:
    base = np.repeat((np.clip(raw_gray, 0.0, 1.0) * 255.0).round().astype(np.uint8)[..., None], 3, axis=-1)
    raw_mask = raw_edges > 0
    rgb_mask = rgb_edges > 0
    base[raw_mask] = (255, 60, 60)
    base[rgb_mask] = (60, 255, 60)
    base[raw_mask & rgb_mask] = (255, 255, 60)
    return base


def compute_input_alignment(pair: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    raw_vis = pair["raw_vis"]
    rgb_vis = pair["rgb_vis"]
    if tuple(rgb_vis.shape[:2]) != tuple(raw_vis.shape[:2]):
        rgb_vis = resize_rgb(rgb_vis, raw_vis.shape[:2], interpolation=cv2.INTER_AREA)

    raw_luma = rgb_to_luma(raw_vis)
    rgb_luma = rgb_to_luma(rgb_vis)
    raw_sobel = sobel_magnitude(raw_luma)
    rgb_sobel = sobel_magnitude(rgb_luma)
    raw_edges = canny_edges(raw_luma)
    rgb_edges = canny_edges(rgb_luma)
    luma_phase = phase_correlate(raw_luma, rgb_luma)
    sobel_phase = phase_correlate(raw_sobel, rgb_sobel)
    canny_phase = phase_correlate(raw_edges, rgb_edges)
    gate = status_for_shift(sobel_phase["shift_norm"], args.input_pass_threshold, args.input_fail_threshold)

    split_dir = output_dir / "input_level" / pair["split"]
    split_dir.mkdir(parents=True, exist_ok=True)
    panel_path = split_dir / f"{pair['dataset_index']:05d}_{safe_name(pair['sample_name'])}_input.jpg"
    diff = np.abs(raw_luma - rgb_luma)
    tile_hw = make_tile_hw(raw_vis.shape[:2], args.panel_width)
    footer = (
        f"{pair['split']} idx={pair['dataset_index']} {pair['sample_name']} | "
        f"sobel shift={sobel_phase['shift_norm']:.3f}px gate={gate}"
    )
    panel = build_labeled_grid(
        [
            ("raw_base_vis", raw_vis),
            ("paired_rgb_sensor_grid", rgb_vis),
            ("checkerboard", checkerboard(raw_vis, rgb_vis)),
            ("raw_luma", colorize_map(raw_luma)),
            ("rgb_luma", colorize_map(rgb_luma)),
            ("abs_luma_diff", colorize_map(diff, cmap=cv2.COLORMAP_INFERNO)),
            ("raw_edges", colorize_map(raw_edges)),
            ("rgb_edges", colorize_map(rgb_edges)),
            ("edge_overlay", edge_overlay(raw_luma, raw_edges, rgb_edges)),
        ],
        cols=3,
        tile_hw=tile_hw,
        footer=footer,
    )
    panel.save(panel_path, quality=95)
    return {
        "panel_path": str(panel_path),
        "gate": gate,
        "phase_luma": luma_phase,
        "phase_sobel": sobel_phase,
        "phase_canny": canny_phase,
        "edge_density": {
            "raw": float(raw_edges.mean()),
            "rgb": float(rgb_edges.mean()),
        },
    }


def feature_norm_map(feature: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(feature, dtype=np.float32), axis=0)


def cosine_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Feature shape mismatch for cosine: {a.shape} vs {b.shape}")
    dot = np.sum(a * b, axis=0)
    denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return (dot / np.clip(denom, 1e-6, None)).astype(np.float32, copy=False)


def pca1_feature_maps(features: list[np.ndarray], *, iterations: int = 12) -> list[np.ndarray]:
    channels = int(features[0].shape[0])
    matrices = [np.asarray(feat, dtype=np.float32).reshape(channels, -1).T for feat in features]
    x = np.concatenate(matrices, axis=0)
    finite_rows = np.isfinite(x).all(axis=1)
    if int(finite_rows.sum()) < 8:
        return [np.zeros(feat.shape[1:], dtype=np.float32) for feat in features]
    x = x[finite_rows]
    mean = x.mean(axis=0, keepdims=True)
    x = x - mean
    rng = np.random.default_rng(0)
    vector = rng.normal(size=(channels,)).astype(np.float32)
    vector /= max(float(np.linalg.norm(vector)), 1e-6)
    for _ in range(max(1, int(iterations))):
        scores = x @ vector
        vector = x.T @ scores
        vector /= max(float(np.linalg.norm(vector)), 1e-6)
    maps: list[np.ndarray] = []
    for matrix, feat in zip(matrices, features):
        projected = (matrix - mean) @ vector
        maps.append(projected.reshape(feat.shape[1:]).astype(np.float32, copy=False))
    return maps


def overlay_two_maps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a01 = normalize_01(a)
    b01 = normalize_01(b)
    return np.stack([a01, 0.5 * (a01 + b01), b01], axis=-1).clip(0.0, 1.0).__mul__(255.0).round().astype(np.uint8)


def finite_max(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return max(finite) if finite else float("nan")


def compute_token_alignment(
    pair: dict[str, Any],
    teacher_features: dict[int, np.ndarray],
    no_bridge_features: dict[int, np.ndarray],
    with_bridge_features: dict[int, np.ndarray],
    layers: list[int],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    split_dir = output_dir / "token_level" / pair["split"]
    split_dir.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        teacher = teacher_features[int(layer)]
        no_bridge = no_bridge_features[int(layer)]
        with_bridge = with_bridge_features[int(layer)]
        if teacher.shape != no_bridge.shape or teacher.shape != with_bridge.shape:
            raise RuntimeError(
                f"{pair['split']} idx={pair['dataset_index']} layer={layer}: "
                f"feature shape mismatch teacher={teacher.shape} no_bridge={no_bridge.shape} with_bridge={with_bridge.shape}"
            )

        teacher_norm = feature_norm_map(teacher)
        no_norm = feature_norm_map(no_bridge)
        with_norm = feature_norm_map(with_bridge)
        teacher_pca, no_pca, with_pca = pca1_feature_maps([teacher, no_bridge, with_bridge])
        no_cos = cosine_map(teacher, no_bridge)
        with_cos = cosine_map(teacher, with_bridge)

        no_norm_phase = phase_correlate(teacher_norm, no_norm)
        no_pca_phase = phase_correlate_best_sign(teacher_pca, no_pca)
        with_norm_phase = phase_correlate(teacher_norm, with_norm)
        with_pca_phase = phase_correlate_best_sign(teacher_pca, with_pca)
        no_shift = finite_max([no_norm_phase["shift_norm"], no_pca_phase["shift_norm"]])
        with_shift = finite_max([with_norm_phase["shift_norm"], with_pca_phase["shift_norm"]])
        no_gate = status_for_shift(no_shift, args.token_pass_threshold, args.token_fail_threshold)
        with_gate = status_for_shift(with_shift, args.token_pass_threshold, args.token_fail_threshold)

        panel_path = split_dir / f"{pair['dataset_index']:05d}_{safe_name(pair['sample_name'])}_layer{int(layer):02d}.jpg"
        tile_hw = make_tile_hw(teacher_norm.shape, args.panel_width)
        footer = (
            f"{pair['split']} idx={pair['dataset_index']} {pair['sample_name']} layer={layer} | "
            f"no_bridge={no_shift:.3f}tok {no_gate}; with_bridge={with_shift:.3f}tok {with_gate}"
        )
        panel = build_labeled_grid(
            [
                ("teacher norm", colorize_map(teacher_norm)),
                ("student no-bridge norm", colorize_map(no_norm)),
                ("student with-bridge norm", colorize_map(with_norm)),
                ("teacher PCA-1", colorize_map(teacher_pca, cmap=cv2.COLORMAP_TURBO, symmetric=True)),
                ("no-bridge PCA-1", colorize_map(no_pca, cmap=cv2.COLORMAP_TURBO, symmetric=True)),
                ("with-bridge PCA-1", colorize_map(with_pca, cmap=cv2.COLORMAP_TURBO, symmetric=True)),
                ("cos teacher/no-bridge", colorize_cosine(no_cos)),
                ("cos teacher/with-bridge", colorize_cosine(with_cos)),
                ("norm overlay T/with", overlay_two_maps(teacher_norm, with_norm)),
            ],
            cols=3,
            tile_hw=tile_hw,
            footer=footer,
        )
        panel.save(panel_path, quality=95)

        records.append(
            {
                "layer": int(layer),
                "panel_path": str(panel_path),
                "patch_hw": [int(teacher.shape[1]), int(teacher.shape[2])],
                "student_no_bridge": {
                    "gate": no_gate,
                    "token_shift": no_shift,
                    "phase_norm": no_norm_phase,
                    "phase_pca1": no_pca_phase,
                    "cosine_mean": float(np.nanmean(no_cos)),
                    "cosine_p10": float(np.nanpercentile(no_cos, 10.0)),
                },
                "student_with_bridge": {
                    "gate": with_gate,
                    "token_shift": with_shift,
                    "phase_norm": with_norm_phase,
                    "phase_pca1": with_pca_phase,
                    "cosine_mean": float(np.nanmean(with_cos)),
                    "cosine_p10": float(np.nanpercentile(with_cos, 10.0)),
                },
            }
        )
    return records


def summarize_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0, "gate": "warning"}
    input_statuses = [record["input_alignment"]["gate"] for record in records]
    no_statuses = [
        layer_record["student_no_bridge"]["gate"]
        for record in records
        for layer_record in record["token_alignment"]
    ]
    with_statuses = [
        layer_record["student_with_bridge"]["gate"]
        for record in records
        for layer_record in record["token_alignment"]
    ]
    input_shifts = [record["input_alignment"]["phase_sobel"]["shift_norm"] for record in records]
    no_shifts = [
        layer_record["student_no_bridge"]["token_shift"]
        for record in records
        for layer_record in record["token_alignment"]
    ]
    with_shifts = [
        layer_record["student_with_bridge"]["token_shift"]
        for record in records
        for layer_record in record["token_alignment"]
    ]
    geometry_gate = worst_status(input_statuses + no_statuses)
    direct_anchor_gate = worst_status(input_statuses + with_statuses)
    return {
        "n": len(records),
        "geometry_gate": geometry_gate,
        "direct_anchor_gate": direct_anchor_gate,
        "input_gate_counts": status_counts(input_statuses),
        "no_bridge_gate_counts": status_counts(no_statuses),
        "with_bridge_gate_counts": status_counts(with_statuses),
        "input_sobel_shift_px_mean": float(np.nanmean(input_shifts)),
        "input_sobel_shift_px_max": float(np.nanmax(input_shifts)),
        "no_bridge_token_shift_mean": float(np.nanmean(no_shifts)),
        "no_bridge_token_shift_max": float(np.nanmax(no_shifts)),
        "with_bridge_token_shift_mean": float(np.nanmean(with_shifts)),
        "with_bridge_token_shift_max": float(np.nanmax(with_shifts)),
        "bridge_content_note": (
            "with_bridge is worse than no_bridge; inspect bridge-content panels"
            if STATUS_ORDER[direct_anchor_gate] > STATUS_ORDER[geometry_gate]
            else "not_triggered"
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# E2' Alignment Sanity",
        "",
        f"- exp_dir: `{payload['exp_dir']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- layers: `{payload['layers']}`",
        f"- overall_geometry_gate: `{payload['overall_geometry_gate']}`",
        f"- overall_direct_anchor_gate: `{payload['overall_direct_anchor_gate']}`",
        "",
        "| split | n | geometry gate | direct-anchor gate | input shift px mean/max | no-bridge token mean/max | with-bridge token mean/max | bridge note |",
        "|---|---:|---|---|---:|---:|---:|---|",
    ]
    for split, summary in payload["summary"].items():
        lines.append(
            "| {split} | {n} | {gg} | {dg} | {ism:.3f}/{isx:.3f} | {nsm:.3f}/{nsx:.3f} | {wsm:.3f}/{wsx:.3f} | {note} |".format(
                split=split,
                n=summary["n"],
                gg=summary["geometry_gate"],
                dg=summary["direct_anchor_gate"],
                ism=summary["input_sobel_shift_px_mean"],
                isx=summary["input_sobel_shift_px_max"],
                nsm=summary["no_bridge_token_shift_mean"],
                nsx=summary["no_bridge_token_shift_max"],
                wsm=summary["with_bridge_token_shift_mean"],
                wsx=summary["with_bridge_token_shift_max"],
                note=summary["bridge_content_note"],
            )
        )
    lines.extend(
        [
            "",
            "Gate thresholds:",
            f"- input phase shift: pass `< {payload['thresholds']['input_pass_px']}` px, warning `< {payload['thresholds']['input_fail_px']}` px, fail otherwise.",
            f"- token phase shift: pass `< {payload['thresholds']['token_pass']}` token, warning `< {payload['thresholds']['token_fail']}` token, fail otherwise.",
            "",
            "Notes:",
            "- `geometry_gate` uses input-level checks plus student no-bridge token checks.",
            "- `direct_anchor_gate` uses input-level checks plus student with-bridge token checks.",
            "- If no-bridge passes but with-bridge warns/fails, treat it first as a bridge-content issue, not proof of RGB/RAW image geometry mismatch.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.samples_per_split < 1:
        raise ValueError("--samples-per-split must be >= 1")
    exp_dir = args.exp_dir.expanduser().resolve()
    cfg = load_experiment_config(exp_dir)
    checkpoint_path = resolve_checkpoint(exp_dir, args.checkpoint, cfg)
    output_suffix = safe_name(str(args.checkpoint))
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else exp_dir / f"alignment_sanity_{output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = resolve_layers(cfg, args)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    use_amp = not args.no_amp
    sample_plan = parse_sample_plan(args.sample_plan)

    split_states = {split: build_split_state(split, cfg) for split in args.splits}
    fixed_samples: list[dict[str, Any]] = []
    for split, state in split_states.items():
        indices = select_indices(state, sample_plan, args.samples_per_split)
        state["indices"] = indices
        for order, idx in enumerate(indices, start=1):
            fixed_samples.append(
                {
                    "split": split,
                    "order": order,
                    "dataset_index": int(idx),
                    "sample_name": dataset_sample_name(state, idx),
                }
            )
    write_json(output_dir / "fixed_samples.json", {"samples": fixed_samples})

    print(f"[align] device={DEVICE} exp={exp_dir.name} checkpoint={checkpoint_path}", flush=True)
    print(f"[align] layers={layers} splits={list(args.splits)}", flush=True)
    print("[align] loading frozen RGB DAv2 teacher", flush=True)
    rgb_model = build_rgb_reference_model(cfg)
    print("[align] loading RAW student", flush=True)
    raw_model = build_raw_student_model(cfg, checkpoint_path)

    per_sample: list[dict[str, Any]] = []
    for split, state in split_states.items():
        for idx in state["indices"]:
            pair = load_pair(state, int(idx), cfg, seed)
            print(f"[align][sample] {split} idx={idx} {pair['sample_name']}", flush=True)
            input_alignment = compute_input_alignment(pair, args, output_dir)

            teacher_result = extract_rgb_teacher_patch_features(
                rgb_model,
                pair["rgb_tensor"],
                layers,
                use_amp=use_amp,
            )
            with torch.no_grad():
                x_norm_padded, bridge_injections = prepare_raw_student_vit_input(raw_model, pair["raw_tensor"])
            no_bridge_features = extract_dav2_patch_features(
                raw_model.dav2,
                x_norm_padded,
                layers,
                bridge_injections=None,
                use_amp=use_amp,
            )
            with_bridge_features = extract_dav2_patch_features(
                raw_model.dav2,
                x_norm_padded,
                layers,
                bridge_injections=bridge_injections,
                use_amp=use_amp,
            )
            token_alignment = compute_token_alignment(
                pair,
                teacher_result["features"],
                no_bridge_features,
                with_bridge_features,
                layers,
                args,
                output_dir,
            )
            per_sample.append(
                {
                    "split": split,
                    "dataset_index": int(idx),
                    "sample_name": pair["sample_name"],
                    "metadata": pair["metadata"],
                    "input_alignment": input_alignment,
                    "teacher_feature_meta": {key: value for key, value in teacher_result.items() if key != "features"},
                    "student_feature_meta": {
                        "sensor_hw": [int(v) for v in pair["raw_tensor"].shape[-2:]],
                        "padded_hw": [int(v) for v in x_norm_padded.shape[-2:]],
                        "patch_hw": [
                            int(x_norm_padded.shape[-2] // raw_model.dav2.pretrained.patch_size),
                            int(x_norm_padded.shape[-1] // raw_model.dav2.pretrained.patch_size),
                        ],
                        "bridge_injections_available": bool(bridge_injections),
                    },
                    "token_alignment": token_alignment,
                }
            )
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    split_summary = {
        split: summarize_split([record for record in per_sample if record["split"] == split])
        for split in args.splits
    }
    geometry_gate = worst_status([summary["geometry_gate"] for summary in split_summary.values()])
    direct_anchor_gate = worst_status([summary["direct_anchor_gate"] for summary in split_summary.values()])
    payload = {
        "exp_dir": str(exp_dir),
        "checkpoint": str(checkpoint_path),
        "splits": list(args.splits),
        "layers": layers,
        "samples_per_split": int(args.samples_per_split),
        "fixed_samples_path": str(output_dir / "fixed_samples.json"),
        "input_panel_dir": str(output_dir / "input_level"),
        "token_panel_dir": str(output_dir / "token_level"),
        "thresholds": {
            "input_pass_px": float(args.input_pass_threshold),
            "input_fail_px": float(args.input_fail_threshold),
            "token_pass": float(args.token_pass_threshold),
            "token_fail": float(args.token_fail_threshold),
        },
        "overall_geometry_gate": geometry_gate,
        "overall_direct_anchor_gate": direct_anchor_gate,
        "summary": split_summary,
        "per_sample": per_sample,
    }
    write_json(output_dir / "alignment_summary.json", payload)
    write_markdown(output_dir / "alignment_summary.md", payload)
    print(json.dumps({"output_dir": str(output_dir), "geometry_gate": geometry_gate, "direct_anchor_gate": direct_anchor_gate}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
