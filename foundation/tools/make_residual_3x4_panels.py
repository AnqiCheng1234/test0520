#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from depth_anything_v2.dpt import DepthAnythingV2
from foundation.engine.datasets import VKITTI2HalfresRGBDepth, VKITTI2Raw
from foundation.engine.models import build_dav2_residual_control_model, build_raw_residual_dav2_model
from foundation.tools._viz_distribution import (
    DEFAULT_RAW_COLORS,
    DEFAULT_RGB_COLORS,
    draw_distribution_tile,
    summarize_channels,
)
from foundation.tools.eval_raw_residual_kitti import (
    KittiHalfresRawDataset,
    add_horizontal_colorbar,
    colormap_gradient,
    improvement_gradient,
    metric_label,
    packed_raw_preview,
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
from foundation.tools.residual_control_kitti_eval import KittiHalfresRGBDepthDataset


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
VKITTI_VARIANT_ORDER = (
    "clone",
    "fog",
    "rain",
    "overcast",
    "morning",
    "sunset",
    "15-deg-left",
    "15-deg-right",
    "30-deg-left",
    "30-deg-right",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3x4 residual qualitative panels for KITTI or VKITTI.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True, choices=("kitti", "vkitti"))
    parser.add_argument("--kitti-val-split", default="/home/caq/6666_raw/dav2_raw/metric_depth/dataset/splits/kitti/val.txt")
    parser.add_argument("--kitti-base", default="/mnt/drive/kitti")
    parser.add_argument("--sample-indices", default=None)
    parser.add_argument("--max-panels", type=int, default=10)
    parser.add_argument("--error-max-abs-rel", type=float, default=0.75)
    parser.add_argument("--depth-pmin", type=float, default=1.0)
    parser.add_argument("--depth-pmax", type=float, default=99.0)
    parser.add_argument("--hist-bins", type=int, default=128)
    parser.add_argument("--tile-width", type=int, default=414)
    parser.add_argument("--tile-height", type=int, default=125)
    parser.add_argument("--header-height", type=int, default=30)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return parser.parse_args()


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def load_run_config(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def resolve_model_state(ckpt_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    return ckpt_obj


def infer_epoch(checkpoint: Path) -> int:
    stem = checkpoint.stem
    if stem.startswith("epoch_"):
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return -1
    return -1


def build_model(config: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    base_model = DepthAnythingV2(**MODEL_CONFIGS[str(config["encoder"])])
    input_domain = str(config["input_domain"])
    if input_domain == "raw4":
        model = build_raw_residual_dav2_model(
            base_model,
            residual_feature_source=str(config["residual_feature_source"]),
            residual_head_d0_mode=str(config.get("residual_head_d0_mode", "concat")),
            residual_alpha=float(config["residual_alpha"]),
            d0_sign=int(config["d0_sign"]),
            sensor_hw=(int(config["input_height"]), int(config["input_width"])),
            backbone_hw=None,
        )
    elif input_domain == "rgb":
        model = build_dav2_residual_control_model(
            base_model,
            residual_feature_source=str(config["residual_feature_source"]),
            residual_alpha=float(config["residual_alpha"]),
            d0_sign=int(config["d0_sign"]),
            sensor_hw=(int(config["input_height"]), int(config["input_width"])),
            backbone_hw=None,
        )
    else:
        raise ValueError(f"Unsupported input_domain={input_domain!r}")

    ckpt_obj = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(strip_module_prefix(resolve_model_state(ckpt_obj)), strict=True)
    return model.to(device).eval()


def build_dataset(config: dict[str, Any], args: argparse.Namespace):
    is_raw = str(config["input_domain"]) == "raw4"
    if args.dataset == "kitti":
        if is_raw:
            return KittiHalfresRawDataset(
                filelist_path=args.kitti_val_split,
                kitti_base=args.kitti_base,
                min_depth=float(config["min_depth"]),
                max_depth=float(config["max_depth"]),
                unprocessing_config=config,
            )
        return KittiHalfresRGBDepthDataset(
            filelist_path=args.kitti_val_split,
            kitti_base=args.kitti_base,
            min_depth=float(config["min_depth"]),
            max_depth=float(config["max_depth"]),
        )

    if is_raw:
        return VKITTI2Raw(
            filelist_path=config["vkitti_val_list"],
            mode="val",
            size=(int(config["input_height"]), int(config["input_width"])),
            min_depth=float(config["min_depth"]),
            max_depth=float(config["max_depth"]),
            randomize_unprocessing=False,
            unprocessing_config=config,
            hflip_prob=0.0,
            include_rgb_input=True,
            include_rgb_preview=True,
            include_geometry=True,
            raw_storage_format=config["raw_storage_format"],
            fullres_even_policy=config["fullres_even_policy"],
            rgb_input_space=config["rgb_input_space"],
            depth_target_space=config["depth_target_space"],
        )
    return VKITTI2HalfresRGBDepth(
        filelist_path=config["vkitti_val_list"],
        mode="val",
        size=(int(config["input_height"]), int(config["input_width"])),
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
        hflip_prob=0.0,
        include_geometry=True,
        raw_storage_format=config["raw_storage_format"],
        fullres_even_policy=config["fullres_even_policy"],
        rgb_input_space=config["rgb_input_space"],
        depth_target_space=config["depth_target_space"],
    )


def parse_indices(arg: str | None, dataset_len: int) -> list[int] | None:
    if not arg:
        return None
    indices = [int(x.strip()) for x in arg.split(",") if x.strip()]
    out: list[int] = []
    for idx in indices:
        if idx < 0 or idx >= dataset_len:
            raise IndexError(f"Sample index {idx} out of range for dataset length {dataset_len}")
        if idx not in out:
            out.append(idx)
    return out


def vkitti_variant_from_path(path: str | Path) -> str:
    parts = Path(path).parts
    for i, part in enumerate(parts[:-1]):
        if part.startswith("Scene") and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def frame_from_path(path: str | Path) -> int:
    stem = Path(path).stem
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return -1


def select_vkitti_condition_indices(filelist_path: str | Path, max_panels: int) -> tuple[list[int], dict[str, Any]]:
    by_variant: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    with Path(filelist_path).expanduser().resolve().open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            image_path, _ = line.split()
            by_variant[vkitti_variant_from_path(image_path)].append((frame_from_path(image_path), idx, image_path))

    selected: list[int] = []
    variant_records: list[dict[str, Any]] = []
    for variant in VKITTI_VARIANT_ORDER:
        rows = sorted(by_variant.get(variant, []))
        if not rows:
            continue
        frame, idx, image_path = rows[len(rows) // 2]
        selected.append(int(idx))
        variant_records.append({"variant": variant, "dataset_index": int(idx), "frame": int(frame), "image_path": image_path})
        if len(selected) >= max_panels:
            break

    if len(selected) < max_panels:
        all_rows = sorted((idx, path) for rows in by_variant.values() for _, idx, path in rows)
        for idx, path in all_rows:
            if idx not in selected:
                selected.append(int(idx))
                variant_records.append(
                    {
                        "variant": vkitti_variant_from_path(path),
                        "dataset_index": int(idx),
                        "frame": frame_from_path(path),
                        "image_path": path,
                    }
                )
            if len(selected) >= max_panels:
                break

    return selected[:max_panels], {
        "mode": "vkitti_condition_median_frame",
        "variant_counts": {k: len(v) for k, v in sorted(by_variant.items())},
        "selected_variants": variant_records[:max_panels],
        "dark_variant_present": any(k in by_variant for k in ("dark", "night")),
    }


def select_default_indices(args: argparse.Namespace, config: dict[str, Any], dataset_len: int) -> tuple[list[int], dict[str, Any]]:
    explicit = parse_indices(args.sample_indices, dataset_len)
    if explicit is not None:
        return explicit[: int(args.max_panels)], {"mode": "explicit", "requested": explicit}
    if args.dataset == "vkitti":
        return select_vkitti_condition_indices(config["vkitti_val_list"], int(args.max_panels))
    if int(args.max_panels) <= 1:
        return [0], {"mode": "uniform"}
    return [int(x) for x in np.linspace(0, dataset_len - 1, int(args.max_panels))], {"mode": "uniform"}


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def stretch_channels_01(arr: np.ndarray, *, channel_axis: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.asarray(arr, dtype=np.float32)
    if channel_axis < 0:
        channel_axis += data.ndim
    if channel_axis != 0:
        data = np.moveaxis(data, channel_axis, 0)
    out = np.zeros_like(data, dtype=np.float32)
    ranges = []
    for c in range(data.shape[0]):
        values = data[c]
        finite = values[np.isfinite(values)]
        if finite.size:
            lo = float(np.percentile(finite, 1.0))
            hi = float(np.percentile(finite, 99.0))
        else:
            lo, hi = 0.0, 1.0
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        out[c] = np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        ranges.append({"p1": lo, "p99": hi})
    return out, {"stretch": "per_channel_p1_p99_to_0_1", "ranges": ranges}


def collect_record(
    *,
    dataset: Any,
    model: torch.nn.Module,
    config: dict[str, Any],
    idx: int,
    epoch: int,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample = dataset[idx]
    if sample.get("status") not in (None, "ok"):
        raise RuntimeError(f"Sample {idx} failed to load: {sample.get('error')}")

    image = sample["image"].unsqueeze(0).to(device).float()
    depth_t = sample["depth"].unsqueeze(0).to(device).float()
    valid_t = sample["valid_mask"].unsqueeze(0).to(device).bool()
    valid_t = valid_t & (depth_t >= float(config["min_depth"])) & (depth_t <= float(config["max_depth"]))
    batch = {"image": image, "valid_mask": valid_t}
    is_raw = str(config["input_domain"]) == "raw4"
    if is_raw:
        batch["raw"] = sample["raw"].unsqueeze(0).to(device).float()

    amp_enabled = bool(config.get("amp", False)) and device.type == "cuda"
    amp_dtype = torch.float16 if str(config.get("amp_dtype", "bf16")) == "fp16" else torch.bfloat16
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        out = model(batch)

    depth = depth_t[0].detach().cpu().numpy().astype(np.float32)
    valid = valid_t[0].detach().cpu().numpy().astype(bool)
    pred = out["pred"][0].float().detach().cpu().numpy().astype(np.float32)
    d0 = (float(config["d0_sign"]) * out["D0"][0].float()).detach().cpu().numpy().astype(np.float32)
    aligned_final, _ = affine_align_disp(depth, pred, valid)
    aligned_d0, _ = affine_align_disp(depth, d0, valid)
    aligned_final = aligned_final.astype(np.float32)
    aligned_d0 = aligned_d0.astype(np.float32)
    final_eval = clip_metric_depth_for_eval(
        aligned_final,
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
    )
    d0_eval = clip_metric_depth_for_eval(
        aligned_d0,
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
    )

    eval_valid_final = valid & np.isfinite(final_eval) & (final_eval > 0.0) & (depth > 0.0)
    eval_valid_d0 = valid & np.isfinite(d0_eval) & (d0_eval > 0.0) & (depth > 0.0)
    eval_valid = eval_valid_final & eval_valid_d0
    err_final = np.zeros_like(depth, dtype=np.float32)
    err_d0 = np.zeros_like(depth, dtype=np.float32)
    err_final[eval_valid_final] = np.abs(final_eval[eval_valid_final] - depth[eval_valid_final]) / np.clip(
        depth[eval_valid_final], 1e-6, None
    )
    err_d0[eval_valid_d0] = np.abs(d0_eval[eval_valid_d0] - depth[eval_valid_d0]) / np.clip(
        depth[eval_valid_d0], 1e-6, None
    )

    metrics_final = compute_metrics(
        depth,
        aligned_final,
        valid,
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
    )
    metrics_d0 = compute_metrics(
        depth,
        aligned_d0,
        valid,
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
    )
    depth_vmin, depth_vmax = choose_depth_range(
        depth,
        valid,
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
        pmin=float(args.depth_pmin),
        pmax=float(args.depth_pmax),
    )

    gate = out["gate"][0].float().detach().cpu().numpy().astype(np.float32)
    delta = out["delta"][0].float().detach().cpu().numpy().astype(np.float32)
    gate_delta = gate * delta
    rgb = sample["rgb_preview"].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

    if is_raw:
        raw = sample["raw"].detach().cpu().numpy().astype(np.float32)
        aux_data = raw
        aux_channels = ("R", "Gr", "Gb", "B")
        aux_colors = DEFAULT_RAW_COLORS
        aux_axis = 0
        aux_title = "RAW distribution"
        aux_subtitle = "min/p50/p99/max"
        aux_stats = summarize_channels(aux_data, channels=aux_channels, channel_axis=aux_axis)
        aux_stretch_meta = {"stretch": "none"}
        input_preview = packed_raw_preview(raw)
        input_preview_title = "Pseudo-RAW preview"
        input_preview_subtitle = "p1..p99 display stretch"
    else:
        head_input = out["head_input"][0].float().detach().cpu().numpy().astype(np.float32)
        aux_data, aux_stretch_meta = stretch_channels_01(head_input, channel_axis=0)
        if str(config["residual_feature_source"]) == "rgb":
            aux_channels = ("D0n", "R", "G", "B")
            aux_colors = ((230, 230, 80), *DEFAULT_RGB_COLORS)
        else:
            aux_channels = ("D0n",)
            aux_colors = ((230, 230, 80),)
        aux_axis = 0
        aux_title = "Head input dist"
        aux_subtitle = "p1..p99 stretch"
        aux_stats = summarize_channels(head_input, channels=aux_channels, channel_axis=0)
        d0_norm = out["D0_norm"][0].float().detach().cpu().numpy().astype(np.float32)
        finite = np.abs(d0_norm[valid & np.isfinite(d0_norm)])
        d0_vlim = float(np.percentile(finite, 99.0)) if finite.size else 1.0
        input_preview = (
            colorize_signed(
                d0_norm,
                np.ones_like(valid, dtype=bool),
                vlim=max(d0_vlim, 1e-6),
                cmap_name="coolwarm",
            ).astype(np.float32)
            / 255.0
        )
        input_preview_title = "D0 norm preview"
        input_preview_subtitle = "+/-p99 signed"

    return {
        "dataset_index": int(idx),
        "epoch": int(epoch),
        "variant": vkitti_variant_from_path(sample["image_path"]) if args.dataset == "vkitti" else None,
        "sample_name": str(sample["sample_name"]),
        "image_path": str(sample["image_path"]),
        "depth_path": str(sample["depth_path"]),
        "rgb": rgb,
        "rgb_stats": summarize_channels(rgb, channels=("R", "G", "B"), channel_axis=2),
        "input_preview": input_preview,
        "input_preview_title": input_preview_title,
        "input_preview_subtitle": input_preview_subtitle,
        "aux_data": aux_data,
        "aux_channels": aux_channels,
        "aux_colors": aux_colors,
        "aux_axis": aux_axis,
        "aux_title": aux_title,
        "aux_subtitle": aux_subtitle,
        "aux_stats": aux_stats,
        "aux_stretch_meta": aux_stretch_meta,
        "depth": depth,
        "gt_valid": valid,
        "valid": eval_valid,
        "aligned_d0": aligned_d0,
        "aligned_final": aligned_final,
        "err_d0": err_d0,
        "err_final": err_final,
        "gate": gate,
        "delta": delta,
        "gate_delta": gate_delta,
        "depth_vmin": float(depth_vmin),
        "depth_vmax": float(depth_vmax),
        "D0": {k: finite_float(None if metrics_d0 is None else metrics_d0.get(k)) for k in ("abs_rel", "d1")},
        "final": {k: finite_float(None if metrics_final is None else metrics_final.get(k)) for k in ("abs_rel", "d1")},
    }


def make_panel(record: dict[str, Any], args: argparse.Namespace, residual_vlim: float) -> tuple[Image.Image, dict[str, Any]]:
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
    d0_metric = f"{depth_range} absrel={record['D0']['abs_rel']:.3f} delta1={record['D0']['d1']:.3f}"
    final_metric = f"{depth_range} absrel={record['final']['abs_rel']:.3f} delta1={record['final']['d1']:.3f}"

    depth_mid = 0.5 * (float(record["depth_vmin"]) + float(record["depth_vmax"]))
    error_mid = 0.5 * float(args.error_max_abs_rel)
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
    aux_hist, aux_dist_meta = draw_distribution_tile(
        record["aux_data"],
        channels=record["aux_channels"],
        colors=record["aux_colors"],
        channel_axis=record["aux_axis"],
        bins=int(args.hist_bins),
        width=tile_w,
        height=tile_h,
        font=font,
        small_font=small_font,
    )

    depth_bar = colormap_gradient("Spectral_r")
    error_bar = colormap_gradient("magma")
    residual_bar = colormap_gradient("coolwarm")
    gate_bar = colormap_gradient("viridis")
    improve_bar = improvement_gradient(vlim=float(args.error_max_abs_rel))
    depth_tile = lambda arr, valid: add_horizontal_colorbar(  # noqa: E731
        image_from_array(colorize_depth(arr, valid, vmin=record["depth_vmin"], vmax=record["depth_vmax"]), tile_width=tile_w, tile_height=tile_h),
        colors=depth_bar,
        labels=depth_labels,
        font=bar_font,
    )
    error_tile = lambda arr: add_horizontal_colorbar(  # noqa: E731
        image_from_array(colorize_error(arr, display_valid, vmax=float(args.error_max_abs_rel)), tile_width=tile_w, tile_height=tile_h),
        colors=error_bar,
        labels=error_labels,
        font=bar_font,
    )

    tiles: list[tuple[str, str, Image.Image]] = [
        (
            "RGB input",
            "",
            image_from_array(np.clip(record["rgb"] * 255.0, 0.0, 255.0).round().astype(np.uint8), tile_width=tile_w, tile_height=tile_h),
        ),
        (
            record["input_preview_title"],
            record["input_preview_subtitle"],
            image_from_array(np.clip(record["input_preview"] * 255.0, 0.0, 255.0).round().astype(np.uint8), tile_width=tile_w, tile_height=tile_h),
        ),
        ("RGB distribution", "min/p50/p99/max", rgb_hist),
        (record["aux_title"], record["aux_subtitle"], aux_hist),
        ("DAV2-S depth", d0_metric, depth_tile(record["aligned_d0"], display_valid)),
        (f"Ours epoch{record['epoch']:02d}", final_metric, depth_tile(record["aligned_final"], display_valid)),
        (
            "Residual gate*delta",
            residual_range,
            add_horizontal_colorbar(
                image_from_array(
                    colorize_signed(record["gate_delta"], display_valid, vlim=residual_vlim, cmap_name="coolwarm"),
                    tile_width=tile_w,
                    tile_height=tile_h,
                ),
                colors=residual_bar,
                labels=residual_labels,
                font=bar_font,
            ),
        ),
        (
            "Gate",
            "0..1",
            add_horizontal_colorbar(
                image_from_array(colorize_gate(record["gate"], display_valid), tile_width=tile_w, tile_height=tile_h),
                colors=gate_bar,
                labels=gate_labels,
                font=bar_font,
            ),
        ),
        ("DAV2 error", error_range, error_tile(record["err_d0"])),
        ("Ours error", error_range, error_tile(record["err_final"])),
        (
            "Err improve +green",
            improve_range,
            add_horizontal_colorbar(
                image_from_array(
                    colorize_improvement(record["err_d0"] - record["err_final"], display_valid, vlim=float(args.error_max_abs_rel)),
                    tile_width=tile_w,
                    tile_height=tile_h,
                ),
                colors=improve_bar,
                labels=improve_labels,
                font=bar_font,
            ),
        ),
        ("GT depth", depth_range, depth_tile(record["depth"], gt_valid)),
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
    return canvas, {"rgb_distribution": rgb_dist_meta, "aux_distribution": aux_dist_meta}


def safe_name(record: dict[str, Any], dataset_name: str) -> str:
    sample = str(record["sample_name"]).replace("/", "_")
    if dataset_name == "vkitti":
        variant = str(record.get("variant") or "unknown")
        return f"vkitti_{variant}_validx{record['dataset_index']:04d}_{sample}"
    return f"kitti_{sample}"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if args.max_panels <= 0:
        raise ValueError("--max-panels must be positive")

    config = load_run_config(run_dir)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")
    dataset = build_dataset(config, args)
    indices, selection_meta = select_default_indices(args, config, len(dataset))
    model = build_model(config, checkpoint, device)
    epoch = infer_epoch(checkpoint)

    records = []
    for order, idx in enumerate(indices, start=1):
        record = collect_record(
            dataset=dataset,
            model=model,
            config=config,
            idx=int(idx),
            epoch=epoch,
            device=device,
            args=args,
        )
        records.append(record)
        print(f"[collect] {order:02d}/{len(indices):02d} idx={idx} sample={record['sample_name']}", flush=True)

    residual_values = []
    for record in records:
        values = np.abs(record["gate_delta"][record["valid"] & np.isfinite(record["gate_delta"])])
        if values.size:
            residual_values.append(values.astype(np.float32))
    residual_vlim = float(np.percentile(np.concatenate(residual_values), 99.0)) if residual_values else 1.0
    residual_vlim = max(residual_vlim, 1e-6)

    manifest_records = []
    for order, record in enumerate(records, start=1):
        panel_path = output_dir / f"{order:02d}_{safe_name(record, args.dataset)}_epoch{epoch:02d}_panel.jpg"
        panel, dist_meta = make_panel(record, args, residual_vlim)
        panel.save(panel_path, quality=95)
        manifest_records.append(
            {
                "order": order,
                "dataset_index": int(record["dataset_index"]),
                "variant": record.get("variant"),
                "sample_name": record["sample_name"],
                "image_path": record["image_path"],
                "depth_path": record["depth_path"],
                "panel_path": str(panel_path),
                "final": record["final"],
                "D0": record["D0"],
                "rgb_stats": record["rgb_stats"],
                "aux_stats": record["aux_stats"],
                "aux_stretch_meta": record["aux_stretch_meta"],
                "distribution_meta": dist_meta,
            }
        )
        print(f"[write] {panel_path}", flush=True)

    manifest = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "dataset": args.dataset,
        "output_dir": str(output_dir),
        "epoch": int(epoch),
        "input_domain": config.get("input_domain"),
        "front_end": config.get("front_end"),
        "residual_feature_source": config.get("residual_feature_source"),
        "residual_head_d0_mode": config.get("residual_head_d0_mode", "concat"),
        "selected_indices": [int(x) for x in indices],
        "selection": selection_meta,
        "panel_layout": [
            "RGB input",
            "input preview",
            "RGB distribution",
            "model/raw input distribution",
            "DAV2-S depth",
            "Ours",
            "Residual gate*delta",
            "Gate",
            "DAV2 error",
            "Ours error",
            "Err improve +green",
            "GT depth",
        ],
        "visualization_defaults": {
            "layout": "3x4",
            "depth_cmap": "Spectral_r",
            "depth_tiles_share_range": ["DAV2-S depth", "Ours", "GT depth"],
            "error_cmap": "magma",
            "error_vmax_abs_rel": float(args.error_max_abs_rel),
            "residual_cmap": "coolwarm",
            "residual_vmax_abs_p99_selected": float(residual_vlim),
            "gate_cmap": "viridis",
            "improvement_colormap": "black_zero_green_positive_red_negative",
        },
        "records": manifest_records,
    }
    save_json(output_dir / "manifest.json", manifest)
    print(f"[done] wrote {len(records)} panels to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
