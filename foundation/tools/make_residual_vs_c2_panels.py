#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
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
from foundation.engine.models import (
    build_c2_frozen_incremental_residual_model,
    build_dav2_residual_control_model,
    build_raw_residual_dav2_model,
)
from foundation.engine.transforms import resolve_unprocessing_config
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
from foundation.tools.residual_training_common import resolve_model_state, save_json, strip_module_prefix


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make VKITTI method-vs-C2 residual panels.")
    parser.add_argument("--c2-run-dir", required=True)
    parser.add_argument("--c2-checkpoint", required=True)
    parser.add_argument("--method-run-dir", required=True)
    parser.add_argument("--method-checkpoint", required=True)
    parser.add_argument("--method-kind", required=True, choices=["raw", "nseries", "control"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-indices", default=None)
    parser.add_argument("--max-panels", type=int, default=8)
    parser.add_argument("--error-max-abs-rel", type=float, default=0.75)
    parser.add_argument("--depth-pmin", type=float, default=1.0)
    parser.add_argument("--depth-pmax", type=float, default=99.0)
    parser.add_argument("--tile-width", type=int, default=414)
    parser.add_argument("--tile-height", type=int, default=125)
    parser.add_argument("--header-height", type=int, default=30)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(config: dict[str, Any]):
    if str(config.get("input_domain")) == "raw4":
        val_config = dict(config)
        if val_config.get("unprocessing_method") == "old_brooks_preset":
            val_config["randomize_unprocessing"] = False
            val_config = resolve_unprocessing_config(val_config)
        return VKITTI2Raw(
            filelist_path=config["vkitti_val_list"],
            mode="val",
            size=(int(config["input_height"]), int(config["input_width"])),
            min_depth=float(config["min_depth"]),
            max_depth=float(config["max_depth"]),
            unprocessing_config=val_config,
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


def build_c2_model(config: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    base = DepthAnythingV2(**MODEL_CONFIGS[str(config["encoder"])])
    model = build_dav2_residual_control_model(
        base,
        residual_feature_source="d0",
        residual_alpha=float(config["residual_alpha"]),
        d0_sign=int(config["d0_sign"]),
        sensor_hw=(int(config["input_height"]), int(config["input_width"])),
        backbone_hw=None,
    )
    ckpt_obj = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(strip_module_prefix(resolve_model_state(ckpt_obj)), strict=True)
    return model.to(device).eval()


def build_method_model(config: dict[str, Any], checkpoint: Path, kind: str, device: torch.device) -> torch.nn.Module:
    if kind == "raw":
        base = DepthAnythingV2(**MODEL_CONFIGS[str(config["encoder"])])
        model = build_raw_residual_dav2_model(
            base,
            residual_feature_source=str(config["residual_feature_source"]),
            residual_head_d0_mode=str(config.get("residual_head_d0_mode", "concat")),
            residual_alpha=float(config["residual_alpha"]),
            d0_sign=int(config["d0_sign"]),
            sensor_hw=(int(config["input_height"]), int(config["input_width"])),
            backbone_hw=None,
        )
    elif kind == "control":
        base = DepthAnythingV2(**MODEL_CONFIGS[str(config["encoder"])])
        model = build_dav2_residual_control_model(
            base,
            residual_feature_source=str(config["residual_feature_source"]),
            residual_alpha=float(config["residual_alpha"]),
            d0_sign=int(config["d0_sign"]),
            sensor_hw=(int(config["input_height"]), int(config["input_width"])),
            backbone_hw=None,
        )
    else:
        c2_config = dict(config)
        c2_model = build_c2_model(c2_config, Path(str(config["c2_checkpoint"])).expanduser().resolve(), torch.device("cpu"))
        model = build_c2_frozen_incremental_residual_model(
            c2_model,
            method_id=str(config["method_id"]),
            incremental_feature_source=str(config["incremental_feature_source"]),
            delta_condition=str(config["delta_condition"]),
            gate_condition=str(config["gate_condition"]),
            raw_feature_encoder_trainable=str(config["raw_feature_encoder_trainable"]),
            residual_alpha=float(config["residual_alpha"]),
            lambda_lp=float(config["lambda_lp"]),
            lowpass_kernel=int(config["lowpass_kernel"]),
            sensor_hw=(int(config["input_height"]), int(config["input_width"])),
            backbone_hw=None,
        )
    ckpt_obj = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(strip_module_prefix(resolve_model_state(ckpt_obj)), strict=True)
    return model.to(device).eval()


def parse_indices(arg: str | None, dataset_len: int, max_panels: int) -> list[int]:
    if arg:
        indices = [int(x.strip()) for x in arg.split(",") if x.strip()]
    else:
        indices = [int(round(x)) for x in np.linspace(0, dataset_len - 1, max(int(max_panels), 1))]
    out = []
    for idx in indices:
        if idx < 0 or idx >= dataset_len:
            raise IndexError(f"Sample index {idx} out of range for validation set length {dataset_len}")
        if idx not in out:
            out.append(idx)
    return out[: int(max_panels)]


def make_panel(record: dict[str, Any], args: argparse.Namespace, residual_vlim: float) -> Image.Image:
    tile_w = int(args.tile_width)
    tile_h = int(args.tile_height)
    header_h = int(args.header_height)
    font = load_font(12)
    small_font = load_font(10)
    canvas = Image.new("RGB", (tile_w * 3, (tile_h + header_h) * 3), (0, 0, 0))
    valid = record["valid"]
    depth_range = f"{record['depth_vmin']:.2f}..{record['depth_vmax']:.2f}m"
    error_range = f"0..{float(args.error_max_abs_rel):.2f} absrel"
    residual_range = f"+/-{float(residual_vlim):.3f}"
    tiles = [
        ("RGB", "", np.clip(record["rgb"] * 255.0, 0.0, 255.0).round().astype(np.uint8)),
        ("GT depth", depth_range, colorize_depth(record["depth"], valid, vmin=record["depth_vmin"], vmax=record["depth_vmax"])),
        ("C2 depth", depth_range, colorize_depth(record["aligned_c2"], valid, vmin=record["depth_vmin"], vmax=record["depth_vmax"])),
        ("Method depth", depth_range, colorize_depth(record["aligned_method"], valid, vmin=record["depth_vmin"], vmax=record["depth_vmax"])),
        ("C2 error", error_range, colorize_error(record["err_c2"], valid, vmax=float(args.error_max_abs_rel))),
        ("Method error", error_range, colorize_error(record["err_method"], valid, vmax=float(args.error_max_abs_rel))),
        ("Improve over C2", "+green / red worse", colorize_improvement(record["err_c2"] - record["err_method"], valid, vlim=float(args.error_max_abs_rel))),
        ("Method gate*delta", residual_range, colorize_signed(record["gate_delta"], valid, vlim=residual_vlim, cmap_name="coolwarm")),
        ("Method gate", "0..1", colorize_gate(record["gate"], valid)),
    ]
    for i, (title, subtitle, rgb) in enumerate(tiles):
        draw_tile(
            canvas,
            col=i % 3,
            row=i // 3,
            tile=image_from_array(rgb, tile_width=tile_w, tile_height=tile_h),
            title=title,
            subtitle=subtitle,
            tile_width=tile_w,
            tile_height=tile_h,
            header_height=header_h,
            font=font,
            small_font=small_font,
        )
    return canvas


def collect_record(
    *,
    sample: dict[str, Any],
    idx: int,
    c2_model: torch.nn.Module,
    method_model: torch.nn.Module,
    method_config: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    image = sample["image"].unsqueeze(0).to(device).float()
    raw = sample.get("raw")
    if raw is not None:
        raw = raw.unsqueeze(0).to(device).float()
    depth_t = sample["depth"].unsqueeze(0).to(device).float()
    valid_t = sample["valid_mask"].unsqueeze(0).to(device).bool()
    valid_t = valid_t & (depth_t >= float(method_config["min_depth"])) & (depth_t <= float(method_config["max_depth"]))
    if int(valid_t[0].sum().item()) < 128:
        return None
    method_batch = {"image": image, "valid_mask": valid_t}
    if raw is not None:
        method_batch["raw"] = raw
    amp_enabled = bool(method_config.get("amp", False)) and device.type == "cuda"
    amp_dtype = torch.float16 if method_config.get("amp_dtype") == "fp16" else torch.bfloat16
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        c2_out = c2_model({"image": image, "valid_mask": valid_t})
        method_out = method_model(method_batch)
    depth = depth_t[0].detach().cpu().numpy().astype(np.float32)
    valid = valid_t[0].detach().cpu().numpy().astype(bool)
    c2_disp = c2_out["pred"][0].float().detach().cpu().numpy().astype(np.float32)
    method_disp = method_out["pred"][0].float().detach().cpu().numpy().astype(np.float32)
    aligned_c2, _ = affine_align_disp(depth, c2_disp, valid)
    aligned_method, _ = affine_align_disp(depth, method_disp, valid)
    c2_metrics = compute_metrics(depth, aligned_c2, valid, min_depth=float(method_config["min_depth"]), max_depth=float(method_config["max_depth"]))
    method_metrics = compute_metrics(depth, aligned_method, valid, min_depth=float(method_config["min_depth"]), max_depth=float(method_config["max_depth"]))
    if c2_metrics is None or method_metrics is None:
        return None
    aligned_c2_eval = clip_metric_depth_for_eval(aligned_c2, min_depth=float(method_config["min_depth"]), max_depth=float(method_config["max_depth"]))
    aligned_method_eval = clip_metric_depth_for_eval(aligned_method, min_depth=float(method_config["min_depth"]), max_depth=float(method_config["max_depth"]))
    eval_valid = valid & np.isfinite(aligned_c2_eval) & np.isfinite(aligned_method_eval) & (aligned_c2_eval > 0.0) & (aligned_method_eval > 0.0) & (depth > 0.0)
    err_c2 = np.zeros_like(depth, dtype=np.float32)
    err_method = np.zeros_like(depth, dtype=np.float32)
    err_c2[eval_valid] = np.abs(aligned_c2_eval[eval_valid] - depth[eval_valid]) / np.clip(depth[eval_valid], 1e-6, None)
    err_method[eval_valid] = np.abs(aligned_method_eval[eval_valid] - depth[eval_valid]) / np.clip(depth[eval_valid], 1e-6, None)
    gate = method_out.get("gate", torch.zeros_like(method_out["pred"]))[0].float().detach().cpu().numpy().astype(np.float32)
    if "delta_effective" in method_out:
        gate_delta_t = method_out["gate"].float() * method_out["delta_effective"].float()
    else:
        gate_delta_t = method_out["gate"].float() * method_out["delta"].float()
    gate_delta = gate_delta_t[0].detach().cpu().numpy().astype(np.float32)
    depth_vmin, depth_vmax = choose_depth_range(
        depth,
        valid,
        min_depth=float(method_config["min_depth"]),
        max_depth=float(method_config["max_depth"]),
        pmin=float(args.depth_pmin),
        pmax=float(args.depth_pmax),
    )
    return {
        "dataset_index": int(idx),
        "sample_name": str(sample["sample_name"]),
        "image_path": str(sample["image_path"]),
        "depth_path": str(sample["depth_path"]),
        "rgb": sample["rgb_preview"].permute(1, 2, 0).numpy().astype(np.float32),
        "depth": depth,
        "valid": eval_valid,
        "aligned_c2": aligned_c2.astype(np.float32),
        "aligned_method": aligned_method.astype(np.float32),
        "err_c2": err_c2,
        "err_method": err_method,
        "gate": gate,
        "gate_delta": gate_delta,
        "depth_vmin": depth_vmin,
        "depth_vmax": depth_vmax,
        "c2_abs_rel": float(c2_metrics["abs_rel"]),
        "method_abs_rel": float(method_metrics["abs_rel"]),
        "method_minus_c2_abs_rel": float(method_metrics["abs_rel"] - c2_metrics["abs_rel"]),
    }


def main() -> None:
    args = parse_args()
    c2_run_dir = Path(args.c2_run_dir).expanduser().resolve()
    method_run_dir = Path(args.method_run_dir).expanduser().resolve()
    c2_checkpoint = Path(args.c2_checkpoint).expanduser().resolve()
    method_checkpoint = Path(args.method_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    c2_config = load_json(c2_run_dir / "config.json")
    method_config = load_json(method_run_dir / "config.json")
    method_c2_checkpoint_override = None
    if args.method_kind == "nseries":
        configured_c2 = method_config.get("c2_checkpoint")
        configured_c2_path = Path(str(configured_c2)).expanduser().resolve() if configured_c2 else None
        if configured_c2_path is not None and configured_c2_path.is_file() and configured_c2_path != c2_checkpoint:
            raise ValueError(
                "N-series method config c2_checkpoint differs from --c2-checkpoint: "
                f"{configured_c2_path} vs {c2_checkpoint}"
            )
        if configured_c2_path is None or configured_c2_path != c2_checkpoint:
            method_c2_checkpoint_override = {
                "configured": str(configured_c2) if configured_c2 else None,
                "used": str(c2_checkpoint),
                "reason": "use explicit --c2-checkpoint for frozen C2 in N-series visualization",
            }
            method_config["c2_checkpoint"] = str(c2_checkpoint)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = build_dataset(method_config)
    c2_model = build_c2_model(c2_config, c2_checkpoint, device)
    method_model = build_method_model(method_config, method_checkpoint, args.method_kind, device)
    indices = parse_indices(args.sample_indices, len(dataset), args.max_panels)
    records: list[dict[str, Any]] = []
    for idx in indices:
        record = collect_record(
            sample=dataset[idx],
            idx=idx,
            c2_model=c2_model,
            method_model=method_model,
            method_config=method_config,
            device=device,
            args=args,
        )
        if record is not None:
            records.append(record)
    residual_values = [np.abs(record["gate_delta"][record["valid"]]) for record in records if bool(record["valid"].any())]
    residual_vlim = float(np.percentile(np.concatenate(residual_values), 99.0)) if residual_values else 1.0
    residual_vlim = max(residual_vlim, 1e-6)
    manifest_records = []
    for order, record in enumerate(records, start=1):
        safe_name = record["sample_name"].replace("/", "_")
        panel_path = output_dir / f"{order:02d}_validx{record['dataset_index']:04d}_{safe_name}_vs_c2.jpg"
        panel = make_panel(record, args, residual_vlim)
        panel.save(panel_path, quality=95)
        manifest_records.append(
            {
                "order": int(order),
                "dataset_index": int(record["dataset_index"]),
                "sample_name": record["sample_name"],
                "panel_path": str(panel_path),
                "c2_abs_rel": record["c2_abs_rel"],
                "method_abs_rel": record["method_abs_rel"],
                "method_minus_c2_abs_rel": record["method_minus_c2_abs_rel"],
            }
        )
        print(f"wrote {panel_path}", flush=True)
    save_json(
        output_dir / "manifest.json",
        {
            "c2_run_dir": str(c2_run_dir),
            "c2_checkpoint": str(c2_checkpoint),
            "method_run_dir": str(method_run_dir),
            "method_checkpoint": str(method_checkpoint),
            "method_kind": args.method_kind,
            "method_c2_checkpoint_override": method_c2_checkpoint_override,
            "selected_indices": indices,
            "records": manifest_records,
            "panel_layout": [
                "RGB",
                "GT depth",
                "C2 depth",
                "method depth",
                "C2 absrel error",
                "method absrel error",
                "improvement over C2",
                "method gate*delta",
                "method gate",
            ],
        },
    )


if __name__ == "__main__":
    main()
