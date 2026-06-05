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
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from depth_anything_v2.dpt import DepthAnythingV2
from foundation.engine.datasets import VKITTI2Raw
from foundation.engine.models import build_raw_residual_dav2_model


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make VKITTI RAW residual qualitative comparison panels.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-indices", default=None, help="Comma-separated validation dataset indices.")
    parser.add_argument("--max-panels", type=int, default=8)
    parser.add_argument("--error-max-abs-rel", type=float, default=0.75)
    parser.add_argument("--depth-pmin", type=float, default=1.0)
    parser.add_argument("--depth-pmax", type=float, default=99.0)
    parser.add_argument("--tile-width", type=int, default=414)
    parser.add_argument("--tile-height", type=int, default=125)
    parser.add_argument("--header-height", type=int, default=30)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def resolve_model_state(ckpt_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    return ckpt_obj


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        p = Path(path)
        if p.is_file():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def choose_depth_range(
    depth: np.ndarray,
    valid: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    pmin: float,
    pmax: float,
) -> tuple[float, float]:
    mask = valid & np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    values = depth[mask].astype(np.float64)
    if values.size < 10:
        return float(min_depth), float(max_depth)
    vmin = max(float(min_depth), float(np.percentile(values, pmin)))
    vmax = min(float(max_depth), float(np.percentile(values, pmax)))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmin = max(float(min_depth), float(np.nanmin(values)))
        vmax = min(float(max_depth), float(np.nanmax(values)))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        return float(min_depth), float(max_depth)
    return float(vmin), float(vmax)


def clip_metric_depth_for_eval(depth: np.ndarray, *, min_depth: float, max_depth: float) -> np.ndarray:
    out = np.asarray(depth, dtype=np.float32).copy()
    finite = np.isfinite(out)
    out[finite] = np.clip(out[finite], float(min_depth), float(max_depth))
    return out


def colorize_depth(depth: np.ndarray, valid: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    clipped = np.clip(np.where(np.isfinite(values), values, vmin), vmin, vmax)
    norm = (clipped - float(vmin)) / max(float(vmax) - float(vmin), 1e-6)
    rgb = (colormaps["Spectral_r"](norm)[..., :3] * 255.0).round().astype(np.uint8)
    rgb[~valid] = np.array([24, 24, 24], dtype=np.uint8)
    return rgb


def colorize_error(error: np.ndarray, valid: np.ndarray, *, vmax: float) -> np.ndarray:
    values = np.clip(np.where(np.isfinite(error), error, 0.0), 0.0, float(vmax))
    norm = values / max(float(vmax), 1e-6)
    rgb = (colormaps["magma"](norm)[..., :3] * 255.0).round().astype(np.uint8)
    rgb[~valid] = np.array([24, 24, 24], dtype=np.uint8)
    return rgb


def colorize_signed(values: np.ndarray, valid: np.ndarray, *, vlim: float, cmap_name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    norm = np.clip(arr / max(float(vlim), 1e-6), -1.0, 1.0)
    norm = (norm + 1.0) * 0.5
    rgb = (colormaps[cmap_name](norm)[..., :3] * 255.0).round().astype(np.uint8)
    rgb[~valid] = np.array([24, 24, 24], dtype=np.uint8)
    return rgb


def colorize_gate(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    norm = np.clip(np.where(np.isfinite(values), values, 0.0), 0.0, 1.0)
    rgb = (colormaps["viridis"](norm)[..., :3] * 255.0).round().astype(np.uint8)
    rgb[~valid] = np.array([24, 24, 24], dtype=np.uint8)
    return rgb


def colorize_improvement(values: np.ndarray, valid: np.ndarray, *, vlim: float) -> np.ndarray:
    arr = np.clip(np.where(np.isfinite(values), values, 0.0), -float(vlim), float(vlim))
    pos = np.clip(arr / max(float(vlim), 1e-6), 0.0, 1.0)
    neg = np.clip(-arr / max(float(vlim), 1e-6), 0.0, 1.0)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    rgb[..., 1] = (pos * 255.0).round().astype(np.uint8)
    rgb[..., 0] = (neg * 255.0).round().astype(np.uint8)
    rgb[..., 2] = (neg * 90.0).round().astype(np.uint8)
    rgb[~valid] = np.array([24, 24, 24], dtype=np.uint8)
    return rgb


def image_from_array(rgb: np.ndarray, *, tile_width: int, tile_height: int) -> Image.Image:
    return Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize((tile_width, tile_height), Image.Resampling.BILINEAR)


def format_range(lo: float, hi: float, unit: str = "") -> str:
    suffix = unit if unit else ""
    return f"{lo:.2f}..{hi:.2f}{suffix}"


def draw_tile(
    canvas: Image.Image,
    *,
    col: int,
    row: int,
    tile: Image.Image,
    title: str,
    subtitle: str,
    tile_width: int,
    tile_height: int,
    header_height: int,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x0 = col * tile_width
    y0 = row * (tile_height + header_height)
    draw.rectangle([x0, y0, x0 + tile_width, y0 + header_height], fill=(18, 18, 18))
    draw.text((x0 + 8, y0 + 3), title, fill=(245, 245, 245), font=font)
    if subtitle:
        draw.text((x0 + 8, y0 + 17), subtitle, fill=(190, 190, 190), font=small_font)
    canvas.paste(tile, (x0, y0 + header_height))


def make_panel(record: dict[str, Any], args: argparse.Namespace, residual_vlim: float) -> Image.Image:
    tile_w = int(args.tile_width)
    tile_h = int(args.tile_height)
    header_h = int(args.header_height)
    font = load_font(12)
    small_font = load_font(10)

    canvas = Image.new("RGB", (tile_w * 3, (tile_h + header_h) * 3), (0, 0, 0))
    valid = record["valid"]
    depth_range = format_range(record["depth_vmin"], record["depth_vmax"], "m")
    error_range = f"0..{float(args.error_max_abs_rel):.2f} absrel"
    improve_range = f"+green +/-{float(args.error_max_abs_rel):.2f}"
    residual_range = f"+/-{float(residual_vlim):.3f}"

    tiles = [
        (
            "RGB input",
            "",
            np.clip(record["rgb"] * 255.0, 0.0, 255.0).round().astype(np.uint8),
        ),
        (
            "GT depth",
            depth_range,
            colorize_depth(record["depth"], valid, vmin=record["depth_vmin"], vmax=record["depth_vmax"]),
        ),
        (
            "DAV2-S depth",
            depth_range,
            colorize_depth(record["aligned_d0"], valid, vmin=record["depth_vmin"], vmax=record["depth_vmax"]),
        ),
        (
            f"Ours epoch{record['epoch']}",
            depth_range,
            colorize_depth(record["aligned_final"], valid, vmin=record["depth_vmin"], vmax=record["depth_vmax"]),
        ),
        (
            "DAV2 error",
            error_range,
            colorize_error(record["err_d0"], valid, vmax=float(args.error_max_abs_rel)),
        ),
        (
            "Ours error",
            error_range,
            colorize_error(record["err_final"], valid, vmax=float(args.error_max_abs_rel)),
        ),
        (
            "Residual gate*delta",
            residual_range,
            colorize_signed(record["gate_delta"], valid, vlim=residual_vlim, cmap_name="coolwarm"),
        ),
        (
            "Gate",
            "0..1",
            colorize_gate(record["gate"], valid),
        ),
        (
            "Err improve +green",
            improve_range,
            colorize_improvement(record["err_d0"] - record["err_final"], valid, vlim=float(args.error_max_abs_rel)),
        ),
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


def parse_indices(arg: str | None, dataset_len: int, max_panels: int) -> list[int]:
    if arg:
        indices = [int(x.strip()) for x in arg.split(",") if x.strip()]
    else:
        if max_panels <= 1:
            indices = [0]
        else:
            indices = [int(round(x)) for x in np.linspace(0, dataset_len - 1, max_panels)]
    out: list[int] = []
    for idx in indices:
        if idx < 0 or idx >= dataset_len:
            raise IndexError(f"Sample index {idx} out of range for validation set length {dataset_len}")
        if idx not in out:
            out.append(idx)
    return out[:max_panels]


def load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(config: dict[str, Any]) -> VKITTI2Raw:
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


def build_model(config: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    base_model = DepthAnythingV2(**MODEL_CONFIGS[str(config["encoder"])])
    model = build_raw_residual_dav2_model(
        base_model,
        residual_feature_source=str(config["residual_feature_source"]),
        residual_head_d0_mode=str(config.get("residual_head_d0_mode", "concat")),
        residual_alpha=float(config["residual_alpha"]),
        d0_sign=int(config["d0_sign"]),
        sensor_hw=(int(config["input_height"]), int(config["input_width"])),
        backbone_hw=None,
    )
    ckpt_obj = torch.load(str(checkpoint), map_location="cpu")
    state = strip_module_prefix(resolve_model_state(ckpt_obj))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def collect_record(
    *,
    dataset: VKITTI2Raw,
    model: torch.nn.Module,
    config: dict[str, Any],
    idx: int,
    epoch: int,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    sample = dataset[idx]
    image = sample["image"].unsqueeze(0).to(device).float()
    raw = sample["raw"].unsqueeze(0).to(device).float()
    depth_t = sample["depth"].unsqueeze(0).to(device).float()
    valid_t = sample["valid_mask"].unsqueeze(0).to(device).bool()
    valid_t = valid_t & (depth_t >= float(config["min_depth"])) & (depth_t <= float(config["max_depth"]))
    if int(valid_t[0].sum().item()) < 128:
        return None

    amp_enabled = bool(config.get("amp", False)) and device.type == "cuda"
    amp_dtype = torch.float16 if config.get("amp_dtype") == "fp16" else torch.bfloat16
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        out = model({"image": image, "raw": raw, "valid_mask": valid_t})

    depth = depth_t[0].detach().cpu().numpy().astype(np.float32)
    valid = valid_t[0].detach().cpu().numpy().astype(bool)
    pred = out["pred"][0].float().detach().cpu().numpy().astype(np.float32)
    d0 = (float(config["d0_sign"]) * out["D0"][0].float()).detach().cpu().numpy().astype(np.float32)
    aligned_final, _ = affine_align_disp(depth, pred, valid)
    aligned_d0, _ = affine_align_disp(depth, d0, valid)
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

    eval_valid_final = valid & np.isfinite(aligned_final_eval) & (aligned_final_eval > 0.0) & (depth > 0.0)
    eval_valid_d0 = valid & np.isfinite(aligned_d0_eval) & (aligned_d0_eval > 0.0) & (depth > 0.0)
    eval_valid = valid & eval_valid_final & eval_valid_d0
    err_final = np.zeros_like(depth, dtype=np.float32)
    err_d0 = np.zeros_like(depth, dtype=np.float32)
    err_final[eval_valid] = np.abs(aligned_final_eval[eval_valid] - depth[eval_valid]) / np.clip(depth[eval_valid], 1e-6, None)
    err_d0[eval_valid] = np.abs(aligned_d0_eval[eval_valid] - depth[eval_valid]) / np.clip(depth[eval_valid], 1e-6, None)

    depth_vmin, depth_vmax = choose_depth_range(
        depth,
        valid,
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
        pmin=float(args.depth_pmin),
        pmax=float(args.depth_pmax),
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
    gate = out["gate"][0].float().detach().cpu().numpy().astype(np.float32)
    delta = out["delta"][0].float().detach().cpu().numpy().astype(np.float32)
    gate_delta = gate * delta
    rgb = sample["rgb_preview"].permute(1, 2, 0).numpy().astype(np.float32)

    return {
        "dataset_index": int(idx),
        "epoch": int(epoch),
        "sample_name": str(sample["sample_name"]),
        "image_path": str(sample["image_path"]),
        "depth_path": str(sample["depth_path"]),
        "rgb": rgb,
        "depth": depth,
        "valid": eval_valid,
        "aligned_d0": aligned_d0,
        "aligned_final": aligned_final,
        "err_d0": err_d0,
        "err_final": err_final,
        "gate": gate,
        "delta": delta,
        "gate_delta": gate_delta,
        "depth_vmin": depth_vmin,
        "depth_vmax": depth_vmax,
        "dav2_abs_rel": finite_float(None if metrics_d0 is None else metrics_d0.get("abs_rel")),
        "dav2_d1": finite_float(None if metrics_d0 is None else metrics_d0.get("d1")),
        "ours_abs_rel": finite_float(None if metrics_final is None else metrics_final.get("abs_rel")),
        "ours_d1": finite_float(None if metrics_final is None else metrics_final.get("d1")),
        "mean_gate": float(gate[eval_valid].mean()) if bool(eval_valid.any()) else None,
        "mean_abs_residual_gate_delta": float(np.abs(gate_delta[eval_valid]).mean()) if bool(eval_valid.any()) else None,
        "max_dav2_error": float(np.nanmax(err_d0[eval_valid])) if bool(eval_valid.any()) else None,
        "max_ours_error": float(np.nanmax(err_final[eval_valid])) if bool(eval_valid.any()) else None,
    }


def manifest_record(record: dict[str, Any], panel_path: Path, order: int, residual_vlim: float, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "order": int(order),
        "dataset_index": int(record["dataset_index"]),
        "sample_name": record["sample_name"],
        "image_path": record["image_path"],
        "depth_path": record["depth_path"],
        "panel_path": str(panel_path),
        "dav2_abs_rel": record["dav2_abs_rel"],
        "dav2_d1": record["dav2_d1"],
        "ours_abs_rel": record["ours_abs_rel"],
        "ours_d1": record["ours_d1"],
        "mean_gate": record["mean_gate"],
        "mean_abs_residual_gate_delta": record["mean_abs_residual_gate_delta"],
        "max_dav2_error": record["max_dav2_error"],
        "max_ours_error": record["max_ours_error"],
        "visualization": {
            "depth_cmap": "Spectral_r",
            "depth_range_scope": "per_panel_shared_by_gt_valid_percentiles",
            "depth_vmin": float(record["depth_vmin"]),
            "depth_vmax": float(record["depth_vmax"]),
            "depth_percentiles": [float(args.depth_pmin), float(args.depth_pmax)],
            "depth_tiles_share_range": ["GT depth", "DAV2-S depth", "Ours"],
            "error_cmap": "magma",
            "error_vmin": 0.0,
            "error_vmax_abs_rel": float(args.error_max_abs_rel),
            "error_tiles_share_range": ["DAV2 error", "Ours error"],
            "residual_cmap": "coolwarm",
            "residual_range_scope": "global_selected_samples_symmetric_p99_abs",
            "residual_vmin": -float(residual_vlim),
            "residual_vmax": float(residual_vlim),
            "gate_cmap": "viridis",
            "gate_vmin": 0.0,
            "gate_vmax": 1.0,
            "improvement_colormap": "black_zero_green_positive_red_negative",
            "improvement_vmin_abs_rel": -float(args.error_max_abs_rel),
            "improvement_vmax_abs_rel": float(args.error_max_abs_rel),
        },
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_run_config(run_dir)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = build_dataset(config)
    model = build_model(config, checkpoint, device)
    indices = parse_indices(args.sample_indices, len(dataset), int(args.max_panels))
    epoch = int(Path(checkpoint).stem.split("_")[-1]) if Path(checkpoint).stem.startswith("epoch_") else -1

    records: list[dict[str, Any]] = []
    for idx in indices:
        record = collect_record(
            dataset=dataset,
            model=model,
            config=config,
            idx=idx,
            epoch=epoch,
            device=device,
            args=args,
        )
        if record is not None:
            records.append(record)

    residual_values = []
    for record in records:
        valid = record["valid"]
        values = np.abs(record["gate_delta"][valid])
        if values.size:
            residual_values.append(values.astype(np.float32))
    if residual_values:
        residual_vlim = float(np.percentile(np.concatenate(residual_values), 99.0))
    else:
        residual_vlim = 1.0
    residual_vlim = max(residual_vlim, 1e-6)

    manifest_records = []
    for order, record in enumerate(records, start=1):
        safe_name = record["sample_name"].replace("/", "_")
        panel_path = output_dir / f"{order:02d}_validx{record['dataset_index']:04d}_{safe_name}_compare_epoch{epoch:02d}_3x3_sharedscale.jpg"
        panel = make_panel(record, args, residual_vlim)
        panel.save(panel_path, quality=95)
        manifest_records.append(manifest_record(record, panel_path, order, residual_vlim, args))
        print(f"wrote {panel_path}", flush=True)

    manifest = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "epoch": epoch,
        "selected_indices": indices,
        "visualization_defaults": {
            "depth_cmap": "Spectral_r",
            "depth_range_scope": "per_panel_shared_by_gt_valid_percentiles",
            "depth_percentiles": [float(args.depth_pmin), float(args.depth_pmax)],
            "depth_tiles_share_range": ["GT depth", "DAV2-S depth", "Ours"],
            "error_cmap": "magma",
            "error_vmin": 0.0,
            "error_vmax_abs_rel": float(args.error_max_abs_rel),
            "error_tiles_share_range": ["DAV2 error", "Ours error"],
            "residual_cmap": "coolwarm",
            "residual_range_scope": "global_selected_samples_symmetric_p99_abs",
            "residual_vmin": -float(residual_vlim),
            "residual_vmax": float(residual_vlim),
            "gate_cmap": "viridis",
            "gate_vmin": 0.0,
            "gate_vmax": 1.0,
            "improvement_colormap": "black_zero_green_positive_red_negative",
            "improvement_vmin_abs_rel": -float(args.error_max_abs_rel),
            "improvement_vmax_abs_rel": float(args.error_max_abs_rel),
        },
        "panel_layout": [
            "RGB input",
            "GT depth",
            "DAV2-S depth",
            "Ours",
            "DAV2 error",
            "Ours error",
            "Residual gate*delta",
            "Gate",
            "Err improve +green",
        ],
        "records": manifest_records,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
