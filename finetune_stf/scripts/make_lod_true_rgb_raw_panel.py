#!/usr/bin/env python3
"""Make true-LOD RGB_Dark vs RAW_Dark panels for the two 0608 experiments."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.dataset.lod_raw import _apply_crop  # noqa: E402
from finetune_stf.dataset.lod_true import (  # noqa: E402
    LODTrueRGBDark,
    LODTrueRawDarkRGB16,
    _load_rgb_u8,
)
from finetune_stf.scripts.make_rod_raw_four_exp_panel import (  # noqa: E402
    blank_tile,
    build_prediction_result,
    colorize_array,
    colorize_d1,
    colorize_error,
    infer_prediction,
    load_checkpoint_into_model,
    load_exp_args,
    load_font,
    make_contact_sheet,
    metric_line,
    resize_tile,
    robust_limits,
    safe_filename_part,
    use_amp_for,
    x3_before_backbone,
    x3_hist_image,
    x3_preview_image,
    x3_stats,
)
from finetune_stf.train import build_model  # noqa: E402
from finetune_stf.util.model_input import select_model_input  # noqa: E402


DEFAULT_RGB_EXP = PROJECT_ROOT / "finetune_stf/exp/0608_0109_lod_true_rgb_dark_dav2s_decoder_e10"
DEFAULT_RAW_EXP = PROJECT_ROOT / "finetune_stf/exp/0608_0113_lod_true_raw_rgb16_ram3_dav2s_decoder_e10"
LOD_TRUE_RAW_RGB16_INPUT_MODE_BY_FAMILY = {
    "lod_true_raw_dark_rgb16": "raw_rgb16_dark",
    "lod_true_raw_normal_rgb16": "raw_rgb16_normal",
}
LOD_TRUE_RAW_RGB16_DATASET_FAMILIES = set(LOD_TRUE_RAW_RGB16_INPUT_MODE_BY_FAMILY)
CSV_FIELDS = (
    "sample_index",
    "sample_name",
    "line",
    "experiment",
    "checkpoint",
    "abs_rel",
    "rmse",
    "silog",
    "d1",
    "d2",
    "d3",
    "align_scale",
    "align_shift",
    "valid_pixels",
    "d1_pass_pixels",
    "backbone_min",
    "backbone_p01",
    "backbone_mean",
    "backbone_p50",
    "backbone_p99",
    "backbone_max",
    "panel_path",
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-exp-dir", type=Path, default=DEFAULT_RGB_EXP)
    parser.add_argument("--raw-exp-dir", type=Path, default=DEFAULT_RAW_EXP)
    parser.add_argument("--checkpoint-name", default="last_epoch_model.pth")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--sample-selection", default="random", choices=["random", "sequential"])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--sample-indices", default="")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--tile-height", type=int, default=232)
    parser.add_argument("--pred-cmap", default="Spectral_r")
    parser.add_argument("--error-cmap", default="magma")
    parser.add_argument("--error-vmax-pct", type=float, default=99.0)
    parser.add_argument("--d1-cmap", default="RdYlGn")
    parser.add_argument("--d1-threshold", type=float, default=1.25)
    parser.add_argument("--backbone-vis-low-pct", type=float, default=1.0)
    parser.add_argument("--backbone-vis-high-pct", type=float, default=99.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_output_root() -> Path:
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    return PROJECT_ROOT / "finetune_stf/analysis/lod_true_rgb_raw_panel" / stamp


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def amp_dtype_from_args(exp_args: argparse.Namespace) -> torch.dtype:
    return torch.float16 if str(getattr(exp_args, "amp_dtype", "bf16")) == "fp16" else torch.bfloat16


def parse_indices(value: str) -> list[int]:
    indices = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            indices.append(int(item))
    return indices


def plan_indices(args: argparse.Namespace, dataset_size: int) -> list[int]:
    explicit = parse_indices(args.sample_indices)
    if explicit:
        indices = explicit
    else:
        count = int(args.num_samples)
        if count < 1:
            raise ValueError("--num-samples must be >= 1")
        if count > int(dataset_size):
            raise ValueError(f"--num-samples={count} exceeds dataset size={dataset_size}")
        if args.sample_selection == "random":
            rng = np.random.default_rng(int(args.sample_seed))
            indices = [int(idx) for idx in rng.choice(int(dataset_size), size=count, replace=False).tolist()]
        else:
            step = int(args.sample_step)
            if step < 1:
                raise ValueError("--sample-step must be >= 1")
            indices = [int(args.sample_index) + idx * step for idx in range(count)]
    bad = [idx for idx in indices if idx < 0 or idx >= int(dataset_size)]
    if bad:
        raise ValueError(f"Sample indices out of range [0, {dataset_size}): {bad}")
    return indices


def resolve_checkpoint(exp_args: argparse.Namespace, checkpoint_name: str) -> Path:
    checkpoint = Path(exp_args.heavy_save_path).expanduser().resolve() / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    return checkpoint


def build_rgb_dataset(exp_args: argparse.Namespace) -> LODTrueRGBDark:
    return LODTrueRGBDark(
        lod_root=exp_args.lod_root,
        manifest_path=exp_args.lod_manifest,
        split="01Valid",
        size=(int(exp_args.input_height), int(exp_args.input_width)),
        mode="val",
        label_space=exp_args.lod_label_space,
        crop_mode=exp_args.lod_val_crop_mode,
    )


def build_raw_dataset(exp_args: argparse.Namespace) -> LODTrueRawDarkRGB16:
    dataset_family = str(exp_args.resolved_config.dataset_family)
    return LODTrueRawDarkRGB16(
        lod_root=exp_args.lod_root,
        manifest_path=exp_args.lod_manifest,
        split="01Valid",
        size=(int(exp_args.input_height), int(exp_args.input_width)),
        mode="val",
        label_space=exp_args.lod_label_space,
        crop_mode=exp_args.lod_val_crop_mode,
        raw_storage_format=exp_args.raw_storage_format,
        lod_raw_norm_mode=exp_args.lod_raw_norm_mode,
        raw_input_mode=LOD_TRUE_RAW_RGB16_INPUT_MODE_BY_FAMILY[dataset_family],
    )


def sample_seed(exp_args: argparse.Namespace, sample_index: int) -> int:
    return int(getattr(exp_args, "seed", 0)) * 1000003 + int(sample_index) * 9176 + 808


def crop_rgb_normal(sample: dict[str, object]) -> np.ndarray:
    image = _load_rgb_u8(Path(str(sample["rgb_normal_path"])))
    geometry = sample.get("geometry_params", {})
    crop_box = geometry.get("crop_box") if isinstance(geometry, dict) else None
    if crop_box is None:
        raise ValueError("LOD sample was built without crop geometry")
    return _apply_crop(image, tuple(int(v) for v in crop_box))


def crop_rgb_dark(sample: dict[str, object]) -> np.ndarray:
    image = _load_rgb_u8(Path(str(sample["rgb_dark_path"])))
    geometry = sample.get("geometry_params", {})
    crop_box = geometry.get("crop_box") if isinstance(geometry, dict) else None
    if crop_box is None:
        raise ValueError("LOD sample was built without crop geometry")
    return _apply_crop(image, tuple(int(v) for v in crop_box))


def rgb_image(rgb_hwc: np.ndarray) -> Image.Image:
    arr = np.clip(np.asarray(rgb_hwc, dtype=np.float32), 0.0, 1.0)
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def stat_payload(values_chw: np.ndarray) -> dict[str, float]:
    return x3_stats(np.asarray(values_chw, dtype=np.float32))


def stats_to_csv_prefix(stats: dict[str, float]) -> dict[str, float]:
    return {
        "backbone_min": stats.get("min", float("nan")),
        "backbone_p01": stats.get("p01", float("nan")),
        "backbone_mean": stats.get("mean", float("nan")),
        "backbone_p50": stats.get("p50", float("nan")),
        "backbone_p99": stats.get("p99", float("nan")),
        "backbone_max": stats.get("max", float("nan")),
    }


def make_panel(
    rows: list[dict[str, object]],
    *,
    tile_size: tuple[int, int],
    sample_name: str,
    pred_cmap: str,
    error_cmap: str,
    d1_cmap: str,
    error_vmax: float,
    d1_threshold: float,
) -> Image.Image:
    tile_w, tile_h = tile_size
    label_w = 290
    header_h = 64
    footer_h = 32
    cols = ("Backbone input", "Pseudo / dist", "Pred", "Error", "D1 map")
    width = label_w + tile_w * len(cols)
    height = header_h + tile_h * len(rows) + footer_h
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(18)
    header_font = load_font(15)
    label_font = load_font(13)
    small_font = load_font(11)
    draw.text((12, 12), f"LOD true RGB/RAW panel: {sample_name}", font=title_font, fill=(20, 20, 20))
    draw.text((12, 38), f"cmaps: {pred_cmap} / {error_cmap} / {d1_cmap}", font=small_font, fill=(70, 70, 70))
    for col_idx, col_name in enumerate(cols):
        draw.text((label_w + col_idx * tile_w + 10, 38), col_name, font=header_font, fill=(20, 20, 20))

    for row_idx, row in enumerate(rows):
        y = header_h + row_idx * tile_h
        draw.rectangle((0, y, label_w - 1, y + tile_h - 1), fill=(235, 235, 235) if row_idx % 2 == 0 else (228, 228, 228))
        text_y = y + 12
        for line in row["label_lines"]:
            draw.text((12, text_y), str(line), font=label_font, fill=(20, 20, 20))
            text_y += 17
        for col_idx, image in enumerate(row["images"]):
            x = label_w + col_idx * tile_w
            canvas.paste(resize_tile(image, tile_size), (x, y))
            draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=(190, 190, 190), width=1)

    draw.text(
        (12, header_h + tile_h * len(rows) + 8),
        f"Pseudo is from RGB_Normal DAv2-L. Error clipped at {error_vmax:.3g}; D1 pass threshold < {d1_threshold:g}.",
        font=small_font,
        fill=(60, 60, 60),
    )
    return canvas


def csv_row(
    *,
    sample_index: int,
    sample_name: str,
    line: str,
    experiment: str,
    checkpoint: Path,
    result,
    backbone_stats: dict[str, float],
    panel_path: Path,
) -> dict[str, object]:
    row = {
        "sample_index": int(sample_index),
        "sample_name": sample_name,
        "line": line,
        "experiment": experiment,
        "checkpoint": str(checkpoint),
        "valid_pixels": int(np.count_nonzero(result.d1_valid)),
        "d1_pass_pixels": int(result.d1_pass_pixels),
        "align_scale": float(result.align_stats.get("scale", float("nan"))),
        "align_shift": float(result.align_stats.get("shift", float("nan"))),
        "panel_path": str(panel_path),
    }
    for key in ("abs_rel", "rmse", "silog", "d1", "d2", "d3"):
        row[key] = float(result.metrics.get(key, float("nan")))
    row.update(stats_to_csv_prefix(backbone_stats))
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def process_sample(
    *,
    sample_index: int,
    rgb_args: argparse.Namespace,
    raw_args: argparse.Namespace,
    rgb_dataset: LODTrueRGBDark,
    raw_dataset: LODTrueRawDarkRGB16,
    rgb_model: torch.nn.Module,
    raw_model: torch.nn.Module,
    rgb_checkpoint: Path,
    raw_checkpoint: Path,
    device: torch.device,
    args: argparse.Namespace,
    panels_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]], Path]:
    rng = random.Random(sample_seed(rgb_args, sample_index))
    rgb_sample = rgb_dataset.build_sample(sample_index, rng=rng, include_geometry=True)
    raw_sample = raw_dataset.build_sample(sample_index, rng=random.Random(sample_seed(rgb_args, sample_index)), include_geometry=True)
    sample_name = str(rgb_sample["sample_name"])
    if sample_name != str(raw_sample["sample_name"]):
        raise ValueError(f"RGB/RAW sample mismatch at idx={sample_index}: {sample_name} vs {raw_sample['sample_name']}")

    target = rgb_sample["depth"].detach().float().cpu().numpy().astype(np.float32)
    valid_mask = rgb_sample["valid_mask"].detach().cpu().numpy().astype(bool)
    valid_mask = valid_mask & np.isfinite(target) & (target > 0)
    target_hw = tuple(int(v) for v in target.shape)
    rgb_normal = crop_rgb_normal(rgb_sample)
    rgb_dark = crop_rgb_dark(rgb_sample)
    rgb_dark_chw = np.transpose(rgb_dark, (2, 0, 1)).astype(np.float32, copy=False)

    rgb_input = select_model_input(
        rgb_sample,
        rgb_args.resolved_config.model_input_tensor,
        dataset_family=rgb_args.resolved_config.dataset_family,
        sample_source="lod_val",
        add_batch_dim=True,
    )
    raw_input = select_model_input(
        raw_sample,
        raw_args.resolved_config.model_input_tensor,
        dataset_family=raw_args.resolved_config.dataset_family,
        sample_source="lod_val",
        add_batch_dim=True,
    )

    rgb_pred = infer_prediction(
        rgb_model,
        rgb_input,
        target_hw,
        device=device,
        amp_dtype=amp_dtype_from_args(rgb_args),
        use_amp=use_amp_for(device, rgb_args, args.no_amp),
    )
    raw_pred = infer_prediction(
        raw_model,
        raw_input,
        target_hw,
        device=device,
        amp_dtype=amp_dtype_from_args(raw_args),
        use_amp=use_amp_for(device, raw_args, args.no_amp),
    )
    rgb_result = build_prediction_result(rgb_pred, target, valid_mask, d1_threshold=args.d1_threshold)
    raw_result = build_prediction_result(raw_pred, target, valid_mask, d1_threshold=args.d1_threshold)

    raw_backbone = x3_before_backbone(
        raw_model,
        raw_input,
        device=device,
        amp_dtype=amp_dtype_from_args(raw_args),
        use_amp=use_amp_for(device, raw_args, args.no_amp),
    )
    rgb_stats = stat_payload(rgb_dark_chw)
    raw_stats = stat_payload(raw_backbone)

    pred_vmin, pred_vmax = robust_limits([target, rgb_result.aligned_pred, raw_result.aligned_pred], valid_mask)
    _, error_vmax = robust_limits(
        [rgb_result.rel_error, raw_result.rel_error],
        valid_mask,
        low_pct=0.0,
        high_pct=float(args.error_vmax_pct),
    )

    tile_size = (int(args.tile_width), int(args.tile_height))
    rows = [
        {
            "label_lines": [
                "Reference",
                "RGB_Normal",
                "pseudo label = RGB_Normal DAv2-L",
            ],
            "images": [
                rgb_image(rgb_normal),
                colorize_array(target, valid_mask, vmin=pred_vmin, vmax=pred_vmax, cmap_name=args.pred_cmap),
                blank_tile(tile_size),
                blank_tile(tile_size),
                blank_tile(tile_size),
            ],
        },
        {
            "label_lines": [
                "RGB line",
                "input = RGB_Dark",
                f"ckpt = {Path(rgb_args.save_path).name.split('_lod_true_', 1)[0]} final",
                metric_line(rgb_result.metrics),
            ],
            "images": [
                rgb_image(rgb_dark),
                x3_hist_image(rgb_dark_chw, size=tile_size, title="RGB_Dark input dist", labels=("R", "G", "B")),
                colorize_array(rgb_result.aligned_pred, valid_mask, vmin=pred_vmin, vmax=pred_vmax, cmap_name=args.pred_cmap),
                colorize_error(rgb_result.rel_error, rgb_result.error_valid, vmax=error_vmax, cmap_name=args.error_cmap),
                colorize_d1(rgb_result.d1_score, rgb_result.d1_valid, cmap_name=args.d1_cmap),
            ],
        },
        {
            "label_lines": [
                "RAW line",
                "input = RAW_Dark RGB16",
                "shown = RamCore3 x3 before backbone",
                f"ckpt = {Path(raw_args.save_path).name.split('_lod_true_', 1)[0]} final",
                metric_line(raw_result.metrics),
                f"x3 p1/p99 {raw_stats['p01']:.2g}/{raw_stats['p99']:.2g}",
            ],
            "images": [
                x3_preview_image(raw_backbone, low_pct=args.backbone_vis_low_pct, high_pct=args.backbone_vis_high_pct),
                x3_hist_image(raw_backbone, size=tile_size, title="RAW_Dark RamCore3 x3 dist"),
                colorize_array(raw_result.aligned_pred, valid_mask, vmin=pred_vmin, vmax=pred_vmax, cmap_name=args.pred_cmap),
                colorize_error(raw_result.rel_error, raw_result.error_valid, vmax=error_vmax, cmap_name=args.error_cmap),
                colorize_d1(raw_result.d1_score, raw_result.d1_valid, cmap_name=args.d1_cmap),
            ],
        },
    ]

    panel = make_panel(
        rows,
        tile_size=tile_size,
        sample_name=sample_name,
        pred_cmap=args.pred_cmap,
        error_cmap=args.error_cmap,
        d1_cmap=args.d1_cmap,
        error_vmax=error_vmax,
        d1_threshold=args.d1_threshold,
    )
    panel_path = panels_dir / f"lod_true_rgb_raw_panel_idx{sample_index:04d}_{safe_filename_part(sample_name)}.png"
    panel.save(panel_path)

    rows_csv = [
        csv_row(
            sample_index=sample_index,
            sample_name=sample_name,
            line="rgb_dark",
            experiment=Path(rgb_args.save_path).name,
            checkpoint=rgb_checkpoint,
            result=rgb_result,
            backbone_stats=rgb_stats,
            panel_path=panel_path,
        ),
        csv_row(
            sample_index=sample_index,
            sample_name=sample_name,
            line="raw_dark_rgb16",
            experiment=Path(raw_args.save_path).name,
            checkpoint=raw_checkpoint,
            result=raw_result,
            backbone_stats=raw_stats,
            panel_path=panel_path,
        ),
    ]
    summary = {
        "sample_index": int(sample_index),
        "sample_name": sample_name,
        "panel_path": str(panel_path),
        "rgb_normal_path": str(rgb_sample["rgb_normal_path"]),
        "rgb_dark_path": str(rgb_sample["rgb_dark_path"]),
        "raw_dark_path": str(raw_sample["raw_dark_path"]),
        "pseudo_depth_path": str(rgb_sample["pseudo_depth_path"]),
        "semantics": {
            "rgb_normal": "teacher source for pseudo label",
            "rgb_dark": "RGB line student input",
            "raw_dark_rgb16": "RAW line student input; visualized after RamCore3 as backbone x3",
        },
        "prediction_colormap": str(args.pred_cmap),
        "error_colormap": str(args.error_cmap),
        "d1_colormap": str(args.d1_cmap),
        "error_definition": "abs(aligned inverse-relative prediction - pseudo inverse-relative label) / pseudo inverse-relative label",
        "d1_definition": f"per-pixel delta1 pass after affine alignment, threshold < {float(args.d1_threshold):g}",
        "pred_vmin": float(pred_vmin),
        "pred_vmax": float(pred_vmax),
        "error_vmax": float(error_vmax),
        "rgb_line": {
            "experiment": Path(rgb_args.save_path).name,
            "checkpoint": str(rgb_checkpoint),
            "metrics": rgb_result.metrics,
            "align_stats": rgb_result.align_stats,
            "rgb_dark_input_stats": rgb_stats,
        },
        "raw_line": {
            "experiment": Path(raw_args.save_path).name,
            "checkpoint": str(raw_checkpoint),
            "metrics": raw_result.metrics,
            "align_stats": raw_result.align_stats,
            "backbone_input_stats": raw_stats,
        },
    }
    return summary, rows_csv, panel_path


def main() -> None:
    args = parse_args()
    rgb_exp_dir = args.rgb_exp_dir.expanduser().resolve()
    raw_exp_dir = args.raw_exp_dir.expanduser().resolve()
    rgb_args = load_exp_args(rgb_exp_dir)
    raw_args = load_exp_args(raw_exp_dir)
    if rgb_args.resolved_config.dataset_family != "lod_true_rgb_dark":
        raise ValueError(f"Expected RGB LOD experiment, got {rgb_args.resolved_config.dataset_family!r}")
    if raw_args.resolved_config.dataset_family not in LOD_TRUE_RAW_RGB16_DATASET_FAMILIES:
        raise ValueError(f"Expected RAW LOD experiment, got {raw_args.resolved_config.dataset_family!r}")
    if rgb_args.lod_manifest != raw_args.lod_manifest:
        raise ValueError("RGB and RAW LOD experiments must use the same lod_manifest")

    rgb_dataset = build_rgb_dataset(rgb_args)
    raw_dataset = build_raw_dataset(raw_args)
    if len(rgb_dataset) != len(raw_dataset):
        raise ValueError(f"RGB/RAW LOD dataset length mismatch: {len(rgb_dataset)} vs {len(raw_dataset)}")
    sample_indices = plan_indices(args, len(rgb_dataset))
    output_root = args.output_root.expanduser().resolve() if args.output_root else default_output_root()
    panels_dir = output_root / "sample_panels"
    metadata_dir = output_root / "metadata"

    planned = {
        "sample_indices": sample_indices,
        "sample_selection": str(args.sample_selection) if not parse_indices(args.sample_indices) else "explicit_indices",
        "sample_seed": int(args.sample_seed),
        "output_root": str(output_root),
        "panels_dir": str(panels_dir),
        "metadata_dir": str(metadata_dir),
        "rgb_experiment": str(rgb_exp_dir),
        "raw_experiment": str(raw_exp_dir),
    }
    if args.dry_run:
        print(json.dumps(planned, indent=2, sort_keys=True))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else int(device.index))

    rgb_checkpoint = resolve_checkpoint(rgb_args, args.checkpoint_name)
    raw_checkpoint = resolve_checkpoint(raw_args, args.checkpoint_name)
    rgb_model = build_model(rgb_args)
    raw_model = build_model(raw_args)
    rgb_ckpt_meta = load_checkpoint_into_model(rgb_model, rgb_checkpoint)
    raw_ckpt_meta = load_checkpoint_into_model(raw_model, raw_checkpoint)
    rgb_model.to(device).eval()
    raw_model.to(device).eval()

    sample_summaries = []
    csv_rows = []
    panel_paths = []
    with torch.no_grad():
        for sample_index in sample_indices:
            summary, rows, panel_path = process_sample(
                sample_index=sample_index,
                rgb_args=rgb_args,
                raw_args=raw_args,
                rgb_dataset=rgb_dataset,
                raw_dataset=raw_dataset,
                rgb_model=rgb_model,
                raw_model=raw_model,
                rgb_checkpoint=rgb_checkpoint,
                raw_checkpoint=raw_checkpoint,
                device=device,
                args=args,
                panels_dir=panels_dir,
            )
            sample_meta_dir = metadata_dir / f"sample_{int(sample_index):04d}"
            sample_meta_dir.mkdir(parents=True, exist_ok=True)
            (sample_meta_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            write_csv(sample_meta_dir / "metrics_by_row.csv", rows)
            sample_summaries.append(summary)
            csv_rows.extend(rows)
            panel_paths.append(panel_path)

    del rgb_model, raw_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    combined_csv_path = output_root / "metrics_by_sample_row.csv"
    write_csv(combined_csv_path, csv_rows)
    contact_sheet_path = output_root / f"lod_true_rgb_raw_panel_{len(sample_indices)}samples_contact_sheet.png"
    make_contact_sheet(panel_paths, contact_sheet_path)
    summary_path = output_root / f"summary_{len(sample_indices)}samples.json"
    payload = {
        **planned,
        "checkpoint_name": str(args.checkpoint_name),
        "rgb_checkpoint": str(rgb_checkpoint),
        "raw_checkpoint": str(raw_checkpoint),
        "rgb_checkpoint_meta": rgb_ckpt_meta,
        "raw_checkpoint_meta": raw_ckpt_meta,
        "num_samples": len(sample_indices),
        "combined_csv_path": str(combined_csv_path),
        "contact_sheet_path": str(contact_sheet_path),
        "semantics": {
            "rgb_normal": "teacher source for pseudo label",
            "rgb_dark": "RGB line student input",
            "raw_dark_rgb16": "RAW line student input; visualized after RamCore3 as backbone x3",
        },
        "sample_summaries": sample_summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] panels_dir={panels_dir}")
    print(f"[OK] contact_sheet={contact_sheet_path}")
    print(f"[OK] combined_csv={combined_csv_path}")
    print(f"[OK] summary={summary_path}")


if __name__ == "__main__":
    main()
