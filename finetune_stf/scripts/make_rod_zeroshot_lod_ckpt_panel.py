#!/usr/bin/env python3
"""Visualize ROD zero-shot predictions from the three LOD checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.dataset.lod_raw import _apply_crop
from finetune_stf.dataset.rod_raw import RODRawRGB3
from finetune_stf.dataset.rod_raw_rgb import render_pipeline, unpack_raw24
from finetune_stf.dataset.rod_raw_student_rgb import (
    DEFAULT_ROD_NIGHT_TEACHER_MANIFEST,
    DEFAULT_ROD_ROOT,
    RODRawStudentRGB,
)
from finetune_stf.train import build_model, parse_args as parse_train_args
from finetune_stf.train import resolve_model_state, strip_module_prefix
from finetune_stf.util.metric import affine_align_to_inverse_target, compute_inverse_relative_metrics
from finetune_stf.util.model_input import select_model_input


PRETRAINED = "/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth"
CKPT_0608_2246 = (
    "/mnt/drive/3333_raw/0000_exp_ckpt/"
    "0608_2246_lod_true_rgb_dark_block8excl10_dav2s_lora_tap_r8a16_decoder_e40_flip05_poly/"
    "best_model.pth"
)
CKPT_0608_2026 = (
    "/mnt/drive/3333_raw/0000_exp_ckpt/"
    "0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly/"
    "best_model.pth"
)
CKPT_0609_0044 = (
    "/mnt/drive/3333_raw/0000_exp_ckpt/"
    "0609_0044_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly/"
    "best_model.pth"
)


@dataclass(frozen=True)
class ExpSpec:
    key: str
    label: str
    checkpoint: str
    train_argv: tuple[str, ...]
    dataset_kind: str


@dataclass
class PredResult:
    pred: np.ndarray
    aligned: np.ndarray
    rel_error: np.ndarray
    d1_map: np.ndarray
    valid: np.ndarray
    metrics: dict[str, float]
    align: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--rod-root", default=DEFAULT_ROD_ROOT)
    parser.add_argument("--rod-manifest", default=DEFAULT_ROD_NIGHT_TEACHER_MANIFEST)
    parser.add_argument("--sample-indices", nargs="*", type=int, default=[0, 500, 1000, 1500])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile-width", type=int, default=220)
    parser.add_argument("--tile-height", type=int, default=142)
    parser.add_argument("--pred-cmap", default="Spectral_r")
    parser.add_argument("--error-cmap", default="magma")
    parser.add_argument("--d1-cmap", default="RdYlGn")
    parser.add_argument("--error-vmax-pct", type=float, default=95.0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def default_output_root() -> Path:
    stamp = datetime.now().strftime("%m%d_%H%M")
    return PROJECT_ROOT / "finetune_stf/analysis/rod_zeroshot_lod_ckpt_viz" / f"{stamp}_3exp_n4"


def train_args_from(argv: tuple[str, ...]):
    old_argv = sys.argv
    try:
        sys.argv = ["train.py", *argv]
        return parse_train_args()
    finally:
        sys.argv = old_argv


def common_eval_args() -> list[str]:
    return [
        "--encoder", "vits",
        "--stage", "eval_only",
        "--eval-only",
        "--input-height", "512",
        "--input-width", "960",
        "--rod-root", DEFAULT_ROD_ROOT,
        "--rod-night-manifest", DEFAULT_ROD_NIGHT_TEACHER_MANIFEST,
        "--rod-raw-source", "raw24",
        "--rod-label-space", "inverse_relative",
        "--rod-train-crop-mode", "random",
        "--rod-val-crop-mode", "center",
        "--no-eval-stf",
        "--eval-rod",
        "--bs", "1",
        "--accum-steps", "1",
        "--lr", "1e-5",
        "--loss-type", "ssi",
        "--loss-target-normalization",
        "--loss-norm-min-scale", "1e-3",
        "--epochs", "0",
        "--amp",
        "--amp-dtype", "bf16",
        "--seed", "42",
        "--num-workers", "0",
        "--no-enable-fixed-viz-dump",
        "--no-enable-train-source-viz-dump",
        "--pretrained-from", PRETRAINED,
        "--save-path", "/tmp/codex_tmp_rod_zeroshot_viz_args",
    ]


def exp_specs() -> list[ExpSpec]:
    base = common_eval_args()
    rgb_args = [
        *base,
        "--input-domain", "rgb",
        "--front-end", "dav2_rgb",
        "--model-input-tensor", "image",
        "--dataset-family", "rod_raw_student_rgb",
        "--dataset-input-mode", "raw24_student_rgb",
        "--raw-storage-format", "n_a",
        "--bridge", "none",
        "--decoder-feature-adapter", "none",
        "--lora", "dav2_lora",
        "--lora-block-mode", "tap",
        "--lora-tap-layers", "2", "5", "8", "11",
        "--lora-rank", "8",
        "--lora-alpha", "16",
        "--lora-lr", "5e-5",
        "--dav2-train-mode", "decoder",
        "--resume-from", CKPT_0608_2246,
    ]
    raw_base = [
        *base,
        "--input-domain", "raw3",
        "--front-end", "raw_rgb16_ram3",
        "--model-input-tensor", "raw",
        "--dataset-family", "rod_raw_rgb3",
        "--dataset-input-mode", "raw24_base_rgb3",
        "--raw-storage-format", "n_a",
        "--bridge", "none",
        "--decoder-feature-adapter", "none",
        "--lora", "dav2_lora",
        "--lora-block-mode", "tap",
        "--lora-tap-layers", "2", "5", "8", "11",
        "--lora-rank", "8",
        "--lora-alpha", "16",
        "--lora-lr", "5e-5",
        "--raw-front-end-lr", "5e-5",
        "--raw-ram-rgb-tail", "identity",
        "--dav2-train-mode", "decoder",
    ]
    return [
        ExpSpec("rgb_0608_2246", "0608_2246 LOD RGB -> ROD student RGB", CKPT_0608_2246, tuple(rgb_args), "student_rgb"),
        ExpSpec("rawdark_0608_2026", "0608_2026 LOD RAW_Dark -> ROD raw3", CKPT_0608_2026, tuple([*raw_base, "--resume-from", CKPT_0608_2026]), "raw3"),
        ExpSpec("rawnormal_0609_0044", "0609_0044 LOD RAW_normal -> ROD raw3", CKPT_0609_0044, tuple([*raw_base, "--resume-from", CKPT_0609_0044]), "raw3"),
    ]


def load_model(spec: ExpSpec, device: torch.device) -> tuple[torch.nn.Module, argparse.Namespace]:
    args = train_args_from(spec.train_argv)
    model = build_model(args)
    ckpt = torch.load(spec.checkpoint, map_location="cpu")
    model.load_state_dict(strip_module_prefix(resolve_model_state(ckpt)), strict=True)
    model.to(device).eval()
    return model, args


def tensor_chw(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.float32, copy=False)


def run_model(model, args, sample, device: torch.device, use_amp: bool) -> np.ndarray:
    model_input = select_model_input(
        sample,
        args.resolved_config.model_input_tensor,
        dataset_family=args.resolved_config.dataset_family,
        sample_source="rod_night_val",
        add_batch_dim=True,
    ).to(device).float()
    amp_dtype = torch.float16 if getattr(args, "amp_dtype", "bf16") == "fp16" else torch.bfloat16
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
        pred = model(model_input).float()
    if pred.ndim != 3:
        raise ValueError(f"Expected prediction BCHW-like (B,H,W), got {tuple(pred.shape)}")
    return pred[0].detach().cpu().numpy().astype(np.float32, copy=False)


def build_result(pred: np.ndarray, target: np.ndarray, valid_mask: np.ndarray) -> PredResult:
    if pred.shape != target.shape:
        pred_t = torch.from_numpy(pred)[None, None].float()
        pred = F.interpolate(pred_t, target.shape, mode="bilinear", align_corners=True)[0, 0].numpy()
    aligned, align = affine_align_to_inverse_target(pred, target, valid_mask)
    metrics = compute_inverse_relative_metrics(aligned, target, valid_mask) or {}
    valid = valid_mask & np.isfinite(aligned) & np.isfinite(target) & (aligned > 0) & (target > 0)
    rel_error = np.full(target.shape, np.nan, dtype=np.float32)
    d1_map = np.full(target.shape, np.nan, dtype=np.float32)
    if np.any(valid):
        rel_error[valid] = np.abs(aligned[valid] - target[valid]) / np.clip(target[valid], 1e-6, None)
        thresh = np.maximum(target[valid] / aligned[valid], aligned[valid] / target[valid])
        d1_map[valid] = (thresh < 1.25).astype(np.float32)
    return PredResult(
        pred=pred,
        aligned=aligned.astype(np.float32, copy=False),
        rel_error=rel_error,
        d1_map=d1_map,
        valid=valid,
        metrics={key: float(value) for key, value in metrics.items()},
        align={key: float(value) for key, value in align.items()},
    )


def robust_limits(arrays: list[np.ndarray], valid: np.ndarray, low=1.0, high=99.0) -> tuple[float, float]:
    values = []
    for array in arrays:
        mask = valid & np.isfinite(array)
        if np.any(mask):
            values.append(np.asarray(array, dtype=np.float32)[mask].reshape(-1))
    if not values:
        return 0.0, 1.0
    flat = np.concatenate(values)
    lo, hi = np.percentile(flat, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(flat)), float(np.nanmax(flat))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    return float(lo), float(hi)


def colorize(array: np.ndarray, valid: np.ndarray, *, vmin: float, vmax: float, cmap_name: str) -> Image.Image:
    arr = np.asarray(array, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(arr)
    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    cmap = colormaps.get_cmap(cmap_name)
    rgb = (cmap(norm)[:, :, :3] * 255.0).round().astype(np.uint8)
    rgb[~mask] = 0
    return Image.fromarray(rgb)


def colorize_error(error: np.ndarray, valid: np.ndarray, *, vmax: float, cmap_name: str) -> Image.Image:
    arr = np.asarray(error, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(arr)
    norm = np.clip(np.where(mask, arr, 0.0) / max(vmax, 1e-6), 0.0, 1.0)
    cmap = colormaps.get_cmap(cmap_name)
    rgb = (cmap(norm)[:, :, :3] * 255.0).round().astype(np.uint8)
    rgb[~mask] = 0
    return Image.fromarray(rgb)


def colorize_d1(d1_map: np.ndarray, valid: np.ndarray, cmap_name: str) -> Image.Image:
    arr = np.asarray(d1_map, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(arr)
    cmap = colormaps.get_cmap(cmap_name)
    rgb = (cmap(np.clip(np.where(mask, arr, 0.0), 0.0, 1.0))[:, :, :3] * 255.0).round().astype(np.uint8)
    rgb[~mask] = 0
    return Image.fromarray(rgb)


def rgb_image_from_float(rgb: np.ndarray) -> Image.Image:
    arr = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def raw3_image(raw_chw: np.ndarray) -> Image.Image:
    arr = np.transpose(raw_chw, (1, 2, 0)).astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    hi = np.percentile(arr[finite], 99.5) if np.any(finite) else 1.0
    hi = float(hi) if np.isfinite(hi) and hi > 1e-6 else 1.0
    return rgb_image_from_float(np.clip(arr / hi, 0.0, 1.0))


def crop_student_rgb(sample: dict[str, object]) -> np.ndarray:
    raw = unpack_raw24(str(sample["raw_path"]))
    rgb = render_pipeline(raw, "student_dark_degreen_v1").astype(np.float32) / 255.0
    geometry = sample.get("geometry_params") or {}
    crop_box = geometry.get("crop_box")
    if crop_box is None:
        raise ValueError("sample is missing geometry crop_box")
    return _apply_crop(rgb, tuple(int(v) for v in crop_box))


def font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def resize_tile(image: Image.Image, tile_size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(tile_size, Image.Resampling.BILINEAR)


def metric_text(metrics: dict[str, float]) -> str:
    return "D1 {d1:.3f}  abs_rel {abs_rel:.2f}".format(
        d1=float(metrics.get("d1", float("nan"))),
        abs_rel=float(metrics.get("abs_rel", float("nan"))),
    )


def make_sample_panel(
    *,
    sample_index: int,
    sample_name: str,
    student_rgb: Image.Image,
    raw3: Image.Image,
    pseudo: Image.Image,
    pred_rows: list[tuple[ExpSpec, PredResult, Image.Image, Image.Image, Image.Image]],
    tile_size: tuple[int, int],
    error_vmax: float,
) -> Image.Image:
    label_w = 310
    header_h = 76
    footer_h = 32
    cols = ["Input", "Pseudo GT", "Aligned pred", "Rel error", "D1 pass"]
    row_h = tile_size[1]
    width = label_w + tile_size[0] * len(cols)
    height = header_h + row_h * (1 + len(pred_rows)) + footer_h
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    title_font = font(18)
    label_font = font(13)
    small_font = font(11)
    header_font = font(14)
    draw.text((12, 12), f"ROD zero-shot LOD checkpoints: idx={sample_index} {sample_name}", font=title_font, fill=(20, 20, 20))
    draw.text((12, 40), "Predictions are affine-aligned to ROD pseudo inverse-relative label.", font=small_font, fill=(70, 70, 70))
    for c, name in enumerate(cols):
        draw.text((label_w + c * tile_size[0] + 8, 52), name, font=header_font, fill=(20, 20, 20))

    y = header_h
    draw.rectangle((0, y, label_w - 1, y + row_h - 1), fill=(232, 232, 232))
    draw.text((12, y + 16), "Input views", font=label_font, fill=(20, 20, 20))
    draw.text((12, y + 38), "student RGB / raw3 p99.5 stretch", font=small_font, fill=(60, 60, 60))
    inputs = [student_rgb, pseudo, raw3, Image.new("RGB", tile_size, (245, 245, 245)), Image.new("RGB", tile_size, (245, 245, 245))]
    for c, image in enumerate(inputs):
        x = label_w + c * tile_size[0]
        canvas.paste(resize_tile(image, tile_size), (x, y))
        draw.rectangle((x, y, x + tile_size[0] - 1, y + row_h - 1), outline=(190, 190, 190))

    for row_idx, (spec, result, pred_img, err_img, d1_img) in enumerate(pred_rows, start=1):
        y = header_h + row_idx * row_h
        bg = (238, 238, 238) if row_idx % 2 else (230, 230, 230)
        draw.rectangle((0, y, label_w - 1, y + row_h - 1), fill=bg)
        label_lines = [spec.key, spec.label, metric_text(result.metrics)]
        for i, line in enumerate(label_lines):
            draw.text((12, y + 12 + i * 20), line, font=label_font if i == 0 else small_font, fill=(20, 20, 20))
        input_img = student_rgb if spec.dataset_kind == "student_rgb" else raw3
        images = [input_img, pseudo, pred_img, err_img, d1_img]
        for c, image in enumerate(images):
            x = label_w + c * tile_size[0]
            canvas.paste(resize_tile(image, tile_size), (x, y))
            draw.rectangle((x, y, x + tile_size[0] - 1, y + row_h - 1), outline=(190, 190, 190))

    draw.text(
        (12, height - 24),
        f"Rel error = abs(pred-pseudo)/pseudo, clipped at p{95:g}={error_vmax:.3g}; D1 pass threshold < 1.25.",
        font=small_font,
        fill=(60, 60, 60),
    )
    return canvas


def make_contact_sheet(panel_paths: list[Path], output_path: Path, *, thumb_width: int = 1050, columns: int = 1) -> None:
    thumbs = []
    for path in panel_paths:
        image = Image.open(path).convert("RGB")
        scale = thumb_width / image.width
        thumbs.append(image.resize((thumb_width, int(round(image.height * scale))), Image.Resampling.BILINEAR))
    gap = 14
    rows = int(math.ceil(len(thumbs) / columns))
    width = columns * thumb_width + (columns - 1) * gap
    height = sum(thumbs[r * columns].height for r in range(rows)) + (rows - 1) * gap
    sheet = Image.new("RGB", (width, height), (245, 245, 245))
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, columns)
        y = sum(thumbs[r * columns].height for r in range(row)) + row * gap
        x = col * (thumb_width + gap)
        sheet.paste(thumb, (x, y))
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else default_output_root()
    panels_dir = output_root / "sample_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else int(device.index))

    student_ds = RODRawStudentRGB(
        rod_root=args.rod_root,
        manifest_path=args.rod_manifest,
        split="01Valid",
        mode="val",
        size=(512, 960),
        crop_mode="center",
    )
    raw3_ds = RODRawRGB3(
        rod_root=args.rod_root,
        manifest_path=args.rod_manifest,
        split="01Valid",
        mode="val",
        size=(512, 960),
        crop_mode="center",
    )
    specs = exp_specs()
    loaded = [(*load_model(spec, device), spec) for spec in specs]
    use_amp = not args.no_amp
    tile_size = (int(args.tile_width), int(args.tile_height))
    panel_paths = []
    csv_rows = []

    for sample_index in args.sample_indices:
        if sample_index < 0 or sample_index >= len(raw3_ds):
            raise ValueError(f"sample index out of range: {sample_index}")
        rng = random.Random(42 + int(sample_index))
        student_sample = student_ds.build_sample(sample_index, rng=rng, include_geometry=True)
        raw_sample = raw3_ds.build_sample(sample_index, rng=random.Random(42 + int(sample_index)), include_geometry=True)
        sample_name = str(raw_sample["sample_name"])
        target = raw_sample["depth"].detach().cpu().numpy().astype(np.float32)
        valid = raw_sample["valid_mask"].detach().cpu().numpy().astype(bool) & np.isfinite(target) & (target > 0)

        results = []
        pred_arrays = [target]
        err_arrays = []
        for model, train_args, spec in loaded:
            sample = student_sample if spec.dataset_kind == "student_rgb" else raw_sample
            pred = run_model(model, train_args, sample, device, use_amp)
            result = build_result(pred, target, valid)
            results.append((spec, result))
            pred_arrays.append(result.aligned)
            err_arrays.append(result.rel_error)

        pred_vmin, pred_vmax = robust_limits(pred_arrays, valid, 1.0, 99.0)
        _, error_vmax = robust_limits(err_arrays, valid, 0.0, float(args.error_vmax_pct))
        pseudo_img = colorize(target, valid, vmin=pred_vmin, vmax=pred_vmax, cmap_name=args.pred_cmap)
        student_rgb_img = rgb_image_from_float(crop_student_rgb(student_sample))
        raw3_img = raw3_image(tensor_chw(raw_sample["raw"]))

        pred_rows = []
        for spec, result in results:
            pred_rows.append(
                (
                    spec,
                    result,
                    colorize(result.aligned, valid, vmin=pred_vmin, vmax=pred_vmax, cmap_name=args.pred_cmap),
                    colorize_error(result.rel_error, result.valid, vmax=error_vmax, cmap_name=args.error_cmap),
                    colorize_d1(result.d1_map, result.valid, args.d1_cmap),
                )
            )
            csv_rows.append(
                {
                    "sample_index": sample_index,
                    "sample_name": sample_name,
                    "experiment": spec.key,
                    "label": spec.label,
                    "checkpoint": spec.checkpoint,
                    **{key: result.metrics.get(key, float("nan")) for key in ("abs_rel", "rmse", "silog", "d1", "d2", "d3")},
                    "align_scale": result.align.get("scale", float("nan")),
                    "align_shift": result.align.get("shift", float("nan")),
                }
            )

        panel = make_sample_panel(
            sample_index=sample_index,
            sample_name=sample_name,
            student_rgb=student_rgb_img,
            raw3=raw3_img,
            pseudo=pseudo_img,
            pred_rows=pred_rows,
            tile_size=tile_size,
            error_vmax=error_vmax,
        )
        panel_path = panels_dir / f"rod_zeroshot_lod_ckpt_idx{sample_index:04d}_{sample_name}.png"
        panel.save(panel_path)
        panel_paths.append(panel_path)

    contact_sheet = output_root / "rod_zeroshot_lod_ckpt_contact_sheet.png"
    make_contact_sheet(panel_paths, contact_sheet)
    csv_path = output_root / "metrics_by_sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sample_index",
            "sample_name",
            "experiment",
            "label",
            "checkpoint",
            "abs_rel",
            "rmse",
            "silog",
            "d1",
            "d2",
            "d3",
            "align_scale",
            "align_shift",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    manifest_path = output_root / "manifest.json"
    manifest = {
        "output_root": str(output_root),
        "contact_sheet": str(contact_sheet),
        "panel_paths": [str(path) for path in panel_paths],
        "metrics_csv": str(csv_path),
        "sample_indices": [int(idx) for idx in args.sample_indices],
        "rod_manifest": str(args.rod_manifest),
        "experiments": [
            {"key": spec.key, "label": spec.label, "checkpoint": spec.checkpoint, "dataset_kind": spec.dataset_kind}
            for spec in specs
        ],
        "visualization_notes": {
            "prediction": "affine-aligned inverse-relative prediction; pseudo and predictions share robust p1-p99 color limits per sample",
            "error": f"relative error clipped at p{float(args.error_vmax_pct):g} per sample",
            "raw3_input": "ROD raw3 [R,(Gr+Gb)/2,B] visualized with per-sample p99.5 stretch",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] output_root={output_root}")
    print(f"[OK] contact_sheet={contact_sheet}")
    print(f"[OK] metrics_csv={csv_path}")
    print(f"[OK] manifest={manifest_path}")


if __name__ == "__main__":
    main()
