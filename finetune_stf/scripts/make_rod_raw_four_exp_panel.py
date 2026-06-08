#!/usr/bin/env python3
"""Make a one-sample ROD raw-RAM comparison panel for four final checkpoints."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import gc
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.config import ensure_resolved_config  # noqa: E402
from finetune_stf.dataset.lod_raw import _apply_crop  # noqa: E402
from finetune_stf.dataset.rod_raw import RODRaw  # noqa: E402
from finetune_stf.dataset.rod_raw_rgb import render_pipeline, unpack_raw24  # noqa: E402
from finetune_stf.dataset.rod_raw_student_rgb import RODRawStudentRGB  # noqa: E402
from finetune_stf.models.raw_ram import (  # noqa: E402
    packed_bayer_to_base_rgb,
    phase1b_tanh_tail_squash,
)
from finetune_stf.train import (  # noqa: E402
    build_model,
    build_rgb_reference_eval_model,
    resolve_model_state,
    strip_module_prefix,
)
from finetune_stf.util.metric import (  # noqa: E402
    affine_align_to_inverse_target,
    compute_inverse_relative_metrics,
)
from finetune_stf.util.model_input import select_model_input  # noqa: E402


DEFAULT_EXPERIMENTS = (
    PROJECT_ROOT / "finetune_stf/exp/0605_1332_rod_night_rawram3_identity_dav2s_ram_decoder_e5",
    PROJECT_ROOT / "finetune_stf/exp/0605_1844_rod_night_rawram3_identity_dav2s_ram_lora_tap_r8a16_decoder_e5",
    PROJECT_ROOT / "finetune_stf/exp/0606_0330_rod_night_rawram3_identity_dav2s_ram_backbone_ld09_decoder_e10",
    PROJECT_ROOT / "finetune_stf/exp/0606_1431_rod_night_rawram3_identity_dav2s_ram_backbone_lowlr_ld09_e10",
)
CSV_FIELDS = (
    "row",
    "kind",
    "experiment",
    "checkpoint",
    "sample_index",
    "sample_name",
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
    "x3_min",
    "x3_p01",
    "x3_mean",
    "x3_p50",
    "x3_p99",
    "x3_max",
)


@dataclass(frozen=True)
class PredictionResult:
    raw_pred: np.ndarray
    aligned_pred: np.ndarray
    metrics: dict[str, float]
    align_stats: dict[str, float]
    rel_error: np.ndarray
    error_valid: np.ndarray
    d1_score: np.ndarray
    d1_valid: np.ndarray
    d1_pass_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-dir",
        action="append",
        type=Path,
        default=None,
        help="Raw experiment directory. Can be passed multiple times; defaults to the four requested runs.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="last_epoch_model.pth",
        help="Checkpoint filename under heavy_save_path. Default is final/last checkpoint.",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="ROD 01Valid dataset index.")
    parser.add_argument("--sample-id", default=None, help="Optional exact ROD sample_id; overrides --sample-index.")
    parser.add_argument("--num-samples", type=int, default=1, help="Generate this many eval samples.")
    parser.add_argument(
        "--sample-selection",
        default="random",
        choices=["random", "sequential"],
        help="How to choose samples when --num-samples > 1 and --sample-indices is not set.",
    )
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for --sample-selection random.")
    parser.add_argument("--sample-step", type=int, default=1, help="Stride between samples when --num-samples > 1.")
    parser.add_argument("--sample-indices", default="", help="Comma-separated explicit eval sample indices.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="auto", help='Use "auto", "cuda", "cuda:0", or "cpu".')
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--tile-height", type=int, default=232)
    parser.add_argument("--pred-cmap", default="Spectral_r")
    parser.add_argument(
        "--d1-cmap",
        default="RdYlGn",
        help="Colormap for D1 pass map. 0=fail, 1=pass, invalid=black.",
    )
    parser.add_argument("--error-cmap", default="magma")
    parser.add_argument("--error-vmax-pct", type=float, default=99.0)
    parser.add_argument("--d1-threshold", type=float, default=1.25)
    parser.add_argument("--x3-vis-low-pct", type=float, default=1.0)
    parser.add_argument("--x3-vis-high-pct", type=float, default=99.0)
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs and checkpoints, then exit.")
    return parser.parse_args()


def parse_sample_indices_arg(value: str) -> list[int]:
    indices = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        indices.append(int(item))
    return indices


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_exp_args(exp_dir: Path) -> argparse.Namespace:
    config_path = exp_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    args = argparse.Namespace(**load_json(config_path))
    ensure_resolved_config(args)
    return args


def default_output_root() -> Path:
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    return PROJECT_ROOT / "finetune_stf/analysis/rod_raw_four_exp_panel" / stamp


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def safe_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sample"


def multi_sample_requested(args: argparse.Namespace) -> bool:
    return bool(parse_sample_indices_arg(args.sample_indices)) or int(args.num_samples) > 1


def build_exp_dirs(args: argparse.Namespace) -> tuple[Path, ...]:
    exp_dirs = tuple(args.exp_dir) if args.exp_dir else DEFAULT_EXPERIMENTS
    return tuple(path.expanduser().resolve() for path in exp_dirs)


def amp_dtype_from_args(exp_args: argparse.Namespace) -> torch.dtype:
    return torch.float16 if str(getattr(exp_args, "amp_dtype", "bf16")) == "fp16" else torch.bfloat16


def use_amp_for(device: torch.device, exp_args: argparse.Namespace, no_amp: bool) -> bool:
    return device.type == "cuda" and bool(getattr(exp_args, "amp", False)) and not bool(no_amp)


def build_raw_dataset(exp_args: argparse.Namespace) -> RODRaw:
    return RODRaw(
        rod_root=exp_args.rod_root,
        manifest_path=exp_args.rod_night_manifest,
        split="01Valid",
        size=(int(exp_args.input_height), int(exp_args.input_width)),
        mode="val",
        raw_source=exp_args.rod_raw_source,
        label_space=exp_args.rod_label_space,
        crop_mode=exp_args.rod_val_crop_mode,
    )


def build_student_rgb_dataset(exp_args: argparse.Namespace) -> RODRawStudentRGB:
    return RODRawStudentRGB(
        rod_root=exp_args.rod_root,
        manifest_path=exp_args.rod_night_manifest,
        split="01Valid",
        size=(int(exp_args.input_height), int(exp_args.input_width)),
        mode="val",
        raw_source=exp_args.rod_raw_source,
        rgb_pipeline=exp_args.rod_rgb_pipeline,
        label_space=exp_args.rod_label_space,
        crop_mode=exp_args.rod_val_crop_mode,
        student_white_percentile=exp_args.rod_student_white_percentile,
        student_gamma=exp_args.rod_student_gamma,
        student_channel_gains=tuple(exp_args.rod_student_channel_gains),
    )


def resolve_sample_index(dataset: RODRaw, sample_index: int, sample_id: str | None) -> int:
    if sample_id:
        for idx, row in enumerate(dataset.rows):
            if str(row.get("sample_id")) == str(sample_id):
                return int(idx)
        raise ValueError(f"sample_id={sample_id!r} was not found in ROD 01Valid")
    sample_index = int(sample_index)
    if sample_index < 0 or sample_index >= len(dataset):
        raise ValueError(f"--sample-index must be in [0, {len(dataset)}), got {sample_index}")
    return sample_index


def resolve_checkpoint(exp_args: argparse.Namespace, checkpoint_name: str) -> Path:
    checkpoint = Path(exp_args.heavy_save_path).expanduser().resolve() / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    return checkpoint


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint: Path) -> dict[str, object]:
    ckpt_obj = torch.load(checkpoint, map_location="cpu")
    state_dict = strip_module_prefix(resolve_model_state(ckpt_obj))
    model.load_state_dict(state_dict, strict=True)
    return {
        "path": str(checkpoint),
        "epoch": ckpt_obj.get("epoch") if isinstance(ckpt_obj, dict) else None,
        "best_metric": ckpt_obj.get("best_metric") if isinstance(ckpt_obj, dict) else None,
        "best_metrics": ckpt_obj.get("best_metrics") if isinstance(ckpt_obj, dict) else None,
    }


def stable_sample_seed(exp_args: argparse.Namespace, sample_index: int) -> int:
    return int(getattr(exp_args, "seed", 0)) * 1000003 + int(sample_index) * 9176 + 1337


def tensor_2d_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)


def infer_prediction(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    target_hw: tuple[int, int],
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> np.ndarray:
    model_input = model_input.to(device=device, non_blocking=True).float()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        pred = model(model_input).float()
    if pred.ndim != 3:
        raise ValueError(f"Expected model output (B,H,W), got {tuple(pred.shape)}")
    if tuple(pred.shape[-2:]) != tuple(target_hw):
        pred = F.interpolate(pred[:, None], target_hw, mode="bilinear", align_corners=True)[:, 0]
    return tensor_2d_np(pred[0])


def build_prediction_result(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    *,
    d1_threshold: float,
) -> PredictionResult:
    aligned, align_stats = affine_align_to_inverse_target(pred, target, valid_mask)
    metrics = compute_inverse_relative_metrics(aligned, target, valid_mask)
    if metrics is None:
        metrics = {key: float("nan") for key in ("abs_rel", "rmse", "silog", "d1", "d2", "d3")}

    valid = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(aligned)
        & np.isfinite(target)
        & (aligned > 0)
        & (target > 0)
    )
    rel_error = np.full(target.shape, np.nan, dtype=np.float32)
    if np.any(valid):
        rel_error[valid] = (
            np.abs(aligned[valid] - target[valid]) / np.clip(target[valid], 1e-6, None)
        ).astype(np.float32)
    score = np.full(target.shape, np.nan, dtype=np.float32)
    pass_pixels = 0
    if np.any(valid):
        thresh = np.maximum(target[valid] / aligned[valid], aligned[valid] / target[valid])
        passed = thresh < float(d1_threshold)
        score[valid] = passed.astype(np.float32)
        pass_pixels = int(np.count_nonzero(passed))
    return PredictionResult(
        raw_pred=np.asarray(pred, dtype=np.float32),
        aligned_pred=np.asarray(aligned, dtype=np.float32),
        metrics={key: float(value) for key, value in metrics.items()},
        align_stats={key: float(value) for key, value in align_stats.items()},
        rel_error=rel_error,
        error_valid=valid,
        d1_score=score,
        d1_valid=valid,
        d1_pass_pixels=pass_pixels,
    )


def robust_limits(arrays: list[np.ndarray], valid_mask: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> tuple[float, float]:
    values = []
    for array in arrays:
        arr = np.asarray(array, dtype=np.float32)
        mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)
        if np.any(mask):
            values.append(arr[mask].reshape(-1))
    if not values:
        return 0.0, 1.0
    flat = np.concatenate(values)
    lo, hi = np.percentile(flat, [float(low_pct), float(high_pct)])
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(flat))
        hi = float(np.nanmax(flat))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


def colorize_array(array: np.ndarray, valid_mask: np.ndarray, *, vmin: float, vmax: float, cmap_name: str) -> Image.Image:
    arr = np.asarray(array, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)
    norm = np.clip((arr - float(vmin)) / max(float(vmax) - float(vmin), 1e-6), 0.0, 1.0)
    cmap = colormaps.get_cmap(cmap_name)
    rgb = (cmap(norm)[:, :, :3] * 255.0).round().astype(np.uint8)
    rgb[~valid] = 0
    return Image.fromarray(rgb)


def colorize_d1(score: np.ndarray, valid_mask: np.ndarray, *, cmap_name: str) -> Image.Image:
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(score)
    values = np.where(valid, np.asarray(score, dtype=np.float32), 0.0)
    cmap = colormaps.get_cmap(cmap_name)
    rgb = (cmap(np.clip(values, 0.0, 1.0))[:, :, :3] * 255.0).round().astype(np.uint8)
    rgb[~valid] = 0
    return Image.fromarray(rgb)


def colorize_error(error: np.ndarray, valid_mask: np.ndarray, *, vmax: float, cmap_name: str) -> Image.Image:
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(error)
    values = np.where(valid, np.asarray(error, dtype=np.float32), 0.0)
    norm = np.clip(values / max(float(vmax), 1e-6), 0.0, 1.0)
    cmap = colormaps.get_cmap(cmap_name)
    rgb = (cmap(norm)[:, :, :3] * 255.0).round().astype(np.uint8)
    rgb[~valid] = 0
    return Image.fromarray(rgb)


def rgb_float_to_image(rgb: np.ndarray) -> Image.Image:
    arr = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def crop_pipeline_rgb(sample: dict[str, object], pipeline: str) -> np.ndarray:
    raw = unpack_raw24(str(sample["raw_path"]))
    rgb = render_pipeline(raw, pipeline).astype(np.float32) / 255.0
    geometry = sample.get("geometry_params", {})
    crop_box = geometry.get("crop_box") if isinstance(geometry, dict) else None
    if crop_box is None:
        raise ValueError("ROD sample was built without crop geometry")
    return _apply_crop(rgb, tuple(int(v) for v in crop_box))


def x3_before_backbone(
    model: torch.nn.Module,
    raw_input: torch.Tensor,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> np.ndarray:
    module = model.module if hasattr(model, "module") else model
    if not hasattr(module, "ram_core"):
        raise ValueError("Model does not expose ram_core")
    raw_input = raw_input.to(device=device, non_blocking=True).float()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        if raw_input.shape[1] == 4 and getattr(module, "uses_base_rgb", False):
            x3_input = packed_bayer_to_base_rgb(raw_input)
        else:
            x3_input = raw_input
        x3, _ = module.ram_core.forward_with_features(x3_input)
        if getattr(module, "raw_ram_rgb_tail", "identity") == "tanh2p5":
            x3 = phase1b_tanh_tail_squash(x3)
    return x3[0].detach().float().cpu().numpy().astype(np.float32, copy=False)


def x3_preview_image(x3_chw: np.ndarray, *, low_pct: float, high_pct: float) -> Image.Image:
    x3 = np.asarray(x3_chw, dtype=np.float32)
    if x3.ndim != 3 or x3.shape[0] != 3:
        raise ValueError(f"Expected x3 CHW, got {x3.shape}")
    hwc = np.transpose(x3, (1, 2, 0))
    valid = np.isfinite(hwc)
    if np.any(valid):
        lo, hi = np.percentile(hwc[valid], [float(low_pct), float(high_pct)])
    else:
        lo, hi = 0.0, 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(hwc)), float(np.nanmax(hwc))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    vis = np.clip((hwc - float(lo)) / max(float(hi) - float(lo), 1e-6), 0.0, 1.0)
    return rgb_float_to_image(vis)


def x3_stats(x3_chw: np.ndarray) -> dict[str, float]:
    flat = np.asarray(x3_chw, dtype=np.float32).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {key: float("nan") for key in ("min", "p01", "mean", "p50", "p99", "max")}
    return {
        "min": float(np.min(flat)),
        "p01": float(np.percentile(flat, 1)),
        "mean": float(np.mean(flat)),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(np.max(flat)),
    }


def x3_hist_image(
    x3_chw: np.ndarray,
    *,
    size: tuple[int, int],
    title: str,
    labels: tuple[str, str, str] = ("ch0", "ch1", "ch2"),
) -> Image.Image:
    width, height = size
    x3 = np.asarray(x3_chw, dtype=np.float32)
    fig_w = max(width / 120.0, 1.0)
    fig_h = max(height / 120.0, 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = ("tab:red", "tab:green", "tab:blue")
    finite = x3[np.isfinite(x3)]
    if finite.size:
        lo, hi = np.percentile(finite, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(finite)), float(np.max(finite))
        for ch, color, label in zip(x3, colors, labels):
            values = ch[np.isfinite(ch)].reshape(-1)
            if values.size:
                values = values[:: max(values.size // 80000, 1)]
                ax.hist(values, bins=80, range=(lo, hi), density=True, histtype="step", linewidth=1.2, color=color, label=label)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("value", fontsize=7)
    ax.set_ylabel("density", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc="upper right", frameon=False)
    fig.tight_layout(pad=0.35)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    rgb = rgba[:, :, :3].copy()
    image = Image.fromarray(rgb)
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image


def blank_tile(size: tuple[int, int], text: str = "") -> Image.Image:
    image = Image.new("RGB", size, (248, 248, 248))
    if text:
        draw = ImageDraw.Draw(image)
        draw.text((12, 12), text, font=load_font(13), fill=(90, 90, 90))
    return image


def resize_tile(image: Image.Image, tile_size: tuple[int, int]) -> Image.Image:
    if image.size == tile_size:
        return image.convert("RGB")
    return image.convert("RGB").resize(tile_size, Image.Resampling.BILINEAR)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_multiline(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str], font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += int(font.size * 1.28) if hasattr(font, "size") else 14


def make_panel(
    rows: list[dict[str, object]],
    *,
    tile_size: tuple[int, int],
    sample_name: str,
    pred_cmap: str,
    error_cmap: str,
    d1_cmap: str,
    d1_threshold: float,
    error_vmax: float,
) -> Image.Image:
    tile_w, tile_h = tile_size
    label_w = 255
    header_h = 64
    footer_h = 32
    cols = ("Input / x3", "Pseudo / dist", "Pred", "Error", "D1 map")
    width = label_w + tile_w * len(cols)
    height = header_h + tile_h * len(rows) + footer_h
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(18)
    header_font = load_font(15)
    label_font = load_font(13)
    small_font = load_font(11)

    draw.text((12, 12), f"ROD raw-RAM final ckpt panel: {sample_name}", font=title_font, fill=(20, 20, 20))
    draw.text(
        (12, 38),
        f"cmaps: {pred_cmap} / {error_cmap} / {d1_cmap}",
        font=small_font,
        fill=(70, 70, 70),
    )
    for col_idx, col_name in enumerate(cols):
        x = label_w + col_idx * tile_w + 10
        draw.text((x, 38), col_name, font=header_font, fill=(20, 20, 20))

    for row_idx, row in enumerate(rows):
        y = header_h + row_idx * tile_h
        bg = (235, 235, 235) if row_idx % 2 == 0 else (228, 228, 228)
        draw.rectangle((0, y, label_w - 1, y + tile_h - 1), fill=bg)
        label_lines = list(row["label_lines"])
        draw_multiline(draw, (12, y + 12), label_lines, label_font, (20, 20, 20))
        for col_idx, image in enumerate(row["images"]):
            x = label_w + col_idx * tile_w
            canvas.paste(resize_tile(image, tile_size), (x, y))
            draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=(190, 190, 190), width=1)

    footer_y = header_h + tile_h * len(rows) + 8
    draw.text(
        (12, footer_y),
        (
            "Error=abs(aligned_pred-pseudo)/pseudo, "
            f"clipped at {error_vmax:.3g}; D1 pass threshold < {d1_threshold:g}."
        ),
        font=small_font,
        fill=(60, 60, 60),
    )
    return canvas


def metric_line(metrics: dict[str, float]) -> str:
    return "absrel {abs_rel:.3g} d1 {d1:.3f} silog {silog:.3g}".format(
        abs_rel=float(metrics.get("abs_rel", float("nan"))),
        d1=float(metrics.get("d1", float("nan"))),
        silog=float(metrics.get("silog", float("nan"))),
    )


def csv_row(
    *,
    row: int,
    kind: str,
    experiment: str,
    checkpoint: str,
    sample_index: int,
    sample_name: str,
    result: PredictionResult,
    x3_stat: dict[str, float] | None = None,
) -> dict[str, object]:
    x3_stat = x3_stat or {}
    out = {
        "row": row,
        "kind": kind,
        "experiment": experiment,
        "checkpoint": checkpoint,
        "sample_index": sample_index,
        "sample_name": sample_name,
        "valid_pixels": int(np.count_nonzero(result.d1_valid)),
        "d1_pass_pixels": int(result.d1_pass_pixels),
    }
    for key in ("abs_rel", "rmse", "silog", "d1", "d2", "d3"):
        out[key] = float(result.metrics.get(key, float("nan")))
    out["align_scale"] = float(result.align_stats.get("scale", float("nan")))
    out["align_shift"] = float(result.align_stats.get("shift", float("nan")))
    for key in ("min", "p01", "mean", "p50", "p99", "max"):
        out[f"x3_{key}"] = x3_stat.get(key, "")
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plan_sample_indices(args: argparse.Namespace, dataset_size: int) -> list[int]:
    explicit = parse_sample_indices_arg(args.sample_indices)
    if explicit:
        indices = explicit
    else:
        if args.sample_id:
            raise ValueError("--sample-id cannot be combined with --num-samples; use --sample-indices instead")
        count = int(args.num_samples)
        if count < 1:
            raise ValueError("--num-samples must be >= 1")
        if count > int(dataset_size):
            raise ValueError(f"--num-samples={count} exceeds dataset size {dataset_size}")
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


def child_command_for_sample(args: argparse.Namespace, exp_dirs: tuple[Path, ...], sample_index: int, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--sample-index",
        str(int(sample_index)),
        "--num-samples",
        "1",
        "--checkpoint-name",
        str(args.checkpoint_name),
        "--output-root",
        str(output_dir),
        "--device",
        str(args.device),
        "--tile-width",
        str(int(args.tile_width)),
        "--tile-height",
        str(int(args.tile_height)),
        "--pred-cmap",
        str(args.pred_cmap),
        "--error-cmap",
        str(args.error_cmap),
        "--error-vmax-pct",
        str(float(args.error_vmax_pct)),
        "--d1-cmap",
        str(args.d1_cmap),
        "--d1-threshold",
        str(float(args.d1_threshold)),
        "--x3-vis-low-pct",
        str(float(args.x3_vis_low_pct)),
        "--x3-vis-high-pct",
        str(float(args.x3_vis_high_pct)),
    ]
    for exp_dir in exp_dirs:
        cmd.extend(["--exp-dir", str(exp_dir)])
    if args.no_amp:
        cmd.append("--no-amp")
    return cmd


def make_contact_sheet(panel_paths: list[Path], output_path: Path, *, thumb_width: int = 900, columns: int = 2) -> None:
    if not panel_paths:
        return
    columns = max(int(columns), 1)
    thumbs = []
    for panel_path in panel_paths:
        image = Image.open(panel_path).convert("RGB")
        ratio = float(thumb_width) / float(image.width)
        thumb_size = (int(thumb_width), max(int(round(image.height * ratio)), 1))
        thumbs.append(image.resize(thumb_size, Image.Resampling.LANCZOS))
    rows = (len(thumbs) + columns - 1) // columns
    cell_w = max(thumb.width for thumb in thumbs)
    cell_h = max(thumb.height for thumb in thumbs)
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), (245, 245, 245))
    for idx, thumb in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        x = col * cell_w
        y = row * cell_h
        sheet.paste(thumb, (x, y))
    sheet.save(output_path)


def run_multi_sample_mode(args: argparse.Namespace) -> None:
    exp_dirs = build_exp_dirs(args)
    exp_args_list = [load_exp_args(exp_dir) for exp_dir in exp_dirs]
    ref_args = exp_args_list[0]
    dataset = build_raw_dataset(ref_args)
    sample_indices = plan_sample_indices(args, len(dataset))
    output_root = (args.output_root.expanduser().resolve() if args.output_root else default_output_root())
    panels_dir = output_root / "sample_panels"
    metadata_dir = output_root / "metadata"

    planned = {
        "sample_indices": sample_indices,
        "sample_selection": str(args.sample_selection) if not parse_sample_indices_arg(args.sample_indices) else "explicit_indices",
        "sample_seed": int(args.sample_seed),
        "output_root": str(output_root),
        "panels_dir": str(panels_dir),
        "metadata_dir": str(metadata_dir),
        "experiments": [str(path) for path in exp_dirs],
    }
    if args.dry_run:
        print(json.dumps(planned, indent=2, sort_keys=True))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    combined_rows = []
    panel_paths: list[Path] = []
    for sample_index in sample_indices:
        sample_dir = metadata_dir / f"sample_{int(sample_index):04d}"
        cmd = child_command_for_sample(args, exp_dirs, sample_index, sample_dir)
        subprocess.run(cmd, check=True)
        summary_path = sample_dir / "summary.json"
        csv_path = sample_dir / "metrics_by_row.csv"
        summary = load_json(summary_path)
        panel_src = Path(summary["panel_path"])
        panel_dst = panels_dir / panel_src.name
        if panel_dst.exists():
            panel_dst.unlink()
        shutil.move(str(panel_src), str(panel_dst))
        summary["panel_path"] = str(panel_dst)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summaries.append(summary)
        panel_paths.append(panel_dst)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                combined = {
                    "panel_path": summary["panel_path"],
                    "summary_path": str(summary_path),
                    **row,
                }
                combined_rows.append(combined)

    contact_sheet_path = output_root / f"rod_raw_four_exp_panel_{len(sample_indices)}samples_contact_sheet.png"
    make_contact_sheet(panel_paths, contact_sheet_path)

    combined_csv_path = output_root / "metrics_by_sample_row.csv"
    combined_fields = ["panel_path", "summary_path", *CSV_FIELDS]
    with combined_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=combined_fields)
        writer.writeheader()
        writer.writerows(combined_rows)

    multi_summary_path = output_root / f"summary_{len(sample_indices)}samples.json"
    payload = {
        **planned,
        "num_samples": len(sample_indices),
        "contact_sheet_path": str(contact_sheet_path),
        "combined_csv_path": str(combined_csv_path),
        "sample_summaries": summaries,
    }
    multi_summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] output_root={output_root}")
    print(f"[OK] contact_sheet={contact_sheet_path}")
    print(f"[OK] combined_csv={combined_csv_path}")
    print(f"[OK] summary={multi_summary_path}")


def main() -> None:
    args = parse_args()
    if multi_sample_requested(args):
        run_multi_sample_mode(args)
        return

    exp_dirs = build_exp_dirs(args)
    exp_args_list = [load_exp_args(exp_dir) for exp_dir in exp_dirs]

    if not exp_args_list:
        raise ValueError("At least one experiment directory is required")
    ref_args = exp_args_list[0]
    for exp_args in exp_args_list:
        if exp_args.resolved_config.dataset_family != "rod_raw":
            raise ValueError(f"{exp_args.save_path} is not a rod_raw experiment")
        if exp_args.resolved_config.model_input_tensor != "raw":
            raise ValueError(f"{exp_args.save_path} does not use raw model input")
        if exp_args.rod_night_manifest != ref_args.rod_night_manifest:
            raise ValueError("All experiments must use the same ROD manifest")
        if (int(exp_args.input_height), int(exp_args.input_width)) != (int(ref_args.input_height), int(ref_args.input_width)):
            raise ValueError("All experiments must use the same input size")

    raw_dataset = build_raw_dataset(ref_args)
    rgb_dataset = build_student_rgb_dataset(ref_args)
    sample_index = resolve_sample_index(raw_dataset, args.sample_index, args.sample_id)
    sample_seed = stable_sample_seed(ref_args, sample_index)
    raw_sample = raw_dataset.build_sample(sample_index, rng=random.Random(sample_seed), include_geometry=True)
    rgb_sample = rgb_dataset.build_sample(sample_index, rng=random.Random(sample_seed), include_geometry=True)
    sample_name = str(raw_sample.get("sample_name", f"sample_{sample_index:06d}"))
    target = raw_sample["depth"].detach().float().cpu().numpy().astype(np.float32)
    valid_mask = raw_sample["valid_mask"].detach().cpu().numpy().astype(bool)
    valid_mask = valid_mask & np.isfinite(target) & (target > 0)
    target_hw = tuple(int(v) for v in target.shape)

    checkpoints = [resolve_checkpoint(exp_args, args.checkpoint_name) for exp_args in exp_args_list]
    output_root = (args.output_root.expanduser().resolve() if args.output_root else default_output_root())

    dry_payload = {
        "experiments": [str(path) for path in exp_dirs],
        "checkpoints": [str(path) for path in checkpoints],
        "sample_index": int(sample_index),
        "sample_name": sample_name,
        "valid_pixels": int(np.count_nonzero(valid_mask)),
        "output_root": str(output_root),
    }
    if args.dry_run:
        print(json.dumps(dry_payload, indent=2, sort_keys=True))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else int(device.index))

    rows_for_panel: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    aligned_for_limits: list[np.ndarray] = [target]

    teacher_rgb = crop_pipeline_rgb(raw_sample, "teacher_bright_degreen_v1")
    student_rgb = crop_pipeline_rgb(raw_sample, "student_dark_degreen_v1")
    pseudo_placeholder = None

    rgb_model = build_rgb_reference_eval_model(ref_args)
    rgb_model.to(device).eval()
    rgb_input = select_model_input(
        rgb_sample,
        "image",
        dataset_family="rod_raw_student_rgb",
        sample_source="rod_val",
        add_batch_dim=True,
    )
    rgb_pred = infer_prediction(
        rgb_model,
        rgb_input,
        target_hw,
        device=device,
        amp_dtype=amp_dtype_from_args(ref_args),
        use_amp=use_amp_for(device, ref_args, args.no_amp),
    )
    rgb_result = build_prediction_result(rgb_pred, target, valid_mask, d1_threshold=args.d1_threshold)
    aligned_for_limits.append(rgb_result.aligned_pred)
    del rgb_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    raw_records = []
    for exp_dir, exp_args, checkpoint in zip(exp_dirs, exp_args_list, checkpoints):
        model = build_model(exp_args)
        ckpt_meta = load_checkpoint_into_model(model, checkpoint)
        model.to(device).eval()
        raw_input = select_model_input(
            raw_sample,
            exp_args.resolved_config.model_input_tensor,
            dataset_family=exp_args.resolved_config.dataset_family,
            sample_source="rod_val",
            add_batch_dim=True,
        )
        amp_dtype = amp_dtype_from_args(exp_args)
        use_amp = use_amp_for(device, exp_args, args.no_amp)
        pred = infer_prediction(
            model,
            raw_input,
            target_hw,
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
        )
        result = build_prediction_result(pred, target, valid_mask, d1_threshold=args.d1_threshold)
        x3 = x3_before_backbone(
            model,
            raw_input,
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
        )
        stat = x3_stats(x3)
        aligned_for_limits.append(result.aligned_pred)
        raw_records.append(
            {
                "exp_dir": exp_dir,
                "exp_args": exp_args,
                "checkpoint": checkpoint,
                "checkpoint_meta": ckpt_meta,
                "result": result,
                "x3": x3,
                "x3_stats": stat,
            }
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pred_vmin, pred_vmax = robust_limits(aligned_for_limits, valid_mask)
    error_arrays = [rgb_result.rel_error] + [record["result"].rel_error for record in raw_records]
    _, error_vmax = robust_limits(error_arrays, valid_mask, low_pct=0.0, high_pct=float(args.error_vmax_pct))
    pseudo_image = colorize_array(target, valid_mask, vmin=pred_vmin, vmax=pred_vmax, cmap_name=args.pred_cmap)
    pseudo_placeholder = pseudo_image

    rows_for_panel.append(
        {
            "label_lines": [
                "Reference",
                "teacher RGB",
                "pseudo label",
            ],
            "images": [
                rgb_float_to_image(teacher_rgb),
                pseudo_placeholder,
                blank_tile((int(args.tile_width), int(args.tile_height))),
                blank_tile((int(args.tile_width), int(args.tile_height))),
                blank_tile((int(args.tile_width), int(args.tile_height))),
            ],
        }
    )
    rows_for_panel.append(
        {
            "label_lines": [
                "Student RGB",
                "DAv2-S init",
                metric_line(rgb_result.metrics),
            ],
            "images": [
                rgb_float_to_image(student_rgb),
                x3_hist_image(
                    np.transpose(student_rgb, (2, 0, 1)),
                    size=(int(args.tile_width), int(args.tile_height)),
                    title="student RGB dist",
                    labels=("R", "G", "B"),
                ),
                colorize_array(rgb_result.aligned_pred, valid_mask, vmin=pred_vmin, vmax=pred_vmax, cmap_name=args.pred_cmap),
                colorize_error(rgb_result.rel_error, rgb_result.error_valid, vmax=error_vmax, cmap_name=args.error_cmap),
                colorize_d1(rgb_result.d1_score, rgb_result.d1_valid, cmap_name=args.d1_cmap),
            ],
        }
    )
    csv_rows.append(
        csv_row(
            row=1,
            kind="student_rgb_dav2s_init",
            experiment="DAv2-S student RGB init",
            checkpoint=str(ref_args.pretrained_from),
            sample_index=sample_index,
            sample_name=sample_name,
            result=rgb_result,
        )
    )

    for row_idx, record in enumerate(raw_records, start=2):
        exp_dir = record["exp_dir"]
        result = record["result"]
        stat = record["x3_stats"]
        exp_label = exp_dir.name.split("_rod_night_", 1)[0]
        label_lines = [
            exp_label,
            "final/last ckpt",
            metric_line(result.metrics),
            f"x3 p1/p99 {stat['p01']:.2g}/{stat['p99']:.2g}",
        ]
        rows_for_panel.append(
            {
                "label_lines": label_lines,
                "images": [
                    x3_preview_image(record["x3"], low_pct=args.x3_vis_low_pct, high_pct=args.x3_vis_high_pct),
                    x3_hist_image(record["x3"], size=(int(args.tile_width), int(args.tile_height)), title=f"{exp_label} x3"),
                    colorize_array(result.aligned_pred, valid_mask, vmin=pred_vmin, vmax=pred_vmax, cmap_name=args.pred_cmap),
                    colorize_error(result.rel_error, result.error_valid, vmax=error_vmax, cmap_name=args.error_cmap),
                    colorize_d1(result.d1_score, result.d1_valid, cmap_name=args.d1_cmap),
                ],
            }
        )
        csv_rows.append(
            csv_row(
                row=row_idx,
                kind="raw_ram_final",
                experiment=exp_dir.name,
                checkpoint=str(record["checkpoint"]),
                sample_index=sample_index,
                sample_name=sample_name,
                result=result,
                x3_stat=stat,
            )
        )

    panel = make_panel(
        rows_for_panel,
        tile_size=(int(args.tile_width), int(args.tile_height)),
        sample_name=sample_name,
        pred_cmap=args.pred_cmap,
        error_cmap=args.error_cmap,
        d1_cmap=args.d1_cmap,
        d1_threshold=args.d1_threshold,
        error_vmax=error_vmax,
    )
    panel_path = output_root / f"rod_raw_four_exp_panel_idx{sample_index:04d}_{safe_filename_part(sample_name)}.png"
    panel.save(panel_path)

    csv_path = output_root / "metrics_by_row.csv"
    write_csv(csv_path, csv_rows)
    summary_path = output_root / "summary.json"
    summary = {
        **dry_payload,
        "panel_path": str(panel_path),
        "csv_path": str(csv_path),
        "prediction_colormap": str(args.pred_cmap),
        "error_colormap": str(args.error_cmap),
        "error_definition": "abs(aligned inverse-relative prediction - pseudo inverse-relative label) / pseudo inverse-relative label",
        "error_vmax": float(error_vmax),
        "error_vmax_percentile": float(args.error_vmax_pct),
        "d1_colormap": str(args.d1_cmap),
        "d1_definition": f"per-pixel delta1 pass map after affine alignment; pass means max(target/pred,pred/target) < {float(args.d1_threshold):g}",
        "pred_visualization": "aligned inverse-relative prediction, colored with shared pseudo/pred robust limits",
        "pred_vmin": float(pred_vmin),
        "pred_vmax": float(pred_vmax),
        "raw_rows": [
            {
                "experiment": record["exp_dir"].name,
                "checkpoint": str(record["checkpoint"]),
                "checkpoint_meta": record["checkpoint_meta"],
                "x3_stats": record["x3_stats"],
                "metrics": record["result"].metrics,
                "align_stats": record["result"].align_stats,
            }
            for record in raw_records
        ],
        "reference_row": {
            "checkpoint": str(ref_args.pretrained_from),
            "metrics": rgb_result.metrics,
            "align_stats": rgb_result.align_stats,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[OK] panel={panel_path}")
    print(f"[OK] summary={summary_path}")
    print(f"[OK] csv={csv_path}")


if __name__ == "__main__":
    main()
