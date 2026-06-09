#!/usr/bin/env python3
"""Make true-LOD same-sample panels across the Plan B e40 experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import gc
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.scripts.make_lod_true_rgb_raw_panel import (  # noqa: E402
    build_raw_dataset,
    build_rgb_dataset,
    crop_rgb_dark,
    crop_rgb_normal,
    parse_indices,
    plan_indices,
    resolve_checkpoint,
    resolve_device,
    rgb_image,
    sample_seed,
    stat_payload,
    stats_to_csv_prefix,
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
)
from finetune_stf.train import build_model, load_initial_weights  # noqa: E402
from finetune_stf.util.model_input import select_model_input  # noqa: E402


DEFAULT_EXPERIMENTS: tuple[tuple[str, str, str], ...] = (
    (
        "R0",
        "RGB baseline_e10 best",
        "finetune_stf/exp/0608_1509_lod_true_rgb_dark_block8excl10_dav2s_decoder_e40_R0_aug-baseline_e10_poly",
    ),
    (
        "Rg",
        "RGB geom best",
        "finetune_stf/exp/0608_1605_lod_true_rgb_dark_block8excl10_dav2s_decoder_e40_Rg_aug-geom_poly",
    ),
    (
        "R1",
        "RGB medium best",
        "finetune_stf/exp/0608_1659_lod_true_rgb_dark_block8excl10_dav2s_decoder_e40_R1_aug-medium_poly",
    ),
    (
        "W0",
        "RAW baseline_e10 best",
        "finetune_stf/exp/0608_1739_lod_true_raw_rgb16_block8excl10_ram3_dav2s_decoder_e40_W0_aug-baseline_e10_poly",
    ),
    (
        "Wg",
        "RAW geom best",
        "finetune_stf/exp/0608_1811_lod_true_raw_rgb16_block8excl10_ram3_dav2s_decoder_e40_Wg_aug-geom_poly",
    ),
    (
        "W1",
        "RAW medium best",
        "finetune_stf/exp/0608_1844_lod_true_raw_rgb16_block8excl10_ram3_dav2s_decoder_e40_W1_aug-medium_poly",
    ),
)

CSV_FIELDS = (
    "sample_index",
    "sample_name",
    "row_label",
    "domain",
    "experiment",
    "checkpoint",
    "checkpoint_epoch",
    "best_lod_d1",
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class ExperimentSpec:
    label: str
    display_name: str
    exp_dir: Path


@dataclass
class SampleBase:
    sample_index: int
    sample_name: str
    rgb_sample: dict[str, object]
    raw_sample: dict[str, object]
    target: np.ndarray
    valid_mask: np.ndarray
    rgb_normal: np.ndarray
    rgb_dark: np.ndarray
    rgb_dark_chw: np.ndarray


@dataclass
class RowResult:
    spec: ExperimentSpec
    domain: str
    exp_name: str
    checkpoint: Path
    checkpoint_meta: dict[str, object]
    result: object
    backbone_stats: dict[str, float]
    input_image: Image.Image
    dist_image: Image.Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help=(
            "Override experiment list. Format label=display=exp_dir. "
            "May be repeated. Defaults to the six Plan B block8excl10 e40 runs."
        ),
    )
    parser.add_argument("--checkpoint-name", default="best_model.pth")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--sample-selection", default="random", choices=["random", "sequential"])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--sample-indices", default="")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tile-width", type=int, default=340)
    parser.add_argument("--tile-height", type=int, default=218)
    parser.add_argument("--label-width", type=int, default=360)
    parser.add_argument("--pred-cmap", default="Spectral_r")
    parser.add_argument("--error-cmap", default="magma")
    parser.add_argument("--error-vmax-pct", type=float, default=99.0)
    parser.add_argument("--d1-cmap", default="RdYlGn")
    parser.add_argument("--d1-threshold", type=float, default=1.25)
    parser.add_argument("--backbone-vis-low-pct", type=float, default=1.0)
    parser.add_argument("--backbone-vis-high-pct", type=float, default=99.0)
    parser.add_argument(
        "--input-vmin",
        type=float,
        default=-2.64,
        help="Fixed lower bound for the shared backbone-input display window (no percentile stretch).",
    )
    parser.add_argument(
        "--input-vmax",
        type=float,
        default=2.64,
        help="Fixed upper bound for the shared backbone-input display window (no percentile stretch).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output_root(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    selection = "explicit" if parse_indices(args.sample_indices) else f"{args.sample_selection}{args.num_samples}_seed{args.sample_seed}"
    checkpoint = Path(str(args.checkpoint_name)).stem
    return (
        PROJECT_ROOT
        / "finetune_stf/analysis/lod_true_rgb_raw_panel"
        / f"{stamp}_planb_block8excl10_{checkpoint}_{selection}_cross_exp"
    )


def parse_experiment_specs(values: list[str]) -> list[ExperimentSpec]:
    if not values:
        return [
            ExperimentSpec(label=label, display_name=display, exp_dir=(PROJECT_ROOT / rel_path).resolve())
            for label, display, rel_path in DEFAULT_EXPERIMENTS
        ]
    specs: list[ExperimentSpec] = []
    for value in values:
        parts = str(value).split("=", 2)
        if len(parts) != 3:
            raise ValueError("--experiment must use label=display=exp_dir")
        label, display, exp_dir = parts
        specs.append(ExperimentSpec(label=label.strip(), display_name=display.strip(), exp_dir=Path(exp_dir).expanduser().resolve()))
    if not specs:
        raise ValueError("At least one experiment is required")
    return specs


def is_rgb_exp(exp_args: argparse.Namespace) -> bool:
    return str(exp_args.resolved_config.dataset_family) == "lod_true_rgb_dark"


def is_raw_exp(exp_args: argparse.Namespace) -> bool:
    return str(exp_args.resolved_config.dataset_family) in {
        "lod_true_raw_dark_rgb16",
        "lod_true_raw_normal_rgb16",
    }


def amp_dtype_from_args(exp_args: argparse.Namespace) -> torch.dtype:
    return torch.float16 if str(getattr(exp_args, "amp_dtype", "bf16")) == "fp16" else torch.bfloat16


def metric_value(checkpoint_meta: dict[str, object], key: str) -> float:
    best_metrics = checkpoint_meta.get("best_metrics", {})
    if isinstance(best_metrics, dict) and key in best_metrics:
        try:
            return float(best_metrics[key])
        except Exception:
            return float("nan")
    return float("nan")


def checkpoint_epoch(checkpoint_meta: dict[str, object]) -> int | None:
    if "epoch" not in checkpoint_meta:
        return None
    try:
        return int(checkpoint_meta["epoch"])
    except Exception:
        return None


def build_sample_bases(
    *,
    sample_indices: list[int],
    rgb_args: argparse.Namespace,
    raw_args: argparse.Namespace,
    rgb_dataset,
    raw_dataset,
) -> list[SampleBase]:
    bases: list[SampleBase] = []
    for sample_index in sample_indices:
        seed = sample_seed(rgb_args, sample_index)
        rgb_sample = rgb_dataset.build_sample(sample_index, rng=random.Random(seed), include_geometry=True)
        raw_sample = raw_dataset.build_sample(sample_index, rng=random.Random(seed), include_geometry=True)
        sample_name = str(rgb_sample["sample_name"])
        if sample_name != str(raw_sample["sample_name"]):
            raise ValueError(f"RGB/RAW sample mismatch at idx={sample_index}: {sample_name} vs {raw_sample['sample_name']}")

        target = rgb_sample["depth"].detach().float().cpu().numpy().astype(np.float32)
        valid_mask = rgb_sample["valid_mask"].detach().cpu().numpy().astype(bool)
        valid_mask = valid_mask & np.isfinite(target) & (target > 0)
        rgb_normal = crop_rgb_normal(rgb_sample)
        rgb_dark = crop_rgb_dark(rgb_sample)
        rgb_dark_chw = np.transpose(rgb_dark, (2, 0, 1)).astype(np.float32, copy=False)
        bases.append(
            SampleBase(
                sample_index=int(sample_index),
                sample_name=sample_name,
                rgb_sample=rgb_sample,
                raw_sample=raw_sample,
                target=target,
                valid_mask=valid_mask,
                rgb_normal=rgb_normal,
                rgb_dark=rgb_dark,
                rgb_dark_chw=rgb_dark_chw,
            )
        )
    return bases


def input_for_domain(
    *,
    exp_args: argparse.Namespace,
    base: SampleBase,
    domain: str,
) -> torch.Tensor:
    sample = base.rgb_sample if domain == "rgb" else base.raw_sample
    return select_model_input(
        sample,
        exp_args.resolved_config.model_input_tensor,
        dataset_family=exp_args.resolved_config.dataset_family,
        sample_source="lod_val",
        add_batch_dim=True,
    )


def to_chw_numpy(tensor: object) -> np.ndarray:
    """Detach a (possibly batched) backbone-input tensor to a CHW float32 array."""
    if hasattr(tensor, "detach"):
        arr = tensor.detach().float().cpu().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.float32, copy=False)


def fixed_window_image(chw: np.ndarray, *, vmin: float, vmax: float) -> Image.Image:
    """Render a CHW backbone-input tensor with a fixed shared linear window.

    No per-image / per-channel percentile stretch: every row and both domains
    use the same [vmin, vmax] -> [0, 1] mapping so the tiles are directly
    comparable. Default window is the ImageNet-normalized RGB gamut.
    """
    arr = np.asarray(chw, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"Expected CHW with 3 channels, got {arr.shape}")
    hwc = np.transpose(arr, (1, 2, 0))
    span = max(float(vmax) - float(vmin), 1e-6)
    vis = np.clip((hwc - float(vmin)) / span, 0.0, 1.0)
    return Image.fromarray((vis * 255.0).round().astype(np.uint8))


def run_experiment(
    *,
    spec: ExperimentSpec,
    exp_args: argparse.Namespace,
    checkpoint_name: str,
    bases: list[SampleBase],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[int, RowResult]:
    if is_rgb_exp(exp_args):
        domain = "rgb"
    elif is_raw_exp(exp_args):
        domain = "raw"
    else:
        raise ValueError(f"{spec.exp_dir}: unsupported dataset_family={exp_args.resolved_config.dataset_family!r}")

    checkpoint = resolve_checkpoint(exp_args, checkpoint_name)
    model = build_model(exp_args)
    checkpoint_meta = load_checkpoint_into_model(model, checkpoint)
    model.to(device).eval()
    results: dict[int, RowResult] = {}

    with torch.no_grad():
        for base in bases:
            model_input = input_for_domain(exp_args=exp_args, base=base, domain=domain)
            target_hw = tuple(int(v) for v in base.target.shape)
            prediction = infer_prediction(
                model,
                model_input,
                target_hw,
                device=device,
                amp_dtype=amp_dtype_from_args(exp_args),
                use_amp=use_amp_for(device, exp_args, args.no_amp),
            )
            result = build_prediction_result(prediction, base.target, base.valid_mask, d1_threshold=args.d1_threshold)
            if domain == "rgb":
                rgb_backbone = to_chw_numpy(model_input)
                stats = stat_payload(rgb_backbone)
                input_image = fixed_window_image(rgb_backbone, vmin=args.input_vmin, vmax=args.input_vmax)
                dist_image = x3_hist_image(
                    rgb_backbone,
                    size=(int(args.tile_width), int(args.tile_height)),
                    title="RGB backbone input (imagenet-norm)",
                    labels=("R", "G", "B"),
                )
            else:
                raw_backbone = x3_before_backbone(
                    model,
                    model_input,
                    device=device,
                    amp_dtype=amp_dtype_from_args(exp_args),
                    use_amp=use_amp_for(device, exp_args, args.no_amp),
                )
                stats = stat_payload(raw_backbone)
                input_image = fixed_window_image(raw_backbone, vmin=args.input_vmin, vmax=args.input_vmax)
                dist_image = x3_hist_image(
                    raw_backbone,
                    size=(int(args.tile_width), int(args.tile_height)),
                    title="RAW backbone input (RamCore3 x3)",
                )
            results[base.sample_index] = RowResult(
                spec=spec,
                domain=domain,
                exp_name=Path(exp_args.save_path).name,
                checkpoint=checkpoint,
                checkpoint_meta=checkpoint_meta,
                result=result,
                backbone_stats=stats,
                input_image=input_image,
                dist_image=dist_image,
            )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def run_rgb_init(
    *,
    rgb_args: argparse.Namespace,
    bases: list[SampleBase],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[int, RowResult]:
    spec = ExperimentSpec(
        label="D0",
        display_name="RGB_Dark DAv2-S init",
        exp_dir=Path(str(rgb_args.pretrained_from)).expanduser().resolve(),
    )
    model = build_model(rgb_args)
    load_initial_weights(model, rgb_args.pretrained_from, input_type="rgb")
    model.to(device).eval()
    results: dict[int, RowResult] = {}
    checkpoint_meta = {
        "path": str(rgb_args.pretrained_from),
        "epoch": None,
        "best_metric": None,
        "best_metrics": {},
    }
    checkpoint = Path(str(rgb_args.pretrained_from)).expanduser().resolve()

    with torch.no_grad():
        for base in bases:
            model_input = input_for_domain(exp_args=rgb_args, base=base, domain="rgb")
            target_hw = tuple(int(v) for v in base.target.shape)
            prediction = infer_prediction(
                model,
                model_input,
                target_hw,
                device=device,
                amp_dtype=amp_dtype_from_args(rgb_args),
                use_amp=use_amp_for(device, rgb_args, args.no_amp),
            )
            result = build_prediction_result(prediction, base.target, base.valid_mask, d1_threshold=args.d1_threshold)
            rgb_backbone = to_chw_numpy(model_input)
            results[base.sample_index] = RowResult(
                spec=spec,
                domain="rgb",
                exp_name="DAv2-S_RGB_Dark_init",
                checkpoint=checkpoint,
                checkpoint_meta=checkpoint_meta,
                result=result,
                backbone_stats=stat_payload(rgb_backbone),
                input_image=fixed_window_image(rgb_backbone, vmin=args.input_vmin, vmax=args.input_vmax),
                dist_image=x3_hist_image(
                    rgb_backbone,
                    size=(int(args.tile_width), int(args.tile_height)),
                    title="RGB backbone input (imagenet-norm)",
                    labels=("R", "G", "B"),
                ),
            )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def make_panel(
    *,
    base: SampleBase,
    init_result: RowResult,
    row_results: list[RowResult],
    tile_size: tuple[int, int],
    label_width: int,
    pred_cmap: str,
    error_cmap: str,
    d1_cmap: str,
    error_vmax: float,
    d1_threshold: float,
    pred_vmin: float,
    pred_vmax: float,
    input_vmin: float,
    input_vmax: float,
) -> Image.Image:
    tile_w, tile_h = tile_size
    header_h = 86
    footer_h = 34
    cols = ("Backbone input", "Pseudo / dist", "Pred", "Error", "D1 map")
    row_count = 1 + len(row_results)
    width = label_width + tile_w * len(cols)
    height = header_h + tile_h * row_count + footer_h
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(18)
    header_font = load_font(14)
    label_font = load_font(12)
    small_font = load_font(10)
    draw.text((12, 10), f"LOD true Plan B cross-exp panel: {base.sample_name}", font=title_font, fill=(20, 20, 20))
    draw.text(
        (12, 38),
        (
            f"checkpoint=best_model; backbone input fixed window [{input_vmin:g},{input_vmax:g}] no stretch; "
            f"pred/error/D1 share per-sample scales; error vmax={error_vmax:.3g}"
        ),
        font=small_font,
        fill=(70, 70, 70),
    )
    for col_idx, col_name in enumerate(cols):
        draw.text((label_width + col_idx * tile_w + 8, 62), col_name, font=header_font, fill=(20, 20, 20))

    init = init_result.result
    reference_row = {
        "label_lines": [
            "Reference + D0",
            "RGB_Normal + pseudo",
            "right cols: RGB_Dark DAv2-S init",
            metric_line(init.metrics),
        ],
        "images": [
            rgb_image(base.rgb_normal),
            colorize_array(base.target, base.valid_mask, vmin=pred_vmin, vmax=pred_vmax, cmap_name=pred_cmap),
            colorize_array(init.aligned_pred, base.valid_mask, vmin=pred_vmin, vmax=pred_vmax, cmap_name=pred_cmap),
            colorize_error(init.rel_error, init.error_valid, vmax=error_vmax, cmap_name=error_cmap),
            colorize_d1(init.d1_score, init.d1_valid, cmap_name=d1_cmap),
        ],
    }
    panel_rows = [reference_row]
    for row_result in row_results:
        result = row_result.result
        epoch = checkpoint_epoch(row_result.checkpoint_meta)
        best_lod_d1 = metric_value(row_result.checkpoint_meta, "lod_d1")
        extra = (
            f"ckpt e{epoch} best={best_lod_d1:.4f}"
            if epoch is not None and np.isfinite(best_lod_d1)
            else "ckpt best_model"
        )
        panel_rows.append(
            {
                "label_lines": [
                    row_result.spec.label,
                    row_result.spec.display_name,
                    extra,
                    metric_line(result.metrics),
                ],
                "images": [
                    row_result.input_image,
                    row_result.dist_image,
                    colorize_array(result.aligned_pred, base.valid_mask, vmin=pred_vmin, vmax=pred_vmax, cmap_name=pred_cmap),
                    colorize_error(result.rel_error, result.error_valid, vmax=error_vmax, cmap_name=error_cmap),
                    colorize_d1(result.d1_score, result.d1_valid, cmap_name=d1_cmap),
                ],
            }
        )

    for row_idx, row in enumerate(panel_rows):
        y = header_h + row_idx * tile_h
        draw.rectangle((0, y, label_width - 1, y + tile_h - 1), fill=(235, 235, 235) if row_idx % 2 == 0 else (228, 228, 228))
        text_y = y + 10
        for line in row["label_lines"]:
            for wrapped_line in wrap_label_line(draw, line, label_font, label_width - 24):
                draw.text((12, text_y), wrapped_line, font=label_font, fill=(20, 20, 20))
                text_y += 15
            text_y += 1
        for col_idx, image in enumerate(row["images"]):
            x = label_width + col_idx * tile_w
            canvas.paste(resize_tile(image, tile_size), (x, y))
            draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=(190, 190, 190), width=1)

    draw.text(
        (12, header_h + tile_h * row_count + 9),
        f"Error = abs(aligned pred - pseudo) / pseudo. D1 pass threshold < {d1_threshold:g}.",
        font=small_font,
        fill=(60, 60, 60),
    )
    return canvas


def csv_row(
    *,
    base: SampleBase,
    row_result: RowResult,
    panel_path: Path,
) -> dict[str, object]:
    result = row_result.result
    row = {
        "sample_index": int(base.sample_index),
        "sample_name": base.sample_name,
        "row_label": row_result.spec.label,
        "domain": row_result.domain,
        "experiment": row_result.exp_name,
        "checkpoint": str(row_result.checkpoint),
        "checkpoint_epoch": checkpoint_epoch(row_result.checkpoint_meta),
        "best_lod_d1": metric_value(row_result.checkpoint_meta, "lod_d1"),
        "valid_pixels": int(np.count_nonzero(result.d1_valid)),
        "d1_pass_pixels": int(result.d1_pass_pixels),
        "align_scale": float(result.align_stats.get("scale", float("nan"))),
        "align_shift": float(result.align_stats.get("shift", float("nan"))),
        "panel_path": str(panel_path),
    }
    for key in ("abs_rel", "rmse", "silog", "d1", "d2", "d3"):
        row[key] = float(result.metrics.get(key, float("nan")))
    row.update(stats_to_csv_prefix(row_result.backbone_stats))
    return row


def write_summary_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def text_width(draw: ImageDraw.ImageDraw, text: str, font: object) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def wrap_label_line(draw: ImageDraw.ImageDraw, line: object, font: object, max_width: int) -> list[str]:
    text = str(line)
    if not text:
        return []
    if text_width(draw, text, font) <= max_width:
        return [text]

    tokens = [token for token in re.split(r"(?<=[_-])", text) if token]
    wrapped: list[str] = []
    current = ""
    for token in tokens:
        candidate = f"{current}{token}"
        if not current or text_width(draw, candidate, font) <= max_width:
            current = candidate
            continue
        wrapped.append(current)
        current = token
    if current:
        wrapped.append(current)
    return wrapped


def main() -> None:
    args = parse_args()
    specs = parse_experiment_specs(args.experiment)
    exp_args_by_label = {spec.label: load_exp_args(spec.exp_dir) for spec in specs}

    rgb_specs = [spec for spec in specs if is_rgb_exp(exp_args_by_label[spec.label])]
    raw_specs = [spec for spec in specs if is_raw_exp(exp_args_by_label[spec.label])]
    if not rgb_specs:
        raise ValueError("At least one RGB LOD experiment is required")
    if not raw_specs:
        raise ValueError("At least one RAW LOD experiment is required")

    reference_manifest = str(exp_args_by_label[specs[0].label].lod_manifest)
    for spec in specs:
        current_manifest = str(exp_args_by_label[spec.label].lod_manifest)
        if current_manifest != reference_manifest:
            raise ValueError(f"{spec.label}: lod_manifest mismatch: {current_manifest} != {reference_manifest}")

    rgb_args = exp_args_by_label[rgb_specs[0].label]
    raw_args = exp_args_by_label[raw_specs[0].label]
    rgb_dataset = build_rgb_dataset(rgb_args)
    raw_dataset = build_raw_dataset(raw_args)
    if len(rgb_dataset) != len(raw_dataset):
        raise ValueError(f"RGB/RAW LOD dataset length mismatch: {len(rgb_dataset)} vs {len(raw_dataset)}")
    sample_indices = plan_indices(args, len(rgb_dataset))
    output_root = args.output_root.expanduser().resolve() if args.output_root else default_output_root(args)
    panels_dir = output_root / "sample_panels"
    metadata_dir = output_root / "metadata"

    planned = {
        "checkpoint_name": str(args.checkpoint_name),
        "experiments": [
            {
                "label": spec.label,
                "display_name": spec.display_name,
                "exp_dir": str(spec.exp_dir),
                "dataset_family": str(exp_args_by_label[spec.label].resolved_config.dataset_family),
            }
            for spec in specs
        ],
        "lod_manifest": reference_manifest,
        "sample_indices": sample_indices,
        "sample_selection": str(args.sample_selection) if not parse_indices(args.sample_indices) else "explicit_indices",
        "sample_seed": int(args.sample_seed),
        "output_root": str(output_root),
        "panels_dir": str(panels_dir),
        "metadata_dir": str(metadata_dir),
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

    bases = build_sample_bases(
        sample_indices=sample_indices,
        rgb_args=rgb_args,
        raw_args=raw_args,
        rgb_dataset=rgb_dataset,
        raw_dataset=raw_dataset,
    )

    print("[RUN] D0 RGB_Dark DAv2-S init")
    init_results = run_rgb_init(
        rgb_args=rgb_args,
        bases=bases,
        device=device,
        args=args,
    )
    all_results: dict[int, list[RowResult]] = {base.sample_index: [] for base in bases}
    checkpoint_meta_by_label: dict[str, dict[str, object]] = {
        "D0": {
            "checkpoint": str(rgb_args.pretrained_from),
            "checkpoint_meta": init_results[bases[0].sample_index].checkpoint_meta if bases else {},
        }
    }
    for spec in specs:
        print(f"[RUN] {spec.label} {spec.exp_dir}")
        result_by_sample = run_experiment(
            spec=spec,
            exp_args=exp_args_by_label[spec.label],
            checkpoint_name=args.checkpoint_name,
            bases=bases,
            device=device,
            args=args,
        )
        for base in bases:
            row_result = result_by_sample[base.sample_index]
            all_results[base.sample_index].append(row_result)
            checkpoint_meta_by_label[spec.label] = {
                "checkpoint": str(row_result.checkpoint),
                "checkpoint_meta": row_result.checkpoint_meta,
            }

    tile_size = (int(args.tile_width), int(args.tile_height))
    sample_summaries: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    panel_paths: list[Path] = []
    for base in bases:
        row_results = all_results[base.sample_index]
        pred_vmin, pred_vmax = robust_limits(
            [base.target, init_results[base.sample_index].result.aligned_pred, *[row.result.aligned_pred for row in row_results]],
            base.valid_mask,
        )
        _, error_vmax = robust_limits(
            [init_results[base.sample_index].result.rel_error, *[row.result.rel_error for row in row_results]],
            base.valid_mask,
            low_pct=0.0,
            high_pct=float(args.error_vmax_pct),
        )
        panel = make_panel(
            base=base,
            init_result=init_results[base.sample_index],
            row_results=row_results,
            tile_size=tile_size,
            label_width=int(args.label_width),
            pred_cmap=args.pred_cmap,
            error_cmap=args.error_cmap,
            d1_cmap=args.d1_cmap,
            error_vmax=float(error_vmax),
            d1_threshold=float(args.d1_threshold),
            pred_vmin=float(pred_vmin),
            pred_vmax=float(pred_vmax),
            input_vmin=float(args.input_vmin),
            input_vmax=float(args.input_vmax),
        )
        panel_path = panels_dir / f"lod_true_planb_cross_exp_idx{base.sample_index:04d}_{safe_filename_part(base.sample_name)}.png"
        panel.save(panel_path)
        panel_paths.append(panel_path)

        rows_for_sample = [
            csv_row(base=base, row_result=init_results[base.sample_index], panel_path=panel_path),
            *[csv_row(base=base, row_result=row, panel_path=panel_path) for row in row_results],
        ]
        csv_rows.extend(rows_for_sample)
        sample_payload = {
            "sample_index": int(base.sample_index),
            "sample_name": base.sample_name,
            "panel_path": str(panel_path),
            "rgb_normal_path": str(base.rgb_sample["rgb_normal_path"]),
            "rgb_dark_path": str(base.rgb_sample["rgb_dark_path"]),
            "raw_dark_path": str(base.raw_sample["raw_dark_path"]),
            "pseudo_depth_path": str(base.rgb_sample["pseudo_depth_path"]),
            "pred_vmin": float(pred_vmin),
            "pred_vmax": float(pred_vmax),
            "error_vmax": float(error_vmax),
            "rows": rows_for_sample,
        }
        sample_meta_dir = metadata_dir / f"sample_{base.sample_index:04d}"
        sample_meta_dir.mkdir(parents=True, exist_ok=True)
        write_summary_json(sample_meta_dir / "summary.json", sample_payload)
        write_csv(sample_meta_dir / "metrics_by_row.csv", rows_for_sample)
        sample_summaries.append(sample_payload)

    combined_csv_path = output_root / "metrics_by_sample_row.csv"
    write_csv(combined_csv_path, csv_rows)
    contact_sheet_path = output_root / f"lod_true_planb_cross_exp_{len(sample_indices)}samples_contact_sheet.png"
    make_contact_sheet(panel_paths, contact_sheet_path)
    summary_path = output_root / f"summary_{len(sample_indices)}samples.json"
    write_summary_json(
        summary_path,
        {
            **planned,
            "combined_csv_path": str(combined_csv_path),
            "contact_sheet_path": str(contact_sheet_path),
            "checkpoint_meta_by_label": checkpoint_meta_by_label,
            "sample_summaries": sample_summaries,
        },
    )
    print(f"[OK] output_root={output_root}")
    print(f"[OK] contact_sheet={contact_sheet_path}")
    print(f"[OK] metrics_csv={combined_csv_path}")
    for panel_path in panel_paths:
        print(f"[PANEL] {panel_path}")


if __name__ == "__main__":
    main()
