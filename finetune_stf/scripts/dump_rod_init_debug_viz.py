#!/usr/bin/env python3
"""Dump ROD train panels comparing DAv2-S init and best checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.config import ensure_resolved_config  # noqa: E402
from finetune_stf.dataset.rod_raw_student_rgb import RODRawStudentRGB  # noqa: E402
from finetune_stf.train import (  # noqa: E402
    build_model,
    build_training_criterion,
    load_initial_weights,
    resolve_model_state,
    strip_module_prefix,
)
from finetune_stf.util.metric import compute_inverse_relative_metrics  # noqa: E402
from finetune_stf.util.model_input import select_model_input  # noqa: E402
from finetune_stf.util.viz_dump import (  # noqa: E402
    _color_limits,
    _colorize_array_rgb,
    _compute_sample_viz_loss,
    _ensure_pred_hw,
    _fixed_rgb_preview_from_sample,
    _stable_int_seed,
    _target_and_prediction_views,
    _tensor_2d_np,
)


DEFAULT_EXP = (
    PROJECT_ROOT
    / "finetune_stf/exp/0604_0752_rod_night_studentrgb_dav2s_decoder_e10"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "debug_outputs/0604_0752_init_vs_best_100samples_10panels_debug"
)
CSV_FIELDS = (
    "order",
    "panel_index",
    "split",
    "dataset_index",
    "sample_name",
    "valid_pixels",
    "crop_box",
    "init_abs_rel",
    "init_d1",
    "best_abs_rel",
    "best_d1",
    "init_align_scale",
    "init_align_shift",
    "best_align_scale",
    "best_align_shift",
    "panel_path",
    "pseudo_depth_path",
    "raw_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-train-samples", type=int, default=50)
    parser.add_argument("--num-val-samples", type=int, default=50)
    parser.add_argument("--samples-per-panel", type=int, default=10)
    parser.add_argument(
        "--train-indices",
        default="",
        help="Comma-separated train indices. If set, overrides interval sampling for train.",
    )
    parser.add_argument(
        "--val-indices",
        default="",
        help="Comma-separated val indices. If set, overrides interval sampling for val.",
    )
    parser.add_argument(
        "--index-stride",
        type=int,
        default=0,
        help="Sampling interval for automatic index selection. 0 means len(dataset)//num_samples.",
    )
    parser.add_argument(
        "--only-panel-index",
        type=int,
        default=None,
        help="If set, process only this panel while preserving global sample order and panel numbering.",
    )
    parser.add_argument("--best-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-label", default="Best")
    parser.add_argument("--checkpoint-column-title", default="Best epoch pred")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tile-width", type=int, default=210)
    parser.add_argument("--tile-height", type=int, default=112)
    parser.add_argument("--error-cmap", default="magma")
    return parser.parse_args()


def _load_args_from_config(exp_dir: Path) -> argparse.Namespace:
    with (exp_dir / "config.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    args = argparse.Namespace(**payload)
    ensure_resolved_config(args)
    return args


def _build_rod_dataset(args: argparse.Namespace, *, split: str, mode: str, crop_mode: str) -> RODRawStudentRGB:
    return RODRawStudentRGB(
        rod_root=args.rod_root,
        manifest_path=args.rod_night_manifest,
        split=split,
        size=(int(args.input_height), int(args.input_width)),
        mode=mode,
        raw_source=args.rod_raw_source,
        rgb_pipeline=args.rod_rgb_pipeline,
        label_space=args.rod_label_space,
        crop_mode=crop_mode,
        student_white_percentile=args.rod_student_white_percentile,
        student_gamma=args.rod_student_gamma,
        student_channel_gains=tuple(args.rod_student_channel_gains),
    )


def _build_rod_train_dataset(args: argparse.Namespace) -> RODRawStudentRGB:
    return _build_rod_dataset(
        args,
        split="00Train",
        mode="train",
        crop_mode=args.rod_train_crop_mode,
    )


def _build_rod_val_dataset(args: argparse.Namespace) -> RODRawStudentRGB:
    return _build_rod_dataset(
        args,
        split="01Valid",
        mode="val",
        crop_mode=args.rod_val_crop_mode,
    )


def _parse_indices(
    indices: str,
    num_samples: int,
    dataset_len: int,
    *,
    index_stride: int = 0,
) -> tuple[list[int], dict[str, object]]:
    parsed = []
    for item in str(indices).split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= dataset_len:
            raise ValueError(f"Dataset index {idx} is out of range [0, {dataset_len})")
        parsed.append(idx)
    if parsed:
        return parsed[: int(num_samples)], {
            "mode": "explicit_indices",
            "index_stride": None,
        }

    count = min(int(num_samples), int(dataset_len))
    stride = int(index_stride) if int(index_stride) > 0 else max(int(dataset_len) // max(count, 1), 1)
    selected = [idx for idx in range(0, int(dataset_len), stride)][:count]
    return selected, {
        "mode": "interval",
        "index_stride": stride,
    }


def _float_or_none(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _load_best_checkpoint(model: torch.nn.Module, path: Path) -> dict[str, object]:
    ckpt_obj = torch.load(path, map_location="cpu")
    state_dict = strip_module_prefix(resolve_model_state(ckpt_obj))
    model.load_state_dict(state_dict, strict=True)
    meta = {
        "path": str(path),
        "epoch": ckpt_obj.get("epoch") if isinstance(ckpt_obj, dict) else None,
        "best_metric": ckpt_obj.get("best_metric") if isinstance(ckpt_obj, dict) else None,
        "best_metrics": ckpt_obj.get("best_metrics") if isinstance(ckpt_obj, dict) else None,
    }
    return meta


def _colorize_error_rgb(error: np.ndarray, lo: float, hi: float, cmap_name: str) -> np.ndarray:
    error = np.asarray(error, dtype=np.float32)
    finite = np.isfinite(error)
    norm = np.zeros(error.shape, dtype=np.uint8)
    if hi > lo:
        clipped = np.clip(error, lo, hi)
        clipped = np.where(finite, clipped, lo)
        norm = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgb = (cmap(norm)[:, :, :3] * 255.0).astype(np.uint8)
    rgb[~finite] = (32, 32, 32)
    return rgb


def _resize_tile(image_rgb: np.ndarray, tile_hw: tuple[int, int]) -> np.ndarray:
    tile_h, tile_w = tile_hw
    image = np.asarray(image_rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).round().astype(np.uint8)
    return cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)


def _draw_text(canvas: np.ndarray, text: str, xy: tuple[int, int], *, scale=0.42, color=(235, 235, 235)) -> None:
    cv2.putText(canvas, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _make_panel_grid(
    records: list[dict[str, object]],
    *,
    panel_index: int,
    tile_hw: tuple[int, int],
    error_cmap: str,
    checkpoint_label: str,
    checkpoint_column_title: str,
) -> np.ndarray:
    tile_h, tile_w = tile_hw
    label_w = 215
    header_h = 58
    columns = (
        ("Input RGB", ""),
        ("Pseudo label", ""),
        ("Direct pred", ""),
        ("Init error", "abs(aligned-pseudo)"),
        (checkpoint_column_title, ""),
        (f"{checkpoint_label} error", "abs(aligned-pseudo)"),
    )
    width = label_w + tile_w * len(columns)
    height = header_h + tile_h * len(records)
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:, :] = (18, 18, 18)

    _draw_text(panel, f"panel {panel_index:02d}", (10, 26), scale=0.50)
    for col_idx, (title, subtitle) in enumerate(columns):
        x = label_w + col_idx * tile_w + 8
        _draw_text(panel, title, (x, 26), scale=0.48)
        if subtitle:
            _draw_text(panel, subtitle, (x, 48), scale=0.33, color=(205, 205, 205))

    for row_idx, record in enumerate(records):
        y = header_h + row_idx * tile_h
        row_bg = (24, 24, 24) if row_idx % 2 == 0 else (30, 30, 30)
        panel[y : y + tile_h, :label_w] = row_bg
        _draw_text(
            panel,
            f"{str(record['split'])} {int(record['order']):03d} idx={int(record['dataset_index'])}",
            (8, y + 20),
            scale=0.40,
        )
        _draw_text(panel, str(record["sample_name"])[:24], (8, y + 44), scale=0.40, color=(220, 220, 220))
        _draw_text(
            panel,
            f"init absrel {float(record['init_abs_rel']):.3g} d1 {float(record['init_d1']):.3g}",
            (8, y + 67),
            scale=0.31,
            color=(205, 205, 205),
        )
        _draw_text(
            panel,
            f"{checkpoint_label.lower()} absrel {float(record['best_abs_rel']):.3g} d1 {float(record['best_d1']):.3g}",
            (8, y + 89),
            scale=0.31,
            color=(205, 205, 205),
        )

        pseudo_view = record["pseudo_view"]
        init_direct = record["init_direct_view"]
        best_direct = record["best_direct_view"]
        init_error = record["init_error_view"]
        best_error = record["best_error_view"]

        pseudo_lo, pseudo_hi = _color_limits([pseudo_view])
        direct_lo, direct_hi = _color_limits([init_direct, best_direct])
        error_lo, error_hi = _color_limits([init_error, best_error])

        images = (
            _resize_tile(record["rgb_preview"], tile_hw),
            _resize_tile(_colorize_array_rgb(pseudo_view, pseudo_lo, pseudo_hi, cmap_name="Spectral_r"), tile_hw),
            _resize_tile(_colorize_array_rgb(init_direct, direct_lo, direct_hi, cmap_name="Spectral_r"), tile_hw),
            _resize_tile(_colorize_error_rgb(init_error, error_lo, error_hi, error_cmap), tile_hw),
            _resize_tile(_colorize_array_rgb(best_direct, direct_lo, direct_hi, cmap_name="Spectral_r"), tile_hw),
            _resize_tile(_colorize_error_rgb(best_error, error_lo, error_hi, error_cmap), tile_hw),
        )
        for col_idx, image in enumerate(images):
            x = label_w + col_idx * tile_w
            panel[y : y + tile_h, x : x + tile_w] = image
    return panel


def _run_one_model(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    args: argparse.Namespace,
    target_space: str,
    *,
    criterion: torch.nn.Module,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> dict[str, object]:
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        pred = model(model_input)
    pred = _ensure_pred_hw(pred.float(), depth.shape[-2:])
    loss, loss_info = criterion(pred.float(), depth, valid_mask, target_space=target_space)
    metrics, aligned, _, effective_mask = _compute_sample_viz_loss(
        pred,
        depth,
        valid_mask,
        args,
        target_space,
    )
    _, aligned_view = _target_and_prediction_views(
        target_space,
        depth[0],
        valid_mask[0],
        aligned[0],
    )
    direct_view = _tensor_2d_np(pred[0]).astype(np.float32, copy=False)
    target_view = _tensor_2d_np(depth[0], valid_mask[0])
    valid_np = valid_mask[0].detach().cpu().numpy().astype(bool)
    eval_metrics = compute_inverse_relative_metrics(aligned_view, target_view, valid_np)
    if eval_metrics is None:
        eval_metrics = {"abs_rel": float("nan"), "d1": float("nan")}
    return {
        "direct_view": direct_view,
        "aligned_view": aligned_view,
        "loss": float(loss.detach().item()),
        "loss_info": loss_info,
        "metrics": metrics,
        "eval_metrics": eval_metrics,
        "effective_mask": effective_mask,
    }


def main() -> None:
    args_cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this debug dump.")

    exp_args = _load_args_from_config(args_cli.exp_dir)
    if exp_args.resolved_config.dataset_family != "rod_raw_student_rgb":
        raise ValueError(f"Expected ROD experiment, got {exp_args.resolved_config.dataset_family!r}")
    if exp_args.resolved_config.model_input_tensor != "image":
        raise ValueError(f"Expected RGB image input, got {exp_args.resolved_config.model_input_tensor!r}")

    output_root = args_cli.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    panels_dir = output_root / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint = args_cli.best_checkpoint
    if best_checkpoint is None:
        best_checkpoint = Path(exp_args.heavy_save_path) / "best_model.pth"
    best_checkpoint = best_checkpoint.expanduser().resolve()
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Missing best checkpoint: {best_checkpoint}")

    device = torch.device(args_cli.device)
    torch.cuda.set_device(device)
    train_dataset = _build_rod_train_dataset(exp_args)
    val_dataset = _build_rod_val_dataset(exp_args)
    train_indices, train_selection = _parse_indices(
        args_cli.train_indices,
        args_cli.num_train_samples,
        len(train_dataset),
        index_stride=args_cli.index_stride,
    )
    val_indices, val_selection = _parse_indices(
        args_cli.val_indices,
        args_cli.num_val_samples,
        len(val_dataset),
        index_stride=args_cli.index_stride,
    )
    sample_specs_all = (
        [("train", train_dataset, idx) for idx in train_indices]
        + [("val", val_dataset, idx) for idx in val_indices]
    )
    ordered_sample_specs = list(enumerate(sample_specs_all))
    if args_cli.only_panel_index is not None:
        if int(args_cli.only_panel_index) < 0:
            raise ValueError("--only-panel-index must be non-negative")
        panel_start = int(args_cli.only_panel_index) * int(args_cli.samples_per_panel)
        panel_stop = panel_start + int(args_cli.samples_per_panel)
        ordered_sample_specs = ordered_sample_specs[panel_start:panel_stop]
        if not ordered_sample_specs:
            raise ValueError(
                f"Panel {args_cli.only_panel_index} is empty for {len(sample_specs_all)} selected samples"
            )
    selected_panel_indices = sorted(
        {order // int(args_cli.samples_per_panel) for order, _ in ordered_sample_specs}
    )

    init_model = build_model(exp_args)
    load_initial_weights(init_model, exp_args.pretrained_from, input_type=exp_args.input_type)
    init_model.to(device).eval()

    best_model = build_model(exp_args)
    best_meta = _load_best_checkpoint(best_model, best_checkpoint)
    best_model.to(device).eval()

    criterion = build_training_criterion(exp_args, device.index or 0)
    criterion.eval()
    amp_dtype = torch.float16 if getattr(exp_args, "amp_dtype", "bf16") == "fp16" else torch.bfloat16
    use_amp = bool(getattr(exp_args, "amp", False)) and not args_cli.no_amp

    summary = {
        "experiment": str(args_cli.exp_dir),
        "init_checkpoint_source": str(exp_args.pretrained_from),
        "best_checkpoint": best_meta,
        "output_root": str(output_root),
        "dataset_len": {"train": len(train_dataset), "val": len(val_dataset)},
        "num_samples": len(ordered_sample_specs),
        "source_num_samples": len(sample_specs_all),
        "split_counts": {"train": len(train_indices), "val": len(val_indices)},
        "samples_per_panel": int(args_cli.samples_per_panel),
        "panel_count": len(selected_panel_indices),
        "selected_panel_indices": selected_panel_indices,
        "checkpoint_label": str(args_cli.checkpoint_label),
        "selection": {
            "train": train_selection,
            "val": val_selection,
        },
        "indices": {
            "train": train_indices,
            "val": val_indices,
        },
        "prediction_colormap": "Spectral_r",
        "error_colormap": str(args_cli.error_cmap),
        "metric_definition": "absrel/d1 from compute_inverse_relative_metrics(aligned prediction, pseudo label, valid mask)",
        "error_definition": "abs(aligned inverse-relative prediction - pseudo inverse-relative label)",
        "samples": [],
        "panels": [],
    }
    csv_rows = []
    panel_records: list[dict[str, object]] = []

    with torch.no_grad():
        for order, (split_label, dataset, idx) in ordered_sample_specs:
            sample_seed = _stable_int_seed(exp_args.seed, "rod", split_label, idx, order, "init_best_debug")
            sample = dataset.build_sample(idx, rng=random.Random(sample_seed), include_geometry=True)
            target_space = str(sample.get("target_space", exp_args.rod_label_space))
            if target_space != "inverse_relative":
                raise ValueError(f"Expected inverse_relative target, got {target_space!r}")

            model_input = select_model_input(
                sample,
                exp_args.resolved_config.model_input_tensor,
                dataset_family=exp_args.resolved_config.dataset_family,
                sample_source="rod",
                add_batch_dim=True,
            ).to(device=device, non_blocking=True).float()
            depth = sample["depth"]
            valid_mask = sample["valid_mask"]
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            if valid_mask.ndim == 2:
                valid_mask = valid_mask.unsqueeze(0)
            depth = depth.to(device=device, non_blocking=True).float()
            valid_mask = valid_mask.to(device=device, non_blocking=True).bool()

            init_result = _run_one_model(
                init_model,
                model_input,
                depth,
                valid_mask,
                exp_args,
                target_space,
                criterion=criterion,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
            )
            best_result = _run_one_model(
                best_model,
                model_input,
                depth,
                valid_mask,
                exp_args,
                target_space,
                criterion=criterion,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
            )

            target_view, _ = _target_and_prediction_views(target_space, depth[0], valid_mask[0], depth[0])
            if target_view is None:
                raise RuntimeError(f"Failed to build pseudo-label view for dataset index {idx}")
            rgb_preview = _fixed_rgb_preview_from_sample(sample, "image")
            if rgb_preview is None:
                raise RuntimeError(f"Failed to recover RGB preview for dataset index {idx}")

            valid_np = valid_mask[0].detach().cpu().numpy().astype(bool)
            init_aligned = init_result["aligned_view"]
            best_aligned = best_result["aligned_view"]
            init_error = np.full_like(target_view, np.nan, dtype=np.float32)
            best_error = np.full_like(target_view, np.nan, dtype=np.float32)
            init_valid = valid_np & np.isfinite(target_view) & np.isfinite(init_aligned)
            best_valid = valid_np & np.isfinite(target_view) & np.isfinite(best_aligned)
            init_error[init_valid] = np.abs(init_aligned[init_valid] - target_view[init_valid])
            best_error[best_valid] = np.abs(best_aligned[best_valid] - target_view[best_valid])

            sample_name = str(sample.get("sample_name", f"sample_{idx:06d}"))
            panel_index = order // int(args_cli.samples_per_panel)
            crop_box = sample.get("geometry_params", {}).get("crop_box", "")
            record = {
                "order": order,
                "panel_index": panel_index,
                "split": split_label,
                "dataset_index": idx,
                "sample_name": sample_name,
                "rgb_preview": rgb_preview.astype(np.float32, copy=False),
                "pseudo_view": target_view.astype(np.float32, copy=False),
                "init_direct_view": init_result["direct_view"].astype(np.float32, copy=False),
                "init_aligned_view": init_aligned.astype(np.float32, copy=False),
                "init_error_view": init_error,
                "best_direct_view": best_result["direct_view"].astype(np.float32, copy=False),
                "best_aligned_view": best_aligned.astype(np.float32, copy=False),
                "best_error_view": best_error,
                "init_abs_rel": float(init_result["eval_metrics"]["abs_rel"]),
                "init_d1": float(init_result["eval_metrics"]["d1"]),
                "best_abs_rel": float(best_result["eval_metrics"]["abs_rel"]),
                "best_d1": float(best_result["eval_metrics"]["d1"]),
                "init_loss": float(init_result["metrics"]["loss_total"]),
                "best_loss": float(best_result["metrics"]["loss_total"]),
                "init_align_scale": float(init_result["metrics"]["align_scale"]),
                "init_align_shift": float(init_result["metrics"]["align_shift"]),
                "best_align_scale": float(best_result["metrics"]["align_scale"]),
                "best_align_shift": float(best_result["metrics"]["align_shift"]),
                "valid_pixels": int(valid_np.sum()),
                "crop_box": crop_box,
                "raw_path": str(sample.get("raw_path", "")),
                "pseudo_depth_path": str(sample.get("pseudo_depth_path", "")),
            }
            panel_records.append(record)

            csv_rows.append(
                {
                    "order": order,
                    "panel_index": panel_index,
                    "split": split_label,
                    "dataset_index": idx,
                    "sample_name": sample_name,
                    "valid_pixels": int(valid_np.sum()),
                    "crop_box": json.dumps(crop_box),
                    "init_abs_rel": f"{record['init_abs_rel']:.9g}",
                    "init_d1": f"{record['init_d1']:.9g}",
                    "best_abs_rel": f"{record['best_abs_rel']:.9g}",
                    "best_d1": f"{record['best_d1']:.9g}",
                    "init_align_scale": f"{record['init_align_scale']:.9g}",
                    "init_align_shift": f"{record['init_align_shift']:.9g}",
                    "best_align_scale": f"{record['best_align_scale']:.9g}",
                    "best_align_shift": f"{record['best_align_shift']:.9g}",
                    "panel_path": "",
                    "pseudo_depth_path": record["pseudo_depth_path"],
                    "raw_path": record["raw_path"],
                }
            )
            summary["samples"].append(
                {
                    "order": order,
                    "panel_index": panel_index,
                    "split": split_label,
                    "dataset_index": idx,
                    "sample_name": sample_name,
                    "sample_seed": int(sample_seed),
                    "valid_pixels": int(valid_np.sum()),
                    "crop_box": crop_box,
                    "raw_path": record["raw_path"],
                    "pseudo_depth_path": record["pseudo_depth_path"],
                    "init_abs_rel": _float_or_none(record["init_abs_rel"]),
                    "init_d1": _float_or_none(record["init_d1"]),
                    "best_abs_rel": _float_or_none(record["best_abs_rel"]),
                    "best_d1": _float_or_none(record["best_d1"]),
                    "init_loss": _float_or_none(record["init_loss"]),
                    "best_loss": _float_or_none(record["best_loss"]),
                    "init_align_scale": _float_or_none(record["init_align_scale"]),
                    "init_align_shift": _float_or_none(record["init_align_shift"]),
                    "best_align_scale": _float_or_none(record["best_align_scale"]),
                    "best_align_shift": _float_or_none(record["best_align_shift"]),
                    "init_loss_info": init_result["loss_info"],
                    "best_loss_info": best_result["loss_info"],
                }
            )

    csv_rows_by_panel = {int(row["panel_index"]): [] for row in csv_rows}
    for row in csv_rows:
        csv_rows_by_panel[int(row["panel_index"])].append(row)

    for panel_index in selected_panel_indices:
        records = [record for record in panel_records if int(record["panel_index"]) == int(panel_index)]
        if not records:
            continue
        start = int(records[0]["order"])
        stop = int(records[-1]["order"])
        panel = _make_panel_grid(
            records,
            panel_index=panel_index,
            tile_hw=(int(args_cli.tile_height), int(args_cli.tile_width)),
            error_cmap=str(args_cli.error_cmap),
            checkpoint_label=str(args_cli.checkpoint_label),
            checkpoint_column_title=str(args_cli.checkpoint_column_title),
        )
        panel_path = panels_dir / f"panel_{panel_index:02d}_samples_{start:03d}_{stop:03d}.jpg"
        cv2.imwrite(str(panel_path), panel[:, :, ::-1])
        summary["panels"].append(str(panel_path))
        for row in csv_rows_by_panel.get(int(panel_index), []):
            row["panel_path"] = str(panel_path)

    summary_path = output_root / "summary.json"
    csv_path = output_root / "metrics_by_sample.csv"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"[OK] wrote {len(ordered_sample_specs)} samples in {len(summary['panels'])} panels to {output_root}")
    print(f"[OK] summary={summary_path}")
    print(f"[OK] csv={csv_path}")
    for panel_path in summary["panels"]:
        print(f"[PANEL] {panel_path}")


if __name__ == "__main__":
    main()
