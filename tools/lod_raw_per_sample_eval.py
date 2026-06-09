#!/usr/bin/env python3
"""Per-sample LOD RAW checkpoint evaluation for attribution analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.config.resolved import ensure_resolved_config  # noqa: E402
from finetune_stf.dataset.lod_true import LODTrueRawDarkRGB16  # noqa: E402
from finetune_stf.scripts.make_rod_raw_four_exp_panel import (  # noqa: E402
    build_prediction_result,
    load_json,
)
from finetune_stf.train import (  # noqa: E402
    build_model,
    parse_args as train_parse_args,
    resolve_model_state,
    strip_module_prefix,
)
from finetune_stf.util.model_input import select_model_input  # noqa: E402


SEMANTIC_SCALAR_FIELDS = (
    "dataset_family",
    "dataset_input_mode",
    "input_domain",
    "front_end",
    "model_input_tensor",
    "bridge",
    "decoder_feature_adapter",
    "lora",
    "raw_storage_format",
)
SEMANTIC_SEQUENCE_FIELDS = ("lora_tap_layers",)
SEMANTIC_NUMERIC_FIELDS = ("lora_rank", "lora_alpha")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--dataset-family", required=True)
    parser.add_argument("--dataset-input-mode", required=True)
    parser.add_argument("--input-domain", required=True)
    parser.add_argument("--front-end", required=True)
    parser.add_argument("--model-input-tensor", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--decoder-feature-adapter", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--lora-block-mode", default="tap")
    parser.add_argument("--lora-tap-layers", nargs="+", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--raw-front-end-lr", type=float, default=None)
    parser.add_argument("--lora-lr", type=float, default=None)
    parser.add_argument("--raw-storage-format", default="raw_rgb16_png_3ch")
    parser.add_argument("--lod-raw-norm-mode", default="uint16_div_65535")
    parser.add_argument("--raw-ram-rgb-tail", default="identity")
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--pretrained-from", default=None)
    parser.add_argument("--dav2-train-mode", default=None)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default=None)
    parser.add_argument("--lod-root", type=Path, required=True)
    parser.add_argument("--lod-manifest", type=Path, required=True)
    parser.add_argument("--lod-label-space", default="inverse_relative")
    parser.add_argument("--split", default="01Valid")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-indices", default="")
    parser.add_argument("--d1-threshold", type=float, default=1.25)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def find_run_file(run_dir: Path, filename: str) -> Path | None:
    run_dir = run_dir.expanduser().resolve()
    candidates = [
        run_dir / filename,
        PROJECT_ROOT / "finetune_stf" / "exp" / run_dir.name / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_run_payloads(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    resolved_path = find_run_file(run_dir, "resolved_config.json")
    config_path = find_run_file(run_dir, "config.json")
    resolved_payload: dict[str, Any] = {}
    config_payload: dict[str, Any] = {}
    if resolved_path is not None:
        resolved_payload = load_json(resolved_path)
    if config_path is not None:
        config_payload = load_json(config_path)
        if not resolved_payload and isinstance(config_payload.get("resolved_config"), dict):
            resolved_payload = dict(config_payload["resolved_config"])
    if not resolved_payload:
        raise FileNotFoundError(
            f"Could not find resolved_config.json for run_dir={run_dir}. "
            f"Checked {run_dir / 'resolved_config.json'} and "
            f"{PROJECT_ROOT / 'finetune_stf' / 'exp' / run_dir.name / 'resolved_config.json'}"
        )
    paths = {
        "resolved_config": str(resolved_path) if resolved_path is not None else "",
        "config": str(config_path) if config_path is not None else "",
    }
    return resolved_payload, config_payload, paths


def value_from_args_or_config(args: argparse.Namespace, config: dict[str, Any], key: str, default: Any) -> Any:
    value = getattr(args, key, None)
    if value is not None:
        return value
    return config.get(key, default)


def validate_semantics(args: argparse.Namespace, resolved_payload: dict[str, Any]) -> None:
    mismatches: list[str] = []
    for field in SEMANTIC_SCALAR_FIELDS:
        cli_value = str(getattr(args, field))
        run_value = resolved_payload.get(field)
        if run_value == "not_applicable":
            run_value = "none"
        if str(run_value) != cli_value:
            mismatches.append(f"{field}: cli={cli_value!r} run={run_value!r}")
    for field in SEMANTIC_SEQUENCE_FIELDS:
        cli_value = tuple(int(v) for v in (getattr(args, field) or ()))
        run_raw = resolved_payload.get(field)
        run_value = tuple(int(v) for v in run_raw) if isinstance(run_raw, list) else ()
        if cli_value != run_value:
            mismatches.append(f"{field}: cli={list(cli_value)!r} run={list(run_value)!r}")
    for field in SEMANTIC_NUMERIC_FIELDS:
        cli_value = float(getattr(args, field))
        run_raw = resolved_payload.get(field)
        if run_raw == "not_applicable":
            run_raw = float("nan")
        run_value = float(run_raw)
        if not math.isclose(cli_value, run_value, rel_tol=1e-8, abs_tol=1e-8):
            mismatches.append(f"{field}: cli={cli_value!r} run={run_value!r}")
    if mismatches:
        raise ValueError(
            "CLI semantic fields do not match the checkpoint run resolved config:\n"
            + "\n".join(f"  - {item}" for item in mismatches)
        )


def train_args_from_eval_args(args: argparse.Namespace, config_payload: dict[str, Any]) -> argparse.Namespace:
    encoder = str(value_from_args_or_config(args, config_payload, "encoder", "vits"))
    pretrained = str(
        value_from_args_or_config(
            args,
            config_payload,
            "pretrained_from",
            "/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth",
        )
    )
    dav2_train_mode = str(value_from_args_or_config(args, config_payload, "dav2_train_mode", "decoder"))
    input_height = int(value_from_args_or_config(args, config_payload, "input_height", 512))
    input_width = int(value_from_args_or_config(args, config_payload, "input_width", 960))
    amp = bool(value_from_args_or_config(args, config_payload, "amp", True))
    amp_dtype = str(value_from_args_or_config(args, config_payload, "amp_dtype", "bf16"))
    raw_front_end_lr = float(value_from_args_or_config(args, config_payload, "raw_front_end_lr", 5e-5))
    lora_lr = float(value_from_args_or_config(args, config_payload, "lora_lr", 5e-5))
    save_path = PROJECT_ROOT / "finetune_stf" / "exp" / "lod_raw_diag" / "per_sample_eval" / "_build_args"
    argv = [
        "finetune_stf/train.py",
        "--stage",
        "eval_only",
        "--encoder",
        encoder,
        "--dataset-family",
        str(args.dataset_family),
        "--dataset-input-mode",
        str(args.dataset_input_mode),
        "--input-domain",
        str(args.input_domain),
        "--front-end",
        str(args.front_end),
        "--model-input-tensor",
        str(args.model_input_tensor),
        "--bridge",
        str(args.bridge),
        "--decoder-feature-adapter",
        str(args.decoder_feature_adapter),
        "--lora",
        str(args.lora),
        "--lora-block-mode",
        str(args.lora_block_mode),
        "--lora-rank",
        str(int(args.lora_rank)),
        "--lora-alpha",
        str(float(args.lora_alpha)),
        "--raw-front-end-lr",
        str(raw_front_end_lr),
        "--lora-lr",
        str(lora_lr),
        "--pretrained-from",
        pretrained,
        "--lod-root",
        str(args.lod_root.expanduser().resolve()),
        "--lod-manifest",
        str(args.lod_manifest.expanduser().resolve()),
        "--lod-label-space",
        str(args.lod_label_space),
        "--lod-val-crop-mode",
        "center",
        "--raw-storage-format",
        str(args.raw_storage_format),
        "--lod-raw-norm-mode",
        str(args.lod_raw_norm_mode),
        "--raw-ram-rgb-tail",
        str(args.raw_ram_rgb_tail),
        "--dav2-train-mode",
        dav2_train_mode,
        "--input-height",
        str(input_height),
        "--input-width",
        str(input_width),
        "--save-path",
        str(save_path),
        "--heavy-save-root",
        "/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/per_sample_eval_build_args",
        "--eval-lod",
        "--no-eval-stf",
    ]
    if args.lora_tap_layers:
        argv.extend(["--lora-tap-layers", *[str(int(v)) for v in args.lora_tap_layers]])
    argv.extend(["--amp" if amp else "--no-amp", "--amp-dtype", amp_dtype])
    old_argv = sys.argv
    try:
        sys.argv = argv
        train_args = train_parse_args()
    finally:
        sys.argv = old_argv
    ensure_resolved_config(train_args)
    return train_args


def parse_indices(value: str) -> list[int]:
    indices = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            indices.append(int(item))
    return indices


def planned_indices(args: argparse.Namespace, dataset_len: int) -> list[int]:
    indices = parse_indices(args.sample_indices)
    if not indices:
        max_samples = args.max_samples
        if max_samples is None or int(max_samples) < 0:
            indices = list(range(dataset_len))
        else:
            indices = list(range(min(int(max_samples), dataset_len)))
    bad = [idx for idx in indices if idx < 0 or idx >= dataset_len]
    if bad:
        raise ValueError(f"Sample indices out of range [0, {dataset_len}): {bad}")
    return indices


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint: Path) -> dict[str, Any]:
    ckpt_obj = torch.load(checkpoint, map_location="cpu")
    state_dict = strip_module_prefix(resolve_model_state(ckpt_obj))
    model.load_state_dict(state_dict, strict=True)
    return {
        "path": str(checkpoint),
        "epoch": ckpt_obj.get("epoch") if isinstance(ckpt_obj, dict) else None,
        "best_metric": ckpt_obj.get("best_metric") if isinstance(ckpt_obj, dict) else None,
        "best_metrics": ckpt_obj.get("best_metrics") if isinstance(ckpt_obj, dict) else None,
    }


def amp_dtype_from_args(train_args: argparse.Namespace) -> torch.dtype:
    return torch.float16 if str(getattr(train_args, "amp_dtype", "bf16")) == "fp16" else torch.bfloat16


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
    return pred[0].detach().float().cpu().numpy().astype(np.float32, copy=False)


def robust_normalize(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)
    if int(np.count_nonzero(valid)) < 10:
        return np.zeros_like(arr, dtype=np.float32)
    p1, p99 = np.percentile(arr[valid], (1.0, 99.0))
    return np.clip((arr - p1) / max(float(p99 - p1), 1e-8), 0.0, 1.0).astype(np.float32)


def sobel_magnitude(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def edge_metrics(aligned_pred: np.ndarray, target: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(aligned_pred) & np.isfinite(target) & (aligned_pred > 0) & (target > 0)
    if int(np.count_nonzero(valid)) < 10:
        return {"edge_sobel_l1": float("nan"), "edge_overlap_iou": float("nan")}
    pred_norm = robust_normalize(aligned_pred, valid)
    target_norm = robust_normalize(target, valid)
    pred_edge = sobel_magnitude(pred_norm)
    target_edge = sobel_magnitude(target_norm)
    edge_sobel_l1 = float(np.mean(np.abs(pred_edge[valid] - target_edge[valid])))
    pred_thr = float(np.percentile(pred_edge[valid], 90.0))
    target_thr = float(np.percentile(target_edge[valid], 90.0))
    pred_mask = (pred_edge > pred_thr) & valid
    target_mask = (target_edge > target_thr) & valid
    inter = int(np.count_nonzero(pred_mask & target_mask))
    union = int(np.count_nonzero(pred_mask | target_mask))
    return {
        "edge_sobel_l1": edge_sobel_l1,
        "edge_overlap_iou": float(inter / max(union, 1)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    resolved_payload, config_payload, run_config_paths = load_run_payloads(run_dir)
    validate_semantics(args, resolved_payload)
    train_args = train_args_from_eval_args(args, config_payload)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dataset = LODTrueRawDarkRGB16(
        lod_root=train_args.lod_root,
        manifest_path=train_args.lod_manifest,
        split=str(args.split),
        size=(int(train_args.input_height), int(train_args.input_width)),
        mode="val",
        label_space=train_args.lod_label_space,
        crop_mode=train_args.lod_val_crop_mode,
        raw_storage_format=train_args.raw_storage_format,
        lod_raw_norm_mode=train_args.lod_raw_norm_mode,
        raw_input_mode=str(args.dataset_input_mode),
    )
    indices = planned_indices(args, len(dataset))
    model = build_model(train_args)
    ckpt_meta = load_checkpoint_into_model(model, checkpoint)
    model.to(device)
    model.eval()
    use_amp = device.type == "cuda" and bool(getattr(train_args, "amp", False))
    amp_dtype = amp_dtype_from_args(train_args)
    label = args.label or checkpoint.parent.name
    rows: list[dict[str, Any]] = []
    print(
        f"[PER_SAMPLE] label={label} split={args.split} samples={len(indices)}/{len(dataset)} "
        f"device={device} checkpoint={checkpoint}",
        flush=True,
    )
    for ordinal, idx in enumerate(indices):
        sample = dataset.build_sample(int(idx), include_geometry=True)
        target = sample["depth"].detach().float().cpu().numpy().astype(np.float32, copy=False)
        valid_mask = sample["valid_mask"].detach().cpu().numpy().astype(bool)
        valid_mask = valid_mask & np.isfinite(target) & (target > 0)
        model_input = select_model_input(
            sample,
            train_args.resolved_config.model_input_tensor,
            dataset_family=train_args.resolved_config.dataset_family,
            sample_source=f"lod_{args.split}",
            add_batch_dim=True,
        )
        pred = infer_prediction(
            model,
            model_input,
            tuple(int(v) for v in target.shape),
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
        )
        result = build_prediction_result(pred, target, valid_mask, d1_threshold=float(args.d1_threshold))
        edges = edge_metrics(result.aligned_pred, target, valid_mask)
        row: dict[str, Any] = {
            "split": str(args.split),
            "checkpoint_label": str(label),
            "sample_index": int(idx),
            "sample_id": str(sample["sample_id"]),
            "pair_id": str(sample["sample_id"]),
            "normal_id": int(sample["normal_id"]),
            "dark_id": int(sample["dark_id"]),
            "student_input_path": str(sample["student_input_path"]),
            "dataset_family": str(args.dataset_family),
            "dataset_input_mode": str(args.dataset_input_mode),
            "input_domain": str(args.input_domain),
            "model_input_tensor": str(args.model_input_tensor),
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": ckpt_meta.get("epoch"),
            "checkpoint_best_metric": ckpt_meta.get("best_metric"),
            "valid_pixels": int(np.count_nonzero(result.d1_valid)),
            "d1_pass_pixels": int(result.d1_pass_pixels),
            "align_scale": float(result.align_stats.get("scale", float("nan"))),
            "align_shift": float(result.align_stats.get("shift", float("nan"))),
        }
        for key in ("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog", "silog_x100", "d1", "d2", "d3"):
            row[key] = float(result.metrics.get(key, float("nan")))
        row.update(edges)
        rows.append(row)
        if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(indices):
            print(f"[PER_SAMPLE] processed={ordinal + 1}/{len(indices)}", flush=True)
    write_csv(args.output_csv.expanduser().resolve(), rows)
    summary = {
        "label": str(label),
        "rows": len(rows),
        "dataset_size": len(dataset),
        "checkpoint": str(checkpoint),
        "checkpoint_meta": ckpt_meta,
        "run_config_paths": run_config_paths,
        "mean_metrics": {key: mean_metric(rows, key) for key in ("abs_rel", "rmse", "silog", "d1", "d2", "d3", "edge_sobel_l1", "edge_overlap_iou")},
    }
    summary_path = args.output_csv.expanduser().resolve().with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(
        "[PER_SAMPLE] wrote {csv} rows={rows} mean_d1={d1:.6f} mean_abs_rel={abs_rel:.6f}".format(
            csv=args.output_csv.expanduser().resolve(),
            rows=len(rows),
            d1=summary["mean_metrics"]["d1"],
            abs_rel=summary["mean_metrics"]["abs_rel"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
