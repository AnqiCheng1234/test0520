#!/usr/bin/env python3
"""Dump post-RAM cleanup artifacts for Stage-B visualization."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.models.raw_ram import phase1b_tanh_tail_squash  # noqa: E402
from finetune_stf.train import METRIC_KEYS, build_model, set_random_seed  # noqa: E402
from finetune_stf.util.metric import affine_align_to_inverse_target, compute_inverse_relative_metrics  # noqa: E402
from finetune_stf.util.model_input import select_model_input  # noqa: E402
from tools.lod_postram_external_eval import save_artifact  # noqa: E402
from tools.lod_raw_cross_eval import (  # noqa: E402
    INPUT_DARK,
    collated_item,
    get_config,
    load_checkpoint_into_model,
    make_dataset,
    make_loader,
    metric_average,
    train_args_from_run_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-split", default="01Valid")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-eval-samples", type=int, default=16)
    parser.add_argument("--artifact-max-samples", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default=None)
    parser.add_argument("--lod-root", type=Path, default=None)
    parser.add_argument("--lod-manifest", type=Path, default=None)
    parser.add_argument("--pretrained-from", type=Path, default=None)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def amp_dtype_from_train_args(train_args: argparse.Namespace) -> torch.dtype:
    return torch.float16 if str(getattr(train_args, "amp_dtype", "bf16")) == "fp16" else torch.bfloat16


def tensor_percentiles(x: torch.Tensor, qs=(0.50, 0.95, 0.99)) -> list[float]:
    flat = x.detach().float().reshape(-1)
    if flat.numel() == 0:
        return [float("nan") for _ in qs]
    if flat.numel() > 1_000_000:
        flat = flat[:: max(int(flat.numel() // 1_000_000), 1)]
    values = torch.quantile(flat, torch.tensor(qs, device=flat.device))
    return [float(value.detach().cpu().item()) for value in values]


def cleanup_meta(x_ram: torch.Tensor, x_clean: torch.Tensor, model: torch.nn.Module) -> dict[str, Any]:
    delta_abs = (x_clean - x_ram).detach().abs()
    mean_abs_x = x_ram.detach().abs().mean().clamp_min(1e-12)
    p50, p95, p99 = tensor_percentiles(delta_abs)
    return {
        "post_ram_enabled": float(getattr(model, "post_ram_cleanup_enabled", False)),
        "post_ram_scale": float(getattr(getattr(model, "post_ram_cleanup", None), "scale", 0.0)),
        "delta_ratio": float((delta_abs.mean() / mean_abs_x).detach().cpu().item()),
        "delta_p50": p50,
        "delta_p95": p95,
        "delta_p99": p99,
        "clamp_total_ratio": float("nan"),
    }


def main() -> int:
    args = parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config, _ = get_config(args.run_dir.expanduser().resolve())
    train_args = train_args_from_run_config(
        config=config,
        eval_input=INPUT_DARK,
        output_dir=output_dir / "_build_args",
        args=args,
    )
    dataset = make_dataset(
        train_args,
        input_spec=INPUT_DARK,
        split=args.eval_split,
        crop_mode=str(config.get("lod_val_crop_mode") or "center"),
    )
    loader = make_loader(dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers)
    model = build_model(train_args)
    checkpoint_meta = load_checkpoint_into_model(model, args.checkpoint.expanduser().resolve())
    model.to(device)
    model.eval()

    use_amp = device.type == "cuda" and bool(getattr(train_args, "amp", False))
    amp_dtype = amp_dtype_from_train_args(train_args)
    rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    processed = 0
    start = time.time()

    with torch.no_grad():
        for batch in loader:
            if args.max_eval_samples is not None and processed >= int(args.max_eval_samples):
                break
            model_input = select_model_input(
                batch,
                train_args.resolved_config.model_input_tensor,
                dataset_family=train_args.resolved_config.dataset_family,
                sample_source="postram_cleanup_artifacts",
            ).to(device=device, non_blocking=True).float()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                x_ram, _ = model.ram_core.forward_with_features(model_input)
                if getattr(model, "post_ram_cleanup_enabled", False):
                    x_clean = model.post_ram_cleanup(x_ram)
                else:
                    x_clean = x_ram
                x_dav2 = x_clean
                if getattr(model, "raw_ram_rgb_tail", "identity") == "tanh2p5":
                    x_dav2 = phase1b_tanh_tail_squash(x_dav2)
                pred = model.dav2(model.spatial_adapter.pad_rgb(x_dav2)).float()
                pred = model.spatial_adapter.crop_depth(pred)
            meta = cleanup_meta(x_ram, x_clean, model)
            meta_rows.append(meta)
            depth_batch = batch["depth"].float()
            valid_batch = batch["valid_mask"].bool()
            if tuple(pred.shape[-2:]) != tuple(depth_batch.shape[-2:]):
                pred = F.interpolate(
                    pred[:, None],
                    tuple(int(v) for v in depth_batch.shape[-2:]),
                    mode="bilinear",
                    align_corners=True,
                )[:, 0]
            pred_np = pred.detach().cpu().numpy()
            depth_np = depth_batch.detach().cpu().numpy()
            valid_np = valid_batch.detach().cpu().numpy().astype(bool)

            batch_size = int(depth_batch.shape[0])
            for i in range(batch_size):
                if args.max_eval_samples is not None and processed >= int(args.max_eval_samples):
                    break
                target = depth_np[i]
                valid = valid_np[i] & np.isfinite(target) & (target > 0)
                if int(np.count_nonzero(valid)) < 10:
                    processed += 1
                    continue
                aligned_inverse, align_stats = affine_align_to_inverse_target(pred_np[i], target, valid)
                metrics = compute_inverse_relative_metrics(aligned_inverse, target, valid)
                if metrics is None:
                    processed += 1
                    continue
                ratio = np.maximum(
                    aligned_inverse[valid] / np.maximum(target[valid], 1e-12),
                    target[valid] / np.maximum(aligned_inverse[valid], 1e-12),
                )
                d1_hit = np.zeros_like(valid, dtype=bool)
                d1_hit[valid] = ratio < 1.25
                row: dict[str, Any] = {
                    "ext_id": "stageB_cleanup",
                    "denoiser": "cleanup_cnn" if getattr(model, "post_ram_cleanup_enabled", False) else "none",
                    "sigma": "n_a",
                    "alpha": 1.0,
                    "affine": "n_a",
                    "sample_index": int(processed),
                    "sample_id": str(collated_item(batch.get("sample_id", ""), i)),
                    "normal_id": int(collated_item(batch.get("normal_id", -1), i)),
                    "dark_id": int(collated_item(batch.get("dark_id", -1), i)),
                    "valid_pixels": int(align_stats.get("valid_pixels", int(np.count_nonzero(valid)))),
                    "align_scale": float(align_stats.get("scale", float("nan"))),
                    "align_shift": float(align_stats.get("shift", float("nan"))),
                }
                for key in METRIC_KEYS:
                    if key in metrics:
                        row[key] = float(metrics[key])
                row.update(meta)
                if len(artifact_rows) < int(args.artifact_max_samples):
                    artifact_path = output_dir / "artifacts" / f"{processed:06d}_{row['sample_id']}.npz"
                    save_artifact(
                        artifact_path,
                        batch=batch,
                        local_index=i,
                        x_ram=x_ram.detach(),
                        x_out=x_clean.detach(),
                        pred=pred.detach(),
                        target=target,
                        valid=valid,
                        aligned_inverse=aligned_inverse,
                        d1_hit=d1_hit,
                        row=row,
                    )
                    row["artifact_path"] = str(artifact_path)
                    artifact_rows.append(dict(row))
                rows.append(row)
                processed += 1

    summary: dict[str, Any] = {
        "ext_id": "stageB_cleanup",
        "denoiser": "cleanup_cnn" if str(config.get("post_ram_cleanup")) == "cnn" else "none",
        "sigma": "n_a",
        "alpha": 1.0,
        "affine": "n_a",
        "x_ram_label": "x_ram",
        "x_out_label": "x_clean",
        "samples": int(len(rows)),
        "dataset_size": int(len(dataset)),
        "elapsed_sec": float(time.time() - start),
        "device": str(device),
        "run_dir": str(args.run_dir.expanduser().resolve()),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_meta": checkpoint_meta,
        "eval_split": str(args.eval_split),
        "bn_protocol": "strict_no_bn_recalib",
        "artifact_count": int(len(artifact_rows)),
        "output_dir": str(output_dir),
    }
    for key in METRIC_KEYS:
        summary[key] = metric_average(rows, key)
    for key in ("post_ram_enabled", "post_ram_scale", "delta_ratio", "delta_p50", "delta_p95", "delta_p99", "clamp_total_ratio"):
        values = [float(row[key]) for row in meta_rows if key in row and math.isfinite(float(row[key]))]
        summary[key] = float(np.mean(values)) if values else float("nan")

    write_csv(output_dir / "per_sample.csv", rows)
    write_json(output_dir / "metrics.json", summary)
    write_json(output_dir / "manifest.json", {"summary": summary, "artifacts": artifact_rows})
    print(
        f"[POSTRAM_CLEANUP_ARTIFACTS] wrote {output_dir} "
        f"d1={summary.get('d1', float('nan')):.6f} delta_ratio={summary.get('delta_ratio', float('nan')):.6f} "
        f"artifacts={len(artifact_rows)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
