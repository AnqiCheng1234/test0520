#!/usr/bin/env python3
"""Evaluate a post-RAM external wrapper on true-LOD RAW strict split."""

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
from finetune_stf.models.raw_ram import phase1b_tanh_tail_squash  # noqa: E402
from finetune_stf.train import METRIC_KEYS, build_model, set_random_seed  # noqa: E402
from finetune_stf.util.metric import affine_align_to_inverse_target, compute_inverse_relative_metrics  # noqa: E402
from finetune_stf.util.model_input import select_model_input  # noqa: E402


DEFAULT_RUN = "0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly"
DEFAULT_EXP_ROOT = PROJECT_ROOT / "finetune_stf" / "exp"
DEFAULT_CKPT_ROOT = Path("/mnt/drive/3333_raw/0000_exp_ckpt")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "finetune_stf" / "analysis" / "lod_postram_external"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_EXP_ROOT / DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT_ROOT / DEFAULT_RUN / "best_model.pth")
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ext-id", required=True)
    parser.add_argument("--denoiser", choices=("none", "identity", "drunet", "restormer", "nafnet"), required=True)
    parser.add_argument("--sigma", type=float, default=None, help="Noise level in /255 units, e.g. 10 means 10/255.")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--affine", choices=("n_a", "q001q999", "q01q99"), default="n_a")
    parser.add_argument("--denoiser-torchscript", type=Path, default=None)
    parser.add_argument("--denoiser-takes-sigma", action="store_true")
    parser.add_argument("--denoiser-concat-sigma-channel", action="store_true")
    parser.add_argument("--lod-root", type=Path, default=None)
    parser.add_argument("--lod-manifest", type=Path, default=None)
    parser.add_argument("--pretrained-from", type=Path, default=None)
    parser.add_argument("--eval-split", default="01Valid")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default=None)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--artifact-max-samples", type=int, default=16)
    parser.add_argument("--artifact-sample-indices", default="")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


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


def parse_index_set(value: str) -> set[int]:
    result = set()
    for item in str(value or "").split(","):
        item = item.strip()
        if item:
            result.add(int(item))
    return result


def affine_keys(name: str) -> tuple[str, str]:
    if name == "q001q999":
        return "q001", "q999"
    if name == "q01q99":
        return "q01", "q99"
    raise ValueError(f"affine={name!r} does not define quantile keys")


def load_affine(stats_json: Path, affine: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = load_json(stats_json)
    qlo_key, qhi_key = affine_keys(affine)
    channels = payload.get("channels") or []
    if len(channels) != 3:
        raise ValueError(f"Expected 3 channel stats in {stats_json}, got {len(channels)}")
    qlo = torch.tensor([float(row[qlo_key]) for row in channels], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    qhi = torch.tensor([float(row[qhi_key]) for row in channels], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    if torch.any(~torch.isfinite(qlo)) or torch.any(~torch.isfinite(qhi)) or torch.any(qhi <= qlo):
        raise ValueError(f"Invalid affine stats in {stats_json}: qlo={qlo.flatten()}, qhi={qhi.flatten()}")
    return qlo, qhi, {"stats_json": str(stats_json), "qlo_key": qlo_key, "qhi_key": qhi_key}


class ExternalDenoiser:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.name = str(args.denoiser)
        self.sigma = args.sigma
        self.module = None
        self.takes_sigma = bool(args.denoiser_takes_sigma)
        self.concat_sigma_channel = bool(args.denoiser_concat_sigma_channel)
        if self.name in {"drunet", "restormer", "nafnet"}:
            if args.denoiser_torchscript is None:
                raise ValueError(
                    f"denoiser={self.name} requires --denoiser-torchscript in this repo revision; "
                    "no fallback filter is used"
                )
            self.module = torch.jit.load(str(args.denoiser_torchscript.expanduser()), map_location=device)
            self.module.eval()
            for param in self.module.parameters():
                param.requires_grad_(False)

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        if self.name == "identity":
            return u
        if self.name == "none":
            return u
        if self.module is None:
            raise RuntimeError(f"denoiser={self.name} has no loaded module")
        sigma_value = 0.0 if self.sigma is None else float(self.sigma) / 255.0
        sigma = torch.full((u.shape[0], 1, u.shape[2], u.shape[3]), sigma_value, device=u.device, dtype=u.dtype)
        with torch.no_grad():
            if self.concat_sigma_channel:
                return self.module(torch.cat([u, sigma], dim=1))
            if self.takes_sigma:
                return self.module(u, sigma)
            return self.module(u)


def tensor_percentiles(x: torch.Tensor, qs=(0.50, 0.95, 0.99)) -> list[float]:
    flat = x.detach().float().reshape(-1)
    if flat.numel() == 0:
        return [float("nan") for _ in qs]
    if flat.numel() > 1_000_000:
        flat = flat[:: max(int(flat.numel() // 1_000_000), 1)]
    values = torch.quantile(flat, torch.tensor(qs, device=flat.device))
    return [float(v.detach().cpu().item()) for v in values]


def apply_wrapper(
    *,
    x_ram: torch.Tensor,
    args: argparse.Namespace,
    denoiser: ExternalDenoiser,
    qlo: torch.Tensor | None,
    qhi: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if args.denoiser == "none":
        zero = torch.zeros((3,), device=x_ram.device, dtype=x_ram.dtype)
        return x_ram, {
            "clamp_low_ratio": 0.0,
            "clamp_high_ratio": 0.0,
            "clamp_total_ratio": 0.0,
            "clamp_low_ratio_ch": [0.0, 0.0, 0.0],
            "clamp_high_ratio_ch": [0.0, 0.0, 0.0],
            "clamp_total_ratio_ch": [0.0, 0.0, 0.0],
            "delta_ratio": 0.0,
            "delta_p50": 0.0,
            "delta_p95": 0.0,
            "delta_p99": 0.0,
        }
    if qlo is None or qhi is None:
        raise ValueError(f"denoiser={args.denoiser} requires an affine stats file")
    denom = (qhi - qlo).clamp_min(1e-12)
    u_raw = (x_ram - qlo) / denom
    low = u_raw < 0.0
    high = u_raw > 1.0
    u = u_raw.clamp(0.0, 1.0)
    u_dn = denoiser(u).clamp(0.0, 1.0)
    x_dn = u_dn * denom + qlo
    x_out = x_ram + float(args.alpha) * (x_dn - x_ram)
    delta_abs = (x_out - x_ram).detach().abs()
    mean_abs_x = x_ram.detach().abs().mean().clamp_min(1e-12)
    low_ch = low.float().mean(dim=(0, 2, 3))
    high_ch = high.float().mean(dim=(0, 2, 3))
    total_ch = (low | high).float().mean(dim=(0, 2, 3))
    p50, p95, p99 = tensor_percentiles(delta_abs, qs=(0.50, 0.95, 0.99))
    return x_out, {
        "clamp_low_ratio": float(low.float().mean().detach().cpu().item()),
        "clamp_high_ratio": float(high.float().mean().detach().cpu().item()),
        "clamp_total_ratio": float((low | high).float().mean().detach().cpu().item()),
        "clamp_low_ratio_ch": [float(v) for v in low_ch.detach().cpu().tolist()],
        "clamp_high_ratio_ch": [float(v) for v in high_ch.detach().cpu().tolist()],
        "clamp_total_ratio_ch": [float(v) for v in total_ch.detach().cpu().tolist()],
        "delta_ratio": float((delta_abs.mean() / mean_abs_x).detach().cpu().item()),
        "delta_p50": p50,
        "delta_p95": p95,
        "delta_p99": p99,
    }


def save_artifact(
    path: Path,
    *,
    batch: dict[str, Any],
    local_index: int,
    x_ram: torch.Tensor,
    x_out: torch.Tensor,
    pred: torch.Tensor,
    target: np.ndarray,
    valid: np.ndarray,
    aligned_inverse: np.ndarray,
    d1_hit: np.ndarray,
    row: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_preview = batch.get("rgb_preview", batch.get("raw"))
    raw_preview_np = None
    if isinstance(raw_preview, torch.Tensor):
        raw_preview_np = raw_preview[local_index].detach().float().cpu().numpy().astype(np.float16)
    np.savez_compressed(
        path,
        sample_id=str(row["sample_id"]),
        sample_index=np.array(int(row["sample_index"]), dtype=np.int32),
        raw_preview=raw_preview_np,
        x_ram=x_ram[local_index].detach().float().cpu().numpy().astype(np.float16),
        x_out=x_out[local_index].detach().float().cpu().numpy().astype(np.float16),
        delta=(x_out[local_index] - x_ram[local_index]).detach().float().cpu().numpy().astype(np.float16),
        pred=pred[local_index].detach().float().cpu().numpy().astype(np.float32),
        target=target.astype(np.float32),
        valid=valid.astype(bool),
        aligned_inverse=aligned_inverse.astype(np.float32),
        error=np.abs(aligned_inverse - target).astype(np.float32),
        d1_hit=d1_hit.astype(bool),
        row=json.dumps(row, sort_keys=True),
    )


def main() -> None:
    args = parse_args()
    if args.denoiser == "none":
        if args.affine != "n_a" or abs(float(args.alpha)) > 1e-12 or args.sigma is not None:
            raise ValueError("denoiser=none is EXT_NOOP_A0 bypass; requires affine=n_a, alpha=0, sigma omitted")
    else:
        if args.affine == "n_a":
            raise ValueError(f"denoiser={args.denoiser} requires --affine q001q999/q01q99")
        if args.denoiser == "identity" and args.sigma is not None:
            raise ValueError("denoiser=identity requires sigma omitted")
        if args.denoiser == "drunet" and args.sigma is None:
            raise ValueError("denoiser=drunet requires --sigma 10/25/50")
        if args.denoiser in {"restormer", "nafnet"} and args.sigma is not None:
            raise ValueError("denoiser=restormer/nafnet requires sigma omitted")

    set_random_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config, _ = get_config(args.run_dir)
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
    checkpoint_meta = load_checkpoint_into_model(model, args.checkpoint)
    model.to(device)
    model.eval()
    denoiser = ExternalDenoiser(args, device)
    qlo = qhi = None
    affine_meta: dict[str, Any] = {}
    if args.affine != "n_a":
        qlo, qhi, affine_meta = load_affine(args.stats_json, args.affine, device)

    use_amp = device.type == "cuda" and bool(getattr(train_args, "amp", False))
    amp_dtype = amp_dtype_from_train_args(train_args)
    artifact_indices = parse_index_set(args.artifact_sample_indices)
    rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    aggregate_clamp: list[dict[str, Any]] = []
    processed = 0
    start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_eval_samples is not None and processed >= int(args.max_eval_samples):
                break
            model_input = select_model_input(
                batch,
                train_args.resolved_config.model_input_tensor,
                dataset_family=train_args.resolved_config.dataset_family,
                sample_source=f"{args.ext_id}_postram_external",
            ).to(device=device, non_blocking=True).float()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                x_ram, _ = model.ram_core.forward_with_features(model_input)
                x_out, wrapper_meta = apply_wrapper(
                    x_ram=x_ram,
                    args=args,
                    denoiser=denoiser,
                    qlo=qlo,
                    qhi=qhi,
                )
                x_dav2 = x_out
                if getattr(model, "raw_ram_rgb_tail", "identity") == "tanh2p5":
                    x_dav2 = phase1b_tanh_tail_squash(x_dav2)
                pred = model.dav2(model.spatial_adapter.pad_rgb(x_dav2)).float()
                pred = model.spatial_adapter.crop_depth(pred)
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
                ratio = np.maximum(aligned_inverse[valid] / np.maximum(target[valid], 1e-12), target[valid] / np.maximum(aligned_inverse[valid], 1e-12))
                d1_hit = np.zeros_like(valid, dtype=bool)
                d1_hit[valid] = ratio < 1.25
                row: dict[str, Any] = {
                    "ext_id": args.ext_id,
                    "denoiser": args.denoiser,
                    "sigma": args.sigma if args.sigma is not None else "n_a",
                    "alpha": float(args.alpha),
                    "affine": args.affine,
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
                for key, value in wrapper_meta.items():
                    if isinstance(value, list):
                        for ch, item in enumerate(value):
                            row[f"{key}_{ch}"] = float(item)
                    else:
                        row[key] = float(value)
                save_this = processed in artifact_indices or (
                    not artifact_indices and len(artifact_rows) < int(args.artifact_max_samples)
                )
                if save_this:
                    artifact_path = output_dir / "artifacts" / f"{processed:06d}_{row['sample_id']}.npz"
                    save_artifact(
                        artifact_path,
                        batch=batch,
                        local_index=i,
                        x_ram=x_ram.detach(),
                        x_out=x_out.detach(),
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
                aggregate_clamp.append(wrapper_meta)
                processed += 1
            if args.progress_interval > 0 and processed > 0 and processed % int(args.progress_interval) == 0:
                print(f"[POSTRAM_EVAL] {args.ext_id} processed={processed}", flush=True)

    summary: dict[str, Any] = {
        "ext_id": args.ext_id,
        "denoiser": args.denoiser,
        "sigma": args.sigma if args.sigma is not None else "n_a",
        "alpha": float(args.alpha),
        "affine": args.affine,
        "samples": int(len(rows)),
        "dataset_size": int(len(dataset)),
        "elapsed_sec": float(time.time() - start),
        "device": str(device),
        "run_dir": str(args.run_dir.expanduser().resolve()),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_meta": checkpoint_meta,
        "eval_split": str(args.eval_split),
        "bn_protocol": "strict_no_bn_recalib",
        "affine_meta": affine_meta,
        "artifact_count": int(len(artifact_rows)),
        "output_dir": str(output_dir),
        "train_args_resolved_config": train_args.resolved_config.to_dict(),
    }
    for key in METRIC_KEYS:
        summary[key] = metric_average(rows, key)
    for key in ("clamp_low_ratio", "clamp_high_ratio", "clamp_total_ratio", "delta_ratio", "delta_p50", "delta_p95", "delta_p99"):
        values = [float(meta[key]) for meta in aggregate_clamp if key in meta and math.isfinite(float(meta[key]))]
        summary[key] = float(np.mean(values)) if values else float("nan")

    write_csv(output_dir / "per_sample.csv", rows)
    write_json(output_dir / "metrics.json", summary)
    write_json(
        output_dir / "manifest.json",
        {
            "summary": summary,
            "artifacts": artifact_rows,
        },
    )
    print(
        f"[POSTRAM_EVAL] wrote {output_dir} d1={summary.get('d1', float('nan')):.6f} "
        f"abs_rel={summary.get('abs_rel', float('nan')):.6f} samples={len(rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
