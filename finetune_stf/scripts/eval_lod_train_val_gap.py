#!/usr/bin/env python3
"""Evaluate fixed train-proxy vs val LOD gaps for existing true-LOD runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.config import ensure_resolved_config
from finetune_stf.dataset.lod_true import LODTrueRGBDark, LODTrueRawDarkRGB16
from finetune_stf.train import build_model, resolve_model_state, strip_module_prefix
from finetune_stf.util.metric import affine_align_to_inverse_target, compute_inverse_relative_metrics
from finetune_stf.util.model_input import coerce_model_input_tensor, select_model_input


DEFAULT_HEAVY_ROOT = Path("/mnt/drive/3333_raw/0000_exp_ckpt")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "finetune_stf/analysis/lod_overfit_diag"
METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog", "silog_x100", "d1", "d2", "d3")
LOD_TRUE_RAW_RGB16_INPUT_MODE_BY_FAMILY = {
    "lod_true_raw_dark_rgb16": "raw_rgb16_dark",
    "lod_true_raw_normal_rgb16": "raw_rgb16_normal",
}
LOD_TRUE_RAW_RGB16_DATASET_FAMILIES = set(LOD_TRUE_RAW_RGB16_INPUT_MODE_BY_FAMILY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate best/last checkpoint train-proxy vs val d1 gap for true-LOD runs."
    )
    parser.add_argument("run_dir", type=Path, help="Experiment directory containing config.json.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help='Checkpoint alias ("best"/"last") or .pth path. May be repeated. Default: best and last.',
    )
    parser.add_argument(
        "--checkpoint-label",
        action="append",
        default=None,
        help="Optional label for each --checkpoint. Must be repeated the same number of times.",
    )
    parser.add_argument("--heavy-root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-proxy-count", type=int, default=112)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-val-samples", type=int, default=None, help="Debug cap for both train_proxy and val.")
    parser.add_argument("--viz-count", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true", help="Disable autocast during eval.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def make_args_from_config(config: dict[str, Any]) -> Namespace:
    args = Namespace(**config)
    ensure_resolved_config(args)
    return args


def resolve_checkpoint(run_dir: Path, checkpoint_arg: str, heavy_root: Path) -> Path:
    if checkpoint_arg in {"best", "last"}:
        filename = "best_model.pth" if checkpoint_arg == "best" else "last_epoch_model.pth"
        candidates = [
            run_dir / filename,
            heavy_root.expanduser().resolve() / run_dir.name / filename,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        checked = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(f"Missing {checkpoint_arg} checkpoint. Checked:\n{checked}")
    path = Path(checkpoint_arg).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


def checkpoint_specs(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    checkpoint_args = args.checkpoint or ["best", "last"]
    labels = args.checkpoint_label
    if labels is not None and len(labels) != len(checkpoint_args):
        raise ValueError("--checkpoint-label must be repeated exactly once per --checkpoint")

    specs = []
    for idx, checkpoint_arg in enumerate(checkpoint_args):
        label = labels[idx] if labels else (checkpoint_arg if checkpoint_arg in {"best", "last"} else Path(checkpoint_arg).stem)
        specs.append(
            {
                "label": str(label),
                "arg": str(checkpoint_arg),
                "path": resolve_checkpoint(run_dir, str(checkpoint_arg), args.heavy_root),
            }
        )
    return specs


def build_lod_dataset(args: Namespace, *, split: str, mode: str, crop_mode: str):
    cfg = ensure_resolved_config(args)
    common = {
        "lod_root": args.lod_root,
        "manifest_path": args.lod_manifest,
        "split": split,
        "size": (int(args.input_height), int(args.input_width)),
        "mode": mode,
        "label_space": args.lod_label_space,
        "crop_mode": crop_mode,
    }
    if cfg.dataset_family == "lod_true_rgb_dark":
        return LODTrueRGBDark(**common)
    if cfg.dataset_family in LOD_TRUE_RAW_RGB16_DATASET_FAMILIES:
        return LODTrueRawDarkRGB16(
            raw_storage_format=args.raw_storage_format,
            lod_raw_norm_mode=args.lod_raw_norm_mode,
            raw_input_mode=LOD_TRUE_RAW_RGB16_INPUT_MODE_BY_FAMILY[cfg.dataset_family],
            **common,
        )
    raise ValueError(f"Unsupported dataset_family for true-LOD gap eval: {cfg.dataset_family!r}")


def select_train_proxy_indices(length: int, count: int, seed: int) -> list[int]:
    if count <= 0:
        raise ValueError("--train-proxy-count must be positive")
    count = min(int(count), int(length))
    rng = np.random.default_rng(int(seed))
    return sorted(int(idx) for idx in rng.choice(length, size=count, replace=False).tolist())


def load_model_for_checkpoint(args: Namespace, checkpoint_path: Path) -> torch.nn.Module:
    model = build_model(args)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = strip_module_prefix(resolve_model_state(ckpt))
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    return model


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return {"checkpoint_epoch": None, "best_lod_d1": None, "best_metrics": None}
    best_metrics = ckpt.get("best_metrics")
    return {
        "checkpoint_epoch": ckpt.get("epoch"),
        "best_lod_d1": ckpt.get("best_lod_d1"),
        "best_metric": ckpt.get("best_metric"),
        "best_metrics": best_metrics if isinstance(best_metrics, dict) else None,
    }


def finite_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def average_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    summary = {}
    for key in METRIC_KEYS:
        values = [float(record[key]) for record in records if key in record and math.isfinite(float(record[key]))]
        summary[key] = float(np.mean(values)) if values else float("nan")
    return summary


def batch_meta_value(sample: dict[str, Any], key: str, index: int) -> Any:
    value = sample.get(key)
    if isinstance(value, (list, tuple)):
        return value[index]
    if torch.is_tensor(value):
        item = value[index]
        return item.item() if item.ndim == 0 else item.detach().cpu().tolist()
    return value


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    args: Namespace,
    *,
    split_name: str,
    checkpoint_label: str,
    max_samples: int | None,
    use_amp: bool,
    viz_keep: set[str],
    viz_cache: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    cfg = ensure_resolved_config(args)
    model_input_tensor = coerce_model_input_tensor(cfg.model_input_tensor)
    amp_dtype = torch.float16 if getattr(args, "amp_dtype", "bf16") == "fp16" else torch.bfloat16
    records: list[dict[str, Any]] = []
    processed = 0

    for sample in loader:
        if max_samples is not None and processed >= max_samples:
            break
        image = select_model_input(
            sample,
            model_input_tensor,
            dataset_family=cfg.dataset_family,
            sample_source=split_name,
        ).cuda(non_blocking=True).float()
        depth = sample["depth"].cuda(non_blocking=True).float()
        valid_mask = sample["valid_mask"].cuda(non_blocking=True).bool()

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            pred_disp = model(image).float()
        pred_disp = F.interpolate(pred_disp[:, None], depth.shape[-2:], mode="bilinear", align_corners=True)[:, 0]

        batch_size = int(depth.shape[0])
        for batch_idx in range(batch_size):
            if max_samples is not None and processed >= max_samples:
                break
            target = depth[batch_idx]
            valid = valid_mask[batch_idx] & torch.isfinite(target) & (target > 0)
            sample_id = str(batch_meta_value(sample, "sample_id", batch_idx))
            if int(valid.sum().item()) < 10:
                processed += 1
                continue

            pred_np = pred_disp[batch_idx].detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy().astype(bool)
            aligned_inverse, align_stats = affine_align_to_inverse_target(pred_np, target_np, valid_np)
            metrics = compute_inverse_relative_metrics(aligned_inverse, target_np, valid_np)
            if metrics is None:
                processed += 1
                continue

            record = {
                "split": split_name,
                "checkpoint_label": checkpoint_label,
                "sample_index": processed,
                "sample_id": sample_id,
                "normal_id": batch_meta_value(sample, "normal_id", batch_idx),
                "dark_id": batch_meta_value(sample, "dark_id", batch_idx),
                "valid_pixels": int(valid_np.sum()),
                "align_scale": finite_float(align_stats.get("scale")),
                "align_shift": finite_float(align_stats.get("shift")),
                **{key: float(metrics[key]) for key in METRIC_KEYS if key in metrics},
            }
            records.append(record)

            if sample_id in viz_keep:
                cache_key = (split_name, sample_id)
                cache = viz_cache.setdefault(
                    cache_key,
                    {
                        "target": target_np.astype(np.float32, copy=False),
                        "valid": valid_np,
                        "normal_id": record["normal_id"],
                        "dark_id": record["dark_id"],
                    },
                )
                cache[f"pred_{checkpoint_label}"] = aligned_inverse.astype(np.float32, copy=False)

            processed += 1

    return average_metrics(records), records


def read_manifest_rows(manifest_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    with manifest_path.expanduser().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item = {
                "pair_id": row["pair_id"],
                "split": row["split"],
                "normal_id": int(row["normal_id"]),
                "dark_id": int(row["dark_id"]),
            }
            if item["split"] == "00Train":
                train_rows.append(item)
            elif item["split"] == "01Valid":
                val_rows.append(item)
    return train_rows, val_rows


def percentile_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("min", "p1", "p5", "p10", "p50", "p90", "p95", "p99", "max", "mean")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p1": float(np.percentile(array, 1)),
        "p5": float(np.percentile(array, 5)),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def split_neighbor_audit(manifest_path: Path) -> dict[str, Any]:
    train_rows, val_rows = read_manifest_rows(manifest_path)
    if not train_rows or not val_rows:
        raise ValueError(f"Manifest must contain both 00Train and 01Valid rows: {manifest_path}")

    train_normal_ids = np.asarray([row["normal_id"] for row in train_rows], dtype=np.int64)
    val_records = []
    nearest_id_distances = []
    nearest_pair_distances = []
    for row in val_rows:
        distances = np.abs(train_normal_ids - int(row["normal_id"]))
        nearest_idx = int(np.argmin(distances))
        nearest = train_rows[nearest_idx]
        id_distance = int(distances[nearest_idx])
        pair_distance = id_distance / 2.0
        nearest_id_distances.append(float(id_distance))
        nearest_pair_distances.append(float(pair_distance))
        val_records.append(
            {
                "val_pair_id": row["pair_id"],
                "val_normal_id": row["normal_id"],
                "val_dark_id": row["dark_id"],
                "nearest_train_pair_id": nearest["pair_id"],
                "nearest_train_normal_id": nearest["normal_id"],
                "nearest_train_dark_id": nearest["dark_id"],
                "nearest_id_distance": id_distance,
                "nearest_pair_distance": pair_distance,
            }
        )

    thresholds = (1, 2, 5, 10, 20, 50)
    within_k_pairs = {
        str(k): int(sum(distance <= float(k) for distance in nearest_pair_distances)) for k in thresholds
    }
    return {
        "manifest_path": str(manifest_path.expanduser().resolve()),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "distance_definition": "nearest_pair_distance = abs(val.normal_id - train.normal_id) / 2",
        "nearest_id_distance_summary": percentile_summary(nearest_id_distances),
        "nearest_pair_distance_summary": percentile_summary(nearest_pair_distances),
        "val_within_k_pairs": within_k_pairs,
        "val_within_k_pairs_fraction": {
            key: float(value / len(val_rows)) for key, value in within_k_pairs.items()
        },
        "records": val_records,
    }


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    fieldnames = [
        "split",
        "checkpoint_label",
        "sample_index",
        "sample_id",
        "normal_id",
        "dark_id",
        "valid_pixels",
        "align_scale",
        "align_shift",
        *METRIC_KEYS,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def colorize(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    finite = valid & np.isfinite(array)
    if int(finite.sum()) == 0:
        return np.zeros((*array.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(array[finite], [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(array[finite])), float(np.max(array[finite]))
    if hi <= lo:
        hi = lo + 1.0
    gray = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    gray = (gray * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~finite] = np.array([32, 32, 32], dtype=np.uint8)
    return rgb


def labeled_panel(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(output, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def write_visualizations(output_dir: Path, viz_cache: dict[tuple[str, str], dict[str, Any]], checkpoint_labels: list[str]) -> None:
    if not viz_cache:
        return
    viz_dir = output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for (split_name, sample_id), cache in sorted(viz_cache.items()):
        target = cache["target"]
        valid = cache["valid"]
        panels = [labeled_panel(colorize(target, valid), "target")]
        for label in checkpoint_labels:
            pred = cache.get(f"pred_{label}")
            if pred is None:
                continue
            panels.append(labeled_panel(colorize(pred, valid), f"{label} aligned"))
            panels.append(labeled_panel(colorize(np.abs(pred - target), valid), f"{label} abs diff"))
        contact = np.concatenate(panels, axis=1)
        safe_sample_id = str(sample_id).replace("/", "_")
        cv2.imwrite(str(viz_dir / f"{split_name}_{safe_sample_id}.png"), cv2.cvtColor(contact, cv2.COLOR_RGB2BGR))


def default_output_dir(output_root: Path, run_name: str) -> Path:
    timestamp = time.strftime("%m%d_%H%M")
    return output_root.expanduser().resolve() / f"{timestamp}_{run_name}"


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LOD gap evaluation.")

    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    run_config = load_json(config_path)
    train_args = make_args_from_config(run_config)
    cfg = ensure_resolved_config(train_args)
    if cfg.dataset_family not in {"lod_true_rgb_dark", *LOD_TRUE_RAW_RGB16_DATASET_FAMILIES}:
        raise ValueError(f"Expected true-LOD dataset family, got {cfg.dataset_family!r}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else default_output_dir(args.output_root, run_dir.name)
    output_dir.mkdir(parents=True, exist_ok=False)

    specs = checkpoint_specs(args, run_dir)
    train_dataset_full = build_lod_dataset(train_args, split="00Train", mode="val", crop_mode="center")
    train_indices = select_train_proxy_indices(len(train_dataset_full), args.train_proxy_count, args.seed)
    train_proxy = Subset(train_dataset_full, train_indices)
    val_dataset = build_lod_dataset(train_args, split="01Valid", mode="val", crop_mode="center")

    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(
        train_proxy,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        **loader_kwargs,
    )

    viz_keep: set[str] = set()
    if args.viz_count > 0:
        for dataset in (train_proxy, val_dataset):
            limit = min(args.viz_count, len(dataset))
            for idx in range(limit):
                row = dataset.dataset.rows[dataset.indices[idx]] if isinstance(dataset, Subset) else dataset.rows[idx]
                viz_keep.add(str(row["pair_id"]))

    use_amp = bool(getattr(train_args, "amp", True)) and not args.no_amp
    all_records: list[dict[str, Any]] = []
    checkpoint_results = {}
    viz_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for spec in specs:
        print(f"[EVAL] run={run_dir.name} checkpoint={spec['label']} path={spec['path']}", flush=True)
        model = load_model_for_checkpoint(train_args, spec["path"])
        train_summary, train_records = evaluate_loader(
            model,
            train_loader,
            train_args,
            split_name="lod_train_proxy",
            checkpoint_label=spec["label"],
            max_samples=args.max_val_samples,
            use_amp=use_amp,
            viz_keep=viz_keep,
            viz_cache=viz_cache,
        )
        val_summary, val_records = evaluate_loader(
            model,
            val_loader,
            train_args,
            split_name="lod_val",
            checkpoint_label=spec["label"],
            max_samples=args.max_val_samples,
            use_amp=use_amp,
            viz_keep=viz_keep,
            viz_cache=viz_cache,
        )
        all_records.extend(train_records)
        all_records.extend(val_records)
        checkpoint_results[spec["label"]] = {
            "checkpoint_arg": spec["arg"],
            "checkpoint_path": str(spec["path"]),
            "checkpoint_metadata": checkpoint_metadata(spec["path"]),
            "lod_train_proxy": train_summary,
            "lod_val": val_summary,
            "gap": {
                "d1_train_minus_val": float(train_summary["d1"] - val_summary["d1"]),
                "abs_rel_train_minus_val": float(train_summary["abs_rel"] - val_summary["abs_rel"]),
                "train_proxy_count": len(train_records),
                "val_count": len(val_records),
            },
        }
        print(
            "[EVAL] done checkpoint=%s train_d1=%.4f val_d1=%.4f gap=%.4f"
            % (spec["label"], train_summary["d1"], val_summary["d1"], train_summary["d1"] - val_summary["d1"]),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    audit = split_neighbor_audit(Path(train_args.lod_manifest))
    report = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "dataset_family": cfg.dataset_family,
        "input_domain": cfg.input_domain,
        "front_end": cfg.front_end,
        "model_input_tensor": cfg.model_input_tensor,
        "lod_root": train_args.lod_root,
        "lod_manifest": train_args.lod_manifest,
        "train_proxy": {
            "split": "00Train",
            "count_requested": int(args.train_proxy_count),
            "count_actual": int(len(train_proxy)),
            "seed": int(args.seed),
            "indices": train_indices,
            "crop_mode": "center",
            "augmentation": "off",
        },
        "val": {
            "split": "01Valid",
            "count_actual": int(len(val_dataset)),
            "crop_mode": "center",
            "augmentation": "off",
        },
        "checkpoints": checkpoint_results,
        "split_neighbor_audit": audit,
        "notes": [
            "train_proxy and val use center crop and no augmentation.",
            "The fixed split is pair-random seed42; neighbor audit determines whether fixed-split gap is only a lower-bound signal.",
        ],
    }
    save_json(output_dir / "gap_report.json", report)
    save_json(output_dir / "split_neighbor_audit.json", audit)
    write_records_csv(output_dir / "per_sample_metrics.csv", all_records)
    write_visualizations(output_dir, viz_cache, [spec["label"] for spec in specs])

    print(f"[DONE] report={output_dir / 'gap_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
