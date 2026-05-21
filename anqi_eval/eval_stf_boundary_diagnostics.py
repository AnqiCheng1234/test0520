#!/usr/bin/env python3
"""STF boundary diagnostics for RGB/RAW baseline comparison.

This script evaluates two boundary-oriented diagnostics on the STF val split:

1. Pseudo-depth boundary error (PDBE-style): compare model prediction edges
   against a fixed dense pseudo target edge map.
2. Image-edge-band sparse metrics: compute abs_rel/d1 only at sparse LiDAR
   points that fall near strong RGB image edges.

The regular sparse STF abs_rel/d1 are also reported as a sanity check.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from finetune_stf.dataset.stf import STF
from finetune_stf.dataset.stf_raw import STF_RAW
from finetune_stf.train import (
    RAW_MODEL_INPUT_TYPES,
    RGB_INPUT_TYPES,
    build_model,
    load_initial_weights,
    resolve_model_state,
    resolve_stf_raw_input_mode,
    strip_module_prefix,
)


DEFAULT_REFERENCE_CONFIG = (
    PROJECT_ROOT
    / "finetune_stf/exp/0521_0133_stf_train_test_pseudovitl_rgb_decoder_e5/config.json"
)
DEFAULT_PSEUDO_MANIFEST = Path(
    "/mnt/drive/3333_raw/seeing_through_fog/"
    "pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/"
    "stf_rgb_lut_manifest_6216.csv"
)
DEFAULT_CKPT_ROOT = Path("/mnt/drive/3333_raw/0000_exp_ckpt")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plans/result/stf_boundary_diagnostics_0521"

SUMMARY_RUNS = (
    "0521_0133_stf_train_test_pseudovitl_rgb_decoder_e5",
    "0521_0306_stf_train_test_pseudovitl_rgb_lora_decoder_e5",
    "0521_0402_stf_train_test_pseudovitl_rgb_full_lrd09_e5",
    "0521_0012_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_e5",
    "0521_0112_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_decoder_e5",
    "0521_0522_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_lora_decoder_e10",
    "0521_0656_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_full_lrd09_e10",
    "0521_0835_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_lora_decoder_e10",
    "0521_1004_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_full_lrd09_e10",
    "0521_1137_stf_train_test_pseudoda3_sparse_metric_rgb_lora_decoder_e5",
)

DIRECT_BASELINES = (
    "dav2s_rgb_direct",
    "dav2s_raw_preview_direct",
)

SUMMARY_LABELS = {
    "dav2s_rgb_direct": "DAv2-S RGB direct",
    "dav2s_raw_preview_direct": "DAv2-S RAW-preview direct",
    "0521_0133_stf_train_test_pseudovitl_rgb_decoder_e5": "0521_0133 rgb_decoder",
    "0521_0306_stf_train_test_pseudovitl_rgb_lora_decoder_e5": "0521_0306 rgb_lora_decoder",
    "0521_0402_stf_train_test_pseudovitl_rgb_full_lrd09_e5": "0521_0402 rgb_full",
    "0521_0012_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_e5": "0521_0012 raw_identity",
    "0521_0112_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_decoder_e5": "0521_0112 raw_identity_decoder",
    "0521_0522_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_lora_decoder_e10": "0521_0522 raw_identity_lora",
    "0521_0656_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_full_lrd09_e10": "0521_0656 raw_identity_full",
    "0521_0835_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_lora_decoder_e10": "0521_0835 raw_bridge_lora",
    "0521_1004_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_full_lrd09_e10": "0521_1004 raw_bridge_full",
    "0521_1137_stf_train_test_pseudoda3_sparse_metric_rgb_lora_decoder_e5": "0521_1137 da3_rgb_lora",
}


@dataclass(frozen=True)
class EvalItem:
    item_id: str
    label: str
    config_path: Path | None
    checkpoint_path: Path | None
    direct_mode: str | None = None
    input_source: str = "config"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        default="summary",
        choices=("summary",),
        help="Named experiment suite. 'summary' matches rgb_raw_baseline_fairness_summary.md.",
    )
    parser.add_argument("--run", action="append", default=None, help="Run name(s) to evaluate instead of the full suite.")
    parser.add_argument("--include-direct", action="store_true", default=True)
    parser.add_argument("--no-include-direct", action="store_false", dest="include_direct")
    parser.add_argument("--reference-config", type=Path, default=DEFAULT_REFERENCE_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--pseudo-manifest", type=Path, default=DEFAULT_PSEUDO_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--checkpoint-name", default="best_model.pth")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--edge-percentile", type=float, default=95.0)
    parser.add_argument("--image-edge-percentile", type=float, default=90.0)
    parser.add_argument("--image-edge-dilate", type=int, default=3)
    parser.add_argument("--boundary-tolerance", type=float, default=3.0)
    parser.add_argument("--dbe-truncation", type=float, default=10.0)
    parser.add_argument("--min-edge-points", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=50)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")


def load_depth_npz(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "arr_0" not in data.files:
            raise KeyError(f"{path} does not contain arr_0")
        return np.array(data["arr_0"], dtype=np.float32, copy=True)


def load_pseudo_manifest(path: Path, split: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.expanduser().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_name", "split", "rgb_path", "sparse_depth_path", "pseudo_depth_npy"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        for row in reader:
            if row["split"] == split:
                rows[row["sample_name"]] = row
    if not rows:
        raise ValueError(f"No pseudo rows found for split={split}: {path}")
    return rows


def to_namespace(cfg: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**cfg)


def build_eval_items(args: argparse.Namespace) -> list[EvalItem]:
    items: list[EvalItem] = []
    run_names = tuple(args.run) if args.run else SUMMARY_RUNS
    if args.include_direct and not args.run:
        items.extend(
            [
                EvalItem(
                    item_id="dav2s_rgb_direct",
                    label=SUMMARY_LABELS["dav2s_rgb_direct"],
                    config_path=None,
                    checkpoint_path=None,
                    direct_mode="rgb",
                    input_source="rgb",
                ),
                EvalItem(
                    item_id="dav2s_raw_preview_direct",
                    label=SUMMARY_LABELS["dav2s_raw_preview_direct"],
                    config_path=None,
                    checkpoint_path=None,
                    direct_mode="raw_preview",
                    input_source="raw_preview",
                ),
            ]
        )

    for run_name in run_names:
        config_path = PROJECT_ROOT / "finetune_stf/exp" / run_name / "config.json"
        ckpt_path = args.checkpoint_root.expanduser() / run_name / args.checkpoint_name
        items.append(
            EvalItem(
                item_id=run_name,
                label=SUMMARY_LABELS.get(run_name, run_name),
                config_path=config_path,
                checkpoint_path=ckpt_path,
                direct_mode=None,
                input_source="config",
            )
        )
    return items


def config_for_item(item: EvalItem, reference_config: Path) -> SimpleNamespace:
    if item.config_path is None:
        cfg = load_json(reference_config.expanduser())
        cfg["input_type"] = "rgb"
        cfg["dav2_train_mode"] = "none"
        cfg["resume_from"] = None
        return to_namespace(cfg)

    if not item.config_path.is_file():
        raise FileNotFoundError(f"Missing config: {item.config_path}")
    return to_namespace(load_json(item.config_path))


def dataset_for_item(cfg: SimpleNamespace, item: EvalItem, split: str):
    size = (int(cfg.input_height), int(cfg.input_width))
    if item.input_source == "raw_preview":
        return STF_RAW(
            split,
            stf_root=cfg.stf_root,
            raw_npz_root=cfg.raw_npz_root,
            size=size,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            merge_test_into_train=False,
            norm_mode="passthrough",
            channel_mode="rgb_avg_g",
            use_imagenet_norm=True,
            input_mode="raw_naive",
            stf_raw_decode_mode="legacy_online_decomp16",
            depth_mode="fast",
            fast_eval_backend="sparse",
        )

    if cfg.input_type in RGB_INPUT_TYPES:
        return STF(
            split,
            stf_root=cfg.stf_root,
            size=size,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            merge_test_into_train=False,
        )

    if cfg.input_type in RAW_MODEL_INPUT_TYPES:
        return STF_RAW(
            split,
            stf_root=cfg.stf_root,
            raw_npz_root=cfg.raw_npz_root,
            size=size,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            merge_test_into_train=False,
            norm_mode=cfg.norm_mode,
            channel_mode=cfg.channel_mode,
            use_imagenet_norm=cfg.use_imagenet_norm,
            input_mode=resolve_stf_raw_input_mode(cfg.input_type),
            stf_raw_decode_mode=cfg.stf_raw_decode_mode,
            depth_mode="fast",
            fast_eval_backend=getattr(cfg, "stf_fast_eval_backend", "sparse"),
        )

    raise ValueError(f"Unsupported input_type={cfg.input_type!r}")


def load_model_for_item(cfg: SimpleNamespace, item: EvalItem, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg)
    if item.direct_mode is not None:
        load_initial_weights(model, cfg.pretrained_from, input_type=cfg.input_type)
    else:
        if item.checkpoint_path is None or not item.checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {item.checkpoint_path}")
        ckpt = torch.load(item.checkpoint_path, map_location="cpu")
        model.load_state_dict(strip_module_prefix(resolve_model_state(ckpt)), strict=True)
    model.to(device)
    model.eval()
    return model


def sample_name(sample: dict[str, Any]) -> str:
    name = sample["sample_name"]
    if isinstance(name, (list, tuple)):
        return str(name[0])
    return str(name)


def tensor_from_sample(sample: dict[str, Any], cfg: SimpleNamespace, item: EvalItem, device: torch.device) -> torch.Tensor:
    if item.input_source == "raw_preview":
        tensor = sample["image"]
    elif cfg.input_type in RAW_MODEL_INPUT_TYPES:
        tensor = sample["raw"] if "raw" in sample else sample["image"]
    else:
        tensor = sample["image"]
    return tensor.unsqueeze(0).to(device=device, dtype=torch.float32)


def predict_disp(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_hw: tuple[int, int],
) -> np.ndarray:
    with torch.no_grad():
        pred = model(image)
        pred = pred.float()
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred[:, 0]
        if pred.ndim != 3:
            raise RuntimeError(f"Unexpected prediction shape: {tuple(pred.shape)}")
        if tuple(pred.shape[-2:]) != tuple(target_hw):
            pred = F.interpolate(
                pred[:, None],
                target_hw,
                mode="bilinear",
                align_corners=True,
            )[:, 0]
    return pred[0].detach().cpu().numpy().astype(np.float32, copy=False)


def affine_align_values(target: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    vm = valid & np.isfinite(target) & np.isfinite(pred)
    if int(vm.sum()) < 10:
        return np.full_like(pred, np.nan, dtype=np.float32), {"scale": float("nan"), "shift": float("nan")}
    x = pred[vm].reshape(-1, 1).astype(np.float64)
    y = target[vm].reshape(-1, 1).astype(np.float64)
    A = np.concatenate([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    scale, shift = float(coef[0].item()), float(coef[1].item())
    aligned = pred.astype(np.float64) * scale + shift
    return aligned.astype(np.float32), {"scale": scale, "shift": shift}


def resize_to_shape(values: np.ndarray, shape: tuple[int, int], *, interpolation: int) -> np.ndarray:
    if tuple(values.shape[:2]) == tuple(shape):
        return values
    return cv2.resize(values, (shape[1], shape[0]), interpolation=interpolation)


def fill_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = valid.astype(bool) & np.isfinite(values)
    out = values.copy()
    if not np.any(valid):
        out[~np.isfinite(out)] = 0.0
        return out
    fill = float(np.median(out[valid]))
    out[~valid] = fill
    return out


def eroded_valid(valid: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.erode(valid.astype(np.uint8), kernel, iterations=1).astype(bool)


def sobel_magnitude(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    gx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def depth_edge_map(values: np.ndarray, valid: np.ndarray, percentile: float) -> np.ndarray:
    valid = valid.astype(bool) & np.isfinite(values) & (values > 0)
    edge_valid = eroded_valid(valid)
    if int(edge_valid.sum()) < 100:
        return np.zeros_like(valid, dtype=bool)
    log_values = np.zeros_like(values, dtype=np.float32)
    log_values[valid] = np.log(np.clip(values[valid].astype(np.float32), 1e-6, None))
    log_values = fill_invalid(log_values, valid)
    grad = sobel_magnitude(log_values)
    pool = grad[edge_valid]
    if pool.size == 0:
        return np.zeros_like(valid, dtype=bool)
    threshold = float(np.percentile(pool, percentile))
    if not np.isfinite(threshold):
        return np.zeros_like(valid, dtype=bool)
    return (grad > threshold) & edge_valid


def image_edge_band(rgb_path: str | Path, shape: tuple[int, int], percentile: float, dilate: int) -> np.ndarray:
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read RGB edge source: {rgb_path}")
    if tuple(bgr.shape[:2]) != tuple(shape):
        bgr = cv2.resize(bgr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad = sobel_magnitude(gray)
    threshold = float(np.percentile(grad.reshape(-1), percentile))
    edges = grad > threshold
    dilate = max(0, int(dilate))
    if dilate > 0:
        k = 2 * dilate + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        edges = cv2.dilate(edges.astype(np.uint8), kernel, iterations=1).astype(bool)
    return edges


def distance_to_edges(edge_map: np.ndarray) -> np.ndarray:
    if not np.any(edge_map):
        return np.full(edge_map.shape, np.inf, dtype=np.float32)
    inverse = (~edge_map.astype(bool)).astype(np.uint8)
    return cv2.distanceTransform(inverse, cv2.DIST_L2, 3)


def pseudo_boundary_metrics(
    pred_disp: np.ndarray,
    pseudo_target: np.ndarray,
    *,
    edge_percentile: float,
    tolerance: float,
    truncation: float,
) -> dict[str, float]:
    pseudo_target = np.asarray(pseudo_target, dtype=np.float32)
    valid = np.isfinite(pseudo_target) & (pseudo_target > 0) & np.isfinite(pred_disp)
    if int(valid.sum()) < 100:
        return {
            "pseudo_dbe_acc": float("nan"),
            "pseudo_dbe_comp": float("nan"),
            "pseudo_edge_precision": float("nan"),
            "pseudo_edge_recall": float("nan"),
            "pseudo_edge_f1": float("nan"),
            "pseudo_pred_edge_density": float("nan"),
            "pseudo_target_edge_density": float("nan"),
        }

    pred_aligned, _ = affine_align_values(pseudo_target, pred_disp, valid)
    pred_valid = valid & np.isfinite(pred_aligned) & (pred_aligned > 0)
    pred_edges = depth_edge_map(pred_aligned, pred_valid, edge_percentile)
    target_edges = depth_edge_map(pseudo_target, valid, edge_percentile)

    if not np.any(pred_edges) or not np.any(target_edges):
        return {
            "pseudo_dbe_acc": float("nan"),
            "pseudo_dbe_comp": float("nan"),
            "pseudo_edge_precision": float("nan"),
            "pseudo_edge_recall": float("nan"),
            "pseudo_edge_f1": float("nan"),
            "pseudo_pred_edge_density": float(np.mean(pred_edges)),
            "pseudo_target_edge_density": float(np.mean(target_edges)),
        }

    dist_to_target = distance_to_edges(target_edges)
    dist_to_pred = distance_to_edges(pred_edges)
    acc = float(np.mean(np.minimum(dist_to_target[pred_edges], truncation)))
    comp = float(np.mean(np.minimum(dist_to_pred[target_edges], truncation)))
    precision = float(np.mean(dist_to_target[pred_edges] <= tolerance))
    recall = float(np.mean(dist_to_pred[target_edges] <= tolerance))
    denom = precision + recall
    f1 = float(2.0 * precision * recall / denom) if denom > 0 else 0.0
    return {
        "pseudo_dbe_acc": acc,
        "pseudo_dbe_comp": comp,
        "pseudo_edge_precision": precision,
        "pseudo_edge_recall": recall,
        "pseudo_edge_f1": f1,
        "pseudo_pred_edge_density": float(np.mean(pred_edges)),
        "pseudo_target_edge_density": float(np.mean(target_edges)),
    }


def edge_band_sparse_metrics(
    gt_depth: np.ndarray,
    aligned_depth: np.ndarray,
    valid_mask: np.ndarray,
    edge_band: np.ndarray,
    *,
    min_points: int,
) -> dict[str, float]:
    vm = (
        valid_mask.astype(bool)
        & edge_band.astype(bool)
        & np.isfinite(gt_depth)
        & np.isfinite(aligned_depth)
        & (gt_depth > 0)
        & (aligned_depth > 0)
    )
    count = int(vm.sum())
    if count < min_points:
        return {
            "image_edge_band_abs_rel": float("nan"),
            "image_edge_band_d1": float("nan"),
            "image_edge_band_points": float(count),
            "image_edge_band_coverage": float(count / max(int(valid_mask.sum()), 1)),
        }

    gt = gt_depth[vm].astype(np.float64)
    pred = aligned_depth[vm].astype(np.float64)
    thresh = np.maximum(gt / pred, pred / gt)
    return {
        "image_edge_band_abs_rel": float(np.mean(np.abs(pred - gt) / gt)),
        "image_edge_band_d1": float(np.mean(thresh < 1.25)),
        "image_edge_band_points": float(count),
        "image_edge_band_coverage": float(count / max(int(valid_mask.sum()), 1)),
    }


def finite_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for record in records for key in record["metrics"]})
    summary = {key: finite_mean([record["metrics"].get(key, float("nan")) for record in records]) for key in keys}
    summary["n"] = float(len(records))
    summary["edge_band_sample_count"] = float(
        sum(math.isfinite(float(record["metrics"].get("image_edge_band_abs_rel", float("nan")))) for record in records)
    )
    summary["pseudo_boundary_sample_count"] = float(
        sum(math.isfinite(float(record["metrics"].get("pseudo_dbe_acc", float("nan")))) for record in records)
    )
    return summary


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    fieldnames = [
        "item_id",
        "label",
        "n",
        "abs_rel",
        "d1",
        "pseudo_dbe_acc",
        "pseudo_dbe_comp",
        "pseudo_edge_f1",
        "pseudo_edge_precision",
        "pseudo_edge_recall",
        "image_edge_band_abs_rel",
        "image_edge_band_d1",
        "image_edge_band_points",
        "image_edge_band_coverage",
        "edge_band_sample_count",
        "pseudo_boundary_sample_count",
        "checkpoint",
        "config",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_per_sample_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    metric_keys = sorted({key for record in records for key in record["metrics"]})
    fieldnames = ["item_id", "label", "index", "sample_name"] + metric_keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "item_id": record["item_id"],
                    "label": record["label"],
                    "index": record["index"],
                    "sample_name": record["sample_name"],
                    **record["metrics"],
                }
            )


def evaluate_item(
    item: EvalItem,
    cfg: SimpleNamespace,
    pseudo_rows: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = torch.device(args.device)
    dataset = dataset_for_item(cfg, item, args.split)
    max_samples = len(dataset) if args.max_samples is None else min(int(args.max_samples), len(dataset))

    print(f"[item] {item.label} n={max_samples} input_type={cfg.input_type} source={item.input_source}", flush=True)
    model = load_model_for_item(cfg, item, device)

    records: list[dict[str, Any]] = []
    for idx in range(max_samples):
        sample = dataset[idx]
        name = sample_name(sample)
        gt_depth = sample["depth"].detach().cpu().numpy().astype(np.float32, copy=False)
        valid_mask = sample["valid_mask"].detach().cpu().numpy().astype(bool)
        target_hw = tuple(int(v) for v in gt_depth.shape[-2:])
        image = tensor_from_sample(sample, cfg, item, device)
        pred_disp = predict_disp(model, image, target_hw)

        aligned_depth, _ = affine_align_disp(gt_depth, pred_disp, valid_mask)
        sparse_metrics = compute_metrics(
            gt_depth,
            aligned_depth,
            valid_mask,
            min_depth=float(cfg.min_depth),
            max_depth=float(cfg.max_depth),
        )
        if sparse_metrics is None:
            continue

        pseudo_row = pseudo_rows.get(name)
        if pseudo_row is None:
            raise KeyError(f"Missing pseudo manifest row for sample {name}")
        pseudo_target = np.load(pseudo_row["pseudo_depth_npy"]).astype(np.float32, copy=False)
        pseudo_target = resize_to_shape(pseudo_target, target_hw, interpolation=cv2.INTER_LINEAR)

        pdb_metrics = pseudo_boundary_metrics(
            pred_disp,
            pseudo_target,
            edge_percentile=float(args.edge_percentile),
            tolerance=float(args.boundary_tolerance),
            truncation=float(args.dbe_truncation),
        )
        edge_band = image_edge_band(
            pseudo_row["rgb_path"],
            target_hw,
            percentile=float(args.image_edge_percentile),
            dilate=int(args.image_edge_dilate),
        )
        edge_metrics = edge_band_sparse_metrics(
            gt_depth,
            aligned_depth,
            valid_mask,
            edge_band,
            min_points=int(args.min_edge_points),
        )

        metrics = {
            "abs_rel": float(sparse_metrics["abs_rel"]),
            "d1": float(sparse_metrics["d1"]),
            "rmse": float(sparse_metrics["rmse"]),
            "silog": float(sparse_metrics["silog"]),
            **pdb_metrics,
            **edge_metrics,
        }
        records.append(
            {
                "item_id": item.item_id,
                "label": item.label,
                "index": idx,
                "sample_name": name,
                "metrics": metrics,
            }
        )

        if args.print_every > 0 and (idx + 1) % int(args.print_every) == 0:
            print(f"  [{idx + 1}/{max_samples}] {item.label}", flush=True)

    if not records:
        raise RuntimeError(f"No valid records for {item.item_id}")

    summary_metrics = aggregate_records(records)
    summary = {
        "item_id": item.item_id,
        "label": item.label,
        "checkpoint": str(item.checkpoint_path) if item.checkpoint_path is not None else str(cfg.pretrained_from),
        "config": str(item.config_path) if item.config_path is not None else str(args.reference_config),
        **summary_metrics,
    }
    print(
        "[done] {label}: abs_rel={abs_rel:.4f} d1={d1:.4f} "
        "pDBE_acc={pseudo_dbe_acc:.3f} pDBE_comp={pseudo_dbe_comp:.3f} "
        "pF1={pseudo_edge_f1:.4f} edge_abs={image_edge_band_abs_rel:.4f} "
        "edge_d1={image_edge_band_d1:.4f}".format(**summary),
        flush=True,
    )
    return summary, records


def main() -> None:
    args = parse_args()
    if args.num_threads is not None:
        torch.set_num_threads(int(args.num_threads))
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pseudo_rows = load_pseudo_manifest(args.pseudo_manifest, args.split)
    items = build_eval_items(args)

    all_summaries: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for item in items:
        cfg = config_for_item(item, args.reference_config)
        summary, records = evaluate_item(item, cfg, pseudo_rows, args)
        all_summaries.append(summary)
        all_records.extend(records)

        save_json(args.output_dir / "summary.json", all_summaries)
        write_summary_csv(args.output_dir / "summary.csv", all_summaries)
        write_per_sample_csv(args.output_dir / "per_sample.csv", all_records)

    meta = {
        "split": args.split,
        "pseudo_manifest": str(args.pseudo_manifest.expanduser().resolve()),
        "edge_percentile": args.edge_percentile,
        "image_edge_percentile": args.image_edge_percentile,
        "image_edge_dilate": args.image_edge_dilate,
        "boundary_tolerance": args.boundary_tolerance,
        "dbe_truncation": args.dbe_truncation,
        "min_edge_points": args.min_edge_points,
        "max_samples": args.max_samples,
        "device": args.device,
    }
    save_json(args.output_dir / "meta.json", meta)
    print(f"[output] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
