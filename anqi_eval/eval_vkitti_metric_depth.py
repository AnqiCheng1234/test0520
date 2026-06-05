#!/usr/bin/env python3
"""Direct metric-depth evaluation for official Depth Anything V2 VKITTI checkpoints."""

import argparse
import inspect
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.depth_metric_common import MODEL_CONFIGS, compute_metrics, format_metric_value


EVAL_PROTOCOL = "metric_direct_no_alignment_pred_clipped"
GT_DEPTH_SCALE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METRIC_KEYS = (
    "abs_rel",
    "sq_rel",
    "rmse",
    "rmse_log",
    "log10",
    "silog",
    "silog_x100",
    "d1",
    "d2",
    "d3",
    "edge_sobel_l1",
    "edge_overlap_iou",
)


def parse_args():
    parser = argparse.ArgumentParser("VKITTI direct metric-depth evaluator for official DAv2 metric checkpoints")
    parser.add_argument("--metric-package-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", required=True, choices=list(MODEL_CONFIGS))
    parser.add_argument("--model-max-depth", required=True, type=float)
    parser.add_argument("--split", required=True)
    parser.add_argument("--eval-gt-min-depth", required=True, type=float)
    parser.add_argument("--eval-gt-max-depth", required=True, type=float)
    parser.add_argument("--pred-clip-min-depth", required=True, type=float)
    parser.add_argument("--pred-clip-max-depth", required=True, type=float)
    parser.add_argument("--input-size", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--viz-max-samples", type=int, default=16)
    return parser.parse_args()


def path_is_relative_to(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(data), f, indent=2, sort_keys=True)
        f.write("\n")


def resolve_config(args):
    metric_package_root = Path(args.metric_package_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    split = Path(args.split).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    model_config = dict(MODEL_CONFIGS[args.encoder])
    model_config["max_depth"] = float(args.model_max_depth)

    return {
        "metric_package_root": str(metric_package_root),
        "checkpoint": str(checkpoint),
        "encoder": args.encoder,
        "model_config": model_config,
        "model_max_depth": float(args.model_max_depth),
        "split": str(split),
        "eval_gt_min_depth": float(args.eval_gt_min_depth),
        "eval_gt_max_depth": float(args.eval_gt_max_depth),
        "pred_clip_min_depth": float(args.pred_clip_min_depth),
        "pred_clip_max_depth": float(args.pred_clip_max_depth),
        "input_size": int(args.input_size),
        "max_samples": args.max_samples,
        "num_workers": int(args.num_workers),
        "save_viz": bool(args.save_viz),
        "save_npy": bool(args.save_npy),
        "viz_max_samples": int(args.viz_max_samples),
        "output_dir": str(output_dir),
        "eval_protocol": EVAL_PROTOCOL,
        "depth_unit": "meters",
        "gt_depth_scale": GT_DEPTH_SCALE,
        "metrics_helper": "anqi_eval/depth_metric_common.py",
        "device": DEVICE,
    }


def validate_config(config):
    metric_package_root = Path(config["metric_package_root"])
    checkpoint = Path(config["checkpoint"])
    split = Path(config["split"])
    dpt_path = metric_package_root / "depth_anything_v2_metric" / "dpt.py"

    errors = []
    if not metric_package_root.is_dir():
        errors.append(f"Official metric package root does not exist: {metric_package_root}")
    if not dpt_path.is_file():
        errors.append(f"Official metric dpt.py does not exist: {dpt_path}")
    if not checkpoint.is_file():
        errors.append(f"Checkpoint does not exist: {checkpoint}")
    if not split.is_file():
        errors.append(f"Split file does not exist: {split}")
    if config["eval_gt_min_depth"] >= config["eval_gt_max_depth"]:
        errors.append("eval GT min depth must be smaller than eval GT max depth")
    if config["pred_clip_min_depth"] >= config["pred_clip_max_depth"]:
        errors.append("prediction clip min depth must be smaller than prediction clip max depth")
    if config["input_size"] <= 0:
        errors.append("input size must be positive")
    if config["max_samples"] is not None and int(config["max_samples"]) <= 0:
        errors.append("max samples must be positive when provided")
    if int(config["num_workers"]) != 0:
        errors.append("num-workers is accepted for CLI compatibility, but this evaluator is sequential; use 0")
    if errors:
        raise ValueError("\n".join(errors))


def import_metric_model(metric_package_root):
    if "depth_anything_v2" in sys.modules:
        raise RuntimeError("Local relative package depth_anything_v2 was imported before metric model import")

    root_str = str(metric_package_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from depth_anything_v2_metric.dpt import DepthAnythingV2

    signature = inspect.signature(DepthAnythingV2)
    if "max_depth" not in signature.parameters:
        raise RuntimeError("Imported DepthAnythingV2 does not accept max_depth")
    if not DepthAnythingV2.__module__.startswith("depth_anything_v2_metric"):
        raise RuntimeError(f"Imported unexpected DepthAnythingV2 module: {DepthAnythingV2.__module__}")

    module = sys.modules[DepthAnythingV2.__module__]
    module_path = Path(module.__file__).resolve()
    if not path_is_relative_to(module_path, metric_package_root):
        raise RuntimeError(f"Imported metric model from outside vendored root: {module_path}")
    if "depth_anything_v2" in sys.modules:
        raise RuntimeError("Metric evaluator unexpectedly imported local relative depth_anything_v2")
    return DepthAnythingV2


def load_model(config):
    metric_package_root = Path(config["metric_package_root"])
    DepthAnythingV2 = import_metric_model(metric_package_root)
    model = DepthAnythingV2(**config["model_config"])
    state_dict = torch.load(config["checkpoint"], map_location="cpu")
    load_result = model.load_state_dict(state_dict, strict=True)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint load was not strict clean: missing={missing}, unexpected={unexpected}")
    return model.to(DEVICE).eval()


def iter_split(split_path, max_samples=None):
    with Path(split_path).open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if max_samples is not None:
        lines = lines[: int(max_samples)]
    for index, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Malformed split line {index + 1}: expected image and depth paths, got {line!r}")
        yield index, Path(parts[0]), Path(parts[1])


def sample_name_from_path(image_path):
    parts = image_path.parts
    if "VKITTI2" in parts:
        start = parts.index("VKITTI2")
        tail = list(parts[start + 1 :])
        keep = [part for part in tail if part not in {"rgb", "frames"}]
        return "_".join(keep).replace(".jpg", "")
    return image_path.stem


def read_sample(image_path, depth_path):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read VKITTI RGB image: {image_path}")
    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"Failed to read VKITTI depth PNG: {depth_path}")
    gt = depth.astype(np.float32) * GT_DEPTH_SCALE
    return bgr, gt


def compute_stats(values, mask=None):
    array = np.asarray(values, dtype=np.float64)
    if mask is not None:
        array = array[np.asarray(mask, dtype=bool)]
    else:
        array = array[np.isfinite(array)]
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"min": None, "max": None, "mean": None, "count": 0}
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "count": int(array.size),
    }


def update_stats_aggregate(aggregate, stats):
    if stats["count"] <= 0:
        return
    aggregate["count"] += int(stats["count"])
    aggregate["sum"] += float(stats["mean"]) * int(stats["count"])
    value_min = float(stats["min"])
    value_max = float(stats["max"])
    aggregate["min"] = value_min if aggregate["min"] is None else min(aggregate["min"], value_min)
    aggregate["max"] = value_max if aggregate["max"] is None else max(aggregate["max"], value_max)


def finalize_stats_aggregate(aggregate):
    mean = float(aggregate["sum"] / aggregate["count"]) if aggregate["count"] > 0 else None
    return {"min": aggregate["min"], "max": aggregate["max"], "mean": mean, "count": aggregate["count"]}


def metric_means(records):
    means = {}
    counts = {}
    for key in METRIC_KEYS:
        values = [
            float(record["metrics"][key])
            for record in records
            if record.get("metrics") and key in record["metrics"] and np.isfinite(float(record["metrics"][key]))
        ]
        means[key] = float(np.mean(values)) if values else float("nan")
        counts[key] = len(values)
    valid_pixels = [
        int(record["metrics"]["valid_eval_pixels"])
        for record in records
        if record.get("metrics") and "valid_eval_pixels" in record["metrics"]
    ]
    means["avg_valid_eval_pixels"] = float(np.mean(valid_pixels)) if valid_pixels else float("nan")
    counts["avg_valid_eval_pixels"] = len(valid_pixels)
    return means, counts


def apply_matplotlib_colormap(values, colormap_name):
    from matplotlib import colormaps

    colored = colormaps[colormap_name](values)
    rgb = (colored[..., :3] * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def add_panel_label(image, label):
    output = image.copy()
    cv2.rectangle(output, (0, 0), (210, 34), (0, 0, 0), thickness=-1)
    cv2.putText(
        output,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return output


def render_depth(depth, valid_mask, min_depth, max_depth):
    clipped = np.clip(depth.astype(np.float32), float(min_depth), float(max_depth))
    norm = (clipped - float(min_depth)) / max(float(max_depth) - float(min_depth), 1e-6)
    norm[~valid_mask] = 0.0
    image = apply_matplotlib_colormap(norm, "Spectral_r")
    image[~valid_mask] = 0
    return image


def render_abs_error(gt, pred, valid_mask, min_depth, max_depth):
    pred_eval = pred.astype(np.float32, copy=True)
    pred_eval[np.isfinite(pred_eval)] = np.clip(
        pred_eval[np.isfinite(pred_eval)],
        float(min_depth),
        float(max_depth),
    )
    error = np.abs(pred_eval - gt.astype(np.float32))
    error_valid = valid_mask & np.isfinite(error)
    norm = np.zeros(error.shape, dtype=np.float32)
    if np.any(error_valid):
        vmax = float(np.percentile(error[error_valid], 95))
        vmax = max(vmax, 1e-6)
        norm[error_valid] = np.clip(error[error_valid] / vmax, 0.0, 1.0)
    image = apply_matplotlib_colormap(norm, "magma")
    image[~error_valid] = 0
    return image


def write_optional_outputs(output_dir, sample_name, bgr, gt, pred, valid, args, saved_viz_count):
    if args.save_npy:
        npy_dir = output_dir / "npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_dir / f"{sample_name}_pred.npy", pred.astype(np.float32))

    if args.save_viz and saved_viz_count < args.viz_max_samples:
        viz_dir = output_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        pred_viz = render_depth(pred, np.isfinite(pred), args.pred_clip_min_depth, args.pred_clip_max_depth)
        gt_viz = render_depth(gt, valid, args.eval_gt_min_depth, args.eval_gt_max_depth)
        error_viz = render_abs_error(
            gt,
            pred,
            valid,
            args.pred_clip_min_depth,
            args.pred_clip_max_depth,
        )
        panel = np.concatenate(
            [
                add_panel_label(bgr, "RGB"),
                add_panel_label(gt_viz, "GT depth"),
                add_panel_label(pred_viz, "Pred depth"),
                add_panel_label(error_viz, "Abs error"),
            ],
            axis=1,
        )
        cv2.imwrite(str(viz_dir / f"{sample_name}.jpg"), panel)
        return saved_viz_count + 1
    return saved_viz_count


def evaluate(config, args, model):
    output_dir = Path(config["output_dir"])
    per_sample_path = output_dir / "per_sample_metrics.jsonl"
    valid_records = []
    sample_count = 0
    invalid_sample_count = 0
    saved_viz_count = 0
    pred_agg = {"min": None, "max": None, "sum": 0.0, "count": 0}
    gt_agg = {"min": None, "max": None, "sum": 0.0, "count": 0}

    with per_sample_path.open("w", encoding="utf-8") as f:
        for sample_index, image_path, depth_path in iter_split(config["split"], config["max_samples"]):
            sample_count += 1
            sample_name = sample_name_from_path(image_path)
            bgr, gt = read_sample(image_path, depth_path)
            valid = (
                np.isfinite(gt)
                & (gt >= config["eval_gt_min_depth"])
                & (gt <= config["eval_gt_max_depth"])
            )

            with torch.no_grad():
                pred = model.infer_image(bgr, config["input_size"])
            pred = np.asarray(pred, dtype=np.float32)

            if pred.shape != gt.shape:
                raise ValueError(f"{sample_name}: prediction shape {pred.shape} does not match GT shape {gt.shape}")
            if not np.all(np.isfinite(pred)):
                raise ValueError(f"{sample_name}: prediction contains NaN/Inf")

            pred_stats = compute_stats(pred)
            gt_stats = compute_stats(gt, valid)
            update_stats_aggregate(pred_agg, pred_stats)
            update_stats_aggregate(gt_agg, gt_stats)

            metrics = None
            status = "ok"
            skip_reason = None
            if int(valid.sum()) < 10:
                status = "skipped"
                skip_reason = "fewer_than_10_valid_gt_pixels"
                invalid_sample_count += 1
            else:
                metrics = compute_metrics(
                    gt,
                    pred,
                    valid,
                    min_depth=config["pred_clip_min_depth"],
                    max_depth=config["pred_clip_max_depth"],
                )
                if metrics is None:
                    status = "skipped"
                    skip_reason = "compute_metrics_returned_none"
                    invalid_sample_count += 1
                else:
                    valid_records.append({"metrics": metrics})

            saved_viz_count = write_optional_outputs(
                output_dir,
                sample_name,
                bgr,
                gt,
                pred,
                valid,
                args,
                saved_viz_count,
            )

            record = {
                "sample_index": int(sample_index),
                "sample_name": sample_name,
                "image_path": str(image_path),
                "depth_path": str(depth_path),
                "valid_pixels": int(valid.sum()),
                "metrics": metrics,
                "pred_min": pred_stats["min"],
                "pred_max": pred_stats["max"],
                "pred_mean": pred_stats["mean"],
                "gt_min": gt_stats["min"],
                "gt_max": gt_stats["max"],
                "gt_mean": gt_stats["mean"],
                "status": status,
                "skip_reason": skip_reason,
            }
            f.write(json.dumps(jsonable(record), sort_keys=True) + "\n")

            if sample_count % 25 == 0:
                print(f"[{sample_count}] valid={len(valid_records)} skipped={invalid_sample_count}", flush=True)

    means, counts = metric_means(valid_records)
    pred_summary = finalize_stats_aggregate(pred_agg)
    gt_summary = finalize_stats_aggregate(gt_agg)
    summary = {
        "sample_count": sample_count,
        "valid_sample_count": len(valid_records),
        "invalid_sample_count": invalid_sample_count,
        "metrics": means,
        "metric_counts": counts,
        "prediction_min_meter": pred_summary["min"],
        "prediction_max_meter": pred_summary["max"],
        "prediction_mean_meter": pred_summary["mean"],
        "prediction_pixel_count": pred_summary["count"],
        "gt_min_meter": gt_summary["min"],
        "gt_max_meter": gt_summary["max"],
        "gt_mean_meter": gt_summary["mean"],
        "gt_valid_pixel_count": gt_summary["count"],
    }
    return summary


def write_summary_md(output_dir, config, metrics_summary):
    metrics = metrics_summary["metrics"]
    lines = [
        "# VKITTI Metric DAv2-S Official VITS Evaluation",
        "",
        "## Protocol",
        "",
        f"- eval_protocol: {config['eval_protocol']}",
        f"- depth_unit: {config['depth_unit']}",
        f"- checkpoint: {config['checkpoint']}",
        f"- split: {config['split']}",
        f"- model_max_depth: {config['model_max_depth']}",
        f"- eval_gt_depth: [{config['eval_gt_min_depth']}, {config['eval_gt_max_depth']}]",
        f"- pred_clip_depth: [{config['pred_clip_min_depth']}, {config['pred_clip_max_depth']}]",
        "",
        "## Results",
        "",
        f"- sample_count: {metrics_summary['sample_count']}",
        f"- valid_sample_count: {metrics_summary['valid_sample_count']}",
        f"- invalid_sample_count: {metrics_summary['invalid_sample_count']}",
        f"- abs_rel: {format_metric_value(metrics.get('abs_rel'))}",
        f"- rmse: {format_metric_value(metrics.get('rmse'))}",
        f"- silog: {format_metric_value(metrics.get('silog'))}",
        f"- silog_x100: {format_metric_value(metrics.get('silog_x100'), '.2f')}",
        f"- d1: {format_metric_value(metrics.get('d1'))}",
        f"- d2: {format_metric_value(metrics.get('d2'))}",
        f"- d3: {format_metric_value(metrics.get('d3'))}",
        f"- edge_sobel_l1: {format_metric_value(metrics.get('edge_sobel_l1'))}",
        f"- edge_overlap_iou: {format_metric_value(metrics.get('edge_overlap_iou'))}",
        "",
    ]
    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    config = resolve_config(args)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved_config.json", config)

    try:
        validate_config(config)
        model = load_model(config)
        metrics_summary = evaluate(config, args, model)
    except Exception as exc:
        error_path = output_dir / "error.json"
        write_json(error_path, {"error": str(exc), "type": type(exc).__name__})
        raise

    write_json(output_dir / "metrics.json", metrics_summary)
    write_summary_md(output_dir, config, metrics_summary)
    print(f"Saved metrics to {output_dir / 'metrics.json'}", flush=True)
    print(f"Saved summary to {output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
