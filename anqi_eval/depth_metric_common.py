#!/usr/bin/env python3
"""Shared metric-depth metrics used by relative and direct-depth evaluators."""

import cv2
import numpy as np


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

EDGE_METRIC_KEYS = ("edge_sobel_l1", "edge_overlap_iou")


def sobel_magnitude(values):
    values = np.asarray(values, dtype=np.float32)
    grad_x = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def _fill_invalid_for_sobel(values, valid_mask):
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    output = values.copy()
    if not np.any(valid):
        output[~np.isfinite(output)] = 0.0
        return output
    fill_value = float(np.median(output[valid]))
    output[~valid] = fill_value
    return output


def _compute_edge_metrics(gt, eval_depth, valid_mask):
    unavailable = {key: float("nan") for key in EDGE_METRIC_KEYS}
    if gt.ndim != 2 or eval_depth.ndim != 2 or valid_mask.ndim != 2:
        return unavailable

    edge_valid_base = (
        valid_mask.astype(bool)
        & np.isfinite(eval_depth)
        & np.isfinite(gt)
        & (eval_depth > 0)
        & (gt > 0)
    )
    if int(edge_valid_base.sum()) < 1000:
        return unavailable

    kernel = np.ones((3, 3), dtype=np.uint8)
    edge_valid = cv2.erode(edge_valid_base.astype(np.uint8), kernel, iterations=1).astype(bool)
    if int(edge_valid.sum()) < 1000:
        return unavailable

    pred_disp = np.full(eval_depth.shape, np.nan, dtype=np.float32)
    gt_disp = np.full(gt.shape, np.nan, dtype=np.float32)
    pred_disp[edge_valid_base] = 1.0 / eval_depth[edge_valid_base].astype(np.float32)
    gt_disp[edge_valid_base] = 1.0 / gt[edge_valid_base].astype(np.float32)

    pred_disp = _fill_invalid_for_sobel(pred_disp, edge_valid_base)
    gt_disp = _fill_invalid_for_sobel(gt_disp, edge_valid_base)
    g_pred = sobel_magnitude(pred_disp)
    g_gt = sobel_magnitude(gt_disp)

    pred_edges = g_pred[edge_valid]
    gt_edges = g_gt[edge_valid]
    if pred_edges.size < 1000 or gt_edges.size < 1000:
        return unavailable

    edge_sobel_l1 = float(np.mean(np.abs(pred_edges - gt_edges)))
    thr = float(np.percentile(gt_edges, 95))
    if not np.isfinite(thr):
        return {"edge_sobel_l1": edge_sobel_l1, "edge_overlap_iou": float("nan")}

    pred_binary = pred_edges > thr
    gt_binary = gt_edges > thr
    union = np.logical_or(pred_binary, gt_binary).sum()
    if int(union) == 0:
        edge_iou = float("nan")
    else:
        edge_iou = float(np.logical_and(pred_binary, gt_binary).sum() / union)
    return {"edge_sobel_l1": edge_sobel_l1, "edge_overlap_iou": edge_iou}


def compute_metrics(gt, aligned_depth, valid_mask, min_depth=None, max_depth=None):
    eval_depth = aligned_depth.astype(np.float64, copy=True)
    if min_depth is not None or max_depth is not None:
        lo = -np.inf if min_depth is None else float(min_depth)
        hi = np.inf if max_depth is None else float(max_depth)
        finite = np.isfinite(eval_depth)
        eval_depth[finite] = np.clip(eval_depth[finite], lo, hi)

    vm = valid_mask & np.isfinite(eval_depth) & (eval_depth > 0) & (gt > 0)
    if vm.sum() < 10:
        return None

    g = gt[vm].astype(np.float64)
    p = eval_depth[vm].astype(np.float64)
    diff = p - g
    diff_log = np.log(p) - np.log(g)
    thresh = np.maximum(g / p, p / g)

    metrics = {
        "abs_rel": float(np.mean(np.abs(diff) / g)),
        "sq_rel": float(np.mean(diff ** 2 / g)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "rmse_log": float(np.sqrt(np.mean(diff_log ** 2))),
        "log10": float(np.mean(np.abs(np.log10(p) - np.log10(g)))),
        "silog": float(np.sqrt(max(np.mean(diff_log ** 2) - 0.5 * np.mean(diff_log) ** 2, 0.0))),
        "silog_x100": float(
            np.sqrt(max(np.mean(diff_log ** 2) - 0.5 * np.mean(diff_log) ** 2, 0.0))
            * 100.0
        ),
        "d1": float(np.mean(thresh < 1.25)),
        "d2": float(np.mean(thresh < 1.25 ** 2)),
        "d3": float(np.mean(thresh < 1.25 ** 3)),
        "valid_eval_pixels": int(vm.sum()),
    }
    metrics.update(_compute_edge_metrics(gt, eval_depth, vm))
    return metrics


def format_metric_value(value, spec=".4f"):
    if value is None:
        return "n/a"
    value = float(value)
    if not np.isfinite(value):
        return "n/a"
    return format(value, spec)
