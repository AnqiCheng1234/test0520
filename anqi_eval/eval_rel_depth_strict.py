#!/usr/bin/env python3
"""
DAv2 relative depth evaluation with stricter benchmark-style handling.

Compared with eval_rel_depth.py, this version keeps the same high-level
protocol (per-image affine alignment in disparity space) but tightens a few
details while leaving KITTI behavior aligned with the original script:
  1. Invalid aligned disparity is tracked explicitly instead of being silently
     hidden.
  2. Prediction / GT / mask shapes are checked before evaluation.
  3. SILog is reported in both raw and x100 conventions.
"""
import argparse
from datetime import date
import glob
import os
import sys

import cv2
import numpy as np
import torch

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)

sys.path.insert(0, PROJECT_ROOT)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EDGE_METRIC_KEYS = ("edge_sobel_l1", "edge_overlap_iou")

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

NYU_DIR = "/mnt/drive/nyu/nyu_test"
KITTI_BASE = "/mnt/drive/kitti"
KITTI_SPLIT = os.path.join(
    PROJECT_ROOT,
    "metric_depth/dataset/splits/kitti/val.txt",
)
KITTI_MAX_D = 80.0
NUSCENES_BASE = "/mnt/drive/1111_new_works/0000_nuscenes"
ROBOTCAR_BASE = "/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640"
DATASET_DEPTH_RANGES = {
    "nyu": (1e-3, 10.0),
    "kitti": (0.1, KITTI_MAX_D),
    "nuscenes": (0.1, 80.0),
    "robotcar": (0.1, 50.0),
}


def load_model(encoder, checkpoint):
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ModuleNotFoundError as exc:
        missing_pkg = exc.name or "required dependency"
        raise ModuleNotFoundError(
            "Failed to import DepthAnythingV2 dependencies while loading the model. "
            f"Missing package: {missing_pkg}. Install the required runtime packages "
            "for this repo and retry."
        ) from exc

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(DEVICE).eval()


def resolve_checkpoint_path(encoder, checkpoint_arg):
    if checkpoint_arg:
        return checkpoint_arg

    filename = f"depth_anything_v2_{encoder}.pth"
    home_dir = os.path.expanduser("~")
    search_patterns = [
        os.path.join(PROJECT_ROOT, "checkpoints", filename),
        os.path.join(os.getcwd(), "checkpoints", filename),
        os.path.join(home_dir, "checkpoints", filename),
        os.path.join(home_dir, "*", "Depth-Anything-V2", "checkpoints", filename),
        os.path.join(home_dir, "*", "*", "Depth-Anything-V2", "checkpoints", filename),
        os.path.join(home_dir, "*", "checkpoints", filename),
        os.path.join(home_dir, "*", "*", "checkpoints", filename),
        os.path.join(home_dir, "*", "*", "*", "checkpoints", filename),
    ]

    checked = []
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        checked.extend(matches or [pattern])
        for path in matches:
            if os.path.isfile(path):
                return path

    raise FileNotFoundError(
        "Checkpoint not found. Checked these locations for "
        f"{filename}:\n" + "\n".join(f"  - {path}" for path in checked) + "\n"
        "Pass the correct file via --checkpoint /path/to/depth_anything_v2_*.pth."
    )


def apply_runtime_paths(args):
    global NYU_DIR, KITTI_BASE, KITTI_SPLIT, NUSCENES_BASE, ROBOTCAR_BASE

    NYU_DIR = args.nyu_dir
    KITTI_BASE = args.kitti_base
    KITTI_SPLIT = args.kitti_split
    NUSCENES_BASE = args.nuscenes_base
    ROBOTCAR_BASE = args.robotcar_base


def validate_runtime_paths(datasets):
    required_paths = []
    if "nyu" in datasets:
        required_paths.append(("NYU dir", NYU_DIR, os.path.isdir))
    if "kitti" in datasets:
        required_paths.append(("KITTI base", KITTI_BASE, os.path.isdir))
        required_paths.append(("KITTI split", KITTI_SPLIT, os.path.isfile))
    if "nuscenes" in datasets:
        required_paths.append(("nuScenes base", NUSCENES_BASE, os.path.isdir))
    if "robotcar" in datasets:
        required_paths.append(("RobotCar base", ROBOTCAR_BASE, os.path.isdir))

    missing = [f"{label}: {path}" for label, path, check in required_paths if not check(path)]
    if missing:
        joined = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            "Required dataset paths are missing for the requested evaluation:\n"
            f"{joined}\n"
            "Pass the correct paths via CLI flags such as --nyu-dir / --kitti-base."
        )


class EvalLogger:
    def __init__(self, enabled, save_dir):
        self.enabled = enabled
        self.save_dir = save_dir
        self.global_lines = []
        self.dataset_lines = {}

    def emit(self, message="", dataset_name=None):
        print(message, flush=True)
        if not self.enabled:
            return
        self.global_lines.append(message)
        if dataset_name is not None:
            self.dataset_lines.setdefault(dataset_name, []).append(message)

    def write_dataset_report(self, dataset_name):
        if not self.enabled:
            return None

        os.makedirs(self.save_dir, exist_ok=True)
        stamp = date.today().isoformat()
        path = os.path.join(
            self.save_dir,
            f"eval_rel_depth_strict_{dataset_name}_results_{stamp}.txt",
        )
        lines = self.global_lines[:1]
        if lines:
            lines.append("")
        lines.extend(self.dataset_lines.get(dataset_name, []))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")
        return path

    def write_summary_report(self, all_results, counts, checkpoint):
        if not self.enabled or not all_results:
            return None

        os.makedirs(self.save_dir, exist_ok=True)
        stamp = date.today().isoformat()
        path = os.path.join(
            self.save_dir,
            f"eval_rel_depth_strict_summary_{stamp}.md",
        )
        lines = []
        for dataset_name, results in all_results.items():
            lines.append(dataset_name.upper())
            lines.append(f"- n={counts[dataset_name]}")
            lines.append(f"- abs_rel={format_metric_value(results['abs_rel'])}")
            lines.append(f"- rmse={format_metric_value(results['rmse'])}")
            lines.append(f"- silog={format_metric_value(results['silog'])}")
            lines.append(f"- silog_x100={format_metric_value(results['silog_x100'], '.2f')}")
            lines.append(f"- d1={format_metric_value(results['d1'])}")
            lines.append(f"- d2={format_metric_value(results['d2'])}")
            lines.append(f"- d3={format_metric_value(results['d3'])}")
            lines.append(f"- edge_sobel_l1={format_metric_value(results.get('edge_sobel_l1'))}")
            lines.append(f"- edge_overlap_iou={format_metric_value(results.get('edge_overlap_iou'))}")
            lines.append(f"- invalid_align={format_metric_value(results['avg_invalid_aligned_ratio'])}")
            lines.append("")

        lines.append("Notes")
        lines.append(f"- Script: {os.path.abspath(__file__)}")
        lines.append(f"- Checkpoint: {checkpoint}")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")
        return path


def check_sample_shapes(pred_disp, gt_depth, valid_mask, sample_name):
    if pred_disp.shape != gt_depth.shape or pred_disp.shape != valid_mask.shape:
        raise ValueError(
            f"{sample_name}: shape mismatch pred={pred_disp.shape} gt={gt_depth.shape} mask={valid_mask.shape}"
        )


def affine_align_disp(gt_depth, pred_disp, valid_mask):
    """
    Fit s * pred_disp + t ~= 1 / gt_depth on valid pixels.

    Returns:
      aligned_depth: inverse of aligned disparity where disparity > 0
      stats: basic fit diagnostics
    """
    gt_disp = np.zeros_like(gt_depth, dtype=np.float64)
    gt_disp[valid_mask] = 1.0 / np.clip(gt_depth[valid_mask], a_min=1e-9, a_max=None)

    y = gt_disp[valid_mask].reshape(-1, 1).astype(np.float64)
    x = pred_disp[valid_mask].reshape(-1, 1).astype(np.float64)
    A = np.concatenate([x, np.ones_like(x)], axis=-1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    scale, shift = float(coef[0].item()), float(coef[1].item())

    aligned_disp = pred_disp.astype(np.float64) * scale + shift
    aligned_depth = np.full(pred_disp.shape, np.nan, dtype=np.float64)
    pos = aligned_disp > 0
    aligned_depth[pos] = 1.0 / aligned_disp[pos]

    invalid_count = int(valid_mask.sum() - np.count_nonzero(valid_mask & pos))
    stats = {
        "scale": scale,
        "shift": shift,
        "invalid_aligned_pixels": invalid_count,
        "invalid_aligned_ratio": float(invalid_count / max(int(valid_mask.sum()), 1)),
    }
    return aligned_depth, stats


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
        "silog_x100": float(np.sqrt(max(np.mean(diff_log ** 2) - 0.5 * np.mean(diff_log) ** 2, 0.0)) * 100.0),
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


def _parse_calib(path):
    data = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                try:
                    data[key.strip()] = np.array([float(x) for x in val.strip().split()])
                except ValueError:
                    pass
    return data


_calib_cache = {}


def get_calib(date):
    if date not in _calib_cache:
        d = f"{KITTI_BASE}/{date}"
        cam = _parse_calib(f"{d}/calib_cam_to_cam.txt")
        velo = _parse_calib(f"{d}/calib_velo_to_cam.txt")
        P = cam["P_rect_02"].reshape(3, 4)
        R = cam["R_rect_00"].reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = velo["R"].reshape(3, 3)
        T[:3, 3] = velo["T"]
        _calib_cache[date] = (P, R, T)
    return _calib_cache[date]


def velo_to_depth(velo_path, P, R_rect00, T_velo_cam, img_h, img_w):
    pts = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
    pts = pts[pts[:, 0] > 0]

    R_rect = np.eye(4)
    R_rect[:3, :3] = R_rect00

    hom = np.ones((len(pts), 4), dtype=np.float32)
    hom[:, :3] = pts[:, :3]

    cam = R_rect @ T_velo_cam @ hom.T
    depth = cam[2]
    valid = depth > 0

    img_pts = P @ cam[:, valid]
    img_pts[:2] /= img_pts[2:3]

    u = np.round(img_pts[0]).astype(int)
    v = np.round(img_pts[1]).astype(int)
    d = depth[valid]

    in_b = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    depth_map = np.zeros((img_h, img_w), dtype=np.float32)
    depth_map[v[in_b], u[in_b]] = d[in_b]
    return depth_map


def iter_nyu(model):
    import h5py

    files = sorted(f for f in os.listdir(NYU_DIR) if f.endswith(".h5"))
    for fname in files:
        with h5py.File(os.path.join(NYU_DIR, fname), "r") as hf:
            rgb = hf["rgb"][()]
            gt = hf["depth"][()]
            mask = hf["mask"][()] > 0.5
        bgr = rgb[::-1].transpose(1, 2, 0).copy()
        pred_disp = model.infer_image(bgr)
        gt = gt.astype(np.float32)
        mask = mask.astype(bool) & np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0)
        yield fname, pred_disp, gt, mask


def iter_kitti(model):
    with open(KITTI_SPLIT) as f:
        lines = f.read().splitlines()

    for line in lines:
        parts = line.split()[0].split("raw_data/")[-1].split("/")
        date, drive, frame = parts[0], parts[1], parts[-1].replace(".png", "")

        rgb_path = f"{KITTI_BASE}/{date}/{drive}/image_02/data/{frame}.jpg"
        velo_path = f"{KITTI_BASE}/{date}/{drive}/velodyne_points/data/{frame}.bin"

        bgr = cv2.imread(rgb_path)
        if bgr is None or not os.path.exists(velo_path):
            continue

        img_h, img_w = bgr.shape[:2]
        P, R, T = get_calib(date)
        gt = velo_to_depth(velo_path, P, R, T, img_h, img_w)
        mask = (gt > 0) & (gt < KITTI_MAX_D)
        pred_disp = model.infer_image(bgr)
        yield f"{date}/{drive}/{frame}", pred_disp, gt, mask


def iter_nuscenes(model):
    timestamps = sorted(
        f.replace(".jpg", "")
        for f in os.listdir(f"{NUSCENES_BASE}/color")
        if f.endswith(".jpg")
    )
    for ts in timestamps:
        bgr = cv2.imread(f"{NUSCENES_BASE}/color/{ts}.jpg")
        gt = np.load(f"{NUSCENES_BASE}/gt/{ts}.npy").astype(np.float32)
        mask = np.isfinite(gt) & (gt > 0.1) & (gt < 80.0)
        pred_disp = model.infer_image(bgr)
        yield ts, pred_disp, gt, mask


def iter_robotcar(model):
    timestamps = sorted(
        f.replace(".png", "")
        for f in os.listdir(f"{ROBOTCAR_BASE}/rgb")
        if f.endswith(".png")
    )
    for ts in timestamps:
        bgr = cv2.imread(f"{ROBOTCAR_BASE}/rgb/{ts}.png")
        gt = np.load(f"{ROBOTCAR_BASE}/gt/{ts}.npy").astype(np.float32)
        mask = np.isfinite(gt) & (gt > 0.1) & (gt < 80.0)
        pred_disp = model.infer_image(bgr)
        yield ts, pred_disp, gt, mask


ITERATORS = {
    "nyu": iter_nyu,
    "kitti": iter_kitti,
    "nuscenes": iter_nuscenes,
    "robotcar": iter_robotcar,
}


def evaluate(model, dataset_name, logger=None):
    accumulator = []
    invalid_ratios = []
    valid_eval_pixels = []
    min_depth, max_depth = DATASET_DEPTH_RANGES[dataset_name]

    for i, (sample_name, pred_disp, gt, mask) in enumerate(ITERATORS[dataset_name](model)):
        check_sample_shapes(pred_disp, gt, mask, sample_name)
        if mask.sum() < 10:
            continue

        aligned_depth, align_stats = affine_align_disp(gt, pred_disp, mask)
        metrics = compute_metrics(
            gt,
            aligned_depth,
            mask,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        if metrics is None:
            continue

        accumulator.append(metrics)
        invalid_ratios.append(align_stats["invalid_aligned_ratio"])
        valid_eval_pixels.append(metrics["valid_eval_pixels"])

        if align_stats["invalid_aligned_ratio"] > 0.01:
            message = (
                f"  [warn] {sample_name}: invalid_aligned_ratio="
                f"{align_stats['invalid_aligned_ratio']:.4f}"
            )
            if logger is None:
                print(message)
            else:
                logger.emit(message, dataset_name=dataset_name)

        if (i + 1) % 50 == 0:
            message = f"  [{i + 1}] ..."
            if logger is None:
                print(message)
            else:
                logger.emit(message, dataset_name=dataset_name)

    if not accumulator:
        return {}, 0

    metric_keys = [k for k in accumulator[0].keys() if k != "valid_eval_pixels"]
    mean = {}
    for key in metric_keys:
        values = [
            float(metrics[key])
            for metrics in accumulator
            if key in metrics and np.isfinite(float(metrics[key]))
        ]
        mean[key] = float(np.mean(values)) if values else float("nan")
    mean["avg_valid_eval_pixels"] = float(np.mean(valid_eval_pixels))
    mean["avg_invalid_aligned_ratio"] = float(np.mean(invalid_ratios))
    return mean, len(accumulator)


def main():
    parser = argparse.ArgumentParser("DAv2 Relative Depth Evaluation (strict disparity-space affine)")
    parser.add_argument("--encoder", default="vitl", choices=list(MODEL_CONFIGS))
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["nyu", "kitti", "nuscenes", "robotcar", "all"],
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--nyu-dir", default=os.environ.get("DAV2_NYU_DIR", NYU_DIR))
    parser.add_argument("--kitti-base", default=os.environ.get("DAV2_KITTI_BASE", KITTI_BASE))
    parser.add_argument("--kitti-split", default=os.environ.get("DAV2_KITTI_SPLIT", KITTI_SPLIT))
    parser.add_argument(
        "--nuscenes-base",
        default=os.environ.get("DAV2_NUSCENES_BASE", NUSCENES_BASE),
    )
    parser.add_argument(
        "--robotcar-base",
        default=os.environ.get("DAV2_ROBOTCAR_BASE", ROBOTCAR_BASE),
    )
    parser.add_argument("--save-dir", default=os.environ.get("DAV2_EVAL_SAVE_DIR", FILE_DIR))
    parser.add_argument("--no-save-results", action="store_true")
    args = parser.parse_args()

    datasets = ["nyu", "kitti", "nuscenes", "robotcar"] if args.dataset == "all" else [args.dataset]
    logger = EvalLogger(enabled=not args.no_save_results, save_dir=args.save_dir)
    try:
        args.checkpoint = resolve_checkpoint_path(
            args.encoder,
            args.checkpoint or os.environ.get("DAV2_CHECKPOINT"),
        )
        apply_runtime_paths(args)
        validate_runtime_paths(datasets)

        logger.emit(f"Loading {args.encoder} from {args.checkpoint} ...")
        model = load_model(args.encoder, args.checkpoint)
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    all_results = {}
    all_counts = {}
    for ds in datasets:
        logger.emit(f"\n=== {ds.upper()} ===", dataset_name=ds)
        results, n = evaluate(model, ds, logger=logger)
        if not results:
            logger.emit("  No valid samples.", dataset_name=ds)
            continue
        all_results[ds] = results
        all_counts[ds] = n
        logger.emit(
            f"  n={n}  abs_rel={format_metric_value(results['abs_rel'])}  "
            f"rmse={format_metric_value(results['rmse'])}  "
            f"silog={format_metric_value(results['silog'])}  "
            f"silog_x100={format_metric_value(results['silog_x100'], '.2f')}  "
            f"d1={format_metric_value(results['d1'])}  d2={format_metric_value(results['d2'])}  "
            f"d3={format_metric_value(results['d3'])}  "
            f"edge_l1={format_metric_value(results.get('edge_sobel_l1'))}  "
            f"edge_iou={format_metric_value(results.get('edge_overlap_iou'))}  "
            f"invalid_align={format_metric_value(results['avg_invalid_aligned_ratio'])}",
            dataset_name=ds,
        )
        logger.write_dataset_report(ds)

    if len(all_results) > 1:
        logger.emit("\n" + "=" * 104)
        logger.emit(
            f"{'Dataset':<12} {'abs_rel':>8} {'rmse':>8} {'silog':>8} {'silog*100':>10} "
            f"{'d1':>7} {'d2':>7} {'d3':>7} {'edge_l1':>9} {'edge_iou':>9} {'inv_align':>10}"
        )
        logger.emit("-" * 104)
        for ds, r in all_results.items():
            logger.emit(
                f"{ds:<12} {format_metric_value(r['abs_rel']):>8} {format_metric_value(r['rmse']):>8} "
                f"{format_metric_value(r['silog']):>8} {format_metric_value(r['silog_x100'], '.2f'):>10} "
                f"{format_metric_value(r['d1']):>7} {format_metric_value(r['d2']):>7} "
                f"{format_metric_value(r['d3']):>7} {format_metric_value(r.get('edge_sobel_l1')):>9} "
                f"{format_metric_value(r.get('edge_overlap_iou')):>9} "
                f"{format_metric_value(r['avg_invalid_aligned_ratio']):>10}"
            )

    summary_path = logger.write_summary_report(all_results, all_counts, args.checkpoint)
    if logger.enabled:
        for ds in all_results:
            dataset_path = os.path.join(
                args.save_dir,
                f"eval_rel_depth_strict_{ds}_results_{date.today().isoformat()}.txt",
            )
            logger.emit(f"Saved {ds} results to {dataset_path}")
        if summary_path is not None:
            logger.emit(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
