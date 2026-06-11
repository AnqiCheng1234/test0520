#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import spearmanr


DEFAULT_MANIFEST = (
    "/mnt/drive/3333_raw/seeing_through_fog/"
    "pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/"
    "stf_rgb_lut_manifest_6216.csv"
)
DEFAULT_OUTPUT_PARENT = "/mnt/drive/3333_raw/seeing_through_fog"
DEFAULT_PRIORDA_REPO = "/home/caq/6666_raw/dav2_raw_0603/codex_tmp/Prior-Depth-Anything-review"
TEMP_MARKERS = ("smoke", "debug", "tmp", "codex_smoke")
OUTPUT_DIRS = (
    "dense_depth_priorda_raw",
    "dense_depth_priorda_lidar_overwrite",
    "dense_depth_priorda_lidar_gated_overwrite",
    "dav2_geometric_proxy",
    "sparse_lidar_prior",
    "sparse_lidar_mask",
    "geometric_meta",
    "priorda_meta",
    "qa_meta",
    "debug_vis",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate STF dense pseudo depth labels with PriorDA + DAv2-L geometric proxy."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--preview-contact-sheet", action="store_true")
    parser.add_argument("--preview-approved", default="")
    parser.add_argument("--allow-existing-output-root", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--z-min", type=float, default=1.0)
    parser.add_argument("--z-max", type=float, default=80.0)
    parser.add_argument("--r-percentile-low", type=float, default=0.1)
    parser.add_argument("--r-percentile-high", type=float, default=99.9)
    parser.add_argument("--polarity-flip", action="store_true")
    parser.add_argument("--min-sparse-points", type=int, default=100)
    parser.add_argument("--min-holdout-points", type=int, default=50)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--overwrite-rel-threshold", type=float, default=0.10)
    parser.add_argument("--priorda-version", default="1.1")
    parser.add_argument("--frozen-model-size", default="vitb")
    parser.add_argument("--conditioned-model-size", default="vitb")
    parser.add_argument("--mde-dir", default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--priorda-repo", default=DEFAULT_PRIORDA_REPO)
    parser.add_argument("--coarse-only", action="store_true")
    parser.add_argument("--skip-holdout-qa", action="store_true")
    parser.add_argument("--debug-tile-width", type=int, default=320)
    return parser.parse_args()


def path_has_temp_marker(path):
    text = str(path).lower()
    return any(marker in text for marker in TEMP_MARKERS)


def sha1_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_manifest(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"sample_name", "split", "rgb_path", "sparse_depth_path", "pseudo_depth_npy"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(missing)}")
        for row in reader:
            rows.append(
                {
                    "sample_id": row["sample_name"],
                    "split": row["split"],
                    "rgb_path": row["rgb_path"],
                    "sparse_depth_path": row["sparse_depth_path"],
                    "dav2_npy_path": row["pseudo_depth_npy"],
                    "dav2_vis_path": row.get("pseudo_vis_png", ""),
                }
            )
    return rows


def select_rows(rows, sample_ids, splits, max_samples):
    if sample_ids:
        wanted = set(sample_ids)
        rows = [row for row in rows if row["sample_id"] in wanted]
        missing = sorted(wanted - {row["sample_id"] for row in rows})
        if missing:
            raise ValueError(f"sample_id not found in manifest: {', '.join(missing[:10])}")
    if splits:
        wanted_splits = set(splits)
        rows = [row for row in rows if row["split"] in wanted_splits]
    if max_samples is not None:
        rows = sorted(rows, key=lambda row: sha1_text(row["sample_id"]))[:max_samples]
    return rows


def ensure_output_root(output_root, allow_existing):
    output_root = Path(output_root)
    if output_root.exists() and not allow_existing:
        existing = [p for p in output_root.iterdir()]
        if existing:
            raise FileExistsError(f"Refusing to write into non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_DIRS:
        (output_root / name).mkdir(parents=True, exist_ok=True)
    return output_root


def git_value(repo, *args):
    repo_path = Path(repo)
    if not repo_path.is_dir():
        return ""
    try:
        return subprocess.check_output(["git", "-C", str(repo_path), *args], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def runtime_versions():
    payload = {
        "python": sys.version,
        "numpy": np.__version__,
        "priorda_fine_heit_env": os.environ.get("PRIORDA_FINE_HEIT", ""),
    }
    try:
        import torch

        payload.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "torch_cuda_available": bool(torch.cuda.is_available()),
            }
        )
    except Exception as exc:
        payload["torch_error"] = repr(exc)
    try:
        import torch_cluster

        payload["torch_cluster_version"] = getattr(torch_cluster, "__version__", "unknown")
        payload["torch_cluster_available"] = True
    except Exception as exc:
        payload["torch_cluster_version"] = ""
        payload["torch_cluster_available"] = False
        payload["torch_cluster_import_error"] = repr(exc)
    try:
        import prior_depth_anything.depth_completion as depth_completion

        payload["priorda_torch_cluster_available_in_depth_completion"] = (
            depth_completion.torch_cluster is not None
        )
        payload["priorda_knn_fallback"] = depth_completion.torch_cluster is None
    except Exception as exc:
        payload["priorda_depth_completion_error"] = repr(exc)
    return payload


def resolve_priorda_fine_heit(shape, coarse_only):
    env_value = os.environ.get("PRIORDA_FINE_HEIT", "518").strip().lower()
    if coarse_only:
        return {
            "priorda_fine_heit_env": env_value,
            "priorda_fine_heit_resolved": None,
            "priorda_fine_heit_mode": "coarse_only",
        }
    if env_value in ("", "518"):
        resolved = 518
        mode = "default_518"
    elif env_value == "native":
        resolved = int(shape[0] // 14 * 14)
        mode = "native"
    else:
        resolved = int(env_value) // 14 * 14
        mode = "explicit_int"
    return {
        "priorda_fine_heit_env": env_value,
        "priorda_fine_heit_resolved": resolved,
        "priorda_fine_heit_mode": mode,
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_sparse_npz(path):
    with np.load(path, allow_pickle=False) as data:
        if "arr_0" not in data.files:
            raise KeyError(f"{path} does not contain arr_0")
        return data["arr_0"].astype(np.float32, copy=True)


def build_dav2_geometric_proxy(r, z_min, z_max, p_low, p_high, polarity_flip):
    finite_pos = np.isfinite(r) & (r > 0)
    if int(finite_pos.sum()) == 0:
        raise ValueError("DAv2 map has no finite positive values")

    r_lo = float(np.percentile(r[finite_pos], p_low))
    r_hi = float(np.percentile(r[finite_pos], p_high))
    if not np.isfinite(r_lo) or not np.isfinite(r_hi) or r_hi <= r_lo:
        raise ValueError(f"Invalid DAv2 percentile range: r_lo={r_lo}, r_hi={r_hi}")

    r_filled = np.where(finite_pos, r, r_lo)
    r_clip = np.clip(r_filled, r_lo, r_hi)
    r_unit = (r_clip - r_lo) / max(r_hi - r_lo, 1e-6)
    if polarity_flip:
        r_unit = 1.0 - r_unit

    disp_min = 1.0 / z_max
    disp_max = 1.0 / z_min
    disp_proxy = disp_min + r_unit * (disp_max - disp_min)
    geometric = np.clip(1.0 / disp_proxy, z_min, z_max).astype(np.float32)
    meta = {
        "geometric_transform": "percentile_normalized_inverse_proxy",
        "external_metric_alignment": False,
        "metric_alignment_owner": "PriorDA internal global/KNN alignment",
        "z_min": z_min,
        "z_max": z_max,
        "r_percentile_low": r_lo,
        "r_percentile_high": r_hi,
        "r_percentile_low_q": p_low,
        "r_percentile_high_q": p_high,
        "r_nonfinite_or_nonpositive_ratio": float((~finite_pos).mean()),
        "r_nonfinite_policy": "fill_with_r_percentile_low_before_clipping",
        "polarity_flipped": bool(polarity_flip),
        "geometric_proxy_min": float(np.nanmin(geometric)),
        "geometric_proxy_max": float(np.nanmax(geometric)),
        "geometric_proxy_saturation_min_ratio": float((geometric <= z_min + 1e-6).mean()),
        "geometric_proxy_saturation_max_ratio": float((geometric >= z_max - 1e-6).mean()),
    }
    return geometric, meta


def safe_spearman(r, sparse_depth, mask):
    valid = mask & np.isfinite(r) & np.isfinite(sparse_depth) & (sparse_depth > 0)
    if int(valid.sum()) < 2:
        return None
    try:
        corr = spearmanr(r[valid].reshape(-1), (1.0 / sparse_depth[valid]).reshape(-1)).correlation
    except Exception:
        return None
    if corr is None or not np.isfinite(corr):
        return None
    return float(corr)


def holdout_split(sample_id, mask, holdout_fraction, min_holdout_points, min_prior_points):
    coords = np.argwhere(mask)
    if coords.shape[0] <= min_prior_points:
        return None, None, "too_few_sparse_points_for_holdout"
    seed = int(sha1_text(sample_id)[:8], 16)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(coords.shape[0])
    num_holdout = max(min_holdout_points, int(round(coords.shape[0] * holdout_fraction)))
    num_holdout = min(num_holdout, coords.shape[0] - min_prior_points)
    if num_holdout < min_holdout_points:
        return None, None, "too_few_holdout_points"
    holdout_coords = coords[perm[:num_holdout]]
    holdout_mask = np.zeros_like(mask, dtype=bool)
    holdout_mask[holdout_coords[:, 0], holdout_coords[:, 1]] = True
    prior_mask = mask & ~holdout_mask
    return prior_mask, holdout_mask, "ok"


def depth_metrics(pred, gt, mask):
    eval_mask = mask & np.isfinite(pred) & (pred > 0) & np.isfinite(gt) & (gt > 0)
    count = int(eval_mask.sum())
    if count == 0:
        return {
            "eval_points": 0,
            "absrel": None,
            "rmse": None,
            "delta1": None,
        }
    pred_v = pred[eval_mask].astype(np.float64)
    gt_v = gt[eval_mask].astype(np.float64)
    thresh = np.maximum(pred_v / gt_v, gt_v / pred_v)
    return {
        "eval_points": count,
        "absrel": float(np.mean(np.abs(pred_v - gt_v) / gt_v)),
        "rmse": float(np.sqrt(np.mean((pred_v - gt_v) ** 2))),
        "delta1": float(np.mean(thresh < 1.25)),
    }


def infer_priorda(priorda, image_path, prior, geometric, expected_shape):
    import torch

    with torch.no_grad():
        output = priorda.infer_one_sample(
            image=str(image_path),
            prior=prior.astype(np.float32, copy=False),
            geometric=geometric.astype(np.float32, copy=False),
            pattern=None,
            visualize=False,
        )
    depth = output.detach().float().cpu().numpy().astype(np.float32)
    depth = np.squeeze(depth)
    if depth.shape != expected_shape:
        raise RuntimeError(f"PriorDA output shape {depth.shape} != expected {expected_shape}")
    return depth


def image_size(path):
    with Image.open(path) as image:
        return image.size[1], image.size[0]


def colorize_depth(arr, valid=None, vmin=None, vmax=None, cmap_name="turbo"):
    from matplotlib import colormaps

    arr = arr.astype(np.float32, copy=False)
    if valid is None:
        valid = np.isfinite(arr)
    else:
        valid = valid & np.isfinite(arr)
    if vmin is None or vmax is None:
        if int(valid.sum()) == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vals = arr[valid]
            vmin = float(np.percentile(vals, 1.0)) if vmin is None else vmin
            vmax = float(np.percentile(vals, 99.0)) if vmax is None else vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = (colormaps[cmap_name](norm)[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def resize_tile(arr, width):
    image = Image.fromarray(arr)
    scale = width / image.width
    return image.resize((width, max(1, int(round(image.height * scale)))), Image.Resampling.BILINEAR)


def add_label(tile, text):
    canvas = Image.new("RGB", (tile.width, tile.height + 24), (255, 255, 255))
    canvas.paste(tile, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), text, fill=(0, 0, 0))
    return canvas


def sparse_overlay(rgb_path, sparse_depth, sparse_mask, z_min, z_max):
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    color = colorize_depth(sparse_depth, sparse_mask, z_min, z_max)
    out = rgb.copy()
    out[sparse_mask] = (0.35 * out[sparse_mask] + 0.65 * color[sparse_mask]).astype(np.uint8)
    return out


def make_debug_vis(
    path,
    rgb_path,
    sparse_depth,
    sparse_mask,
    dav2_raw,
    geometric,
    raw_depth,
    overwrite_depth,
    gated_depth,
    gated_mask,
    z_min,
    z_max,
    tile_width,
):
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    diff = np.abs(overwrite_depth - raw_depth)
    tiles = [
        ("RGB", rgb),
        ("sparse LiDAR overlay", sparse_overlay(rgb_path, sparse_depth, sparse_mask, z_min, z_max)),
        ("DAv2-L raw inverse", colorize_depth(dav2_raw, np.isfinite(dav2_raw))),
        ("DAv2 geometric proxy", colorize_depth(geometric, np.isfinite(geometric), z_min, z_max)),
        ("PriorDA raw clipped", colorize_depth(raw_depth, np.isfinite(raw_depth), z_min, z_max)),
        ("LiDAR overwrite", colorize_depth(overwrite_depth, np.isfinite(overwrite_depth), z_min, z_max)),
        ("gated overwrite", colorize_depth(gated_depth, np.isfinite(gated_depth), z_min, z_max)),
        ("raw vs overwrite diff", colorize_depth(diff, np.isfinite(diff), 0.0, max(1.0, float(np.nanpercentile(diff, 99.0))))),
        ("sparse/gated mask", mask_rgb(sparse_mask, gated_mask)),
    ]
    pil_tiles = [add_label(resize_tile(arr, tile_width), label) for label, arr in tiles]
    cols = 3
    rows = int(math.ceil(len(pil_tiles) / cols))
    tile_h = max(tile.height for tile in pil_tiles)
    canvas = Image.new("RGB", (cols * tile_width, rows * tile_h), (255, 255, 255))
    for idx, tile in enumerate(pil_tiles):
        x = (idx % cols) * tile_width
        y = (idx // cols) * tile_h
        canvas.paste(tile, (x, y))
    canvas.save(path)


def mask_rgb(sparse_mask, gated_mask):
    out = np.zeros((*sparse_mask.shape, 3), dtype=np.uint8)
    out[sparse_mask] = (220, 220, 220)
    out[gated_mask] = (255, 64, 64)
    return out


def make_contact_sheet(output_root, processed_rows, tile_width):
    panels = []
    for row in processed_rows:
        panel_path = row.get("debug_vis_path")
        if not panel_path or not Path(panel_path).is_file():
            continue
        panels.append(resize_tile(np.asarray(Image.open(panel_path).convert("RGB")), tile_width * 3))
    if not panels:
        return None
    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height
    path = Path(output_root) / "preview10_contact_sheet.png"
    canvas.save(path)
    return str(path)


def output_paths(output_root, sample_id):
    return {
        "dense_depth_priorda_raw": output_root / "dense_depth_priorda_raw" / f"{sample_id}.npy",
        "dense_depth_priorda_lidar_overwrite": output_root
        / "dense_depth_priorda_lidar_overwrite"
        / f"{sample_id}.npy",
        "dense_depth_priorda_lidar_gated_overwrite": output_root
        / "dense_depth_priorda_lidar_gated_overwrite"
        / f"{sample_id}.npy",
        "dav2_geometric_proxy": output_root / "dav2_geometric_proxy" / f"{sample_id}.npy",
        "sparse_lidar_prior": output_root / "sparse_lidar_prior" / f"{sample_id}.npy",
        "sparse_lidar_mask": output_root / "sparse_lidar_mask" / f"{sample_id}.png",
        "geometric_meta": output_root / "geometric_meta" / f"{sample_id}.json",
        "priorda_meta": output_root / "priorda_meta" / f"{sample_id}.json",
        "qa_meta": output_root / "qa_meta" / f"{sample_id}.json",
        "debug_vis": output_root / "debug_vis" / f"{sample_id}.png",
    }


def init_priorda(args):
    from prior_depth_anything import PriorDepthAnything

    return PriorDepthAnything(
        device=args.device,
        version=args.priorda_version,
        mde_dir=args.mde_dir,
        ckpt_dir=args.ckpt_dir,
        frozen_model_size=args.frozen_model_size,
        conditioned_model_size=args.conditioned_model_size,
        coarse_only=args.coarse_only,
    ).eval()


def process_sample(row, args, output_root, priorda, runtime, repo_info):
    sample_id = row["sample_id"]
    paths = output_paths(output_root, sample_id)
    started = time.time()
    sparse_raw = load_sparse_npz(row["sparse_depth_path"])
    dav2_raw = np.load(row["dav2_npy_path"], allow_pickle=False).astype(np.float32, copy=False)
    rgb_h, rgb_w = image_size(row["rgb_path"])
    expected_shape = sparse_raw.shape
    if dav2_raw.shape != expected_shape:
        raise RuntimeError(f"{sample_id}: DAv2 shape {dav2_raw.shape} != sparse shape {expected_shape}")
    if expected_shape != (rgb_h, rgb_w):
        raise RuntimeError(f"{sample_id}: sparse shape {expected_shape} != RGB shape {(rgb_h, rgb_w)}")

    sparse_mask = (
        np.isfinite(sparse_raw) & (sparse_raw >= args.z_min) & (sparse_raw <= args.z_max)
    )
    prior = np.zeros_like(sparse_raw, dtype=np.float32)
    prior[sparse_mask] = sparse_raw[sparse_mask]
    num_sparse_raw = int((np.isfinite(sparse_raw) & (sparse_raw > 0)).sum())
    num_sparse_in_range = int(sparse_mask.sum())

    geometric, geometric_meta = build_dav2_geometric_proxy(
        dav2_raw,
        args.z_min,
        args.z_max,
        args.r_percentile_low,
        args.r_percentile_high,
        args.polarity_flip,
    )
    geometric_meta.update(
        {
            "sample_id": sample_id,
            "rgb_path": row["rgb_path"],
            "sparse_depth_path": row["sparse_depth_path"],
            "dav2_source_manifest": args.manifest,
            "dav2_source_npy": row["dav2_npy_path"],
            "dav2_model": "Depth Anything V2 Large relative inverse depth",
            "dav2_reused_existing_output": True,
            "num_sparse_raw_points": num_sparse_raw,
            "num_sparse_in_range_points": num_sparse_in_range,
            "num_sparse_removed_by_range": int(num_sparse_raw - num_sparse_in_range),
            "spearman_corr_r_invz": safe_spearman(dav2_raw, sparse_raw, sparse_mask),
            "notes": "",
        }
    )

    np.save(paths["sparse_lidar_prior"], prior.astype(np.float32, copy=False))
    np.save(paths["dav2_geometric_proxy"], geometric.astype(np.float32, copy=False))
    Image.fromarray((sparse_mask.astype(np.uint8) * 255)).save(paths["sparse_lidar_mask"])
    write_json(paths["geometric_meta"], geometric_meta)

    qa_meta = {
        "sample_id": sample_id,
        "qa_mode": "holdout_sparse_lidar",
        "holdout_fraction": args.holdout_fraction,
        "holdout_seed_policy": "sha1(sample_id) first 8 hex chars -> uint32 seed",
        "min_sparse_points": args.min_sparse_points,
        "min_holdout_points": args.min_holdout_points,
        "num_sparse_in_range_points": num_sparse_in_range,
        "qa_status": "not_run",
        "num_prior_points": None,
        "num_holdout_points": None,
        "absrel_holdout": None,
        "rmse_holdout": None,
        "delta1_holdout": None,
        "absrel_raw_on_all_sparse_for_reference_only": None,
        "notes": "",
    }

    if num_sparse_in_range < 5:
        raise RuntimeError(f"{sample_id}: too few in-range sparse points for PriorDA: {num_sparse_in_range}")

    if (not args.skip_holdout_qa) and num_sparse_in_range >= args.min_sparse_points:
        prior_mask_for_qa, holdout_mask, status = holdout_split(
            sample_id,
            sparse_mask,
            args.holdout_fraction,
            args.min_holdout_points,
            min_prior_points=5,
        )
        qa_meta["qa_status"] = status
        if status == "ok":
            prior_for_qa = np.zeros_like(sparse_raw, dtype=np.float32)
            prior_for_qa[prior_mask_for_qa] = sparse_raw[prior_mask_for_qa]
            qa_pred = infer_priorda(priorda, row["rgb_path"], prior_for_qa, geometric, expected_shape)
            metrics = depth_metrics(qa_pred, sparse_raw, holdout_mask)
            qa_meta.update(
                {
                    "num_prior_points": int(prior_mask_for_qa.sum()),
                    "num_holdout_points": int(holdout_mask.sum()),
                    "holdout_eval_points": metrics["eval_points"],
                    "absrel_holdout": metrics["absrel"],
                    "rmse_holdout": metrics["rmse"],
                    "delta1_holdout": metrics["delta1"],
                }
            )
    elif args.skip_holdout_qa:
        qa_meta["qa_status"] = "skipped_by_flag"
    else:
        qa_meta["qa_status"] = "too_few_sparse_points"

    priorda_raw_model = infer_priorda(priorda, row["rgb_path"], prior, geometric, expected_shape)
    invalid_output = ~np.isfinite(priorda_raw_model) | (priorda_raw_model <= 0)
    raw_metrics = depth_metrics(priorda_raw_model, sparse_raw, sparse_mask)
    qa_meta["absrel_raw_on_all_sparse_for_reference_only"] = raw_metrics["absrel"]

    raw_clipped = np.clip(priorda_raw_model, args.z_min, args.z_max).astype(np.float32)
    overwrite = raw_clipped.copy()
    overwrite[sparse_mask] = sparse_raw[sparse_mask]
    gated_mask = (
        sparse_mask
        & np.isfinite(raw_clipped)
        & (raw_clipped > 0)
        & (np.abs(raw_clipped - sparse_raw) / np.maximum(sparse_raw, 1e-6) <= args.overwrite_rel_threshold)
    )
    gated = raw_clipped.copy()
    gated[gated_mask] = sparse_raw[gated_mask]

    np.save(paths["dense_depth_priorda_raw"], raw_clipped.astype(np.float32, copy=False))
    np.save(paths["dense_depth_priorda_lidar_overwrite"], overwrite.astype(np.float32, copy=False))
    np.save(paths["dense_depth_priorda_lidar_gated_overwrite"], gated.astype(np.float32, copy=False))

    priorda_meta = {
        "sample_id": sample_id,
        "priorda_repo": "SpatialVision/Prior-Depth-Anything",
        "priorda_repo_local_path": args.priorda_repo,
        "priorda_repo_commit": repo_info["commit"],
        "priorda_repo_dirty": repo_info["dirty"],
        "priorda_weights_repo": "Rain729/Prior-Depth-Anything",
        "priorda_checkpoint": f"prior_depth_anything_{args.conditioned_model_size}_1_1.pth",
        "frozen_mde_checkpoint": f"depth_anything_v2_{args.frozen_model_size}.pth",
        "frozen_mde_loaded_at_init": True,
        "frozen_mde_forward_skipped_by_geometric": True,
        "priorda_version": args.priorda_version,
        "frozen_model_size": args.frozen_model_size,
        "conditioned_model_size": args.conditioned_model_size,
        **resolve_priorda_fine_heit(raw_clipped.shape, args.coarse_only),
        "torch_version": runtime.get("torch_version", ""),
        "torch_cuda_version": runtime.get("torch_cuda_version", ""),
        "torch_cluster_version": runtime.get("torch_cluster_version", ""),
        "torch_cluster_available": runtime.get("torch_cluster_available", False),
        "knn_backend": "torch_cluster" if runtime.get("torch_cluster_available") else "scipy_cKDTree_fallback",
        "input_prior": "sparse_lidar_prior",
        "input_geometric": "dav2_geometric_proxy",
        "pattern": None,
        "output_raw": "dense_depth_priorda_raw",
        "output_raw_meaning": "no_lidar_overwrite_post_clipped",
        "output_final": "dense_depth_priorda_lidar_overwrite",
        "lidar_overwrite": True,
        "gated_overwrite": True,
        "overwrite_rel_threshold": args.overwrite_rel_threshold,
        "gated_overwrite_points": int(gated_mask.sum()),
        "output_unit": "meters",
        "output_shape": list(raw_clipped.shape),
        "output_nonfinite_or_nonpositive_ratio": float(invalid_output.mean()),
        "output_raw_model_min": float(np.nanmin(priorda_raw_model)),
        "output_raw_model_max": float(np.nanmax(priorda_raw_model)),
        "post_clip_min": args.z_min,
        "post_clip_max": args.z_max,
        "elapsed_seconds": float(time.time() - started),
        "notes": "",
    }
    write_json(paths["priorda_meta"], priorda_meta)
    write_json(paths["qa_meta"], qa_meta)
    make_debug_vis(
        paths["debug_vis"],
        row["rgb_path"],
        sparse_raw,
        sparse_mask,
        dav2_raw,
        geometric,
        raw_clipped,
        overwrite,
        gated,
        gated_mask,
        args.z_min,
        args.z_max,
        args.debug_tile_width,
    )

    status = "ok"
    if priorda_meta["output_nonfinite_or_nonpositive_ratio"] > 0:
        status = "qa_failed"
    elif qa_meta["qa_status"] != "ok":
        status = "low_quality"
    return {
        "sample_id": sample_id,
        "split": row["split"],
        "rgb_path": row["rgb_path"],
        "sparse_depth_path": row["sparse_depth_path"],
        "dav2_npy_path": row["dav2_npy_path"],
        "dense_depth_priorda_raw": str(paths["dense_depth_priorda_raw"]),
        "dense_depth_priorda_lidar_overwrite": str(paths["dense_depth_priorda_lidar_overwrite"]),
        "dense_depth_priorda_lidar_gated_overwrite": str(paths["dense_depth_priorda_lidar_gated_overwrite"]),
        "dav2_geometric_proxy": str(paths["dav2_geometric_proxy"]),
        "sparse_lidar_prior": str(paths["sparse_lidar_prior"]),
        "sparse_lidar_mask": str(paths["sparse_lidar_mask"]),
        "geometric_meta": str(paths["geometric_meta"]),
        "priorda_meta": str(paths["priorda_meta"]),
        "qa_meta": str(paths["qa_meta"]),
        "debug_vis_path": str(paths["debug_vis"]),
        "status": status,
        "num_sparse_in_range_points": num_sparse_in_range,
        "spearman_corr_r_invz": geometric_meta["spearman_corr_r_invz"],
        "absrel_holdout": qa_meta["absrel_holdout"],
        "rmse_holdout": qa_meta["rmse_holdout"],
        "delta1_holdout": qa_meta["delta1_holdout"],
        "elapsed_seconds": priorda_meta["elapsed_seconds"],
    }


def write_manifest(path, rows):
    fieldnames = [
        "sample_id",
        "split",
        "rgb_path",
        "sparse_depth_path",
        "dav2_npy_path",
        "dense_depth_priorda_raw",
        "dense_depth_priorda_lidar_overwrite",
        "dense_depth_priorda_lidar_gated_overwrite",
        "dav2_geometric_proxy",
        "sparse_lidar_prior",
        "sparse_lidar_mask",
        "geometric_meta",
        "priorda_meta",
        "qa_meta",
        "debug_vis_path",
        "status",
        "num_sparse_in_range_points",
        "spearman_corr_r_invz",
        "absrel_holdout",
        "rmse_holdout",
        "delta1_holdout",
        "elapsed_seconds",
        "error",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def summarize(rows):
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") == "failed"]
    absrels = [row.get("absrel_holdout") for row in ok_rows if row.get("absrel_holdout") not in ("", None)]
    return {
        "total_rows": len(rows),
        "ok_rows": len(ok_rows),
        "failed_rows": len(failed_rows),
        "status_counts": {status: sum(1 for row in rows if row.get("status") == status) for status in sorted({row.get("status") for row in rows})},
        "mean_absrel_holdout_ok": float(np.mean([float(x) for x in absrels])) if absrels else None,
    }


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    if args.max_samples is not None and not path_has_temp_marker(output_root):
        raise ValueError(
            "--max-samples output roots must contain one of: "
            + ", ".join(TEMP_MARKERS)
        )
    if args.max_samples is None and not args.preview_approved:
        raise ValueError("Full generation requires --preview-approved pointing to the reviewed preview output.")
    if args.z_max <= args.z_min:
        raise ValueError("--z-max must be larger than --z-min")
    if not (0.0 < args.holdout_fraction < 1.0):
        raise ValueError("--holdout-fraction must be in (0, 1)")

    output_root = ensure_output_root(output_root, args.allow_existing_output_root)
    rows_all = read_manifest(args.manifest)
    rows = select_rows(rows_all, args.sample_id, args.split, args.max_samples)
    if not rows:
        raise ValueError("No manifest rows selected")

    runtime = runtime_versions()
    repo_info = {
        "path": args.priorda_repo,
        "commit": git_value(args.priorda_repo, "rev-parse", "HEAD"),
        "dirty": bool(git_value(args.priorda_repo, "status", "--short")),
    }
    run_config = {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "manifest": args.manifest,
        "output_root": str(output_root),
        "selected_rows": len(rows),
        "total_manifest_rows": len(rows_all),
        "preview_approved": args.preview_approved,
        "args": vars(args),
        "runtime": runtime,
        "priorda_fine_heit_env": os.environ.get("PRIORDA_FINE_HEIT", ""),
        "priorda_repo": repo_info,
        "label_name": "LiDAR-anchored dense pseudo depth",
        "output_unit": "meters",
    }
    write_json(output_root / "run_config.json", run_config)

    priorda = init_priorda(args)
    processed_rows = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] {row['sample_id']}", flush=True)
        try:
            processed = process_sample(row, args, output_root, priorda, runtime, repo_info)
        except Exception as exc:
            processed = {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "rgb_path": row["rgb_path"],
                "sparse_depth_path": row["sparse_depth_path"],
                "dav2_npy_path": row["dav2_npy_path"],
                "status": "failed",
                "error": repr(exc),
            }
            write_json(output_root / "qa_meta" / f"{row['sample_id']}.json", processed)
            print(f"[ERROR] {row['sample_id']}: {exc}", flush=True)
        processed_rows.append(processed)
        write_manifest(output_root / "manifest.csv", processed_rows)

    contact_sheet = None
    if args.preview_contact_sheet:
        contact_sheet = make_contact_sheet(output_root, processed_rows, args.debug_tile_width)
        write_manifest(output_root / "preview10_manifest.csv", processed_rows)

    summary = summarize(processed_rows)
    summary.update(
        {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "preview_contact_sheet": contact_sheet,
        }
    )
    write_json(output_root / ("preview10_summary.json" if args.preview_contact_sheet else "run_summary.json"), summary)
    run_config["status"] = "complete"
    run_config["completed_at"] = summary["completed_at"]
    run_config["summary"] = summary
    write_json(output_root / "run_config.json", run_config)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if summary["failed_rows"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
