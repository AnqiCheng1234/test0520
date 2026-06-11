#!/usr/bin/env python3
"""Compare STF dataset RGB and RAW-rendered teacher RGB under DAV2-L."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
from finetune_stf.dataset.raw_utils import (  # noqa: E402
    COMPANDED_MAX,
    DECOMPANDED_MAX,
    get_stf_decompanding_lut,
)


DEFAULT_MANIFEST = Path(
    "/mnt/drive/3333_raw/seeing_through_fog/"
    "pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/"
    "stf_rgb_lut_manifest_6216.csv"
)
DEFAULT_RAW_NPZ_ROOT = Path("/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz")
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "finetune_stf" / "analysis" / "0611_stf_teacher_rgb_dav2_compare"
)
DEFAULT_CHECKPOINT = Path("/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth")

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

TEACHER_GAINS = np.array([1.08, 0.95, 1.10], dtype=np.float32)
TEACHER_WHITE_PERCENTILE = 99.5
TEACHER_GAMMA = 0.454545
REQUIRED_MANIFEST_COLUMNS = {
    "sample_name",
    "split",
    "rgb_path",
    "sparse_depth_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-npz-root", type=Path, default=DEFAULT_RAW_NPZ_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--encoder", default="vitl", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=924)
    parser.add_argument("--dark-count", type=int, default=5)
    parser.add_argument("--overexp-count", type=int, default=5)
    parser.add_argument("--z-min", type=float, default=1.0)
    parser.add_argument("--z-max", type=float, default=80.0)
    parser.add_argument("--min-sparse-points", type=int, default=100)
    parser.add_argument("--min-prior-points", type=int, default=50)
    parser.add_argument("--min-holdout-points", type=int, default=50)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--raw-normalization", choices=("companded", "decompanded"), default="companded")
    parser.add_argument("--teacher-white-percentile", type=float, default=TEACHER_WHITE_PERCENTILE)
    parser.add_argument("--teacher-gamma", type=float, default=TEACHER_GAMMA)
    parser.add_argument("--teacher-gains", nargs=3, type=float, default=TEACHER_GAINS.tolist())
    parser.add_argument("--enhance-white-percentile", type=float, default=99.5)
    parser.add_argument("--enhance-gamma", type=float, default=0.65)
    parser.add_argument("--enhance-clahe-clip", type=float, default=2.0)
    parser.add_argument("--enhance-clahe-grid", type=int, default=8)
    parser.add_argument("--cmap", default="Spectral_r")
    parser.add_argument("--viz-vmin-pct", type=float, default=1.0)
    parser.add_argument("--viz-vmax-pct", type=float, default=99.0)
    parser.add_argument("--tile-width", type=int, default=384)
    parser.add_argument("--gt-radius", type=int, default=5)
    parser.add_argument("--overwrite-preds", action="store_true")
    return parser.parse_args()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(missing)}")
        for row in reader:
            sample_id = row["sample_name"]
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": row["split"],
                    "rgb_path": Path(row["rgb_path"]),
                    "sparse_depth_path": Path(row["sparse_depth_path"]),
                    "raw_npz_path": Path(),  # filled after args are available
                }
            )
    return rows


def image_stats(rgb_path: Path) -> dict[str, float]:
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_REDUCED_COLOR_8)
    if bgr is None:
        raise ValueError(f"OpenCV failed to read {rgb_path}")
    rgb = bgr[..., ::-1].astype(np.float32)
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    max_ch = np.max(rgb, axis=-1)
    return {
        "rgb_luma_mean": float(np.mean(luma)),
        "rgb_luma_p99": float(np.percentile(luma, 99.0)),
        "rgb_luma_p999": float(np.percentile(luma, 99.9)),
        "rgb_sat_ratio": float(np.mean((max_ch >= 250.0) | (luma >= 245.0))),
    }


def load_sparse_depth(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "arr_0" in data.files:
            depth = data["arr_0"]
        elif "depth" in data.files:
            depth = data["depth"]
        else:
            raise KeyError(f"{path} does not contain arr_0 or depth")
    return np.asarray(depth, dtype=np.float32)


def sparse_valid_mask(sparse_depth: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    return (
        np.isfinite(sparse_depth)
        & (sparse_depth >= float(z_min))
        & (sparse_depth <= float(z_max))
    )


def count_sparse_points(path: Path, z_min: float, z_max: float) -> int:
    sparse = load_sparse_depth(path)
    return int(np.count_nonzero(sparse_valid_mask(sparse, z_min, z_max)))


def select_samples(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    start = time.time()
    for idx, row in enumerate(rows, start=1):
        raw_path = args.raw_npz_root / f"{row['sample_id']}.npz"
        if not row["rgb_path"].is_file() or not row["sparse_depth_path"].is_file() or not raw_path.is_file():
            continue
        item = dict(row)
        item["raw_npz_path"] = raw_path
        item.update(image_stats(item["rgb_path"]))
        candidates.append(item)
        if idx % 500 == 0:
            print(
                f"[select] scanned {idx}/{len(rows)} rows, usable={len(candidates)} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    if not candidates:
        raise ValueError("No usable STF candidates found")

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def try_add(item: dict[str, Any], group: str) -> bool:
        if item["sample_id"] in selected_ids:
            return False
        sparse_points = count_sparse_points(item["sparse_depth_path"], args.z_min, args.z_max)
        if sparse_points < int(args.min_sparse_points):
            return False
        picked = dict(item)
        picked["selection_group"] = group
        picked["num_sparse_in_range_points"] = sparse_points
        selected.append(picked)
        selected_ids.add(item["sample_id"])
        return True

    for item in sorted(candidates, key=lambda x: (x["rgb_luma_mean"], x["rgb_luma_p99"], x["sample_id"])):
        if sum(1 for x in selected if x["selection_group"] == "dark") >= args.dark_count:
            break
        try_add(item, "dark")

    overexp_sorted = sorted(
        candidates,
        key=lambda x: (-x["rgb_sat_ratio"], -x["rgb_luma_p999"], -x["rgb_luma_p99"], x["sample_id"]),
    )
    for item in overexp_sorted:
        if sum(1 for x in selected if x["selection_group"] == "overexposed") >= args.overexp_count:
            break
        try_add(item, "overexposed")

    expected = int(args.dark_count) + int(args.overexp_count)
    if len(selected) != expected:
        raise RuntimeError(f"Selected {len(selected)} samples, expected {expected}")
    return selected


def load_model(args: argparse.Namespace, device: torch.device) -> DepthAnythingV2:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing DAV2 checkpoint: {args.checkpoint}")
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    state = torch.load(str(args.checkpoint), map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


def read_bgr(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV failed to read {path}")
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected uint8 BGR image, got shape={bgr.shape} dtype={bgr.dtype}: {path}")
    return bgr


def load_stf_bayer(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "bayer_rect" not in data.files:
            raise KeyError(f"{path} does not contain bayer_rect")
        bayer = data["bayer_rect"]
    if bayer.ndim != 3 or bayer.shape[-1] != 4:
        raise ValueError(f"Expected bayer_rect shape HxWx4, got {bayer.shape}: {path}")
    return np.asarray(bayer)


def stf_bayer_to_rgb01(bayer: np.ndarray, raw_normalization: str) -> np.ndarray:
    if raw_normalization == "companded":
        bayer01 = np.clip(bayer.astype(np.float32) / float(COMPANDED_MAX), 0.0, 1.0)
    elif raw_normalization == "decompanded":
        lut = get_stf_decompanding_lut()
        codes = np.clip(bayer, 0, len(lut) - 1).astype(np.uint16, copy=False)
        bayer01 = lut[codes].astype(np.float32) / float(DECOMPANDED_MAX)
    else:
        raise ValueError(f"Unsupported raw_normalization: {raw_normalization!r}")
    r = bayer01[..., 0]
    g = 0.5 * (bayer01[..., 1] + bayer01[..., 2])
    b = bayer01[..., 3]
    return np.stack([r, g, b], axis=-1).astype(np.float32, copy=False)


def render_teacher_rgb(
    bayer: np.ndarray,
    *,
    raw_normalization: str,
    gains: np.ndarray,
    white_percentile: float,
    gamma: float,
) -> np.ndarray:
    rgb = stf_bayer_to_rgb01(bayer, raw_normalization=raw_normalization)
    rgb = np.clip(rgb * gains.reshape(1, 1, 3), 0.0, 1.0)
    if white_percentile < 100.0:
        white = float(np.percentile(rgb, white_percentile))
        if white > 1e-6:
            rgb = np.clip(rgb / white, 0.0, 1.0)
    rgb = np.power(np.clip(rgb, 0.0, 1.0), float(gamma), dtype=np.float32)
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def enhance_low_light_rgb(
    bgr: np.ndarray,
    *,
    white_percentile: float,
    gamma: float,
    clahe_clip: float,
    clahe_grid: int,
) -> np.ndarray:
    """Deterministic low-light enhancement for already-rendered STF dark RGB."""

    grid = max(int(clahe_grid), 1)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(grid, grid))
    l_chan = clahe.apply(l_chan)
    enhanced_bgr = cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
    rgb = enhanced_bgr[..., ::-1].astype(np.float32) / 255.0
    if white_percentile < 100.0:
        white = float(np.percentile(rgb, white_percentile))
        if white > 1e-6:
            rgb = np.clip(rgb / white, 0.0, 1.0)
    rgb = np.power(np.clip(rgb, 0.0, 1.0), float(gamma), dtype=np.float32)
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def infer_dav2(model: DepthAnythingV2, bgr: np.ndarray, input_size: int) -> np.ndarray:
    with torch.inference_mode():
        pred = model.infer_image(bgr, input_size=int(input_size)).astype(np.float32, copy=False)
    if pred.shape != bgr.shape[:2]:
        raise ValueError(f"DAV2 output shape {pred.shape} != image shape {bgr.shape[:2]}")
    if not np.isfinite(pred).all():
        raise ValueError("DAV2 output contains non-finite values")
    if float(np.max(pred)) <= 0.0:
        raise ValueError("DAV2 output is non-positive")
    return pred


def holdout_split(
    sample_id: str,
    mask: np.ndarray,
    holdout_fraction: float,
    min_holdout_points: int,
    min_prior_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    if coords.shape[0] < min_holdout_points + min_prior_points:
        raise ValueError(f"too few sparse points for holdout: {coords.shape[0]}")
    seed = int(sha1_text(sample_id)[:8], 16)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(coords.shape[0])
    num_holdout = max(int(min_holdout_points), int(round(coords.shape[0] * float(holdout_fraction))))
    num_holdout = min(num_holdout, coords.shape[0] - int(min_prior_points))
    if num_holdout < int(min_holdout_points):
        raise ValueError(f"too few holdout points: {num_holdout}")
    holdout_coords = coords[perm[:num_holdout]]
    holdout = np.zeros_like(mask, dtype=bool)
    holdout[holdout_coords[:, 0], holdout_coords[:, 1]] = True
    prior = mask & ~holdout
    return prior, holdout


def align_relative_inverse_to_depth(
    pred_rel: np.ndarray,
    sparse_depth: np.ndarray,
    prior_mask: np.ndarray,
    *,
    z_min: float,
    z_max: float,
) -> dict[str, Any]:
    valid = (
        prior_mask
        & np.isfinite(pred_rel)
        & (pred_rel > 0.0)
        & np.isfinite(sparse_depth)
        & (sparse_depth >= float(z_min))
        & (sparse_depth <= float(z_max))
    )
    if int(np.count_nonzero(valid)) < 2:
        raise ValueError("too few valid prior points for affine alignment")
    x = pred_rel[valid].astype(np.float64)
    y = (1.0 / sparse_depth[valid].astype(np.float64))
    design = np.stack([x, np.ones_like(x)], axis=1)
    scale, shift = np.linalg.lstsq(design, y, rcond=None)[0]
    aligned_inv = pred_rel.astype(np.float64) * float(scale) + float(shift)
    aligned_inv = np.clip(aligned_inv, 1.0 / float(z_max), 1.0 / float(z_min)).astype(np.float32)
    aligned_depth = (1.0 / aligned_inv).astype(np.float32)
    return {
        "aligned_inv": aligned_inv,
        "aligned_depth": aligned_depth,
        "affine_scale": float(scale),
        "affine_shift": float(shift),
        "fit_points": int(np.count_nonzero(valid)),
    }


def depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid = mask & np.isfinite(pred_depth) & (pred_depth > 0.0) & np.isfinite(gt_depth) & (gt_depth > 0.0)
    count = int(np.count_nonzero(valid))
    if count == 0:
        return {"eval_points": 0, "absrel": math.nan, "d1": math.nan}
    pred = pred_depth[valid].astype(np.float64)
    gt = gt_depth[valid].astype(np.float64)
    thresh = np.maximum(pred / gt, gt / pred)
    return {
        "eval_points": count,
        "absrel": float(np.mean(np.abs(pred - gt) / gt)),
        "d1": float(np.mean(thresh < 1.25)),
    }


def scalar_to_rgb(values: np.ndarray, *, vmin: float, vmax: float, cmap_name: str) -> np.ndarray:
    cmap = colormaps.get_cmap(cmap_name)
    norm = np.clip((values.astype(np.float32) - float(vmin)) / max(float(vmax) - float(vmin), 1e-9), 0.0, 1.0)
    return np.clip(cmap(norm)[..., :3] * 255.0, 0.0, 255.0).astype(np.uint8)


def colorize_dense(inv_map: np.ndarray, *, vmin: float, vmax: float, cmap_name: str) -> Image.Image:
    valid = np.isfinite(inv_map)
    rgb = np.full((*inv_map.shape, 3), 235, dtype=np.uint8)
    rgb[valid] = scalar_to_rgb(inv_map[valid], vmin=vmin, vmax=vmax, cmap_name=cmap_name)
    return Image.fromarray(rgb, mode="RGB")


def colorize_sparse_gt(
    sparse_depth: np.ndarray,
    mask: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap_name: str,
    radius: int,
) -> Image.Image:
    image = Image.fromarray(np.full((*sparse_depth.shape, 3), 235, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    inv = 1.0 / sparse_depth[mask].astype(np.float32)
    colors = scalar_to_rgb(inv, vmin=vmin, vmax=vmax, cmap_name=cmap_name)
    r = int(radius)
    for (y, x), color in zip(coords, colors):
        draw.ellipse((int(x) - r, int(y) - r, int(x) + r, int(y) + r), fill=tuple(int(v) for v in color))
    return image


def resize_tile(image: Image.Image, tile_w: int, tile_h: int, resample: int = Image.Resampling.BILINEAR) -> Image.Image:
    return image.resize((int(tile_w), int(tile_h)), resample=resample)


def get_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        p = Path(path)
        if p.is_file():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def truncate_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    suffix = "..."
    while text and draw.textlength(text + suffix, font=font) > max_width:
        text = text[:-1]
    return text + suffix


def make_contact_sheet(
    records: list[dict[str, Any]],
    output_path: Path,
    *,
    vmin: float,
    vmax: float,
    cmap_name: str,
    tile_width: int,
    gt_radius: int,
) -> None:
    if not records:
        raise ValueError("No records for contact sheet")
    height, width = records[0]["sparse_depth"].shape
    tile_w = int(tile_width)
    tile_h = int(round(tile_w * height / width))
    margin = 18
    gap = 10
    title_h = 86
    header_h = 30
    caption_h = 38
    row_h = caption_h + tile_h + gap
    cols = [
        "dataset RGB",
        "teacher_rgb",
        "enhanced RGB",
        "GT sparse inv",
        "DAV2-L dataset RGB inv",
        "DAV2-L teacher RGB inv",
        "DAV2-L enhanced RGB inv",
    ]
    sheet_w = margin * 2 + len(cols) * tile_w + (len(cols) - 1) * gap
    sheet_h = title_h + header_h + len(records) * row_h + margin
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    title_font = get_font(20)
    header_font = get_font(15)
    text_font = get_font(13)

    title = (
        "STF RAW teacher_rgb vs dataset RGB DAV2-L, sparse GT and predictions in shared inverse-depth scale"
    )
    draw.text((margin, 12), title, fill=(0, 0, 0), font=title_font)
    draw.text(
        (margin, 40),
        f"Colormap={cmap_name}; shared inverse scale vmin={vmin:.5g}, vmax={vmax:.5g}; larger inverse means closer.",
        fill=(35, 35, 35),
        font=text_font,
    )
    bar_w = min(420, sheet_w - 2 * margin)
    bar_h = 14
    grad = np.linspace(vmin, vmax, bar_w, dtype=np.float32).reshape(1, -1)
    bar = colorize_dense(grad, vmin=vmin, vmax=vmax, cmap_name=cmap_name).resize((bar_w, bar_h))
    sheet.paste(bar, (margin, 62))
    draw.text((margin + bar_w + 10, 58), "far -> near", fill=(35, 35, 35), font=text_font)

    y = title_h
    for col_idx, col in enumerate(cols):
        x = margin + col_idx * (tile_w + gap)
        draw.text((x, y + 5), col, fill=(0, 0, 0), font=header_font)

    y += header_h
    for rec in records:
        official = rec["official_metrics"]
        teacher = rec["teacher_metrics"]
        caption = (
            f"{rec['selection_group']} | {rec['sample_id']} | "
            f"mean={rec['rgb_luma_mean']:.1f} p99={rec['rgb_luma_p99']:.1f} sat={rec['rgb_sat_ratio']:.4f} | "
            f"dataset AbsRel={official['absrel']:.4f} D1={official['d1']:.4f} | "
            f"teacher AbsRel={teacher['absrel']:.4f} D1={teacher['d1']:.4f} | "
            f"enhanced AbsRel={rec['enhanced_metrics']['absrel']:.4f} D1={rec['enhanced_metrics']['d1']:.4f}"
        )
        draw.text((margin, y + 7), truncate_text(draw, caption, text_font, sheet_w - 2 * margin), fill=(20, 20, 20), font=text_font)
        y_tiles = y + caption_h
        images = [
            Image.fromarray(rec["official_rgb"][..., ::-1], mode="RGB"),
            Image.fromarray(rec["teacher_rgb_full"], mode="RGB"),
            Image.fromarray(rec["enhanced_rgb_full"], mode="RGB"),
            colorize_sparse_gt(
                rec["sparse_depth"],
                rec["valid_sparse_mask"],
                vmin=vmin,
                vmax=vmax,
                cmap_name=cmap_name,
                radius=gt_radius,
            ),
            colorize_dense(rec["official_aligned_inv"], vmin=vmin, vmax=vmax, cmap_name=cmap_name),
            colorize_dense(rec["teacher_aligned_inv"], vmin=vmin, vmax=vmax, cmap_name=cmap_name),
            colorize_dense(rec["enhanced_aligned_inv"], vmin=vmin, vmax=vmax, cmap_name=cmap_name),
        ]
        for col_idx, image in enumerate(images):
            x = margin + col_idx * (tile_w + gap)
            tile = resize_tile(image, tile_w, tile_h)
            sheet.paste(tile, (x, y_tiles))
        y += row_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "selection_group",
        "sample_id",
        "split",
        "rgb_luma_mean",
        "rgb_luma_p99",
        "rgb_luma_p999",
        "rgb_sat_ratio",
        "num_sparse_in_range_points",
        "holdout_eval_points",
        "official_absrel",
        "official_d1",
        "official_affine_scale",
        "official_affine_shift",
        "teacher_absrel",
        "teacher_d1",
        "teacher_affine_scale",
        "teacher_affine_shift",
        "enhanced_absrel",
        "enhanced_d1",
        "enhanced_affine_scale",
        "enhanced_affine_shift",
        "rgb_path",
        "raw_npz_path",
        "sparse_depth_path",
        "official_pred_npy",
        "teacher_rgb_png",
        "teacher_pred_npy",
        "enhanced_rgb_png",
        "enhanced_pred_npy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "selection_group": rec["selection_group"],
                    "sample_id": rec["sample_id"],
                    "split": rec["split"],
                    "rgb_luma_mean": f"{rec['rgb_luma_mean']:.9g}",
                    "rgb_luma_p99": f"{rec['rgb_luma_p99']:.9g}",
                    "rgb_luma_p999": f"{rec['rgb_luma_p999']:.9g}",
                    "rgb_sat_ratio": f"{rec['rgb_sat_ratio']:.9g}",
                    "num_sparse_in_range_points": rec["num_sparse_in_range_points"],
                    "holdout_eval_points": rec["official_metrics"]["eval_points"],
                    "official_absrel": f"{rec['official_metrics']['absrel']:.9g}",
                    "official_d1": f"{rec['official_metrics']['d1']:.9g}",
                    "official_affine_scale": f"{rec['official_align']['affine_scale']:.9g}",
                    "official_affine_shift": f"{rec['official_align']['affine_shift']:.9g}",
                    "teacher_absrel": f"{rec['teacher_metrics']['absrel']:.9g}",
                    "teacher_d1": f"{rec['teacher_metrics']['d1']:.9g}",
                    "teacher_affine_scale": f"{rec['teacher_align']['affine_scale']:.9g}",
                    "teacher_affine_shift": f"{rec['teacher_align']['affine_shift']:.9g}",
                    "enhanced_absrel": f"{rec['enhanced_metrics']['absrel']:.9g}",
                    "enhanced_d1": f"{rec['enhanced_metrics']['d1']:.9g}",
                    "enhanced_affine_scale": f"{rec['enhanced_align']['affine_scale']:.9g}",
                    "enhanced_affine_shift": f"{rec['enhanced_align']['affine_shift']:.9g}",
                    "rgb_path": rec["rgb_path"],
                    "raw_npz_path": rec["raw_npz_path"],
                    "sparse_depth_path": rec["sparse_depth_path"],
                    "official_pred_npy": rec["official_pred_npy"],
                    "teacher_rgb_png": rec["teacher_rgb_png"],
                    "teacher_pred_npy": rec["teacher_pred_npy"],
                    "enhanced_rgb_png": rec["enhanced_rgb_png"],
                    "enhanced_pred_npy": rec["enhanced_pred_npy"],
                }
            )


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else math.nan


def write_summary(path: Path, records: list[dict[str, Any]], args: argparse.Namespace, vmin: float, vmax: float) -> None:
    summary = {
        "config": {
            "manifest": str(args.manifest.resolve()),
            "raw_npz_root": str(args.raw_npz_root.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "encoder": args.encoder,
            "device": args.device,
            "input_size": int(args.input_size),
            "raw_channel_order": ["R", "Gr", "Gb", "B"],
            "raw_to_rgb_projection": ["R", "0.5*(Gr+Gb)", "B"],
            "raw_normalization": args.raw_normalization,
            "teacher_white_percentile": float(args.teacher_white_percentile),
            "teacher_gamma": float(args.teacher_gamma),
            "teacher_gains": [float(v) for v in args.teacher_gains],
            "low_light_enhancement": {
                "source": "dataset official RGB",
                "recipe": "LAB CLAHE on L channel, global white-percentile stretch, RGB gamma brighten",
                "white_percentile": float(args.enhance_white_percentile),
                "gamma": float(args.enhance_gamma),
                "clahe_clip": float(args.enhance_clahe_clip),
                "clahe_grid": int(args.enhance_clahe_grid),
            },
            "z_min": float(args.z_min),
            "z_max": float(args.z_max),
            "holdout_fraction": float(args.holdout_fraction),
            "min_prior_points": int(args.min_prior_points),
            "min_holdout_points": int(args.min_holdout_points),
            "selection": {
                "dark_count": int(args.dark_count),
                "overexp_count": int(args.overexp_count),
                "dark_sort": "low official RGB luma_mean",
                "overexposed_sort": "high official RGB saturation ratio, then high luma percentiles",
            },
            "visualization": {
                "cmap": args.cmap,
                "shared_inverse_vmin": float(vmin),
                "shared_inverse_vmax": float(vmax),
                "shared_inverse_vmin_pct": float(args.viz_vmin_pct),
                "shared_inverse_vmax_pct": float(args.viz_vmax_pct),
            },
            "metric_definition": (
                "Fit affine mapping from DAV2-L relative inverse output to sparse LiDAR inverse depth "
                "on deterministic per-sample prior points; compute AbsRel and D1 on held-out sparse points."
            ),
        },
        "aggregate": {
            "num_samples": len(records),
            "official_mean_absrel": finite_mean([r["official_metrics"]["absrel"] for r in records]),
            "official_mean_d1": finite_mean([r["official_metrics"]["d1"] for r in records]),
            "teacher_mean_absrel": finite_mean([r["teacher_metrics"]["absrel"] for r in records]),
            "teacher_mean_d1": finite_mean([r["teacher_metrics"]["d1"] for r in records]),
            "enhanced_mean_absrel": finite_mean([r["enhanced_metrics"]["absrel"] for r in records]),
            "enhanced_mean_d1": finite_mean([r["enhanced_metrics"]["d1"] for r in records]),
        },
        "samples": [
            {
                "selection_group": r["selection_group"],
                "sample_id": r["sample_id"],
                "split": r["split"],
                "rgb_luma_mean": float(r["rgb_luma_mean"]),
                "rgb_luma_p99": float(r["rgb_luma_p99"]),
                "rgb_sat_ratio": float(r["rgb_sat_ratio"]),
                "num_sparse_in_range_points": int(r["num_sparse_in_range_points"]),
                "official_metrics": r["official_metrics"],
                "teacher_metrics": r["teacher_metrics"],
                "enhanced_metrics": r["enhanced_metrics"],
                "official_pred_npy": str(r["official_pred_npy"]),
                "teacher_rgb_png": str(r["teacher_rgb_png"]),
                "teacher_pred_npy": str(r["teacher_pred_npy"]),
                "enhanced_rgb_png": str(r["enhanced_rgb_png"]),
                "enhanced_pred_npy": str(r["enhanced_pred_npy"]),
            }
            for r in records
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if int(args.input_size) % 14 != 0:
        raise ValueError("--input-size must be a multiple of 14")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    teacher_rgb_dir = args.output_dir / "teacher_rgb"
    enhanced_rgb_dir = args.output_dir / "enhanced_rgb"
    official_pred_dir = args.output_dir / "official_rgb_dav2l_pred"
    teacher_pred_dir = args.output_dir / "teacher_rgb_dav2l_pred"
    enhanced_pred_dir = args.output_dir / "enhanced_rgb_dav2l_pred"
    for d in (teacher_rgb_dir, enhanced_rgb_dir, official_pred_dir, teacher_pred_dir, enhanced_pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest)
    print(f"[setup] rows={len(rows)} output={args.output_dir}", flush=True)
    selected = select_samples(rows, args)
    print("[select] selected samples:", flush=True)
    for item in selected:
        print(
            f"  {item['selection_group']:11s} {item['sample_id']} "
            f"mean={item['rgb_luma_mean']:.2f} p99={item['rgb_luma_p99']:.2f} "
            f"sat={item['rgb_sat_ratio']:.5f} sparse={item['num_sparse_in_range_points']}",
            flush=True,
        )

    device = torch.device(args.device)
    model = load_model(args, device)
    gains = np.asarray(args.teacher_gains, dtype=np.float32)
    records: list[dict[str, Any]] = []
    all_inv_values: list[np.ndarray] = []

    for idx, item in enumerate(selected, start=1):
        sample_id = item["sample_id"]
        print(f"[infer] {idx}/{len(selected)} {sample_id}", flush=True)
        official_bgr = read_bgr(item["rgb_path"])
        sparse = load_sparse_depth(item["sparse_depth_path"])
        if sparse.shape != official_bgr.shape[:2]:
            raise ValueError(f"Sparse shape {sparse.shape} != RGB shape {official_bgr.shape[:2]} for {sample_id}")
        valid_sparse = sparse_valid_mask(sparse, args.z_min, args.z_max)
        prior_mask, holdout_mask = holdout_split(
            sample_id,
            valid_sparse,
            args.holdout_fraction,
            args.min_holdout_points,
            args.min_prior_points,
        )

        bayer = load_stf_bayer(item["raw_npz_path"])
        teacher_native = render_teacher_rgb(
            bayer,
            raw_normalization=args.raw_normalization,
            gains=gains,
            white_percentile=args.teacher_white_percentile,
            gamma=args.teacher_gamma,
        )
        teacher_full = cv2.resize(
            teacher_native,
            (official_bgr.shape[1], official_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        enhanced_full = enhance_low_light_rgb(
            official_bgr,
            white_percentile=args.enhance_white_percentile,
            gamma=args.enhance_gamma,
            clahe_clip=args.enhance_clahe_clip,
            clahe_grid=args.enhance_clahe_grid,
        )

        teacher_rgb_png = teacher_rgb_dir / f"{sample_id}.png"
        enhanced_rgb_png = enhanced_rgb_dir / f"{sample_id}.png"
        official_pred_npy = official_pred_dir / f"{sample_id}.npy"
        teacher_pred_npy = teacher_pred_dir / f"{sample_id}.npy"
        enhanced_pred_npy = enhanced_pred_dir / f"{sample_id}.npy"
        cv2.imwrite(str(teacher_rgb_png), teacher_full[..., ::-1])
        cv2.imwrite(str(enhanced_rgb_png), enhanced_full[..., ::-1])

        if official_pred_npy.is_file() and not args.overwrite_preds:
            official_pred = np.load(official_pred_npy).astype(np.float32, copy=False)
        else:
            official_pred = infer_dav2(model, official_bgr, args.input_size)
            np.save(official_pred_npy, official_pred)

        if teacher_pred_npy.is_file() and not args.overwrite_preds:
            teacher_pred = np.load(teacher_pred_npy).astype(np.float32, copy=False)
        else:
            teacher_pred = infer_dav2(model, teacher_full[..., ::-1], args.input_size)
            np.save(teacher_pred_npy, teacher_pred)

        if enhanced_pred_npy.is_file() and not args.overwrite_preds:
            enhanced_pred = np.load(enhanced_pred_npy).astype(np.float32, copy=False)
        else:
            enhanced_pred = infer_dav2(model, enhanced_full[..., ::-1], args.input_size)
            np.save(enhanced_pred_npy, enhanced_pred)

        official_align = align_relative_inverse_to_depth(
            official_pred,
            sparse,
            prior_mask,
            z_min=args.z_min,
            z_max=args.z_max,
        )
        teacher_align = align_relative_inverse_to_depth(
            teacher_pred,
            sparse,
            prior_mask,
            z_min=args.z_min,
            z_max=args.z_max,
        )
        enhanced_align = align_relative_inverse_to_depth(
            enhanced_pred,
            sparse,
            prior_mask,
            z_min=args.z_min,
            z_max=args.z_max,
        )
        official_metrics = depth_metrics(official_align["aligned_depth"], sparse, holdout_mask)
        teacher_metrics = depth_metrics(teacher_align["aligned_depth"], sparse, holdout_mask)
        enhanced_metrics = depth_metrics(enhanced_align["aligned_depth"], sparse, holdout_mask)

        all_inv_values.append(official_align["aligned_inv"][np.isfinite(official_align["aligned_inv"])])
        all_inv_values.append(teacher_align["aligned_inv"][np.isfinite(teacher_align["aligned_inv"])])
        all_inv_values.append(enhanced_align["aligned_inv"][np.isfinite(enhanced_align["aligned_inv"])])
        all_inv_values.append((1.0 / sparse[valid_sparse]).astype(np.float32))

        rec = dict(item)
        rec.update(
            {
                "official_rgb": official_bgr,
                "teacher_rgb_full": teacher_full,
                "enhanced_rgb_full": enhanced_full,
                "sparse_depth": sparse,
                "valid_sparse_mask": valid_sparse,
                "prior_mask": prior_mask,
                "holdout_mask": holdout_mask,
                "official_align": official_align,
                "teacher_align": teacher_align,
                "enhanced_align": enhanced_align,
                "official_aligned_inv": official_align["aligned_inv"],
                "teacher_aligned_inv": teacher_align["aligned_inv"],
                "enhanced_aligned_inv": enhanced_align["aligned_inv"],
                "official_metrics": official_metrics,
                "teacher_metrics": teacher_metrics,
                "enhanced_metrics": enhanced_metrics,
                "official_pred_npy": official_pred_npy,
                "teacher_rgb_png": teacher_rgb_png,
                "teacher_pred_npy": teacher_pred_npy,
                "enhanced_rgb_png": enhanced_rgb_png,
                "enhanced_pred_npy": enhanced_pred_npy,
            }
        )
        records.append(rec)
        print(
            f"[metric] {sample_id} dataset AbsRel={official_metrics['absrel']:.4f} D1={official_metrics['d1']:.4f}; "
            f"teacher AbsRel={teacher_metrics['absrel']:.4f} D1={teacher_metrics['d1']:.4f}; "
            f"enhanced AbsRel={enhanced_metrics['absrel']:.4f} D1={enhanced_metrics['d1']:.4f}",
            flush=True,
        )

    inv_concat = np.concatenate([arr.reshape(-1).astype(np.float32, copy=False) for arr in all_inv_values])
    inv_concat = inv_concat[np.isfinite(inv_concat)]
    vmin = float(np.percentile(inv_concat, args.viz_vmin_pct))
    vmax = float(np.percentile(inv_concat, args.viz_vmax_pct))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    csv_path = args.output_dir / "selected_dark_overexp_10_metrics.csv"
    json_path = args.output_dir / "summary.json"
    cmap_token = str(args.cmap).lower()
    sheet_path = (
        args.output_dir
        / f"stf_teacher_rgb_enhanced_dav2_{int(args.dark_count)}dark_{int(args.overexp_count)}overexp_{cmap_token}.png"
    )
    write_csv(csv_path, records)
    write_summary(json_path, records, args, vmin, vmax)
    make_contact_sheet(
        records,
        sheet_path,
        vmin=vmin,
        vmax=vmax,
        cmap_name=args.cmap,
        tile_width=args.tile_width,
        gt_radius=args.gt_radius,
    )
    print(f"[done] sheet={sheet_path}", flush=True)
    print(f"[done] csv={csv_path}", flush=True)
    print(f"[done] summary={json_path}", flush=True)


if __name__ == "__main__":
    main()
