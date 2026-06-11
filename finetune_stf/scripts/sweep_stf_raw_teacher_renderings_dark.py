#!/usr/bin/env python3
"""Sweep STF RAW rendering recipes on dark samples and compare DAV2-L depth."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
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

from finetune_stf.scripts.make_stf_teacher_rgb_dav2_dark_overexp_compare import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    DEFAULT_RAW_NPZ_ROOT,
    TEACHER_GAINS,
    align_relative_inverse_to_depth,
    colorize_dense,
    colorize_sparse_gt,
    depth_metrics,
    get_font,
    holdout_split,
    infer_dav2,
    load_model,
    load_sparse_depth,
    load_stf_bayer,
    read_bgr,
    read_manifest,
    select_samples,
    sparse_valid_mask,
    stf_bayer_to_rgb01,
    truncate_text,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "finetune_stf" / "analysis" / "0611_stf_raw_render_sweep_dark10"
)
DEFAULT_SELECTED_CSV = (
    PROJECT_ROOT
    / "finetune_stf"
    / "analysis"
    / "0611_stf_dark_enhance_dav2_compare"
    / "selected_dark_overexp_10_metrics.csv"
)


@dataclass(frozen=True)
class RenderRecipe:
    name: str
    raw_normalization: str
    black_percentile: float
    white_percentile: float
    gamma: float
    gains: tuple[float, float, float] = tuple(float(v) for v in TEACHER_GAINS.tolist())
    black_mode: str = "global"
    unsharp_sigma: float = 0.0
    unsharp_amount: float = 0.0


COARSE_RECIPES = (
    RenderRecipe(
        name="raw_rod_like_g045",
        raw_normalization="companded",
        black_percentile=0.0,
        white_percentile=99.5,
        gamma=0.454545,
    ),
    RenderRecipe(
        name="raw_black01_g065",
        raw_normalization="companded",
        black_percentile=1.0,
        white_percentile=99.7,
        gamma=0.65,
    ),
    RenderRecipe(
        name="raw_black05_g075",
        raw_normalization="companded",
        black_percentile=5.0,
        white_percentile=99.8,
        gamma=0.75,
    ),
    RenderRecipe(
        name="raw_black10_g085",
        raw_normalization="companded",
        black_percentile=10.0,
        white_percentile=99.9,
        gamma=0.85,
    ),
    RenderRecipe(
        name="raw_black05_g075_unsharp",
        raw_normalization="companded",
        black_percentile=5.0,
        white_percentile=99.8,
        gamma=0.75,
        unsharp_sigma=1.1,
        unsharp_amount=0.85,
    ),
    RenderRecipe(
        name="raw_decomp_black05_g075",
        raw_normalization="decompanded",
        black_percentile=5.0,
        white_percentile=99.8,
        gamma=0.75,
    ),
)

FOCUS_RECIPES = (
    RenderRecipe(
        name="raw_rod_like_g045",
        raw_normalization="companded",
        black_percentile=0.0,
        white_percentile=99.5,
        gamma=0.454545,
    ),
    RenderRecipe(
        name="raw_b03_g065_us06",
        raw_normalization="companded",
        black_percentile=3.0,
        white_percentile=99.8,
        gamma=0.65,
        unsharp_sigma=1.1,
        unsharp_amount=0.6,
    ),
    RenderRecipe(
        name="raw_b03_g070_us06",
        raw_normalization="companded",
        black_percentile=3.0,
        white_percentile=99.8,
        gamma=0.70,
        unsharp_sigma=1.1,
        unsharp_amount=0.6,
    ),
    RenderRecipe(
        name="raw_b03_g075_us06",
        raw_normalization="companded",
        black_percentile=3.0,
        white_percentile=99.8,
        gamma=0.75,
        unsharp_sigma=1.1,
        unsharp_amount=0.6,
    ),
    RenderRecipe(
        name="raw_b05_g070_us06",
        raw_normalization="companded",
        black_percentile=5.0,
        white_percentile=99.8,
        gamma=0.70,
        unsharp_sigma=1.1,
        unsharp_amount=0.6,
    ),
    RenderRecipe(
        name="raw_b05_g075_us06",
        raw_normalization="companded",
        black_percentile=5.0,
        white_percentile=99.8,
        gamma=0.75,
        unsharp_sigma=1.1,
        unsharp_amount=0.6,
    ),
    RenderRecipe(
        name="raw_b05_g075_us085",
        raw_normalization="companded",
        black_percentile=5.0,
        white_percentile=99.8,
        gamma=0.75,
        unsharp_sigma=1.1,
        unsharp_amount=0.85,
    ),
    RenderRecipe(
        name="raw_b05_g080_us06",
        raw_normalization="companded",
        black_percentile=5.0,
        white_percentile=99.8,
        gamma=0.80,
        unsharp_sigma=1.1,
        unsharp_amount=0.6,
    ),
    RenderRecipe(
        name="raw_b08_g080_us06",
        raw_normalization="companded",
        black_percentile=8.0,
        white_percentile=99.85,
        gamma=0.80,
        unsharp_sigma=1.1,
        unsharp_amount=0.6,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--raw-npz-root", type=Path, default=DEFAULT_RAW_NPZ_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--encoder", default="vitl", choices=("vits", "vitb", "vitl", "vitg"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=924)
    parser.add_argument("--dark-count", type=int, default=10)
    parser.add_argument("--z-min", type=float, default=1.0)
    parser.add_argument("--z-max", type=float, default=80.0)
    parser.add_argument("--min-sparse-points", type=int, default=100)
    parser.add_argument("--min-prior-points", type=int, default=50)
    parser.add_argument("--min-holdout-points", type=int, default=50)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--cmap", default="Spectral_r")
    parser.add_argument("--viz-vmin-pct", type=float, default=1.0)
    parser.add_argument("--viz-vmax-pct", type=float, default=99.0)
    parser.add_argument("--tile-width", type=int, default=260)
    parser.add_argument("--gt-radius", type=int, default=5)
    parser.add_argument("--recipe-set", choices=("coarse", "focus"), default="coarse")
    parser.add_argument("--overwrite-preds", action="store_true")
    return parser.parse_args()


def load_selected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.selected_csv and args.selected_csv.is_file():
        rows: list[dict[str, Any]] = []
        with args.selected_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("selection_group") != "dark":
                    continue
                sample_id = row["sample_id"]
                rows.append(
                    {
                        "sample_id": sample_id,
                        "split": row["split"],
                        "selection_group": "dark",
                        "rgb_path": Path(row["rgb_path"]),
                        "raw_npz_path": Path(row["raw_npz_path"]),
                        "sparse_depth_path": Path(row["sparse_depth_path"]),
                        "rgb_luma_mean": float(row["rgb_luma_mean"]),
                        "rgb_luma_p99": float(row["rgb_luma_p99"]),
                        "rgb_luma_p999": float(row["rgb_luma_p999"]),
                        "rgb_sat_ratio": float(row["rgb_sat_ratio"]),
                        "num_sparse_in_range_points": int(row["num_sparse_in_range_points"]),
                    }
                )
                if len(rows) >= int(args.dark_count):
                    break
        if len(rows) == int(args.dark_count):
            print(f"[select] loaded {len(rows)} dark rows from {args.selected_csv}", flush=True)
            return rows
        print(f"[select] selected CSV had {len(rows)} dark rows; falling back to manifest scan", flush=True)

    rows = read_manifest(args.manifest)
    select_args = argparse.Namespace(
        raw_npz_root=args.raw_npz_root,
        z_min=args.z_min,
        z_max=args.z_max,
        min_sparse_points=args.min_sparse_points,
        dark_count=args.dark_count,
        overexp_count=0,
    )
    return select_samples(rows, select_args)


def stretch_with_black_white(rgb: np.ndarray, recipe: RenderRecipe) -> np.ndarray:
    gains = np.asarray(recipe.gains, dtype=np.float32).reshape(1, 1, 3)
    x = np.clip(rgb.astype(np.float32) * gains, 0.0, 1.0)

    if recipe.black_mode == "per_channel":
        axes = (0, 1)
        black = np.percentile(x, recipe.black_percentile, axis=axes, keepdims=True)
        white = np.percentile(x, recipe.white_percentile, axis=axes, keepdims=True)
    elif recipe.black_mode == "global":
        black = float(np.percentile(x, recipe.black_percentile))
        white = float(np.percentile(x, recipe.white_percentile))
    else:
        raise ValueError(f"Unsupported black_mode: {recipe.black_mode!r}")

    denom = np.maximum(np.asarray(white, dtype=np.float32) - np.asarray(black, dtype=np.float32), 1e-6)
    x = np.clip((x - black) / denom, 0.0, 1.0)
    x = np.power(x, float(recipe.gamma), dtype=np.float32)
    out = np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)

    if recipe.unsharp_amount > 0.0 and recipe.unsharp_sigma > 0.0:
        bgr = out[..., ::-1]
        blur = cv2.GaussianBlur(bgr, (0, 0), float(recipe.unsharp_sigma))
        sharp = cv2.addWeighted(bgr, 1.0 + float(recipe.unsharp_amount), blur, -float(recipe.unsharp_amount), 0)
        out = sharp[..., ::-1]
    return out


def render_recipe(bayer: np.ndarray, recipe: RenderRecipe, target_hw: tuple[int, int]) -> np.ndarray:
    rgb01 = stf_bayer_to_rgb01(bayer, raw_normalization=recipe.raw_normalization)
    native_rgb = stretch_with_black_white(rgb01, recipe)
    target_h, target_w = target_hw
    return cv2.resize(native_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def resize_tile(image: Image.Image, tile_w: int, tile_h: int, resample: int = Image.Resampling.BILINEAR) -> Image.Image:
    return image.resize((int(tile_w), int(tile_h)), resample=resample)


def make_render_sheet(records: list[dict[str, Any]], output_path: Path, recipes: tuple[RenderRecipe, ...], tile_width: int) -> None:
    height, width = records[0]["official_bgr"].shape[:2]
    tile_w = int(tile_width)
    tile_h = int(round(tile_w * height / width))
    cols = ["dataset RGB", *[r.name for r in recipes]]
    margin = 16
    gap = 8
    title_h = 72
    header_h = 34
    caption_h = 32
    row_h = caption_h + tile_h + gap
    sheet_w = margin * 2 + len(cols) * tile_w + (len(cols) - 1) * gap
    sheet_h = title_h + header_h + len(records) * row_h + margin
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    title_font = get_font(19)
    header_font = get_font(12)
    text_font = get_font(12)
    draw.text((margin, 12), "STF dark RAW rendering sweep", fill=(0, 0, 0), font=title_font)
    draw.text(
        (margin, 40),
        "Black-percentile removal is intended to reduce the milky veil before DAV2-L inference.",
        fill=(35, 35, 35),
        font=text_font,
    )
    y = title_h
    for col_idx, col in enumerate(cols):
        x = margin + col_idx * (tile_w + gap)
        draw.text((x, y + 8), truncate_text(draw, col, header_font, tile_w), fill=(0, 0, 0), font=header_font)
    y += header_h
    for rec in records:
        caption = (
            f"{rec['sample_id']} | mean={rec['rgb_luma_mean']:.1f} "
            f"p99={rec['rgb_luma_p99']:.1f} sparse={rec['num_sparse_in_range_points']}"
        )
        draw.text((margin, y + 6), truncate_text(draw, caption, text_font, sheet_w - 2 * margin), fill=(20, 20, 20), font=text_font)
        y_tile = y + caption_h
        images = [Image.fromarray(rec["official_bgr"][..., ::-1], mode="RGB")]
        images.extend(Image.fromarray(rec["renders"][recipe.name], mode="RGB") for recipe in recipes)
        for col_idx, image in enumerate(images):
            x = margin + col_idx * (tile_w + gap)
            sheet.paste(resize_tile(image, tile_w, tile_h), (x, y_tile))
        y += row_h
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def make_depth_sheet(
    records: list[dict[str, Any]],
    output_path: Path,
    recipes: tuple[RenderRecipe, ...],
    *,
    vmin: float,
    vmax: float,
    cmap_name: str,
    tile_width: int,
    gt_radius: int,
) -> None:
    height, width = records[0]["sparse_depth"].shape
    tile_w = int(tile_width)
    tile_h = int(round(tile_w * height / width))
    cols = ["GT sparse inv", "dataset RGB pred", *[f"{r.name} pred" for r in recipes]]
    margin = 16
    gap = 8
    title_h = 84
    header_h = 34
    caption_h = 42
    row_h = caption_h + tile_h + gap
    sheet_w = margin * 2 + len(cols) * tile_w + (len(cols) - 1) * gap
    sheet_h = title_h + header_h + len(records) * row_h + margin
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    title_font = get_font(19)
    header_font = get_font(12)
    text_font = get_font(12)
    draw.text((margin, 12), "STF dark DAV2-L predictions from RAW rendering sweep", fill=(0, 0, 0), font=title_font)
    draw.text(
        (margin, 40),
        f"Colormap={cmap_name}; shared inverse scale vmin={vmin:.5g}, vmax={vmax:.5g}; larger inverse means closer.",
        fill=(35, 35, 35),
        font=text_font,
    )
    bar_w = min(360, sheet_w - 2 * margin)
    grad = np.linspace(vmin, vmax, bar_w, dtype=np.float32).reshape(1, -1)
    sheet.paste(colorize_dense(grad, vmin=vmin, vmax=vmax, cmap_name=cmap_name).resize((bar_w, 12)), (margin, 62))
    y = title_h
    for col_idx, col in enumerate(cols):
        x = margin + col_idx * (tile_w + gap)
        draw.text((x, y + 8), truncate_text(draw, col, header_font, tile_w), fill=(0, 0, 0), font=header_font)
    y += header_h
    for rec in records:
        best_name = rec["best_absrel_branch"]
        caption = (
            f"{rec['sample_id']} | best AbsRel={best_name}:{rec['metrics'][best_name]['absrel']:.4f} | "
            f"dataset={rec['metrics']['dataset_rgb']['absrel']:.4f}/{rec['metrics']['dataset_rgb']['d1']:.3f}"
        )
        draw.text((margin, y + 6), truncate_text(draw, caption, text_font, sheet_w - 2 * margin), fill=(20, 20, 20), font=text_font)
        y_tile = y + caption_h
        images = [
            colorize_sparse_gt(
                rec["sparse_depth"],
                rec["valid_sparse_mask"],
                vmin=vmin,
                vmax=vmax,
                cmap_name=cmap_name,
                radius=gt_radius,
            ),
            colorize_dense(rec["aligned_inv"]["dataset_rgb"], vmin=vmin, vmax=vmax, cmap_name=cmap_name),
        ]
        images.extend(
            colorize_dense(rec["aligned_inv"][recipe.name], vmin=vmin, vmax=vmax, cmap_name=cmap_name)
            for recipe in recipes
        )
        for col_idx, image in enumerate(images):
            x = margin + col_idx * (tile_w + gap)
            sheet.paste(resize_tile(image, tile_w, tile_h), (x, y_tile))
        y += row_h
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else math.nan


def write_metrics_csv(path: Path, records: list[dict[str, Any]], branches: list[str]) -> None:
    fields = ["sample_id", "split", "branch", "absrel", "d1", "eval_points", "affine_scale", "affine_shift", "pred_npy", "rgb_png"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            for branch in branches:
                metric = rec["metrics"][branch]
                align = rec["align"][branch]
                writer.writerow(
                    {
                        "sample_id": rec["sample_id"],
                        "split": rec["split"],
                        "branch": branch,
                        "absrel": f"{metric['absrel']:.9g}",
                        "d1": f"{metric['d1']:.9g}",
                        "eval_points": int(metric["eval_points"]),
                        "affine_scale": f"{align['affine_scale']:.9g}",
                        "affine_shift": f"{align['affine_shift']:.9g}",
                        "pred_npy": rec["pred_paths"][branch],
                        "rgb_png": rec["rgb_paths"].get(branch, ""),
                    }
                )


def main() -> None:
    args = parse_args()
    if int(args.input_size) % 14 != 0:
        raise ValueError("--input-size must be a multiple of 14")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = args.output_dir / "rendered_rgb"
    pred_dir = args.output_dir / "dav2l_pred"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    selected = load_selected_rows(args)
    print(f"[setup] samples={len(selected)} output={args.output_dir}", flush=True)
    for row in selected:
        print(
            f"  dark {row['sample_id']} mean={row['rgb_luma_mean']:.2f} "
            f"p99={row['rgb_luma_p99']:.2f} sparse={row['num_sparse_in_range_points']}",
            flush=True,
        )

    device = torch.device(args.device)
    model = load_model(args, device)
    recipes = FOCUS_RECIPES if args.recipe_set == "focus" else COARSE_RECIPES
    branches = ["dataset_rgb", *[r.name for r in recipes]]
    records: list[dict[str, Any]] = []
    all_inv_values: list[np.ndarray] = []

    for idx, item in enumerate(selected, start=1):
        sample_id = item["sample_id"]
        print(f"[infer] {idx}/{len(selected)} {sample_id}", flush=True)
        official_bgr = read_bgr(item["rgb_path"])
        sparse = load_sparse_depth(item["sparse_depth_path"])
        valid_sparse = sparse_valid_mask(sparse, args.z_min, args.z_max)
        prior_mask, holdout_mask = holdout_split(
            sample_id,
            valid_sparse,
            args.holdout_fraction,
            args.min_holdout_points,
            args.min_prior_points,
        )
        bayer = load_stf_bayer(item["raw_npz_path"])

        rendered: dict[str, np.ndarray] = {}
        rgb_paths: dict[str, str] = {}
        pred_paths: dict[str, str] = {}
        pred_rel: dict[str, np.ndarray] = {}

        dataset_pred_path = pred_dir / "dataset_rgb" / f"{sample_id}.npy"
        dataset_pred_path.parent.mkdir(parents=True, exist_ok=True)
        if dataset_pred_path.is_file() and not args.overwrite_preds:
            pred_rel["dataset_rgb"] = np.load(dataset_pred_path).astype(np.float32, copy=False)
        else:
            pred_rel["dataset_rgb"] = infer_dav2(model, official_bgr, args.input_size)
            np.save(dataset_pred_path, pred_rel["dataset_rgb"])
        pred_paths["dataset_rgb"] = str(dataset_pred_path)

        for recipe in recipes:
            rendered_rgb = render_recipe(bayer, recipe, official_bgr.shape[:2])
            rendered[recipe.name] = rendered_rgb
            branch_rgb_dir = rgb_dir / recipe.name
            branch_pred_dir = pred_dir / recipe.name
            branch_rgb_dir.mkdir(parents=True, exist_ok=True)
            branch_pred_dir.mkdir(parents=True, exist_ok=True)
            rgb_path = branch_rgb_dir / f"{sample_id}.png"
            pred_path = branch_pred_dir / f"{sample_id}.npy"
            cv2.imwrite(str(rgb_path), rendered_rgb[..., ::-1])
            if pred_path.is_file() and not args.overwrite_preds:
                pred_rel[recipe.name] = np.load(pred_path).astype(np.float32, copy=False)
            else:
                pred_rel[recipe.name] = infer_dav2(model, rendered_rgb[..., ::-1], args.input_size)
                np.save(pred_path, pred_rel[recipe.name])
            rgb_paths[recipe.name] = str(rgb_path)
            pred_paths[recipe.name] = str(pred_path)

        align: dict[str, dict[str, Any]] = {}
        metrics: dict[str, dict[str, Any]] = {}
        aligned_inv: dict[str, np.ndarray] = {}
        for branch in branches:
            align[branch] = align_relative_inverse_to_depth(
                pred_rel[branch],
                sparse,
                prior_mask,
                z_min=args.z_min,
                z_max=args.z_max,
            )
            aligned_inv[branch] = align[branch]["aligned_inv"]
            metrics[branch] = depth_metrics(align[branch]["aligned_depth"], sparse, holdout_mask)
            all_inv_values.append(aligned_inv[branch][np.isfinite(aligned_inv[branch])])
        all_inv_values.append((1.0 / sparse[valid_sparse]).astype(np.float32))

        best_absrel_branch = min(branches, key=lambda b: float(metrics[b]["absrel"]))
        record = dict(item)
        record.update(
            {
                "official_bgr": official_bgr,
                "renders": rendered,
                "sparse_depth": sparse,
                "valid_sparse_mask": valid_sparse,
                "align": align,
                "metrics": metrics,
                "aligned_inv": aligned_inv,
                "rgb_paths": rgb_paths,
                "pred_paths": pred_paths,
                "best_absrel_branch": best_absrel_branch,
            }
        )
        records.append(record)
        pieces = [f"{b} AbsRel={metrics[b]['absrel']:.4f} D1={metrics[b]['d1']:.3f}" for b in branches]
        print("[metric] " + sample_id + " | " + " | ".join(pieces), flush=True)

    inv_concat = np.concatenate([arr.reshape(-1).astype(np.float32, copy=False) for arr in all_inv_values])
    inv_concat = inv_concat[np.isfinite(inv_concat)]
    vmin = float(np.percentile(inv_concat, args.viz_vmin_pct))
    vmax = float(np.percentile(inv_concat, args.viz_vmax_pct))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    render_sheet = args.output_dir / "stf_raw_render_variants_10dark.png"
    depth_sheet = args.output_dir / "stf_raw_render_dav2l_depth_variants_10dark_spectral_r.png"
    metrics_csv = args.output_dir / "raw_render_sweep_metrics_long.csv"
    summary_json = args.output_dir / "summary.json"
    make_render_sheet(records, render_sheet, recipes, args.tile_width)
    make_depth_sheet(
        records,
        depth_sheet,
        recipes,
        vmin=vmin,
        vmax=vmax,
        cmap_name=args.cmap,
        tile_width=args.tile_width,
        gt_radius=args.gt_radius,
    )
    write_metrics_csv(metrics_csv, records, branches)

    aggregate = {
        branch: {
            "mean_absrel": finite_mean([r["metrics"][branch]["absrel"] for r in records]),
            "mean_d1": finite_mean([r["metrics"][branch]["d1"] for r in records]),
            "wins_by_absrel": int(sum(1 for r in records if r["best_absrel_branch"] == branch)),
        }
        for branch in branches
    }
    summary = {
        "config": {
            "selected_csv": str(args.selected_csv.resolve()) if args.selected_csv else "",
            "output_dir": str(args.output_dir.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "encoder": args.encoder,
            "input_size": int(args.input_size),
            "metric_definition": (
                "Affinely align DAV2-L relative inverse output to sparse LiDAR inverse depth on deterministic "
                "prior points, then compute AbsRel and D1 on held-out sparse points."
            ),
            "visualization": {
                "cmap": args.cmap,
                "shared_inverse_vmin": vmin,
                "shared_inverse_vmax": vmax,
            },
            "recipes": [asdict(r) for r in recipes],
        },
        "aggregate": aggregate,
        "best_by_mean_absrel": min(aggregate, key=lambda b: aggregate[b]["mean_absrel"]),
        "best_by_mean_d1": max(aggregate, key=lambda b: aggregate[b]["mean_d1"]),
        "samples": [
            {
                "sample_id": r["sample_id"],
                "split": r["split"],
                "best_absrel_branch": r["best_absrel_branch"],
                "metrics": r["metrics"],
            }
            for r in records
        ],
        "artifacts": {
            "render_sheet": str(render_sheet),
            "depth_sheet": str(depth_sheet),
            "metrics_csv": str(metrics_csv),
        },
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[aggregate]", json.dumps(aggregate, indent=2), flush=True)
    print(f"[done] render_sheet={render_sheet}", flush=True)
    print(f"[done] depth_sheet={depth_sheet}", flush=True)
    print(f"[done] metrics_csv={metrics_csv}", flush=True)
    print(f"[done] summary={summary_json}", flush=True)


if __name__ == "__main__":
    main()
