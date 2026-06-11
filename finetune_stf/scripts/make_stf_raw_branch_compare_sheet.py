#!/usr/bin/env python3
"""Make a compact STF dark comparison sheet for one RAW rendering branch."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.scripts.make_stf_teacher_rgb_dav2_dark_overexp_compare import (  # noqa: E402
    align_relative_inverse_to_depth,
    colorize_dense,
    colorize_sparse_gt,
    depth_metrics,
    get_font,
    holdout_split,
    load_sparse_depth,
    read_bgr,
    sparse_valid_mask,
    truncate_text,
)


DEFAULT_SELECTED_CSV = (
    PROJECT_ROOT
    / "finetune_stf"
    / "analysis"
    / "0611_stf_dark_enhance_dav2_compare"
    / "selected_dark_overexp_10_metrics.csv"
)
DEFAULT_SWEEP_ROOT = (
    PROJECT_ROOT
    / "finetune_stf"
    / "analysis"
    / "0611_stf_raw_render_focus_sweep_dark10"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--branch", default="raw_b05_g075_us085")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--cmap", default="Spectral_r")
    parser.add_argument("--viz-vmin-pct", type=float, default=1.0)
    parser.add_argument("--viz-vmax-pct", type=float, default=99.0)
    parser.add_argument("--tile-width", type=int, default=384)
    parser.add_argument("--gt-radius", type=int, default=5)
    parser.add_argument("--z-min", type=float, default=1.0)
    parser.add_argument("--z-max", type=float, default=80.0)
    parser.add_argument("--min-prior-points", type=int, default=50)
    parser.add_argument("--min-holdout-points", type=int, default=50)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    return parser.parse_args()


def read_selected_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("selection_group") != "dark":
                continue
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "rgb_path": Path(row["rgb_path"]),
                    "sparse_depth_path": Path(row["sparse_depth_path"]),
                    "rgb_luma_mean": float(row["rgb_luma_mean"]),
                    "rgb_luma_p99": float(row["rgb_luma_p99"]),
                    "rgb_sat_ratio": float(row["rgb_sat_ratio"]),
                    "num_sparse_in_range_points": int(row["num_sparse_in_range_points"]),
                }
            )
    if not rows:
        raise ValueError(f"No dark rows in {path}")
    return rows


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def resize_tile(image: Image.Image, tile_w: int, tile_h: int, resample: int = Image.Resampling.BILINEAR) -> Image.Image:
    return image.resize((int(tile_w), int(tile_h)), resample=resample)


def make_sheet(
    records: list[dict[str, Any]],
    output_path: Path,
    *,
    branch: str,
    vmin: float,
    vmax: float,
    cmap_name: str,
    tile_width: int,
    gt_radius: int,
) -> None:
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
        branch,
        "GT sparse inv",
        "DAV2-L dataset RGB inv",
        f"DAV2-L {branch} inv",
    ]
    sheet_w = margin * 2 + len(cols) * tile_w + (len(cols) - 1) * gap
    sheet_h = title_h + header_h + len(records) * row_h + margin
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    title_font = get_font(20)
    header_font = get_font(15)
    text_font = get_font(13)

    draw.text((margin, 12), f"STF dark RAW branch compare: {branch}", fill=(0, 0, 0), font=title_font)
    draw.text(
        (margin, 40),
        f"Colormap={cmap_name}; shared inverse scale vmin={vmin:.5g}, vmax={vmax:.5g}; larger inverse means closer.",
        fill=(35, 35, 35),
        font=text_font,
    )
    bar_w = min(420, sheet_w - 2 * margin)
    grad = np.linspace(vmin, vmax, bar_w, dtype=np.float32).reshape(1, -1)
    bar = colorize_dense(grad, vmin=vmin, vmax=vmax, cmap_name=cmap_name).resize((bar_w, 14))
    sheet.paste(bar, (margin, 62))
    draw.text((margin + bar_w + 10, 58), "far -> near", fill=(35, 35, 35), font=text_font)

    y = title_h
    for col_idx, col in enumerate(cols):
        x = margin + col_idx * (tile_w + gap)
        draw.text((x, y + 5), truncate_text(draw, col, header_font, tile_w), fill=(0, 0, 0), font=header_font)

    y += header_h
    for rec in records:
        dataset = rec["dataset_metrics"]
        branch_metrics = rec["branch_metrics"]
        caption = (
            f"dark | {rec['sample_id']} | mean={rec['rgb_luma_mean']:.1f} "
            f"p99={rec['rgb_luma_p99']:.1f} sat={rec['rgb_sat_ratio']:.4f} | "
            f"dataset AbsRel={dataset['absrel']:.4f} D1={dataset['d1']:.4f} | "
            f"{branch} AbsRel={branch_metrics['absrel']:.4f} D1={branch_metrics['d1']:.4f}"
        )
        draw.text((margin, y + 7), truncate_text(draw, caption, text_font, sheet_w - 2 * margin), fill=(20, 20, 20), font=text_font)
        y_tiles = y + caption_h
        images = [
            Image.fromarray(rec["dataset_rgb"], mode="RGB"),
            Image.fromarray(rec["branch_rgb"], mode="RGB"),
            colorize_sparse_gt(
                rec["sparse_depth"],
                rec["valid_sparse_mask"],
                vmin=vmin,
                vmax=vmax,
                cmap_name=cmap_name,
                radius=gt_radius,
            ),
            colorize_dense(rec["dataset_aligned_inv"], vmin=vmin, vmax=vmax, cmap_name=cmap_name),
            colorize_dense(rec["branch_aligned_inv"], vmin=vmin, vmax=vmax, cmap_name=cmap_name),
        ]
        for col_idx, image in enumerate(images):
            x = margin + col_idx * (tile_w + gap)
            sheet.paste(resize_tile(image, tile_w, tile_h), (x, y_tiles))
        y += row_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    branch = args.branch
    if branch == "raw_b05_g075_us08":
        branch = "raw_b05_g075_us085"

    if args.output_path is None:
        args.output_path = args.sweep_root / f"stf_{branch}_compare_10dark_spectral_r.png"

    rows = read_selected_rows(args.selected_csv)
    records: list[dict[str, Any]] = []
    all_inv_values: list[np.ndarray] = []
    for row in rows:
        sample_id = row["sample_id"]
        dataset_pred_path = args.sweep_root / "dav2l_pred" / "dataset_rgb" / f"{sample_id}.npy"
        branch_pred_path = args.sweep_root / "dav2l_pred" / branch / f"{sample_id}.npy"
        branch_rgb_path = args.sweep_root / "rendered_rgb" / branch / f"{sample_id}.png"
        for path in (dataset_pred_path, branch_pred_path, branch_rgb_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        dataset_bgr = read_bgr(row["rgb_path"])
        dataset_rgb = dataset_bgr[..., ::-1]
        branch_rgb = load_rgb(branch_rgb_path)
        sparse_depth = load_sparse_depth(row["sparse_depth_path"])
        valid_sparse_mask = sparse_valid_mask(sparse_depth, args.z_min, args.z_max)
        prior_mask, holdout_mask = holdout_split(
            sample_id,
            valid_sparse_mask,
            args.holdout_fraction,
            args.min_holdout_points,
            args.min_prior_points,
        )
        dataset_pred = np.load(dataset_pred_path).astype(np.float32, copy=False)
        branch_pred = np.load(branch_pred_path).astype(np.float32, copy=False)
        dataset_align = align_relative_inverse_to_depth(
            dataset_pred,
            sparse_depth,
            prior_mask,
            z_min=args.z_min,
            z_max=args.z_max,
        )
        branch_align = align_relative_inverse_to_depth(
            branch_pred,
            sparse_depth,
            prior_mask,
            z_min=args.z_min,
            z_max=args.z_max,
        )
        dataset_metrics = depth_metrics(dataset_align["aligned_depth"], sparse_depth, holdout_mask)
        branch_metrics = depth_metrics(branch_align["aligned_depth"], sparse_depth, holdout_mask)
        all_inv_values.append(dataset_align["aligned_inv"][np.isfinite(dataset_align["aligned_inv"])])
        all_inv_values.append(branch_align["aligned_inv"][np.isfinite(branch_align["aligned_inv"])])
        all_inv_values.append((1.0 / sparse_depth[valid_sparse_mask]).astype(np.float32))
        rec = dict(row)
        rec.update(
            {
                "dataset_rgb": dataset_rgb,
                "branch_rgb": branch_rgb,
                "sparse_depth": sparse_depth,
                "valid_sparse_mask": valid_sparse_mask,
                "dataset_aligned_inv": dataset_align["aligned_inv"],
                "branch_aligned_inv": branch_align["aligned_inv"],
                "dataset_metrics": dataset_metrics,
                "branch_metrics": branch_metrics,
            }
        )
        records.append(rec)

    inv_concat = np.concatenate([arr.reshape(-1).astype(np.float32, copy=False) for arr in all_inv_values])
    inv_concat = inv_concat[np.isfinite(inv_concat)]
    vmin = float(np.percentile(inv_concat, args.viz_vmin_pct))
    vmax = float(np.percentile(inv_concat, args.viz_vmax_pct))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    make_sheet(
        records,
        args.output_path,
        branch=branch,
        vmin=vmin,
        vmax=vmax,
        cmap_name=args.cmap,
        tile_width=args.tile_width,
        gt_radius=args.gt_radius,
    )
    print(f"[done] {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
