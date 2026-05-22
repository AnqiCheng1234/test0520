#!/usr/bin/env python3
"""Create qualitative examples where 0521_0835 improves image-edge abs_rel."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp
from anqi_eval.eval_stf_boundary_diagnostics import (
    EvalItem,
    config_for_item,
    dataset_for_item,
    image_edge_band,
    load_model_for_item,
    load_pseudo_manifest,
    predict_disp,
    sample_name,
    tensor_from_sample,
)


RUN_0306 = "0521_0306_stf_train_test_pseudovitl_rgb_lora_decoder_e5"
RUN_0835 = "0521_0835_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_lora_decoder_e10"
LABEL_0306 = "0521_0306 RGB LoRA"
LABEL_0835 = "0521_0835 RAW bridge LoRA"
DEFAULT_PSEUDO_MANIFEST = Path(
    "/mnt/drive/3333_raw/seeing_through_fog/"
    "pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/"
    "stf_rgb_lut_manifest_6216.csv"
)
DEFAULT_CKPT_ROOT = Path("/mnt/drive/3333_raw/0000_exp_ckpt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-sample",
        type=Path,
        default=Path(__file__).with_name("per_sample.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("examples_0835_better_image_edge_abs"),
    )
    parser.add_argument("--pseudo-manifest", type=Path, default=DEFAULT_PSEUDO_MANIFEST)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit STF val indices. Defaults to robust examples where 0835 improves edge abs_rel and edge d1.",
    )
    parser.add_argument("--image-edge-percentile", type=float, default=90.0)
    parser.add_argument("--image-edge-dilate", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=360)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    rows: dict[str, dict[str, dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["sample_name"], {})[row["item_id"]] = row
    return rows


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def select_examples(rows: dict[str, dict[str, dict[str, str]]], max_examples: int) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for name, pair in rows.items():
        if RUN_0306 not in pair or RUN_0835 not in pair:
            continue
        a = pair[RUN_0306]
        b = pair[RUN_0835]
        edge_a = f(a, "image_edge_band_abs_rel")
        edge_b = f(b, "image_edge_band_abs_rel")
        d1_a = f(a, "image_edge_band_d1")
        d1_b = f(b, "image_edge_band_d1")
        if not (math.isfinite(edge_a) and math.isfinite(edge_b)):
            continue
        if edge_b < edge_a and d1_b > d1_a:
            candidates.append(
                {
                    "index": int(a["index"]),
                    "sample_name": name,
                    "delta_edge_abs": edge_a - edge_b,
                    "row_0306": a,
                    "row_0835": b,
                }
            )
    candidates.sort(key=lambda item: float(item["delta_edge_abs"]), reverse=True)
    return candidates[:max_examples]


def rows_for_indices(rows: dict[str, dict[str, dict[str, str]]], indices: list[int]) -> list[dict[str, object]]:
    by_index = {}
    for name, pair in rows.items():
        if RUN_0306 in pair and RUN_0835 in pair:
            by_index[int(pair[RUN_0306]["index"])] = (name, pair)
    selected = []
    for idx in indices:
        name, pair = by_index[idx]
        edge_a = f(pair[RUN_0306], "image_edge_band_abs_rel")
        edge_b = f(pair[RUN_0835], "image_edge_band_abs_rel")
        selected.append(
            {
                "index": idx,
                "sample_name": name,
                "delta_edge_abs": edge_a - edge_b,
                "row_0306": pair[RUN_0306],
                "row_0835": pair[RUN_0835],
            }
        )
    return selected


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(18, bold=True)
FONT_TEXT = font(14)
FONT_SMALL = font(12)


def resize_rgb(arr: np.ndarray, width: int) -> Image.Image:
    h, w = arr.shape[:2]
    height = int(round(h * width / w))
    return Image.fromarray(arr).resize((width, height), Image.Resampling.BILINEAR)


def load_npz_array(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "arr_0" in data.files:
            return np.asarray(data["arr_0"], dtype=np.float32)
        return np.asarray(data[data.files[0]], dtype=np.float32)


def depth_tile(depth: np.ndarray, valid: np.ndarray, width: int, vmin: float, vmax: float, title: str, lines: list[str]) -> Image.Image:
    value = depth.astype(np.float32).copy()
    value[~valid] = np.nan
    norm = (np.clip(value, vmin, vmax) - vmin) / max(vmax - vmin, 1e-6)
    gray = np.nan_to_num(norm, nan=0.0)
    u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~valid] = 255
    return add_header(resize_rgb(rgb, width), title, lines)


def rgb_edge_tile(rgb: np.ndarray, edge_band: np.ndarray, valid_edge: np.ndarray, width: int, title: str, lines: list[str]) -> Image.Image:
    out = rgb.copy()
    overlay = out.copy()
    overlay[edge_band] = (255, 220, 40)
    out = cv2.addWeighted(overlay, 0.32, out, 0.68, 0.0)
    ys, xs = np.where(valid_edge)
    for x, y in zip(xs[:: max(1, len(xs) // 1800)], ys[:: max(1, len(ys) // 1800)]):
        cv2.circle(out, (int(x), int(y)), 1, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    return add_header(resize_rgb(out, width), title, lines)


def error_tile(
    rgb: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    width: int,
    title: str,
    lines: list[str],
    cap: float = 1.0,
) -> Image.Image:
    out = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
    ys, xs = np.where(mask)
    rel = np.abs(pred[mask].astype(np.float64) - gt[mask].astype(np.float64)) / np.maximum(gt[mask].astype(np.float64), 1e-6)
    values = np.clip(rel / cap, 0.0, 1.0)
    colors = cv2.applyColorMap(np.clip(values * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    step = max(1, len(xs) // 2400)
    for x, y, color in zip(xs[::step], ys[::step], colors[::step]):
        cv2.circle(out, (int(x), int(y)), 2, tuple(int(c) for c in color), -1, lineType=cv2.LINE_AA)
    return add_header(resize_rgb(out, width), title, lines)


def add_header(tile: Image.Image, title: str, lines: list[str]) -> Image.Image:
    header_h = 64
    canvas = Image.new("RGB", (tile.width, tile.height + header_h), "white")
    canvas.paste(tile, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(0, 0, 0), font=FONT_TITLE)
    for i, line in enumerate(lines[:2]):
        draw.text((8, 30 + 16 * i), line, fill=(40, 40, 40), font=FONT_SMALL)
    return canvas


def make_item(run_name: str, label: str, ckpt_root: Path) -> EvalItem:
    return EvalItem(
        item_id=run_name,
        label=label,
        config_path=PROJECT_ROOT / "finetune_stf/exp" / run_name / "config.json",
        checkpoint_path=ckpt_root / run_name / "best_model.pth",
        direct_mode=None,
        input_source="config",
    )


def predict_for_item(item: EvalItem, cfg: SimpleNamespace, model: torch.nn.Module, idx: int, device: torch.device):
    dataset = dataset_for_item(cfg, item, "val")
    sample = dataset[idx]
    name = sample_name(sample)
    gt = sample["depth"].detach().cpu().numpy().astype(np.float32, copy=False)
    valid = sample["valid_mask"].detach().cpu().numpy().astype(bool)
    target_hw = tuple(int(v) for v in gt.shape[-2:])
    tensor = tensor_from_sample(sample, cfg, item, device)
    pred_disp = predict_disp(model, tensor, target_hw)
    aligned, _ = affine_align_disp(gt, pred_disp, valid)
    return name, gt, valid, aligned


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.per_sample)
    examples = rows_for_indices(rows, args.indices) if args.indices else select_examples(rows, args.max_examples)
    if not examples:
        raise RuntimeError("No examples selected")

    device = torch.device(args.device)
    pseudo_rows = load_pseudo_manifest(args.pseudo_manifest, args.split)
    item_0306 = make_item(RUN_0306, LABEL_0306, args.checkpoint_root)
    item_0835 = make_item(RUN_0835, LABEL_0835, args.checkpoint_root)
    cfg_0306 = config_for_item(item_0306, item_0306.config_path)
    cfg_0835 = config_for_item(item_0835, item_0835.config_path)
    model_0306 = load_model_for_item(cfg_0306, item_0306, device)
    model_0835 = load_model_for_item(cfg_0835, item_0835, device)

    manifest_rows = []
    row_images = []
    for ex in examples:
        idx = int(ex["index"])
        name = str(ex["sample_name"])
        name_a, gt, valid, pred_0306 = predict_for_item(item_0306, cfg_0306, model_0306, idx, device)
        name_b, gt_b, valid_b, pred_0835 = predict_for_item(item_0835, cfg_0835, model_0835, idx, device)
        if name_a != name or name_b != name:
            raise RuntimeError(f"Index/name mismatch for index {idx}: {name}, {name_a}, {name_b}")
        if gt_b.shape != gt.shape or valid_b.shape != valid.shape:
            raise RuntimeError(f"Shape mismatch for {name}")

        pseudo = pseudo_rows[name]
        bgr = cv2.imread(pseudo["rgb_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(pseudo["rgb_path"])
        bgr = cv2.resize(bgr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        edge_band = image_edge_band(
            pseudo["rgb_path"],
            gt.shape,
            percentile=float(args.image_edge_percentile),
            dilate=int(args.image_edge_dilate),
        )
        edge_valid = valid & edge_band & np.isfinite(gt) & (gt > 0) & np.isfinite(pred_0306) & np.isfinite(pred_0835)
        depth_valid_0306 = np.isfinite(pred_0306) & (pred_0306 > 0)
        depth_valid_0835 = np.isfinite(pred_0835) & (pred_0835 > 0)
        finite_depth = np.concatenate([pred_0306[depth_valid_0306], pred_0835[depth_valid_0835]])
        vmin = float(np.percentile(finite_depth, 2)) if finite_depth.size else 1.0
        vmax = float(np.percentile(finite_depth, 98)) if finite_depth.size else 80.0
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 1.0, 80.0

        row_a = ex["row_0306"]
        row_b = ex["row_0835"]
        pts = int(round(f(row_a, "image_edge_band_points")))
        tiles = [
            rgb_edge_tile(
                rgb,
                edge_band,
                edge_valid,
                args.tile_width,
                f"{idx} {name}",
                [f"yellow=edge band, cyan=sparse pts", f"edge pts={pts}"],
            ),
            depth_tile(
                pred_0306,
                depth_valid_0306,
                args.tile_width,
                vmin,
                vmax,
                "0306 aligned depth",
                [f"edge abs_rel={f(row_a, 'image_edge_band_abs_rel'):.4f}", f"edge d1={f(row_a, 'image_edge_band_d1'):.4f}"],
            ),
            depth_tile(
                pred_0835,
                depth_valid_0835,
                args.tile_width,
                vmin,
                vmax,
                "0835 aligned depth",
                [f"edge abs_rel={f(row_b, 'image_edge_band_abs_rel'):.4f}", f"edge d1={f(row_b, 'image_edge_band_d1'):.4f}"],
            ),
            error_tile(
                rgb,
                gt,
                pred_0306,
                edge_valid,
                args.tile_width,
                "0306 edge rel error",
                [f"all abs_rel={f(row_a, 'abs_rel'):.4f}", "color cap rel=1.0"],
            ),
            error_tile(
                rgb,
                gt,
                pred_0835,
                edge_valid,
                args.tile_width,
                "0835 edge rel error",
                [f"all abs_rel={f(row_b, 'abs_rel'):.4f}", f"delta={float(ex['delta_edge_abs']):+.4f}"],
            ),
        ]
        row_w = sum(tile.width for tile in tiles)
        row_h = max(tile.height for tile in tiles)
        row = Image.new("RGB", (row_w, row_h), "white")
        x = 0
        for tile in tiles:
            row.paste(tile, (x, 0))
            x += tile.width
        row_path = args.output_dir / f"{idx:04d}_{name}_edge_abs_0835_better.png"
        row.save(row_path)
        row_images.append(row)

        manifest_rows.append(
            {
                "index": idx,
                "sample_name": name,
                "delta_edge_abs_rel_0306_minus_0835": float(ex["delta_edge_abs"]),
                "edge_abs_rel_0306": f(row_a, "image_edge_band_abs_rel"),
                "edge_abs_rel_0835": f(row_b, "image_edge_band_abs_rel"),
                "edge_d1_0306": f(row_a, "image_edge_band_d1"),
                "edge_d1_0835": f(row_b, "image_edge_band_d1"),
                "all_abs_rel_0306": f(row_a, "abs_rel"),
                "all_abs_rel_0835": f(row_b, "abs_rel"),
                "edge_points": pts,
                "panel": str(row_path),
            }
        )

    gap = 14
    atlas_w = max(row.width for row in row_images)
    atlas_h = sum(row.height for row in row_images) + gap * (len(row_images) - 1)
    atlas = Image.new("RGB", (atlas_w, atlas_h), "white")
    y = 0
    for row in row_images:
        atlas.paste(row, (0, y))
        y += row.height + gap
    atlas_path = args.output_dir / "edge_abs_0835_better_examples_atlas.png"
    atlas.save(atlas_path)

    csv_path = args.output_dir / "selected_examples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (args.output_dir / "selected_examples.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest_rows, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {atlas_path}")
    print(f"wrote {csv_path}")
    for row in manifest_rows:
        print(
            "{index} {sample_name}: edge_abs {edge_abs_rel_0306:.4f}->{edge_abs_rel_0835:.4f}, "
            "edge_d1 {edge_d1_0306:.4f}->{edge_d1_0835:.4f}, panel={panel}".format(**row)
        )


if __name__ == "__main__":
    main()
