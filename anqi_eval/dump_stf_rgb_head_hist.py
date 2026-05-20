#!/usr/bin/env python3
"""
Dump RGB-head pre/post-sigmoid histograms for STF raw models.

Focuses on the current raw RAM path:
    x_raw -> ram_core -> x4 -> rgb_head.conv(x4) -> sigmoid -> adapted_rgb

This diagnostic is intentionally standalone and avoids modifying the main eval
or qualitative scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.dump_stf_rgb_triplets import (
    DEFAULT_RAW_NPZ_ROOT,
    DEFAULT_STF_ROOT,
    DEVICE,
    load_manifest_rows,
    load_model,
    make_indices,
    resolve_checkpoint,
)
from finetune_stf.dataset.raw_utils import load_rectified_bayer_npz, normalize_raw_4ch
from finetune_stf.models.lora_bridge import RAW_RAM_BRIDGE_INPUT_TYPES

SUPPORTED_INPUT_TYPES = ("raw_ram", *RAW_RAM_BRIDGE_INPUT_TYPES)
VALID_SPLITS = ("train", "val", "test")
GROUP_ORDER = ("all", "day", "twilight", "night")


def parse_args():
    parser = argparse.ArgumentParser(description="Dump STF RGB-head pre/post-sigmoid histograms.")
    parser.add_argument("exp_dir", type=Path, help="Path to an experiment directory containing config.json.")
    parser.add_argument(
        "--checkpoint",
        default="best",
        help='Checkpoint to load: "best", "latest", or a custom .pth path. Default: best',
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=VALID_SPLITS,
        help="Dataset split to analyze. Default: val",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="How many evenly spaced samples to use. Default: 50",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit sample indices. Overrides --num-samples.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of histogram bins. Default: 80",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Optional custom output subdirectory name inside exp_dir.",
    )
    return parser.parse_args()


def resolve_output_dir(exp_dir: Path, split: str, checkpoint: str, output_subdir: str | None) -> Path:
    if output_subdir:
        return exp_dir / output_subdir
    ckpt_name = Path(checkpoint).stem if checkpoint not in {"best", "latest"} else checkpoint
    return exp_dir / f"diagnostic_rgb_head_hist_{split}_{ckpt_name}"


def make_raw_tensor(raw_npz_path: Path, *, norm_mode: str) -> torch.Tensor:
    bayer_rect = load_rectified_bayer_npz(raw_npz_path)
    bayer_4ch_norm = normalize_raw_4ch(bayer_rect, norm_mode=norm_mode)
    chw = np.ascontiguousarray(bayer_4ch_norm.transpose(2, 0, 1)).astype(np.float32, copy=False)
    return torch.frombuffer(chw.tobytes(), dtype=torch.float32).view(chw.shape).unsqueeze(0).to(DEVICE)


def extract_pre_post_sigmoid(model, input_type: str, x_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if input_type == "raw_ram":
        with torch.no_grad():
            x4 = model.ram_core(x_raw)
            pre = model.rgb_head.conv(x4)
            post = torch.sigmoid(pre)
        return pre, post

    if input_type in RAW_RAM_BRIDGE_INPUT_TYPES:
        with torch.no_grad():
            x4, _ = model.ram_core.forward_with_features(x_raw)
            pre = model.rgb_head.conv(x4)
            post = torch.sigmoid(pre)
        return pre, post

    raise ValueError(f"Unsupported input_type for sigmoid histogram diagnostic: {input_type}")


def tensor_flat_numpy(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().contiguous().cpu().view(-1)
    data = bytes((x.view(torch.uint8)).tolist())
    return np.frombuffer(data, dtype=np.float32)


def summarize_distribution(values: np.ndarray, *, kind: str) -> dict[str, float]:
    summary = {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p01": float(np.quantile(values, 0.01)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
    }
    if kind == "pre":
        summary["frac_abs_gt_2"] = float((np.abs(values) > 2.0).mean())
        summary["frac_abs_gt_4"] = float((np.abs(values) > 4.0).mean())
        summary["frac_abs_gt_6"] = float((np.abs(values) > 6.0).mean())
    elif kind == "post":
        summary["frac_lt_001"] = float((values < 0.01).mean())
        summary["frac_lt_005"] = float((values < 0.05).mean())
        summary["frac_gt_095"] = float((values > 0.95).mean())
        summary["frac_gt_099"] = float((values > 0.99).mean())
        summary["frac_mid_045_055"] = float(((values >= 0.45) & (values <= 0.55)).mean())
    else:
        raise ValueError(f"Unsupported kind={kind}")
    return summary


def render_histogram(
    counts: np.ndarray,
    edges: np.ndarray,
    *,
    title: str,
    stats_lines: list[str],
    bar_color: tuple[int, int, int],
    width: int = 1200,
    height: int = 520,
) -> Image.Image:
    margin_left = 72
    margin_right = 24
    margin_top = 84
    margin_bottom = 56
    hist_h = height - margin_top - margin_bottom
    hist_w = width - margin_left - margin_right
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((24, 20), title, fill="black")
    for i, line in enumerate(stats_lines):
        draw.text((24, 44 + i * 16), line, fill=(60, 60, 60))

    max_count = float(counts.max()) if counts.size else 1.0
    if max_count <= 0:
        max_count = 1.0

    x0 = margin_left
    y0 = margin_top
    x1 = width - margin_right
    y1 = height - margin_bottom

    draw.rectangle((x0, y0, x1, y1), outline=(160, 160, 160), width=1)
    for tick in range(5):
        y = y1 - hist_h * tick / 4.0
        draw.line((x0, y, x1, y), fill=(235, 235, 235), width=1)

    nbins = max(len(counts), 1)
    bin_w = hist_w / nbins
    for i, count in enumerate(counts):
        bar_h = 0 if count <= 0 else hist_h * float(count) / max_count
        left = x0 + i * bin_w
        right = x0 + (i + 1) * bin_w
        top = y1 - bar_h
        draw.rectangle((left, top, right, y1), fill=bar_color)

    xmin = float(edges[0])
    xmax = float(edges[-1])
    draw.text((x0, y1 + 8), f"{xmin:.2f}", fill="black")
    xmax_text = f"{xmax:.2f}"
    draw.text((x1 - 8 * len(xmax_text), y1 + 8), xmax_text, fill="black")
    return canvas


def save_group_histograms(
    output_dir: Path,
    *,
    group_name: str,
    pre_values: np.ndarray,
    post_values: np.ndarray,
    bins: int,
):
    pre_counts, pre_edges = np.histogram(pre_values, bins=bins, range=(-8.0, 8.0))
    post_counts, post_edges = np.histogram(post_values, bins=bins, range=(0.0, 1.0))

    pre_summary = summarize_distribution(pre_values, kind="pre")
    post_summary = summarize_distribution(post_values, kind="post")

    pre_lines = [
        f"n={pre_summary['count']:,} mean={pre_summary['mean']:.3f} std={pre_summary['std']:.3f}",
        f"p01/p50/p99={pre_summary['p01']:.3f}/{pre_summary['p50']:.3f}/{pre_summary['p99']:.3f}",
        f"|x|>2: {pre_summary['frac_abs_gt_2']:.3%}  |x|>4: {pre_summary['frac_abs_gt_4']:.3%}  |x|>6: {pre_summary['frac_abs_gt_6']:.3%}",
    ]
    post_lines = [
        f"n={post_summary['count']:,} mean={post_summary['mean']:.3f} std={post_summary['std']:.3f}",
        f"p01/p50/p99={post_summary['p01']:.3f}/{post_summary['p50']:.3f}/{post_summary['p99']:.3f}",
        f"<0.01: {post_summary['frac_lt_001']:.3%}  <0.05: {post_summary['frac_lt_005']:.3%}  >0.95: {post_summary['frac_gt_095']:.3%}  >0.99: {post_summary['frac_gt_099']:.3%}",
    ]

    render_histogram(
        pre_counts,
        pre_edges,
        title=f"{group_name}: pre-sigmoid rgb_head.conv(x4)",
        stats_lines=pre_lines,
        bar_color=(70, 130, 220),
    ).save(output_dir / f"{group_name}_pre_sigmoid_hist.png")

    render_histogram(
        post_counts,
        post_edges,
        title=f"{group_name}: post-sigmoid rgb_head(x4)",
        stats_lines=post_lines,
        bar_color=(220, 120, 70),
    ).save(output_dir / f"{group_name}_post_sigmoid_hist.png")

    return {"pre": pre_summary, "post": post_summary}


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    config_path = exp_dir / "config.json"
    checkpoint_path = resolve_checkpoint(exp_dir, args.checkpoint).resolve()
    output_dir = resolve_output_dir(exp_dir, args.split, args.checkpoint, args.output_subdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing experiment config: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    with config_path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    input_type = cfg.get("input_type", "rgb")
    if input_type not in SUPPORTED_INPUT_TYPES:
        raise ValueError(
            "This histogram diagnostic currently only supports raw_ram/raw_ram_bridge "
            f"models with rgb_head.conv + sigmoid, got input_type={input_type}"
        )

    stf_root = cfg.get("stf_root", DEFAULT_STF_ROOT)
    raw_npz_root = cfg.get("raw_npz_root", DEFAULT_RAW_NPZ_ROOT)
    norm_mode = cfg.get("norm_mode", "companded")

    print(f"Loading model from {checkpoint_path} on {DEVICE} ...", flush=True)
    model = load_model(cfg, str(checkpoint_path))
    rows = load_manifest_rows(stf_root, args.split, raw_npz_root)
    indices = make_indices(len(rows), args.num_samples, args.indices)

    grouped = {group: {"pre": [], "post": [], "indices": []} for group in GROUP_ORDER}
    sample_lines = ["index\tsample_name\tdaytime\traw_npz_path"]

    for idx in indices:
        row = rows[idx]
        raw_npz_path = row["raw_npz_path"]
        if not raw_npz_path.is_file():
            raise FileNotFoundError(f"Missing raw npz: {raw_npz_path}")

        x_raw = make_raw_tensor(raw_npz_path, norm_mode=norm_mode)
        pre, post = extract_pre_post_sigmoid(model, input_type, x_raw)
        pre_np = tensor_flat_numpy(pre)
        post_np = tensor_flat_numpy(post)
        daytime = str(row.get("daytime", "")).strip().lower()
        if daytime not in ("day", "twilight", "night"):
            daytime = "all"

        for group in ("all", daytime):
            grouped[group]["pre"].append(pre_np)
            grouped[group]["post"].append(post_np)
            grouped[group]["indices"].append(idx)

        sample_lines.append(f"{idx}\t{row['sample_name']}\t{daytime}\t{raw_npz_path}")
        print(f"  [{idx}] {row['sample_name']} daytime={daytime}", flush=True)

    (output_dir / "samples.txt").write_text("\n".join(sample_lines) + "\n", encoding="utf-8")

    summary = {
        "checkpoint": str(checkpoint_path),
        "input_type": input_type,
        "split": args.split,
        "num_samples": len(indices),
        "device": DEVICE,
        "groups": {},
    }
    lines = [
        f"checkpoint={checkpoint_path}",
        f"input_type={input_type}",
        f"split={args.split}",
        f"num_samples={len(indices)}",
        "",
    ]

    for group_name in GROUP_ORDER:
        pre_chunks = grouped[group_name]["pre"]
        post_chunks = grouped[group_name]["post"]
        if not pre_chunks or not post_chunks:
            continue
        pre_values = np.concatenate(pre_chunks)
        post_values = np.concatenate(post_chunks)
        group_summary = save_group_histograms(
            output_dir,
            group_name=group_name,
            pre_values=pre_values,
            post_values=post_values,
            bins=args.bins,
        )
        group_summary["num_samples"] = len(grouped[group_name]["indices"])
        group_summary["indices"] = list(grouped[group_name]["indices"])
        summary["groups"][group_name] = group_summary

        pre_s = group_summary["pre"]
        post_s = group_summary["post"]
        lines.append(f"[{group_name}] n={group_summary['num_samples']}")
        lines.append(
            "pre  "
            f"mean={pre_s['mean']:.4f} std={pre_s['std']:.4f} "
            f"|x|>2={pre_s['frac_abs_gt_2']:.4%} |x|>4={pre_s['frac_abs_gt_4']:.4%} |x|>6={pre_s['frac_abs_gt_6']:.4%}"
        )
        lines.append(
            "post "
            f"mean={post_s['mean']:.4f} std={post_s['std']:.4f} "
            f"<0.01={post_s['frac_lt_001']:.4%} <0.05={post_s['frac_lt_005']:.4%} "
            f">0.95={post_s['frac_gt_095']:.4%} >0.99={post_s['frac_gt_099']:.4%}"
        )
        lines.append("")

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved histogram diagnostics to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
