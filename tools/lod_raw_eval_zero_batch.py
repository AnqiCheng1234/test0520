#!/usr/bin/env python3
"""Run zero-training LOD RAW ablation evals and summarize logs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import socket
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DARK_RUN = "0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly"
NORMAL_RUN = "0609_0044_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly"

DARK_TRANSFORMS = [
    "raw_dark_identity",
    "raw_dark_denoise_median3",
    "raw_dark_bilateral_weak_trainfit_sigma",
    "raw_dark_bilateral_medium_trainfit_sigma",
    "raw_dark_bilateral_strong_trainfit_sigma",
    "raw_dark_tone_percentile_trainfit",
    "raw_dark_tone_percentile_luma_trainfit",
    "raw_dark_tone_gamma_trainfit",
    "raw_dark_denoise_then_tone_trainfit",
    "raw_dark_tone_then_denoise_trainfit",
    "raw_dark_tone_percentile_oracle",
    "raw_dark_tone_denoise_oracle",
]

NORMAL_TRANSFORMS = [
    "raw_normal_identity",
    "raw_normal_to_dark_exposure_trainfit",
    "raw_normal_to_dark_noise_only_trainfit",
    "raw_normal_to_dark_exposure_noise_trainfit",
]

ALL_TRANSFORMS = DARK_TRANSFORMS + NORMAL_TRANSFORMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=Path("/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/ablation_inputs"),
    )
    parser.add_argument(
        "--light-root",
        type=Path,
        default=PROJECT_ROOT / "finetune_stf" / "exp" / "lod_raw_diag" / "eval_zero",
    )
    parser.add_argument(
        "--heavy-root",
        type=Path,
        default=Path("/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/eval_zero"),
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/eval_zero/zero_eval_summary.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/eval_zero/zero_eval_summary.json"),
    )
    parser.add_argument(
        "--transforms",
        default="all",
        help="Comma-separated transform names, or 'all'.",
    )
    parser.add_argument("--skip-transforms", default="", help="Comma-separated transform names to skip.")
    parser.add_argument(
        "--pretrained-from",
        default="/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth",
    )
    parser.add_argument("--lod-root", default="/home/caq/6666_raw/0000_dataset/LOD")
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=Path("/mnt/drive/3333_raw/0000_exp_ckpt"),
    )
    parser.add_argument("--run-prefix", default=None)
    return parser.parse_args()


def split_names(value: str) -> list[str]:
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def choose_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def transform_config(name: str, ckpt_root: Path) -> dict[str, str]:
    if name in DARK_TRANSFORMS:
        return {
            "model_group": "raw_dark_L0_best",
            "checkpoint": str(ckpt_root / DARK_RUN / "best_model.pth"),
            "dataset_family": "lod_true_raw_dark_rgb16",
            "dataset_input_mode": "raw_rgb16_dark",
        }
    if name in NORMAL_TRANSFORMS:
        return {
            "model_group": "raw_normal_NL_best",
            "checkpoint": str(ckpt_root / NORMAL_RUN / "best_model.pth"),
            "dataset_family": "lod_true_raw_normal_rgb16",
            "dataset_input_mode": "raw_rgb16_normal",
        }
    raise ValueError(f"Unknown transform: {name}")


def command_for(args: argparse.Namespace, transform: str, port: int) -> list[str]:
    cfg = transform_config(transform, args.ckpt_root)
    manifest = args.ablation_dir / "manifests" / f"{transform}.csv"
    save_path = args.light_root / transform
    return [
        "torchrun",
        "--nproc_per_node=1",
        "--master_port",
        str(port),
        "finetune_stf/train.py",
        "--stage",
        "eval_only",
        "--encoder",
        "vits",
        "--dataset-family",
        cfg["dataset_family"],
        "--dataset-input-mode",
        cfg["dataset_input_mode"],
        "--input-domain",
        "raw3",
        "--front-end",
        "raw_rgb16_ram3",
        "--model-input-tensor",
        "raw",
        "--bridge",
        "none",
        "--decoder-feature-adapter",
        "none",
        "--pretrained-from",
        args.pretrained_from,
        "--resume-from",
        cfg["checkpoint"],
        "--lod-root",
        args.lod_root,
        "--lod-manifest",
        str(manifest),
        "--lod-label-space",
        "inverse_relative",
        "--lod-train-crop-mode",
        "random",
        "--lod-val-crop-mode",
        "center",
        "--raw-storage-format",
        "raw_rgb16_png_3ch",
        "--lod-raw-norm-mode",
        "uint16_div_65535",
        "--raw-ram-rgb-tail",
        "identity",
        "--lora",
        "dav2_lora",
        "--lora-block-mode",
        "tap",
        "--lora-tap-layers",
        "2",
        "5",
        "8",
        "11",
        "--lora-rank",
        "8",
        "--lora-alpha",
        "16",
        "--lora-lr",
        "5e-5",
        "--raw-front-end-lr",
        "5e-5",
        "--dav2-train-mode",
        "decoder",
        "--backbone-layer-decay",
        "1.0",
        "--lr",
        "1e-5",
        "--lr-schedule",
        "poly",
        "--warmup-steps",
        "0",
        "--loss-type",
        "ssi",
        "--loss-target-normalization",
        "--loss-norm-min-scale",
        "1e-3",
        "--bs",
        "8",
        "--accum-steps",
        "1",
        "--input-height",
        "512",
        "--input-width",
        "960",
        "--aug-preset",
        "off",
        "--aug-hflip-prob",
        "0.5",
        "--eval-lod",
        "--no-eval-stf",
        "--eval-lod-train-proxy",
        "--lod-train-proxy-count",
        "112",
        "--amp",
        "--amp-dtype",
        "bf16",
        "--num-workers",
        "4",
        "--log-interval",
        "500",
        "--no-enable-fixed-viz-dump",
        "--no-enable-train-source-viz-dump",
        "--save-path",
        str(save_path),
        "--heavy-save-root",
        str(args.heavy_root),
    ]


def parse_metrics(text: str) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {
        "val_samples": None,
        "val_abs_rel": None,
        "val_rmse": None,
        "val_silog": None,
        "val_d1": None,
        "train_proxy_d1": None,
        "train_proxy_val_gap": None,
    }
    done = re.search(
        r"\[EVAL\]\[eval_only_lod_val\] done epoch=init samples=(\d+) "
        r"abs_rel=([0-9.eE+-]+) rmse=([0-9.eE+-]+) silog=([0-9.eE+-]+) d1=([0-9.eE+-]+)",
        text,
    )
    if done:
        out["val_samples"] = int(done.group(1))
        out["val_abs_rel"] = float(done.group(2))
        out["val_rmse"] = float(done.group(3))
        out["val_silog"] = float(done.group(4))
        out["val_d1"] = float(done.group(5))
    summary = re.search(r"\[EVAL\]\[eval_only_lod_val\] summary=(\{.*?\})", text)
    if summary:
        payload = eval(summary.group(1), {"__builtins__": {}}, {"nan": math.nan, "inf": math.inf})  # noqa: S307
        out["val_abs_rel"] = float(payload["abs_rel"])
        out["val_rmse"] = float(payload["rmse"])
        out["val_silog"] = float(payload["silog"])
        out["val_d1"] = float(payload["d1"])
    gap = re.search(
        r"\[EVAL\]\[eval_only_lod_gap\] train_proxy_d1=([0-9.eE+-]+) "
        r"val_d1=([0-9.eE+-]+) gap=([0-9.eE+-]+)",
        text,
    )
    if gap:
        out["train_proxy_d1"] = float(gap.group(1))
        out["train_proxy_val_gap"] = float(gap.group(3))
    return out


def run_one(args: argparse.Namespace, transform: str, run_prefix: str) -> dict[str, object]:
    cfg = transform_config(transform, args.ckpt_root)
    port = choose_port()
    log_path = args.logs_dir / f"{run_prefix}_eval_zero_{transform}.log"
    cmd = command_for(args, transform, port)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] {transform} port={port} log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write("COMMAND: " + " ".join(cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert proc.stdout is not None
        lines: list[str] = []
        for line in proc.stdout:
            lines.append(line)
            log_f.write(line)
            if "[EVAL]" in line or "[RESOLVED]" in line:
                print(line.rstrip(), flush=True)
        returncode = proc.wait()
    text = "".join(lines)
    metrics = parse_metrics(text)
    row: dict[str, object] = {
        "transform": transform,
        "model_group": cfg["model_group"],
        "dataset_family": cfg["dataset_family"],
        "dataset_input_mode": cfg["dataset_input_mode"],
        "checkpoint": cfg["checkpoint"],
        "manifest": str(args.ablation_dir / "manifests" / f"{transform}.csv"),
        "save_path": str(args.light_root / transform),
        "heavy_path": str(args.heavy_root / transform),
        "log_path": str(log_path),
        "returncode": returncode,
        **metrics,
    }
    print(
        f"[DONE] {transform} returncode={returncode} val_d1={row['val_d1']} "
        f"val_abs_rel={row['val_abs_rel']}",
        flush=True,
    )
    return row


def write_summary(rows: list[dict[str, object]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "transform",
        "model_group",
        "dataset_family",
        "dataset_input_mode",
        "returncode",
        "val_samples",
        "val_d1",
        "val_abs_rel",
        "val_rmse",
        "val_silog",
        "train_proxy_d1",
        "train_proxy_val_gap",
        "manifest",
        "checkpoint",
        "save_path",
        "heavy_path",
        "log_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)


def main() -> int:
    args = parse_args()
    if args.transforms == "all":
        transforms = list(ALL_TRANSFORMS)
    else:
        transforms = split_names(args.transforms)
    skip = set(split_names(args.skip_transforms))
    transforms = [name for name in transforms if name not in skip]
    unknown = sorted(set(transforms) - set(ALL_TRANSFORMS))
    if unknown:
        raise SystemExit(f"Unknown transforms: {unknown}")
    run_prefix = args.run_prefix or dt.datetime.now().strftime("%m%d_%H%M")
    rows: list[dict[str, object]] = []
    for transform in transforms:
        row = run_one(args, transform, run_prefix)
        rows.append(row)
        write_summary(rows, args.summary_csv, args.summary_json)
        if int(row["returncode"]) != 0:
            print(f"[ERROR] stopping after failed transform={transform}", file=sys.stderr, flush=True)
            return int(row["returncode"])
    write_summary(rows, args.summary_csv, args.summary_json)
    print(f"[SUMMARY] csv={args.summary_csv}", flush=True)
    print(f"[SUMMARY] json={args.summary_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
