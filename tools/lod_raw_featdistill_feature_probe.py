#!/usr/bin/env python3
"""Probe student RAW_dark tokens against a frozen RAW_normal teacher."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.config import NOT_APPLICABLE, ResolvedConfig  # noqa: E402
from finetune_stf.dataset.lod_true import LODTrueRawDarkNormalPairRGB16  # noqa: E402
from finetune_stf.train import (  # noqa: E402
    build_frozen_teacher,
    build_model,
    load_compatible_model_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-run-dir", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-ckpt", type=Path, required=True)
    parser.add_argument("--teacher-recipe", required=True, choices=("lora_tap_r8a16", "decoder_w0"))
    parser.add_argument("--teacher-input-mode", default="raw_rgb16_normal", choices=("raw_rgb16_normal",))
    parser.add_argument("--feat-distill-teacher", default="normal_expert", choices=("normal_expert", "same_param_clean"))
    parser.add_argument("--lod-root", type=Path, default=Path("/home/caq/6666_raw/0000_dataset/LOD"))
    parser.add_argument(
        "--lod-manifest",
        type=Path,
        default=Path(
            "/home/caq/6666_raw/0000_dataset/LOD/"
            "pseudo_depth_dav2l_rgb_normal_rel_1200x800/"
            "lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv"
        ),
    )
    parser.add_argument("--split", default="01Valid")
    parser.add_argument("--layers", nargs="+", type=int, default=[5, 8, 11])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_student_args(run_dir: Path, args: argparse.Namespace) -> argparse.Namespace:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing student config.json: {config_path}")
    payload = load_json(config_path)
    resolved_payload = payload["resolved_config"]
    resolved = ResolvedConfig.from_dict(resolved_payload)
    resolved = replace(
        resolved,
        teacher_ckpt=str(args.teacher_ckpt.expanduser().resolve()),
        teacher_recipe=str(args.teacher_recipe),
        teacher_input_mode=str(args.teacher_input_mode),
        feat_distill="dav2_middeep_cosine",
        feat_distill_layers=tuple(int(layer) for layer in args.layers),
        feat_distill_lambda=1.0,
        feat_distill_teacher=str(args.feat_distill_teacher),
    )
    payload["resolved_config"] = resolved
    payload["teacher_ckpt"] = str(args.teacher_ckpt.expanduser().resolve())
    payload["teacher_recipe"] = str(args.teacher_recipe)
    payload["teacher_input_mode"] = str(args.teacher_input_mode)
    payload["feat_distill"] = "dav2_middeep_cosine"
    payload["feat_distill_layers"] = [int(layer) for layer in args.layers]
    payload["feat_distill_lambda"] = 1.0
    payload["feat_distill_teacher"] = str(args.feat_distill_teacher)
    payload.setdefault("raw_ram_local_residual", "none")
    payload.setdefault("raw_ram_local_hidden_ch", 0)
    payload.setdefault("raw_ram_local_residual_scale", 0.0)
    payload.setdefault("raw_ram_local_gate_init", 0.0)
    payload.setdefault("raw_ram_local_gate_mode", "n_a")
    payload["student_init_from"] = None
    payload["student_init_strict"] = "compatible"
    payload.setdefault("_explicit_cli_args", ())
    return argparse.Namespace(**payload)


def token_cosine_distance(student: torch.Tensor, teacher: torch.Tensor) -> float:
    if student.shape != teacher.shape:
        raise ValueError(f"Feature shape mismatch: student={tuple(student.shape)} teacher={tuple(teacher.shape)}")
    fs = F.normalize(student.float(), dim=-1)
    ft = F.normalize(teacher.float(), dim=-1)
    return float((1.0 - (fs * ft).sum(dim=-1)).mean().detach().cpu().item())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], layers: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {"samples": len(rows)}
    for layer in layers:
        key = f"layer{layer}_token_cosine_distance"
        values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
        summary[f"{key}_mean"] = float(np.mean(values)) if values else float("nan")
        summary[f"{key}_std"] = float(np.std(values)) if values else float("nan")
    return summary


def main() -> None:
    args = parse_args()
    run_dir = args.student_run_dir.expanduser().resolve()
    student_ckpt = args.student_checkpoint.expanduser().resolve()
    teacher_ckpt = args.teacher_ckpt.expanduser().resolve()
    if not student_ckpt.is_file():
        raise FileNotFoundError(f"Missing student checkpoint: {student_ckpt}")
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(f"Missing teacher checkpoint: {teacher_ckpt}")
    device = resolve_device(args.device)
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    student_args = load_student_args(run_dir, args)
    if student_args.resolved_config.student_init_from != NOT_APPLICABLE:
        student_args.student_init_from = None
        student_args.resolved_config = replace(student_args.resolved_config, student_init_from=NOT_APPLICABLE)
    student = build_model(student_args)
    load_compatible_model_checkpoint(student, student_ckpt, strict_mode="compatible")
    teacher, teacher_args, teacher_status = build_frozen_teacher(student_args)
    del teacher_args, teacher_status
    student.to(device).eval()
    teacher.to(device).eval()

    dataset = LODTrueRawDarkNormalPairRGB16(
        lod_root=args.lod_root.expanduser().resolve(),
        manifest_path=args.lod_manifest.expanduser().resolve(),
        split=str(args.split),
        size=(int(student_args.input_height), int(student_args.input_width)),
        mode="val",
        label_space=str(student_args.lod_label_space),
        crop_mode=str(student_args.lod_val_crop_mode),
        raw_storage_format="raw_rgb16_png_3ch",
        lod_raw_norm_mode="uint16_div_65535",
    )
    max_samples = len(dataset) if args.max_samples is None else min(int(args.max_samples), len(dataset))
    rows: list[dict[str, Any]] = []
    for idx in range(max_samples):
        sample = dataset[idx]
        raw_dark = sample["raw"][None].to(device=device, non_blocking=True).float()
        raw_normal = sample["raw_normal"][None].to(device=device, non_blocking=True).float()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
            student_out = student(raw_dark, return_features=True, return_layers=args.layers)
            teacher_out = teacher(raw_normal, return_features=True, return_layers=args.layers)
        row: dict[str, Any] = {
            "sample_index": int(idx),
            "sample_id": str(sample["sample_id"]),
            "student_checkpoint": str(student_ckpt),
            "teacher_checkpoint": str(teacher_ckpt),
            "teacher_recipe": str(args.teacher_recipe),
            "student_input": "RAW_Dark",
            "teacher_input": "RAW_normal",
        }
        for layer in args.layers:
            row[f"layer{int(layer)}_token_cosine_distance"] = token_cosine_distance(
                student_out["features"][int(layer)],
                teacher_out["features"][int(layer)],
            )
        rows.append(row)
        if (idx + 1) % 25 == 0 or idx + 1 == max_samples:
            print(f"[FEAT_PROBE] processed={idx + 1}/{max_samples}", flush=True)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, [int(layer) for layer in args.layers])
    summary.update(
        {
            "student_run_dir": str(run_dir),
            "student_checkpoint": str(student_ckpt),
            "teacher_checkpoint": str(teacher_ckpt),
            "teacher_recipe": str(args.teacher_recipe),
            "layers": [int(layer) for layer in args.layers],
        }
    )
    write_csv(output_dir / "feature_probe.csv", rows)
    (output_dir / "feature_probe_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[FEAT_PROBE] wrote {output_dir / 'feature_probe.csv'}", flush=True)


if __name__ == "__main__":
    main()
