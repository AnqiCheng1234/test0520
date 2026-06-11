#!/usr/bin/env python3
"""Verify the ROD RawPy RGB renderer, config, and dataset wiring."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.dataset.rod_raw_student_rgb import DEFAULT_ROD_NIGHT_TEACHER_MANIFEST, DEFAULT_ROD_ROOT
from finetune_stf.dataset.rod_rawpy_render import (
    RAWPY_DNG_PROFILE,
    RAWPY_POSTPROCESS_PROFILE,
    RAWPY_RGB_PIPELINE,
    get_rawpy_runtime_versions,
    render_rawpy_default_rgb_from_raw24,
)
from finetune_stf.dataset.rod_rawpy_rgb import RODRawRawPyRGB


DEFAULT_ORACLE_ROOT = (
    PROJECT_ROOT
    / "finetune_stf"
    / "analysis"
    / "rod_rawpy_debug"
    / "0611_1253_rod_rawpy_preview_n10_dav2l_spectral_r"
)
DEFAULT_PRETRAINED = Path("/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "parity", "config", "dataset"], default="all")
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--rod-root", default=DEFAULT_ROD_ROOT)
    parser.add_argument("--rod-night-manifest", default=DEFAULT_ROD_NIGHT_TEACHER_MANIFEST)
    parser.add_argument("--pretrained-from", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/codex_smoke_rod_rawpy_verify"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--keep-success-artifacts", action="store_true")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def verify_parity(args: argparse.Namespace) -> None:
    oracle_root = Path(args.oracle_root).expanduser().resolve()
    run_config_path = oracle_root / "run_config.json"
    manifest_path = oracle_root / "manifest_rawpy_preview.jsonl"
    if not run_config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing RawPy preview oracle under {oracle_root}")

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    versions = get_rawpy_runtime_versions()
    expected = {
        "rawpy_version": run_config.get("rawpy_version"),
        "libraw_version": run_config.get("libraw_version"),
    }
    observed = {
        "rawpy_version": versions["rawpy_version"],
        "libraw_version": versions["libraw_version"],
    }
    if observed != expected:
        raise RuntimeError(f"RawPy runtime differs from oracle: observed={observed}, expected={expected}")

    records = _load_jsonl(manifest_path)[: max(1, int(args.limit))]
    fail_dir = Path(args.output_root) / "parity_failures"
    for record in records:
        raw_path = Path(str(record["raw_path"]))
        oracle_rgb = iio.imread(record["rawpy_928x1440_rgb_path"])
        rendered = render_rawpy_default_rgb_from_raw24(raw_path, temp_root=Path(args.output_root) / "tmp_dng")
        if not np.array_equal(rendered, oracle_rgb):
            fail_dir.mkdir(parents=True, exist_ok=True)
            stem = str(record["sample_id"]).replace("/", "_")
            diff = np.abs(rendered.astype(np.int16) - oracle_rgb.astype(np.int16)).astype(np.uint8)
            iio.imwrite(fail_dir / f"{stem}_rendered.png", rendered)
            iio.imwrite(fail_dir / f"{stem}_oracle.png", oracle_rgb)
            iio.imwrite(fail_dir / f"{stem}_absdiff.png", diff)
            raise AssertionError(
                f"Parity mismatch for {record['sample_id']}: max_abs_diff={int(diff.max())}; kept {fail_dir}"
            )
    print(f"[PARITY][OK] samples={len(records)} oracle={oracle_root}")


def _run_parse_args(argv: list[str]) -> object:
    from finetune_stf.train import parse_args as train_parse_args

    old_argv = sys.argv[:]
    try:
        sys.argv = ["finetune_stf/train.py", *argv]
        return train_parse_args()
    finally:
        sys.argv = old_argv


def _common_train_args(args: argparse.Namespace, save_name: str) -> list[str]:
    save_path = Path(args.output_root) / save_name
    return [
        "--encoder",
        "vits",
        "--pretrained-from",
        str(Path(args.pretrained_from).expanduser()),
        "--stage",
        "rod_only",
        "--input-domain",
        "rgb",
        "--front-end",
        "dav2_rgb",
        "--model-input-tensor",
        "image",
        "--dataset-family",
        "rod_raw_rawpy_rgb",
        "--dataset-input-mode",
        "raw24_rawpy_rgb",
        "--raw-storage-format",
        "n_a",
        "--bridge",
        "none",
        "--decoder-feature-adapter",
        "none",
        "--rod-root",
        str(args.rod_root),
        "--rod-night-manifest",
        str(args.rod_night_manifest),
        "--rod-raw-source",
        "raw24",
        "--rod-rgb-pipeline",
        RAWPY_RGB_PIPELINE,
        "--rod-rawpy-dng-profile",
        RAWPY_DNG_PROFILE,
        "--rod-rawpy-postprocess-profile",
        RAWPY_POSTPROCESS_PROFILE,
        "--rod-label-space",
        "inverse_relative",
        "--input-height",
        "512",
        "--input-width",
        "960",
        "--rod-train-crop-mode",
        "random",
        "--rod-val-crop-mode",
        "center",
        "--no-eval-stf",
        "--eval-rod",
        "--best-metric",
        "rod",
        "--bs",
        "8",
        "--accum-steps",
        "1",
        "--lr",
        "1e-5",
        "--loss-type",
        "ssi",
        "--loss-target-normalization",
        "--loss-norm-min-scale",
        "1e-3",
        "--epochs",
        "10",
        "--amp",
        "--amp-dtype",
        "bf16",
        "--seed",
        "42",
        "--num-workers",
        "4",
        "--log-interval",
        "500",
        "--no-enable-fixed-viz-dump",
        "--no-enable-train-source-viz-dump",
        "--heavy-save-root",
        str(Path(args.output_root) / "heavy"),
        "--save-path",
        str(save_path),
    ]


def _expect_parse_failure(argv: list[str], label: str) -> None:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            _run_parse_args(argv)
    except SystemExit as exc:
        if int(exc.code or 0) != 0:
            print(f"[CONFIG][NEG_OK] {label}")
            return
        raise AssertionError(f"{label} unexpectedly exited with status 0")
    raise AssertionError(f"{label} unexpectedly parsed successfully")


def verify_config(args: argparse.Namespace) -> None:
    decoder_args = _run_parse_args([*_common_train_args(args, "audit_decoder"), "--lora", "none", "--dav2-train-mode", "decoder", "--backbone-layer-decay", "1.0"])
    cfg = decoder_args.resolved_config
    assert cfg.input_type_alias == "rgb", cfg.to_dict()
    assert cfg.dataset_family == "rod_raw_rawpy_rgb", cfg.to_dict()
    assert cfg.dataset_input_mode == "raw24_rawpy_rgb", cfg.to_dict()
    assert cfg.front_end == "dav2_rgb", cfg.to_dict()
    assert cfg.model_input_tensor == "image", cfg.to_dict()

    lora_args = _run_parse_args(
        [
            *_common_train_args(args, "audit_lora"),
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
            "--dav2-train-mode",
            "decoder",
            "--backbone-layer-decay",
            "1.0",
        ]
    )
    lora_cfg = lora_args.resolved_config
    assert lora_cfg.input_type_alias == "rgb_lora", lora_cfg.to_dict()
    assert lora_cfg.lora == "dav2_lora", lora_cfg.to_dict()
    assert tuple(lora_cfg.lora_tap_layers) == (2, 5, 8, 11), lora_cfg.to_dict()

    bad_pipeline = _common_train_args(args, "bad_pipeline")
    idx = bad_pipeline.index("--rod-rgb-pipeline") + 1
    bad_pipeline[idx] = "student_dark_degreen_v1"
    _expect_parse_failure([*bad_pipeline, "--lora", "none"], "rawpy_family_rejects_student_pipeline")

    bad_student_param = [*_common_train_args(args, "bad_student_param"), "--rod-student-gamma", "0.9", "--lora", "none"]
    _expect_parse_failure(bad_student_param, "rawpy_family_rejects_explicit_student_param")

    student_rawpy = _common_train_args(args, "bad_student_rawpy")
    student_rawpy[student_rawpy.index("--dataset-family") + 1] = "rod_raw_student_rgb"
    student_rawpy[student_rawpy.index("--dataset-input-mode") + 1] = "raw24_student_rgb"
    _expect_parse_failure([*student_rawpy, "--lora", "none"], "student_family_rejects_rawpy_pipeline")

    print("[CONFIG][OK] decoder_and_lora_audits")


def _assert_batch(batch: dict[str, object], split: str) -> None:
    image = batch["image"]
    depth = batch["depth"]
    valid_mask = batch["valid_mask"]
    if not isinstance(image, torch.Tensor) or tuple(image.shape[-3:]) != (3, 512, 960):
        raise AssertionError(f"{split} image shape mismatch: {getattr(image, 'shape', None)}")
    if not isinstance(depth, torch.Tensor) or tuple(depth.shape[-2:]) != (512, 960):
        raise AssertionError(f"{split} depth shape mismatch: {getattr(depth, 'shape', None)}")
    if not isinstance(valid_mask, torch.Tensor) or tuple(valid_mask.shape[-2:]) != (512, 960):
        raise AssertionError(f"{split} valid_mask shape mismatch: {getattr(valid_mask, 'shape', None)}")


def verify_dataset(args: argparse.Namespace) -> None:
    temp_root = Path(args.output_root) / "dataset_tmp_dng"
    for split, mode, crop_mode, workers in (
        ("01Valid", "val", "center", 0),
        ("00Train", "train", "random", 4),
    ):
        dataset = RODRawRawPyRGB(
            rod_root=args.rod_root,
            manifest_path=args.rod_night_manifest,
            split=split,
            mode=mode,
            crop_mode=crop_mode,
            raw_source="raw24",
            rgb_pipeline=RAWPY_RGB_PIPELINE,
            rawpy_dng_profile=RAWPY_DNG_PROFILE,
            rawpy_postprocess_profile=RAWPY_POSTPROCESS_PROFILE,
            rawpy_temp_root=temp_root,
        )
        loader_kwargs = {"multiprocessing_context": "spawn"} if workers > 0 else {}
        loader = DataLoader(dataset, batch_size=1, num_workers=workers, **loader_kwargs)
        batch = next(iter(loader))
        _assert_batch(batch, split)

    leftovers = list(temp_root.glob("*.dng")) if temp_root.exists() else []
    if leftovers:
        raise AssertionError(f"Temporary DNG files were not cleaned: {leftovers[:5]}")
    print("[DATASET][OK] splits=01Valid/00Train workers=0/4")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode in {"all", "parity"}:
            verify_parity(args)
        if args.mode in {"all", "config"}:
            verify_config(args)
        if args.mode in {"all", "dataset"}:
            verify_dataset(args)
    except Exception:
        print(f"[VERIFY][FAIL] kept artifacts at {output_root}", file=sys.stderr)
        raise
    else:
        if not args.keep_success_artifacts and any(marker in str(output_root) for marker in ("smoke", "debug", "tmp", "codex_smoke")):
            shutil.rmtree(output_root, ignore_errors=True)
        print("[VERIFY][OK]")


if __name__ == "__main__":
    main()
