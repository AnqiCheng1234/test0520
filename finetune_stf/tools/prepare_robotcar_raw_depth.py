from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_SOURCE_ROOT = Path("/mnt/drive/3333_raw/robotcar/batch_rgb_raw_gt_depth_stride10_lms_front")
DEFAULT_OUTPUT_ROOT = Path("/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640")
DEFAULT_MODELS_DIR = Path("/mnt/drive/3333_raw/robotcar/tools/robotcar-dataset-sdk/models")
FULL_HW = (960, 1280)
RAW_NATIVE_HW = (480, 640)
DEFAULT_EVAL_HW = (480, 640)
RECTIFIED_BAYER_KEY = "bayer_rect"
DEPTH_PROXY_KEY = "depth"
VALID_MASK_KEY = "valid_mask"
PROXY_COVERAGE_KEY = "coverage"
RAW_CFA_PATTERN = "GBRG"
PACK_ORDER = "[R,Gr,Gb,B]"
META_SUFFIX = ".meta.json"
PLANE_DEFS = (
    ("R", 1, 0),
    ("Gr", 1, 1),
    ("Gb", 0, 0),
    ("B", 0, 1),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare RobotCar raw/rgb/depth validation assets.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--target-height", type=int, default=DEFAULT_EVAL_HW[0])
    parser.add_argument("--target-width", type=int, default=DEFAULT_EVAL_HW[1])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.target_height <= 0 or args.target_width <= 0:
        parser.error("--target-height/--target-width must be positive")
    if args.sample_limit is not None and args.sample_limit < 1:
        parser.error("--sample-limit must be >= 1")
    if args.threshold is not None and args.threshold <= 0:
        parser.error("--threshold must be > 0")
    return args


def hw_to_str(hw: tuple[int, int]) -> str:
    return f"{int(hw[0])}x{int(hw[1])}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_npz(path: Path, **arrays) -> None:
    ensure_parent(path)
    np.savez_compressed(path, **arrays)


def save_rgb_copy(rgb_src_path: Path, rgb_dst_path: Path, target_hw: tuple[int, int]) -> None:
    image = Image.open(rgb_src_path).convert("RGB")
    resized = image.resize((target_hw[1], target_hw[0]), Image.Resampling.BICUBIC)
    ensure_parent(rgb_dst_path)
    resized.save(rgb_dst_path)


def create_or_update_symlink(link_path: Path, target_path: Path) -> None:
    ensure_parent(link_path)
    if link_path.exists() or link_path.is_symlink():
        try:
            if link_path.resolve() == target_path.resolve():
                return
        except FileNotFoundError:
            pass
        if link_path.is_dir():
            raise RuntimeError(f"Refusing to replace directory with symlink: {link_path}")
        link_path.unlink()
    os.symlink(target_path, link_path)


def discover_sample_dirs(source_root: Path) -> list[Path]:
    sample_dirs = []
    for path in sorted(source_root.iterdir()):
        if not path.is_dir():
            continue
        required = ("raw.png", "rgb.png", "gt_depth.npy", "meta.json")
        if all((path / name).is_file() for name in required):
            sample_dirs.append(path)
    return sample_dirs


def load_lut_xy(models_dir: Path) -> np.ndarray:
    lut_path = models_dir / "stereo_narrow_left_distortion_lut.bin"
    lut = np.fromfile(lut_path, dtype=np.double)
    expected_size = FULL_HW[0] * FULL_HW[1] * 2
    if lut.size != expected_size:
        raise RuntimeError(f"Unexpected RobotCar LUT size for {lut_path}: {lut.size} != {expected_size}")
    lut_xy = lut.reshape(2, -1).T.reshape(FULL_HW[0], FULL_HW[1], 2).astype(np.float32)
    return lut_xy


def build_half_res_maps(lut_xy: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    maps = []
    for _, row_offset, col_offset in PLANE_DEFS:
        sub = lut_xy[row_offset::2, col_offset::2]
        map_x = np.ascontiguousarray((sub[..., 0] - float(col_offset)) * 0.5, dtype=np.float32)
        map_y = np.ascontiguousarray((sub[..., 1] - float(row_offset)) * 0.5, dtype=np.float32)
        maps.append((map_x, map_y))
    return maps


def load_raw_bayer(raw_path: Path) -> np.ndarray:
    raw = np.array(Image.open(raw_path), dtype=np.uint8)
    if raw.shape != FULL_HW:
        raise RuntimeError(f"Unexpected RobotCar raw shape for {raw_path}: {raw.shape}")
    return raw


def pack_gbrg(raw: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            raw[1::2, 0::2],
            raw[1::2, 1::2],
            raw[0::2, 0::2],
            raw[0::2, 1::2],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def rectify_packed_raw(raw: np.ndarray, remap_pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    packed = pack_gbrg(raw)
    rectified = np.empty_like(packed, dtype=np.float32)
    for ch, (map_x, map_y) in enumerate(remap_pairs):
        rectified[..., ch] = cv2.remap(
            packed[..., ch],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return np.clip(rectified / 255.0, 0.0, 1.0).astype(np.float32, copy=False)


def load_depth_full(depth_path: Path) -> np.ndarray:
    depth = np.load(depth_path).astype(np.float32, copy=False)
    if depth.shape != FULL_HW:
        raise RuntimeError(f"Unexpected RobotCar depth shape for {depth_path}: {depth.shape}")
    return depth


def build_depth_proxy(
    depth_full: np.ndarray,
    target_hw: tuple[int, int],
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_full = np.isfinite(depth_full) & (depth_full > 0)
    depth_num = cv2.resize(
        np.where(valid_full, depth_full, 0.0).astype(np.float32),
        (target_hw[1], target_hw[0]),
        interpolation=cv2.INTER_AREA,
    )
    depth_den = cv2.resize(
        valid_full.astype(np.float32),
        (target_hw[1], target_hw[0]),
        interpolation=cv2.INTER_AREA,
    )
    valid_proxy = depth_den > float(threshold)
    depth_proxy = np.zeros_like(depth_num, dtype=np.float32)
    np.divide(depth_num, depth_den, out=depth_proxy, where=valid_proxy)
    return depth_proxy, valid_proxy, depth_den


def sample_meta_path(raw_native_path: Path) -> Path:
    return raw_native_path.with_suffix(META_SUFFIX)


def build_sample_meta_payload(
    *,
    scene: str,
    sample_name: str,
    target_hw: tuple[int, int],
    source_meta: dict[str, object],
) -> dict[str, object]:
    return {
        "scene": scene,
        "sample_name": sample_name,
        "cfa_pattern": RAW_CFA_PATTERN,
        "pack_order": PACK_ORDER,
        "raw_domain": "sensor_linear_unit_interval",
        "raw_align_mode": "robotcar_lut_halfres_rectify",
        "raw_native_hw": hw_to_str(RAW_NATIVE_HW),
        "raw_eval_hw": hw_to_str(target_hw),
        "full_hw": hw_to_str(FULL_HW),
        "white_level": 255.0,
        "black_level_0": 0.0,
        "black_level_1": 0.0,
        "black_level_2": 0.0,
        "black_level_3": 0.0,
        "timestamp": source_meta.get("timestamp"),
        "image_idx": source_meta.get("image_idx"),
        "poses_type": source_meta.get("poses_type"),
        "laser_sensors": source_meta.get("laser_sensors"),
        "valid_depth_pixels": source_meta.get("valid_depth_pixels"),
    }


def write_sample_meta(meta_path: Path, payload: dict[str, object]) -> None:
    ensure_parent(meta_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_manifest_row(
    *,
    scene: str,
    sample_name: str,
    rgb_src_path: Path,
    rgb_eval_path: Path,
    raw_src_path: Path,
    raw_native_path: Path,
    raw_eval_path: Path,
    depth_src_path: Path,
    depth_proxy_path: Path,
    meta_src_path: Path,
    target_hw: tuple[int, int],
    source_meta: dict[str, object],
) -> dict[str, object]:
    return {
        "scene": scene,
        "sample_name": sample_name,
        "rgb_src_path": str(rgb_src_path),
        "rgb_eval_path": str(rgb_eval_path),
        "raw_src_path": str(raw_src_path),
        "raw_native_path": str(raw_native_path),
        "raw_eval_path": str(raw_eval_path),
        "depth_src_path": str(depth_src_path),
        "depth_proxy_path": str(depth_proxy_path),
        "meta_src_path": str(meta_src_path),
        "rgb_hw": hw_to_str(FULL_HW),
        "rgb_eval_hw": hw_to_str(target_hw),
        "raw_native_hw": hw_to_str(RAW_NATIVE_HW),
        "raw_eval_hw": hw_to_str(target_hw),
        "depth_full_hw": hw_to_str(FULL_HW),
        "depth_fast_hw": hw_to_str(target_hw),
        "cfa_pattern": RAW_CFA_PATTERN,
        "pack_order": PACK_ORDER,
        "raw_domain": "sensor_linear_unit_interval",
        "raw_align_mode": "robotcar_lut_halfres_rectify",
        "white_level": 255.0,
        "black_level_0": 0.0,
        "black_level_1": 0.0,
        "black_level_2": 0.0,
        "black_level_3": 0.0,
        "timestamp": source_meta.get("timestamp"),
        "image_idx": source_meta.get("image_idx"),
        "poses_type": source_meta.get("poses_type"),
        "laser_sensors": "|".join(source_meta.get("laser_sensors", [])),
        "valid_depth_pixels": int(source_meta.get("valid_depth_pixels", 0)),
        "quality_ok": bool(source_meta.get("quality_ok", False)),
    }


def main():
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    models_dir = Path(args.models_dir).expanduser().resolve()
    target_hw = (int(args.target_height), int(args.target_width))
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else (target_hw[0] * target_hw[1]) / float(FULL_HW[0] * FULL_HW[1])
    )

    output_dirs = {
        "raw_native": output_root / "raw_4ch_native",
        "raw_eval": output_root / f"raw_4ch_{target_hw[0]}{target_hw[1]}",
        "rgb_eval": output_root / f"rgb_{target_hw[0]}{target_hw[1]}",
        "depth_proxy": output_root / f"depth_proxy_{target_hw[0]}{target_hw[1]}",
        "legacy_rgb": output_root / "rgb",
        "legacy_gt": output_root / "gt",
        "manifests": output_root / "manifests",
    }
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    sample_dirs = discover_sample_dirs(source_root)
    if not sample_dirs:
        raise RuntimeError(f"No RobotCar sample directories found under {source_root}")
    if args.sample_limit is not None:
        sample_dirs = sample_dirs[: args.sample_limit]

    lut_xy = load_lut_xy(models_dir)
    remap_pairs = build_half_res_maps(lut_xy)

    manifest_rows = []
    failed_rows = []
    skipped_quality_rows = []
    scene_stats = defaultdict(lambda: {"processed": 0, "failed": 0, "skipped_quality": 0})
    align_mode_counts = Counter()

    for index, sample_dir in enumerate(sample_dirs, start=1):
        meta_path = sample_dir / "meta.json"
        with meta_path.open("r", encoding="utf-8") as f:
            source_meta = json.load(f)
        scene = str(source_meta.get("run") or sample_dir.name.split("_", 1)[0])
        sample_name = sample_dir.name
        if not bool(source_meta.get("quality_ok", False)):
            skipped_quality_rows.append({"scene": scene, "sample_name": sample_name, "reason": "quality_ok=false"})
            scene_stats[scene]["skipped_quality"] += 1
            continue

        rgb_src_path = (sample_dir / "rgb.png").resolve()
        raw_src_path = (sample_dir / "raw.png").resolve()
        depth_src_path = (sample_dir / "gt_depth.npy").resolve()
        raw_native_path = output_dirs["raw_native"] / scene / f"{sample_name}.npz"
        raw_eval_path = output_dirs["raw_eval"] / scene / f"{sample_name}.npz"
        rgb_eval_path = output_dirs["rgb_eval"] / scene / f"{sample_name}.png"
        depth_proxy_path = output_dirs["depth_proxy"] / scene / f"{sample_name}.npz"
        raw_meta_out_path = sample_meta_path(raw_native_path)
        legacy_rgb_path = output_dirs["legacy_rgb"] / f"{sample_name}.png"
        legacy_gt_path = output_dirs["legacy_gt"] / f"{sample_name}.npy"

        existing_outputs = (
            raw_native_path.is_file(),
            raw_eval_path.is_file(),
            rgb_eval_path.is_file(),
            depth_proxy_path.is_file(),
            raw_meta_out_path.is_file(),
        )
        if args.skip_existing and all(existing_outputs):
            manifest_rows.append(
                build_manifest_row(
                    scene=scene,
                    sample_name=sample_name,
                    rgb_src_path=rgb_src_path,
                    rgb_eval_path=rgb_eval_path,
                    raw_src_path=raw_src_path,
                    raw_native_path=raw_native_path,
                    raw_eval_path=raw_eval_path,
                    depth_src_path=depth_src_path,
                    depth_proxy_path=depth_proxy_path,
                    meta_src_path=meta_path.resolve(),
                    target_hw=target_hw,
                    source_meta=source_meta,
                )
            )
            create_or_update_symlink(legacy_rgb_path, rgb_src_path)
            create_or_update_symlink(legacy_gt_path, depth_src_path)
            scene_stats[scene]["processed"] += 1
            align_mode_counts["robotcar_lut_halfres_rectify"] += 1
            continue

        try:
            raw = load_raw_bayer(raw_src_path)
            rect_native = rectify_packed_raw(raw, remap_pairs)
            rect_eval = (
                rect_native
                if target_hw == RAW_NATIVE_HW
                else cv2.resize(rect_native, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
            )

            save_npz(raw_native_path, **{RECTIFIED_BAYER_KEY: rect_native})
            save_npz(raw_eval_path, **{RECTIFIED_BAYER_KEY: rect_eval.astype(np.float32, copy=False)})
            save_rgb_copy(rgb_src_path, rgb_eval_path, target_hw)

            depth_full = load_depth_full(depth_src_path)
            depth_proxy, valid_proxy, coverage = build_depth_proxy(depth_full, target_hw, threshold=threshold)
            save_npz(
                depth_proxy_path,
                **{
                    DEPTH_PROXY_KEY: depth_proxy,
                    VALID_MASK_KEY: valid_proxy.astype(bool, copy=False),
                    PROXY_COVERAGE_KEY: coverage.astype(np.float32, copy=False),
                },
            )

            sample_meta_payload = build_sample_meta_payload(
                scene=scene,
                sample_name=sample_name,
                target_hw=target_hw,
                source_meta=source_meta,
            )
            write_sample_meta(raw_meta_out_path, sample_meta_payload)
            create_or_update_symlink(legacy_rgb_path, rgb_src_path)
            create_or_update_symlink(legacy_gt_path, depth_src_path)

            manifest_rows.append(
                build_manifest_row(
                    scene=scene,
                    sample_name=sample_name,
                    rgb_src_path=rgb_src_path,
                    rgb_eval_path=rgb_eval_path,
                    raw_src_path=raw_src_path,
                    raw_native_path=raw_native_path,
                    raw_eval_path=raw_eval_path,
                    depth_src_path=depth_src_path,
                    depth_proxy_path=depth_proxy_path,
                    meta_src_path=meta_path.resolve(),
                    target_hw=target_hw,
                    source_meta=source_meta,
                )
            )
            scene_stats[scene]["processed"] += 1
            align_mode_counts["robotcar_lut_halfres_rectify"] += 1
        except Exception as exc:
            failed_rows.append({"scene": scene, "sample_name": sample_name, "reason": str(exc)})
            scene_stats[scene]["failed"] += 1

        if index % 100 == 0 or index == len(sample_dirs):
            print(f"[prepare_robotcar] processed {index}/{len(sample_dirs)}", flush=True)

    manifest_rows.sort(key=lambda row: (str(row["scene"]), str(row["sample_name"])))
    manifest_path = output_dirs["manifests"] / "robotcar_raw_depth_v1_val.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [])
        if manifest_rows:
            writer.writeheader()
            writer.writerows(manifest_rows)

    failed_path = output_dirs["manifests"] / "robotcar_failed.csv"
    with failed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "sample_name", "reason"])
        writer.writeheader()
        writer.writerows(failed_rows)

    skipped_quality_path = output_dirs["manifests"] / "robotcar_skipped_quality.csv"
    with skipped_quality_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "sample_name", "reason"])
        writer.writeheader()
        writer.writerows(skipped_quality_rows)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "models_dir": str(models_dir),
        "target_hw": hw_to_str(target_hw),
        "raw_native_hw": hw_to_str(RAW_NATIVE_HW),
        "raw_eval_matches_native": bool(target_hw == RAW_NATIVE_HW),
        "depth_full_hw": hw_to_str(FULL_HW),
        "coverage_threshold": float(threshold),
        "discovered_samples": int(len(sample_dirs)),
        "processed_samples": int(len(manifest_rows)),
        "failed_samples": int(len(failed_rows)),
        "skipped_quality_samples": int(len(skipped_quality_rows)),
        "align_mode_counts": dict(align_mode_counts),
        "scene_stats": scene_stats,
        "manifest_path": str(manifest_path),
        "failed_path": str(failed_path),
        "skipped_quality_path": str(skipped_quality_path),
    }
    summary_path = output_dirs["manifests"] / "robotcar_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
