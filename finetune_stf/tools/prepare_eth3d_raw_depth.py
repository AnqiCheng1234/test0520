from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    import rawpy
except ImportError as exc:  # pragma: no cover - fail fast in runtime usage
    raise SystemExit(
        "rawpy is required for ETH3D preprocessing. Install a working rawpy/LibRaw build first."
    ) from exc


DEFAULT_SOURCE_ROOT = Path("/mnt/drive/3333_raw/eth3d")
DEFAULT_OUTPUT_ROOT = Path("/mnt/drive/3333_raw/eth3d_raw_depth_640960")
EXPECTED_FULL_HW = (4032, 6048)
DEFAULT_EVAL_HW = (640, 960)
RAW_NATIVE_KEY = "raw_4ch"
DEPTH_PROXY_KEY = "depth"
VALID_MASK_KEY = "valid_mask"
PROXY_COVERAGE_KEY = "coverage"
EXPECTED_RAW_PATTERN = np.array([[0, 1], [3, 2]], dtype=np.int16)
EXPECTED_COLOR_DESC = b"RGBG"
META_SUFFIX = ".meta.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ETH3D raw/rgb/depth validation assets.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--target-height", type=int, default=DEFAULT_EVAL_HW[0])
    parser.add_argument("--target-width", type=int, default=DEFAULT_EVAL_HW[1])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--preview-scene", default=None)
    parser.add_argument("--preview-stem", default=None)
    parser.add_argument("--preview-path", default=None)
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


def list_stems(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        return set()
    suffix = suffix.lower()
    return {path.stem for path in directory.iterdir() if path.is_file() and path.suffix.lower() == suffix}


def discover_scene_inputs(source_root: Path) -> list[dict[str, Path | str]]:
    raw_root = source_root / "multi_view_training_dslr_raw"
    jpg_root = source_root / "multi_view_training_dslr_jpg"
    depth_root = source_root / "multi_view_training_dslr_depth"
    scene_inputs = []
    for raw_scene_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        scene = raw_scene_dir.name
        scene_inputs.append(
            {
                "scene": scene,
                "raw_dir": raw_scene_dir / "images" / "dslr_images",
                "jpg_dir": jpg_root / scene / "images" / "dslr_images",
                "depth_dir": depth_root / scene / scene / "ground_truth_depth" / "dslr_images",
                "raw_calib": raw_scene_dir / "dslr_calibration_raw" / "cameras.txt",
                "jpg_calib": jpg_root / scene / "dslr_calibration_jpg" / "cameras.txt",
            }
        )
    return scene_inputs


def assert_scene_layout(scene_info: dict[str, Path | str]) -> None:
    scene = str(scene_info["scene"])
    for key in ("raw_dir", "jpg_dir", "depth_dir", "raw_calib", "jpg_calib"):
        path = Path(scene_info[key])
        if not path.exists():
            raise FileNotFoundError(f"Missing ETH3D scene asset for {scene}: {path}")
    if Path(scene_info["raw_calib"]).read_bytes() != Path(scene_info["jpg_calib"]).read_bytes():
        raise RuntimeError(f"Calibration mismatch between raw/jpg scene roots for {scene}")


def size_field(sizes, name: str) -> int | None:
    value = getattr(sizes, name, None)
    return None if value is None else int(value)


def build_tiled_color_indices(raw_pattern: np.ndarray, shape: tuple[int, int], *, top: int = 0, left: int = 0) -> np.ndarray:
    h, w = shape
    yy = (np.arange(top, top + h, dtype=np.int32)[:, None]) % raw_pattern.shape[0]
    xx = (np.arange(left, left + w, dtype=np.int32)[None, :]) % raw_pattern.shape[1]
    return raw_pattern[yy, xx]


def check_raw_layout(raw) -> tuple[np.ndarray, bytes]:
    raw_pattern = np.asarray(raw.raw_pattern, dtype=np.int16)
    color_desc = bytes(raw.color_desc)
    if raw_pattern.shape != (2, 2) or not np.array_equal(raw_pattern, EXPECTED_RAW_PATTERN):
        raise RuntimeError(f"Unsupported raw pattern: {raw_pattern!r}")
    if color_desc != EXPECTED_COLOR_DESC:
        raise RuntimeError(f"Unsupported color_desc: {color_desc!r}")
    return raw_pattern, color_desc


def resolve_raw_crop(raw, expected_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict[str, int | str | None]]:
    raw_pattern, _ = check_raw_layout(raw)
    sizes = raw.sizes
    expected_h, expected_w = expected_hw
    size_meta = {
        "raw_height": size_field(sizes, "raw_height"),
        "raw_width": size_field(sizes, "raw_width"),
        "height": size_field(sizes, "height"),
        "width": size_field(sizes, "width"),
        "top_margin": size_field(sizes, "top_margin"),
        "left_margin": size_field(sizes, "left_margin"),
        "iheight": size_field(sizes, "iheight"),
        "iwidth": size_field(sizes, "iwidth"),
        "crop_top_margin": size_field(sizes, "crop_top_margin"),
        "crop_left_margin": size_field(sizes, "crop_left_margin"),
        "crop_height": size_field(sizes, "crop_height"),
        "crop_width": size_field(sizes, "crop_width"),
    }

    raw_image = np.asarray(raw.raw_image, dtype=np.float32)
    raw_colors = np.asarray(raw.raw_colors) if raw.raw_colors is not None else None
    candidate = None
    for height_key, width_key, top_key, left_key in (
        ("iheight", "iwidth", "top_margin", "left_margin"),
        ("height", "width", "top_margin", "left_margin"),
        ("crop_height", "crop_width", "crop_top_margin", "crop_left_margin"),
    ):
        crop_h = size_meta[height_key]
        crop_w = size_meta[width_key]
        crop_top = size_meta[top_key]
        crop_left = size_meta[left_key]
        if crop_h != expected_h or crop_w != expected_w:
            continue
        if crop_top is None or crop_left is None:
            continue
        if crop_top + expected_h > raw_image.shape[0] or crop_left + expected_w > raw_image.shape[1]:
            continue
        candidate = {
            "raw_align_mode": "camera_native_candidate",
            "crop_top": int(crop_top),
            "crop_left": int(crop_left),
            "crop_height": expected_h,
            "crop_width": expected_w,
        }
        break

    if candidate is not None:
        crop_top = int(candidate["crop_top"])
        crop_left = int(candidate["crop_left"])
        if crop_top % 2 != 0 or crop_left % 2 != 0:
            raise ValueError(f"camera-native candidate starts on odd coordinates {(crop_top, crop_left)}")
        raw_crop = raw_image[crop_top : crop_top + expected_h, crop_left : crop_left + expected_w]
        if raw_colors is not None:
            color_crop = raw_colors[crop_top : crop_top + expected_h, crop_left : crop_left + expected_w]
        else:
            color_crop = build_tiled_color_indices(raw_pattern, expected_hw, top=crop_top, left=crop_left)
        meta = {**size_meta, **candidate}
        return raw_crop, np.asarray(color_crop), meta

    visible = np.asarray(raw.raw_image_visible, dtype=np.float32)
    colors_visible = np.asarray(raw.raw_colors_visible) if raw.raw_colors_visible is not None else None
    delta_h = visible.shape[0] - expected_h
    delta_w = visible.shape[1] - expected_w
    if delta_h < 0 or delta_w < 0:
        raise RuntimeError(
            f"Visible RAW is smaller than target crop: visible={visible.shape}, expected={expected_hw}"
        )
    crop_top = delta_h // 2
    crop_left = delta_w // 2
    crop_bottom = delta_h - crop_top
    crop_right = delta_w - crop_left
    if crop_top % 2 != 0 or crop_left % 2 != 0:
        raise ValueError(f"center crop starts on odd coordinates {(crop_top, crop_left)}")
    raw_crop = visible[crop_top : visible.shape[0] - crop_bottom, crop_left : visible.shape[1] - crop_right]
    if colors_visible is not None:
        color_crop = colors_visible[crop_top : visible.shape[0] - crop_bottom, crop_left : visible.shape[1] - crop_right]
    else:
        color_crop = build_tiled_color_indices(raw_pattern, expected_hw)
    if raw_crop.shape != expected_hw:
        raise RuntimeError(f"Unexpected fallback crop shape {raw_crop.shape}, expected {expected_hw}")
    meta = {
        **size_meta,
        "raw_align_mode": "center_crop_fallback",
        "crop_top": int(crop_top),
        "crop_left": int(crop_left),
        "crop_height": expected_h,
        "crop_width": expected_w,
    }
    return raw_crop, np.asarray(color_crop), meta


def normalize_raw_crop(raw_crop: np.ndarray, color_crop: np.ndarray, raw) -> np.ndarray:
    black_levels = np.asarray(raw.black_level_per_channel, dtype=np.float32)
    if black_levels.shape[0] < 4:
        raise RuntimeError(f"Expected 4 black levels, got {black_levels}")
    color_crop = np.asarray(color_crop, dtype=np.int16)
    black_map = black_levels[color_crop]
    denom = np.maximum(float(raw.white_level) - black_map, 1.0)
    normalized = (raw_crop.astype(np.float32) - black_map) / denom
    return np.clip(normalized, 0.0, 1.0)


def pack_semantic_rggb(raw_crop_norm: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            raw_crop_norm[0::2, 0::2],
            raw_crop_norm[0::2, 1::2],
            raw_crop_norm[1::2, 0::2],
            raw_crop_norm[1::2, 1::2],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def save_rgb_copy(rgb_src_path: Path, rgb_dst_path: Path, target_hw: tuple[int, int]) -> None:
    image = Image.open(rgb_src_path).convert("RGB")
    resized = image.resize((target_hw[1], target_hw[0]), Image.Resampling.BICUBIC)
    ensure_parent(rgb_dst_path)
    resized.save(rgb_dst_path, quality=95)


def load_depth_full(depth_path: Path, expected_hw: tuple[int, int]) -> np.ndarray:
    expected_size = expected_hw[0] * expected_hw[1]
    depth = np.fromfile(depth_path, dtype=np.float32)
    if depth.size != expected_size:
        raise RuntimeError(f"Unexpected depth payload size for {depth_path}: {depth.size} != {expected_size}")
    return depth.reshape(expected_hw)


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


def save_npz(path: Path, **arrays) -> None:
    ensure_parent(path)
    np.savez_compressed(path, **arrays)


def sample_meta_path(raw_native_path: Path) -> Path:
    return raw_native_path.with_suffix(META_SUFFIX)


def build_sample_meta_payload(
    *,
    crop_meta: dict[str, int | str | None],
    white_level: float,
    black_levels: np.ndarray,
) -> dict[str, int | float | str | None]:
    return {
        "cfa_pattern": "RGGB",
        "pack_order": "[R,Gr,Gb,B]",
        "raw_domain": "sensor_linear_per_channel",
        "raw_align_mode": str(crop_meta["raw_align_mode"]),
        "crop_top": int(crop_meta["crop_top"]),
        "crop_left": int(crop_meta["crop_left"]),
        "crop_height": int(crop_meta["crop_height"]),
        "crop_width": int(crop_meta["crop_width"]),
        "white_level": float(white_level),
        "black_level_0": float(black_levels[0]),
        "black_level_1": float(black_levels[1]),
        "black_level_2": float(black_levels[2]),
        "black_level_3": float(black_levels[3]),
        "raw_height": crop_meta.get("raw_height"),
        "raw_width": crop_meta.get("raw_width"),
        "height": crop_meta.get("height"),
        "width": crop_meta.get("width"),
        "top_margin": crop_meta.get("top_margin"),
        "left_margin": crop_meta.get("left_margin"),
        "iheight": crop_meta.get("iheight"),
        "iwidth": crop_meta.get("iwidth"),
        "crop_top_margin": crop_meta.get("crop_top_margin"),
        "crop_left_margin": crop_meta.get("crop_left_margin"),
        "crop_height_meta": crop_meta.get("crop_height"),
        "crop_width_meta": crop_meta.get("crop_width"),
    }


def write_sample_meta(meta_path: Path, payload: dict[str, int | float | str | None]) -> None:
    ensure_parent(meta_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_sample_meta(raw_src_path: Path, meta_path: Path) -> dict[str, int | float | str | None]:
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    with rawpy.imread(str(raw_src_path)) as raw:
        check_raw_layout(raw)
        _, _, crop_meta = resolve_raw_crop(raw, EXPECTED_FULL_HW)
        black_levels = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        white_level = float(raw.white_level)
    payload = build_sample_meta_payload(
        crop_meta=crop_meta,
        white_level=white_level,
        black_levels=black_levels,
    )
    write_sample_meta(meta_path, payload)
    return payload


def pack_preview_rgb(raw_eval_4ch: np.ndarray) -> np.ndarray:
    preview = np.stack(
        [
            raw_eval_4ch[..., 0],
            0.5 * (raw_eval_4ch[..., 1] + raw_eval_4ch[..., 2]),
            raw_eval_4ch[..., 3],
        ],
        axis=-1,
    )
    preview = np.clip(preview, 0.0, 1.0) ** (1.0 / 2.2)
    return (preview * 255.0).round().astype(np.uint8)


def build_triptych(
    *,
    rgb_src_path: Path,
    raw_eval_4ch: np.ndarray,
    depth_proxy: np.ndarray,
    valid_proxy: np.ndarray,
    threshold: float,
    out_path: Path,
    scene: str,
    stem: str,
) -> None:
    target_hw = tuple(int(v) for v in raw_eval_4ch.shape[:2])
    rgb = np.array(Image.open(rgb_src_path).convert("RGB"), dtype=np.uint8)
    rgb_eval = cv2.resize(rgb, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_CUBIC)
    raw_preview = pack_preview_rgb(raw_eval_4ch)

    depth_vis = np.zeros_like(depth_proxy, dtype=np.float32)
    if np.any(valid_proxy):
        lo = float(np.percentile(depth_proxy[valid_proxy], 2))
        hi = float(np.percentile(depth_proxy[valid_proxy], 98))
        if hi <= lo:
            hi = lo + 1e-6
        depth_vis[valid_proxy] = (depth_proxy[valid_proxy] - lo) / (hi - lo)
    depth_u8 = np.clip(depth_vis, 0.0, 1.0)
    depth_u8 = (depth_u8 * 255.0).astype(np.uint8)
    depth_panel = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    depth_panel = cv2.cvtColor(depth_panel, cv2.COLOR_BGR2RGB)
    depth_panel[~valid_proxy] = 0

    panels = [rgb_eval, raw_preview, depth_panel]
    labels = [
        f"RGB {rgb.shape[1]}x{rgb.shape[0]} -> {target_hw[1]}x{target_hw[0]}",
        f"RAW [R,Gr,Gb,B] -> {target_hw[1]}x{target_hw[0]}",
        f"Depth proxy valid>{threshold:.4f}",
    ]

    gap = 24
    top_band = 64
    canvas = Image.new(
        "RGB",
        (len(panels) * target_hw[1] + (len(panels) - 1) * gap, target_hw[0] + top_band),
        color=(18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)
    for idx, (panel, label) in enumerate(zip(panels, labels)):
        x0 = idx * (target_hw[1] + gap)
        canvas.paste(Image.fromarray(panel), (x0, top_band))
        draw.text((x0 + 12, 18), label, fill=(240, 240, 240))
    draw.text((12, canvas.height - 24), f"{scene}/{stem}", fill=(180, 180, 180))
    ensure_parent(out_path)
    canvas.save(out_path)


def build_manifest_row(
    *,
    scene: str,
    stem: str,
    rgb_src_path: Path,
    rgb_eval_path: Path,
    raw_native_path: Path,
    raw_eval_path: Path,
    depth_src_path: Path,
    depth_proxy_path: Path,
    target_hw: tuple[int, int],
    raw_native_hw: str,
    meta_payload: dict[str, int | float | str | None],
) -> dict[str, int | float | str | None]:
    return {
        "scene": scene,
        "sample_name": stem,
        "rgb_src_path": str(rgb_src_path),
        "rgb_640_path": str(rgb_eval_path),
        "raw_native_path": str(raw_native_path),
        "raw_640_path": str(raw_eval_path),
        "depth_src_path": str(depth_src_path),
        "depth_proxy_path": str(depth_proxy_path),
        "rgb_hw": hw_to_str(EXPECTED_FULL_HW),
        "raw_native_hw": raw_native_hw,
        "raw_eval_hw": hw_to_str(target_hw),
        "depth_full_hw": hw_to_str(EXPECTED_FULL_HW),
        "depth_fast_hw": hw_to_str(target_hw),
        "cfa_pattern": meta_payload["cfa_pattern"],
        "pack_order": meta_payload["pack_order"],
        "raw_domain": meta_payload["raw_domain"],
        "raw_align_mode": meta_payload["raw_align_mode"],
        "crop_top": meta_payload["crop_top"],
        "crop_left": meta_payload["crop_left"],
        "crop_height": meta_payload["crop_height"],
        "crop_width": meta_payload["crop_width"],
        "white_level": meta_payload["white_level"],
        "black_level_0": meta_payload["black_level_0"],
        "black_level_1": meta_payload["black_level_1"],
        "black_level_2": meta_payload["black_level_2"],
        "black_level_3": meta_payload["black_level_3"],
    }


def main():
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    target_hw = (int(args.target_height), int(args.target_width))
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else (target_hw[0] * target_hw[1]) / float(EXPECTED_FULL_HW[0] * EXPECTED_FULL_HW[1])
    )

    output_dirs = {
        "raw_native": output_root / "raw_4ch_native",
        "raw_eval": output_root / "raw_4ch_640960",
        "rgb_eval": output_root / "rgb_640960",
        "depth_proxy": output_root / "depth_proxy_640960",
        "manifests": output_root / "manifests",
    }
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    scene_inputs = discover_scene_inputs(source_root)
    if not scene_inputs:
        raise RuntimeError(f"No ETH3D training scenes found under {source_root}")

    manifest_rows = []
    missing_raw_rows = []
    skipped_alignment_rows = []
    align_mode_counts = Counter()
    scene_stats = defaultdict(lambda: {"jpg_depth_pairs": 0, "complete_pairs": 0, "missing_raw": 0, "skipped_alignment": 0})

    complete_samples = []
    for scene_info in scene_inputs:
        assert_scene_layout(scene_info)
        scene = str(scene_info["scene"])
        raw_stems = list_stems(Path(scene_info["raw_dir"]), ".nef")
        jpg_stems = list_stems(Path(scene_info["jpg_dir"]), ".jpg")
        depth_stems = list_stems(Path(scene_info["depth_dir"]), ".jpg")
        jpg_depth = sorted(jpg_stems & depth_stems)
        complete = sorted(raw_stems & jpg_stems & depth_stems)
        missing_raw = sorted((jpg_stems & depth_stems) - raw_stems)
        scene_stats[scene]["jpg_depth_pairs"] = len(jpg_depth)
        scene_stats[scene]["complete_pairs_total"] = len(complete)
        scene_stats[scene]["complete_pairs"] = len(complete)
        scene_stats[scene]["missing_raw"] = len(missing_raw)
        for stem in missing_raw:
            missing_raw_rows.append({"scene": scene, "sample_name": stem, "reason": "missing_nef"})
        complete_samples.extend((scene, stem, scene_info) for stem in complete)

    total_complete = len(complete_samples)
    if args.sample_limit is not None:
        complete_samples = complete_samples[: args.sample_limit]
        selected_counts = Counter(scene for scene, _, _ in complete_samples)
        for scene in scene_stats:
            scene_stats[scene]["complete_pairs"] = int(selected_counts.get(scene, 0))

    preview_scene = args.preview_scene
    preview_stem = args.preview_stem
    preview_path = (
        Path(args.preview_path).expanduser().resolve()
        if args.preview_path
        else output_root / "manifests" / "eth3d_preview_triptych.png"
    )

    for index, (scene, stem, scene_info) in enumerate(complete_samples, start=1):
        raw_src_path = Path(scene_info["raw_dir"]) / f"{stem}.NEF"
        rgb_src_path = Path(scene_info["jpg_dir"]) / f"{stem}.JPG"
        depth_src_path = Path(scene_info["depth_dir"]) / f"{stem}.JPG"
        raw_native_path = output_dirs["raw_native"] / scene / f"{stem}.npz"
        raw_eval_path = output_dirs["raw_eval"] / scene / f"{stem}.npz"
        rgb_eval_path = output_dirs["rgb_eval"] / scene / f"{stem}.jpg"
        depth_proxy_path = output_dirs["depth_proxy"] / scene / f"{stem}.npz"
        meta_path = sample_meta_path(raw_native_path)

        if args.skip_existing and all(
            path.is_file() for path in (raw_native_path, raw_eval_path, rgb_eval_path, depth_proxy_path)
        ):
            meta_payload = load_sample_meta(raw_src_path, meta_path)
            align_mode_counts[str(meta_payload["raw_align_mode"])] += 1
            manifest_rows.append(
                build_manifest_row(
                    scene=scene,
                    stem=stem,
                    rgb_src_path=rgb_src_path,
                    rgb_eval_path=rgb_eval_path,
                    raw_native_path=raw_native_path,
                    raw_eval_path=raw_eval_path,
                    depth_src_path=depth_src_path,
                    depth_proxy_path=depth_proxy_path,
                    target_hw=target_hw,
                    raw_native_hw="2016x3024",
                    meta_payload=meta_payload,
                )
            )
            continue

        with rawpy.imread(str(raw_src_path)) as raw:
            raw_pattern, color_desc = check_raw_layout(raw)
            try:
                raw_crop, color_crop, crop_meta = resolve_raw_crop(raw, EXPECTED_FULL_HW)
            except ValueError as exc:
                skipped_alignment_rows.append(
                    {
                        "scene": scene,
                        "sample_name": stem,
                        "reason": str(exc),
                    }
                )
                scene_stats[scene]["skipped_alignment"] += 1
                continue

            normalized_crop = normalize_raw_crop(raw_crop, color_crop, raw)
            packed_native = pack_semantic_rggb(normalized_crop)
            raw_eval_4ch = cv2.resize(
                packed_native,
                (target_hw[1], target_hw[0]),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32, copy=False)
            black_levels = np.asarray(raw.black_level_per_channel, dtype=np.float32)
            white_level = float(raw.white_level)
            meta_payload = build_sample_meta_payload(
                crop_meta=crop_meta,
                white_level=white_level,
                black_levels=black_levels,
            )

        depth_full = load_depth_full(depth_src_path, EXPECTED_FULL_HW)
        depth_proxy, valid_proxy, depth_coverage = build_depth_proxy(depth_full, target_hw, threshold=threshold)

        save_npz(raw_native_path, **{RAW_NATIVE_KEY: packed_native})
        save_npz(raw_eval_path, **{RAW_NATIVE_KEY: raw_eval_4ch})
        save_rgb_copy(rgb_src_path, rgb_eval_path, target_hw)
        write_sample_meta(meta_path, meta_payload)
        save_npz(
            depth_proxy_path,
            **{
                DEPTH_PROXY_KEY: depth_proxy.astype(np.float32, copy=False),
                VALID_MASK_KEY: valid_proxy.astype(np.uint8, copy=False),
                PROXY_COVERAGE_KEY: depth_coverage.astype(np.float32, copy=False),
            },
        )

        align_mode_counts[str(crop_meta["raw_align_mode"])] += 1
        manifest_rows.append(
            build_manifest_row(
                scene=scene,
                stem=stem,
                rgb_src_path=rgb_src_path,
                rgb_eval_path=rgb_eval_path,
                raw_native_path=raw_native_path,
                raw_eval_path=raw_eval_path,
                depth_src_path=depth_src_path,
                depth_proxy_path=depth_proxy_path,
                target_hw=target_hw,
                raw_native_hw=hw_to_str(packed_native.shape[:2]),
                meta_payload=meta_payload,
            )
        )

        if (preview_scene is None and preview_stem is None and index == 1) or (
            preview_scene == scene and preview_stem == stem
        ):
            build_triptych(
                rgb_src_path=rgb_src_path,
                raw_eval_4ch=raw_eval_4ch,
                depth_proxy=depth_proxy,
                valid_proxy=valid_proxy,
                threshold=threshold,
                out_path=preview_path,
                scene=scene,
                stem=stem,
            )

        if index == 1 or index % 25 == 0 or index == len(complete_samples):
            print(f"[prepare_eth3d] processed {index}/{len(complete_samples)}: {scene}/{stem}", flush=True)

    manifest_rows.sort(key=lambda row: (row["scene"], row["sample_name"]))
    missing_raw_rows.sort(key=lambda row: (row["scene"], row["sample_name"]))
    skipped_alignment_rows.sort(key=lambda row: (row["scene"], row["sample_name"]))

    manifest_path = output_dirs["manifests"] / "eth3d_raw_depth_v2_val.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [])
        if manifest_rows:
            writer.writeheader()
            writer.writerows(manifest_rows)

    missing_raw_path = output_dirs["manifests"] / "eth3d_missing_raw.csv"
    with missing_raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "sample_name", "reason"])
        writer.writeheader()
        writer.writerows(missing_raw_rows)

    skipped_alignment_path = output_dirs["manifests"] / "eth3d_skipped_alignment.csv"
    with skipped_alignment_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "sample_name", "reason"])
        writer.writeheader()
        writer.writerows(skipped_alignment_rows)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "expected_full_hw": list(EXPECTED_FULL_HW),
        "target_hw": list(target_hw),
        "threshold": float(threshold),
        "jpg_depth_pairs": int(sum(scene_stats[scene]["jpg_depth_pairs"] for scene in scene_stats)),
        "complete_pairs": int(sum(scene_stats[scene]["complete_pairs"] for scene in scene_stats)),
        "complete_pairs_total": int(total_complete),
        "processed_pairs": int(len(manifest_rows)),
        "missing_raw_pairs": int(len(missing_raw_rows)),
        "skipped_alignment_pairs": int(len(skipped_alignment_rows)),
        "sample_limit": args.sample_limit,
        "align_mode_counts": dict(align_mode_counts),
        "scene_stats": {scene: dict(stats) for scene, stats in sorted(scene_stats.items())},
        "preview_path": str(preview_path) if preview_path.is_file() else None,
        "manifest_path": str(manifest_path),
        "missing_raw_path": str(missing_raw_path),
        "skipped_alignment_path": str(skipped_alignment_path),
        "rawpy_version": getattr(rawpy, "__version__", "unknown"),
    }
    summary_path = output_dirs["manifests"] / "eth3d_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(
        json.dumps(
            {
                "processed_pairs": len(manifest_rows),
                "missing_raw_pairs": len(missing_raw_rows),
                "skipped_alignment_pairs": len(skipped_alignment_rows),
                "manifest_path": str(manifest_path),
                "summary_path": str(summary_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
