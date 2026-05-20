#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetune_stf.dataset.raw_utils import (  # noqa: E402
    COMPANDED_MAX,
    RECTIFIED_BAYER_KEY,
)
from foundation.engine.transforms.unprocessing import build_unprocessing_transform_from_preset  # noqa: E402


DEFAULT_VKITTI_LIST = str(ROOT / "finetune_stf" / "dataset" / "splits" / "vkitti2" / "train.txt")
DEFAULT_OUTPUT = str(ROOT / "finetune_stf" / "tools" / "realraw_unprocessing_calibration.json")

RAW_PATH_KEYS = ("raw_native_path", "raw_eval_path", "raw_src_path")
PACK_ORDER_CANONICAL = "[R,Gr,Gb,B]"


def percentile_summary(values: List[float], ps: Tuple[int, ...] = (10, 50, 90)) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def sanitise_range(
    low: float,
    high: float,
    *,
    min_low: float | None = None,
    min_width: float | None = None,
) -> List[float]:
    low_f = float(low)
    high_f = float(high)
    if min_low is not None and low_f < min_low:
        low_f = float(min_low)
    if min_width is not None and (high_f - low_f) < min_width:
        high_f = low_f + float(min_width)
    return [low_f, high_f]


def _parse_float(value: object, default: float) -> float:
    if value is None:
        return float(default)
    text = str(value).strip()
    if not text:
        return float(default)
    try:
        return float(text)
    except ValueError:
        return float(default)


def _resolve_candidate_path(path_text: str, *, manifest_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (manifest_dir / path).resolve()


def _select_raw_path(row: Dict[str, str], *, manifest_dir: Path, prefer_order: Sequence[str]) -> Path:
    for key in prefer_order:
        value = str(row.get(key, "")).strip()
        if value:
            candidate = _resolve_candidate_path(value, manifest_dir=manifest_dir)
            if candidate.is_file():
                return candidate
    for key in RAW_PATH_KEYS:
        value = str(row.get(key, "")).strip()
        if value:
            candidate = _resolve_candidate_path(value, manifest_dir=manifest_dir)
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        "No usable raw path found for row. Checked keys: "
        f"{list(prefer_order)} then fallback {list(RAW_PATH_KEYS)}"
    )


def _parse_pack_order(pack_order: str) -> Tuple[str, ...]:
    text = str(pack_order).strip()
    if not text:
        return ("R", "Gr", "Gb", "B")
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    tokens = [item.strip() for item in text.split(",") if item.strip()]
    return tuple(tokens)


def _canonicalize_pack_order(raw_4ch: np.ndarray, pack_order: str) -> np.ndarray:
    tokens = _parse_pack_order(pack_order)
    if len(tokens) != 4:
        raise ValueError(f"Unsupported pack order format: {pack_order!r}")
    token_to_idx = {token: idx for idx, token in enumerate(tokens)}
    required = ("R", "Gr", "Gb", "B")
    if any(name not in token_to_idx for name in required):
        raise ValueError(f"Pack order must include {required}, got {tokens}")
    order = [token_to_idx[name] for name in required]
    return raw_4ch[..., order]


def _load_raw_4ch(path: Path, *, raw_key: str) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            selected = None
            if raw_key in data.files:
                selected = raw_key
            else:
                for candidate in ("bayer_rect", "raw_4ch", "raw", "packed_bayer", "bayer"):
                    if candidate in data.files:
                        selected = candidate
                        break
            if selected is None:
                for candidate in data.files:
                    array = np.asarray(data[candidate])
                    if array.ndim == 3 and (array.shape[-1] == 4 or array.shape[0] == 4):
                        selected = candidate
                        break
            if selected is None:
                raise KeyError(
                    f"{path} does not contain key '{raw_key}' and no 4-channel array was found. "
                    f"Available keys: {list(data.files)}"
                )
            raw = np.asarray(data[selected], dtype=np.float32)
    elif suffix == ".npy":
        raw = np.load(path)
    else:
        raise ValueError(f"Unsupported raw file suffix '{suffix}' for {path}")
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3:
        raise ValueError(f"Expected raw tensor with 3 dims (H,W,4), got {raw.shape} from {path}")
    if raw.shape[-1] == 4:
        return raw
    if raw.shape[0] == 4:
        return np.transpose(raw, (1, 2, 0))
    raise ValueError(f"Expected raw tensor with 4 channels, got {raw.shape} from {path}")


def _normalize_raw_sensor_linear(
    raw_4ch: np.ndarray,
    *,
    raw_domain: str,
    white_level: float,
    black_levels: Sequence[float],
) -> np.ndarray:
    domain = str(raw_domain or "").strip().lower()
    if not domain:
        domain = "sensor_linear_per_channel"

    if domain == "companded":
        return np.clip(raw_4ch / max(COMPANDED_MAX, 1e-6), 0.0, 1.0)

    black = np.asarray([float(v) for v in black_levels], dtype=np.float32).reshape(1, 1, 4)
    white = float(max(white_level, 1e-6))
    raw_max = float(np.max(raw_4ch))
    if raw_max <= 1.5:
        black_norm = black / white
        denom = np.maximum(1.0 - black_norm, 1e-6)
        normalized = (raw_4ch - black_norm) / denom
    else:
        denom = np.maximum(white - black, 1e-6)
        normalized = (raw_4ch - black) / denom

    return np.clip(normalized, 0.0, 1.0)


def compute_basic_stats(packed_01: np.ndarray) -> Dict[str, float]:
    r, gr, gb, b = (packed_01[..., i] for i in range(4))
    g = 0.5 * (gr + gb)
    eps = 1e-6
    return {
        "mean_R": float(r.mean()),
        "mean_G": float(g.mean()),
        "mean_B": float(b.mean()),
        "mean_all": float(packed_01.mean()),
        "p1_all": float(np.percentile(packed_01, 1)),
        "p5_all": float(np.percentile(packed_01, 5)),
        "p99_all": float(np.percentile(packed_01, 99)),
        "sat_ratio": float((packed_01 >= 1.0 - 1e-6).mean()),
        "G_over_R": float(g.mean() / (r.mean() + eps)),
        "G_over_B": float(g.mean() / (b.mean() + eps)),
    }


def extract_patch_stats(
    mean_source: np.ndarray,
    var_source: np.ndarray,
    *,
    patch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = mean_source.shape
    ph = h // patch_size
    pw = w // patch_size
    if ph < 2 or pw < 2:
        return np.empty((0, 4), dtype=np.float32), np.empty((0, 4), dtype=np.float32)
    mean_img = mean_source[: ph * patch_size, : pw * patch_size, :].reshape(ph, patch_size, pw, patch_size, 4)
    var_img = var_source[: ph * patch_size, : pw * patch_size, :].reshape(ph, patch_size, pw, patch_size, 4)
    patch_mean = mean_img.mean(axis=(1, 3))
    patch_var = var_img.mean(axis=(1, 3))
    return patch_mean.reshape(-1, 4), patch_var.reshape(-1, 4)


def extract_patch_mean_var_from_image(
    packed_01: np.ndarray,
    *,
    patch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = packed_01.shape
    ph = h // patch_size
    pw = w // patch_size
    if ph < 2 or pw < 2:
        return np.empty((0, 4), dtype=np.float32), np.empty((0, 4), dtype=np.float32)
    img = packed_01[: ph * patch_size, : pw * patch_size, :].reshape(ph, patch_size, pw, patch_size, 4)
    patch_mean = img.mean(axis=(1, 3))
    patch_var = img.var(axis=(1, 3))
    return patch_mean.reshape(-1, 4), patch_var.reshape(-1, 4)


def fit_lower_envelope_line(
    patch_means: np.ndarray,
    patch_vars: np.ndarray,
    *,
    num_bins: int = 20,
    min_bins: int = 5,
    min_range: float = 0.03,
    lower_env_pct: float = 10.0,
) -> Tuple[float | None, float | None]:
    fits = []
    for ch in range(4):
        m = patch_means[:, ch]
        v = patch_vars[:, ch]
        if m.size < 16 or (m.max() - m.min()) < min_range:
            continue
        edges = np.linspace(m.min(), m.max(), num_bins + 1)
        bin_indices = np.clip(np.digitize(m, edges[1:-1]), 0, num_bins - 1)
        bin_means, bin_vars = [], []
        for b in range(num_bins):
            mask = bin_indices == b
            if mask.sum() < 3:
                continue
            bin_means.append(float(m[mask].mean()))
            bin_vars.append(float(np.percentile(v[mask], lower_env_pct)))
        if len(bin_means) < min_bins:
            continue
        slope, intercept = np.polyfit(bin_means, bin_vars, 1)
        fits.append((max(float(intercept), 0.0), max(float(slope), 1e-8)))
    if not fits:
        return None, None
    arr = np.asarray(fits, dtype=np.float64)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def per_image_residual_noise_fit(packed_01: np.ndarray, *, patch_size: int) -> Tuple[float | None, float | None]:
    smooth = cv2.blur(packed_01, (3, 3))
    residual_sq = (packed_01 - smooth) ** 2
    patch_means, patch_vars = extract_patch_stats(smooth, residual_sq, patch_size=patch_size)
    if patch_means.shape[0] == 0:
        return None, None
    return fit_lower_envelope_line(patch_means, patch_vars, min_range=0.02)


def per_image_ptc(packed_01: np.ndarray, *, patch_size: int) -> Tuple[float | None, float | None]:
    patch_means, patch_vars = extract_patch_mean_var_from_image(packed_01, patch_size=patch_size)
    if patch_means.shape[0] == 0:
        return None, None
    return fit_lower_envelope_line(patch_means, patch_vars)


def summarise_basic_stats(items: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = list(items[0].keys()) if items else []
    return {k: percentile_summary([x[k] for x in items]) for k in keys}


def summarise_noise_fits(fits: List[Tuple[float, float]]) -> Dict[str, object]:
    read_vars = [x[0] for x in fits]
    shot_scales = [x[1] for x in fits]
    return {
        "num_images_fit": len(fits),
        "read_var": percentile_summary(read_vars),
        "shot_scale": percentile_summary(shot_scales),
    }


def build_vkitti_reference(vkitti_image_paths: List[str], num_samples: int, device: str) -> Dict[str, float | List[float] | int]:
    transform = build_unprocessing_transform_from_preset("stf_legacy", randomize=False).to(device)
    transform.eval()
    per_ch_means = []
    per_image_all_mean = []
    used = 0
    print(f"[VKITTI] building canonical reference on up to {num_samples} images", flush=True)
    for path in vkitti_image_paths:
        if used >= num_samples:
            break
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img)
        tensor = (
            torch.frombuffer(bytearray(img.tobytes()), dtype=torch.float32)
            .view(img.shape[0], img.shape[1], img.shape[2])
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            packed, _ = transform(tensor)
        arr = np.asarray(packed[0].detach().cpu().tolist(), dtype=np.float32)
        per_ch_means.append(arr.mean(axis=(1, 2)))
        per_image_all_mean.append(float(arr.mean()))
        used += 1
        if used % 25 == 0 or used == num_samples:
            print(f"[VKITTI] processed {used}/{num_samples}", flush=True)

    if not per_ch_means:
        raise RuntimeError("No VKITTI images were successfully processed.")
    per_ch_means_np = np.stack(per_ch_means, axis=0)
    return {
        "canonical_exposure_gain": float(transform.canonical_params.exposure_gain),
        "per_ch_mean_avg": [float(v) for v in per_ch_means_np.mean(axis=0)],
        "R_mean_avg": float(per_ch_means_np[:, 0].mean()),
        "G_mean_avg": float(per_ch_means_np[:, 1:3].mean()),
        "B_mean_avg": float(per_ch_means_np[:, 3].mean()),
        "all_mean_avg": float(np.mean(per_image_all_mean)),
        "num_samples": int(len(per_ch_means)),
    }


def _parse_black_levels_from_row(row: Dict[str, str]) -> List[float]:
    fields = [f"black_level_{i}" for i in range(4)]
    if all(str(row.get(k, "")).strip() for k in fields):
        return [_parse_float(row.get(k), 0.0) for k in fields]

    single = _parse_float(row.get("black_level"), 0.0)
    return [single, single, single, single]


def _build_recommended_preset(
    *,
    basic_summary: Dict[str, Dict[str, float]],
    noise_summary: Dict[str, object],
    vkitti_ref: Dict[str, float | List[float] | int],
    cfa_counter: Counter[str],
    black_level_meta_samples: List[float],
) -> Dict[str, object]:
    vkitti_g_ref = max(float(vkitti_ref["G_mean_avg"]), 1e-8)
    exposure_rec = [
        basic_summary["mean_G"]["p10"] / vkitti_g_ref,
        basic_summary["mean_G"]["p90"] / vkitti_g_ref,
    ]

    shot_scale_floor = 1e-5
    read_var_floor = 1e-10
    read_noise_rec = [
        math.sqrt(max(noise_summary["read_var"]["p10"], read_var_floor)),
        math.sqrt(max(noise_summary["read_var"]["p90"], read_var_floor)),
    ]
    shot_log_gain_rec = [
        math.log(max(noise_summary["shot_scale"]["p10"], shot_scale_floor)),
        math.log(max(noise_summary["shot_scale"]["p90"], shot_scale_floor)),
    ]
    if black_level_meta_samples:
        black_summary = percentile_summary(black_level_meta_samples)
        black_low = black_summary["p10"]
        black_high = black_summary["p90"]
    else:
        black_low = basic_summary["p1_all"]["p10"]
        black_high = basic_summary["p1_all"]["p90"]

    cfa_patterns = [name for name, _ in cfa_counter.most_common() if name]
    if not cfa_patterns:
        cfa_patterns = ["RGGB"]

    return {
        "red_gain_range": sanitise_range(
            basic_summary["G_over_R"]["p10"],
            basic_summary["G_over_R"]["p90"],
            min_low=1e-2,
            min_width=1e-2,
        ),
        "blue_gain_range": sanitise_range(
            basic_summary["G_over_B"]["p10"],
            basic_summary["G_over_B"]["p90"],
            min_low=1e-2,
            min_width=1e-2,
        ),
        "black_level_range": sanitise_range(black_low, black_high, min_low=0.0, min_width=1e-4),
        "exposure_gain_range": sanitise_range(exposure_rec[0], exposure_rec[1], min_low=1e-3, min_width=1e-3),
        "read_noise_std_range": sanitise_range(read_noise_rec[0], read_noise_rec[1], min_low=0.0, min_width=1e-4),
        "shot_log_gain_range": sanitise_range(shot_log_gain_rec[0], shot_log_gain_rec[1], min_width=1e-2),
        "cfa_patterns": cfa_patterns,
    }


def _parse_manifest_rows(manifest_paths: Iterable[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for manifest_text in manifest_paths:
        manifest_path = Path(manifest_text).expanduser().resolve()
        with manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "manifest_path": manifest_path,
                        "row": row,
                    }
                )
    return rows


def _build_prefer_order(value: str) -> Tuple[str, ...]:
    options = {
        "native": ("raw_native_path", "raw_eval_path", "raw_src_path"),
        "eval": ("raw_eval_path", "raw_native_path", "raw_src_path"),
        "src": ("raw_src_path", "raw_native_path", "raw_eval_path"),
    }
    key = str(value).strip().lower()
    if key not in options:
        raise ValueError(f"Unsupported --prefer-raw-path={value}. Expected one of {sorted(options)}")
    return options[key]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Brooks-style unprocessing ranges to real RAW manifests.")
    parser.add_argument("--manifest", action="append", required=True, help="Repeatable CSV manifest input.")
    parser.add_argument("--dataset-name", default="realraw_sensor_linear")
    parser.add_argument("--vkitti-list", default=DEFAULT_VKITTI_LIST)
    parser.add_argument("--num-vkitti", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=-1, help="Cap on real RAW rows (-1 means all).")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-interval", type=int, default=200)
    parser.add_argument("--prefer-raw-path", default="native", choices=["native", "eval", "src"])
    parser.add_argument("--raw-key", default=RECTIFIED_BAYER_KEY)
    args = parser.parse_args()

    prefer_order = _build_prefer_order(args.prefer_raw_path)
    rng = random.Random(args.seed)
    entries = _parse_manifest_rows(args.manifest)
    if not entries:
        raise RuntimeError(f"No rows loaded from manifests: {args.manifest}")
    rng.shuffle(entries)
    if args.num_samples > 0:
        entries = entries[: args.num_samples]
    print(f"[REALRAW] loaded {len(entries)} rows from {len(args.manifest)} manifest(s)", flush=True)

    basic_stats_items: List[Dict[str, float]] = []
    ptc_fits: List[Tuple[float, float]] = []
    residual_fits: List[Tuple[float, float]] = []
    residual_var_proxies: List[float] = []
    cfa_counter: Counter[str] = Counter()
    pack_order_counter: Counter[str] = Counter()
    raw_domain_counter: Counter[str] = Counter()
    white_levels: List[float] = []
    black_level_meta_samples: List[float] = []

    processed = 0
    skipped = 0
    for idx, entry in enumerate(entries, start=1):
        manifest_path = Path(entry["manifest_path"])
        manifest_dir = manifest_path.parent
        row = entry["row"]
        try:
            raw_path = _select_raw_path(row, manifest_dir=manifest_dir, prefer_order=prefer_order)
            raw_4ch = _load_raw_4ch(raw_path, raw_key=args.raw_key)
            pack_order = str(row.get("pack_order", PACK_ORDER_CANONICAL) or PACK_ORDER_CANONICAL)
            raw_4ch = _canonicalize_pack_order(raw_4ch, pack_order)

            black_levels = _parse_black_levels_from_row(row)
            white_level = _parse_float(row.get("white_level"), 1.0)
            raw_domain = str(row.get("raw_domain", "sensor_linear_per_channel") or "sensor_linear_per_channel")
            packed_01 = _normalize_raw_sensor_linear(
                raw_4ch,
                raw_domain=raw_domain,
                white_level=white_level,
                black_levels=black_levels,
            )
            basic = compute_basic_stats(packed_01)
            ptc_fit = per_image_ptc(packed_01, patch_size=args.patch_size)
            residual_fit = per_image_residual_noise_fit(packed_01, patch_size=args.patch_size)
            residual_var_proxies.append(float(np.var(packed_01 - cv2.blur(packed_01, (3, 3)))))
            if ptc_fit[0] is not None and ptc_fit[1] is not None:
                ptc_fits.append((ptc_fit[0], ptc_fit[1]))
            if residual_fit[0] is not None and residual_fit[1] is not None:
                residual_fits.append((residual_fit[0], residual_fit[1]))
            basic_stats_items.append(basic)

            cfa = str(row.get("cfa_pattern", "")).strip()
            if cfa:
                cfa_counter[cfa] += 1
            pack_order_counter[pack_order] += 1
            raw_domain_counter[raw_domain] += 1
            white_levels.append(float(white_level))
            black_level_meta_samples.extend(float(v) for v in black_levels)
            processed += 1
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"[WARN] skip row idx={idx}: {exc}", flush=True)
            continue

        if idx % args.progress_interval == 0 or idx == len(entries):
            print(
                f"[REALRAW] processed={processed} skipped={skipped} scanned={idx}/{len(entries)}",
                flush=True,
            )

    if not basic_stats_items:
        raise RuntimeError("No real raw samples were processed successfully.")

    with open(args.vkitti_list, "r", encoding="utf-8") as f:
        vkitti_image_paths = [line.strip().split()[0] for line in f if line.strip()]
    rng.shuffle(vkitti_image_paths)
    print(f"[VKITTI] loaded {len(vkitti_image_paths)} candidate RGB images from split", flush=True)
    vkitti_ref = build_vkitti_reference(vkitti_image_paths, args.num_vkitti, device=args.device)

    basic_summary = summarise_basic_stats(basic_stats_items)
    ptc_summary = summarise_noise_fits(ptc_fits)
    residual_summary = summarise_noise_fits(residual_fits)
    if residual_fits:
        noise_source = "residual_noise"
        noise_summary = residual_summary
    elif ptc_fits:
        noise_source = "patch_ptc"
        noise_summary = ptc_summary
    else:
        proxy_summary = percentile_summary(residual_var_proxies)
        proxy_floor = 1e-10
        noise_source = "global_residual_proxy"
        noise_summary = {
            "num_images_fit": 0,
            "read_var": {
                "p10": max(float(proxy_summary["p10"]), proxy_floor),
                "p50": max(float(proxy_summary["p50"]), proxy_floor),
                "p90": max(float(proxy_summary["p90"]), proxy_floor),
            },
            "shot_scale": {
                "p10": max(float(proxy_summary["p10"]), 1e-5),
                "p50": max(float(proxy_summary["p50"]), 1e-5),
                "p90": max(float(proxy_summary["p90"]), 1e-5),
            },
        }
    recommended = _build_recommended_preset(
        basic_summary=basic_summary,
        noise_summary=noise_summary,
        vkitti_ref=vkitti_ref,
        cfa_counter=cfa_counter,
        black_level_meta_samples=black_level_meta_samples,
    )

    report = {
        "args": vars(args),
        "dataset_name": args.dataset_name,
        "processed_rows": processed,
        "skipped_rows": skipped,
        "manifest_paths": [str(Path(item).expanduser().resolve()) for item in args.manifest],
        "source_summary": {
            "raw_domain_distribution": dict(raw_domain_counter),
            "cfa_distribution": dict(cfa_counter),
            "pack_order_distribution": dict(pack_order_counter),
            "white_level_summary": percentile_summary(white_levels),
            "black_level_summary": percentile_summary(black_level_meta_samples),
        },
        "vkitti_canonical_reference": vkitti_ref,
        "basic_summary": basic_summary,
        "patch_ptc_summary": ptc_summary,
        "residual_noise_summary": residual_summary,
        "selected_noise_source_for_recommendation": noise_source,
        "recommended_brooks_ranges": recommended,
        "notes": [
            "Raw path selection prefers raw_native_path by default, then falls back to raw_eval_path and raw_src_path.",
            "Normalization uses manifest white_level/black_level metadata for sensor-linear domains.",
            "Pack order is canonicalized to [R,Gr,Gb,B] before statistics and noise fitting.",
            "Noise recommendations prefer residual-noise lower-envelope fit, and fall back to patch-PTC when needed.",
        ],
    }

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[DONE] wrote report JSON to {out_path}", flush=True)

    print("=" * 88)
    print(
        f"dataset={args.dataset_name} processed={processed} skipped={skipped} "
        f"cfa={dict(cfa_counter)} pack_order={dict(pack_order_counter)}"
    )
    print(
        f"mean_all p10/p50/p90 = "
        f"{basic_summary['mean_all']['p10']:.6f} / "
        f"{basic_summary['mean_all']['p50']:.6f} / "
        f"{basic_summary['mean_all']['p90']:.6f}"
    )
    print(
        f"sat_ratio p10/p50/p90 = "
        f"{basic_summary['sat_ratio']['p10']:.6f} / "
        f"{basic_summary['sat_ratio']['p50']:.6f} / "
        f"{basic_summary['sat_ratio']['p90']:.6f}"
    )
    print(
        f"noise_source={noise_source} residual_fit_n={residual_summary['num_images_fit']} "
        f"ptc_fit_n={ptc_summary['num_images_fit']}"
    )
    print(f"recommended preset: {json.dumps(recommended, ensure_ascii=False)}")
    print("=" * 88)


if __name__ == "__main__":
    main()
