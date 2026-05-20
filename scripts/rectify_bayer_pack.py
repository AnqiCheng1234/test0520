#!/usr/bin/env python3
"""
Phase 1 Step 1: Bayer Pack + Half-Resolution Rectification

Processes all STF RAW TIFFs into rectified 4-channel Bayer-packed NPZ files,
and generates alignment verification visualizations for a subset.

RAW images are unrectified; LiDAR GT is in rectified coordinates.
This script bridges that gap by:
  1. Packing GBRG Bayer into 4 channels [R, Gr, Gb, B] at H/2 x W/2
  2. Rectifying each channel with half-resolution camera intrinsics
  3. Saving as compressed NPZ (key='bayer_rect', uint16, shape 512x960x4)

Usage:
    python rectify_bayer_pack.py [--n-vis 20] [--workers 4]
"""

import argparse
import csv
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tifffile

# ─── Paths ───────────────────────────────────────────────────
STF_DATA_ROOT = Path("/mnt/drive/3333_raw/seeing_through_fog")
CALIB_PATH    = STF_DATA_ROOT / "calib_cam_stereo_left.json"
RAW_DIR       = STF_DATA_ROOT / "cam_stereo_left"
LUT_DIR       = STF_DATA_ROOT / "cam_stereo_left_lut"
LIDAR_DIR     = STF_DATA_ROOT / "lidar_hdl64_last_stereo_left"
MANIFEST_DIR  = Path("/home/caq/6666_raw/seeingthroughfog/manifests")

# Output on the large drive to avoid filling NVMe
OUTPUT_ROOT = Path("/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect")
NPZ_DIR     = OUTPUT_ROOT / "npz"
VIS_DIR     = OUTPUT_ROOT / "vis"

FULL_H, FULL_W = 1024, 1920
HALF_H, HALF_W = 512, 960
COMPANDED_MAX  = 3967


# ─── Calibration ─────────────────────────────────────────────
def load_calibration(path):
    with open(path) as f:
        c = json.load(f)
    K = np.array(c["K"], dtype=np.float64).reshape(3, 3)
    D = np.array(c["D"], dtype=np.float64).reshape(-1)
    R = np.array(c["R"], dtype=np.float64).reshape(3, 3)
    P = np.array(c["P"], dtype=np.float64).reshape(3, 4)
    return K, D, R, P


def build_remap_half(K, D, R, P):
    """Build remap tables for half-resolution Bayer channels.

    Bayer packing subsamples every 2nd pixel in each direction,
    so the effective intrinsics are halved: fx/2, fy/2, cx/2, cy/2.
    Distortion coefficients D are unitless and stay unchanged.
    R (rectification rotation) is unchanged.
    P (new camera matrix for output) is also halved.
    """
    K_half = K.copy()
    K_half[0, 0] /= 2   # fx
    K_half[1, 1] /= 2   # fy
    K_half[0, 2] /= 2   # cx
    K_half[1, 2] /= 2   # cy

    P_half = P.copy()
    P_half[0, 0] /= 2   # fx
    P_half[1, 1] /= 2   # fy
    P_half[0, 2] /= 2   # cx
    P_half[1, 2] /= 2   # cy

    mx, my = cv2.initUndistortRectifyMap(
        K_half, D, R, P_half, (HALF_W, HALF_H), cv2.CV_32FC1
    )
    return mx, my


# ─── Bayer Pack / Rectify ────────────────────────────────────
def bayer_pack_gbrg(raw):
    """GBRG Bayer → 4 channels [R, Gr, Gb, B] at H/2 x W/2."""
    return np.stack([
        raw[1::2, 0::2],   # R
        raw[0::2, 0::2],   # Gr
        raw[1::2, 1::2],   # Gb
        raw[0::2, 1::2],   # B
    ], axis=-1)


def rectify_packed(packed, mx, my):
    """Rectify each of the 4 Bayer channels independently."""
    out = np.empty_like(packed)
    for ch in range(4):
        out[..., ch] = cv2.remap(
            packed[..., ch], mx, my,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return out


# ─── Visualization helpers ───────────────────────────────────
def packed_to_pseudo_rgb_u8(packed, max_val=COMPANDED_MAX):
    """4-ch [R,Gr,Gb,B] → pseudo-RGB uint8 for visualization."""
    R  = packed[..., 0].astype(np.float32)
    Gr = packed[..., 1].astype(np.float32)
    Gb = packed[..., 2].astype(np.float32)
    B  = packed[..., 3].astype(np.float32)
    G  = (Gr + Gb) / 2.0
    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb / max_val * 255, 0, 255).astype(np.uint8)


def overlay_lidar(bgr_img, depth, scale_x=1.0, scale_y=1.0, radius=2):
    """Draw LiDAR points (colored by depth) on a BGR image."""
    vis = bgr_img.copy()
    ys, xs = np.where((depth > 0) & np.isfinite(depth))
    if len(ys) == 0:
        return vis

    ds = depth[ys, xs]
    d_lo, d_hi = 1.0, 80.0
    d_norm = np.clip((ds - d_lo) / (d_hi - d_lo) * 255, 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(d_norm.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)

    cxs = (xs * scale_x).astype(np.int32)
    cys = (ys * scale_y).astype(np.int32)
    h, w = vis.shape[:2]
    mask = (cxs >= 0) & (cxs < w) & (cys >= 0) & (cys < h)

    for i in np.where(mask)[0]:
        cv2.circle(vis, (int(cxs[i]), int(cys[i])), radius, colors[i].tolist(), -1)
    return vis


def make_verification_panel(stem, rect_packed, lut_bgr, depth, out_path):
    """Side-by-side: [LUT + LiDAR | Rectified-RAW pseudo-RGB + LiDAR]."""
    # Rectified RAW pseudo-RGB → upscale to full-res for fair comparison
    raw_rgb = packed_to_pseudo_rgb_u8(rect_packed)
    raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
    raw_bgr_full = cv2.resize(raw_bgr, (FULL_W, FULL_H), interpolation=cv2.INTER_LINEAR)

    # Overlay LiDAR on both (LiDAR is at full-res rectified coords)
    lut_ov  = overlay_lidar(lut_bgr, depth, scale_x=1.0, scale_y=1.0, radius=2)
    raw_ov  = overlay_lidar(raw_bgr_full, depth, scale_x=1.0, scale_y=1.0, radius=2)

    combined = np.hstack([lut_ov, raw_ov])

    cv2.putText(combined, "LUT (rectified reference)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(combined, "Rectified RAW pseudo-RGB", (FULL_W + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(combined, stem, (10, FULL_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(str(out_path), combined)

    # Also generate a zoomed crop around the densest LiDAR region
    ys, xs = np.where((depth > 0) & np.isfinite(depth))
    if len(ys) > 50:
        cy_med = int(np.median(ys))
        cx_med = int(np.median(xs))
        crop_h, crop_w = 200, 400
        y1 = max(0, cy_med - crop_h // 2)
        y2 = min(FULL_H, y1 + crop_h)
        x1 = max(0, cx_med - crop_w // 2)
        x2 = min(FULL_W, x1 + crop_w)

        crop_lut = lut_ov[y1:y2, x1:x2]
        crop_raw = raw_ov[y1:y2, x1:x2]
        crop_combined = np.hstack([crop_lut, crop_raw])

        # Scale up for visibility
        crop_combined = cv2.resize(crop_combined, None, fx=2, fy=2,
                                   interpolation=cv2.INTER_NEAREST)
        crop_path = out_path.parent / (out_path.stem + "_crop" + out_path.suffix)
        cv2.imwrite(str(crop_path), crop_combined)


# ─── Manifest ────────────────────────────────────────────────
def load_all_stems():
    stems = []
    seen = set()
    for split in ["train", "val", "test"]:
        p = MANIFEST_DIR / f"stf_raw_depth_v1_{split}.csv"
        if not p.exists():
            print(f"  [WARN] manifest not found: {p}")
            continue
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                s = row["filename_stem"]
                if s not in seen:
                    stems.append(s)
                    seen.add(s)
    return stems


# ─── Main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-vis", type=int, default=20,
                        help="Number of verification visualizations")
    args = parser.parse_args()

    print("=" * 64)
    print("  Bayer Pack + Half-Resolution Rectification")
    print("=" * 64)

    # 1. Calibration
    K, D, R, P = load_calibration(CALIB_PATH)
    print(f"  K diag: fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    print(f"  P diag: fx={P[0,0]:.2f}  fy={P[1,1]:.2f}  cx={P[0,2]:.2f}  cy={P[1,2]:.2f}")
    print(f"  D: {D.tolist()}")

    mx, my = build_remap_half(K, D, R, P)
    print(f"  Half-res remap: mapx {mx.shape}  mapy {my.shape}")

    # 2. Manifest
    stems = load_all_stems()
    n = len(stems)
    print(f"  Stems loaded: {n}")
    if n == 0:
        print("  ERROR: no stems found. Exiting.")
        sys.exit(1)

    # 3. Output dirs
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Decide vis samples (evenly spaced)
    n_vis = min(args.n_vis, n)
    if n_vis > 0:
        vis_indices = set(np.linspace(0, n - 1, n_vis, dtype=int).tolist())
    else:
        vis_indices = set()
    print(f"  Vis samples: {len(vis_indices)}")
    print("-" * 64)

    # 5. Process loop
    stats = defaultdict(int)
    t0 = time.time()

    for idx, stem in enumerate(stems):
        raw_path = RAW_DIR / f"{stem}.tiff"
        if not raw_path.exists():
            stats["missing_raw"] += 1
            continue

        try:
            # Read RAW
            raw = tifffile.imread(str(raw_path))
            if raw.shape != (FULL_H, FULL_W):
                stats["bad_shape"] += 1
                print(f"  [SKIP] {stem}: shape {raw.shape}")
                continue

            # Pack + rectify
            packed = bayer_pack_gbrg(raw)          # (512, 960, 4) uint16
            rect   = rectify_packed(packed, mx, my)  # (512, 960, 4) uint16

            # Save
            out_npz = NPZ_DIR / f"{stem}.npz"
            np.savez_compressed(str(out_npz), bayer_rect=rect)
            stats["ok"] += 1

            # Verification visualization
            if idx in vis_indices:
                lut_path   = LUT_DIR / f"{stem}.png"
                lidar_path = LIDAR_DIR / f"{stem}.npz"
                if lut_path.exists() and lidar_path.exists():
                    lut_bgr = cv2.imread(str(lut_path), cv2.IMREAD_COLOR)
                    with np.load(str(lidar_path), allow_pickle=False) as d:
                        depth = d["arr_0"].astype(np.float32)
                    vis_path = VIS_DIR / f"{stem}_align.png"
                    make_verification_panel(stem, rect, lut_bgr, depth, vis_path)
                    stats["vis"] += 1

        except Exception as e:
            stats["error"] += 1
            if stats["error"] <= 10:
                print(f"  [ERROR] {stem}: {e}")
                traceback.print_exc()

        # Progress
        if (idx + 1) % 500 == 0 or idx == n - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (n - idx - 1) / max(rate, 0.01)
            print(f"  [{idx+1:>5}/{n}]  ok={stats['ok']}  err={stats['error']}  "
                  f"vis={stats['vis']}  | {rate:.1f} samp/s  ETA {eta/60:.1f} min")

    # 6. Summary
    elapsed = time.time() - t0
    summary = {
        "total_stems": n,
        "ok": stats["ok"],
        "missing_raw": stats["missing_raw"],
        "bad_shape": stats["bad_shape"],
        "errors": stats["error"],
        "visualizations": stats["vis"],
        "elapsed_sec": round(elapsed, 1),
        "output_npz_dir": str(NPZ_DIR),
        "output_vis_dir": str(VIS_DIR),
    }

    print("=" * 64)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 64)

    # Save summary
    import json as _json
    with open(OUTPUT_ROOT / "rectify_summary.json", "w") as f:
        _json.dump(summary, f, indent=2)
    print(f"  Summary saved to {OUTPUT_ROOT / 'rectify_summary.json'}")


if __name__ == "__main__":
    main()
