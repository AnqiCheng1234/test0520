#!/usr/bin/env python3
import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

SPLIT_LAYOUTS = {
    "00Train": {
        "rgb_dir": Path("00Train-sdr_rgb") / "00Train",
        "rggb_dir": Path("00Train-rggb") / "00Train",
    },
    "01Valid": {
        "rgb_dir": Path("01Valid-sdr_rgb") / "01Valid",
        "rggb_dir": Path("01Valid-rggb") / "01Valid",
    },
}


@dataclass(frozen=True)
class SampleSpec:
    split: str
    sample_name: str
    rgb_path: str
    rggb_path: str
    output_npy: str
    output_png: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DAV2 relative-depth pseudo labels for selected LOD samples."
    )
    parser.add_argument(
        "--lod-root",
        type=Path,
        default=Path("/mnt/drive/3333_raw/LOD"),
        help="LOD dataset root containing 00Train/01Valid rgb and rggb folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/drive/3333_raw/LOD/pseudo_depth_dav2_day_rel_1440x928"),
        help="Output root. Split subfolders and manifest files will be created here.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth"),
        help="DAV2 checkpoint path.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["00Train", "01Valid"],
        choices=sorted(SPLIT_LAYOUTS.keys()),
        help="LOD splits to process.",
    )
    parser.add_argument(
        "--sample-prefixes",
        nargs="+",
        default=["day"],
        help="Filename prefixes to process, e.g. day night.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vitl",
        choices=sorted(MODEL_CONFIGS.keys()),
        help="DAV2 encoder variant.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=700,
        help="DAV2 preprocessing size used by infer_image().",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=928,
        help="Saved pseudo-label height aligned to packed raw/rggb.",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=1440,
        help="Saved pseudo-label width aligned to packed raw/rggb.",
    )
    parser.add_argument(
        "--vis-cmap",
        type=str,
        default="magma_r",
        help="Matplotlib colormap for visualization PNGs.",
    )
    parser.add_argument(
        "--vis-vmin-pct",
        type=float,
        default=1.0,
        help="Lower percentile for robust visualization normalization.",
    )
    parser.add_argument(
        "--vis-vmax-pct",
        type=float,
        default=99.0,
        help="Upper percentile for robust visualization normalization.",
    )
    parser.add_argument(
        "--vis-gamma",
        type=float,
        default=1.0,
        help="Gamma applied after robust normalization for visualization only.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing npy/png outputs instead of skipping them.",
    )
    return parser.parse_args()


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def colorize_depth(
    depth: np.ndarray,
    *,
    cmap_name: str,
    vmin_pct: float,
    vmax_pct: float,
    gamma: float,
) -> np.ndarray:
    depth = depth.astype(np.float32, copy=False)
    vmin = float(np.percentile(depth, vmin_pct))
    vmax = float(np.percentile(depth, vmax_pct))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(depth))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(depth))

    if vmax <= vmin + 1e-8:
        norm = np.zeros_like(depth, dtype=np.float32)
    else:
        norm = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)

    if gamma > 0.0 and abs(gamma - 1.0) > 1e-6:
        norm = np.power(norm, gamma)

    depth_u8 = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgb = (cmap(depth_u8)[:, :, :3] * 255.0).astype(np.uint8)
    return rgb[:, :, ::-1]


def build_model(args, device: str):
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing DAV2 checkpoint: {checkpoint}")

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    state_dict = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


def collect_samples(args):
    lod_root = args.lod_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    samples = []
    for split in args.splits:
        layout = SPLIT_LAYOUTS[split]
        rgb_dir = lod_root / layout["rgb_dir"]
        rggb_dir = lod_root / layout["rggb_dir"]
        if not rgb_dir.is_dir():
            raise FileNotFoundError(f"Missing RGB dir for {split}: {rgb_dir}")
        if not rggb_dir.is_dir():
            raise FileNotFoundError(f"Missing RGGB dir for {split}: {rggb_dir}")

        split_output_dir = output_root / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        rgb_paths = []
        seen = set()
        for prefix in args.sample_prefixes:
            for rgb_path in sorted(rgb_dir.glob(f"{prefix}-*.jpg")):
                if rgb_path.name in seen:
                    continue
                seen.add(rgb_path.name)
                rgb_paths.append(rgb_path)
        rgb_paths.sort()
        for rgb_path in rgb_paths:
            sample_name = rgb_path.stem
            rggb_path = rggb_dir / f"{sample_name}.npy"
            if not rggb_path.is_file():
                raise FileNotFoundError(f"Missing corresponding RGGB file: {rggb_path}")
            samples.append(
                SampleSpec(
                    split=split,
                    sample_name=sample_name,
                    rgb_path=str(rgb_path.resolve()),
                    rggb_path=str(rggb_path.resolve()),
                    output_npy=str((split_output_dir / f"{sample_name}.npy").resolve()),
                    output_png=str((split_output_dir / f"{sample_name}.png").resolve()),
                )
            )

    if args.max_samples is not None:
        samples = samples[: max(int(args.max_samples), 0)]
    return samples


def write_manifest(output_root: Path, samples):
    prefix_tag = "_".join(argsafe_prefixes_from_samples(samples))
    manifest_path = output_root / f"lod_{prefix_tag}_dav2_rel_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "sample_name",
                "rgb_path",
                "rggb_path",
                "output_npy",
                "output_png",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))
    return manifest_path


def argsafe_prefixes_from_samples(samples):
    prefixes = []
    for sample in samples:
        prefix = sample.sample_name.split("-", 1)[0]
        if prefix not in prefixes:
            prefixes.append(prefix)
    return prefixes or ["samples"]


def write_json(path: Path, payload):
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def atomic_save_npy(path: Path, array: np.ndarray):
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    np.save(tmp_path, array.astype(np.float32, copy=False))
    tmp_path.replace(path)


def atomic_save_png(path: Path, image: np.ndarray):
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    ok = cv2.imwrite(str(tmp_path), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise RuntimeError(f"Failed to write PNG: {tmp_path}")
    tmp_path.replace(path)


def main():
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    samples = collect_samples(args)
    if not samples:
        raise ValueError("No daytime LOD samples found.")

    manifest_path = write_manifest(output_root, samples)
    write_json(
        output_root / "run_config.json",
        {
            "lod_root": str(args.lod_root.expanduser().resolve()),
            "output_root": str(output_root),
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "splits": list(args.splits),
            "sample_prefixes": list(args.sample_prefixes),
            "encoder": args.encoder,
            "input_size": args.input_size,
            "target_hw": [args.target_height, args.target_width],
            "vis_cmap": args.vis_cmap,
            "vis_vmin_pct": args.vis_vmin_pct,
            "vis_vmax_pct": args.vis_vmax_pct,
            "vis_gamma": args.vis_gamma,
            "max_samples": args.max_samples,
            "overwrite": bool(args.overwrite),
            "device": device,
            "manifest_path": str(manifest_path),
            "sample_count": len(samples),
        },
    )

    print(
        f"[setup] device={device} encoder={args.encoder} prefixes={','.join(args.sample_prefixes)} input_size={args.input_size} "
        f"target_hw=({args.target_height}, {args.target_width}) samples={len(samples)}"
    )
    print(f"[setup] manifest={manifest_path}")
    sys.stdout.flush()

    model = build_model(args, device)

    start_time = time.time()
    generated = 0
    skipped = 0
    failed = []

    for index, sample in enumerate(samples, start=1):
        out_npy = Path(sample.output_npy)
        out_png = Path(sample.output_png)
        ensure_parent(out_npy)
        ensure_parent(out_png)

        if not args.overwrite and out_npy.is_file() and out_png.is_file():
            skipped += 1
            if index == 1 or index % 100 == 0 or index == len(samples):
                print(
                    f"[progress] {index}/{len(samples)} split={sample.split} "
                    f"sample={sample.sample_name} status=skip generated={generated} skipped={skipped}"
                )
                sys.stdout.flush()
            continue

        try:
            rgb = cv2.imread(sample.rgb_path, cv2.IMREAD_COLOR)
            if rgb is None:
                raise RuntimeError(f"Failed to read RGB image: {sample.rgb_path}")

            depth_native = model.infer_image(rgb, args.input_size).astype(np.float32, copy=False)
            depth_raw = cv2.resize(
                depth_native,
                (args.target_width, args.target_height),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32, copy=False)
            vis = colorize_depth(
                depth_raw,
                cmap_name=args.vis_cmap,
                vmin_pct=args.vis_vmin_pct,
                vmax_pct=args.vis_vmax_pct,
                gamma=args.vis_gamma,
            )

            atomic_save_npy(out_npy, depth_raw)
            atomic_save_png(out_png, vis)
            generated += 1

            if index == 1 or index % 25 == 0 or index == len(samples):
                elapsed = time.time() - start_time
                print(
                    f"[progress] {index}/{len(samples)} split={sample.split} "
                    f"sample={sample.sample_name} status=ok generated={generated} skipped={skipped} "
                    f"elapsed_sec={elapsed:.1f}"
                )
                sys.stdout.flush()
        except Exception as exc:  # noqa: BLE001
            failed.append(
                {
                    "index": index,
                    "split": sample.split,
                    "sample_name": sample.sample_name,
                    "rgb_path": sample.rgb_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"[error] {index}/{len(samples)} split={sample.split} sample={sample.sample_name} "
                f"error={type(exc).__name__}: {exc}"
            )
            sys.stdout.flush()

    elapsed = time.time() - start_time
    summary = {
        "status": "completed" if not failed else "completed_with_failures",
        "generated": generated,
        "skipped": skipped,
        "failed": len(failed),
        "sample_count": len(samples),
        "elapsed_sec": elapsed,
        "device": device,
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
    }
    write_json(output_root / "run_summary.json", summary)
    failed_path = output_root / "failed_samples.json"
    if failed:
        write_json(failed_path, failed)
    elif failed_path.exists():
        failed_path.unlink()

    print(
        f"[done] status={summary['status']} generated={generated} skipped={skipped} "
        f"failed={len(failed)} elapsed_sec={elapsed:.1f}"
    )


if __name__ == "__main__":
    main()
