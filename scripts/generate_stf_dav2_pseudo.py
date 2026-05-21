#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import subprocess
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

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

DEPTH_VALUE_UNITS = {
    "value": "relative_inverse_depth_from_dav2",
    "direction": "larger_is_closer",
    "compatibility_note": (
        "DepthAnythingV2 infer_image() returns affine/scale-free relative inverse "
        "depth or disparity-like values; shapes and file schema match STF DAV2-L "
        "pseudo labels."
    ),
}

MANIFEST_FIELDS = [
    "sample_name",
    "split",
    "rgb_path",
    "sparse_depth_path",
    "pseudo_depth_npy",
    "pseudo_vis_png",
]


@dataclass(frozen=True)
class SampleSpec:
    sample_name: str
    split: str
    rgb_path: str
    sparse_depth_path: str
    pseudo_depth_npy: str
    pseudo_vis_png: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DAV2 pseudo depth labels for STF RGB-LUT manifest rows."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Source STF RGB-LUT manifest CSV, usually the old DAV2-L manifest.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root for flat npy/png files and metadata.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Depth Anything V2 checkpoint path.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vits",
        choices=sorted(MODEL_CONFIGS.keys()),
        help="DAV2 encoder variant.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="DAV2 preprocessing size used by infer_image().",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device, e.g. cuda, cuda:0, cpu, or mps.",
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
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing complete npy/png output pairs.",
    )
    return parser.parse_args()


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def write_json(path: Path, payload):
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def atomic_save_npy(path: Path, array: np.ndarray):
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    np.save(tmp_path, array.astype(np.float32, copy=False))
    tmp_path.replace(path)


def atomic_save_png(path: Path, image_bgr: np.ndarray):
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    ok = cv2.imwrite(str(tmp_path), image_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise RuntimeError(f"Failed to write PNG: {tmp_path}")
    tmp_path.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(args, cwd: Path | None = None) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip(), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def checkpoint_metadata(checkpoint: Path) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    metadata = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_filename": checkpoint.name if checkpoint.exists() else None,
        "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.exists() else None,
        "checkpoint_sha256": None,
        "checkpoint_sha256_error": None,
        "project_root": str(PROJECT_ROOT),
        "project_repo_commit": None,
        "project_repo_commit_error": None,
    }

    if checkpoint.exists():
        try:
            metadata["checkpoint_sha256"] = sha256_file(checkpoint)
        except Exception as exc:  # noqa: BLE001
            metadata["checkpoint_sha256_error"] = f"{type(exc).__name__}: {exc}"
    else:
        metadata["checkpoint_sha256_error"] = f"Missing checkpoint file: {checkpoint}"

    commit, commit_error = run_command(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
    metadata["project_repo_commit"] = commit
    metadata["project_repo_commit_error"] = commit_error
    return metadata


def validate_manifest_header(fieldnames):
    missing = [name for name in MANIFEST_FIELDS if name not in fieldnames]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")


def collect_samples(args) -> tuple[list[SampleSpec], int]:
    manifest = args.manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    samples = []
    seen_names = set()
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_manifest_header(reader.fieldnames or [])
        for row in reader:
            sample_name = row["sample_name"]
            if sample_name in seen_names:
                raise ValueError(f"Duplicate sample_name in manifest: {sample_name}")
            seen_names.add(sample_name)
            samples.append(
                SampleSpec(
                    sample_name=sample_name,
                    split=row["split"],
                    rgb_path=row["rgb_path"],
                    sparse_depth_path=row["sparse_depth_path"],
                    pseudo_depth_npy=str((output_root / f"{sample_name}.npy").resolve()),
                    pseudo_vis_png=str((output_root / f"{sample_name}.png").resolve()),
                )
            )

    source_count = len(samples)
    if args.max_samples is not None:
        samples = samples[: max(int(args.max_samples), 0)]
    return samples, source_count


def write_outputs_manifest(output_root: Path, samples: list[SampleSpec]) -> tuple[Path, Path]:
    manifest_path = output_root / "stf_rgb_lut_manifest_6216.csv"
    inputs_path = output_root / "stf_rgb_lut_inputs_6216.txt"

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))

    with inputs_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(f"{sample.rgb_path}\n")

    return manifest_path, inputs_path


def colorize_depth(
    depth: np.ndarray,
    *,
    cmap_name: str,
    vmin_pct: float,
    vmax_pct: float,
    gamma: float,
) -> np.ndarray:
    depth = depth.astype(np.float32, copy=False)
    finite = np.isfinite(depth)
    valid = depth[finite]
    if valid.size <= 10:
        norm = np.zeros_like(depth, dtype=np.float32)
    else:
        vmin = float(np.percentile(valid, vmin_pct))
        vmax = float(np.percentile(valid, vmax_pct))
        if not np.isfinite(vmin):
            vmin = float(np.nanmin(valid))
        if not np.isfinite(vmax):
            vmax = float(np.nanmax(valid))
        if vmax <= vmin + 1e-8:
            norm = np.zeros_like(depth, dtype=np.float32)
        else:
            norm = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0).astype(
                np.float32,
                copy=False,
            )

    if gamma > 0.0 and abs(gamma - 1.0) > 1e-6:
        norm = np.power(norm, gamma)
    norm[~finite] = 0.0

    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    image_rgb = (cmap(norm)[:, :, :3] * 255.0).astype(np.uint8)
    return image_rgb[:, :, ::-1]


def configure_torch(device: torch.device):
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is false.")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def cuda_peak_allocated(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def cuda_peak_reserved(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_reserved(device))


def current_allocated(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.memory_allocated(device))


def build_model(args, device: torch.device):
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing DAV2 checkpoint: {checkpoint}")

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    state_dict = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def build_config(args, samples, source_count, manifest_path, inputs_path, metadata):
    return {
        "source_manifest": str(args.manifest.expanduser().resolve()),
        "source_sample_count": source_count,
        "output_root": str(args.output_root.expanduser().resolve()),
        "manifest_path": str(manifest_path),
        "inputs_path": str(inputs_path),
        "sample_count": len(samples),
        "max_samples": args.max_samples,
        "overwrite": bool(args.overwrite),
        "device": args.device,
        "encoder": args.encoder,
        "model_config": MODEL_CONFIGS[args.encoder],
        "input_size": args.input_size,
        "target_resolution_policy": "original_image_resolution",
        "resize_interpolation": "cv2.INTER_LINEAR",
        "vis_cmap": args.vis_cmap,
        "vis_vmin_pct": args.vis_vmin_pct,
        "vis_vmax_pct": args.vis_vmax_pct,
        "vis_gamma": args.vis_gamma,
        "depth_value_units": DEPTH_VALUE_UNITS,
        **metadata,
    }


def print_progress(
    *,
    index: int,
    total: int,
    sample: SampleSpec,
    status: str,
    generated: int,
    skipped: int,
    failed: int,
    start_time: float,
    device: torch.device,
):
    elapsed = max(time.time() - start_time, 1e-9)
    processed = max(index, 1)
    imgs_per_sec = processed / elapsed
    remaining = max(total - processed, 0)
    eta_sec = remaining / imgs_per_sec if imgs_per_sec > 0 else 0.0
    print(
        f"[progress] {index}/{total} split={sample.split} sample={sample.sample_name} "
        f"status={status} generated={generated} skipped={skipped} failed={failed} "
        f"imgs_per_sec={imgs_per_sec:.4f} eta_sec={eta_sec:.1f} eta_hms={hms(eta_sec)} "
        f"peak_cuda_memory_allocated={cuda_peak_allocated(device)} "
        f"peak_cuda_memory_reserved={cuda_peak_reserved(device)}"
    )
    sys.stdout.flush()


def main() -> int:
    args = parse_args()
    if args.input_size <= 0:
        raise ValueError("--input-size must be > 0")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    samples, source_count = collect_samples(args)
    if not samples:
        raise ValueError("No STF manifest rows selected.")

    manifest_path, inputs_path = write_outputs_manifest(output_root, samples)
    metadata = checkpoint_metadata(args.checkpoint)
    run_config = build_config(args, samples, source_count, manifest_path, inputs_path, metadata)
    write_json(output_root / "run_config.json", run_config)

    device = torch.device(args.device)
    configure_torch(device)

    print(
        f"[setup] device={device} encoder={args.encoder} checkpoint={run_config['checkpoint_path']} "
        f"input_size={args.input_size} samples={len(samples)}"
    )
    print(f"[setup] manifest={manifest_path}")
    print(f"[setup] inputs={inputs_path}")
    sys.stdout.flush()

    model = build_model(args, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    generated = 0
    skipped = 0
    failed_samples = []

    for index, sample in enumerate(samples, start=1):
        out_npy = Path(sample.pseudo_depth_npy)
        out_png = Path(sample.pseudo_vis_png)
        ensure_parent(out_npy)
        ensure_parent(out_png)

        if not args.overwrite and out_npy.is_file() and out_png.is_file():
            skipped += 1
            if index == 1 or index % 25 == 0 or index == len(samples):
                print_progress(
                    index=index,
                    total=len(samples),
                    sample=sample,
                    status="skip",
                    generated=generated,
                    skipped=skipped,
                    failed=len(failed_samples),
                    start_time=start_time,
                    device=device,
                )
            continue

        try:
            bgr = cv2.imread(sample.rgb_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Failed to read RGB image: {sample.rgb_path}")
            orig_h, orig_w = int(bgr.shape[0]), int(bgr.shape[1])

            with torch.inference_mode():
                depth = model.infer_image(bgr, args.input_size).astype(np.float32, copy=False)
            if depth.shape != (orig_h, orig_w):
                depth = cv2.resize(
                    depth,
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.float32, copy=False)

            vis = colorize_depth(
                depth,
                cmap_name=args.vis_cmap,
                vmin_pct=args.vis_vmin_pct,
                vmax_pct=args.vis_vmax_pct,
                gamma=args.vis_gamma,
            )
            atomic_save_npy(out_npy, depth)
            atomic_save_png(out_png, vis)
            generated += 1

            if index == 1 or index % 25 == 0 or index == len(samples):
                print_progress(
                    index=index,
                    total=len(samples),
                    sample=sample,
                    status="ok",
                    generated=generated,
                    skipped=skipped,
                    failed=len(failed_samples),
                    start_time=start_time,
                    device=device,
                )
        except Exception as exc:  # noqa: BLE001
            failed_samples.append(
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
        "status": "completed" if not failed_samples else "completed_with_failures",
        "generated": generated,
        "skipped": skipped,
        "failed": len(failed_samples),
        "sample_count": len(samples),
        "source_sample_count": source_count,
        "elapsed_sec": elapsed,
        "imgs_per_sec": (len(samples) / elapsed) if elapsed > 0 else None,
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "inputs_path": str(inputs_path),
        "encoder": args.encoder,
        "input_size": args.input_size,
        "checkpoint_path": run_config["checkpoint_path"],
        "checkpoint_filename": run_config["checkpoint_filename"],
        "checkpoint_size_bytes": run_config["checkpoint_size_bytes"],
        "checkpoint_sha256": run_config["checkpoint_sha256"],
        "checkpoint_sha256_error": run_config["checkpoint_sha256_error"],
        "project_repo_commit": run_config["project_repo_commit"],
        "project_repo_commit_error": run_config["project_repo_commit_error"],
        "depth_value_units": DEPTH_VALUE_UNITS,
        "vis_cmap": args.vis_cmap,
        "vis_vmin_pct": args.vis_vmin_pct,
        "vis_vmax_pct": args.vis_vmax_pct,
        "vis_gamma": args.vis_gamma,
        "peak_cuda_memory_allocated_bytes": cuda_peak_allocated(device),
        "peak_cuda_memory_reserved_bytes": cuda_peak_reserved(device),
        "final_cuda_memory_allocated_bytes": current_allocated(device),
    }
    write_json(output_root / "run_summary.json", summary)

    failed_path = output_root / "failed_samples.json"
    if failed_samples:
        write_json(failed_path, failed_samples)
    elif failed_path.exists():
        failed_path.unlink()

    print(
        f"[done] status={summary['status']} generated={generated} skipped={skipped} "
        f"failed={len(failed_samples)} elapsed_sec={elapsed:.1f} "
        f"peak_cuda_memory_allocated={summary['peak_cuda_memory_allocated_bytes']}"
    )
    sys.stdout.flush()
    return 0 if not failed_samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
