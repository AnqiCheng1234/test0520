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
DA3_REPO_ROOT = Path("/home/caq/dav3")
DA3_SRC = DA3_REPO_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DA3_SRC) not in sys.path:
    sys.path.insert(0, str(DA3_SRC))

from depth_anything_3.api import DepthAnything3  # noqa: E402


DEPTH_VALUE_UNITS = {
    "value": "affine_invariant_depth_from_da3mono",
    "direction": "larger_is_farther",
    "compatibility_note": (
        "Shape/dtype/file schema match DAV2 pseudo labels, but values are not DAV2 "
        "relative inverse depth/disparity."
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
        description="Generate DA3MONO-LARGE pseudo depth labels for STF RGB-LUT manifest rows."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Source DAV2 STF RGB-LUT manifest CSV.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root for flat npy/png files and metadata.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/home/caq/dav3/DA3MONO-LARGE"),
        help="Local DA3MONO model directory passed to DepthAnything3.from_pretrained().",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=1008,
        help="DA3 processing resolution.",
    )
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
        choices=[
            "upper_bound_resize",
            "upper_bound_crop",
            "lower_bound_resize",
            "lower_bound_crop",
        ],
        help="DA3 process_res_method.",
    )
    parser.add_argument(
        "--align-to-input-ext-scale",
        dest="align_to_input_ext_scale",
        action="store_true",
        default=True,
        help="Pass align_to_input_ext_scale=True to DA3 inference.",
    )
    parser.add_argument(
        "--no-align-to-input-ext-scale",
        dest="align_to_input_ext_scale",
        action="store_false",
        help="Pass align_to_input_ext_scale=False to DA3 inference.",
    )
    parser.add_argument(
        "--use-ray-pose",
        action="store_true",
        help="Pass use_ray_pose=True to DA3 inference.",
    )
    parser.add_argument(
        "--ref-view-strategy",
        type=str,
        default="saddle_balanced",
        help="DA3 ref_view_strategy.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of manifest rows per DA3 inference call. Default keeps images independent.",
    )
    parser.add_argument(
        "--vis-cmap",
        type=str,
        default="Spectral",
        help="Matplotlib colormap for depth-only visualization PNGs.",
    )
    parser.add_argument(
        "--vis-percentile",
        type=float,
        default=2.0,
        help="Percentile p for p/(100-p) visualization clipping.",
    )
    parser.add_argument(
        "--vis-inverse",
        action="store_true",
        default=False,
        help="Apply DA3 current visualization inverse-depth logic.",
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


def model_metadata(model_dir: Path) -> dict:
    model_dir = model_dir.expanduser().resolve()
    config_path = model_dir / "config.json"
    weight_path = model_dir / "model.safetensors"

    metadata = {
        "model_dir": str(model_dir),
        "model_identifier": model_dir.name,
        "model_config_path": str(config_path) if config_path.exists() else None,
        "model_config_sha256": None,
        "model_weight_path": str(weight_path) if weight_path.exists() else None,
        "model_weight_filename": weight_path.name if weight_path.exists() else None,
        "model_weight_size_bytes": weight_path.stat().st_size if weight_path.exists() else None,
        "model_weight_sha256": None,
        "model_weight_sha256_error": None,
        "da3_repo_root": str(DA3_REPO_ROOT),
        "da3_repo_commit": None,
        "da3_repo_commit_error": None,
    }

    if config_path.exists():
        metadata["model_config_sha256"] = sha256_file(config_path)
    else:
        metadata["model_config_sha256_error"] = f"Missing config file: {config_path}"

    if weight_path.exists():
        try:
            metadata["model_weight_sha256"] = sha256_file(weight_path)
        except Exception as exc:  # noqa: BLE001
            metadata["model_weight_sha256_error"] = f"{type(exc).__name__}: {exc}"
    else:
        metadata["model_weight_sha256_error"] = f"Missing model weight file: {weight_path}"

    commit, commit_error = run_command(["git", "rev-parse", "HEAD"], cwd=DA3_REPO_ROOT)
    metadata["da3_repo_commit"] = commit
    metadata["da3_repo_commit_error"] = commit_error
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


def visualize_depth_da3_current(
    depth: np.ndarray,
    *,
    cmap_name: str,
    percentile: float,
    vis_inverse: bool,
) -> np.ndarray:
    depth_vis = depth.astype(np.float32, copy=True)
    finite = np.isfinite(depth_vis)
    valid_mask = finite & (depth_vis > 0)

    if vis_inverse:
        depth_vis[valid_mask] = 1.0 / depth_vis[valid_mask]

    if int(valid_mask.sum()) <= 10:
        depth_min = 0.0
        depth_max = 0.0
    else:
        valid_values = depth_vis[valid_mask]
        depth_min = float(np.percentile(valid_values, percentile))
        depth_max = float(np.percentile(valid_values, 100.0 - percentile))

    if not np.isfinite(depth_min):
        depth_min = 0.0
    if not np.isfinite(depth_max):
        depth_max = depth_min
    if depth_min == depth_max:
        depth_min -= 1e-6
        depth_max += 1e-6

    normalized = ((depth_vis - depth_min) / (depth_max - depth_min)).clip(0.0, 1.0)
    if vis_inverse:
        normalized = 1.0 - normalized
    normalized[~finite] = 0.0

    cmap = matplotlib.colormaps[cmap_name]
    image_rgb = (cmap(normalized)[..., :3] * 255.0).astype(np.uint8)
    return image_rgb[:, :, ::-1]


def load_rgb_images(samples: list[SampleSpec]) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    images = []
    original_hw = []
    for sample in samples:
        bgr = cv2.imread(sample.rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read RGB image: {sample.rgb_path}")
        original_hw.append((int(bgr.shape[0]), int(bgr.shape[1])))
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return images, original_hw


def configure_torch(device: torch.device):
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is false.")
        torch.backends.cudnn.benchmark = False
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


def build_config(args, samples, source_count, manifest_path, inputs_path, metadata):
    vis_clip_percentiles = [args.vis_percentile, 100.0 - args.vis_percentile]
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
        "batch_size": args.batch_size,
        "process_res": args.process_res,
        "process_res_method": args.process_res_method,
        "align_to_input_ext_scale": bool(args.align_to_input_ext_scale),
        "use_ray_pose": bool(args.use_ray_pose),
        "ref_view_strategy": args.ref_view_strategy,
        "infer_gs": False,
        "target_resolution_policy": "original_image_resolution",
        "resize_interpolation": "cv2.INTER_LINEAR",
        "vis_cmap": args.vis_cmap,
        "vis_percentile": args.vis_percentile,
        "vis_inverse": bool(args.vis_inverse),
        "vis_clip_percentiles": vis_clip_percentiles,
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
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    samples, source_count = collect_samples(args)
    if not samples:
        raise ValueError("No STF manifest rows selected.")

    manifest_path, inputs_path = write_outputs_manifest(output_root, samples)
    metadata = model_metadata(args.model_dir)
    run_config = build_config(args, samples, source_count, manifest_path, inputs_path, metadata)
    write_json(output_root / "run_config.json", run_config)

    device = torch.device(args.device)
    configure_torch(device)

    print(
        f"[setup] device={device} model_dir={run_config['model_dir']} process_res={args.process_res} "
        f"process_res_method={args.process_res_method} batch_size={args.batch_size} samples={len(samples)}"
    )
    print(f"[setup] manifest={manifest_path}")
    print(f"[setup] inputs={inputs_path}")
    sys.stdout.flush()

    model = DepthAnything3.from_pretrained(str(args.model_dir.expanduser().resolve())).to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    generated = 0
    skipped = 0
    failed_samples = []

    for start in range(0, len(samples), args.batch_size):
        chunk = samples[start : start + args.batch_size]
        run_chunk = []
        for offset, sample in enumerate(chunk):
            index = start + offset + 1
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
            run_chunk.append((index, sample))

        if not run_chunk:
            continue

        try:
            images, original_hws = load_rgb_images([sample for _, sample in run_chunk])
            prediction = model.inference(
                images,
                align_to_input_ext_scale=args.align_to_input_ext_scale,
                infer_gs=False,
                use_ray_pose=args.use_ray_pose,
                ref_view_strategy=args.ref_view_strategy,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
            )
            depths = prediction.depth
            if depths.shape[0] != len(run_chunk):
                raise RuntimeError(
                    f"DA3 returned {depths.shape[0]} depth maps for {len(run_chunk)} inputs."
                )

            for depth_index, ((index, sample), (orig_h, orig_w)) in enumerate(
                zip(run_chunk, original_hws)
            ):
                raw_depth = depths[depth_index].astype(np.float32, copy=False)
                if raw_depth.shape != (orig_h, orig_w):
                    raw_depth = cv2.resize(
                        raw_depth,
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_LINEAR,
                    ).astype(np.float32, copy=False)
                vis = visualize_depth_da3_current(
                    raw_depth,
                    cmap_name=args.vis_cmap,
                    percentile=args.vis_percentile,
                    vis_inverse=args.vis_inverse,
                )
                atomic_save_npy(Path(sample.pseudo_depth_npy), raw_depth)
                atomic_save_png(Path(sample.pseudo_vis_png), vis)
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
            for index, sample in run_chunk:
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
        "process_res": args.process_res,
        "process_res_method": args.process_res_method,
        "align_to_input_ext_scale": bool(args.align_to_input_ext_scale),
        "use_ray_pose": bool(args.use_ray_pose),
        "ref_view_strategy": args.ref_view_strategy,
        "model_dir": run_config["model_dir"],
        "model_identifier": run_config["model_identifier"],
        "da3_repo_commit": run_config["da3_repo_commit"],
        "model_config_sha256": run_config["model_config_sha256"],
        "model_weight_filename": run_config["model_weight_filename"],
        "model_weight_size_bytes": run_config["model_weight_size_bytes"],
        "model_weight_sha256": run_config["model_weight_sha256"],
        "model_weight_sha256_error": run_config["model_weight_sha256_error"],
        "depth_value_units": DEPTH_VALUE_UNITS,
        "vis_cmap": args.vis_cmap,
        "vis_percentile": args.vis_percentile,
        "vis_inverse": bool(args.vis_inverse),
        "vis_clip_percentiles": [args.vis_percentile, 100.0 - args.vis_percentile],
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
