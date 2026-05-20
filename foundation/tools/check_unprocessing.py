from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundation.engine.transforms import build_unprocessing_transform_from_preset, packed_bayer_to_base_rgb


DEFAULT_VKITTI_TRAIN_LIST = PROJECT_ROOT / "finetune_stf" / "dataset" / "splits" / "vkitti2" / "train.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "foundation" / "debug" / "unprocessing_sanity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity checks for the Phase 1 unprocessing pipeline.")
    parser.add_argument("--vkitti-train-list", default=str(DEFAULT_VKITTI_TRAIN_LIST))
    parser.add_argument("--roundtrip-index", type=int, default=0)
    parser.add_argument("--num-stats-images", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_filelist(filelist_path: Path) -> List[Tuple[Path, Path]]:
    with filelist_path.open("r", encoding="utf-8") as f:
        pairs = [line.strip().split() for line in f if line.strip()]
    if not pairs:
        raise ValueError(f"No samples found in {filelist_path}")
    return [(Path(image_path), Path(depth_path)) for image_path, depth_path in pairs]


def load_rgb_tensor(image_path: Path, device: torch.device) -> torch.Tensor:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = tensor.view(height, width, channels).permute(2, 0, 1).to(device=device, dtype=torch.float32) / 255.0
    return tensor


def tensor_to_uint8_image(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().clamp(0.0, 1.0)
    if image.ndim == 3:
        image = image.permute(1, 2, 0)
    image = image.mul(255.0).add(0.5).to(dtype=torch.uint8).contiguous()
    return np.asarray(image.tolist(), dtype=np.uint8)


def save_rgb(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_np = tensor_to_uint8_image(image)
    cv2.imwrite(str(path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))


def save_triptych(path: Path, images: Iterable[torch.Tensor]) -> None:
    image_np = [tensor_to_uint8_image(image) for image in images]
    triptych = np.concatenate(image_np, axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(triptych, cv2.COLOR_RGB2BGR))


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def compute_roundtrip_metrics(reference: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((reference - reconstructed) ** 2).item()
    mae = torch.mean(torch.abs(reference - reconstructed)).item()
    psnr = 99.0 if mse <= 1e-12 else -10.0 * math.log10(mse)
    return {"mse": mse, "mae": mae, "psnr": psnr}


def main() -> None:
    args = parse_args()
    filelist_path = Path(args.vkitti_train_list).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    generator = torch.Generator(device=device.type if device.type != "cpu" else "cpu")
    generator.manual_seed(args.seed)

    samples = load_filelist(filelist_path)
    if args.roundtrip_index < 0 or args.roundtrip_index >= len(samples):
        raise ValueError(f"--roundtrip-index must be in [0, {len(samples) - 1}], got {args.roundtrip_index}")

    canonical_transform = build_unprocessing_transform_from_preset("stf_legacy", randomize=False).to(device)
    randomized_transform = build_unprocessing_transform_from_preset("stf_legacy", randomize=True).to(device)

    roundtrip_image_path, _ = samples[args.roundtrip_index]
    image = load_rgb_tensor(roundtrip_image_path, device=device)
    raw_canonical, metadata_canonical = canonical_transform(image, generator=generator)
    reconstructed = canonical_transform.reprocess(raw_canonical, metadata_canonical, apply_tone_mapping=True)
    image_for_compare = image[..., : reconstructed.shape[-2], : reconstructed.shape[-1]]
    raw_preview = packed_bayer_to_base_rgb(raw_canonical.unsqueeze(0)).squeeze(0)
    raw_preview = torch.nn.functional.interpolate(
        raw_preview.unsqueeze(0),
        size=reconstructed.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    roundtrip_metrics = compute_roundtrip_metrics(image_for_compare, reconstructed)
    save_rgb(output_dir / "roundtrip_original.png", image_for_compare)
    save_rgb(output_dir / "roundtrip_reconstructed.png", reconstructed)
    save_rgb(output_dir / "roundtrip_raw_preview.png", raw_preview)
    save_triptych(output_dir / "roundtrip_triptych.png", (image_for_compare, raw_preview, reconstructed))

    stats_count = min(args.num_stats_images, len(samples))
    total_sum = torch.zeros(4, dtype=torch.float64)
    total_count = torch.zeros(4, dtype=torch.float64)
    channel_min = torch.full((4,), float("inf"), dtype=torch.float64)
    channel_max = torch.full((4,), float("-inf"), dtype=torch.float64)
    global_min = float("inf")
    global_max = float("-inf")
    sampled_params: List[Dict[str, object]] = []

    for image_path, _ in samples[:stats_count]:
        rgb = load_rgb_tensor(image_path, device=device)
        raw, metadata = randomized_transform(rgb, generator=generator)
        raw_cpu = raw.detach().cpu().to(dtype=torch.float64)

        flat = raw_cpu.view(4, -1)
        total_sum += flat.sum(dim=1)
        total_count += torch.full((4,), flat.shape[1], dtype=torch.float64)
        channel_min = torch.minimum(channel_min, flat.min(dim=1).values)
        channel_max = torch.maximum(channel_max, flat.max(dim=1).values)
        global_min = min(global_min, float(raw_cpu.min().item()))
        global_max = max(global_max, float(raw_cpu.max().item()))

        sampled_params.append(
            {
                "image_path": str(image_path),
                "cfa_pattern": metadata["cfa_pattern"],
                "red_gain": float(metadata["red_gain"].item()),
                "blue_gain": float(metadata["blue_gain"].item()),
                "black_level": float(metadata["black_level"].item()),
                "shot_log_gain": float(metadata["shot_log_gain"].item()),
                "shot_noise_scale": float(metadata["shot_noise_scale"].item()),
                "read_noise_std": float(metadata["read_noise_std"].item()),
            }
        )

    stats_summary = {
        "num_images": stats_count,
        "global_min": global_min,
        "global_max": global_max,
        "channel_min": channel_min.tolist(),
        "channel_max": channel_max.tolist(),
        "channel_mean": (total_sum / total_count.clamp_min(1.0)).tolist(),
        "packed_channel_order": ["R", "Gr", "Gb", "B"],
    }

    summary = {
        "roundtrip_image_path": str(roundtrip_image_path),
        "roundtrip_metrics": roundtrip_metrics,
        "roundtrip_metadata": to_jsonable(metadata_canonical),
        "stats_summary": stats_summary,
        "sampled_params_head": sampled_params[: min(10, len(sampled_params))],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
