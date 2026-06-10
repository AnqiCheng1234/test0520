#!/usr/bin/env python3
"""Collect RamCore3 post-BN x_ram statistics for true-LOD RAW runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.lod_raw_cross_eval import (  # noqa: E402
    INPUT_DARK,
    config_value,
    get_config,
    load_checkpoint_into_model,
    make_dataset,
    make_loader,
    train_args_from_run_config,
)
from finetune_stf.train import build_model, set_random_seed  # noqa: E402
from finetune_stf.util.model_input import select_model_input  # noqa: E402


DEFAULT_RUN = "0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly"
DEFAULT_EXP_ROOT = PROJECT_ROOT / "finetune_stf" / "exp"
DEFAULT_CKPT_ROOT = Path("/mnt/drive/3333_raw/0000_exp_ckpt")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "finetune_stf" / "analysis" / "lod_postram_external"
QUANTILES = {
    "q001": 0.001,
    "q01": 0.01,
    "q05": 0.05,
    "q50": 0.50,
    "q95": 0.95,
    "q99": 0.99,
    "q999": 0.999,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_EXP_ROOT / DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT_ROOT / DEFAULT_RUN / "best_model.pth")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_ROOT / "xram_stats_00Train.json")
    parser.add_argument("--lod-root", type=Path, default=None)
    parser.add_argument("--lod-manifest", type=Path, default=None)
    parser.add_argument("--pretrained-from", type=Path, default=None)
    parser.add_argument("--split", default="00Train")
    parser.add_argument("--crop-mode", default="center", choices=("center", "random"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-pixels-per-channel", type=int, default=2_000_000)
    parser.add_argument("--sample-pixels-per-batch", type=int, default=32768)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default=None)
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ChannelStats:
    def __init__(self, channels: int, *, max_pixels: int, sample_pixels_per_batch: int, seed: int):
        self.channels = int(channels)
        self.max_pixels = int(max_pixels)
        self.sample_pixels_per_batch = int(sample_pixels_per_batch)
        self.rng = np.random.default_rng(int(seed))
        self.count = np.zeros(self.channels, dtype=np.int64)
        self.sum = np.zeros(self.channels, dtype=np.float64)
        self.sumsq = np.zeros(self.channels, dtype=np.float64)
        self.min = np.full(self.channels, np.inf, dtype=np.float64)
        self.max = np.full(self.channels, -np.inf, dtype=np.float64)
        self.samples: list[list[np.ndarray]] = [[] for _ in range(self.channels)]
        self.sample_counts = np.zeros(self.channels, dtype=np.int64)

    def update(self, x: torch.Tensor) -> None:
        arr = x.detach().float().cpu().numpy()
        if arr.ndim != 4 or arr.shape[1] != self.channels:
            raise ValueError(f"Expected BCHW with {self.channels} channels, got {arr.shape}")
        for ch in range(self.channels):
            flat = arr[:, ch].reshape(-1)
            finite = flat[np.isfinite(flat)]
            if finite.size == 0:
                continue
            self.count[ch] += int(finite.size)
            self.sum[ch] += float(finite.sum(dtype=np.float64))
            self.sumsq[ch] += float(np.square(finite, dtype=np.float64).sum(dtype=np.float64))
            self.min[ch] = min(self.min[ch], float(finite.min()))
            self.max[ch] = max(self.max[ch], float(finite.max()))
            remaining = self.max_pixels - int(self.sample_counts[ch])
            if remaining <= 0:
                continue
            take = min(remaining, self.sample_pixels_per_batch, int(finite.size))
            if take <= 0:
                continue
            if take == finite.size:
                sampled = finite.astype(np.float32, copy=True)
            else:
                indices = self.rng.choice(int(finite.size), size=int(take), replace=False)
                sampled = finite[indices].astype(np.float32, copy=True)
            self.samples[ch].append(sampled)
            self.sample_counts[ch] += int(sampled.size)

    def summary(self) -> dict[str, Any]:
        rows = []
        for ch in range(self.channels):
            count = int(self.count[ch])
            mean = float(self.sum[ch] / count) if count else float("nan")
            variance = float(self.sumsq[ch] / count - mean * mean) if count else float("nan")
            std = math.sqrt(max(variance, 0.0)) if math.isfinite(variance) else float("nan")
            sample = np.concatenate(self.samples[ch]) if self.samples[ch] else np.empty((0,), dtype=np.float32)
            row: dict[str, Any] = {
                "channel": int(ch),
                "count": count,
                "mean": mean,
                "std": std,
                "min": float(self.min[ch]) if count else float("nan"),
                "max": float(self.max[ch]) if count else float("nan"),
                "quantile_sample_count": int(sample.size),
            }
            for name, q in QUANTILES.items():
                row[name] = float(np.quantile(sample, q)) if sample.size else float("nan")
            rows.append(row)
        return {
            "channels": rows,
            "quantiles": QUANTILES,
            "quantile_estimation": {
                "method": "per-batch random pixel sampling",
                "max_pixels_per_channel": int(self.max_pixels),
                "sample_pixels_per_batch": int(self.sample_pixels_per_batch),
            },
        }


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    config, _ = get_config(args.run_dir)
    train_args = train_args_from_run_config(
        config=config,
        eval_input=INPUT_DARK,
        output_dir=args.output_json.parent / "_build_args_collect_stats",
        args=args,
    )
    dataset = make_dataset(
        train_args,
        input_spec=INPUT_DARK,
        split=args.split,
        crop_mode=args.crop_mode,
    )
    loader: DataLoader = make_loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    model = build_model(train_args)
    checkpoint_meta = load_checkpoint_into_model(model, args.checkpoint)
    model.to(device)
    model.eval()

    stats = ChannelStats(
        channels=3,
        max_pixels=args.max_pixels_per_channel,
        sample_pixels_per_batch=args.sample_pixels_per_batch,
        seed=args.seed,
    )
    processed = 0
    start = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_samples is not None and processed >= int(args.max_samples):
                break
            model_input = select_model_input(
                batch,
                train_args.resolved_config.model_input_tensor,
                dataset_family=train_args.resolved_config.dataset_family,
                sample_source="postram_collect_stats",
            ).to(device=device, non_blocking=True).float()
            if args.max_samples is not None:
                keep = max(min(int(args.max_samples) - processed, int(model_input.shape[0])), 0)
                if keep <= 0:
                    break
                model_input = model_input[:keep]
            _, ram_features = model.ram_core.forward_with_features(model_input)
            x_ram = ram_features.get("x3_ram")
            if x_ram is None:
                raise RuntimeError("RamCore3 did not return x3_ram in forward_with_features")
            stats.update(x_ram)
            processed += int(model_input.shape[0])
            if args.progress_interval > 0 and (batch_idx + 1) % int(args.progress_interval) == 0:
                print(f"[COLLECT_STATS] batches={batch_idx + 1} samples={processed}", flush=True)

    payload = stats.summary()
    payload.update(
        {
            "run_dir": str(args.run_dir.expanduser().resolve()),
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_meta": checkpoint_meta,
            "split": str(args.split),
            "crop_mode": str(args.crop_mode),
            "samples": int(processed),
            "dataset_size": int(len(dataset)),
            "elapsed_sec": float(time.time() - start),
            "device": str(device),
            "train_args_resolved_config": train_args.resolved_config.to_dict(),
        }
    )
    write_json(args.output_json.expanduser().resolve(), payload)
    print(f"[COLLECT_STATS] wrote {args.output_json.expanduser().resolve()} samples={processed}", flush=True)


if __name__ == "__main__":
    main()
