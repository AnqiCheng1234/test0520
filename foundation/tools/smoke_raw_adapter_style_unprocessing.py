#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundation.engine.transforms import (
    RAW_ADAPTER_PACKED_CHANNEL_ORDER,
    RawAdapterStyleUnprocessingTransform,
    resolve_unprocessing_config,
)


def load_offline_module():
    path = PROJECT_ROOT / "plans" / "0524_unprocessing" / "unprocess_rgb_to_packed_raw.py"
    spec = importlib.util.spec_from_file_location("offline_raw_adapter_unprocess", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load offline unprocessing script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def make_config(*, inverse_tone: str, ccm: str) -> dict[str, Any]:
    return resolve_unprocessing_config(
        {
            "unprocessing_method": "raw_adapter_style",
            "vkitti_unprocessing_preset": "not_applicable",
            "vkitti_unprocessing_mix_weights": None,
            "randomize_unprocessing": False,
            "raw_adapter_backend": "analytic",
            "raw_adapter_cfa_pattern": "RGGB",
            "raw_adapter_packed_channel_order": RAW_ADAPTER_PACKED_CHANNEL_ORDER,
            "raw_adapter_rgb_transfer": "srgb_piecewise",
            "raw_adapter_inverse_tone": inverse_tone,
            "raw_adapter_ccm": ccm,
            "raw_adapter_red_gain_range": [1.9, 2.4],
            "raw_adapter_blue_gain_range": [1.5, 1.9],
            "raw_adapter_fixed_red_gain": 2.15,
            "raw_adapter_fixed_blue_gain": 1.70,
            "raw_adapter_variant_policy": "normal",
            "raw_adapter_variant_weights": "normal=1.0,dark=0.0,over=0.0",
            "raw_adapter_fixed_light_scale": 1.0,
            "raw_adapter_dark_light_scale_range": [0.05, 0.4],
            "raw_adapter_over_light_scale_range": [1.5, 2.5],
            "raw_adapter_shot_noise": 0.001,
            "raw_adapter_read_noise": 0.0005,
            "raw_adapter_noise_mean_mode": "zero",
            "raw_adapter_black_level": 0.0,
            "raw_adapter_white_level": 1.0,
            "raw_adapter_random_seed_policy": "dataloader_generator",
        }
    )


def offline_reference(rgb_chw: np.ndarray, config: dict[str, Any], offline: Any) -> np.ndarray:
    rgb_hwc = np.transpose(rgb_chw, (1, 2, 0)).astype(np.float32)
    raw_rgb = offline.analytic_rgb_to_raw_rgb(
        rgb_hwc,
        inverse_tone=config["raw_adapter_inverse_tone"] == "global_0p15",
        ccm_name=config["raw_adapter_ccm"],
    )
    raw_rgb = offline.inverse_white_balance(
        raw_rgb,
        float(config["raw_adapter_fixed_red_gain"]),
        float(config["raw_adapter_fixed_blue_gain"]),
    )
    raw_rgb = offline.apply_light_synthesis(
        raw_rgb,
        float(config["raw_adapter_fixed_light_scale"]),
        np.random.default_rng(123),
        shot_noise=0.0,
        read_noise=0.0,
        noise_mean_mode=str(config["raw_adapter_noise_mean_mode"]),
    )
    raw_rgb = offline.apply_black_white_levels(
        raw_rgb,
        float(config["raw_adapter_black_level"]),
        float(config["raw_adapter_white_level"]),
    )
    return offline.pack_rggb(raw_rgb).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test RAW-Adapter-style online unprocessing parity.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if "codex_smoke" not in str(output):
        raise ValueError("--output must contain codex_smoke")

    offline = load_offline_module()
    rng = np.random.default_rng(20260525)
    rgb = rng.uniform(0.0, 1.0, size=(3, 18, 26)).astype(np.float32)
    rgb[:, 0, 0] = 0.0
    rgb[:, -1, -1] = 1.0

    cases = [
        ("none", "identity"),
        ("global_0p15", "identity"),
        ("global_0p15", "generic_d65"),
    ]
    rows = []
    for inverse_tone, ccm in cases:
        config = make_config(inverse_tone=inverse_tone, ccm=ccm)
        transform = RawAdapterStyleUnprocessingTransform(config)
        with torch.no_grad():
            actual, metadata = transform(torch.from_numpy(rgb))
        expected = offline_reference(rgb, config, offline)
        actual_np = actual.detach().cpu().numpy().astype(np.float32)
        abs_diff = np.abs(actual_np - expected)
        max_abs = float(abs_diff.max())
        mean_abs = float(abs_diff.mean())
        if tuple(actual_np.shape) != (4, 9, 13):
            raise AssertionError(f"Unexpected packed shape: {actual_np.shape}")
        if not np.all(np.isfinite(actual_np)):
            raise AssertionError("Transform output contains non-finite values")
        if actual_np.min() < -1e-7 or actual_np.max() > 1.0 + 1e-7:
            raise AssertionError(f"Transform output range invalid: {actual_np.min()}..{actual_np.max()}")
        if max_abs > 2e-6:
            raise AssertionError(
                f"Parity failed for inverse_tone={inverse_tone} ccm={ccm}: max_abs={max_abs}"
            )
        if metadata["inverse_tone"] != inverse_tone:
            raise AssertionError(f"metadata inverse_tone mismatch: {metadata['inverse_tone']} != {inverse_tone}")
        if metadata["noise_model"] != "none" or bool(metadata["noise_realization_applied"]):
            raise AssertionError("Fixed smoke must not apply noise realization")
        rows.append(
            {
                "inverse_tone": inverse_tone,
                "ccm": ccm,
                "shape": list(actual_np.shape),
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "raw_adapter_config_hash": metadata["raw_adapter_config_hash"],
            }
        )

    if not math.isfinite(max(row["max_abs_diff"] for row in rows)):
        raise AssertionError("Non-finite parity diff")
    save_json(output, {"status": "ok", "cases": rows})
    print(json.dumps({"status": "ok", "output": str(output), "cases": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
