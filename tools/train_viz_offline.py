#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.train import (  # noqa: E402
    build_datasets,
    build_model,
    build_rgb_reference_eval_model,
    dav2_rgb_pred_label,
    resolve_model_state,
    strip_module_prefix,
)
from finetune_stf.util.viz_dump import (  # noqa: E402
    collect_fixed_train_source_samples,
    dump_train_source_samples,
)


def _load_run_args(run_dir: Path, output_root: Path, *, sources: str | None, n_per_source: int | None):
    config_path = run_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    data["save_path"] = str(output_root)
    data["heavy_save_path"] = data.get("heavy_save_path") or str(output_root)
    data["eval_only"] = False
    data["num_workers"] = 0

    for flag in ("eval_stf", "eval_kitti", "eval_nyu", "eval_eth3d", "eval_robotcar", "eval_robotcar_night"):
        data[flag] = False

    data.setdefault("enable_train_source_viz_dump", True)
    data["train_viz_rgb_baseline"] = bool(data.get("train_viz_rgb_baseline", True))
    data.setdefault("train_viz_rgb_baseline_checkpoint", None)
    data.setdefault("train_viz_rgb_baseline_label", None)
    data.setdefault("train_viz_sources", "auto")
    data.setdefault("train_viz_n_per_source", 8)
    data.setdefault("train_viz_seed", data.get("seed", 42))

    if sources is not None:
        data["train_viz_sources"] = sources
    if n_per_source is not None:
        data["train_viz_n_per_source"] = int(n_per_source)

    return SimpleNamespace(**data)


def _checkpoint_epoch(checkpoint) -> int:
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    if epoch is None:
        raise ValueError("Checkpoint has no integer 'epoch'; pass --epoch explicitly.")
    return int(epoch)


def main():
    parser = argparse.ArgumentParser(description="Regenerate train_viz panels from a saved train.py checkpoint.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", default=None, type=Path)
    parser.add_argument("--epoch", default=None, type=int)
    parser.add_argument("--sources", default=None, help="Override train_viz_sources, e.g. lod_night")
    parser.add_argument("--n-per-source", default=None, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args_cli = parser.parse_args()

    run_dir = args_cli.run_dir.expanduser().resolve()
    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    output_root = (
        args_cli.output_root.expanduser().resolve()
        if args_cli.output_root is not None
        else run_dir / "train_viz_regen"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    args = _load_run_args(run_dir, output_root, sources=args_cli.sources, n_per_source=args_cli.n_per_source)
    datasets = build_datasets(args)
    train_state = {"train_sources": tuple(args.train_sources)}
    fixed_samples = collect_fixed_train_source_samples(train_state, datasets, args)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    epoch = int(args_cli.epoch) if args_cli.epoch is not None else _checkpoint_epoch(checkpoint)

    device = torch.device(args_cli.device)
    model = build_model(args)
    model.load_state_dict(strip_module_prefix(resolve_model_state(checkpoint)), strict=True)
    model.to(device).eval()

    baseline_model = None
    if any(
        record.get("rgb_baseline_input") is not None
        for records in fixed_samples.values()
        for record in records
    ):
        baseline_model = build_rgb_reference_eval_model(args).to(device).eval()

    outputs = dump_train_source_samples(
        model,
        fixed_samples,
        args,
        epoch,
        str(output_root),
        baseline_model=baseline_model,
        baseline_label=args.train_viz_rgb_baseline_label or dav2_rgb_pred_label(args),
    )

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "epoch": epoch,
                "output_root": str(output_root),
                "outputs": {source: len(items) for source, items in outputs.items()},
                "panel_dirs": {
                    source: str(output_root / "train_viz" / f"epoch_{epoch:02d}" / source)
                    for source in outputs
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
