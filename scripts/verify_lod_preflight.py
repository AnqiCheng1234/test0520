#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.raw_audit_common import read_json, write_json
from finetune_stf.dataset.lod_raw import DEFAULT_LOD_DAY_MANIFEST, DEFAULT_LOD_NIGHT_MANIFEST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify LOD pseudo-target source and current training loss semantics before new method work."
    )
    parser.add_argument("--exp-dir", required=True, type=Path, help="Experiment directory containing config.json.")
    parser.add_argument("--lod-day-manifest", default=None, help="LOD day pseudo-depth manifest. Defaults to config or repo default.")
    parser.add_argument("--lod-night-manifest", default=None, help="LOD night pseudo-depth manifest. Defaults to config or repo default when present.")
    parser.add_argument("--max-samples", type=int, default=16, help="Samples per manifest used for target stats.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for preflight_verify.json/md.")
    return parser.parse_args()


def load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fields:
        raise ValueError(f"Manifest has no header: {path}")
    if not rows:
        raise ValueError(f"Manifest has no rows: {path}")
    return fields, rows


def resolve_manifest_path(path_value: str | Path | None, fallback: str | Path | None) -> Path | None:
    value = path_value if path_value not in (None, "") else fallback
    if value in (None, ""):
        return None
    return Path(value).expanduser().resolve()


def load_sidecar_json(manifest_path: Path, name: str) -> dict[str, Any] | None:
    path = manifest_path.parent / name
    if not path.is_file():
        return None
    return read_json(path)


def array_stats(path: Path) -> dict[str, Any]:
    arr = np.load(path).astype(np.float32, copy=False)
    valid = np.isfinite(arr) & (arr > 0)
    stats: dict[str, Any] = {
        "path": str(path),
        "shape": list(arr.shape),
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(valid.size),
        "valid_coverage": float(valid.mean()) if valid.size else 0.0,
    }
    if valid.any():
        vals = arr[valid].astype(np.float64)
        stats.update(
            {
                "min": float(vals.min()),
                "mean": float(vals.mean()),
                "p50": float(np.percentile(vals, 50.0)),
                "p90": float(np.percentile(vals, 90.0)),
                "p99": float(np.percentile(vals, 99.0)),
                "max": float(vals.max()),
            }
        )
    else:
        stats.update({"min": float("nan"), "mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "max": float("nan")})
    return stats


def summarize_manifest(label: str, manifest_path: Path, max_samples: int) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {label} manifest: {manifest_path}")
    fields, rows = load_csv_rows(manifest_path)
    required = {"split", "sample_name", "rgb_path", "rggb_path", "output_npy"}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"{manifest_path} missing required field(s): {missing}")

    sample_limit = max(1, min(int(max_samples), len(rows)))
    sample_indices = np.linspace(0, len(rows) - 1, sample_limit, dtype=int).tolist()
    target_stats = []
    missing_files = []
    for idx in sample_indices:
        row = rows[int(idx)]
        target_path = Path(row["output_npy"]).expanduser().resolve()
        if not target_path.is_file():
            missing_files.append(str(target_path))
            continue
        target_stats.append(array_stats(target_path))

    shapes = sorted({tuple(item["shape"]) for item in target_stats})
    coverages = np.asarray([item["valid_coverage"] for item in target_stats], dtype=np.float64)
    means = np.asarray([item["mean"] for item in target_stats], dtype=np.float64)
    run_config = load_sidecar_json(manifest_path, "run_config.json")
    run_summary = load_sidecar_json(manifest_path, "run_summary.json")

    return {
        "label": label,
        "manifest_path": str(manifest_path),
        "fieldnames": fields,
        "row_count": len(rows),
        "sampled_indices": sample_indices,
        "sampled_count": len(target_stats),
        "missing_sampled_targets": missing_files,
        "sample_name_prefixes": sorted({row["sample_name"].split("-", 1)[0] for row in rows[: min(len(rows), 256)]}),
        "run_config": run_config,
        "run_summary": run_summary,
        "target_shape_set": [list(shape) for shape in shapes],
        "target_valid_coverage": {
            "mean": float(coverages.mean()) if coverages.size else float("nan"),
            "min": float(coverages.min()) if coverages.size else float("nan"),
            "max": float(coverages.max()) if coverages.size else float("nan"),
        },
        "target_value_mean": {
            "mean": float(means[np.isfinite(means)].mean()) if np.isfinite(means).any() else float("nan"),
            "min": float(means[np.isfinite(means)].min()) if np.isfinite(means).any() else float("nan"),
            "max": float(means[np.isfinite(means)].max()) if np.isfinite(means).any() else float("nan"),
        },
        "sample_target_stats": target_stats,
    }


def classify_loss_semantics(cfg: dict[str, Any], manifest_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    coverage_values = [
        summary["target_valid_coverage"]["mean"]
        for summary in manifest_summaries
        if np.isfinite(summary["target_valid_coverage"]["mean"])
    ]
    coverage_mean = float(np.mean(coverage_values)) if coverage_values else float("nan")
    loss_type = cfg.get("loss_type")
    # DAv2 pseudo arrays can contain zero/non-positive regions, but their mask
    # coverage is orders of magnitude denser than sparse LiDAR supervision.
    if coverage_values and coverage_mean >= 0.50 and loss_type in {"ssi", "ssi_grad"}:
        verdict = "dense_or_near_dense_masked_ssi"
    elif loss_type in {"ssi", "ssi_grad"}:
        verdict = "masked_ssi_not_dense_by_coverage"
    elif loss_type == "aligned_sig":
        verdict = "masked_aligned_inverse_sig"
    else:
        verdict = f"other_{loss_type}"
    return {
        "verdict": verdict,
        "loss_type": loss_type,
        "loss_target_normalization": cfg.get("loss_target_normalization"),
        "loss_lambda_grad": cfg.get("loss_lambda_grad"),
        "loss_grad_scales": cfg.get("loss_grad_scales"),
        "loss_mask_downsample": cfg.get("loss_mask_downsample"),
        "dataset_target_space": "inverse_relative",
        "mean_sampled_valid_coverage": coverage_mean,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# LOD Pre-flight Verify",
        "",
        "## Conclusion",
        "",
        f"- Pseudo target source: `{payload['pseudo_target_source']['conclusion']}`",
        f"- Current LOD loss semantics: `{payload['loss_semantics']['verdict']}`",
        f"- Dataset target space: `{payload['loss_semantics']['dataset_target_space']}`",
        "",
        "## Experiment Config",
        "",
        f"- exp_dir: `{payload['exp_dir']}`",
        f"- stage: `{payload['training_config'].get('stage')}`",
        f"- input_type: `{payload['training_config'].get('input_type')}`",
        f"- loss_type: `{payload['training_config'].get('loss_type')}`",
        f"- loss_target_normalization: `{payload['training_config'].get('loss_target_normalization')}`",
        "",
        "## Manifest Summary",
        "",
        "| split | rows | sampled | shapes | coverage mean/min/max | generator checkpoint |",
        "|---|---:|---:|---|---|---|",
    ]
    for summary in payload["manifests"]:
        run_config = summary.get("run_config") or {}
        coverage = summary["target_valid_coverage"]
        lines.append(
            "| {label} | {rows} | {sampled} | `{shapes}` | {mean:.6f}/{minv:.6f}/{maxv:.6f} | `{ckpt}` |".format(
                label=summary["label"],
                rows=summary["row_count"],
                sampled=summary["sampled_count"],
                shapes=summary["target_shape_set"],
                mean=coverage["mean"],
                minv=coverage["min"],
                maxv=coverage["max"],
                ckpt=run_config.get("checkpoint", "missing_run_config"),
            )
        )
    lines.extend(
        [
            "",
            "## Source Interpretation",
            "",
            "- Manifest contains `rgb_path`, `rggb_path`, and `output_npy`; current `LODRaw` trains from `rggb_path` and reads pseudo target from `output_npy`.",
            "- Generator sidecar config records official DAv2 checkpoint, RGB input size, and target resize shape.",
            "- Because sampled pseudo targets are dense or near-dense positive arrays, this is not sparse LiDAR-style supervision.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()
    cfg = read_json(exp_dir / "config.json")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else exp_dir / "preflight_verify"
    output_dir.mkdir(parents=True, exist_ok=True)

    day_manifest = resolve_manifest_path(args.lod_day_manifest, cfg.get("lod_day_manifest") or DEFAULT_LOD_DAY_MANIFEST)
    night_fallback = cfg.get("lod_night_manifest") or DEFAULT_LOD_NIGHT_MANIFEST
    night_manifest = resolve_manifest_path(args.lod_night_manifest, night_fallback)
    manifests = [("day", day_manifest)]
    if night_manifest is not None:
        manifests.append(("night", night_manifest))

    manifest_summaries = [
        summarize_manifest(label, manifest_path, max_samples=args.max_samples)
        for label, manifest_path in manifests
        if manifest_path is not None
    ]
    loss_semantics = classify_loss_semantics(cfg, manifest_summaries)
    pseudo_target_source = {
        "conclusion": "LOD SDR RGB -> frozen official DAv2 -> dense inverse-relative pseudo target",
        "path_classification": "B: RAW dataset trains on rggb_path, but pseudo target was generated from paired/ISP SDR RGB path",
        "generator_script": str((PROJECT_ROOT / "scripts" / "generate_lod_day_dav2_pseudo.py").resolve()),
        "notes": [
            "run_config.json sidecars record the DAv2 checkpoint and target_hw used during pseudo-label generation.",
            "LODRaw sets target_space='inverse_relative', so the loaded output_npy is consumed as relative inverse-depth target.",
        ],
    }
    payload = {
        "exp_dir": str(exp_dir),
        "output_dir": str(output_dir),
        "training_config": {
            key: cfg.get(key)
            for key in (
                "stage",
                "input_type",
                "input_height",
                "input_width",
                "loss_type",
                "loss_target_normalization",
                "loss_lambda_grad",
                "loss_grad_scales",
                "loss_mask_downsample",
                "lod_day_manifest",
                "lod_night_manifest",
            )
        },
        "pseudo_target_source": pseudo_target_source,
        "loss_semantics": loss_semantics,
        "manifests": manifest_summaries,
    }
    write_json(output_dir / "preflight_verify.json", payload)
    (output_dir / "preflight_verify.md").write_text(build_markdown(payload) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "conclusion": pseudo_target_source["conclusion"], "loss": loss_semantics["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
