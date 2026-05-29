#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundation.tools.residual_training_common import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize VKITTI N-series incremental residual runs.")
    parser.add_argument("--c2-run", required=True)
    parser.add_argument("--n2-runs", nargs="*", default=[])
    parser.add_argument("--n3-run", default=None)
    parser.add_argument("--n4-run", default=None)
    parser.add_argument("--n5-run", default=None)
    parser.add_argument("--n7-run", default=None)
    parser.add_argument("--feature-ablation-dir", default=None)
    parser.add_argument("--energy-frequency-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def get_path(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def best_record(summary: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    for key in ("best_target_region_score", "best_abs_rel"):
        record = summary.get(key)
        if isinstance(record, dict):
            return key, record
    val = summary.get("val") or []
    if val:
        return "latest_val_fallback", {"epoch": val[-1].get("epoch"), "vkitti": val[-1]}
    return "missing", None


def read_feature_ablation(feature_dir: Path | None) -> dict[str, Any]:
    if feature_dir is None or not feature_dir.exists():
        return {}
    candidates = list(feature_dir.rglob("summary.csv"))
    out: dict[str, Any] = {}
    for csv_path in candidates:
        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mode = row.get("feature_ablation_mode")
                if mode:
                    out[f"{csv_path.parent.name}:{mode}"] = row
    return out


def row_from_run(run_dir: Path, *, selected_by: str | None = None) -> dict[str, Any]:
    config = load_json(run_dir / "config.json")
    summary = load_json(run_dir / "run_summary.json")
    best_key, record = best_record(summary)
    if record is None:
        raise ValueError(f"No best/val metrics found in {run_dir}")
    vkitti = record.get("vkitti") or {}
    kitti = record.get("kitti") or {}
    model_counts = config.get("model_param_counts", {})
    overall = vkitti.get("overall", {})
    region = vkitti.get("region", {})
    diagnostics = vkitti.get("diagnostics", {})
    kitti_overall = kitti.get("overall", {})
    row = {
        "method_id": config.get("method_id"),
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "selected_by": selected_by or best_key,
        "best_epoch": record.get("epoch"),
        "lambda_lp": config.get("lambda_lp"),
        "q_good": config.get("q_good"),
        "lambda_lowfreq_loss": config.get("lambda_lowfreq_loss"),
        "incremental_feature_source": config.get("incremental_feature_source"),
        "delta_condition": config.get("delta_condition"),
        "gate_condition": config.get("gate_condition"),
        "raw_feature_encoder_trainable": config.get("raw_feature_encoder_trainable"),
        "trainable_params": model_counts.get("trainable_params"),
        "c2_checkpoint": config.get("c2_checkpoint"),
        "VKITTI final abs_rel": get_path(overall, ["final", "abs_rel"]),
        "VKITTI D1 abs_rel": get_path(overall, ["D1", "abs_rel"]),
        "final - D1 abs_rel": get_path(overall, ["delta_final_minus_D1", "abs_rel"]),
        "boundary final": get_path(region, ["final", "boundary_abs_rel"]),
        "boundary final - D1": get_path(region, ["delta_final_minus_D1", "boundary_abs_rel"]),
        "far50 final": get_path(region, ["final", "far50_abs_rel"]),
        "far50 final - D1": get_path(region, ["delta_final_minus_D1", "far50_abs_rel"]),
        "dark final": get_path(region, ["final", "dark_abs_rel"]),
        "saturated final": get_path(region, ["final", "saturated_abs_rel"]),
        "target_region_score": vkitti.get("target_region_score"),
        "mean_gate": diagnostics.get("mean_gate"),
        "mean_abs_gate_delta": diagnostics.get("mean_abs_gate_delta"),
        "low_ratio": diagnostics.get("low_ratio"),
        "high_ratio": diagnostics.get("high_ratio"),
        "KITTI final abs_rel": get_path(kitti_overall, ["final", "abs_rel"]),
        "KITTI D1 abs_rel": get_path(kitti_overall, ["D1", "abs_rel"]),
        "KITTI final - D1": get_path(kitti_overall, ["delta_final_minus_D1", "abs_rel"]),
        "checkpoint_path": record.get("checkpoint_path"),
    }
    return row


def classify(row: dict[str, Any], *, n3_abs_rel: float | None, x3_shuffle_gain: float | None) -> str:
    overall_delta = finite(row.get("final - D1 abs_rel"))
    boundary_delta = finite(row.get("boundary final - D1"))
    target_delta = finite(row.get("target_region_score"))
    kitti_delta = finite(row.get("KITTI final - D1"))
    n2_abs_rel = finite(row.get("VKITTI final abs_rel"))
    overall_improve_eps = 0.002
    region_improve_eps = 0.003
    feature_gain_eps = 0.001
    kitti_regress_eps = 0.005
    tie_eps = 0.002
    if (
        overall_delta is not None
        and boundary_delta is not None
        and x3_shuffle_gain is not None
        and kitti_delta is not None
        and overall_delta <= -overall_improve_eps
        and boundary_delta <= -region_improve_eps
        and x3_shuffle_gain >= feature_gain_eps
        and kitti_delta <= kitti_regress_eps
        and (n3_abs_rel is None or (n2_abs_rel is not None and n2_abs_rel <= n3_abs_rel - feature_gain_eps))
    ):
        return "strong_success"
    if (
        overall_delta is not None
        and overall_delta <= tie_eps
        and x3_shuffle_gain is not None
        and x3_shuffle_gain >= feature_gain_eps
        and (
            (boundary_delta is not None and boundary_delta <= -region_improve_eps)
            or (target_delta is not None and target_delta <= -region_improve_eps)
        )
    ):
        return "medium_success"
    if (
        overall_delta is not None
        and boundary_delta is not None
        and target_delta is not None
        and overall_delta > tie_eps
        and boundary_delta > -region_improve_eps
        and target_delta > -region_improve_eps
    ):
        return "failed"
    if x3_shuffle_gain is not None and x3_shuffle_gain < 0.5 * feature_gain_eps:
        return "failed"
    return "inconclusive"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# VKITTI N-Series Summary\n\n")
        if not rows:
            f.write("No N-series runs provided.\n")
            return
        f.write("| method | run | epoch | final | D1 | final-D1 | boundary-D1 | KITTI-D1 | x3 shuffle | class |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row.get('method_id')} | {row.get('run_name')} | {row.get('best_epoch')} | "
                f"{row.get('VKITTI final abs_rel')} | {row.get('VKITTI D1 abs_rel')} | "
                f"{row.get('final - D1 abs_rel')} | {row.get('boundary final - D1')} | "
                f"{row.get('KITTI final - D1')} | {row.get('feature ablation shuffled - true')} | "
                f"{row.get('classification')} |\n"
            )
        f.write("\nConclusion labels are threshold-based and should be checked against qualitative panels.\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%m%d_%H%M")
    run_dirs = [Path(p).expanduser().resolve() for p in args.n2_runs]
    for optional in (args.n3_run, args.n4_run, args.n5_run, args.n7_run):
        if optional:
            run_dirs.append(Path(optional).expanduser().resolve())
    feature_rows = read_feature_ablation(Path(args.feature_ablation_dir).expanduser().resolve() if args.feature_ablation_dir else None)
    x3_true = None
    x3_shuffle = None
    x3_zero = None
    shuffle_policy = None
    shuffle_seed = None
    donor_sha = None
    for key, row in feature_rows.items():
        if not key.startswith("m") and "n2" not in key.lower() and "x3" not in key.lower():
            continue
        mode = row.get("feature_ablation_mode")
        if mode == "true":
            x3_true = finite(row.get("abs_rel"))
        elif mode == "shuffle":
            x3_shuffle = finite(row.get("abs_rel"))
            shuffle_policy = row.get("shuffle_policy")
            shuffle_seed = row.get("shuffle_seed")
            donor_sha = row.get("donor_mapping_sha256")
        elif mode == "zero":
            x3_zero = finite(row.get("abs_rel"))
    x3_shuffle_gain = None if x3_true is None or x3_shuffle is None else x3_shuffle - x3_true
    x3_zero_gain = None if x3_true is None or x3_zero is None else x3_zero - x3_true
    rows = [row_from_run(run_dir) for run_dir in run_dirs]
    n3_abs_rel = next((finite(row.get("VKITTI final abs_rel")) for row in rows if row.get("method_id") == "N3"), None)
    for row in rows:
        row["feature ablation shuffled - true"] = x3_shuffle_gain
        row["x3_shuffle_gain"] = x3_shuffle_gain
        row["x3_zero_gain"] = x3_zero_gain
        row["shuffle_policy"] = shuffle_policy
        row["shuffle_seed"] = shuffle_seed
        row["donor_mapping_sha256"] = donor_sha
        row["classification"] = classify(row, n3_abs_rel=n3_abs_rel, x3_shuffle_gain=x3_shuffle_gain)
    payload = {
        "c2_run": str(Path(args.c2_run).expanduser().resolve()),
        "rows": rows,
        "feature_ablation_dir": args.feature_ablation_dir,
        "energy_frequency_dir": args.energy_frequency_dir,
    }
    json_path = output_dir / f"{timestamp}_vkitti_nseries_summary.json"
    csv_path = output_dir / f"{timestamp}_vkitti_nseries_summary.csv"
    md_path = output_dir / f"{timestamp}_vkitti_nseries_summary.md"
    save_json(json_path, payload)
    write_csv(csv_path, rows)
    write_md(md_path, rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
