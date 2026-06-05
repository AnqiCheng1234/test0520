#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundation.tools.residual_training_common import METRIC_KEYS, REGION_KEYS, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize N7 control experiments.")
    parser.add_argument("--n7-ablation-root", default=None)
    parser.add_argument("--n7-zero-run-dir", default=None)
    parser.add_argument("--n7-rgb-run-dir", default=None)
    parser.add_argument("--n7-true-run-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--p0-prefix", default=None)
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
    for key in ("best_abs_rel", "best_target_region_score", "best_boundary_abs_rel"):
        record = summary.get(key)
        if isinstance(record, dict):
            return key, record
    val = summary.get("vkitti_val") or summary.get("val") or []
    if val:
        return "latest_val_fallback", {"epoch": val[-1].get("epoch"), "vkitti": val[-1]}
    return "missing", None


def read_ablation_root(root: Path | None) -> list[dict[str, Any]]:
    if root is None or not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("mode_*_scope_*/feature_ablation_summary.json")):
        payload = load_json(summary_path)
        vk = payload.get("vkitti", {})
        kt = payload.get("kitti", {})
        diag = vk.get("diagnostics", {})
        mode = str(payload.get("feature_ablation_mode") or summary_path.parent.name.split("_scope_")[0].replace("mode_", ""))
        scope = str(payload.get("feature_ablation_scope") or summary_path.parent.name.split("_scope_")[-1])
        row = {
            "phase": "P0",
            "mode": mode,
            "scope": scope,
            "run_dir": str(summary_path.parent),
            "source_checkpoint": payload.get("source_checkpoint"),
            "c2_checkpoint": payload.get("c2_checkpoint"),
            "mean_kind": payload.get("feature_ablation_mean_kind"),
            "VK abs_rel": get_path(vk, ["overall", "final", "abs_rel"]),
            "VK d1": get_path(vk, ["overall", "final", "d1"]),
            "VK D1 abs_rel": get_path(vk, ["overall", "D1", "abs_rel"]),
            "VK D0 abs_rel": get_path(vk, ["overall", "D0", "abs_rel"]),
            "final-D1": get_path(vk, ["overall", "delta_final_minus_D1", "abs_rel"]),
            "boundary": get_path(vk, ["region", "final", "boundary_abs_rel"]),
            "high-error": get_path(vk, ["region", "final", "dav2_high_error_abs_rel"]),
            "far50": get_path(vk, ["region", "final", "far50_abs_rel"]),
            "dark": get_path(vk, ["region", "final", "dark_abs_rel"]),
            "saturated": get_path(vk, ["region", "final", "saturated_abs_rel"]),
            "KITTI abs_rel": get_path(kt, ["overall", "final", "abs_rel"]),
            "KITTI d1": get_path(kt, ["overall", "final", "d1"]),
            "KITTI D1 abs_rel": get_path(kt, ["overall", "D1", "abs_rel"]),
            "mean_gate": diag.get("mean_gate"),
            "mean_abs_gate_delta": diag.get("mean_abs_gate_delta"),
            "low_ratio": diag.get("low_ratio"),
            "high_ratio": diag.get("high_ratio"),
            "payload": payload,
        }
        rows.append(row)
    return rows


def run_row(run_dir: Path | None, *, label: str) -> dict[str, Any] | None:
    if run_dir is None or not run_dir.exists():
        return None
    config = load_json(run_dir / "config.json")
    summary = load_json(run_dir / "run_summary.json")
    selected_by, record = best_record(summary)
    if record is None:
        return None
    vk = record.get("vkitti") or {}
    kt = record.get("kitti") or {}
    diag = vk.get("diagnostics", {})
    return {
        "phase": label,
        "method": config.get("experiment_label") or config.get("method_id") or label,
        "run_dir": str(run_dir),
        "selected_by": selected_by,
        "selected_epoch": record.get("epoch"),
        "selected_ckpt": record.get("checkpoint_path"),
        "train_feature_ablation_mode": config.get("train_feature_ablation_mode"),
        "eval_feature_ablation_mode": config.get("eval_feature_ablation_mode"),
        "method_id": config.get("method_id"),
        "incremental_feature_source": config.get("incremental_feature_source"),
        "delta_condition": config.get("delta_condition"),
        "gate_condition": config.get("gate_condition"),
        "trainable_params": get_path(config, ["model_param_counts", "trainable_params"]),
        "VK abs_rel": get_path(vk, ["overall", "final", "abs_rel"]),
        "VK d1": get_path(vk, ["overall", "final", "d1"]),
        "VK D1 abs_rel": get_path(vk, ["overall", "D1", "abs_rel"]),
        "final-D1": get_path(vk, ["overall", "delta_final_minus_D1", "abs_rel"]),
        "boundary": get_path(vk, ["region", "final", "boundary_abs_rel"]),
        "high-error": get_path(vk, ["region", "final", "dav2_high_error_abs_rel"]),
        "far50": get_path(vk, ["region", "final", "far50_abs_rel"]),
        "dark": get_path(vk, ["region", "final", "dark_abs_rel"]),
        "saturated": get_path(vk, ["region", "final", "saturated_abs_rel"]),
        "KITTI abs_rel": get_path(kt, ["overall", "final", "abs_rel"]),
        "KITTI d1": get_path(kt, ["overall", "final", "d1"]),
        "mean_gate": diag.get("mean_gate"),
        "mean_abs_gate_delta": diag.get("mean_abs_gate_delta"),
        "low_ratio": diag.get("low_ratio"),
        "high_ratio": diag.get("high_ratio"),
    }


def invariance_check(rows: list[dict[str, Any]], *, tol: float = 1e-7) -> dict[str, Any]:
    main = [row for row in rows if row.get("scope") == "both" and row.get("mode") in {"true", "shuffle", "zero", "mean"}]
    result: dict[str, Any] = {"tolerance": tol, "D0": {}, "D1": {}, "ok": True}
    for pred_key, label in (("D0", "D0"), ("D1", "D1")):
        diffs: dict[str, float] = {}
        base = next((row for row in main if row.get("mode") == "true"), None)
        for metric in METRIC_KEYS:
            path = ["vkitti", "overall", pred_key, metric]
            base_value = finite(get_path(base.get("payload", {}) if base else {}, path))
            max_diff = 0.0
            if base_value is not None:
                for row in main:
                    value = finite(get_path(row.get("payload", {}), path))
                    if value is not None:
                        max_diff = max(max_diff, abs(value - base_value))
            diffs[metric] = max_diff
        result[label] = diffs
        if any(value > tol for value in diffs.values()):
            result["ok"] = False
    return result


def threshold_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true = next((row for row in rows if row.get("mode") == "true" and row.get("scope") == "both"), None)
    shuffle = next((row for row in rows if row.get("mode") == "shuffle" and row.get("scope") == "both"), None)
    def improvement(metric: str) -> float | None:
        if true is None or shuffle is None:
            return None
        a = finite(true.get(metric))
        b = finite(shuffle.get(metric))
        return None if a is None or b is None else b - a
    checks = {
        "overall_true_minus_shuffle_abs_rel": improvement("VK abs_rel"),
        "boundary_true_minus_shuffle_abs_rel": improvement("boundary"),
        "saturated_true_minus_shuffle_abs_rel": improvement("saturated"),
    }
    return {
        **checks,
        "overall_pass": checks["overall_true_minus_shuffle_abs_rel"] is not None and checks["overall_true_minus_shuffle_abs_rel"] >= 0.001,
        "boundary_pass": checks["boundary_true_minus_shuffle_abs_rel"] is not None and checks["boundary_true_minus_shuffle_abs_rel"] >= 0.006,
        "saturated_pass": checks["saturated_true_minus_shuffle_abs_rel"] is not None and checks["saturated_true_minus_shuffle_abs_rel"] >= 0.005,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = [{k: v for k, v in row.items() if k != "payload"} for row in rows]
    if not flat_rows:
        return
    fieldnames = sorted({key for row in flat_rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def fmt(value: Any) -> str:
    return "" if value is None else str(value)


def write_md(path: Path, *, p0_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]], inv: dict[str, Any], thresholds: dict[str, Any]) -> None:
    lines = ["# N7 controls summary", ""]
    lines += [
        "## P0 N7 eval-time x3 ablation",
        "",
        "| mode | scope | VK abs_rel | VK d1 | final-D1 | boundary | high-error | far50 | dark | saturated | KITTI abs_rel | mean_gate | mean_abs_gate_delta | low_ratio | high_ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in p0_rows:
        lines.append(
            f"| {row.get('mode')} | {row.get('scope')} | {fmt(row.get('VK abs_rel'))} | {fmt(row.get('VK d1'))} | "
            f"{fmt(row.get('final-D1'))} | {fmt(row.get('boundary'))} | {fmt(row.get('high-error'))} | "
            f"{fmt(row.get('far50'))} | {fmt(row.get('dark'))} | {fmt(row.get('saturated'))} | "
            f"{fmt(row.get('KITTI abs_rel'))} | {fmt(row.get('mean_gate'))} | {fmt(row.get('mean_abs_gate_delta'))} | "
            f"{fmt(row.get('low_ratio'))} | {fmt(row.get('high_ratio'))} |"
        )
    lines += [
        "",
        "D0/D1 invariance check:",
        f"- tolerance: {inv.get('tolerance')}",
        f"- ok: {inv.get('ok')}",
        f"- D0 max diffs: {inv.get('D0')}",
        f"- D1 max diffs: {inv.get('D1')}",
        "",
        "Threshold check:",
        f"- overall true-shuffle improvement: {thresholds.get('overall_true_minus_shuffle_abs_rel')} pass={thresholds.get('overall_pass')}",
        f"- boundary true-shuffle improvement: {thresholds.get('boundary_true_minus_shuffle_abs_rel')} pass={thresholds.get('boundary_pass')}",
        f"- saturated true-shuffle improvement: {thresholds.get('saturated_true_minus_shuffle_abs_rel')} pass={thresholds.get('saturated_pass')}",
        "",
        "## P1 N7-zero-x3-train vs N7 true",
        "",
        "| method | selected ckpt | VK abs_rel | VK d1 | final-D1 | boundary | high-error | far50 | saturated | KITTI abs_rel |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in control_rows if r.get("phase") in {"N7 true", "P1"}]:
        lines.append(
            f"| {row.get('method')} | {row.get('selected_ckpt')} | {fmt(row.get('VK abs_rel'))} | {fmt(row.get('VK d1'))} | "
            f"{fmt(row.get('final-D1'))} | {fmt(row.get('boundary'))} | {fmt(row.get('high-error'))} | "
            f"{fmt(row.get('far50'))} | {fmt(row.get('saturated'))} | {fmt(row.get('KITTI abs_rel'))} |"
        )
    lines += [
        "",
        "## P2 N7-RGB vs N7 true",
        "",
        "| method | selected ckpt | VK abs_rel | VK d1 | final-D1 | boundary | high-error | far50 | saturated | KITTI abs_rel |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in control_rows if r.get("phase") in {"N7 true", "P2"}]:
        lines.append(
            f"| {row.get('method')} | {row.get('selected_ckpt')} | {fmt(row.get('VK abs_rel'))} | {fmt(row.get('VK d1'))} | "
            f"{fmt(row.get('final-D1'))} | {fmt(row.get('boundary'))} | {fmt(row.get('high-error'))} | "
            f"{fmt(row.get('far50'))} | {fmt(row.get('saturated'))} | {fmt(row.get('KITTI abs_rel'))} |"
        )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "If true ~= shuffle/zero/mean, N7 cannot be used as evidence that image-corresponding x3 matters.",
        "If N7 true ~= N7-zero-x3-train, N7 improvement is mostly D1-conditioned head capacity, not x3.",
        "If N7 true ~= N7RGB, RAW/RAM x3 is effective but not clearly better than matched RGB cue under clean VKITTI.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    p0_rows = read_ablation_root(Path(args.n7_ablation_root).expanduser().resolve() if args.n7_ablation_root else None)
    control_rows = [
        row
        for row in (
            run_row(Path(args.n7_true_run_dir).expanduser().resolve() if args.n7_true_run_dir else None, label="N7 true"),
            run_row(Path(args.n7_zero_run_dir).expanduser().resolve() if args.n7_zero_run_dir else None, label="P1"),
            run_row(Path(args.n7_rgb_run_dir).expanduser().resolve() if args.n7_rgb_run_dir else None, label="P2"),
        )
        if row is not None
    ]
    inv = invariance_check(p0_rows)
    thresholds = threshold_check(p0_rows)
    payload = {
        "n7_ablation_root": args.n7_ablation_root,
        "n7_true_run_dir": args.n7_true_run_dir,
        "n7_zero_run_dir": args.n7_zero_run_dir,
        "n7_rgb_run_dir": args.n7_rgb_run_dir,
        "p0_rows": [{k: v for k, v in row.items() if k != "payload"} for row in p0_rows],
        "control_rows": control_rows,
        "invariance_check": inv,
        "threshold_check": thresholds,
    }
    rows = [{k: v for k, v in row.items() if k != "payload"} for row in p0_rows] + control_rows
    save_json(out_dir / "n7_controls_summary.json", payload)
    write_csv(out_dir / "n7_controls_records.csv", rows)
    write_md(out_dir / "n7_controls_summary.md", p0_rows=p0_rows, control_rows=control_rows, inv=inv, thresholds=thresholds)
    if args.p0_prefix:
        save_json(out_dir / f"{args.p0_prefix}_summary.json", payload)
        write_csv(out_dir / f"{args.p0_prefix}_records.csv", rows)
        write_md(out_dir / f"{args.p0_prefix}_summary.md", p0_rows=p0_rows, control_rows=control_rows, inv=inv, thresholds=thresholds)
    if not inv.get("ok", True):
        raise RuntimeError("D0/D1 invariance check failed; ablation appears to affect the frozen path.")


if __name__ == "__main__":
    main()
