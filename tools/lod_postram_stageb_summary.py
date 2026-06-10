#!/usr/bin/env python3
"""Summarize Stage-B post-RAM cleanup strict eval outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP_ROOT = PROJECT_ROOT / "finetune_stf" / "exp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--stage-a-noop-metrics", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def fmt(value: Any, digits: int = 6) -> str:
    result = finite_float(value)
    if not math.isfinite(result):
        return "n_a"
    return f"{result:.{digits}f}"


def seed_from_run(run_id: str) -> str:
    match = re.search(r"seed(\d+)", run_id)
    return match.group(1) if match else "n_a"


def kind_from_run(run_id: str) -> str:
    if "T_LORA" in run_id:
        return "T_LORA_s01_cleanup"
    if "B0_LORA" in run_id:
        return "B0_LORA_s01_baseline"
    return "strict_noop_reference"


def label_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    kind = str(row["kind"])
    if kind == "strict_noop_reference":
        return (0, 0, str(row["run_id"]))
    seed = row["seed"]
    seed_int = int(seed) if str(seed).isdigit() else 999999
    kind_rank = 0 if kind.startswith("B0_") else 1
    return (1, seed_int * 10 + kind_rank, str(row["run_id"]))


def strict_row_from_summary(path: Path) -> dict[str, Any] | None:
    payload = load_json(path)
    for row in payload.get("matrix", []):
        if str(row.get("matrix_id")) == "M_DD" and str(row.get("variant")) == "strict":
            return row
    return None


def collect_rows(eval_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(eval_root.glob("*/summary.json")):
        strict = strict_row_from_summary(summary_path)
        if strict is None:
            continue
        eval_dir = summary_path.parent
        run_id = str(strict.get("checkpoint_run_id", eval_dir.name))
        resolved_config_path = DEFAULT_EXP_ROOT / run_id / "resolved_config.json"
        row = {
            "label": eval_dir.name,
            "run_id": run_id,
            "seed": seed_from_run(run_id),
            "kind": kind_from_run(run_id),
            "epoch": strict.get("checkpoint_epoch", "n_a"),
            "eval_protocol": "M_DD_strict",
            "bn_protocol": "strict_no_bn_recalib",
            "best_lod_d1": strict.get("checkpoint_best_lod_d1", float("nan")),
            "strict_d1": strict.get("d1", float("nan")),
            "AbsRel": strict.get("abs_rel", float("nan")),
            "RMSE": strict.get("rmse", float("nan")),
            "delta_ratio": strict.get("post_ram_delta_ratio", float("nan")),
            "clamp_ratio": "n_a",
            "viz_path": "n_a",
            "resolved_config_path": str(resolved_config_path) if resolved_config_path.is_file() else "n_a",
            "checkpoint_path": strict.get("checkpoint_path", "n_a"),
            "eval_output_dir": str(eval_dir),
            "per_sample_path": str(eval_dir / "per_sample" / "M_DD_strict.csv"),
        }
        rows.append(row)
    return sorted(rows, key=label_sort_key)


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        seed = str(row["seed"])
        if seed == "n_a":
            continue
        kind = str(row["kind"])
        bucket = "cleanup" if kind.startswith("T_") else "baseline" if kind.startswith("B0_") else "other"
        if bucket in {"baseline", "cleanup"}:
            by_seed.setdefault(seed, {})[bucket] = row

    result = []
    for seed, pair in sorted(by_seed.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 999999):
        if "baseline" not in pair or "cleanup" not in pair:
            continue
        baseline = pair["baseline"]
        cleanup = pair["cleanup"]
        result.append(
            {
                "seed": seed,
                "baseline_run": baseline["run_id"],
                "cleanup_run": cleanup["run_id"],
                "baseline_training_best": baseline["best_lod_d1"],
                "cleanup_training_best": cleanup["best_lod_d1"],
                "training_delta": finite_float(cleanup["best_lod_d1"]) - finite_float(baseline["best_lod_d1"]),
                "baseline_strict_d1": baseline["strict_d1"],
                "cleanup_strict_d1": cleanup["strict_d1"],
                "strict_delta": finite_float(cleanup["strict_d1"]) - finite_float(baseline["strict_d1"]),
                "cleanup_delta_ratio": cleanup["delta_ratio"],
            }
        )
    return result


def stage_a_noop_row(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = load_json(path)
    return {
        "d1": payload.get("d1", float("nan")),
        "abs_rel": payload.get("abs_rel", float("nan")),
        "rmse": payload.get("rmse", float("nan")),
        "path": str(path),
    }


def write_markdown(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    stage_a_noop: dict[str, Any] | None,
) -> None:
    cleanup_rows = [row for row in rows if str(row["kind"]).startswith("T_")]
    best_cleanup = max(cleanup_rows, key=lambda row: finite_float(row["strict_d1"]), default=None)
    lines = [
        "# LOD Post-RAM Stage B Strict Summary",
        "",
        "## Strict Rows",
        "",
        "| run_id | seed | kind | epoch | training_best | strict_D1 | AbsRel | RMSE | delta_ratio | resolved_config |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {seed} | {kind} | {epoch} | {best} | {strict} | {absrel} | {rmse} | {delta} | `{resolved}` |".format(
                run_id=row["run_id"],
                seed=row["seed"],
                kind=row["kind"],
                epoch=row["epoch"],
                best=fmt(row["best_lod_d1"]),
                strict=fmt(row["strict_d1"]),
                absrel=fmt(row["AbsRel"]),
                rmse=fmt(row["RMSE"]),
                delta=fmt(row["delta_ratio"]),
                resolved=row["resolved_config_path"],
            )
        )

    lines.extend(
        [
            "",
            "## Matched Seed Deltas",
            "",
            "| seed | training_delta | strict_delta | baseline_strict | cleanup_strict | cleanup_delta_ratio |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pairs:
        lines.append(
            "| {seed} | {train_delta} | {strict_delta} | {baseline} | {cleanup} | {delta_ratio} |".format(
                seed=row["seed"],
                train_delta=fmt(row["training_delta"]),
                strict_delta=fmt(row["strict_delta"]),
                baseline=fmt(row["baseline_strict_d1"]),
                cleanup=fmt(row["cleanup_strict_d1"]),
                delta_ratio=fmt(row["cleanup_delta_ratio"]),
            )
        )

    lines.extend(["", "## Decision Notes", ""])
    if stage_a_noop is not None:
        lines.append(
            "- Stage A strict no-op reference: D1 `{}`, AbsRel `{}`, RMSE `{}` from `{}`.".format(
                fmt(stage_a_noop["d1"]),
                fmt(stage_a_noop["abs_rel"]),
                fmt(stage_a_noop["rmse"]),
                stage_a_noop["path"],
            )
        )
    if best_cleanup is not None:
        lines.append(
            "- Best cleanup strict row: `{}` with strict D1 `{}` and training-best `{}`.".format(
                best_cleanup["run_id"],
                fmt(best_cleanup["strict_d1"]),
                fmt(best_cleanup["best_lod_d1"]),
            )
        )
    if pairs:
        strict_deltas = [finite_float(row["strict_delta"]) for row in pairs]
        strict_deltas = [value for value in strict_deltas if math.isfinite(value)]
        train_deltas = [finite_float(row["training_delta"]) for row in pairs]
        train_deltas = [value for value in train_deltas if math.isfinite(value)]
        if strict_deltas:
            lines.append(f"- Mean matched strict delta: `{sum(strict_deltas) / len(strict_deltas):.6f}` over `{len(strict_deltas)}` seeds.")
        if train_deltas:
            lines.append(f"- Mean matched training-best delta: `{sum(train_deltas) / len(train_deltas):.6f}` over `{len(train_deltas)}` seeds.")
    lines.append("- `viz_path` is `n_a`: this strict pass generated metrics/per-sample CSV only; Stage A external sanity generated post-RAM panels.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    eval_root = args.eval_root.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve() if args.output_csv else eval_root / "stageB_strict_summary.csv"
    output_md = args.output_md.expanduser().resolve() if args.output_md else eval_root / "summary.md"
    rows = collect_rows(eval_root)
    if not rows:
        raise FileNotFoundError(f"No strict summary.json rows found under {eval_root}")
    pairs = paired_rows(rows)
    stage_a_noop = stage_a_noop_row(args.stage_a_noop_metrics.expanduser().resolve() if args.stage_a_noop_metrics else None)
    write_csv(output_csv, rows)
    write_csv(eval_root / "stageB_paired_deltas.csv", pairs)
    write_markdown(output_md, rows=rows, pairs=pairs, stage_a_noop=stage_a_noop)
    print(f"[STAGEB_SUMMARY] wrote {output_md}")
    print(f"[STAGEB_SUMMARY] wrote {output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
