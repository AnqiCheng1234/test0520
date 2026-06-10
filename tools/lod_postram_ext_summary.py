#!/usr/bin/env python3
"""Summarize post-RAM external eval directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--baseline-ext-id", default="EXT_NOOP_A0")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


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


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    args = parse_args()
    rows = []
    for eval_dir in args.eval_dirs:
        metrics_path = eval_dir.expanduser().resolve() / "metrics.json"
        metrics = load_json(metrics_path)
        rows.append(
            {
                "ext_id": metrics.get("ext_id"),
                "denoiser": metrics.get("denoiser"),
                "sigma": metrics.get("sigma"),
                "alpha": metrics.get("alpha"),
                "affine": metrics.get("affine"),
                "d1": metrics.get("d1"),
                "abs_rel": metrics.get("abs_rel"),
                "rmse": metrics.get("rmse"),
                "clamp_total_ratio": metrics.get("clamp_total_ratio"),
                "delta_ratio": metrics.get("delta_ratio"),
                "samples": metrics.get("samples"),
                "artifact_count": metrics.get("artifact_count"),
                "eval_dir": str(eval_dir.expanduser().resolve()),
                "viz_manifest": str(eval_dir.expanduser().resolve() / "viz" / str(metrics.get("ext_id")) / "manifest.json"),
            }
        )
    baseline = next((row for row in rows if str(row.get("ext_id")) == args.baseline_ext_id), None)
    baseline_d1 = float(baseline["d1"]) if baseline and baseline.get("d1") is not None else None
    for row in rows:
        if baseline_d1 is not None:
            row["delta_d1_vs_baseline"] = float(row["d1"]) - baseline_d1
    write_csv(args.output_csv.expanduser().resolve(), rows)

    lines = [
        "# Post-RAM External Eval Summary",
        "",
        f"- baseline_ext_id: `{args.baseline_ext_id}`",
        "",
        "| EXT | denoiser | sigma | alpha | affine | D1 | delta D1 | AbsRel | RMSE | clamp | delta_ratio | samples | viz |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {ext_id} | {denoiser} | {sigma} | {alpha} | {affine} | {d1} | {delta} | "
            "{abs_rel} | {rmse} | {clamp} | {delta_ratio} | {samples} | `{viz}` |".format(
                ext_id=row.get("ext_id"),
                denoiser=row.get("denoiser"),
                sigma=row.get("sigma"),
                alpha=row.get("alpha"),
                affine=row.get("affine"),
                d1=fmt(row.get("d1")),
                delta=fmt(row.get("delta_d1_vs_baseline", "n_a")),
                abs_rel=fmt(row.get("abs_rel")),
                rmse=fmt(row.get("rmse")),
                clamp=fmt(row.get("clamp_total_ratio")),
                delta_ratio=fmt(row.get("delta_ratio")),
                samples=row.get("samples"),
                viz=row.get("viz_manifest"),
            )
        )
    args.output_md.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_md.expanduser().resolve().write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[POSTRAM_SUMMARY] wrote {args.output_md.expanduser().resolve()}", flush=True)


if __name__ == "__main__":
    main()
