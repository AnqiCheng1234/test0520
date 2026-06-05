#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundation.tools.residual_training_common import save_json


RECORD_FIELDS = [
    "method_label",
    "dataset_index",
    "sample_name",
    "image_path",
    "depth_path",
    "panel_path",
    "c2_abs_rel",
    "method_abs_rel",
    "method_minus_c2_abs_rel",
    "method_better_than_c2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize improvement-over-C2 panel outputs.")
    parser.add_argument("--method-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def format_float(value: Any, digits: int = 6) -> str:
    out = finite_float(value)
    return "" if out is None else f"{out:.{digits}f}"


def normalize_record(record: dict[str, Any], *, method_label: str) -> dict[str, Any]:
    delta = finite_float(record.get("method_minus_c2_abs_rel"))
    better = bool(delta is not None and delta < 0.0)
    return {
        "method_label": str(record.get("method_label") or method_label),
        "dataset_index": int(record["dataset_index"]),
        "sample_name": str(record.get("sample_name", "")),
        "image_path": str(record.get("image_path", "")),
        "depth_path": str(record.get("depth_path", "")),
        "panel_path": str(record.get("panel_path", "")),
        "c2_abs_rel": finite_float(record.get("c2_abs_rel")),
        "method_abs_rel": finite_float(record.get("method_abs_rel")),
        "method_minus_c2_abs_rel": delta,
        "method_better_than_c2": better,
    }


def rows_from_method_dir(method_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = method_dir / "summary.json"
    manifest_path = method_dir / "manifest.json"
    if summary_path.is_file():
        payload = load_json(summary_path)
        summary = dict(payload.get("summary") or {})
        manifest = payload.get("manifest") or {}
        method_label = str(summary.get("method_label") or manifest.get("method_label") or method_dir.name)
        rows = [normalize_record(row, method_label=method_label) for row in payload.get("records", [])]
    elif manifest_path.is_file():
        manifest = load_json(manifest_path)
        method_label = str(manifest.get("method_label") or method_dir.name)
        rows = [normalize_record(row, method_label=method_label) for row in manifest.get("records", [])]
        summary = {}
    else:
        raise FileNotFoundError(f"Missing summary.json or manifest.json in {method_dir}")
    summary = summarize_rows(rows, method_label=method_label, output_dir=method_dir, base=summary)
    return summary, rows


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    method_label: str,
    output_dir: Path,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deltas = [float(row["method_minus_c2_abs_rel"]) for row in rows if finite_float(row.get("method_minus_c2_abs_rel")) is not None]
    sorted_by_delta = sorted(rows, key=lambda row: float(row["method_minus_c2_abs_rel"])) if rows else []
    summary = dict(base or {})
    summary.update(
        {
            "method_label": method_label,
            "num_panels": int(len(rows)),
            "num_method_better_than_c2": int(sum(1 for row in rows if bool(row.get("method_better_than_c2")))),
            "num_method_worse_than_c2": int(
                sum(1 for row in rows if finite_float(row.get("method_minus_c2_abs_rel")) is not None and float(row["method_minus_c2_abs_rel"]) > 0.0)
            ),
            "mean_method_minus_c2_abs_rel": statistics.fmean(deltas) if deltas else None,
            "median_method_minus_c2_abs_rel": statistics.median(deltas) if deltas else None,
            "best_sample_by_method_minus_c2": sorted_by_delta[0] if sorted_by_delta else None,
            "worst_sample_by_method_minus_c2": sorted_by_delta[-1] if sorted_by_delta else None,
            "output_dir": str(output_dir),
        }
    )
    return summary


def write_records_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in RECORD_FIELDS})


def write_summary_md(path: Path, summaries: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Improvement over C2 visualization summary\n\n")
        f.write("## Runs\n\n")
        f.write("| method | num panels | better | worse | mean method-C2 | median method-C2 | output |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for summary in summaries:
            f.write(
                f"| {summary.get('method_label')} | {summary.get('num_panels')} | "
                f"{summary.get('num_method_better_than_c2')} | {summary.get('num_method_worse_than_c2')} | "
                f"{format_float(summary.get('mean_method_minus_c2_abs_rel'))} | "
                f"{format_float(summary.get('median_method_minus_c2_abs_rel'))} | {summary.get('output_dir')} |\n"
            )
        f.write("\n## Per-sample records\n\n")
        f.write("| method | idx | sample | C2 abs_rel | method abs_rel | method-C2 | panel |\n")
        f.write("|---|---:|---|---:|---:|---:|---|\n")
        for row in records:
            f.write(
                f"| {row.get('method_label')} | {row.get('dataset_index')} | {row.get('sample_name')} | "
                f"{format_float(row.get('c2_abs_rel'))} | {format_float(row.get('method_abs_rel'))} | "
                f"{format_float(row.get('method_minus_c2_abs_rel'))} | {row.get('panel_path')} |\n"
            )
        f.write("\n## Interpretation notes\n\n")
        f.write("Negative method-C2 means method is better than C2.\n")
        f.write("Positive method-C2 means method is worse than C2.\n")
        f.write("This is a visualization/diagnostic summary, not a replacement for full validation metrics.\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    method_dirs = [Path(path).expanduser().resolve() for path in args.method_dirs]
    summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for method_dir in method_dirs:
        summary, rows = rows_from_method_dir(method_dir)
        summaries.append(summary)
        records.extend(rows)
    payload = {"runs": summaries, "records": records}
    json_path = output_dir / "improvement_vs_c2_summary.json"
    md_path = output_dir / "improvement_vs_c2_summary.md"
    csv_path = output_dir / "improvement_vs_c2_records.csv"
    save_json(json_path, payload)
    write_records_csv(csv_path, records)
    write_summary_md(md_path, summaries, records)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
