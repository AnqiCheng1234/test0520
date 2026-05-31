#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from foundation.tools.residual_training_common import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LED-HB formal eval metrics.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_path(payload: dict[str, Any], path: list[str]) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def make_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": path.parent.name,
        "metrics_path": str(path),
        "run_kind": payload.get("run_kind"),
        "max_depth": payload.get("max_depth"),
        "samples": payload.get("samples"),
        "feature_ablation_mode": payload.get("feature_ablation_mode", "not_applicable"),
        "feature_ablation_scope": payload.get("feature_ablation_scope", "not_applicable"),
        "final_abs_rel": get_path(payload, ["overall", "final", "abs_rel"]),
        "D1_abs_rel": get_path(payload, ["overall", "D1", "abs_rel"]),
        "D0_abs_rel": get_path(payload, ["overall", "D0", "abs_rel"]),
        "final_minus_D1_abs_rel": get_path(payload, ["overall", "delta_final_minus_D1", "abs_rel"]),
        "final_minus_D0_abs_rel": get_path(payload, ["overall", "delta_final_minus_D0", "abs_rel"]),
        "boundary_final_abs_rel": get_path(payload, ["region", "final", "boundary_abs_rel"]),
        "dark_q20_final_abs_rel": get_path(payload, ["region", "final", "dark_q20_abs_rel"]),
        "saturated_final_abs_rel": get_path(payload, ["region", "final", "saturated_abs_rel"]),
        "far50_final_abs_rel": get_path(payload, ["region", "final", "far50_abs_rel"]),
        "far100_final_abs_rel": get_path(payload, ["region", "final", "far100_abs_rel"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "run_kind",
        "max_depth",
        "samples",
        "feature_ablation_mode",
        "feature_ablation_scope",
        "final_abs_rel",
        "D1_abs_rel",
        "D0_abs_rel",
        "final_minus_D1_abs_rel",
        "final_minus_D0_abs_rel",
        "boundary_final_abs_rel",
        "dark_q20_final_abs_rel",
        "saturated_final_abs_rel",
        "far50_final_abs_rel",
        "far100_final_abs_rel",
        "metrics_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# LED-HB Formal Summary",
        "",
        "true nighttime RGB-D + inverse-ISP synthetic RAW-like packed Bayer",
        "",
        "## Overall Main [1,200]",
        "",
        "| name | kind | samples | final AbsRel | D1 AbsRel | D0 AbsRel | final-D1 | final-D0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if str(row.get("max_depth")) not in ("200.0", "200"):
            continue
        lines.append(
            f"| {row['name']} | {row.get('run_kind')} | {fmt(row.get('samples'))} | {fmt(row.get('final_abs_rel'))} | "
            f"{fmt(row.get('D1_abs_rel'))} | {fmt(row.get('D0_abs_rel'))} | {fmt(row.get('final_minus_D1_abs_rel'))} | {fmt(row.get('final_minus_D0_abs_rel'))} |"
        )
    lines.extend([
        "",
        "## Overall Secondary [1,80]",
        "",
        "| name | kind | samples | final AbsRel | D1 AbsRel | D0 AbsRel | final-D1 | final-D0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        if str(row.get("max_depth")) not in ("80.0", "80"):
            continue
        lines.append(
            f"| {row['name']} | {row.get('run_kind')} | {fmt(row.get('samples'))} | {fmt(row.get('final_abs_rel'))} | "
            f"{fmt(row.get('D1_abs_rel'))} | {fmt(row.get('D0_abs_rel'))} | {fmt(row.get('final_minus_D1_abs_rel'))} | {fmt(row.get('final_minus_D0_abs_rel'))} |"
        )
    lines.extend([
        "",
        "## Region Main [1,200]",
        "",
        "| name | boundary | dark_q20 | saturated | far50 | far100 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        if str(row.get("max_depth")) in ("200.0", "200"):
            lines.append(
                f"| {row['name']} | {fmt(row.get('boundary_final_abs_rel'))} | {fmt(row.get('dark_q20_final_abs_rel'))} | "
                f"{fmt(row.get('saturated_final_abs_rel'))} | {fmt(row.get('far50_final_abs_rel'))} | {fmt(row.get('far100_final_abs_rel'))} |"
            )
    lines.extend([
        "",
        "## X3 Ablation",
        "",
        "| name | mode | scope | max_depth | final AbsRel | final-D1 |",
        "|---|---|---|---:|---:|---:|",
    ])
    for row in rows:
        if row.get("feature_ablation_mode") not in (None, "not_applicable", "true"):
            lines.append(
                f"| {row['name']} | {row.get('feature_ablation_mode')} | {row.get('feature_ablation_scope')} | "
                f"{fmt(row.get('max_depth'))} | {fmt(row.get('final_abs_rel'))} | {fmt(row.get('final_minus_D1_abs_rel'))} |"
            )
    lines.extend(["", "## Panel Manifest", "", "Panel manifests are recorded separately as `panel_manifest.json` files under panel output directories.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    metrics_paths = sorted(input_root.rglob("metrics.json"))
    rows = [make_record(path, load_json(path)) for path in metrics_paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "led_hb_formal_summary.json", {"input_root": str(input_root), "records": rows})
    write_csv(output_dir / "led_hb_formal_records.csv", rows)
    write_md(output_dir / "led_hb_formal_summary.md", rows)
    print(json.dumps({"records": len(rows), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
