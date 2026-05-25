from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SUMMARY_COLUMNS = (
    "method",
    "feature_source",
    "raw_storage_format",
    "fullres_even_policy",
    "rgb_input_space",
    "depth_target_space",
    "input_height",
    "input_width",
    "source_original_hw",
    "even_fullres_hw",
    "packed_hw",
    "best_epoch",
    "D0_abs_rel",
    "final_abs_rel",
    "delta_abs_rel",
    "D0_d1",
    "final_d1",
    "delta_d1",
    "boundary_abs_rel",
    "high_error_abs_rel",
    "far50_abs_rel",
    "dark_abs_rel",
    "saturated_abs_rel",
    "mean_gate",
    "mean_abs_delta",
    "mean_abs_gate_delta",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize VKITTI M-series residual runs.")
    parser.add_argument("runs", nargs="+", help="Run directories under finetune_stf/exp.")
    parser.add_argument("--output-dir", default="plans/0524_new")
    parser.add_argument("--timestamp", default=None, help="Output timestamp. Defaults to timestamp parsed from first run name.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_method(run_dir: Path, feature_source: str) -> str:
    name = run_dir.name.lower()
    if "m1" in name or feature_source == "x3":
        return "M1"
    if "m3" in name or feature_source == "x3_ffm_mid":
        return "M3"
    return "M2"


def find_best_epoch(val_payload: dict[str, Any]) -> dict[str, Any]:
    epochs = val_payload.get("epochs")
    if not epochs:
        latest = val_payload.get("latest")
        if isinstance(latest, dict):
            return latest
        raise ValueError("val_metrics.json has neither epochs nor latest.")
    valid = [
        item
        for item in epochs
        if item.get("overall", {}).get("final", {}).get("abs_rel") is not None
    ]
    if not valid:
        raise ValueError("No epoch with final abs_rel found.")
    return min(valid, key=lambda item: float(item["overall"]["final"]["abs_rel"]))


def get_dataset_geometry_field(config: dict[str, Any], field: str) -> Any:
    geometry = config.get("dataset_geometry", {})
    if isinstance(geometry, dict):
        if isinstance(geometry.get("val"), dict) and field in geometry["val"]:
            return geometry["val"][field]
        if isinstance(geometry.get("train"), dict) and field in geometry["train"]:
            return geometry["train"][field]
        if field in geometry:
            return geometry[field]
    if field == "source_original_hw":
        return get_dataset_geometry_field(config, "original_hw")
    return "n/a"


def row_from_run(run_dir: Path) -> dict[str, Any]:
    config = load_json(run_dir / "config.json")
    val_payload = load_json(run_dir / "val_metrics.json")
    best = find_best_epoch(val_payload)
    feature_source = str(config.get("residual_feature_source", "unknown"))
    overall = best["overall"]
    region = best["region"]
    diagnostics = best.get("diagnostics", {})
    row = {
        "run_dir": str(run_dir),
        "method": infer_method(run_dir, feature_source),
        "feature_source": feature_source,
        "raw_storage_format": config.get("raw_storage_format", "n/a"),
        "fullres_even_policy": config.get("fullres_even_policy", "n/a"),
        "rgb_input_space": config.get("rgb_input_space", "n/a"),
        "depth_target_space": config.get("depth_target_space", "n/a"),
        "input_height": config.get("input_height", "n/a"),
        "input_width": config.get("input_width", "n/a"),
        "source_original_hw": get_dataset_geometry_field(config, "source_original_hw"),
        "even_fullres_hw": get_dataset_geometry_field(config, "even_fullres_hw"),
        "packed_hw": get_dataset_geometry_field(config, "packed_hw"),
        "best_epoch": int(best["epoch"]),
        "D0_abs_rel": overall["D0"].get("abs_rel"),
        "final_abs_rel": overall["final"].get("abs_rel"),
        "delta_abs_rel": overall["delta"].get("final_abs_rel_minus_D0_abs_rel"),
        "D0_d1": overall["D0"].get("d1"),
        "final_d1": overall["final"].get("d1"),
        "delta_d1": overall["delta"].get("final_d1_minus_D0_d1"),
        "boundary_abs_rel": region["final"].get("boundary_abs_rel"),
        "high_error_abs_rel": region["final"].get("dav2_high_error_abs_rel"),
        "far50_abs_rel": region["final"].get("far50_abs_rel"),
        "dark_abs_rel": region["final"].get("dark_abs_rel"),
        "saturated_abs_rel": region["final"].get("saturated_abs_rel"),
        "mean_gate": diagnostics.get("mean_gate"),
        "mean_abs_delta": diagnostics.get("mean_abs_delta"),
        "mean_abs_gate_delta": diagnostics.get("mean_abs_gate_delta"),
    }
    return row


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# VKITTI M-series Summary",
        "",
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(SUMMARY_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in SUMMARY_COLUMNS) + " |")
    lines.append("")
    return "\n".join(lines)


def infer_timestamp(first_run: Path) -> str:
    match = re.match(r"^(\d{4}_\d{4})", first_run.name)
    if match:
        return match.group(1)
    return "unknown_time"


def main() -> None:
    args = parse_args()
    run_dirs = [Path(path).expanduser().resolve() for path in args.runs]
    rows = [row_from_run(path) for path in run_dirs]
    rows.sort(key=lambda row: {"M2": 0, "M1": 1, "M3": 2}.get(str(row["method"]), 99))
    timestamp = args.timestamp or infer_timestamp(run_dirs[0])
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"vkitti_mseries_summary_{timestamp}.json"
    md_path = output_dir / f"vkitti_mseries_summary_{timestamp}.md"
    csv_path = output_dir / f"vkitti_mseries_summary_{timestamp}.csv"
    payload = {"timestamp": timestamp, "runs": [str(path) for path in run_dirs], "rows": rows}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    md_path.write_text(markdown_table(rows), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("run_dir", *SUMMARY_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in ("run_dir", *SUMMARY_COLUMNS)})
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
