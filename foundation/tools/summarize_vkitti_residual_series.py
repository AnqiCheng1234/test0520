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
    "experiment_id",
    "residual_feature_source",
    "input_domain",
    "model_input_tensor",
    "raw_storage_format",
    "front_end",
    "dataset_geometry_mode",
    "fullres_even_policy",
    "rgb_input_space",
    "depth_target_space",
    "input_height",
    "input_width",
    "source_original_hw",
    "even_fullres_hw",
    "packed_hw",
    "trainable_params",
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
    parser = argparse.ArgumentParser(description="Summarize VKITTI M-series vs C-series residual runs.")
    parser.add_argument("--m2-run", required=True)
    parser.add_argument("--c1-run", required=True)
    parser.add_argument("--c2-run", required=True)
    parser.add_argument("--m1-run", default=None)
    parser.add_argument("--m3-run", default=None)
    parser.add_argument("--output-dir", default="plans/0524_new")
    parser.add_argument("--timestamp", default=None, help="Output timestamp. Defaults to timestamp parsed from required runs.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_method(run_dir: Path, config: dict[str, Any]) -> str:
    exp_id = config.get("experiment_id")
    if exp_id in ("C1", "C2", "M1", "M2", "M3"):
        return str(exp_id)
    feature_source = str(config.get("residual_feature_source", "unknown"))
    name = run_dir.name.lower()
    if "m1" in name or feature_source == "x3":
        return "M1"
    if "m3" in name or feature_source == "x3_ffm_mid":
        return "M3"
    if "m2" in name or feature_source == "ffm_mid":
        return "M2"
    return "unknown"


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


def parse_trainable_params(run_dir: Path, config: dict[str, Any]) -> Any:
    counts = config.get("model_param_counts")
    if isinstance(counts, dict) and counts.get("trainable_params") is not None:
        return counts["trainable_params"]
    if config.get("trainable_params") is not None:
        return config["trainable_params"]
    train_log = run_dir / "train.log"
    if train_log.is_file():
        text = train_log.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"trainable_params=(\d+)", text)
        if matches:
            return int(matches[-1])
    return "n/a"


def row_from_run(run_dir: Path) -> dict[str, Any]:
    config = load_json(run_dir / "config.json")
    val_payload = load_json(run_dir / "val_metrics.json")
    best = find_best_epoch(val_payload)
    method = infer_method(run_dir, config)
    overall = best["overall"]
    region = best["region"]
    diagnostics = best.get("diagnostics", {})
    row = {
        "run_dir": str(run_dir),
        "method": method,
        "experiment_id": config.get("experiment_id", method),
        "residual_feature_source": config.get("residual_feature_source", "n/a"),
        "input_domain": config.get("input_domain", "n/a"),
        "model_input_tensor": config.get("model_input_tensor", "n/a"),
        "raw_storage_format": config.get("raw_storage_format", "n/a"),
        "front_end": config.get("front_end", "n/a"),
        "dataset_geometry_mode": config.get("dataset_geometry_mode", "n/a"),
        "fullres_even_policy": config.get("fullres_even_policy", "n/a"),
        "rgb_input_space": config.get("rgb_input_space", "n/a"),
        "depth_target_space": config.get("depth_target_space", "n/a"),
        "input_height": config.get("input_height", "n/a"),
        "input_width": config.get("input_width", "n/a"),
        "source_original_hw": get_dataset_geometry_field(config, "source_original_hw"),
        "even_fullres_hw": get_dataset_geometry_field(config, "even_fullres_hw"),
        "packed_hw": get_dataset_geometry_field(config, "packed_hw"),
        "trainable_params": parse_trainable_params(run_dir, config),
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
        "_config": config,
        "_best": best,
    }
    return row


def normalize_path(path: Any) -> str:
    return str(Path(path).expanduser().resolve()) if path not in (None, "n/a") else "n/a"


def freeze_for_set(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else str(value)


def vkitti_eval_protocol(row: dict[str, Any]) -> Any:
    protocol = row["_config"].get(
        "eval_protocol",
        row["_best"].get("alignment_protocol", "per_image_affine_disp_depth_anything_v2"),
    )
    if isinstance(protocol, dict):
        return protocol.get("vkitti_val", protocol.get("val", protocol))
    return protocol


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No runs to validate.")
    for row in rows:
        if (int(row["input_height"]), int(row["input_width"])) != (187, 621):
            raise ValueError(f"Refusing to summarize non-halfres run {row['run_dir']}: input={row['input_height']}x{row['input_width']}")

    def unique_config_values(key: str) -> set[Any]:
        values: set[Any] = set()
        for row in rows:
            value = row["_config"].get(key, "n/a")
            if isinstance(value, list):
                value = tuple(value)
            values.add(value)
        return values

    for key in ("encoder", "fullres_even_policy", "rgb_input_space", "depth_target_space", "min_depth", "max_depth"):
        values = unique_config_values(key)
        if len(values) != 1:
            raise ValueError(f"Run mismatch for {key}: {values}")

    train_lists = {normalize_path(row["_config"].get("vkitti_train_list")) for row in rows}
    val_lists = {normalize_path(row["_config"].get("vkitti_val_list")) for row in rows}
    if len(train_lists) != 1:
        raise ValueError(f"Run mismatch for vkitti_train_list: {train_lists}")
    if len(val_lists) != 1:
        raise ValueError(f"Run mismatch for vkitti_val_list: {val_lists}")

    protocols = {freeze_for_set(vkitti_eval_protocol(row)) for row in rows}
    if len(protocols) != 1:
        raise ValueError(f"Run mismatch for eval protocol: {protocols}")


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
        "# VKITTI Residual M vs C Summary",
        "",
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(SUMMARY_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in SUMMARY_COLUMNS) + " |")
    lines.append("")
    return "\n".join(lines)


def infer_timestamp(paths: list[Path]) -> str:
    for path in paths:
        match = re.match(r"^(\d{4}_\d{4})", path.name)
        if match:
            return match.group(1)
    return "unknown_time"


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def main() -> None:
    args = parse_args()
    required = {
        "M2": Path(args.m2_run).expanduser().resolve(),
        "C1": Path(args.c1_run).expanduser().resolve(),
        "C2": Path(args.c2_run).expanduser().resolve(),
    }
    optional = {
        "M1": Path(args.m1_run).expanduser().resolve() if args.m1_run else None,
        "M3": Path(args.m3_run).expanduser().resolve() if args.m3_run else None,
    }
    missing_optional: list[str] = []
    run_dirs: list[Path] = []
    for label, path in required.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing required {label} run: {path}")
        run_dirs.append(path)
    for label, path in optional.items():
        if path is None:
            missing_optional.append(label)
            continue
        if not path.is_dir():
            missing_optional.append(label)
            continue
        run_dirs.append(path)

    rows = [row_from_run(path) for path in run_dirs]
    validate_rows(rows)
    order = {"M2": 0, "M1": 1, "M3": 2, "C2": 3, "C1": 4}
    rows.sort(key=lambda row: order.get(str(row["method"]), 99))

    timestamp = args.timestamp or infer_timestamp(run_dirs)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"vkitti_residual_m_vs_c_summary_{timestamp}.json"
    md_path = output_dir / f"vkitti_residual_m_vs_c_summary_{timestamp}.md"
    csv_path = output_dir / f"vkitti_residual_m_vs_c_summary_{timestamp}.csv"
    clean_rows = [clean_row(row) for row in rows]
    payload = {
        "timestamp": timestamp,
        "runs": [str(path) for path in run_dirs],
        "missing_optional_runs": missing_optional,
        "rows": clean_rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    md_path.write_text(markdown_table(clean_rows), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("run_dir", *SUMMARY_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in clean_rows:
            writer.writerow({key: fmt(row.get(key)) for key in ("run_dir", *SUMMARY_COLUMNS)})
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
