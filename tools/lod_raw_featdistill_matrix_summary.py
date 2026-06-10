#!/usr/bin/env python3
"""Summarize LOD noise-aware feature distillation runs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_M_DD = 0.845279
DEFAULT_M_NN = 0.898946


EVAL_RE = re.compile(
    r"\[EVAL\]\[lod_val\] done epoch=(?P<epoch>\d+) .*?"
    r"abs_rel=(?P<abs_rel>[0-9.]+) rmse=(?P<rmse>[0-9.]+).*? d1=(?P<d1>[0-9.]+)"
)
BEST_RE = re.compile(r"\[CHECKPOINT\] saved best=.*?value=(?P<value>[0-9.]+) epoch=(?P<epoch>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--run-glob", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-probe-name", default="feature_probe_summary.json")
    parser.add_argument("--m-dd", type=float, default=DEFAULT_M_DD, help="Baseline dark-on-dark D1 for recovery ratio.")
    parser.add_argument("--m-nn", type=float, default=DEFAULT_M_NN, help="Oracle normal-on-normal D1 for recovery ratio.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_run_dirs(args: argparse.Namespace) -> list[Path]:
    runs = [path.expanduser().resolve() for path in args.run_dir]
    for pattern in args.run_glob:
        runs.extend(Path(path).expanduser().resolve() for path in sorted(glob.glob(str(Path(pattern).expanduser()))))
    unique: list[Path] = []
    seen: set[Path] = set()
    for run in runs:
        if run in seen:
            continue
        seen.add(run)
        unique.append(run)
    if not unique:
        raise ValueError("Provide at least one --run-dir or --run-glob")
    return unique


def infer_group(run_name: str) -> str:
    for group in ("L0_seed123", "L0", "L1", "L2", "L3", "W0_seed123", "W0", "W1", "W2", "W3"):
        if f"_{group}_" in f"_{run_name}_":
            return group
    return "unknown"


def parse_log_metrics(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {"best_epoch": "", "d1": float("nan"), "abs_rel": float("nan"), "rmse": float("nan")}
    latest_eval: dict[str, Any] | None = None
    best_from_ckpt: dict[str, Any] | None = None
    eval_by_epoch: dict[int, dict[str, Any]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        eval_match = EVAL_RE.search(line)
        if eval_match:
            latest_eval = {
                "best_epoch": int(eval_match.group("epoch")),
                "d1": float(eval_match.group("d1")),
                "abs_rel": float(eval_match.group("abs_rel")),
                "rmse": float(eval_match.group("rmse")),
            }
            eval_by_epoch[latest_eval["best_epoch"]] = latest_eval
        best_match = BEST_RE.search(line)
        if best_match:
            best_from_ckpt = {
                "best_epoch": int(best_match.group("epoch")),
                "d1": float(best_match.group("value")),
            }
    if latest_eval is None:
        latest_eval = {"best_epoch": "", "d1": float("nan"), "abs_rel": float("nan"), "rmse": float("nan")}
    if best_from_ckpt is not None and math.isfinite(best_from_ckpt["d1"]):
        best_eval = eval_by_epoch.get(best_from_ckpt["best_epoch"])
        if best_eval is not None:
            latest_eval = dict(best_eval)
        latest_eval["best_epoch"] = best_from_ckpt["best_epoch"]
        latest_eval["d1"] = best_from_ckpt["d1"]
    return latest_eval


def read_feature_probe(run_dir: Path, feature_probe_name: str) -> dict[str, float]:
    candidates = [
        run_dir / feature_probe_name,
        run_dir / "feature_probe" / feature_probe_name,
        run_dir / "analysis" / feature_probe_name,
    ]
    for path in candidates:
        if not path.is_file():
            continue
        payload = load_json(path)
        return {
            "L5 dist": float(payload.get("layer5_token_cosine_distance_mean", float("nan"))),
            "L8 dist": float(payload.get("layer8_token_cosine_distance_mean", float("nan"))),
            "L11 dist": float(payload.get("layer11_token_cosine_distance_mean", float("nan"))),
        }
    return {"L5 dist": float("nan"), "L8 dist": float("nan"), "L11 dist": float("nan")}


def row_for_run(run_dir: Path, feature_probe_name: str, m_dd: float, gap: float) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    resolved_path = run_dir / "resolved_config.json"
    config = load_json(config_path) if config_path.is_file() else {}
    resolved = config.get("resolved_config") or (load_json(resolved_path) if resolved_path.is_file() else {})
    metrics = parse_log_metrics(run_dir / "train.log")
    probe = read_feature_probe(run_dir, feature_probe_name)
    d1 = float(metrics["d1"])
    return {
        "group": infer_group(run_dir.name),
        "run": run_dir.name,
        "local branch": str(resolved.get("raw_ram_local_residual", "unknown")),
        "feature loss": str(resolved.get("feat_distill", "unknown")),
        "lambda_feat": resolved.get("feat_distill_lambda", ""),
        "lora": str(resolved.get("lora", "unknown")),
        "D1": d1,
        "AbsRel": float(metrics["abs_rel"]),
        "RMSE": float(metrics["rmse"]),
        "best epoch": metrics["best_epoch"],
        "ckpt": str(Path(config.get("heavy_save_path", "")) / "best_model.pth") if config.get("heavy_save_path") else "",
        "delta_d1_vs_M_DD": d1 - m_dd if math.isfinite(d1) else float("nan"),
        "recover ratio": (d1 - m_dd) / gap if math.isfinite(d1) and gap != 0 else float("nan"),
        **probe,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.6f}"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = ["group", "local branch", "feature loss", "lora", "D1", "delta_d1_vs_M_DD", "recover ratio", "L5 dist", "L8 dist", "L11 dist", "run"]
    lines = ["# LOD RAW Noise-aware Feature Distillation Summary", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    gap = args.m_nn - args.m_dd
    run_dirs = collect_run_dirs(args)
    rows = [row_for_run(run, args.feature_probe_name, args.m_dd, gap) for run in run_dirs]
    rows.sort(key=lambda row: (str(row["group"]), str(row["run"])))
    output_dir = args.output_dir.expanduser().resolve()
    write_csv(output_dir / "matrix.csv", rows)
    recovery_rows = [
        {
            "group": row["group"],
            "run": row["run"],
            "D1": row["D1"],
            "M_DD": args.m_dd,
            "M_NN": args.m_nn,
            "gap": gap,
            "recover_ratio": row["recover ratio"],
        }
        for row in rows
    ]
    write_csv(output_dir / "recovery.csv", recovery_rows)
    write_markdown(output_dir / "ablation_summary.md", rows)
    print(f"[SUMMARY] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
