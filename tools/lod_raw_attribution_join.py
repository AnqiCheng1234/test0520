#!/usr/bin/env python3
"""Join LOD RAW audit stats with per-sample eval metrics and report correlations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - fallback for lean envs
    scipy_stats = None


DEPENDENT_VARIABLES = ("delta_d1", "one_minus_d1_raw_dark", "d1_raw_dark", "d1_raw_normal")
ATTRIBUTION_VARIABLES = (
    "ev_gap_luma",
    "gain_p95_luma",
    "dark_luma_p99",
    "normal_luma_p99",
    "dark_luma_dynamic_range_98",
    "noise_sigma_ratio_dark_over_normal",
    "noise_sigma_dark_luma_flat",
    "snr_ratio_lowbin",
    "snr_ratio_midbin",
    "snr_ratio_highbin",
    "edge_recall",
    "edge_f1",
    "gradient_corr",
    "gradient_ratio",
    "chromaticity_delta_l1",
    "channel_gain_range_ratio",
    "channel_gain_ratio_R_over_G",
    "channel_gain_ratio_B_over_G",
    "linear_residual_sigma",
    "gamma_residual_sigma",
    "gamma_fit_r2",
    "phase_corr_shift_abs_max",
)
SCATTERS = (
    ("delta_d1", "ev_gap_luma", "delta_d1_vs_ev_gap_luma.png"),
    ("delta_d1", "snr_ratio_lowbin", "delta_d1_vs_snr_ratio_lowbin.png"),
    ("delta_d1", "edge_recall", "delta_d1_vs_edge_recall.png"),
    ("delta_d1", "chromaticity_delta_l1", "delta_d1_vs_chromaticity_delta.png"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-stats-csv", type=Path, required=True)
    parser.add_argument("--dark-eval-csv", type=Path, required=True)
    parser.add_argument("--normal-eval-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.expanduser().open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def index_by_pair(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id") or row.get("sample_id") or "")
        if not pair_id:
            raise ValueError(f"{label} row missing pair_id/sample_id")
        if pair_id in out:
            raise ValueError(f"{label} duplicate pair_id={pair_id}")
        out[pair_id] = row
    return out


def prefixed(row: dict[str, Any], prefix: str, skip: set[str]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in row.items() if key not in skip}


def join_rows(raw_rows: list[dict[str, Any]], dark_rows: list[dict[str, Any]], normal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dark_by_pair = index_by_pair(dark_rows, label="dark_eval")
    normal_by_pair = index_by_pair(normal_rows, label="normal_eval")
    joined: list[dict[str, Any]] = []
    for raw in raw_rows:
        if str(raw.get("split")) != "01Valid":
            continue
        pair_id = str(raw.get("pair_id"))
        if pair_id not in dark_by_pair or pair_id not in normal_by_pair:
            continue
        dark = dark_by_pair[pair_id]
        normal = normal_by_pair[pair_id]
        d1_dark = safe_float(dark.get("d1"))
        d1_normal = safe_float(normal.get("d1"))
        row: dict[str, Any] = dict(raw)
        row.update(prefixed(dark, "raw_dark_eval", skip={"pair_id", "sample_id", "split"}))
        row.update(prefixed(normal, "raw_normal_eval", skip={"pair_id", "sample_id", "split"}))
        row["d1_raw_dark"] = d1_dark
        row["d1_raw_normal"] = d1_normal
        row["one_minus_d1_raw_dark"] = float(1.0 - d1_dark) if math.isfinite(d1_dark) else float("nan")
        row["delta_d1"] = float(d1_normal - d1_dark) if math.isfinite(d1_dark) and math.isfinite(d1_normal) else float("nan")
        row["abs_rel_raw_dark"] = safe_float(dark.get("abs_rel"))
        row["abs_rel_raw_normal"] = safe_float(normal.get("abs_rel"))
        row["edge_sobel_l1_raw_dark"] = safe_float(dark.get("edge_sobel_l1"))
        row["edge_sobel_l1_raw_normal"] = safe_float(normal.get("edge_sobel_l1"))
        row["edge_overlap_iou_raw_dark"] = safe_float(dark.get("edge_overlap_iou"))
        row["edge_overlap_iou_raw_normal"] = safe_float(normal.get("edge_overlap_iou"))
        joined.append(row)
    return joined


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


def arrays_for(rows: list[dict[str, Any]], x_key: str, y_key: str, *, exclude_registration_outliers: bool = True) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        if exclude_registration_outliers and safe_float(row.get("registration_outlier")) >= 0.5:
            continue
        x = safe_float(row.get(x_key))
        y = safe_float(row.get(y_key))
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def corr_pair(x: np.ndarray, y: np.ndarray, *, method: str) -> tuple[float, float]:
    if x.size < 3 or y.size < 3:
        return float("nan"), float("nan")
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan"), float("nan")
    if scipy_stats is not None:
        if method == "spearman":
            stat = scipy_stats.spearmanr(x, y)
        elif method == "pearson":
            stat = scipy_stats.pearsonr(x, y)
        else:
            raise ValueError(method)
        return float(stat.statistic), float(stat.pvalue)
    if method == "spearman":
        x = rankdata(x)
        y = rankdata(y)
    r = float(np.corrcoef(x, y)[0, 1])
    return r, float("nan")


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    sorted_vals = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_vals[end] == sorted_vals[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = float(np.mean(np.arange(start, end, dtype=np.float64)))
        start = end
    return ranks


def residualize(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    design = np.stack([z, np.ones_like(z)], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def partial_spearman(rows: list[dict[str, Any]], x_key: str, y_key: str, control_key: str) -> dict[str, Any]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for row in rows:
        if safe_float(row.get("registration_outlier")) >= 0.5:
            continue
        x = safe_float(row.get(x_key))
        y = safe_float(row.get(y_key))
        z = safe_float(row.get(control_key))
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            xs.append(x)
            ys.append(y)
            zs.append(z)
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    z = np.asarray(zs, dtype=np.float64)
    if x.size < 4:
        return {"x": x_key, "y": y_key, "control": control_key, "n": int(x.size), "partial_spearman_r": float("nan"), "p": float("nan")}
    xr = rankdata(x)
    yr = rankdata(y)
    zr = rankdata(z)
    x_res = residualize(xr, zr)
    y_res = residualize(yr, zr)
    r, p = corr_pair(x_res, y_res, method="pearson")
    return {"x": x_key, "y": y_key, "control": control_key, "n": int(x.size), "partial_spearman_r": r, "p": p}


def build_correlation_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correlations: list[dict[str, Any]] = []
    for y_key in DEPENDENT_VARIABLES:
        for x_key in ATTRIBUTION_VARIABLES:
            x, y = arrays_for(rows, x_key, y_key)
            spearman_r, spearman_p = corr_pair(x, y, method="spearman")
            pearson_r, pearson_p = corr_pair(x, y, method="pearson")
            correlations.append(
                {
                    "dependent": y_key,
                    "variable": x_key,
                    "n": int(x.size),
                    "spearman_rho": spearman_r,
                    "spearman_p": spearman_p,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                }
            )
    partials = [
        partial_spearman(rows, "snr_ratio_lowbin", "delta_d1", "ev_gap_luma"),
        partial_spearman(rows, "edge_recall", "delta_d1", "ev_gap_luma"),
        partial_spearman(rows, "chromaticity_delta_l1", "delta_d1", "ev_gap_luma"),
        partial_spearman(rows, "noise_sigma_ratio_dark_over_normal", "delta_d1", "ev_gap_luma"),
    ]
    deltas = np.asarray([safe_float(row.get("delta_d1")) for row in rows], dtype=np.float64)
    deltas = deltas[np.isfinite(deltas)]
    return {
        "rows": len(rows),
        "registration_outliers_excluded_from_correlations": int(sum(safe_float(row.get("registration_outlier")) >= 0.5 for row in rows)),
        "delta_d1_distribution": distribution(deltas),
        "delta_d1_positive_count": int(np.count_nonzero(deltas > 0)) if deltas.size else 0,
        "correlations": correlations,
        "partial_correlations": partials,
    }


def distribution(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def fmt(value: Any, digits: int = 4) -> str:
    value = safe_float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# LOD RAW Attribution Correlation Report",
        "",
        "Scope: `01Valid` samples joined by `pair_id`. Registration outliers are excluded from correlations.",
        "",
        "## Delta D1 Distribution",
        "",
    ]
    dist = report["delta_d1_distribution"]
    lines.append(
        "count={count}, positive={pos}, mean={mean}, p50={p50}, p90={p90}, min={minv}, max={maxv}".format(
            count=dist.get("count", 0),
            pos=report.get("delta_d1_positive_count", 0),
            mean=fmt(dist.get("mean")),
            p50=fmt(dist.get("p50")),
            p90=fmt(dist.get("p90")),
            minv=fmt(dist.get("min")),
            maxv=fmt(dist.get("max")),
        )
    )
    lines.extend(["", "## Spearman With `delta_d1`", ""])
    delta_rows = [
        row for row in report["correlations"] if row["dependent"] == "delta_d1"
    ]
    delta_rows = sorted(delta_rows, key=lambda row: abs(safe_float(row["spearman_rho"])), reverse=True)
    lines.append("| variable | n | Spearman rho | p-value | Pearson r |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in delta_rows:
        lines.append(
            f"| `{row['variable']}` | {row['n']} | {fmt(row['spearman_rho'])} | {fmt(row['spearman_p'])} | {fmt(row['pearson_r'])} |"
        )
    lines.extend(["", "## Auxiliary Dependent Variables", ""])
    for dep in ("one_minus_d1_raw_dark", "d1_raw_dark", "d1_raw_normal"):
        lines.append(f"### {dep}")
        lines.append("")
        lines.append("| variable | n | Spearman rho | p-value | Pearson r |")
        lines.append("|---|---:|---:|---:|---:|")
        dep_rows = [row for row in report["correlations"] if row["dependent"] == dep]
        dep_rows = sorted(dep_rows, key=lambda row: abs(safe_float(row["spearman_rho"])), reverse=True)
        for row in dep_rows[:12]:
            lines.append(
                f"| `{row['variable']}` | {row['n']} | {fmt(row['spearman_rho'])} | {fmt(row['spearman_p'])} | {fmt(row['pearson_r'])} |"
            )
        lines.append("")
    lines.extend(["## Partial Spearman Controlling `ev_gap_luma`", ""])
    lines.append("| x | y | control | n | partial rho | p-value |")
    lines.append("|---|---|---|---:|---:|---:|")
    for row in report["partial_correlations"]:
        lines.append(
            f"| `{row['x']}` | `{row['y']}` | `{row['control']}` | {row['n']} | "
            f"{fmt(row['partial_spearman_r'])} | {fmt(row['p'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Val size is 112, so single-variable correlations are screening evidence, not final attribution.",
            "- `ev_gap_luma`, `p99`, dynamic range, SNR, edge and color metrics are expected to be collinear.",
            "- `delta_d1 = d1_raw_normal - d1_raw_dark` mixes two independently trained checkpoints; auxiliary dependent variables are reported to check direction consistency.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_scatter(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str) -> None:
    x, y = arrays_for(rows, x_key, y_key, exclude_registration_outliers=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=18, alpha=0.75)
    rho, p = corr_pair(x, y, method="spearman")
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(f"{y_key} vs {x_key}\nSpearman rho={fmt(rho)} p={fmt(p)} n={x.size}")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = read_csv(args.raw_stats_csv)
    dark_rows = read_csv(args.dark_eval_csv)
    normal_rows = read_csv(args.normal_eval_csv)
    joined = join_rows(raw_rows, dark_rows, normal_rows)
    if not joined:
        raise RuntimeError("Join produced zero rows")
    write_csv(output_dir / "attribution_join.csv", joined)
    report = build_correlation_report(joined)
    with (output_dir / "correlation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    write_markdown(output_dir / "correlation_report.md", report)
    for y_key, x_key, filename in SCATTERS:
        write_scatter(output_dir / "scatter_plots" / filename, joined, x_key, y_key)
    print(
        f"[JOIN] wrote rows={len(joined)} output_dir={output_dir} "
        f"delta_mean={fmt(report['delta_d1_distribution'].get('mean'))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
