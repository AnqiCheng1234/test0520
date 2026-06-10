#!/usr/bin/env python3
"""Build visual panels for post-RAM external eval artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stats-json", type=Path, default=None)
    parser.add_argument("--affine", choices=("n_a", "q001q999", "q01q99"), default="n_a")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--grid-cols", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def affine_keys(name: str) -> tuple[str, str]:
    if name == "q001q999":
        return "q001", "q999"
    if name == "q01q99":
        return "q01", "q99"
    raise ValueError(f"affine={name!r} does not define quantile keys")


def load_display_affine(stats_json: Path | None, affine: str) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    if stats_json is None or affine == "n_a":
        return None, None, {}
    payload = load_json(stats_json)
    qlo_key, qhi_key = affine_keys(affine)
    rows = payload.get("channels") or []
    if len(rows) != 3:
        raise ValueError(f"Expected 3 channel stats in {stats_json}, got {len(rows)}")
    qlo = np.asarray([float(row[qlo_key]) for row in rows], dtype=np.float32).reshape(3, 1, 1)
    qhi = np.asarray([float(row[qhi_key]) for row in rows], dtype=np.float32).reshape(3, 1, 1)
    return qlo, qhi, {"stats_json": str(stats_json), "qlo_key": qlo_key, "qhi_key": qhi_key}


def chw_to_hwc(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3 and x.shape[0] in {1, 3}:
        return np.transpose(x, (1, 2, 0))
    return x


def normalize_chw(x: np.ndarray, qlo: np.ndarray | None, qhi: np.ndarray | None) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if qlo is not None and qhi is not None:
        y = (x - qlo) / np.maximum(qhi - qlo, 1e-12)
    else:
        lo = np.nanpercentile(x, 1, axis=(1, 2), keepdims=True)
        hi = np.nanpercentile(x, 99, axis=(1, 2), keepdims=True)
        y = (x - lo) / np.maximum(hi - lo, 1e-12)
    return np.clip(chw_to_hwc(y), 0.0, 1.0)


def normalize_depth(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.nanpercentile(x[finite], [1, 99])
    return np.clip((x - lo) / max(float(hi - lo), 1e-12), 0.0, 1.0)


def artifact_paths(eval_dir: Path, manifest: dict[str, Any], max_samples: int) -> list[Path]:
    rows = manifest.get("artifacts") or []
    paths = []
    for row in rows[: max(0, int(max_samples))]:
        path = row.get("artifact_path")
        if path:
            paths.append(Path(path))
    if paths:
        return paths
    return sorted((eval_dir / "artifacts").glob("*.npz"))[: max(0, int(max_samples))]


def make_panel(
    artifact_path: Path,
    output_path: Path,
    *,
    summary: dict[str, Any],
    qlo: np.ndarray | None,
    qhi: np.ndarray | None,
    delta_abs_range: float,
    dpi: int,
) -> dict[str, Any]:
    data = np.load(artifact_path, allow_pickle=False)
    row = json.loads(str(data["row"]))
    x_ram = data["x_ram"].astype(np.float32)
    x_out = data["x_out"].astype(np.float32)
    delta = data["delta"].astype(np.float32)
    raw_preview = data["raw_preview"].astype(np.float32) if "raw_preview" in data and data["raw_preview"].shape else None
    pred = data["pred"].astype(np.float32)
    error = data["error"].astype(np.float32)
    d1_hit = data["d1_hit"].astype(bool)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    axes = axes.reshape(-1)
    if raw_preview is not None:
        axes[0].imshow(np.clip(chw_to_hwc(raw_preview), 0.0, 1.0))
    else:
        axes[0].imshow(np.zeros_like(pred), cmap="gray")
    axes[0].set_title("RAW_dark")
    axes[1].imshow(normalize_chw(x_ram, qlo, qhi))
    axes[1].set_title(str(summary.get("x_ram_label", "x_ram")))
    axes[2].imshow(normalize_chw(x_out, qlo, qhi))
    axes[2].set_title(str(summary.get("x_out_label", "x_out")))
    delta_range = max(float(delta_abs_range), 1e-12)
    im_delta = axes[3].imshow(chw_to_hwc(delta.mean(axis=0, keepdims=True))[..., 0], cmap="coolwarm", vmin=-delta_range, vmax=delta_range)
    axes[3].set_title("mean delta")
    fig.colorbar(im_delta, ax=axes[3], fraction=0.046, pad=0.04)
    im_abs = axes[4].imshow(np.mean(np.abs(delta), axis=0), cmap="magma", vmin=0.0, vmax=delta_range)
    axes[4].set_title("abs(delta)")
    fig.colorbar(im_abs, ax=axes[4], fraction=0.046, pad=0.04)
    axes[5].imshow(normalize_depth(pred), cmap="magma")
    axes[5].set_title("pred depth")
    im_err = axes[6].imshow(normalize_depth(error), cmap="inferno")
    axes[6].set_title("error")
    fig.colorbar(im_err, ax=axes[6], fraction=0.046, pad=0.04)
    axes[7].imshow(d1_hit, cmap="gray", vmin=0, vmax=1)
    axes[7].set_title("D1 hit")
    for ax in axes:
        ax.axis("off")
    title = (
        f"{summary.get('ext_id')} {summary.get('denoiser')} "
        f"sigma={summary.get('sigma')} alpha={summary.get('alpha')} affine={summary.get('affine')} "
        f"D1={summary.get('d1', float('nan')):.6f} AbsRel={summary.get('abs_rel', float('nan')):.6f} "
        f"clamp={summary.get('clamp_total_ratio', float('nan')):.6f} sample={row.get('sample_id')}"
    )
    fig.suptitle(title, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi))
    plt.close(fig)
    return {
        "artifact_path": str(artifact_path),
        "panel_path": str(output_path),
        "sample_index": int(row.get("sample_index", -1)),
        "sample_id": str(row.get("sample_id", "")),
        "d1": float(row.get("d1", float("nan"))),
        "abs_rel": float(row.get("abs_rel", float("nan"))),
    }


def make_grid(panel_paths: list[Path], output_path: Path, *, cols: int) -> None:
    if not panel_paths:
        return
    images = [Image.open(path).convert("RGB") for path in panel_paths]
    width = max(img.width for img in images)
    height = max(img.height for img in images)
    cols = max(1, int(cols))
    rows = int(math.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * width, rows * height), "white")
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        canvas.paste(img, (col * width, row * height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    for img in images:
        img.close()


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else eval_dir / "viz" / "default"
    manifest = load_json(eval_dir / "manifest.json")
    summary = manifest.get("summary") or load_json(eval_dir / "metrics.json")
    qlo, qhi, affine_meta = load_display_affine(args.stats_json, args.affine)
    paths = artifact_paths(eval_dir, manifest, args.max_samples)
    if not paths:
        raise FileNotFoundError(f"No artifacts found in {eval_dir / 'artifacts'}")
    delta_values = []
    for path in paths:
        data = np.load(path, allow_pickle=False)
        delta_values.append(np.abs(data["delta"].astype(np.float32)).reshape(-1))
    delta_abs_range = float(np.nanpercentile(np.concatenate(delta_values), 99)) if delta_values else 1.0
    panel_rows = []
    panel_paths = []
    for path in paths:
        panel_path = output_dir / "per_sample" / f"{path.stem}.png"
        panel_rows.append(
            make_panel(
                path,
                panel_path,
                summary=summary,
                qlo=qlo,
                qhi=qhi,
                delta_abs_range=delta_abs_range,
                dpi=args.dpi,
            )
        )
        panel_paths.append(panel_path)
    grid_path = output_dir / "panel_grid.png"
    make_grid(panel_paths, grid_path, cols=args.grid_cols)
    viz_manifest = {
        "eval_dir": str(eval_dir),
        "output_dir": str(output_dir),
        "panel_grid": str(grid_path),
        "delta_abs_range_p99": delta_abs_range,
        "affine_meta": affine_meta,
        "summary": summary,
        "panels": panel_rows,
    }
    write_json(output_dir / "manifest.json", viz_manifest)
    print(f"[POSTRAM_VIZ] wrote {grid_path} panels={len(panel_rows)}", flush=True)


if __name__ == "__main__":
    main()
