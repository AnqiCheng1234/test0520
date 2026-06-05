#!/usr/bin/env python3
"""Verify ROD RAW rendering parity against the 20 stored reference PNGs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.dataset.rod_raw_rgb import (  # noqa: E402
    render_pipeline,
    teacher_bright_degreen_v1,
    unpack_raw24,
)


DEFAULT_MANIFEST = Path(
    "/mnt/drive/3333_raw/ROD/example_comparisons/"
    "0603_rawvis_student_teacher_dav2s_teacher_dav2l_6col_spectral_r/"
    "0603_selected_20_night_samples_6col_with_dav2s_dav2l.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify ROD online RGB rendering parity.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/codex_smoke_rod_student_rgb_verify"),
        help="Temporary output root for parity summary JSON.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep temporary output even when parity succeeds.",
    )
    return parser.parse_args()


def read_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compare(label: str, generated: np.ndarray, reference_path: str | Path) -> dict[str, object]:
    reference = read_rgb(reference_path)
    if generated.shape != reference.shape:
        raise ValueError(f"{label} shape mismatch: generated={generated.shape} reference={reference.shape}")
    diff = np.abs(generated.astype(np.int16) - reference.astype(np.int16))
    stats = {
        "label": label,
        "reference_path": str(reference_path),
        "max_abs": int(diff.max()),
        "nonzero": int(np.count_nonzero(diff)),
        "mean_abs": float(diff.mean()),
    }
    if stats["nonzero"] != 0:
        raise ValueError(
            f"{label} parity failed: max_abs={stats['max_abs']} "
            f"nonzero={stats['nonzero']} mean_abs={stats['mean_abs']:.10f}"
        )
    return stats


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No rows found in {args.manifest}")

    all_stats = []
    for row in rows:
        raw = unpack_raw24(row["raw_path"])
        student = render_pipeline(raw, "student_dark_degreen_v1")
        teacher = teacher_bright_degreen_v1(raw)
        prefix = f"{row['split']}_{row['sample_name']}"
        all_stats.append(compare(f"{prefix}_student", student, row["student_dark_degreen_v1_path"]))
        all_stats.append(compare(f"{prefix}_teacher", teacher, row["teacher_bright_degreen_v1_path"]))

    summary = {
        "manifest": str(args.manifest),
        "rows": len(rows),
        "comparisons": len(all_stats),
        "max_abs": max(item["max_abs"] for item in all_stats),
        "nonzero": sum(int(item["nonzero"]) for item in all_stats),
        "items": all_stats,
    }
    summary_path = args.output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(
        f"[OK] parity rows={summary['rows']} comparisons={summary['comparisons']} "
        f"max_abs={summary['max_abs']} nonzero={summary['nonzero']} summary={summary_path}"
    )
    if not args.keep_output and "codex_smoke" in str(args.output_root):
        shutil.rmtree(args.output_root)
        print(f"[OK] removed temporary output {args.output_root}")


if __name__ == "__main__":
    main()
