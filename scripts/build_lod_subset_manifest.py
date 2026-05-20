#!/usr/bin/env python3
"""Build deterministic subset manifests for LoD pseudo-depth training."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COLUMNS = ("split", "sample_name", "rggb_path", "output_npy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input LoD manifest CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Output subset manifest CSV.")
    parser.add_argument("--fraction", default=0.5, type=float, help="Fraction to keep from each stratum.")
    parser.add_argument("--seed", default=42, type=int, help="Deterministic sampling seed.")
    parser.add_argument(
        "--stratify",
        default="split",
        choices=("split", "none"),
        help="Stratification key. Use split to preserve 00Train/01Valid proportions.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        return fieldnames, list(reader)


def stratum_key(row: dict[str, str], mode: str) -> str:
    if mode == "none":
        return "__all__"
    if mode == "split":
        return row["split"]
    raise ValueError(f"Unsupported stratify mode: {mode}")


def select_rows(
    rows: list[dict[str, str]],
    *,
    fraction: float,
    seed: int,
    stratify: str,
) -> tuple[list[dict[str, str]], dict[str, dict[str, int]]]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"--fraction must be in (0, 1], got {fraction}")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[stratum_key(row, stratify)].append(index)

    selected_indices: set[int] = set()
    stats: dict[str, dict[str, int]] = {}
    for key in sorted(groups):
        indices = groups[key]
        keep = int(len(indices) * fraction)
        if fraction > 0.0 and keep == 0 and indices:
            keep = 1
        rng = random.Random(f"{seed}:{key}")
        picked = set(rng.sample(indices, keep))
        selected_indices.update(picked)
        stats[key] = {"input_rows": len(indices), "selected_rows": keep}

    selected_rows = [row for index, row in enumerate(rows) if index in selected_indices]
    return selected_rows, stats


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_metadata(
    output_path: Path,
    *,
    input_path: Path,
    fraction: float,
    seed: int,
    stratify: str,
    input_rows: int,
    output_rows: int,
    strata: dict[str, dict[str, int]],
) -> Path:
    meta_path = output_path.with_suffix(".meta.json")
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "fraction": fraction,
        "seed": seed,
        "stratify": stratify,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "strata": strata,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return meta_path


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser()
    output_path = args.output.expanduser()

    fieldnames, rows = read_manifest(input_path)
    selected_rows, strata = select_rows(
        rows,
        fraction=args.fraction,
        seed=args.seed,
        stratify=args.stratify,
    )
    write_manifest(output_path, fieldnames, selected_rows)
    meta_path = write_metadata(
        output_path,
        input_path=input_path,
        fraction=args.fraction,
        seed=args.seed,
        stratify=args.stratify,
        input_rows=len(rows),
        output_rows=len(selected_rows),
        strata=strata,
    )

    print(
        f"[done] input_rows={len(rows)} output_rows={len(selected_rows)} "
        f"manifest={output_path} metadata={meta_path}"
    )


if __name__ == "__main__":
    main()
