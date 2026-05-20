from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


RECTIFIED_BAYER_KEY = "bayer_rect"
OLD_PACK_ORDER = "[R,Gb,Gr,B]"
NEW_PACK_ORDER = "[R,Gr,Gb,B]"
MIGRATION_NOTE = "Swapped RobotCar green channels to canonical [R,Gr,Gb,B] order."


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate RobotCar packed Bayer channel order to [R,Gr,Gb,B].")
    parser.add_argument("--robotcar-root", default="/mnt/drive/3333_raw/robotcar_raw_depth_720960")
    parser.add_argument("--skip-if-correct", action="store_true", default=True)
    return parser.parse_args()


def swap_green_channels(npz_path: Path) -> bool:
    with np.load(npz_path, allow_pickle=False) as data:
        if RECTIFIED_BAYER_KEY not in data.files:
            raise KeyError(f"{npz_path} does not contain '{RECTIFIED_BAYER_KEY}'")
        arr = np.asarray(data[RECTIFIED_BAYER_KEY], dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise RuntimeError(f"Unexpected packed raw shape for {npz_path}: {arr.shape}")
    arr = arr[..., [0, 2, 1, 3]]
    np.savez_compressed(npz_path, **{RECTIFIED_BAYER_KEY: arr.astype(np.float32, copy=False)})
    return True


def update_meta(meta_path: Path) -> bool:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if payload.get("pack_order") == NEW_PACK_ORDER:
        return False
    payload["pack_order"] = NEW_PACK_ORDER
    if "black_level_1" in payload and "black_level_2" in payload:
        payload["black_level_1"], payload["black_level_2"] = payload["black_level_2"], payload["black_level_1"]
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True


def update_manifest(manifest_path: Path) -> int:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    updated = 0
    for row in rows:
        if row.get("pack_order") == NEW_PACK_ORDER:
            continue
        row["pack_order"] = NEW_PACK_ORDER
        if "black_level_1" in row and "black_level_2" in row:
            row["black_level_1"], row["black_level_2"] = row["black_level_2"], row["black_level_1"]
        updated += 1
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    return updated


def main():
    args = parse_args()
    root = Path(args.robotcar_root).expanduser().resolve()
    raw_roots = [root / "raw_4ch_native", root / "raw_4ch_720960"]
    manifest_path = root / "manifests" / "robotcar_raw_depth_v1_val.csv"
    summary_path = root / "manifests" / "robotcar_summary.json"

    if args.skip_if_correct and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("migration_note") == MIGRATION_NOTE:
            print(
                json.dumps(
                    {
                        "robotcar_root": str(root),
                        "skipped": True,
                        "reason": "migration already applied",
                        "new_pack_order": NEW_PACK_ORDER,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            return

    swapped_npz = 0
    updated_meta = 0

    for raw_root in raw_roots:
        if not raw_root.exists():
            continue
        for npz_path in sorted(raw_root.rglob("*.npz")):
            swap_green_channels(npz_path)
            swapped_npz += 1

    native_root = root / "raw_4ch_native"
    if native_root.exists():
        for meta_path in sorted(native_root.rglob("*.meta.json")):
            if update_meta(meta_path):
                updated_meta += 1

    updated_manifest_rows = 0
    if manifest_path.is_file():
        updated_manifest_rows = update_manifest(manifest_path)

    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["pack_order"] = NEW_PACK_ORDER
        summary["migration_note"] = MIGRATION_NOTE
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "robotcar_root": str(root),
                "swapped_npz": swapped_npz,
                "updated_meta": updated_meta,
                "updated_manifest_rows": updated_manifest_rows,
                "new_pack_order": NEW_PACK_ORDER,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
