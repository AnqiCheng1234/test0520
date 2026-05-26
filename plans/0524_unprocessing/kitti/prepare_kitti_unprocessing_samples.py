#!/usr/bin/env python3
"""Prepare KITTI RGB sample directories for RAW-like unprocessing checks.

The unprocessing script recursively processes every image under --input-dir.
Pointing it at /mnt/drive/kitti would include unrelated cameras and files, so
this helper creates a small symlink tree for selected train/val RGB frames.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class KittiSample:
    sample_name: str
    split: str
    camera: str
    drive: str
    frame: str
    image_path: str
    depth_path: str
    link_path: str


@dataclass(frozen=True)
class MissingRgb:
    split: str
    camera: str
    depth_path: str
    expected_image_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build KITTI train/val RGB symlink sample dirs.")
    parser.add_argument("--kitti-root", type=Path, default=Path("/mnt/drive/kitti"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--cameras", nargs="+", default=["image_02"], choices=["image_02", "image_03"])
    parser.add_argument("--samples-per-split", type=int, default=12, help="Number of samples per split after combining selected cameras. Use 0 for all.")
    parser.add_argument("--strategy", choices=["linspace", "first", "random"], default="linspace")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--layout",
        choices=["nested", "flat"],
        default="nested",
        help=(
            "nested preserves split/date/drive/camera folders. "
            "flat puts all RGB links in one directory and encodes split/drive/camera/frame in the filename."
        ),
    )
    parser.add_argument("--overwrite-symlinks", action="store_true")
    parser.add_argument("--fail-on-missing-rgb", action="store_true")
    return parser.parse_args()


def drive_to_date(drive: str) -> str:
    parts = drive.split("_drive_", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected KITTI drive name: {drive}")
    return parts[0]


def collect_rows(
    kitti_root: Path,
    split: str,
    cameras: Sequence[str],
    *,
    fail_on_missing_rgb: bool,
) -> tuple[list[tuple[str, str, Path, Path]], list[MissingRgb]]:
    split_root = kitti_root / "annotated_depth" / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing KITTI annotated split directory: {split_root}")

    rows: list[tuple[str, str, Path, Path]] = []
    missing: list[MissingRgb] = []
    for camera in cameras:
        pattern = f"*/proj_depth/groundtruth/{camera}/*.png"
        for depth_path in sorted(split_root.glob(pattern)):
            drive = depth_path.parts[-5]
            date = drive_to_date(drive)
            image_path = kitti_root / date / drive / camera / "data" / depth_path.with_suffix(".jpg").name
            if not image_path.is_file():
                record = MissingRgb(
                    split=split,
                    camera=camera,
                    depth_path=str(depth_path),
                    expected_image_path=str(image_path),
                )
                if fail_on_missing_rgb:
                    raise FileNotFoundError(f"Missing RGB image for depth {depth_path}: {image_path}")
                missing.append(record)
                continue
            rows.append((split, camera, depth_path, image_path))
    return rows, missing


def select_rows(rows: list[tuple[str, str, Path, Path]], count: int, strategy: str, seed: int) -> list[tuple[str, str, Path, Path]]:
    if count <= 0 or count >= len(rows):
        return rows
    if strategy == "first":
        return rows[:count]
    if strategy == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(rows, count), key=lambda item: str(item[2]))
    if strategy == "linspace":
        if count == 1:
            return [rows[len(rows) // 2]]
        last = len(rows) - 1
        indices = sorted({round(i * last / (count - 1)) for i in range(count)})
        return [rows[i] for i in indices]
    raise ValueError(f"Unknown strategy: {strategy}")


def sample_name_for(*, split: str, camera: str, drive: str, frame: str) -> str:
    return f"{split}_{drive}_{camera}_{frame}"


def link_sample(
    *,
    kitti_root: Path,
    output_dir: Path,
    split: str,
    camera: str,
    depth_path: Path,
    image_path: Path,
    layout: str,
    overwrite_symlinks: bool,
) -> KittiSample:
    rel = image_path.relative_to(kitti_root)
    drive = depth_path.parts[-5]
    frame = depth_path.stem
    sample_name = sample_name_for(split=split, camera=camera, drive=drive, frame=frame)
    if layout == "nested":
        link_path = output_dir / split / rel
    elif layout == "flat":
        link_path = output_dir / f"{sample_name}{image_path.suffix.lower()}"
    else:
        raise ValueError(f"Unknown layout: {layout}")
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == image_path.resolve():
            pass
        elif overwrite_symlinks and link_path.is_symlink():
            link_path.unlink()
            link_path.symlink_to(image_path)
        else:
            raise FileExistsError(
                f"Refusing to replace existing path: {link_path}. "
                "Use --overwrite-symlinks for stale symlinks."
            )
    else:
        link_path.symlink_to(image_path)

    return KittiSample(
        sample_name=sample_name,
        split=split,
        camera=camera,
        drive=drive,
        frame=frame,
        image_path=str(image_path),
        depth_path=str(depth_path),
        link_path=str(link_path),
    )


def main() -> None:
    args = parse_args()
    kitti_root = args.kitti_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not kitti_root.is_dir():
        raise FileNotFoundError(f"Missing KITTI root: {kitti_root}")

    samples: list[KittiSample] = []
    missing_rgb: list[MissingRgb] = []
    summary: dict[str, object] = {
        "kitti_root": str(kitti_root),
        "output_dir": str(output_dir),
        "splits": list(args.splits),
        "cameras": list(args.cameras),
        "samples_per_split": int(args.samples_per_split),
        "strategy": str(args.strategy),
        "seed": int(args.seed),
        "layout": str(args.layout),
        "counts": {},
    }

    for split in args.splits:
        rows, missing = collect_rows(
            kitti_root,
            split,
            args.cameras,
            fail_on_missing_rgb=bool(args.fail_on_missing_rgb),
        )
        missing_rgb.extend(missing)
        selected = select_rows(rows, args.samples_per_split, args.strategy, args.seed)
        summary["counts"][split] = {
            "available_with_rgb": len(rows),
            "missing_rgb": len(missing),
            "selected": len(selected),
        }
        for row in selected:
            samples.append(
                link_sample(
                    kitti_root=kitti_root,
                    output_dir=output_dir,
                    split=row[0],
                    camera=row[1],
                    depth_path=row[2],
                    image_path=row[3],
                    layout=str(args.layout),
                    overwrite_symlinks=args.overwrite_symlinks,
                )
            )

    manifest = dict(summary)
    manifest["samples"] = [asdict(sample) for sample in samples]
    manifest["missing_rgb_examples"] = [asdict(item) for item in missing_rgb[:100]]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({**summary, "manifest": str(manifest_path), "num_links": len(samples)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
