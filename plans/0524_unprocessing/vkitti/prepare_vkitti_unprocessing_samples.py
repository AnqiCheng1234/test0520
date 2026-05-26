#!/usr/bin/env python3
"""Prepare VKITTI2 RGB sample directories for RAW-like unprocessing checks."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VKITTI_ROOT = Path("/mnt/drive/1111_new_works/VKITTI2")
DEFAULT_FILELIST = PROJECT_ROOT / "finetune_stf" / "dataset" / "splits" / "vkitti2" / "train.txt"


@dataclass(frozen=True)
class VkittiSample:
    sample_name: str
    split: str
    scene: str
    condition: str
    camera: str
    frame: str
    image_path: str
    depth_path: str
    link_path: str


@dataclass(frozen=True)
class MissingPath:
    split: str
    image_path: str
    depth_path: str
    missing_image: bool
    missing_depth: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VKITTI2 RGB symlink sample dirs.")
    parser.add_argument("--vkitti-root", type=Path, default=DEFAULT_VKITTI_ROOT)
    parser.add_argument("--filelist-path", type=Path, default=DEFAULT_FILELIST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--samples", type=int, default=12, help="Number of samples to select. Use 0 for all.")
    parser.add_argument("--strategy", choices=["linspace", "first", "random"], default="linspace")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene filter, e.g. Scene01 Scene20.")
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Optional VKITTI2 condition filter, e.g. clone fog rain overcast sunset.",
    )
    parser.add_argument("--cameras", nargs="*", default=None, help="Optional camera filter, e.g. Camera_0.")
    parser.add_argument(
        "--layout",
        choices=["flat", "nested"],
        default="flat",
        help="flat encodes metadata in one filename; nested mirrors the path under the VKITTI2 root.",
    )
    parser.add_argument("--overwrite-symlinks", action="store_true")
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args()


def parse_vkitti_image_path(path: Path, vkitti_root: Path) -> tuple[str, str, str, str]:
    try:
        rel = path.resolve().relative_to(vkitti_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Image path is not under VKITTI2 root: {path}") from exc

    parts = rel.parts
    expected = ("rgb", "frames", "rgb")
    if len(parts) < 7 or parts[0] != expected[0] or parts[3] != expected[1] or parts[4] != expected[2]:
        raise ValueError(f"Unexpected VKITTI2 RGB path layout: {path}")
    scene = parts[1]
    condition = parts[2]
    camera = parts[5]
    frame = path.stem
    return scene, condition, camera, frame


def sample_name_for(*, split: str, scene: str, condition: str, camera: str, frame: str) -> str:
    return f"{split}_{scene}_{condition}_{camera}_{frame}"


def keep_row(
    *,
    scene: str,
    condition: str,
    camera: str,
    scenes: Sequence[str] | None,
    conditions: Sequence[str] | None,
    cameras: Sequence[str] | None,
) -> bool:
    if scenes is not None and scene not in set(scenes):
        return False
    if conditions is not None and condition not in set(conditions):
        return False
    if cameras is not None and camera not in set(cameras):
        return False
    return True


def read_rows(args: argparse.Namespace, vkitti_root: Path) -> tuple[list[tuple[Path, Path, str, str, str, str]], list[MissingPath]]:
    filelist_path = args.filelist_path.expanduser().resolve()
    if not filelist_path.is_file():
        raise FileNotFoundError(f"Missing VKITTI2 filelist: {filelist_path}")

    rows: list[tuple[Path, Path, str, str, str, str]] = []
    missing: list[MissingPath] = []
    scenes = set(args.scenes) if args.scenes else None
    conditions = set(args.conditions) if args.conditions else None
    cameras = set(args.cameras) if args.cameras else None

    with filelist_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            image_str, depth_str = line.split()
            image_path = Path(image_str).expanduser().resolve()
            depth_path = Path(depth_str).expanduser().resolve()
            scene, condition, camera, frame = parse_vkitti_image_path(image_path, vkitti_root)
            if not keep_row(
                scene=scene,
                condition=condition,
                camera=camera,
                scenes=scenes,
                conditions=conditions,
                cameras=cameras,
            ):
                continue
            missing_image = not image_path.is_file()
            missing_depth = not depth_path.is_file()
            if missing_image or missing_depth:
                record = MissingPath(
                    split=str(args.split_name),
                    image_path=str(image_path),
                    depth_path=str(depth_path),
                    missing_image=bool(missing_image),
                    missing_depth=bool(missing_depth),
                )
                if args.fail_on_missing:
                    raise FileNotFoundError(record)
                missing.append(record)
                continue
            rows.append((image_path, depth_path, scene, condition, camera, frame))
    return rows, missing


def select_rows(rows: list[tuple[Path, Path, str, str, str, str]], count: int, strategy: str, seed: int) -> list[tuple[Path, Path, str, str, str, str]]:
    if count <= 0 or count >= len(rows):
        return rows
    if strategy == "first":
        return rows[:count]
    if strategy == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(rows, count), key=lambda item: str(item[0]))
    if strategy == "linspace":
        if count == 1:
            return [rows[len(rows) // 2]]
        last = len(rows) - 1
        indices = sorted({round(i * last / (count - 1)) for i in range(count)})
        return [rows[i] for i in indices]
    raise ValueError(f"Unknown strategy: {strategy}")


def link_sample(
    *,
    vkitti_root: Path,
    output_dir: Path,
    split: str,
    image_path: Path,
    depth_path: Path,
    scene: str,
    condition: str,
    camera: str,
    frame: str,
    layout: str,
    overwrite_symlinks: bool,
) -> VkittiSample:
    sample_name = sample_name_for(split=split, scene=scene, condition=condition, camera=camera, frame=frame)
    if layout == "flat":
        link_path = output_dir / f"{sample_name}{image_path.suffix.lower()}"
    elif layout == "nested":
        link_path = output_dir / split / image_path.relative_to(vkitti_root)
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

    return VkittiSample(
        sample_name=sample_name,
        split=split,
        scene=scene,
        condition=condition,
        camera=camera,
        frame=frame,
        image_path=str(image_path),
        depth_path=str(depth_path),
        link_path=str(link_path),
    )


def count_field(rows: Sequence[tuple[Path, Path, str, str, str, str]], index: int) -> dict[str, int]:
    return dict(sorted(Counter(row[index] for row in rows).items()))


def main() -> None:
    args = parse_args()
    vkitti_root = args.vkitti_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not vkitti_root.is_dir():
        raise FileNotFoundError(f"Missing VKITTI2 root: {vkitti_root}")

    rows, missing = read_rows(args, vkitti_root)
    selected = select_rows(rows, int(args.samples), str(args.strategy), int(args.seed))

    samples = [
        link_sample(
            vkitti_root=vkitti_root,
            output_dir=output_dir,
            split=str(args.split_name),
            image_path=row[0],
            depth_path=row[1],
            scene=row[2],
            condition=row[3],
            camera=row[4],
            frame=row[5],
            layout=str(args.layout),
            overwrite_symlinks=bool(args.overwrite_symlinks),
        )
        for row in selected
    ]

    summary: dict[str, object] = {
        "vkitti_root": str(vkitti_root),
        "filelist_path": str(args.filelist_path.expanduser().resolve()),
        "output_dir": str(output_dir),
        "split_name": str(args.split_name),
        "samples": int(args.samples),
        "strategy": str(args.strategy),
        "seed": int(args.seed),
        "layout": str(args.layout),
        "filters": {
            "scenes": list(args.scenes) if args.scenes else None,
            "conditions": list(args.conditions) if args.conditions else None,
            "cameras": list(args.cameras) if args.cameras else None,
        },
        "counts": {
            str(args.split_name): {
                "available_with_rgb_depth": len(rows),
                "missing_paths": len(missing),
                "selected": len(selected),
                "scenes": count_field(rows, 2),
                "conditions": count_field(rows, 3),
                "cameras": count_field(rows, 4),
            }
        },
    }
    manifest = dict(summary)
    manifest["samples"] = [asdict(sample) for sample in samples]
    manifest["missing_path_examples"] = [asdict(item) for item in missing[:100]]

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({**summary, "manifest": str(manifest_path), "num_links": len(samples)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
