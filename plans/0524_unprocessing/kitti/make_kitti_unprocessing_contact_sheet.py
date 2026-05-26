#!/usr/bin/env python3
"""Create contact sheets for KITTI RAW-like unprocessing previews."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build original/normal/dark/over preview contact sheets.")
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--variants", nargs="+", default=["normal", "dark", "over"])
    parser.add_argument(
        "--group-by",
        choices=["split", "all"],
        default="split",
        help="Use split to write train/val sheets, or all to put every sample into one sheet.",
    )
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--thumb-height", type=int, default=120)
    parser.add_argument("--max-per-split", type=int, default=0, help="Use 0 for all manifest samples.")
    return parser.parse_args()


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    fitted = ImageOps.contain(image, size, method=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (245, 245, 245))
    left = (size[0] - fitted.width) // 2
    top = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    return canvas


def label(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int]) -> None:
    draw.rectangle((xy[0], xy[1], xy[0] + 210, xy[1] + 18), fill=(0, 0, 0))
    draw.text((xy[0] + 4, xy[1] + 3), text, fill=(255, 255, 255), font=ImageFont.load_default())


def preview_path(raw_output_dir: Path, variant: str, link_path: Path, sample_root: Path) -> Path:
    rel = link_path.relative_to(sample_root).with_suffix(".png")
    return raw_output_dir / "preview" / variant / rel


def build_sheet(
    *,
    split: str,
    samples: Sequence[dict],
    sample_root: Path,
    raw_output_dir: Path,
    output_dir: Path,
    variants: Sequence[str],
    thumb_size: tuple[int, int],
) -> Path:
    columns = ["original", *variants]
    gap = 8
    label_h = 22
    row_h = thumb_size[1] + label_h + gap
    width = len(columns) * thumb_size[0] + (len(columns) + 1) * gap
    height = len(samples) * row_h + gap
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for row_idx, sample in enumerate(samples):
        link = Path(sample["link_path"])
        paths = [link, *[preview_path(raw_output_dir, variant, link, sample_root) for variant in variants]]
        for col_idx, (name, path) in enumerate(zip(columns, paths)):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name} preview for {sample['frame']}: {path}")
            x = gap + col_idx * (thumb_size[0] + gap)
            y = gap + row_idx * row_h
            sheet.paste(fit_image(path, thumb_size), (x, y + label_h))
            sample_name = sample.get("sample_name") or f"{sample['split']}_{sample['drive']}_{sample['camera']}_{sample['frame']}"
            title = f"{name} {sample_name}" if col_idx == 0 else name
            label(draw, title[:42], (x, y))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split}_contact_sheet.jpg"
    sheet.save(out_path, quality=92)
    return out_path


def main() -> None:
    args = parse_args()
    manifest_path = args.sample_manifest.expanduser().resolve()
    raw_output_dir = args.raw_output_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else raw_output_dir / "contact_sheets"
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_root = Path(manifest["output_dir"]).expanduser().resolve()
    by_group: dict[str, list[dict]] = defaultdict(list)
    for sample in manifest["samples"]:
        key = "all" if args.group_by == "all" else str(sample["split"])
        by_group[key].append(sample)

    written = []
    for split, samples in sorted(by_group.items()):
        if args.max_per_split and args.max_per_split > 0:
            samples = samples[: args.max_per_split]
        written.append(
            build_sheet(
                split=split,
                samples=samples,
                sample_root=sample_root,
                raw_output_dir=raw_output_dir,
                output_dir=output_dir,
                variants=args.variants,
                thumb_size=(args.thumb_width, args.thumb_height),
            )
        )
    print(json.dumps({"contact_sheets": [str(path) for path in written]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
