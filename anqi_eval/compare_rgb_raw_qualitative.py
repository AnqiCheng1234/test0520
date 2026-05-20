#!/usr/bin/env python3
"""
Compose 4-panel STF qualitative comparisons:
    RGB input | RGB-pred depth | RAW front-end pseudo-RGB | RAW-pred depth

Expected directory layout matches outputs from visualize_stf_predictions.py:
    rgb_dir:
        image_XXXX.jpg
        pred_XXXX.jpg
    raw_dir:
        front_rgb_XXXX.jpg
        pred_XXXX.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose RGB-vs-RAW STF qualitative comparisons.")
    parser.add_argument("--rgb-dir", type=Path, required=True, help="Directory with RGB qualitative outputs.")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory with RAW qualitative outputs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save 4-panel comparisons.")
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit indices. Defaults to intersection of files in both dirs.",
    )
    return parser.parse_args()


def discover_indices(rgb_dir: Path, raw_dir: Path) -> list[int]:
    rgb_indices = {int(path.stem.split("_")[-1]) for path in rgb_dir.glob("image_*.jpg")}
    raw_indices = {int(path.stem.split("_")[-1]) for path in raw_dir.glob("pred_*.jpg")}
    return sorted(rgb_indices & raw_indices)


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def build_panel(images: list[Image.Image], labels: list[str]) -> Image.Image:
    panel_w, panel_h = images[0].size
    header_h = 30
    canvas = Image.new("RGB", (panel_w * len(images), panel_h + header_h), "white")

    for idx, image in enumerate(images):
        x0 = idx * panel_w
        canvas.paste(image, (x0, header_h))

    draw = ImageDraw.Draw(canvas)
    for idx, label in enumerate(labels):
        draw.text((idx * panel_w + 12, 8), label, fill="black")
    return canvas


def main() -> int:
    args = parse_args()
    rgb_dir = args.rgb_dir.resolve()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.indices:
        indices = sorted(dict.fromkeys(int(idx) for idx in args.indices))
    else:
        indices = discover_indices(rgb_dir, raw_dir)

    if not indices:
        raise FileNotFoundError("No overlapping qualitative sample indices found.")

    labels = ["rgb", "rgb_pred", "raw_front_rgb", "raw_pred"]
    manifest_lines = ["index\trgb_image\trgb_pred\traw_front_rgb\traw_pred"]

    for idx in indices:
        rgb_image_path = rgb_dir / f"image_{idx:04d}.jpg"
        rgb_pred_path = rgb_dir / f"pred_{idx:04d}.jpg"
        raw_front_rgb_path = raw_dir / f"front_rgb_{idx:04d}.jpg"
        raw_pred_path = raw_dir / f"pred_{idx:04d}.jpg"

        missing = [p for p in (rgb_image_path, rgb_pred_path, raw_front_rgb_path, raw_pred_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing comparison assets for index {idx}: {missing}")

        images = [
            open_rgb(rgb_image_path),
            open_rgb(rgb_pred_path),
            open_rgb(raw_front_rgb_path),
            open_rgb(raw_pred_path),
        ]
        canvas = build_panel(images, labels)
        canvas.save(output_dir / f"quadtych_{idx:04d}.jpg", quality=95)

        manifest_lines.append(
            f"{idx}\t{rgb_image_path}\t{rgb_pred_path}\t{raw_front_rgb_path}\t{raw_pred_path}"
        )

    (output_dir / "samples.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(output_dir)
    print(f"Saved {len(indices)} 4-panel comparisons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
