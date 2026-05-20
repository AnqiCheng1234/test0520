#!/usr/bin/env python3
import argparse
from pathlib import Path


DEFAULT_TEMPLATE = "/home/caq/6666_raw/dav2/metric_depth/dataset/splits/vkitti2/train.txt"
DEFAULT_OUTPUT = "/home/caq/6666_raw/dav2/finetune_stf/dataset/splits/vkitti2/train.txt"
DEFAULT_ROOT = "/mnt/drive/1111_new_works/VKITTI2"
OLD_PREFIX = "/mnt/bn/liheyang/DepthDatasets/vKitti2/"


def parse_args():
    parser = argparse.ArgumentParser("Generate finetune_stf VKITTI2 split from the upstream template")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--vkitti2-root", default=DEFAULT_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def remap_path(path_str, vkitti2_root):
    if OLD_PREFIX not in path_str:
        raise ValueError(f"Unexpected VKITTI2 template path: {path_str}")

    rest = path_str.split(OLD_PREFIX, 1)[1]
    if "/frames/rgb/" in rest:
        return Path(vkitti2_root) / "rgb" / rest
    if "/frames/depth/" in rest:
        return Path(vkitti2_root) / "depth" / rest
    raise ValueError(f"Could not infer modality from path: {path_str}")


def main():
    args = parse_args()
    template = Path(args.template).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    vkitti2_root = Path(args.vkitti2_root).expanduser().resolve()

    if not template.is_file():
        raise FileNotFoundError(f"Missing template split file: {template}")

    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    missing = 0
    lines = []

    with template.open("r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            img_old, depth_old = raw_line.split()
            img_new = remap_path(img_old, vkitti2_root)
            depth_new = remap_path(depth_old, vkitti2_root)

            total += 1
            if not img_new.is_file() or not depth_new.is_file():
                missing += 1

            lines.append(f"{img_new} {depth_new}")

    with output.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    print(f"wrote {total} entries to {output}")
    print(f"missing paths: {missing}")
    if args.strict and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

