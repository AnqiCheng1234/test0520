#!/usr/bin/env python3
"""Build a per-sample cross-experiment panel from existing STF fixed-viz panels."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TILE_W = 360
TILE_H = 230
META_H = 58
DIST_H = 190
PANEL_TILE_COUNT = 7
DIST_GAP = 8

DEFAULT_RAW_RUN_PREFIXES = (
    "0521_0012",
    "0521_1542",
    "0522_1423",
    "0521_0112",
    "0521_0522",
    "0521_0656",
    "0521_0835",
)


@dataclass(frozen=True)
class RunInfo:
    name: str
    path: Path
    input_type: str
    route: str
    train_lines: tuple[str, ...]
    best_epoch: int
    best_value: float | None
    panel_path: Path


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = _font(28)
FONT_SECTION = _font(22)
FONT_TEXT = _font(16)
FONT_SMALL = _font(13)


def _timestamp_prefix(name: str) -> str | None:
    match = re.match(r"^(\d{4}_\d{4})", name)
    return match.group(1) if match else None


def _load_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _train_summary_lines(run_name: str, config: dict) -> tuple[str, ...]:
    input_type = str(config.get("input_type", ""))
    dav2_mode = str(config.get("dav2_train_mode", "n/a"))
    lora = "yes" if "lora" in input_type or "lora" in run_name else "no"
    if "bridge" in input_type or "bridge" in run_name:
        frontend = "bridge"
    elif "identity" in run_name:
        frontend = "identity"
    elif "raw" in input_type:
        frontend = "raw"
    else:
        frontend = "rgb"

    train_bits = [f"dav2={dav2_mode}", f"lora={lora}", f"frontend={frontend}"]
    layer_decay = config.get("backbone_layer_decay")
    if dav2_mode == "full" and layer_decay is not None:
        train_bits.append(f"lrd={float(layer_decay):.2g}")

    loss = str(config.get("loss_type", "n/a"))
    lr = config.get("lr")
    bs = config.get("bs")
    accum = config.get("accum_steps")
    epochs = config.get("epochs")

    opt_bits = []
    if lr is not None:
        opt_bits.append(f"lr={float(lr):.1e}")
    if bs is not None and accum is not None:
        opt_bits.append(f"bs={bs}x{accum}")
    if epochs is not None:
        opt_bits.append(f"epochs={epochs}")

    loss_line = f"loss={loss}"

    return (
        "train: " + ", ".join(train_bits),
        loss_line,
        "optim: " + ", ".join(opt_bits) if opt_bits else "optim: n/a",
    )


def _best_epoch_from_log(run_dir: Path) -> tuple[int | None, float | None]:
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return None, None

    saved_best: list[tuple[int, float]] = []
    val_scores: list[tuple[int, float]] = []
    saved_re = re.compile(r"saved best=.*?value=([0-9.]+) epoch=(\d+)")
    val_re = re.compile(r"\[EVAL\]\[val\] done epoch=(\d+).*?abs_rel=([0-9.]+)")
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            saved = saved_re.search(line)
            if saved:
                saved_best.append((int(saved.group(2)), float(saved.group(1))))
            val = val_re.search(line)
            if val:
                val_scores.append((int(val.group(1)), float(val.group(2))))

    if saved_best:
        return saved_best[-1]
    if val_scores:
        epoch, value = min(val_scores, key=lambda item: item[1])
        return epoch, value
    return None, None


def _latest_epoch(run_dir: Path) -> int | None:
    epochs = []
    for path in (run_dir / "viz_fixed").glob("epoch_[0-9][0-9]"):
        try:
            epochs.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(epochs) if epochs else None


def _collect_runs(exp_root: Path, start: str, end: str, sample: str) -> tuple[list[RunInfo], list[str]]:
    runs: list[RunInfo] = []
    warnings: list[str] = []
    for run_dir in sorted(path for path in exp_root.iterdir() if path.is_dir()):
        prefix = _timestamp_prefix(run_dir.name)
        if prefix is None or prefix < start or prefix > end:
            continue

        config = _load_config(run_dir)
        input_type = str(config.get("input_type", ""))
        route = "raw" if "raw" in input_type or "raw_ram" in run_dir.name else "rgb"
        best_epoch, best_value = _best_epoch_from_log(run_dir)
        if best_epoch is None:
            best_epoch = _latest_epoch(run_dir)
            warnings.append(f"{run_dir.name}: no best epoch in log; using latest epoch {best_epoch}")
        if best_epoch is None:
            warnings.append(f"{run_dir.name}: no viz_fixed epoch found; skipped")
            continue

        panel_path = run_dir / "viz_fixed" / f"epoch_{best_epoch:02d}" / "stf" / f"{sample}_panel.png"
        if not panel_path.exists():
            warnings.append(f"{run_dir.name}: missing {panel_path}; skipped")
            continue
        runs.append(
            RunInfo(
                name=run_dir.name,
                path=run_dir,
                input_type=input_type,
                route=route,
                train_lines=_train_summary_lines(run_dir.name, config),
                best_epoch=best_epoch,
                best_value=best_value,
                panel_path=panel_path,
            )
        )
    return runs, warnings


def _run_info_for_sample(run_dir: Path, sample: str) -> tuple[RunInfo | None, list[str]]:
    warnings: list[str] = []
    config = _load_config(run_dir)
    input_type = str(config.get("input_type", ""))
    route = "raw" if "raw" in input_type or "raw_ram" in run_dir.name else "rgb"
    best_epoch, best_value = _best_epoch_from_log(run_dir)
    if best_epoch is None:
        best_epoch = _latest_epoch(run_dir)
        warnings.append(f"{run_dir.name}: no best epoch in log; using latest epoch {best_epoch}")
    if best_epoch is None:
        warnings.append(f"{run_dir.name}: no viz_fixed epoch found; skipped")
        return None, warnings

    panel_path = run_dir / "viz_fixed" / f"epoch_{best_epoch:02d}" / "stf" / f"{sample}_panel.png"
    if not panel_path.exists():
        warnings.append(f"{run_dir.name}: missing {panel_path}; skipped")
        return None, warnings

    return (
        RunInfo(
            name=run_dir.name,
            path=run_dir,
            input_type=input_type,
            route=route,
            train_lines=_train_summary_lines(run_dir.name, config),
            best_epoch=best_epoch,
            best_value=best_value,
            panel_path=panel_path,
        ),
        warnings,
    )


def _matching_run_dirs(exp_root: Path, prefixes: list[str]) -> tuple[list[Path], list[str]]:
    warnings: list[str] = []
    selected: list[Path] = []
    seen: set[Path] = set()
    for prefix in prefixes:
        matches = sorted(path for path in exp_root.iterdir() if path.is_dir() and path.name.startswith(prefix))
        if not matches:
            warnings.append(f"{prefix}: no matching run directory found")
        for run_dir in matches:
            if run_dir in seen:
                continue
            selected.append(run_dir)
            seen.add(run_dir)
    return selected, warnings


def _collect_runs_by_prefixes(exp_root: Path, prefixes: list[str], sample: str) -> tuple[list[RunInfo], list[str]]:
    runs: list[RunInfo] = []
    run_dirs, warnings = _matching_run_dirs(exp_root, prefixes)
    for run_dir in run_dirs:
        run, run_warnings = _run_info_for_sample(run_dir, sample)
        warnings.extend(run_warnings)
        if run is not None:
            runs.append(run)
    return runs, warnings


def _discover_common_samples(exp_root: Path, start: str, end: str, raw_run_prefixes: list[str] | None = None) -> list[str]:
    sample_sets = []
    for run_dir in sorted(path for path in exp_root.iterdir() if path.is_dir()):
        prefix = _timestamp_prefix(run_dir.name)
        if prefix is None or prefix < start or prefix > end:
            continue

        best_epoch, _ = _best_epoch_from_log(run_dir)
        if best_epoch is None:
            best_epoch = _latest_epoch(run_dir)
        if best_epoch is None:
            continue

        split_dir = run_dir / "viz_fixed" / f"epoch_{best_epoch:02d}" / "stf"
        if not split_dir.exists():
            continue
        names = {
            path.name[: -len("_panel.png")]
            for path in split_dir.glob("*_panel.png")
            if path.name.endswith("_panel.png")
        }
        if names:
            sample_sets.append(names)

    if raw_run_prefixes:
        run_dirs, _ = _matching_run_dirs(exp_root, raw_run_prefixes)
        for run_dir in run_dirs:
            best_epoch, _ = _best_epoch_from_log(run_dir)
            if best_epoch is None:
                best_epoch = _latest_epoch(run_dir)
            if best_epoch is None:
                continue

            split_dir = run_dir / "viz_fixed" / f"epoch_{best_epoch:02d}" / "stf"
            if not split_dir.exists():
                continue
            names = {
                path.name[: -len("_panel.png")]
                for path in split_dir.glob("*_panel.png")
                if path.name.endswith("_panel.png")
            }
            if names:
                sample_sets.append(names)

    if not sample_sets:
        return []
    common = set.intersection(*sample_sets)
    return sorted(common)


def _crop_tile(panel: Image.Image, index: int) -> Image.Image:
    return panel.crop((index * TILE_W, 0, (index + 1) * TILE_W, TILE_H))


def _tile_count(panel: Image.Image) -> int:
    return panel.size[0] // TILE_W


def _distribution_count(panel: Image.Image) -> int:
    """Detect old 3-block and new 4-block distribution strips from title ink."""
    image = np.asarray(panel.convert("RGB"))
    width = image.shape[1]
    y0 = TILE_H + META_H
    candidate_scores = {}
    for count in (3, 4):
        block_w = (width - DIST_GAP * (count + 1)) // count
        scores = []
        for idx in range(count):
            x0 = DIST_GAP + idx * (block_w + DIST_GAP)
            x1 = min(x0 + 260, width)
            region = image[y0 + 6 : y0 + 28, x0 + 8 : x1, :]
            if region.size == 0:
                scores.append(0)
                continue
            bright = np.any(region > 130, axis=2)
            scores.append(int(np.count_nonzero(bright)))
        candidate_scores[count] = sum(scores)
    return 4 if candidate_scores[4] > candidate_scores[3] else 3


def _distribution_index(kind: str, count: int) -> int:
    if kind == "rgb":
        return 0
    if kind == "input":
        return 2 if count == 4 else 1
    if kind == "ram":
        return 3 if count == 4 else 2
    raise ValueError(f"unknown distribution kind: {kind}")


def _crop_distribution(panel: Image.Image, index: int, *, count: int | None = None) -> Image.Image:
    width, _ = panel.size
    if count is None:
        count = _distribution_count(panel)
    block_w = (width - DIST_GAP * (count + 1)) // count
    x0 = DIST_GAP + index * (block_w + DIST_GAP)
    y0 = TILE_H + META_H
    return panel.crop((x0, y0, x0 + block_w, y0 + DIST_H))


def _choose_reference_run(runs: list[RunInfo], panels: dict[str, Image.Image]) -> RunInfo:
    for run in runs:
        panel = panels[run.name]
        if _tile_count(panel) >= PANEL_TILE_COUNT and _distribution_count(panel) >= 4:
            return run
    for run in runs:
        if _tile_count(panels[run.name]) >= PANEL_TILE_COUNT:
            return run
    return runs[0]


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=(235, 235, 235)) -> None:
    draw.text(xy, text, font=font, fill=fill)


def _wrapped_label(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, run: RunInfo) -> None:
    short = run.name
    for token in ("stf_train_test_", "pseudovitl_", "raw_ram_rgb_bnclean_", "decoder_"):
        short = short.replace(token, "")
    _draw_text(draw, (x, y), short[:42], FONT_TEXT)
    metric = "n/a" if run.best_value is None else f"{run.best_value:.4f}"
    _draw_text(
        draw,
        (x, y + 23),
        f"{run.route.upper()}  best epoch {run.best_epoch:02d}  val abs_rel {metric}",
        FONT_SMALL,
        fill=(190, 190, 190),
    )
    _draw_text(draw, (x, y + 43), f"input_type={run.input_type}", FONT_SMALL, fill=(165, 165, 165))
    for idx, line in enumerate(run.train_lines):
        _draw_text(draw, (x, y + 68 + idx * 20), line[:55], FONT_SMALL, fill=(150, 150, 150))


def _placeholder(width: int, height: int, text: str) -> Image.Image:
    image = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    _draw_text(draw, (16, height // 2 - 8), text, FONT_TEXT, fill=(170, 170, 170))
    return image


def _paste_with_caption(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    x: int,
    y: int,
    caption: str,
) -> None:
    _draw_text(draw, (x, y), caption, FONT_SMALL, fill=(205, 205, 205))
    canvas.paste(image, (x, y + 22))


def _make_panel(
    runs: list[RunInfo],
    sample: str,
    output: Path,
    *,
    raw_runs_override: list[RunInfo] | None = None,
) -> None:
    if not runs:
        raise SystemExit("No usable runs found.")

    all_runs = list(runs)
    if raw_runs_override is not None:
        known = {run.name for run in all_runs}
        all_runs.extend(run for run in raw_runs_override if run.name not in known)

    panels = {run.name: Image.open(run.panel_path).convert("RGB") for run in all_runs}
    reference_raw_runs = [run for run in runs if run.route == "raw"]
    raw_runs = raw_runs_override if raw_runs_override is not None else reference_raw_runs
    rgb_runs = [run for run in runs if run.route != "raw"]
    reference_run = _choose_reference_run(reference_raw_runs if reference_raw_runs else runs, panels)
    reference = panels[reference_run.name]
    reference_dist_count = _distribution_count(reference)

    margin = 28
    gap = 14
    label_w = 360
    max_dist_w = max(
        _crop_distribution(
            panels[run.name],
            _distribution_index("ram", _distribution_count(panels[run.name])),
        ).width
        for run in (raw_runs if raw_runs else runs)
    )
    shared_dist_w = (
        _crop_distribution(reference, _distribution_index("rgb", reference_dist_count), count=reference_dist_count).width
        + gap
        + _crop_distribution(reference, _distribution_index("input", reference_dist_count), count=reference_dist_count).width
    )
    content_w = max(
        5 * TILE_W + 4 * gap,
        shared_dist_w,
        label_w + TILE_W + max_dist_w + TILE_W + 3 * gap,
        max(1, len(rgb_runs)) * TILE_W + max(0, len(rgb_runs) - 1) * gap,
    )
    width = margin * 2 + content_w

    title_h = 76
    reference_h = 30 + (22 + TILE_H) + 32 + 30 + (22 + DIST_H)
    rgb_h = 0 if not rgb_runs else 42 + 22 + TILE_H
    raw_h = 0 if not raw_runs else 44 + len(raw_runs) * (TILE_H + gap)
    height = margin + title_h + reference_h + gap + rgb_h + gap + raw_h + margin

    canvas = Image.new("RGB", (width, height), (11, 11, 11))
    draw = ImageDraw.Draw(canvas)

    y = margin
    _draw_text(draw, (margin, y), "STF Cross-Experiment Fixed Sample Panel", FONT_TITLE)
    y += 36
    _draw_text(
        draw,
        (margin, y),
        f"sample={sample}  runs={runs[0].name[:9]}..{runs[-1].name[:9]}  epoch=best checkpoint epoch from train.log",
        FONT_TEXT,
        fill=(190, 190, 190),
    )
    y += 40

    _draw_text(draw, (margin, y), f"Shared reference from {reference_run.name} epoch {reference_run.best_epoch:02d}", FONT_SECTION)
    y += 30
    ref_tiles = [
        ("RGB", _crop_tile(reference, 0)),
        ("Input / raw preview", _crop_tile(reference, 1)),
        ("GT depth", _crop_tile(reference, 3)),
        ("Train target", _crop_tile(reference, 4)),
        ("RGB DAv2 baseline", _crop_tile(reference, 5)),
    ]
    x = margin
    for caption, tile in ref_tiles:
        _paste_with_caption(canvas, draw, tile, x, y, caption)
        x += TILE_W + gap
    y += 22 + TILE_H + 32

    _draw_text(draw, (margin, y), "Shared RGB / RAW Distributions", FONT_SECTION)
    y += 30
    x = margin
    shared_dists = [
        (
            "RGB preview distribution",
            _crop_distribution(reference, _distribution_index("rgb", reference_dist_count), count=reference_dist_count),
        ),
        (
            "RAW/Input preview distribution",
            _crop_distribution(reference, _distribution_index("input", reference_dist_count), count=reference_dist_count),
        ),
    ]
    for caption, dist in shared_dists:
        _paste_with_caption(canvas, draw, dist, x, y, caption)
        x += dist.width + gap
    y += 22 + DIST_H + gap

    if rgb_runs:
        _draw_text(draw, (margin, y), "RGB Route: Current Aligned", FONT_SECTION)
        y += 30
        x = margin
        for run in rgb_runs:
            panel = panels[run.name]
            tile = _crop_tile(panel, _tile_count(panel) - 1)
            metric = "n/a" if run.best_value is None else f"{run.best_value:.4f}"
            caption = f"{run.name[:9]}  e{run.best_epoch:02d}  val {metric}"
            _paste_with_caption(canvas, draw, tile, x, y, caption)
            x += TILE_W + gap
        y += 22 + TILE_H + gap

    if raw_runs:
        _draw_text(draw, (margin, y), "RAW Route: RAM Output, RAM Output Distribution, Current Aligned", FONT_SECTION)
        y += 36
        for run in raw_runs:
            panel = panels[run.name]
            dist_count = _distribution_count(panel)
            x = margin
            _wrapped_label(draw, x, y + 6, label_w, run)
            x += label_w + gap

            ram = _crop_tile(panel, 2)
            dist = _crop_distribution(panel, _distribution_index("ram", dist_count), count=dist_count)
            current = _crop_tile(panel, _tile_count(panel) - 1)
            canvas.paste(ram, (x, y))
            x += TILE_W + gap
            canvas.paste(dist, (x, y + (TILE_H - dist.height) // 2))
            x += dist.width + gap
            canvas.paste(current, (x, y))
            y += TILE_H + gap

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", type=Path, default=Path("finetune_stf/exp"))
    parser.add_argument("--start", default="0521_0012")
    parser.add_argument("--end", default="0521_1004")
    parser.add_argument(
        "--raw-run-prefix",
        action="append",
        help=(
            "Run prefix for the RAW route section. Can be repeated; each prefix may match multiple runs. "
            "Defaults to the curated 0521/0522 RAW comparison list."
        ),
    )
    parser.add_argument(
        "--use-range-raw-runs",
        action="store_true",
        help="Use RAW runs from --start/--end instead of the default curated RAW route list.",
    )
    parser.add_argument("--sample", action="append", help="Sample stem. Can be passed multiple times.")
    parser.add_argument("--sample-list", type=Path, help="Text file with one sample stem per line.")
    parser.add_argument("--all-samples", action="store_true", help="Generate all common best-epoch fixed-viz samples.")
    parser.add_argument("--output", type=Path, help="Output file for a single-sample run.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("finetune_stf/exp/cross_exp_panels_best"),
        help="Output directory for batch mode or single-sample runs without --output.",
    )
    args = parser.parse_args()

    raw_run_prefixes = None
    if not args.use_range_raw_runs:
        raw_run_prefixes = list(args.raw_run_prefix or DEFAULT_RAW_RUN_PREFIXES)

    samples = list(args.sample or [])
    if args.sample_list is not None:
        with args.sample_list.open("r", encoding="utf-8") as handle:
            samples.extend(line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#"))
    if args.all_samples:
        samples.extend(_discover_common_samples(args.exp_root, args.start, args.end, raw_run_prefixes))

    samples = sorted(dict.fromkeys(samples))
    if not samples:
        raise SystemExit("No samples requested. Use --sample, --sample-list, or --all-samples.")
    if args.output is not None and len(samples) != 1:
        raise SystemExit("--output can only be used with exactly one sample.")

    print(f"[INFO] generating {len(samples)} sample panel(s)")
    for sample in samples:
        output = args.output or args.output_dir / f"{sample}_cross_exp_best_panel.png"
        print(f"[SAMPLE] {sample}")
        runs, warnings = _collect_runs(args.exp_root, args.start, args.end, sample)
        raw_runs = None
        if raw_run_prefixes is not None:
            raw_runs, raw_warnings = _collect_runs_by_prefixes(args.exp_root, raw_run_prefixes, sample)
            warnings.extend(raw_warnings)
        for warning in warnings:
            print(f"[WARN] {warning}")
        for run in runs:
            metric = "n/a" if run.best_value is None else f"{run.best_value:.4f}"
            panel = Image.open(run.panel_path)
            tile_count = _tile_count(panel)
            dist_count = _distribution_count(panel)
            print(
                f"[RUN] {run.name} route={run.route} input={run.input_type} "
                f"best_epoch={run.best_epoch:02d} val_abs_rel={metric} "
                f"tiles={tile_count} dist_blocks={dist_count}"
            )
        if raw_runs is not None:
            print("[RAW ROUTE]")
            for run in raw_runs:
                metric = "n/a" if run.best_value is None else f"{run.best_value:.4f}"
                print(f"[RAW] {run.name} best_epoch={run.best_epoch:02d} val_abs_rel={metric}")
        _make_panel(runs, sample, output, raw_runs_override=raw_runs)
        print(f"[OK] wrote {output}")


if __name__ == "__main__":
    main()
