#!/usr/bin/env python3
"""Build full multi-split analysis artifacts for one or more training runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPLITS = ("kitti", "nyu", "eth3d_fast", "robotcar_fast", "robotcar_night_fast")
SPLIT_LABELS = {
    "kitti": "KITTI val (RGB checkpoint-decoder)",
    "nyu": "NYUv2 val (RGB checkpoint-decoder)",
    "eth3d_fast": "ETH3D Fast",
    "robotcar_fast": "RobotCar Day Fast",
    "robotcar_night_fast": "RobotCar Night Fast",
}
SPLIT_COLORS = {
    "kitti": "#1f77b4",
    "nyu": "#ff7f0e",
    "eth3d_fast": "#d62728",
    "robotcar_fast": "#2ca02c",
    "robotcar_night_fast": "#9467bd",
}

# Fixed DAv2 RGB zero-shot baselines from plans/result/rgb_raw_baseline_fairness_summary.md §0.
DAV2_RGB_ZS = {
    "kitti": {"abs_rel": 0.0802, "d1": 0.9372},
    "nyu": {"abs_rel": 0.0528, "d1": 0.9726},
    "eth3d_fast": {"abs_rel": 0.0555, "d1": 0.9697},
    "robotcar_fast": {"abs_rel": 0.2688, "d1": 0.4419},
    "robotcar_night_fast": {"abs_rel": 0.2396, "d1": 0.5177},
}

RENAME_SPLIT = {
    "kitti_rgb_checkpoint_decoder": "kitti",
    "kitti_val_rgb_checkpoint_decoder": "kitti",
    "nyu_rgb_checkpoint_decoder": "nyu",
    "eth3d_fast": "eth3d_fast",
    "robotcar_fast": "robotcar_fast",
    "robotcar_night_fast": "robotcar_night_fast",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dirs", type=Path, nargs="+", help="Experiment directories containing config.json and train.log.")
    parser.add_argument(
        "--heavy-root",
        type=Path,
        default=Path("/mnt/drive/3333_raw/0000_exp_ckpt"),
        help="Root where large checkpoints/events are stored.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")

    epoch_start_pat = re.compile(r"\[EPOCH\] start epoch=(\d+)/")
    epoch_done_pat = re.compile(
        r"\[EPOCH\] done epoch=(\d+) avg_loss=([0-9.eE+-]+) used_steps=(\d+) elapsed=([0-9:]+)"
    )
    epoch_domain_pat = re.compile(
        r"\[EPOCH\]\[(lod|vkitti)\] avg_loss=([0-9.eE+-]+) steps=(\d+) "
        r"raw_pred_valid_mean=([0-9.eE+-]+) raw_pred_valid_max=([0-9.eE+-]+)"
    )
    pretrain_pat = re.compile(
        r"\[EVAL\]\[pretrain_(kitti_rgb_checkpoint_decoder|nyu_rgb_checkpoint_decoder|eth3d_fast|robotcar_fast|robotcar_night_fast)\] "
        r"abs_rel=([0-9.]+) rmse=([0-9.]+) silog=([0-9.]+) "
        r"d1=([0-9.]+) d2=([0-9.]+) d3=([0-9.]+)"
    )
    eval_pat = re.compile(
        r"\[EVAL\]\[(kitti_val_rgb_checkpoint_decoder|kitti_rgb_checkpoint_decoder|nyu_rgb_checkpoint_decoder|eth3d_fast|robotcar_fast|robotcar_night_fast)\] "
        r"abs_rel=([0-9.]+) rmse=([0-9.]+) silog=([0-9.]+) "
        r"d1=([0-9.]+) d2=([0-9.]+) d3=([0-9.]+)"
    )
    train_pat = re.compile(
        r"\[TRAIN\] epoch=(\d+) micro_step=(\d+)/(\d+) opt_step=(\d+)/(\d+) "
        r"lr=([0-9.eE+-]+) loss=([0-9.eE+-]+) running_avg=([0-9.eE+-]+)"
    )
    model_pat = re.compile(r"\[MODEL\] total_params=(\d+) trainable_params=(\d+) frozen_params=(\d+)")

    epochs: dict[int, dict[str, Any]] = {}
    pretrain: dict[str, dict[str, Any]] = {}
    train_points: list[dict[str, Any]] = []
    model_params: dict[str, int] | None = None
    current_epoch = -1

    for line in text.splitlines():
        m = model_pat.search(line)
        if m:
            model_params = {
                "total_params": int(m.group(1)),
                "trainable_params": int(m.group(2)),
                "frozen_params": int(m.group(3)),
            }
            continue

        m = epoch_start_pat.search(line)
        if m:
            current_epoch = int(m.group(1))
            epochs.setdefault(current_epoch, {"metrics": {}, "domain_losses": {}})
            continue

        m = pretrain_pat.search(line)
        if m:
            split = RENAME_SPLIT[m.group(1)]
            pretrain[split] = {
                "abs_rel": float(m.group(2)),
                "rmse": float(m.group(3)),
                "silog": float(m.group(4)),
                "d1": float(m.group(5)),
                "d2": float(m.group(6)),
                "d3": float(m.group(7)),
                "_source": "train.log pretrain_eval",
            }
            continue

        m = train_pat.search(line)
        if m:
            micro_step = int(m.group(2))
            micro_total = int(m.group(3))
            train_points.append(
                {
                    "epoch": int(m.group(1)),
                    "micro_step": micro_step,
                    "micro_total": micro_total,
                    "opt_step": int(m.group(4)),
                    "opt_total": int(m.group(5)),
                    "lr": float(m.group(6)),
                    "loss": float(m.group(7)),
                    "running_avg": float(m.group(8)),
                    "epoch_progress": micro_step / micro_total if micro_total else None,
                }
            )
            continue

        m = epoch_done_pat.search(line)
        if m:
            ep = int(m.group(1))
            epochs.setdefault(ep, {"metrics": {}, "domain_losses": {}})
            epochs[ep]["avg_loss"] = float(m.group(2))
            epochs[ep]["used_steps"] = int(m.group(3))
            epochs[ep]["elapsed"] = m.group(4)
            current_epoch = ep
            continue

        m = epoch_domain_pat.search(line)
        if m and current_epoch >= 0:
            epochs.setdefault(current_epoch, {"metrics": {}, "domain_losses": {}})
            epochs[current_epoch]["domain_losses"][m.group(1)] = {
                "avg_loss": float(m.group(2)),
                "steps": int(m.group(3)),
                "raw_pred_valid_mean": float(m.group(4)),
                "raw_pred_valid_max": float(m.group(5)),
            }
            continue

        m = eval_pat.search(line)
        if m and current_epoch >= 0:
            split = RENAME_SPLIT[m.group(1)]
            epochs.setdefault(current_epoch, {"metrics": {}, "domain_losses": {}})
            epochs[current_epoch]["metrics"][split] = {
                "abs_rel": float(m.group(2)),
                "rmse": float(m.group(3)),
                "silog": float(m.group(4)),
                "d1": float(m.group(5)),
                "d2": float(m.group(6)),
                "d3": float(m.group(7)),
            }

    ordered_epochs = []
    for epoch in sorted(epochs):
        row = epochs[epoch]
        if not row.get("metrics") and "avg_loss" not in row:
            continue
        row["epoch"] = epoch
        ordered_epochs.append(row)

    return {
        "pretrain": pretrain,
        "epochs": ordered_epochs,
        "train_points": train_points,
        "model_params": model_params,
    }


def init_metric(parsed: dict[str, Any], split: str, metric: str) -> float | None:
    fixed = DAV2_RGB_ZS.get(split, {}).get(metric)
    if fixed is not None:
        return fixed
    return parsed["pretrain"].get(split, {}).get(metric)


def best_per_split(parsed: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in SPLITS:
        abs_values: list[tuple[int, float]] = []
        d1_values: list[tuple[int, float]] = []
        for item in parsed["epochs"]:
            metrics = item["metrics"].get(split)
            if not metrics:
                continue
            if metrics.get("abs_rel") is not None:
                abs_values.append((item["epoch"], metrics["abs_rel"]))
            if metrics.get("d1") is not None:
                d1_values.append((item["epoch"], metrics["d1"]))

        best_abs = min((v for _, v in abs_values), default=None)
        best_d1 = max((v for _, v in d1_values), default=None)
        best_abs_epochs = [ep for ep, v in abs_values if best_abs is not None and math.isclose(v, best_abs, abs_tol=5e-5)]
        best_d1_epochs = [ep for ep, v in d1_values if best_d1 is not None and math.isclose(v, best_d1, abs_tol=5e-5)]
        last_metrics = {}
        for item in reversed(parsed["epochs"]):
            metrics = item["metrics"].get(split)
            if metrics:
                last_metrics = metrics
                break

        out[split] = {
            "dav2_rgb_zs_abs_rel": init_metric(parsed, split, "abs_rel"),
            "dav2_rgb_zs_d1": init_metric(parsed, split, "d1"),
            "raw_pipeline_pretrain_abs_rel": parsed["pretrain"].get(split, {}).get("abs_rel"),
            "raw_pipeline_pretrain_d1": parsed["pretrain"].get(split, {}).get("d1"),
            "best_abs_rel": best_abs,
            "best_abs_rel_epochs": best_abs_epochs,
            "best_abs_rel_epoch": best_abs_epochs[0] if best_abs_epochs else None,
            "best_d1": best_d1,
            "best_d1_epochs": best_d1_epochs,
            "best_d1_epoch": best_d1_epochs[0] if best_d1_epochs else None,
            "last_abs_rel": last_metrics.get("abs_rel"),
            "last_d1": last_metrics.get("d1"),
        }
    return out


def write_epoch_csv(parsed: dict[str, Any], path: Path) -> None:
    fieldnames = ["epoch", "avg_loss", "used_steps"]
    for split in SPLITS:
        for metric in ("abs_rel", "rmse", "silog", "d1"):
            fieldnames.append(f"{split}_{metric}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in parsed["epochs"]:
            row: dict[str, Any] = {
                "epoch": item["epoch"],
                "avg_loss": item.get("avg_loss"),
                "used_steps": item.get("used_steps"),
            }
            for split in SPLITS:
                metrics = item["metrics"].get(split) or {}
                for metric in ("abs_rel", "rmse", "silog", "d1"):
                    row[f"{split}_{metric}"] = metrics.get(metric)
            writer.writerow(row)


def write_train_points_csv(parsed: dict[str, Any], path: Path) -> None:
    fieldnames = ["epoch", "micro_step", "micro_total", "opt_step", "opt_total", "lr", "loss", "running_avg", "epoch_progress"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in parsed["train_points"]:
            writer.writerow({key: row.get(key) for key in fieldnames})


def plot_metric(parsed: dict[str, Any], *, metric: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=130)
    for split in SPLITS:
        xs = []
        ys = []
        init_value = init_metric(parsed, split, metric)
        if init_value is not None:
            xs.append(-1)
            ys.append(init_value)
        for item in parsed["epochs"]:
            split_metrics = item["metrics"].get(split)
            if not split_metrics or split_metrics.get(metric) is None:
                continue
            xs.append(item["epoch"])
            ys.append(split_metrics[metric])
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=1.4, markersize=4, color=SPLIT_COLORS[split], label=SPLIT_LABELS[split])
        if init_value is not None:
            ax.annotate("init", (xs[0], ys[0]), textcoords="offset points", xytext=(4, 4), fontsize=7, color=SPLIT_COLORS[split])
    ax.set_xlabel("epoch (-1 = eval init)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_loss(parsed: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=130)

    train_x = [row["epoch"] + row["epoch_progress"] for row in parsed["train_points"] if row.get("epoch_progress") is not None]
    running_avg = [row["running_avg"] for row in parsed["train_points"] if row.get("epoch_progress") is not None]
    if train_x:
        ax.plot(train_x, running_avg, linewidth=1.0, alpha=0.55, color="#777", label="train running avg")

    epoch_x = [item["epoch"] for item in parsed["epochs"] if "avg_loss" in item]
    epoch_loss = [item["avg_loss"] for item in parsed["epochs"] if "avg_loss" in item]
    if epoch_x:
        ax.plot(epoch_x, epoch_loss, marker="o", linewidth=1.5, color="#222", label="epoch avg")

    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fmt_num(value: Any, n: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{n}f}"


def _format_loss_label(cfg: dict[str, Any]) -> str:
    loss_type = str(cfg.get("loss_type"))
    if loss_type == "ssi_grad" and cfg.get("loss_lambda_grad") is not None:
        return f"{loss_type}+lambda_grad={cfg.get('loss_lambda_grad')}"
    return loss_type


def fmt_epochs(epochs: list[int]) -> str:
    if not epochs:
        return ""
    return " (e" + "/e".join(str(ep) for ep in epochs) + ")"


def setup_lines(cfg: dict[str, Any], parsed: dict[str, Any], params: dict[str, int] | None, heavy_dir: Path) -> list[str]:
    bs = cfg.get("bs")
    accum = cfg.get("accum_steps")
    eff_bs = bs * accum if bs is not None and accum is not None else None
    size = f"{cfg.get('input_height')}x{cfg.get('input_width')}"
    feature_keys = ", ".join(f"`{key}`" for key in cfg.get("bridge_feature_keys", []))
    bridge_layers = ", ".join(str(x) for x in cfg.get("bridge_layers", []))
    configured_epochs = cfg.get("epochs")
    completed_epochs = [item["epoch"] for item in parsed.get("epochs", []) if item.get("avg_loss") is not None]
    if configured_epochs is not None and len(completed_epochs) != configured_epochs:
        last_complete = max(completed_epochs) if completed_epochs else None
        epoch_line = f"- Epochs complete/configured: `{len(completed_epochs)}/{configured_epochs}`"
        if last_complete is not None:
            epoch_line += f" (last complete e{last_complete})"
    else:
        epoch_line = f"- Epochs parsed/configured: `{configured_epochs}`"
    lines = [
        f"- Stage: `{cfg.get('stage')}` (LOD day + LOD night + VKITTI2, `lod_per_vkitti={cfg.get('lod_per_vkitti')}`)",
        f"- Input type: `{cfg.get('input_type')}` at `{size}`",
        f"- Interface: `{cfg.get('rgb_interface_mode')}`; bridge keys: {feature_keys}; bridge layers: `[{bridge_layers}]`",
        f"- DAv2 train mode: `{cfg.get('dav2_train_mode')}`",
        f"- Loss: `{_format_loss_label(cfg)}`; target normalization `{cfg.get('loss_target_normalization')}`",
        epoch_line,
        f"- Batch: `bs={bs}`, `accum={accum}`, `effective_bs={eff_bs}`",
        f"- LR: base `{cfg.get('lr')}`, bridge/adapter `{cfg.get('bridge_lr')}`, LoRA `{cfg.get('lora_lr')}`",
        f"- Canonical best metric: `{cfg.get('best_metric')}`",
        f"- Checkpoint dir: `{heavy_dir}`",
    ]
    if "lora" in str(cfg.get("input_type", "")):
        lines.insert(
            4,
            f"- LoRA: mode `{cfg.get('lora_block_mode')}`, rank `{cfg.get('lora_rank')}`, alpha `{cfg.get('lora_alpha')}`, lr `{cfg.get('lora_lr')}`",
        )
    if params:
        lines.append(
            f"- Params: trainable `{params['trainable_params'] / 1e6:.2f}M` / total `{params['total_params'] / 1e6:.2f}M`"
        )
    return lines


def write_summary_md(exp_dir: Path, cfg: dict[str, Any], parsed: dict[str, Any], best: dict[str, Any], heavy_dir: Path, path: Path) -> None:
    md: list[str] = [f"# Training Run Analysis - `{exp_dir.name}`", "", "## Setup", ""]
    md.extend(setup_lines(cfg, parsed, parsed.get("model_params"), heavy_dir))
    md.extend(
        [
            "",
            "## Eval Summary",
            "",
            "The init columns below use fixed offline DAv2 RGB zero-shot baselines where available; splits without a fixed baseline use the pretrain eval parsed from `train.log`.",
            "",
            "| Split | Init abs_rel | Best abs_rel (epoch) | Last abs_rel | Init d1 | Best d1 (epoch) | Last d1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in SPLITS:
        row = best[split]
        md.append(
            "| {label} | {zs_a} | {best_a}{best_a_ep} | {last_a} | {zs_d} | {best_d}{best_d_ep} | {last_d} |".format(
                label=SPLIT_LABELS[split],
                zs_a=fmt_num(row["dav2_rgb_zs_abs_rel"]),
                best_a=fmt_num(row["best_abs_rel"]),
                best_a_ep=fmt_epochs(row["best_abs_rel_epochs"]),
                last_a=fmt_num(row["last_abs_rel"]),
                zs_d=fmt_num(row["dav2_rgb_zs_d1"]),
                best_d=fmt_num(row["best_d1"]),
                best_d_ep=fmt_epochs(row["best_d1_epochs"]),
                last_d=fmt_num(row["last_d1"]),
            )
        )

    first_loss = next((item.get("avg_loss") for item in parsed["epochs"] if item.get("avg_loss") is not None), None)
    last_loss = next((item.get("avg_loss") for item in reversed(parsed["epochs"]) if item.get("avg_loss") is not None), None)
    md.extend(
        [
            "",
            "## Loss Trend",
            "",
            f"- Epoch avg loss: `{fmt_num(first_loss, 6)}` -> `{fmt_num(last_loss, 6)}`.",
            "- See `loss_curves_full.png` for the full curve.",
            "",
            "## Generated Files",
            "",
            "- `loss_curves_full.png`",
            "- `eval_abs_rel_curves_full.png`",
            "- `eval_d1_curves_full.png`",
            "- `epoch_metrics_full.csv`",
            "- `train_points.csv`",
            "- `summary_full.json`",
            "- `notes_zh.md`",
            "- `../eth3d_fast_rgb_raw_compare_last_5panels_only/panels_5up/`",
            "- `../robotcar_fast_rgb_raw_compare_last_5panels_only/panels_5up/`",
            "- `../robotcar_night_fast_rgb_raw_compare_last_5panels_only/panels_5up/`",
            "- `../kitti_rgb_compare_dav2_vs_ours_2x3/`",
            "",
        ]
    )
    path.write_text("\n".join(md), encoding="utf-8")


def write_notes_zh(exp_dir: Path, heavy_dir: Path, path: Path) -> None:
    md = f"""# 多 split analysis 记录 - `{exp_dir.name}`

## 输出

- 曲线：`loss_curves_full.png`、`eval_abs_rel_curves_full.png`、`eval_d1_curves_full.png`
- 表格：`epoch_metrics_full.csv`、`train_points.csv`
- 汇总：`summary.md`、`summary_full.json`

## 可视化

- ETH3D fast 5-panel：`../eth3d_fast_rgb_raw_compare_last_5panels_only/panels_5up/`
- RobotCar day fast 5-panel：`../robotcar_fast_rgb_raw_compare_last_5panels_only/panels_5up/`
- RobotCar night fast 5-panel：`../robotcar_night_fast_rgb_raw_compare_last_5panels_only/panels_5up/`
- KITTI RGB zero-shot vs ours 2x3：`../kitti_rgb_compare_dav2_vs_ours_2x3/`

## 口径

- 五个 split：KITTI val、NYUv2 val、ETH3D fast、RobotCar day fast、RobotCar night fast。
- 曲线里的 `epoch=-1` 表示 eval init；有离线 DAv2 RGB zero-shot baseline 的 split 使用固定值，NYUv2 使用 train.log 中的 pretrain eval。
- `summary_full.json` 同时保留 train.log 里的 pretrain 数字和用于画图/表格的 eval init baseline。
- checkpoint/event 根目录记录为：`{heavy_dir}`。
"""
    path.write_text(md, encoding="utf-8")


def run_one(exp_dir: Path, heavy_root: Path) -> Path:
    exp_dir = exp_dir.resolve()
    cfg = read_json(exp_dir / "config.json")
    parsed = parse_log(exp_dir / "train.log")
    best = best_per_split(parsed)
    heavy_dir = Path(cfg.get("heavy_save_path") or heavy_root / exp_dir.name)

    out_dir = exp_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_epoch_csv(parsed, out_dir / "epoch_metrics_full.csv")
    write_train_points_csv(parsed, out_dir / "train_points.csv")
    plot_loss(parsed, out_dir / "loss_curves_full.png")
    plot_metric(
        parsed,
        metric="abs_rel",
        ylabel="abs_rel",
        title="abs_rel per epoch (multi-split, x=-1 is eval init)",
        path=out_dir / "eval_abs_rel_curves_full.png",
    )
    plot_metric(
        parsed,
        metric="d1",
        ylabel="d1",
        title="d1 per epoch (multi-split, x=-1 is eval init)",
        path=out_dir / "eval_d1_curves_full.png",
    )

    summary = {
        "exp_dir": str(exp_dir),
        "heavy_dir": str(heavy_dir),
        "splits": list(SPLITS),
        "config": cfg,
        "model_params": parsed.get("model_params"),
        "dav2_rgb_zero_shot": DAV2_RGB_ZS,
        "pretrain_init_from_train_log": parsed["pretrain"],
        "eval_init_baseline": {
            split: {"abs_rel": init_metric(parsed, split, "abs_rel"), "d1": init_metric(parsed, split, "d1")}
            for split in SPLITS
        },
        "best_per_split": best,
        "epochs": parsed["epochs"],
        "train_points_count": len(parsed["train_points"]),
    }
    (out_dir / "summary_full.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_md(exp_dir, cfg, parsed, best, heavy_dir, out_dir / "summary.md")
    write_notes_zh(exp_dir, heavy_dir, out_dir / "notes_zh.md")
    return out_dir


def main() -> None:
    args = parse_args()
    for exp_dir in args.exp_dirs:
        out_dir = run_one(exp_dir, args.heavy_root)
        print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
