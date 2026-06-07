#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

EXPS = [
    {
        "id": "0606_0330",
        "label": "0606_0330 decoder",
        "name": "0606_0330_rod_night_rawram3_identity_dav2s_ram_backbone_ld09_decoder_e10",
        "color": "#1f77b4",
    },
    {
        "id": "0606_1431",
        "label": "0606_1431 lowlr",
        "name": "0606_1431_rod_night_rawram3_identity_dav2s_ram_backbone_lowlr_ld09_e10",
        "color": "#d62728",
    },
]

TS_RE = re.compile(r"^\[(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)\]")
TRAIN_RE = re.compile(
    r"\[TRAIN\] epoch=(?P<epoch>\d+) "
    r"micro_step=(?P<micro_step>\d+)/(?P<micro_total>\d+) "
    r"opt_step=(?P<opt_step>\d+)/(?P<opt_total>\d+) "
    r"lr=(?P<lr>[-+0-9.eE]+) "
    r"loss=(?P<loss>[-+0-9.eE]+) "
    r"running_avg=(?P<running_avg>[-+0-9.eE]+) "
    r"rod_avg=(?P<rod_avg>[-+0-9.eE]+)"
)
EPOCH_DONE_RE = re.compile(
    r"\[EPOCH\] done epoch=(?P<epoch>\d+) "
    r"avg_loss=(?P<avg_loss>[-+0-9.eE]+) "
    r"used_steps=(?P<used_steps>\d+) elapsed=(?P<elapsed>\S+)"
)
EPOCH_ROD_RE = re.compile(
    r"\[EPOCH\]\[rod\] avg_loss=(?P<avg_loss>[-+0-9.eE]+) "
    r"steps=(?P<steps>\d+) "
    r"raw_pred_valid_mean=(?P<raw_pred_valid_mean>[-+0-9.eE]+) "
    r"raw_pred_valid_max=(?P<raw_pred_valid_max>[-+0-9.eE]+)"
)
EVAL_DONE_RE = re.compile(
    r"\[EVAL\]\[rod_night_val\] done epoch=(?P<epoch>\d+) "
    r"samples=(?P<samples>\d+) "
    r"abs_rel=(?P<abs_rel>[-+0-9.eE]+) "
    r"rmse=(?P<rmse>[-+0-9.eE]+) "
    r"silog=(?P<silog>[-+0-9.eE]+) "
    r"d1=(?P<d1>[-+0-9.eE]+)"
)
EVAL_SUMMARY_RE = re.compile(
    r"\[EVAL\]\[rod_night_val\] "
    r"abs_rel=(?P<abs_rel>[-+0-9.eE]+) "
    r"rmse=(?P<rmse>[-+0-9.eE]+) "
    r"silog=(?P<silog>[-+0-9.eE]+) "
    r"d1=(?P<d1>[-+0-9.eE]+) "
    r"d2=(?P<d2>[-+0-9.eE]+) "
    r"d3=(?P<d3>[-+0-9.eE]+)"
)


def f(value: str) -> float:
    return float(value)


def parse_log(exp: dict[str, str]) -> tuple[list[dict], list[dict]]:
    log_path = ROOT / "finetune_stf" / "exp" / exp["name"] / "train.log"
    train_rows: list[dict] = []
    epochs: dict[int, dict] = {}
    last_epoch_done: int | None = None
    last_eval_epoch: int | None = None

    for line in log_path.read_text().splitlines():
        ts_match = TS_RE.search(line)
        timestamp = ts_match.group("ts") if ts_match else ""

        if match := TRAIN_RE.search(line):
            row = match.groupdict()
            epoch = int(row["epoch"])
            opt_step = int(row["opt_step"])
            opt_total = int(row["opt_total"])
            train_rows.append(
                {
                    "exp_id": exp["id"],
                    "exp_label": exp["label"],
                    "timestamp": timestamp,
                    "epoch": epoch,
                    "micro_step": int(row["micro_step"]),
                    "micro_total": int(row["micro_total"]),
                    "opt_step": opt_step,
                    "opt_total": opt_total,
                    "global_step": epoch * opt_total + opt_step,
                    "lr": f(row["lr"]),
                    "loss": f(row["loss"]),
                    "running_avg": f(row["running_avg"]),
                    "rod_avg": f(row["rod_avg"]),
                }
            )
            continue

        if match := EPOCH_DONE_RE.search(line):
            row = match.groupdict()
            epoch = int(row["epoch"])
            epochs.setdefault(epoch, {"epoch": epoch})
            epochs[epoch].update(
                {
                    "exp_id": exp["id"],
                    "exp_label": exp["label"],
                    "epoch": epoch,
                    "epoch_timestamp": timestamp,
                    "avg_loss": f(row["avg_loss"]),
                    "used_steps": int(row["used_steps"]),
                    "epoch_elapsed": row["elapsed"],
                }
            )
            last_epoch_done = epoch
            continue

        if match := EPOCH_ROD_RE.search(line):
            if last_epoch_done is not None:
                row = match.groupdict()
                epochs.setdefault(last_epoch_done, {"epoch": last_epoch_done})
                epochs[last_epoch_done].update(
                    {
                        "rod_avg_loss": f(row["avg_loss"]),
                        "rod_steps": int(row["steps"]),
                        "raw_pred_valid_mean": f(row["raw_pred_valid_mean"]),
                        "raw_pred_valid_max": f(row["raw_pred_valid_max"]),
                    }
                )
            continue

        if match := EVAL_DONE_RE.search(line):
            row = match.groupdict()
            epoch = int(row["epoch"])
            epochs.setdefault(epoch, {"epoch": epoch})
            epochs[epoch].update(
                {
                    "exp_id": exp["id"],
                    "exp_label": exp["label"],
                    "epoch": epoch,
                    "eval_timestamp": timestamp,
                    "samples": int(row["samples"]),
                    "abs_rel": f(row["abs_rel"]),
                    "rmse": f(row["rmse"]),
                    "silog": f(row["silog"]),
                    "d1": f(row["d1"]),
                }
            )
            last_eval_epoch = epoch
            continue

        if match := EVAL_SUMMARY_RE.search(line):
            if last_eval_epoch is not None:
                row = match.groupdict()
                epochs.setdefault(last_eval_epoch, {"epoch": last_eval_epoch})
                epochs[last_eval_epoch].update(
                    {
                        "abs_rel": f(row["abs_rel"]),
                        "rmse": f(row["rmse"]),
                        "silog": f(row["silog"]),
                        "d1": f(row["d1"]),
                        "d2": f(row["d2"]),
                        "d3": f(row["d3"]),
                    }
                )

    return train_rows, [epochs[k] for k in sorted(epochs)]


def read_pretrain(exp: dict[str, str]) -> dict:
    path = ROOT / "finetune_stf" / "exp" / exp["name"] / "pretrain_eval.json"
    data = json.loads(path.read_text())
    metrics = data["metrics"]
    return {
        "exp_id": exp["id"],
        "exp_label": exp["label"],
        "split": data.get("split", ""),
        "checkpoint_source": data.get("checkpoint_source", ""),
        "abs_rel": metrics.get("abs_rel"),
        "rmse": metrics.get("rmse"),
        "silog": metrics.get("silog"),
        "d1": metrics.get("d1"),
        "d2": metrics.get("d2"),
        "d3": metrics.get("d3"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pct_drop(start: float, end: float) -> float:
    return (start - end) / start * 100.0


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def plot_loss(train_rows: list[dict], epoch_rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for exp in EXPS:
        train = [r for r in train_rows if r["exp_id"] == exp["id"]]
        epochs = [r for r in epoch_rows if r["exp_id"] == exp["id"]]
        axes[0].plot(
            [r["global_step"] for r in train],
            [r["running_avg"] for r in train],
            marker="o",
            linewidth=2,
            markersize=4,
            color=exp["color"],
            label=exp["label"],
        )
        axes[0].scatter(
            [r["global_step"] for r in train],
            [r["loss"] for r in train],
            s=16,
            color=exp["color"],
            alpha=0.25,
        )
        axes[1].plot(
            [r["epoch"] for r in epochs],
            [r["avg_loss"] for r in epochs],
            marker="o",
            linewidth=2,
            color=exp["color"],
            label=exp["label"],
        )
    axes[0].set_title("Training loss, logged every 500 optimizer steps")
    axes[0].set_xlabel("global optimizer step")
    axes[0].set_ylabel("loss / running_avg")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_title("Epoch average training loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("avg_loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(OUT_DIR / "loss_curves.png", dpi=180)
    plt.close(fig)


def plot_eval(epoch_rows: list[dict]) -> None:
    metrics = [
        ("abs_rel", "AbsRel (lower is better)"),
        ("rmse", "RMSE (lower is better)"),
        ("silog", "SILog (lower is better)"),
        ("d1", "d1 (higher is better)"),
        ("d2", "d2 (higher is better)"),
        ("d3", "d3 (higher is better)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (key, title) in zip(axes.ravel(), metrics):
        for exp in EXPS:
            rows = [r for r in epoch_rows if r["exp_id"] == exp["id"] and key in r]
            ax.plot(
                [r["epoch"] for r in rows],
                [r[key] for r in rows],
                marker="o",
                linewidth=2,
                color=exp["color"],
                label=exp["label"],
            )
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="best")
    fig.savefig(OUT_DIR / "eval_trends.png", dpi=180)
    plt.close(fig)


def plot_absrel_zoom(epoch_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for exp in EXPS:
        rows = [r for r in epoch_rows if r["exp_id"] == exp["id"]]
        xs = [r["epoch"] for r in rows]
        ys = [r["abs_rel"] for r in rows]
        ax.plot(xs, ys, marker="o", linewidth=2, color=exp["color"], label=exp["label"])
        best = min(rows, key=lambda r: r["abs_rel"])
        ax.scatter([best["epoch"]], [best["abs_rel"]], s=90, color=exp["color"], edgecolor="black", zorder=5)
        ax.annotate(
            f"best e{best['epoch']}={best['abs_rel']:.4f}",
            (best["epoch"], best["abs_rel"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )
    ax.set_title("ROD night validation AbsRel trend")
    ax.set_xlabel("epoch")
    ax.set_ylabel("abs_rel")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(OUT_DIR / "absrel_zoom.png", dpi=180)
    plt.close(fig)


def write_summary(epoch_rows: list[dict], pretrain_rows: list[dict]) -> None:
    lines = [
        "# 0606_0330 vs 0606_1431 loss and eval summary",
        "",
        "Source logs: `train.log`; eval split: `rod_night_val`; train points are logged every 500 optimizer steps.",
        "",
        "| exp | best AbsRel | best epoch | final epoch | final AbsRel | final RMSE | final SILog | final d1 | final avg_loss | AbsRel drop e0->best | pretrain AbsRel |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for exp in EXPS:
        rows = [r for r in epoch_rows if r["exp_id"] == exp["id"]]
        best = min(rows, key=lambda r: r["abs_rel"])
        final = max(rows, key=lambda r: r["epoch"])
        pretrain = next(r for r in pretrain_rows if r["exp_id"] == exp["id"])
        lines.append(
            "| {label} | {best_abs} | {best_epoch} | {final_epoch} | {final_abs} | {final_rmse} | {final_silog} | {final_d1} | {final_loss} | {drop:.1f}% | {pretrain_abs} |".format(
                label=exp["label"],
                best_abs=fmt(best.get("abs_rel")),
                best_epoch=best["epoch"],
                final_epoch=final["epoch"],
                final_abs=fmt(final.get("abs_rel")),
                final_rmse=fmt(final.get("rmse")),
                final_silog=fmt(final.get("silog")),
                final_d1=fmt(final.get("d1")),
                final_loss=fmt(final.get("avg_loss"), 6),
                drop=pct_drop(rows[0]["abs_rel"], best["abs_rel"]),
                pretrain_abs=fmt(pretrain.get("abs_rel")),
            )
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- 0606_1431 uses 0.1x lr for the backbone and DAV2 decoder parameter groups compared with 0606_0330, according to `resolved_config.json`.",
            "- Main eval plots omit the pretrain point because its AbsRel is 130.3823 for both experiments and would compress the epoch 0-9 trends.",
            "- Lower is better for AbsRel/RMSE/SILog; higher is better for d1/d2/d3.",
            "",
            "Generated files:",
            "- `loss_curves.png`",
            "- `eval_trends.png`",
            "- `absrel_zoom.png`",
            "- `train_loss_points.csv`",
            "- `epoch_loss_eval.csv`",
            "- `pretrain_eval.csv`",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    all_train: list[dict] = []
    all_epochs: list[dict] = []
    all_pretrain: list[dict] = []
    for exp in EXPS:
        train_rows, epoch_rows = parse_log(exp)
        all_train.extend(train_rows)
        all_epochs.extend(epoch_rows)
        all_pretrain.append(read_pretrain(exp))

    write_csv(OUT_DIR / "train_loss_points.csv", all_train)
    write_csv(OUT_DIR / "epoch_loss_eval.csv", all_epochs)
    write_csv(OUT_DIR / "pretrain_eval.csv", all_pretrain)
    plot_loss(all_train, all_epochs)
    plot_eval(all_epochs)
    plot_absrel_zoom(all_epochs)
    write_summary(all_epochs, all_pretrain)
    print(f"wrote outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
