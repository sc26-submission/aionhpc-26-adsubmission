#!/usr/bin/env python3
"""Plot active worker capacity from the one-hour OPSERVE resource trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the worker-availability trace.")
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("opserve-simulation-exps/event-traces/one-hour-trace.jsonl"),
        help="Resource-event JSONL file.",
    )
    parser.add_argument("--persistent-workers", type=int, default=8)
    parser.add_argument("--trace-end-s", type=float, default=3600.0)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("resource_trace_workers"),
        help="Output path prefix; .pdf and .png are added automatically.",
    )
    return parser.parse_args()


def load_capacity_trace(path: Path, persistent_workers: int, trace_end_s: float):
    active: set[int] = set()
    times_s = [0.0]
    transient_counts = [0]

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            event_type = event["event_type"]
            worker_ids = [int(x) for x in event.get("worker_ids", [])]
            if event_type == "gain":
                active.update(worker_ids)
            elif event_type == "loss":
                active.difference_update(worker_ids)
            else:
                continue
            times_s.append(float(event["timestamp"]))
            transient_counts.append(len(active))

    times_s.append(float(trace_end_s))
    transient_counts.append(transient_counts[-1])
    times_min = [t / 60.0 for t in times_s]
    persistent = [persistent_workers] * len(times_min)
    total = [persistent_workers + n for n in transient_counts]
    return times_min, persistent, total


def main() -> None:
    args = parse_args()
    times_min, persistent, total = load_capacity_trace(
        args.events, args.persistent_workers, args.trace_end_s
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 9.5,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    total_line = ax.step(times_min, total, where="post", linewidth=1.6, label="Total", zorder=3)[0]
    transient_fill = ax.fill_between(
        times_min,
        persistent,
        total,
        step="post",
        alpha=0.18,
        linewidth=0.8,
        label="Transient workers",
        zorder=2,
    )
    ax.fill_between(
        times_min,
        0,
        persistent,
        step="post",
        facecolor="white",
        edgecolor="0.85",
        hatch="///",
        linewidth=0.0,
        zorder=1,
    )
    ax.text(
        31,
        args.persistent_workers / 2,
        "Persistent workers",
        ha="center",
        va="center",
        fontsize=10.0,
        style="italic",
    )

    ax.set_xlim(0, args.trace_end_s / 60.0)
    ax.set_ylim(0, max(total) + 1)
    ax.set_xticks(range(0, int(args.trace_end_s / 60.0) + 1, 10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.set_xlabel("Time (min)", labelpad=2)
    ax.set_ylabel("Active workers", labelpad=2)
    ax.legend(
        handles=[total_line, transient_fill],
        labels=["Total", "Transient workers"],
        loc="upper left",
        ncol=1,
        frameon=True,
        borderpad=0.25,
        handlelength=1.6,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.21, top=0.97)
    fig.savefig(args.output_prefix.with_suffix(".pdf"))
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    main()
