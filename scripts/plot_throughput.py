#!/usr/bin/env python3
"""Generate the paper throughput figure from an OPSERVE evaluation summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKLOADS = ["Conversation", "Synthetic", "Tool&Agent"]
MODELS = [
    ("Llama-3.1-8B", "Llama-8B"),
    ("Qwen3-8B", "Qwen-8B"),
    ("Qwen3-14B", "Qwen-14B"),
]
POLICIES = ["Persistent-Only", "Static P/D", "OPSERVE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot completed-request throughput from summary.csv."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("opserve-simulation-exps/logs/evaluation_matrix/summary.csv"),
        help="Evaluation summary CSV produced by run_evaluation.py.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("throughput_three_workloads"),
        help="Output path prefix; .pdf and .png are added automatically.",
    )
    return parser.parse_args()


def load_values(path: Path) -> dict[str, dict[str, list[float]]]:
    frame = pd.read_csv(path)
    # Accept legacy summary files produced before the OPSERVE rename.
    if "policy_label" in frame.columns:
        frame["policy_label"] = frame["policy_label"].replace({"MalleServe": "OPSERVE"})
    required = {"workload_label", "model_label", "policy_label", "mean_throughput_rps"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"summary CSV is missing required columns: {sorted(missing)}")

    values: dict[str, dict[str, list[float]]] = {}
    for workload in WORKLOADS:
        values[workload] = {}
        for policy in POLICIES:
            row_values: list[float] = []
            for model_label, _ in MODELS:
                rows = frame.loc[
                    (frame["workload_label"] == workload)
                    & (frame["model_label"] == model_label)
                    & (frame["policy_label"] == policy),
                    "mean_throughput_rps",
                ]
                if len(rows) != 1:
                    raise ValueError(
                        f"expected one row for {workload}/{model_label}/{policy}, "
                        f"found {len(rows)}"
                    )
                row_values.append(float(rows.iloc[0]))
            values[workload][policy] = row_values
    return values


def main() -> None:
    args = parse_args()
    data = load_values(args.summary_csv)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "hatch.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )

    models = [display for _, display in MODELS]
    display_workloads = {
        "Conversation": "Conversation",
        "Synthetic": "Synthetic",
        "Tool&Agent": "Tool & Agent",
    }
    styles = {
        "Persistent-Only": {"color": "#B8B8B8", "hatch": "///"},
        "Static P/D": {"color": "#E69F00", "hatch": "\\\\"},
        "OPSERVE": {"color": "#0072B2", "hatch": "xx"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), sharey=True)
    x = np.arange(len(models))
    width = 0.24
    offsets = {"Persistent-Only": -width, "Static P/D": 0.0, "OPSERVE": width}

    for ax, workload in zip(axes, WORKLOADS):
        for policy in POLICIES:
            ax.bar(
                x + offsets[policy],
                data[workload][policy],
                width=width,
                label=policy,
                color=styles[policy]["color"],
                hatch=styles[policy]["hatch"],
                edgecolor="black",
                linewidth=0.65,
            )
        ax.set_title(display_workloads[workload], fontsize=10.0, fontweight="bold", pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9.0)
        ax.tick_params(axis="x", pad=3, width=0.7, length=3.0)
        ax.set_ylim(0, 7.1)
        ax.set_yticks(np.arange(0, 8, 1))
        ax.tick_params(axis="y", labelsize=9.0, pad=2, width=0.7, length=3.0)
        ax.grid(axis="y", linewidth=0.45, alpha=0.20)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.55, len(models) - 0.45)

    axes[0].set_ylabel("Throughput (req/s)", fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        fontsize=9.0,
        handlelength=1.5,
        handletextpad=0.45,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.18, top=0.79, wspace=0.16)

    fig.savefig(args.output_prefix.with_suffix(".pdf"))
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
