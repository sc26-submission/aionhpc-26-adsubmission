from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = Path("polaris_2025_analysis/results")
OUTPUT_DIR.mkdir(exist_ok=True)

TIMELINE_CSV = OUTPUT_DIR / "polaris_2025_operational_capacity_timeline.csv"

MONTHLY_OUTPUT_PDF = OUTPUT_DIR / "polaris_monthly_unallocated_capacity.pdf"
MONTHLY_OUTPUT_PNG = OUTPUT_DIR / "polaris_monthly_unallocated_capacity.png"
DYNAMICS_OUTPUT_PDF = OUTPUT_DIR / "polaris_capacity_change_cdf.pdf"
DYNAMICS_OUTPUT_PNG = OUTPUT_DIR / "polaris_capacity_change_cdf.png"

MONTHLY_STATS_CSV = OUTPUT_DIR / "polaris_monthly_unallocated_capacity_stats.csv"
DYNAMICS_STATS_CSV = OUTPUT_DIR / "polaris_capacity_state_durations.csv"

EXPECTED_START = pd.Timestamp("2025-01-01 00:00:00")
EXPECTED_END = pd.Timestamp("2026-01-01 00:00:00")
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


# ============================================================
# Helpers
# ============================================================

def separator(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    parsed = (
        series.astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if parsed.isna().any():
        raise ValueError("Could not parse some machine_down values.")
    return parsed


def parse_timestamp(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, format="mixed", errors="raise")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="raise")


def weighted_quantile(values, weights, quantiles) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    if not len(values):
        raise ValueError("Cannot compute weighted quantiles on empty data.")
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")

    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    total = cumulative[-1]

    if total <= 0:
        raise ValueError("Total weight must be positive.")

    indices = np.searchsorted(cumulative, quantiles * total, side="left")
    return values[np.minimum(indices, len(values) - 1)]


def weighted_tukey_statistics(values, weights) -> dict:
    values = np.asarray(values, dtype=float)
    q1, median, q3 = weighted_quantile(values, weights, [0.25, 0.50, 0.75])
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    lower, upper = values[values >= lower_fence], values[values <= upper_fence]

    return {
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "whisker_low": lower.min() if len(lower) else values.min(),
        "whisker_high": upper.max() if len(upper) else values.max(),
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
    }


# ============================================================
# Timeline loading
# ============================================================

def load_timeline(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")

    timeline = pd.read_csv(path, low_memory=False)
    required = {
        "start",
        "end",
        "duration_seconds",
        "machine_down",
        "unallocated_operational_nodes",
    }

    missing = required - set(timeline.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    timeline["start"] = parse_timestamp(timeline["start"])
    timeline["end"] = parse_timestamp(timeline["end"])
    timeline["duration_seconds"] = pd.to_numeric(
        timeline["duration_seconds"], errors="raise"
    )
    timeline["unallocated_operational_nodes"] = pd.to_numeric(
        timeline["unallocated_operational_nodes"], errors="raise"
    )
    timeline["machine_down"] = parse_bool_series(timeline["machine_down"])
    timeline = timeline.sort_values(["start", "end"]).reset_index(drop=True)

    if timeline["start"].min() != EXPECTED_START:
        raise ValueError(f"Unexpected timeline start: {timeline['start'].min()}")
    if timeline["end"].max() != EXPECTED_END:
        raise ValueError(f"Unexpected timeline end: {timeline['end'].max()}")

    previous_ends = timeline["end"].iloc[:-1].reset_index(drop=True)
    next_starts = timeline["start"].iloc[1:].reset_index(drop=True)

    if (previous_ends != next_starts).any():
        raise ValueError("Timeline contains gaps or overlaps.")
    if (timeline["duration_seconds"] <= 0).any():
        raise ValueError("Timeline contains non-positive durations.")

    return timeline


# ============================================================
# Figure 1: monthly unallocated capacity
# ============================================================

def split_monthly_intervals(operational: pd.DataFrame) -> pd.DataFrame:
    segments = []

    for row in operational.itertuples(index=False):
        current = row.start

        while current < row.end:
            month_start = current.to_period("M").to_timestamp()
            segment_end = min(row.end, month_start + pd.offsets.MonthBegin(1))
            seconds = (segment_end - current).total_seconds()

            if seconds > 0:
                segments.append(
                    {
                        "month": month_start,
                        "unallocated_nodes": row.unallocated_operational_nodes,
                        "duration_seconds": seconds,
                    }
                )
            current = segment_end

    result = pd.DataFrame(segments)
    if result.empty:
        raise ValueError("No monthly segments were produced.")
    return result


def compute_monthly_stats(segments: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for month, group in segments.groupby("month"):
        stats = weighted_tukey_statistics(
            group["unallocated_nodes"], group["duration_seconds"]
        )
        rows.append(
            {
                "month": month,
                "whisker_low": stats["whisker_low"],
                "q1": stats["q1"],
                "median": stats["median"],
                "q3": stats["q3"],
                "whisker_high": stats["whisker_high"],
                "iqr": stats["iqr"],
                "operational_hours": group["duration_seconds"].sum() / 3600,
            }
        )

    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def plot_monthly_capacity(stats: pd.DataFrame) -> None:
    box_stats = [
        {
            "label": row.month.strftime("%b"),
            "whislo": row.whisker_low,
            "q1": row.q1,
            "med": row.median,
            "q3": row.q3,
            "whishi": row.whisker_high,
            "fliers": [],
        }
        for row in stats.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)
    ax.bxp(
        box_stats,
        showfliers=False,
        patch_artist=True,
        widths=0.58,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.9},
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )
    ax.set(xlabel="Month", ylabel="Unallocated nodes")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linewidth=0.45, alpha=0.20)
    ax.set_axisbelow(True)

    fig.savefig(MONTHLY_OUTPUT_PDF)
    fig.savefig(MONTHLY_OUTPUT_PNG, dpi=500)
    plt.close(fig)


# ============================================================
# Figure 2: capacity-state duration ECDF
# ============================================================

def build_capacity_states(timeline: pd.DataFrame) -> pd.DataFrame:
    states, current = [], None

    for row in timeline.itertuples(index=False):
        if row.machine_down:
            if current is not None:
                states.append(current)
                current = None
            continue

        free_nodes = int(row.unallocated_operational_nodes)

        if current is None:
            current = {
                "start": row.start,
                "end": row.end,
                "unallocated_nodes": free_nodes,
            }
            continue

        if free_nodes == current["unallocated_nodes"] and row.start == current["end"]:
            current["end"] = row.end
        else:
            states.append(current)
            current = {
                "start": row.start,
                "end": row.end,
                "unallocated_nodes": free_nodes,
            }

    if current is not None:
        states.append(current)

    states = pd.DataFrame(states)
    if states.empty:
        raise ValueError("No operational capacity states were produced.")

    states["duration_seconds"] = (states["end"] - states["start"]).dt.total_seconds()
    if (states["duration_seconds"] <= 0).any():
        raise ValueError("Found non-positive capacity-state duration.")
    return states


def duration_formatter(value, _position):
    labels = {
        1: "1 s",
        10: "10 s",
        60: "1 min",
        600: "10 min",
        3600: "1 h",
        36000: "10 h",
    }
    return next(
        (label for tick, label in labels.items() if np.isclose(value, tick)),
        "",
    )


def plot_capacity_dynamics(states: pd.DataFrame) -> None:
    durations = np.sort(states["duration_seconds"].to_numpy())
    cdf = np.arange(1, len(durations) + 1) / len(durations)
    max_duration = float(durations.max())

    fig, ax = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)
    ax.step(durations, cdf, where="post", linewidth=1.5, color="black")
    ax.set_xscale("log")
    ax.set_xlabel("Time until unallocated capacity changes")
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0, 1.01)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(left=1, right=max_duration)

    candidate_ticks = [1, 10, 60, 600, 3600, 36000]
    ax.set_xticks([tick for tick in candidate_ticks if tick <= max_duration])
    ax.xaxis.set_major_formatter(FuncFormatter(duration_formatter))

    ax.grid(which="major", linewidth=0.45, alpha=0.20)
    ax.grid(which="minor", visible=False)
    ax.set_axisbelow(True)

    fig.savefig(DYNAMICS_OUTPUT_PDF)
    fig.savefig(DYNAMICS_OUTPUT_PNG, dpi=500)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    timeline = load_timeline(TIMELINE_CSV)
    operational = timeline.loc[~timeline["machine_down"]].copy()
    operational_hours = operational["duration_seconds"].sum() / 3600

    print(f"Operational time: {operational_hours:,.2f} h")

    monthly_stats = compute_monthly_stats(
        split_monthly_intervals(operational)
    )
    monthly_stats.to_csv(MONTHLY_STATS_CSV, index=False)

    separator("MONTHLY TIME-WEIGHTED TUKEY BOXPLOT STATISTICS")
    for row in monthly_stats.itertuples(index=False):
        print(
            f"{row.month.strftime('%b'):>3s} "
            f"low={row.whisker_low:6.0f} "
            f"q1={row.q1:6.0f} "
            f"median={row.median:6.0f} "
            f"q3={row.q3:6.0f} "
            f"high={row.whisker_high:6.0f}"
        )

    plot_monthly_capacity(monthly_stats)

    states = build_capacity_states(timeline)
    states.to_csv(DYNAMICS_STATS_CSV, index=False)

    separator("CAPACITY STATE DURATIONS")
    print(f"Number of states: {len(states):,}")

    for q, value in states["duration_seconds"].quantile(QUANTILES).items():
        print(f"p{int(q * 100):02d}: {value:,.1f} s")

    plot_capacity_dynamics(states)

    separator("OUTPUT")
    for path in (
        MONTHLY_OUTPUT_PDF,
        MONTHLY_OUTPUT_PNG,
        DYNAMICS_OUTPUT_PDF,
        DYNAMICS_OUTPUT_PNG,
        MONTHLY_STATS_CSV,
        DYNAMICS_STATS_CSV,
    ):
        print(path)


if __name__ == "__main__":
    main()