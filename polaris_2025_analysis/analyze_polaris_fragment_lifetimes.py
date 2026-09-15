#!/usr/bin/env python3
"""
Conservative per-node idle-fragment analysis for the 2025 Polaris trace.

The DJC LOCATION field is truncated for some large jobs. This script never
assumes that an unlisted node was idle. Any job whose parsed LOCATION contains
fewer distinct valid node IDs than NODES_USED is treated as location-censored
for the whole cluster during that job interval.

The script reports:
  * LOCATION-field quality / truncation statistics
  * aggregate unallocated operational node-hours (LOCATION-independent)
  * fully observed individual-node fragment-lifetime quantiles
  * capacity-weighted fragment-lifetime quantiles
  * the fraction of fragments/capacity that outlive model startup times
  * a conservative lower bound on total unused capacity usable after startup
  * a survival plot with startup-time markers

By default it writes only compact summary files. Pass --save-details to also
write the individual confirmed-idle segments and fully observed fragments.
"""

from __future__ import annotations

import argparse
import heapq
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOTAL_NODES = 560
ANALYSIS_START = pd.Timestamp("2025-01-01 00:00:00")
ANALYSIS_END = pd.Timestamp("2026-01-01 00:00:00")

# Examples: x3005c0s19b1n0 and x3005c0s7b0n0
NODE_RE = re.compile(r"^x\d{4}c\d+s\d+b\d+n\d+$")

# Table I startup times from the paper.
STARTUP_THRESHOLDS = {
    "Llama-3.1-8B": 139.0,
    "Qwen3-8B": 215.0,
    "Qwen3-14B": 323.0,
}

Interval = Tuple[pd.Timestamp, pd.Timestamp]


def parse_location(value) -> tuple[list[str], int]:
    """Return distinct valid node IDs and the number of malformed tokens."""
    if not isinstance(value, str) or not value.strip():
        return [], 0

    nodes: list[str] = []
    seen = set()
    malformed = 0
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        if NODE_RE.fullmatch(token):
            if token not in seen:
                nodes.append(token)
                seen.add(token)
        else:
            # A truncated LOCATION often ends with a partial node name.
            malformed += 1
    return nodes, malformed


def merge_sorted(intervals: Iterable[Interval]) -> list[Interval]:
    """Merge intervals that are already sorted by start time."""
    merged: list[list[pd.Timestamp]] = []
    for start, end in intervals:
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return [(s, e) for s, e in merged]


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    return merge_sorted(sorted(intervals))


def complement(blocked: Sequence[Interval]) -> Iterator[Interval]:
    cursor = ANALYSIS_START
    for start, end in blocked:
        if end <= cursor:
            continue
        if start > cursor:
            yield cursor, min(start, ANALYSIS_END)
        cursor = max(cursor, end)
        if cursor >= ANALYSIS_END:
            return
    if cursor < ANALYSIS_END:
        yield cursor, ANALYSIS_END


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    x = values[order]
    w = weights[order]
    cumulative = np.cumsum(w)
    idx = np.searchsorted(cumulative, q * cumulative[-1], side="left")
    return float(x[min(idx, len(x) - 1)])


def load_jobs(path: Path) -> pd.DataFrame:
    usecols = [
        "COBALT_JOBID",
        "START_TIMESTAMP",
        "END_TIMESTAMP",
        "NODES_USED",
        "LOCATION",
    ]
    jobs = pd.read_csv(path, usecols=usecols, low_memory=False)
    jobs["START_TIMESTAMP"] = pd.to_datetime(jobs["START_TIMESTAMP"], errors="coerce")
    jobs["END_TIMESTAMP"] = pd.to_datetime(jobs["END_TIMESTAMP"], errors="coerce")
    jobs["NODES_USED"] = pd.to_numeric(jobs["NODES_USED"], errors="coerce")

    valid = (
        jobs["START_TIMESTAMP"].notna()
        & jobs["END_TIMESTAMP"].notna()
        & jobs["NODES_USED"].notna()
        & (jobs["NODES_USED"] > 0)
        & (jobs["END_TIMESTAMP"] > jobs["START_TIMESTAMP"])
        & (jobs["END_TIMESTAMP"] > ANALYSIS_START)
        & (jobs["START_TIMESTAMP"] < ANALYSIS_END)
    )
    jobs = jobs.loc[valid].copy()

    parsed_nodes = []
    malformed_counts = []
    location_lengths = []
    for value in jobs["LOCATION"]:
        nodes, malformed = parse_location(value)
        parsed_nodes.append(nodes)
        malformed_counts.append(malformed)
        location_lengths.append(len(value) if isinstance(value, str) else 0)

    jobs["PARSED_NODES"] = parsed_nodes
    jobs["MALFORMED_LOCATION_TOKENS"] = malformed_counts
    jobs["LOCATION_LENGTH"] = location_lengths
    jobs["NODES_USED_INT"] = jobs["NODES_USED"].round().astype(int)
    jobs["PARSED_NODE_COUNT"] = jobs["PARSED_NODES"].map(len)
    jobs["LOCATION_COMPLETE"] = jobs["PARSED_NODE_COUNT"].eq(jobs["NODES_USED_INT"])

    jobs["CLIP_START"] = jobs["START_TIMESTAMP"].clip(lower=ANALYSIS_START)
    jobs["CLIP_END"] = jobs["END_TIMESTAMP"].clip(upper=ANALYSIS_END)
    jobs["CLIP_DURATION_H"] = (
        jobs["CLIP_END"] - jobs["CLIP_START"]
    ).dt.total_seconds() / 3600.0
    jobs["ALLOCATED_NODE_H"] = jobs["CLIP_DURATION_H"] * jobs["NODES_USED_INT"]
    return jobs


def load_outages(path: Path, jobs: pd.DataFrame) -> pd.DataFrame:
    status = pd.read_csv(path, low_memory=False)
    status["START_TIMESTAMP"] = pd.to_datetime(status["START_TIMESTAMP"], errors="coerce")
    status["END_TIMESTAMP"] = pd.to_datetime(status["END_TIMESTAMP"], errors="coerce")
    status["IS_ALL_MACHINE_DOWN"] = pd.to_numeric(
        status["IS_ALL_MACHINE_DOWN"], errors="coerce"
    )
    status = status.loc[status["IS_ALL_MACHINE_DOWN"].eq(1)].copy()

    # Same missing-end fallback as the original analysis.
    for idx, row in status.loc[status["END_TIMESTAMP"].isna()].iterrows():
        later = jobs.loc[
            jobs["START_TIMESTAMP"] > row["START_TIMESTAMP"], "START_TIMESTAMP"
        ]
        status.at[idx, "END_TIMESTAMP"] = later.min() if len(later) else ANALYSIS_END

    intervals: list[Interval] = []
    for row in status.itertuples(index=False):
        start, end = row.START_TIMESTAMP, row.END_TIMESTAMP
        if pd.isna(start) or pd.isna(end):
            continue
        start = max(start, ANALYSIS_START)
        end = min(end, ANALYSIS_END)
        if end > start:
            intervals.append((start, end))

    merged = merge_intervals(intervals)
    out = pd.DataFrame(merged, columns=["start", "end"])
    if len(out):
        out["duration_hours"] = (out["end"] - out["start"]).dt.total_seconds() / 3600.0
    return out


def aggregate_unallocated_node_hours(jobs: pd.DataFrame, outages: pd.DataFrame) -> float:
    """Aggregate analysis independent of LOCATION, matching the original method."""
    starts = jobs.loc[
        (jobs["CLIP_START"] > ANALYSIS_START) & (jobs["CLIP_START"] < ANALYSIS_END),
        ["CLIP_START", "NODES_USED_INT"],
    ].rename(columns={"CLIP_START": "timestamp", "NODES_USED_INT": "delta_allocated"})
    starts["delta_outage"] = 0

    ends = jobs.loc[
        (jobs["CLIP_END"] > ANALYSIS_START) & (jobs["CLIP_END"] < ANALYSIS_END),
        ["CLIP_END", "NODES_USED_INT"],
    ].rename(columns={"CLIP_END": "timestamp", "NODES_USED_INT": "delta_allocated"})
    ends["delta_allocated"] *= -1
    ends["delta_outage"] = 0

    os = outages.loc[
        (outages["start"] > ANALYSIS_START) & (outages["start"] < ANALYSIS_END), ["start"]
    ].rename(columns={"start": "timestamp"})
    os["delta_allocated"] = 0
    os["delta_outage"] = 1

    oe = outages.loc[
        (outages["end"] > ANALYSIS_START) & (outages["end"] < ANALYSIS_END), ["end"]
    ].rename(columns={"end": "timestamp"})
    oe["delta_allocated"] = 0
    oe["delta_outage"] = -1

    events = pd.concat([starts, ends, os, oe], ignore_index=True)
    events = (
        events.groupby("timestamp", as_index=False)[["delta_allocated", "delta_outage"]]
        .sum()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    initial_alloc = float(
        jobs.loc[
            (jobs["START_TIMESTAMP"] <= ANALYSIS_START)
            & (jobs["END_TIMESTAMP"] > ANALYSIS_START),
            "NODES_USED_INT",
        ].sum()
    )
    initial_outage = int(
        ((outages["start"] <= ANALYSIS_START) & (outages["end"] > ANALYSIS_START)).sum()
    )

    event_times = events["timestamp"].to_numpy(dtype="datetime64[ns]")
    boundaries = np.concatenate(
        [
            np.array([ANALYSIS_START.to_datetime64()]),
            event_times,
            np.array([ANALYSIS_END.to_datetime64()]),
        ]
    )
    duration_h = np.diff(boundaries).astype("timedelta64[ns]").astype(np.int64) / 3.6e12
    alloc = initial_alloc + np.concatenate(
        [[0.0], np.cumsum(events["delta_allocated"].to_numpy(float))]
    )
    down = initial_outage + np.concatenate(
        [[0], np.cumsum(events["delta_outage"].to_numpy(int))]
    )

    if alloc.max() > TOTAL_NODES + 1e-6:
        raise ValueError(f"Allocated-node count exceeds {TOTAL_NODES}: max={alloc.max()}")

    free = np.maximum(0.0, TOTAL_NODES - alloc)
    operational = down == 0
    return float(np.sum(free[operational] * duration_h[operational]))


def build_fragment_durations(
    jobs: pd.DataFrame,
    outages: pd.DataFrame,
    save_details: bool,
):
    complete = jobs.loc[jobs["LOCATION_COMPLETE"]].sort_values("CLIP_START")
    incomplete = jobs.loc[~jobs["LOCATION_COMPLETE"]]

    node_ids = sorted({node for nodes in jobs["PARSED_NODES"] for node in nodes})
    if len(node_ids) != TOTAL_NODES:
        print(f"WARNING: found {len(node_ids)} valid node IDs; expected {TOTAL_NODES}")

    # Because complete is sorted by start time, each per-node list is also sorted.
    busy_by_node: dict[str, list[Interval]] = defaultdict(list)
    for row in complete.itertuples(index=False):
        interval = (row.CLIP_START, row.CLIP_END)
        for node in row.PARSED_NODES:
            busy_by_node[node].append(interval)

    # Incomplete LOCATION means the missing node identities are unknown. For a
    # conservative node-level analysis, no node is asserted idle in these periods.
    censor_intervals = merge_intervals(
        (row.CLIP_START, row.CLIP_END) for row in incomplete.itertuples(index=False)
    )
    censor_starts = {s for s, _ in censor_intervals}
    censor_ends = {e for _, e in censor_intervals}
    outage_intervals = list(outages[["start", "end"]].itertuples(index=False, name=None))

    confirmed_durations: list[float] = []
    exact_durations: list[float] = []
    confirmed_rows = [] if save_details else None
    exact_rows = [] if save_details else None

    for i, node in enumerate(node_ids, start=1):
        node_busy = merge_sorted(busy_by_node.get(node, []))
        # All three inputs are sorted, so heapq.merge avoids re-sorting them.
        blocked = merge_sorted(heapq.merge(node_busy, censor_intervals, outage_intervals))

        for start, end in complement(blocked):
            duration_s = (end - start).total_seconds()
            left_censored = start == ANALYSIS_START or start in censor_ends
            right_censored = end == ANALYSIS_END or end in censor_starts
            fully_observed = not left_censored and not right_censored

            confirmed_durations.append(duration_s)
            if fully_observed:
                exact_durations.append(duration_s)

            if save_details:
                row = {
                    "node": node,
                    "start": start,
                    "end": end,
                    "duration_seconds": duration_s,
                    "duration_minutes": duration_s / 60.0,
                    "duration_hours": duration_s / 3600.0,
                    "left_censored": left_censored,
                    "right_censored": right_censored,
                    "fully_observed": fully_observed,
                }
                confirmed_rows.append(row)
                if fully_observed:
                    exact_rows.append(row.copy())

        if i % 100 == 0:
            print(f"Processed {i}/{len(node_ids)} nodes")

    return (
        np.asarray(confirmed_durations, dtype=float),
        np.asarray(exact_durations, dtype=float),
        censor_intervals,
        node_ids,
        confirmed_rows,
        exact_rows,
    )


def quantile_tables(exact_s: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    qs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    count_rows = []
    weighted_rows = []

    for q in qs:
        value = float(np.quantile(exact_s, q)) if len(exact_s) else float("nan")
        count_rows.append(
            {
                "quantile": q,
                "duration_seconds": value,
                "duration_minutes": value / 60.0,
                "duration_hours": value / 3600.0,
            }
        )

        weighted = weighted_quantile(exact_s, exact_s, q)
        weighted_rows.append(
            {
                "quantile": q,
                "duration_seconds": weighted,
                "duration_minutes": weighted / 60.0,
                "duration_hours": weighted / 3600.0,
            }
        )

    return pd.DataFrame(count_rows), pd.DataFrame(weighted_rows)


def startup_summary(
    confirmed_s: np.ndarray,
    exact_s: np.ndarray,
    aggregate_unused_h: float,
) -> pd.DataFrame:
    rows = []
    exact_h = exact_s.sum() / 3600.0
    confirmed_h = confirmed_s.sum() / 3600.0

    for model, startup_s in STARTUP_THRESHOLDS.items():
        exact_survive = exact_s >= startup_s
        confirmed_survive = confirmed_s >= startup_s

        exact_qual_h = exact_s[exact_survive].sum() / 3600.0
        exact_post_startup_h = np.maximum(exact_s - startup_s, 0.0).sum() / 3600.0
        confirmed_qual_h = confirmed_s[confirmed_survive].sum() / 3600.0
        confirmed_post_startup_h = np.maximum(confirmed_s - startup_s, 0.0).sum() / 3600.0

        rows.append(
            {
                "model": model,
                "startup_seconds": startup_s,
                "startup_minutes": startup_s / 60.0,
                "fully_observed_fragment_count": len(exact_s),
                "fragments_ge_startup_count": int(exact_survive.sum()),
                "fragments_ge_startup_percent": 100.0 * exact_survive.mean(),
                "fully_observed_fragment_node_hours": exact_h,
                "node_hours_in_fragments_ge_startup": exact_qual_h,
                "node_hours_in_fragments_ge_startup_percent": 100.0 * exact_qual_h / exact_h,
                "useful_node_hours_after_startup": exact_post_startup_h,
                "useful_node_hours_after_startup_percent": 100.0 * exact_post_startup_h / exact_h,
                "confirmed_idle_segment_node_hours": confirmed_h,
                "aggregate_unallocated_node_hours": aggregate_unused_h,
                # Conservative lower bounds relative to ALL aggregate unused capacity.
                "lower_bound_total_unused_capacity_in_confirmed_windows_ge_startup_percent": (
                    100.0 * confirmed_qual_h / aggregate_unused_h
                ),
                "lower_bound_total_unused_capacity_remaining_after_startup_percent": (
                    100.0 * confirmed_post_startup_h / aggregate_unused_h
                ),
            }
        )
    return pd.DataFrame(rows)


def location_quality(jobs: pd.DataFrame, node_ids: Sequence[str], censor_intervals: Sequence[Interval]) -> pd.DataFrame:
    incomplete = jobs.loc[~jobs["LOCATION_COMPLETE"]]
    total_alloc_h = jobs["ALLOCATED_NODE_H"].sum()
    incomplete_alloc_h = incomplete["ALLOCATED_NODE_H"].sum()
    censor_h = sum((e - s).total_seconds() for s, e in censor_intervals) / 3600.0
    year_h = (ANALYSIS_END - ANALYSIS_START).total_seconds() / 3600.0

    metrics = [
        ("usable_jobs", len(jobs)),
        ("complete_location_jobs", int(jobs["LOCATION_COMPLETE"].sum())),
        ("incomplete_location_jobs", len(incomplete)),
        ("incomplete_location_jobs_percent", 100.0 * len(incomplete) / len(jobs)),
        ("valid_unique_node_ids", len(node_ids)),
        ("jobs_with_location_length_2048", int(jobs["LOCATION_LENGTH"].eq(2048).sum())),
        ("incomplete_jobs_with_location_length_2048", int(incomplete["LOCATION_LENGTH"].eq(2048).sum())),
        ("total_malformed_location_tokens", int(jobs["MALFORMED_LOCATION_TOKENS"].sum())),
        ("allocated_node_hours_all_jobs", total_alloc_h),
        ("allocated_node_hours_incomplete_location_jobs", incomplete_alloc_h),
        ("allocated_node_hours_incomplete_percent", 100.0 * incomplete_alloc_h / total_alloc_h),
        ("min_nodes_used_in_incomplete_job", incomplete["NODES_USED_INT"].min()),
        ("max_nodes_used_in_incomplete_job", incomplete["NODES_USED_INT"].max()),
        ("median_parsed_nodes_in_incomplete_job", incomplete["PARSED_NODE_COUNT"].median()),
        ("merged_location_censor_intervals", len(censor_intervals)),
        ("location_censored_hours", censor_h),
        ("location_censored_calendar_time_percent", 100.0 * censor_h / year_h),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def survival_plot(exact_s: np.ndarray, outdir: Path) -> None:
    if len(exact_s) == 0:
        return

    x = np.sort(exact_s / 60.0)
    n = len(x)
    count_survival = (n - np.arange(n)) / n

    # Capacity-weighted survival: fraction of fully observed node-hours that
    # belong to fragments at least as long as x.
    weights = x.copy()
    remaining_weight = np.cumsum(weights[::-1])[::-1]
    capacity_survival = remaining_weight / remaining_weight[0]

    fig, ax = plt.subplots(figsize=(5.4, 3.15))
    ax.step(x, count_survival, where="post", linewidth=1.3, label="By fragment count")
    ax.step(x, capacity_survival, where="post", linewidth=1.3, label="By node-hours")

    for model, startup_s in STARTUP_THRESHOLDS.items():
        ax.axvline(startup_s / 60.0, linestyle="--", linewidth=0.85, label=f"{model} startup")

    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Individual fragment lifetime (minutes)")
    ax.set_ylabel("Fraction at or above lifetime")
    ax.grid(axis="y", linewidth=0.45, alpha=0.25)
    ax.legend(frameon=False, fontsize=7.7, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "fragment_lifetime_survival.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fragment_lifetime_survival.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, default="ANL-ALCF-DJC-POLARIS_20250101_20251231/ANL-ALCF-DJC-POLARIS_20250101_20251231.csv")
    ap.add_argument("--status", type=Path, default="ANL-ALCF-MACHINESTATUS-POLARIS_20250101_20251231/ANL-ALCF-MACHINESTATUS-POLARIS_20250101_20251231.csv")
    ap.add_argument("--output-dir", type=Path, default=Path("polaris_fragment_lifetimes"))
    ap.add_argument(
        "--save-details",
        action="store_true",
        help="Also write per-node confirmed segments/fragments as compressed CSVs.",
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and classifying jobs...")
    jobs = load_jobs(args.jobs)
    print(f"Usable jobs overlapping 2025: {len(jobs):,}")

    print("Loading full-machine outages...")
    outages = load_outages(args.status, jobs)
    outages.to_csv(args.output_dir / "full_machine_outages.csv", index=False)

    print("Recomputing aggregate unallocated node-hours...")
    aggregate_unused_h = aggregate_unallocated_node_hours(jobs, outages)
    print(f"Aggregate unallocated operational node-hours: {aggregate_unused_h:,.2f}")

    print("Reconstructing conservative per-node idle fragments...")
    (
        confirmed_s,
        exact_s,
        censor_intervals,
        node_ids,
        confirmed_rows,
        exact_rows,
    ) = build_fragment_durations(jobs, outages, args.save_details)

    quality = location_quality(jobs, node_ids, censor_intervals)
    quality.to_csv(args.output_dir / "location_quality_summary.csv", index=False)

    censor_df = pd.DataFrame(censor_intervals, columns=["start", "end"])
    if len(censor_df):
        censor_df["duration_hours"] = (censor_df["end"] - censor_df["start"]).dt.total_seconds() / 3600.0
    censor_df.to_csv(args.output_dir / "location_censor_intervals.csv", index=False)

    count_q, weighted_q = quantile_tables(exact_s)
    count_q.to_csv(args.output_dir / "fragment_quantiles_by_count.csv", index=False)
    weighted_q.to_csv(args.output_dir / "fragment_quantiles_by_node_hours.csv", index=False)

    startup = startup_summary(confirmed_s, exact_s, aggregate_unused_h)
    startup.to_csv(args.output_dir / "startup_threshold_summary.csv", index=False)

    survival_plot(exact_s, args.output_dir)

    if args.save_details:
        pd.DataFrame(confirmed_rows).to_csv(
            args.output_dir / "confirmed_idle_segments.csv.gz", index=False, compression="gzip"
        )
        pd.DataFrame(exact_rows).to_csv(
            args.output_dir / "fully_observed_fragments.csv.gz", index=False, compression="gzip"
        )

    print("\nLOCATION quality")
    print(quality.to_string(index=False))

    print("\nFragment lifetime quantiles by fragment count")
    print(count_q.to_string(index=False))

    print("\nFragment lifetime quantiles weighted by node-hours")
    print(weighted_q.to_string(index=False))

    print("\nStartup threshold summary")
    display_cols = [
        "model",
        "startup_seconds",
        "fragments_ge_startup_percent",
        "node_hours_in_fragments_ge_startup_percent",
        "useful_node_hours_after_startup_percent",
        "lower_bound_total_unused_capacity_in_confirmed_windows_ge_startup_percent",
        "lower_bound_total_unused_capacity_remaining_after_startup_percent",
    ]
    print(startup[display_cols].to_string(index=False))

    print(f"\nConfirmed idle segments: {len(confirmed_s):,}")
    print(f"Fully observed fragments: {len(exact_s):,}")
    print(f"Confirmed idle node-hours: {confirmed_s.sum()/3600.0:,.2f}")
    print(f"Fully observed fragment node-hours: {exact_s.sum()/3600.0:,.2f}")
    print(f"\nSaved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
