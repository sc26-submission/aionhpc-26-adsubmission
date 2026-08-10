from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================

JOB_CSV = Path(r"ANL-ALCF-DJC-POLARIS_20250101_20251231.csv") / "ANL-ALCF-DJC-POLARIS_20250101_20251231.csv"
STATUS_CSV = Path(r"ANL-ALCF-MACHINESTATUS-POLARIS_20250101_20251231.csv") / "ANL-ALCF-MACHINESTATUS-POLARIS_20250101_20251231.csv"
OUTPUT_DIR = Path("polaris_2025_analysis/results")
OUTPUT_DIR.mkdir(exist_ok=True)

TIMELINE_CSV = OUTPUT_DIR / "polaris_2025_operational_capacity_timeline.csv"
OUTAGES_CSV = OUTPUT_DIR / "polaris_2025_full_machine_outages.csv"

TOTAL_NODES = 560
GPUS_PER_NODE = 4
ANALYSIS_START = pd.Timestamp("2025-01-01 00:00:00")
ANALYSIS_END = pd.Timestamp("2026-01-01 00:00:00")


def separator(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def weighted_quantile(values, weights, q: float) -> float:
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    idx = np.searchsorted(cumulative, q * weights.sum(), side="left")
    return float(values[min(idx, len(values) - 1)])


# ============================================================
# Load inputs
# ============================================================

def load_jobs() -> pd.DataFrame:
    jobs = pd.read_csv(JOB_CSV, low_memory=False)
    for col in ("START_TIMESTAMP", "END_TIMESTAMP"):
        jobs[col] = pd.to_datetime(jobs[col], errors="coerce")
    jobs["NODES_USED"] = pd.to_numeric(jobs["NODES_USED"], errors="coerce")

    valid = (
        jobs["START_TIMESTAMP"].notna()
        & jobs["END_TIMESTAMP"].notna()
        & jobs["NODES_USED"].notna()
        & (jobs["NODES_USED"] > 0)
        & (jobs["END_TIMESTAMP"] > jobs["START_TIMESTAMP"])
    )
    return jobs.loc[valid].copy()


def load_status() -> pd.DataFrame:
    status = pd.read_csv(STATUS_CSV, low_memory=False)
    for col in ("START_TIMESTAMP", "END_TIMESTAMP"):
        status[col] = pd.to_datetime(status[col], errors="coerce")
    status["IS_ALL_MACHINE_DOWN"] = pd.to_numeric(status["IS_ALL_MACHINE_DOWN"], errors="coerce")
    return status.loc[status["IS_ALL_MACHINE_DOWN"].eq(1)].copy()


# ============================================================
# Outages
# ============================================================

def prepare_outages(status: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    status = status.copy()
    status["END_INFERRED"] = False

    separator("OUTAGE END INFERENCE")
    for idx, row in status.loc[status["END_TIMESTAMP"].isna()].iterrows():
        start = row["START_TIMESTAMP"]
        subsequent = jobs.loc[jobs["START_TIMESTAMP"] > start, "START_TIMESTAMP"]
        end = subsequent.min() if len(subsequent) else ANALYSIS_END
        status.at[idx, "END_TIMESTAMP"] = end
        status.at[idx, "END_INFERRED"] = True
        print(f"{start} -> {end} (inferred from subsequent job activity)")

    valid = (
        status["START_TIMESTAMP"].notna()
        & status["END_TIMESTAMP"].notna()
        & (status["END_TIMESTAMP"] > ANALYSIS_START)
        & (status["START_TIMESTAMP"] < ANALYSIS_END)
    )
    status = status.loc[valid].copy()
    status["START_TIMESTAMP"] = status["START_TIMESTAMP"].clip(lower=ANALYSIS_START)
    status["END_TIMESTAMP"] = status["END_TIMESTAMP"].clip(upper=ANALYSIS_END)

    raw = status[["START_TIMESTAMP", "END_TIMESTAMP", "END_INFERRED"]].sort_values("START_TIMESTAMP")
    merged = []
    for row in raw.itertuples(index=False):
        start, end, inferred = row.START_TIMESTAMP, row.END_TIMESTAMP, bool(row.END_INFERRED)
        if not merged or start > merged[-1]["end"]:
            merged.append({"start": start, "end": end, "contains_inferred_end": inferred})
        else:
            merged[-1]["end"] = max(merged[-1]["end"], end)
            merged[-1]["contains_inferred_end"] |= inferred

    outages = pd.DataFrame(merged, columns=["start", "end", "contains_inferred_end"])
    if outages.empty:
        outages["duration_hours"] = pd.Series(dtype=float)
    else:
        outages["duration_hours"] = (outages["end"] - outages["start"]).dt.total_seconds() / 3600.0
    return outages


# ============================================================
# Unified job/outage events
# ============================================================

def event_frame(frame, time_col, value_col=None, sign=1.0, outage_delta=0):
    cols = [time_col] + ([value_col] if value_col else [])
    result = frame[cols].copy().rename(columns={time_col: "timestamp"})
    if value_col:
        result = result.rename(columns={value_col: "delta_allocated"})
        result["delta_allocated"] *= sign
    else:
        result["delta_allocated"] = 0.0
    result["delta_outages"] = outage_delta
    return result


def build_events(jobs: pd.DataFrame, outages: pd.DataFrame) -> pd.DataFrame:
    job_starts = jobs.loc[(jobs["START_TIMESTAMP"] > ANALYSIS_START) & (jobs["START_TIMESTAMP"] < ANALYSIS_END)]
    job_ends = jobs.loc[(jobs["END_TIMESTAMP"] > ANALYSIS_START) & (jobs["END_TIMESTAMP"] < ANALYSIS_END)]
    outage_starts = outages.loc[(outages["start"] > ANALYSIS_START) & (outages["start"] < ANALYSIS_END)]
    outage_ends = outages.loc[(outages["end"] > ANALYSIS_START) & (outages["end"] < ANALYSIS_END)]

    events = pd.concat(
        [
            event_frame(job_starts, "START_TIMESTAMP", "NODES_USED"),
            event_frame(job_ends, "END_TIMESTAMP", "NODES_USED", -1.0),
            event_frame(outage_starts, "start", outage_delta=1),
            event_frame(outage_ends, "end", outage_delta=-1),
        ],
        ignore_index=True,
    )
    return (
        events.groupby("timestamp", as_index=False)[["delta_allocated", "delta_outages"]]
        .sum().sort_values("timestamp").reset_index(drop=True)
    )


# ============================================================
# Piecewise-constant capacity timeline
# ============================================================

def make_interval(start, end, allocated, active_outages):
    machine_down = active_outages > 0
    available = 0 if machine_down else TOTAL_NODES - allocated
    if not machine_down and available < 0:
        raise ValueError(f"Negative available capacity at {start}: allocated={allocated}")

    return {
        "start": start,
        "end": end,
        "duration_seconds": (end - start).total_seconds(),
        "allocated_nodes": allocated,
        "machine_down": machine_down,
        "operational_nodes": 0 if machine_down else TOTAL_NODES,
        "unallocated_operational_nodes": max(available, 0),
    }


def build_timeline(jobs: pd.DataFrame, outages: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    active = jobs.loc[(jobs["START_TIMESTAMP"] <= ANALYSIS_START) & (jobs["END_TIMESTAMP"] > ANALYSIS_START)]
    allocated = float(active["NODES_USED"].sum())
    active_outages = int(((outages["start"] <= ANALYSIS_START) & (outages["end"] > ANALYSIS_START)).sum())

    intervals, current_time = [], ANALYSIS_START
    for row in events.itertuples(index=False):
        if row.timestamp > current_time:
            intervals.append(make_interval(current_time, row.timestamp, allocated, active_outages))

        allocated += row.delta_allocated
        active_outages += int(row.delta_outages)

        if allocated < 0 or allocated > TOTAL_NODES:
            raise ValueError(f"Impossible allocation at {row.timestamp}: {allocated}")
        if active_outages < 0:
            raise ValueError(f"Negative outage count at {row.timestamp}")
        current_time = row.timestamp

    if current_time < ANALYSIS_END:
        intervals.append(make_interval(current_time, ANALYSIS_END, allocated, active_outages))

    timeline = pd.DataFrame(intervals)
    timeline["duration_hours"] = timeline["duration_seconds"] / 3600.0
    timeline["operational_node_hours"] = timeline["operational_nodes"] * timeline["duration_hours"]
    timeline["unallocated_node_hours"] = timeline["unallocated_operational_nodes"] * timeline["duration_hours"]
    return timeline


# ============================================================
# Reports
# ============================================================

def report_results(timeline: pd.DataFrame, outages: pd.DataFrame) -> None:
    separator("OUTAGE SUMMARY")
    print(outages.to_string(index=False))
    total_outage_hours = outages["duration_hours"].sum()
    print(f"\nTotal full-machine outage time: {total_outage_hours:,.2f} h")

    separator("TIMELINE SANITY CHECK")
    total_calendar_hours = (ANALYSIS_END - ANALYSIS_START).total_seconds() / 3600.0
    print(f"Expected calendar hours: {total_calendar_hours:,.2f}")
    print(f"Timeline hours:          {timeline['duration_hours'].sum():,.2f}")
    print(f"Minimum allocated nodes: {timeline['allocated_nodes'].min():.0f}")
    print(f"Maximum allocated nodes: {timeline['allocated_nodes'].max():.0f}")
    print(f"Minimum available nodes: {timeline['unallocated_operational_nodes'].min():.0f}")
    print(f"Maximum available nodes: {timeline['unallocated_operational_nodes'].max():.0f}")

    operational = timeline.loc[~timeline["machine_down"]].copy()
    operational_hours = operational["duration_hours"].sum()
    operational_node_hours = operational["operational_node_hours"].sum()
    unallocated_node_hours = operational["unallocated_node_hours"].sum()
    unallocated_fraction = unallocated_node_hours / operational_node_hours
    mean_unallocated_nodes = unallocated_node_hours / operational_hours

    separator("CORRECTED 2025 CAPACITY SUMMARY")
    print(f"Calendar observation time: {total_calendar_hours:,.2f} h")
    print(f"Full-machine outage time: {total_outage_hours:,.2f} h ({100 * total_outage_hours / total_calendar_hours:.2f}%)")
    print(f"Operational observation time: {operational_hours:,.2f} h\n")
    print(f"Total operational node-hours: {operational_node_hours:,.2f}")
    print(f"Unallocated operational node-hours: {unallocated_node_hours:,.2f}")
    print(f"Unallocated fraction of operational capacity: {100 * unallocated_fraction:.2f}%\n")
    print(f"Equivalent continuously unallocated nodes: {mean_unallocated_nodes:.2f}")
    print(f"Equivalent unallocated GPU-hours: {unallocated_node_hours * GPUS_PER_NODE:,.2f}")

    separator("TIME-WEIGHTED UNALLOCATED POOL SIZE")
    for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        value = weighted_quantile(operational["unallocated_operational_nodes"], operational["duration_seconds"], q)
        print(f"p{int(q * 100):02d}: {value:.0f} nodes")

    separator("FRACTION OF OPERATIONAL TIME WITH AT LEAST N NODES")
    for n in (1, 2, 4, 8, 16, 32, 64, 128):
        hours = operational.loc[operational["unallocated_operational_nodes"] >= n, "duration_hours"].sum()
        print(f">= {n:3d} nodes: {100 * hours / operational_hours:6.2f}% ({hours:,.2f} h)")


# ============================================================
# Main
# ============================================================

def main() -> None:
    jobs = load_jobs()
    separator("JOB DATA")
    print(f"Usable jobs: {len(jobs):,}")

    outages = prepare_outages(load_status(), jobs)
    events = build_events(jobs, outages)
    timeline = build_timeline(jobs, outages, events)
    report_results(timeline, outages)

    timeline.to_csv(TIMELINE_CSV, index=False)
    outages.to_csv(OUTAGES_CSV, index=False)
    print(f"\nSaved:\n  {TIMELINE_CSV}\n  {OUTAGES_CSV}")


if __name__ == "__main__":
    main()