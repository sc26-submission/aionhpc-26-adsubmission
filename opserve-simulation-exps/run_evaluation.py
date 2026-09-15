#!/usr/bin/env python3
"""Run the trace-driven workload/model/policy evaluation matrix.

The default matrix contains:
  * Workloads: Conversation, Synthetic (when present), and Tool&Agent
  * Models: Llama-3.1-8B, Qwen3-8B, and Qwen3-14B
  * Policies: Persistent-Only, Static P/D, and OPSERVE

Request arrivals stop after the one-hour trace horizon. The simulator then
continues until all arrived requests complete or the drain timeout expires.
Completion by the one-hour boundary and latency after draining are reported
separately to avoid completed-request latency bias.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from opserve_sim_core import (
    DEFAULT_KV_TRANSFER_PROFILE,
    SimConfig,
    Simulator,
    clone_requests,
    load_mooncake_trace_requests,
    load_resource_events,
    write_csv,
)


POLICIES: dict[str, tuple[str, bool]] = {
    "fixed": ("Persistent-Only", False),
    "static": ("Static P/D", True),
    "adaptive": ("OPSERVE", True),
}


def parse_csv_selection(value: str, allowed: Iterable[str], name: str) -> list[str]:
    allowed_set = set(allowed)
    if value.strip().lower() == "all":
        return list(allowed)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in selected if item not in allowed_set]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {name}: {', '.join(unknown)}; allowed: {', '.join(allowed)}"
        )
    if not selected:
        raise argparse.ArgumentTypeError(f"at least one {name} is required")
    return selected


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the OPSERVE workload/model/policy matrix."
    )

    parser.add_argument(
        "--profiles",
        type=Path,
        default=root / "profiles" / "model_profiles.json",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=root / "event-traces" / "one-hour-trace.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "logs" / "evaluation_matrix",
    )

    parser.add_argument(
        "--workloads",
        default="all",
        help="Comma-separated workload keys or 'all'.",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys or 'all'.",
    )
    parser.add_argument(
        "--policies",
        default="all",
        help="Comma-separated policy keys or 'all'.",
    )

    parser.add_argument("--trace-duration", type=float, default=3600.0)
    parser.add_argument(
        "--drain-timeout",
        type=float,
        default=14400.0,
        help="Maximum seconds to drain requests after arrivals stop.",
    )
    parser.add_argument("--persistent-workers", type=int, default=8)

    parser.add_argument(
        "--controller",
        choices=["demand_share", "latency_share", "slo_pressure", "pressure"],
        default="latency_share",
        help=(
            "Adaptive controller. latency_share is the controller used in the paper."
        ),
    )
    parser.add_argument("--balance-interval", type=float, default=15.0)
    parser.add_argument("--metric-window", type=float, default=120.0)
    parser.add_argument(
        "--latency-half-life",
        type=float,
        default=30.0,
        help=(
            "Half-life in seconds for time-decayed phase-latency estimates "
            "used by the latency_share controller."
        ),
    )
    parser.add_argument(
        "--latency-queue-quantile",
        type=float,
        default=0.75,
        help=(
            "Quantile of current phase-queue waiting times folded into the "
            "latency_share control signal (default: 0.75)."
        ),
    )
    parser.add_argument("--rebalance-cooldown", type=float, default=30.0)
    parser.add_argument("--max-moves", type=int, default=1)
    parser.add_argument("--min-role-dwell", type=float, default=30.0)
    parser.add_argument("--min-control-samples", type=int, default=16)
    parser.add_argument("--demand-deadband", type=float, default=0.05)
    parser.add_argument(
        "--prompt-threshold",
        type=float,
        default=2048.0,
        help=(
            "Prompt-length threshold below which no inter-worker KV-transfer "
            "delay is charged (default: 2048 tokens)."
        ),
    )
    parser.add_argument("--share-rounding", type=float, default=0.1)
    parser.add_argument("--pressure-hysteresis", type=float, default=3.0)
    parser.add_argument(
        "--ttft-target-s",
        type=float,
        default=None,
        help="Required by --controller slo_pressure.",
    )
    parser.add_argument(
        "--tpot-target-ms",
        type=float,
        default=None,
        help="Required by --controller slo_pressure.",
    )
    parser.add_argument(
        "--slo-pressure-deadband",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--static-prefill-share",
        type=float,
        default=0.5,
        help="Target prefill share for Static P/D; default 0.5.",
    )
    parser.add_argument(
        "--kv-handoff",
        type=float,
        default=0.0,
        help="Optional post-prefill KV handoff delay in seconds.",
    )
    parser.add_argument(
        "--recovery-delay",
        type=float,
        default=0.0,
        help="Optional delay before an interrupted phase is requeued.",
    )

    parser.add_argument(
        "--save-timeseries",
        choices=["none", "representative", "all"],
        default="all",
    )
    parser.add_argument(
        "--save-requests",
        choices=["none", "representative", "all"],
        default="all",
    )
    parser.add_argument(
        "--representative-workload",
        default="toolagent",
    )
    parser.add_argument(
        "--representative-model",
        default="llama31_8b",
    )

    return parser.parse_args()


def load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    profiles = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"no model profiles found in {path}")

    required = {
        "label",
        "params_b",
        "startup_s",
        "prefill_profile_s",
        "decode_profile_s_per_token",
    }
    for key, profile in profiles.items():
        missing = required - set(profile)
        if missing:
            raise ValueError(f"profile {key!r} is missing: {sorted(missing)}")
        if len(profile["prefill_profile_s"]) < 2:
            raise ValueError(f"profile {key!r} needs at least two prefill points")
        if len(profile["decode_profile_s_per_token"]) < 2:
            raise ValueError(f"profile {key!r} needs at least two decode points")
        if float(profile["startup_s"]) <= 0:
            raise ValueError(f"profile {key!r} startup_s must be positive")
        for tokens, latency_s in profile["prefill_profile_s"]:
            if int(tokens) <= 0 or float(latency_s) <= 0:
                raise ValueError(f"profile {key!r} has an invalid prefill point")
        for concurrency, tpot_s in profile["decode_profile_s_per_token"]:
            if int(concurrency) <= 0 or float(tpot_s) <= 0:
                raise ValueError(f"profile {key!r} has an invalid decode point")
    return profiles


def workload_definitions(root: Path) -> dict[str, dict[str, Any]]:
    definitions = {
        "conversation": {
            "label": "Conversation",
            "path": root / "traces" / "conversation_trace.jsonl",
        },
        "synthetic": {
            "label": "Synthetic",
            "path": root / "traces" / "synthetic_trace.jsonl",
            # Synthetic is a native ~17-minute source trace rather than a
            # one-hour trace. Use its own arrival horizon when reporting
            # arrival rate and completed-request throughput.
            "use_native_reporting_horizon": True,
        },
        "toolagent": {
            "label": "Tool&Agent",
            "path": root / "traces" / "toolagent_trace.jsonl",
        },
    }
    # Include only request traces that are present in the artifact package.
    return {
        key: value
        for key, value in definitions.items()
        if Path(value["path"]).exists()
    }


def validate_even_workers(value: int) -> None:
    if value < 2 or value % 2:
        raise ValueError("--persistent-workers must be an even integer >= 2")


def resource_capacity_timeline(
    events: list[tuple[float, str, list[int]]],
    base_workers: int,
    duration_s: float,
) -> list[dict[str, Any]]:
    present: set[int] = set()
    rows: list[dict[str, Any]] = [
        {"time": 0.0, "allocated_workers": base_workers, "event": "start"}
    ]
    for time_s, event_type, worker_ids in events:
        if event_type == "gain":
            present.update(worker_ids)
        elif event_type == "loss":
            present.difference_update(worker_ids)
        rows.append(
            {
                "time": time_s,
                "allocated_workers": base_workers + len(present),
                "event": event_type,
            }
        )
    rows.append(
        {
            "time": duration_s,
            "allocated_workers": base_workers + len(present),
            "event": "end",
        }
    )
    return rows


def should_save(
    mode: str,
    workload_key: str,
    model_key: str,
    args: argparse.Namespace,
) -> bool:
    if mode == "all":
        return True
    if mode == "none":
        return False
    return (
        workload_key == args.representative_workload
        and model_key == args.representative_model
    )


def make_config(
    args: argparse.Namespace,
    profile: dict[str, Any],
) -> SimConfig:
    prefill_profile = tuple(
        (int(tokens), float(latency_s))
        for tokens, latency_s in profile["prefill_profile_s"]
    )
    decode_profile = tuple(
        (int(concurrency), float(tpot_s))
        for concurrency, tpot_s in profile["decode_profile_s_per_token"]
    )
    decode_max_concurrency = max(concurrency for concurrency, _ in decode_profile)

    return SimConfig(
        duration_s=args.trace_duration,
        drain_timeout_s=args.drain_timeout,
        startup_s=float(profile["startup_s"]),
        balance_interval_s=args.balance_interval,
        metric_window_s=args.metric_window,
        latency_half_life_s=args.latency_half_life,
        latency_queue_quantile=args.latency_queue_quantile,
        rebalance_cooldown_s=args.rebalance_cooldown,
        max_moves_per_interval=args.max_moves,
        min_role_dwell_s=args.min_role_dwell,
        min_control_samples=args.min_control_samples,
        controller_mode=args.controller,
        prompt_threshold_tokens=args.prompt_threshold,
        share_rounding=args.share_rounding,
        demand_deadband=args.demand_deadband,
        pressure_hysteresis=args.pressure_hysteresis,
        ttft_target_s=args.ttft_target_s,
        tpot_target_s=(
            None if args.tpot_target_ms is None else args.tpot_target_ms / 1000.0
        ),
        slo_pressure_deadband=args.slo_pressure_deadband,
        prefill_profile=prefill_profile,
        decode_profile=decode_profile,
        kv_transfer_profile=DEFAULT_KV_TRANSFER_PROFILE,
        decode_tpot_s=decode_profile[0][1],
        decode_max_concurrency=decode_max_concurrency,
        static_prefill_share=args.static_prefill_share,
        kv_handoff_s=args.kv_handoff,
        recovery_delay_s=args.recovery_delay,
        neutralize_when_idle=False,
    )


def reporting_horizon_s(
    requests,
    workload: dict[str, Any],
    trace_duration_s: float,
) -> float:
    """Common workload-defined horizon used for arrival rate and throughput.

    Conversation and Tool&Agent are evaluated over the configured one-hour
    trace horizon. Synthetic is a shorter native source trace, so its final
    arrival timestamp defines the common reporting horizon for every policy.
    """
    if workload.get("use_native_reporting_horizon", False):
        if not requests:
            return trace_duration_s
        return max(float(r.arrival_time) for r in requests)
    return trace_duration_s


def workload_summary(requests, reporting_horizon: float) -> dict[str, float]:
    count = len(requests)
    return {
        "request_count": count,
        "reporting_horizon_s": reporting_horizon,
        "mean_arrival_rate_rps": count / reporting_horizon,
        "mean_input_tokens": sum(r.input_length for r in requests) / count,
        "mean_output_tokens": sum(r.output_length for r in requests) / count,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    validate_even_workers(args.persistent_workers)

    profiles = load_profiles(args.profiles)
    workloads = workload_definitions(root)

    workload_keys = parse_csv_selection(
        args.workloads, workloads.keys(), "workload"
    )
    model_keys = parse_csv_selection(args.models, profiles.keys(), "model")
    policy_keys = parse_csv_selection(args.policies, POLICIES.keys(), "policy")

    if args.representative_workload not in workloads:
        raise ValueError("unknown representative workload")
    if args.representative_model not in profiles:
        raise ValueError("unknown representative model")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = load_resource_events(args.events, args.trace_duration)
    write_csv(
        args.output_dir / "resource_capacity.csv",
        resource_capacity_timeline(
            events,
            args.persistent_workers,
            args.trace_duration,
        ),
    )

    base_prefill = args.persistent_workers // 2
    base_decode = args.persistent_workers // 2
    summaries: list[dict[str, Any]] = []
    workload_rows: list[dict[str, Any]] = []

    for workload_key in workload_keys:
        workload = workloads[workload_key]
        requests = load_mooncake_trace_requests(
            workload["path"],
            args.trace_duration,
        )
        report_horizon = reporting_horizon_s(
            requests, workload, args.trace_duration
        )
        stats = workload_summary(requests, report_horizon)
        workload_rows.append(
            {
                "workload": workload_key,
                "workload_label": workload["label"],
                **stats,
            }
        )

        for model_key in model_keys:
            profile = profiles[model_key]
            cfg = make_config(args, profile)

            for policy_key in policy_keys:
                policy_label, use_resource_trace = POLICIES[policy_key]
                sim = Simulator(
                    policy=policy_key,
                    requests=clone_requests(requests),
                    resource_events=events,
                    cfg=cfg,
                    base_prefill=base_prefill,
                    base_decode=base_decode,
                    use_resource_trace=use_resource_trace,
                )
                summary = sim.run()

                # Report throughput over the same workload-defined horizon for
                # every policy. Latency statistics still use all requests that
                # complete during the normal drain period.
                completed_by_reporting_horizon = sum(
                    1
                    for record in sim.completed_records
                    if float(record["completion_time"]) <= report_horizon + 1e-9
                )
                reporting_throughput_rps = (
                    completed_by_reporting_horizon / report_horizon
                )

                row = {
                    "workload": workload_key,
                    "workload_label": workload["label"],
                    "model": model_key,
                    "model_label": profile["label"],
                    "policy": policy_key,
                    "policy_label": policy_label,
                    "startup_s": cfg.startup_s,
                    "decode_max_concurrency": cfg.decode_max_concurrency,
                    **stats,
                    **summary,
                    # Override the simulator's one-hour cutoff throughput with
                    # the workload-defined common reporting horizon.
                    "completed_by_reporting_horizon": completed_by_reporting_horizon,
                    "mean_throughput_rps": reporting_throughput_rps,
                }
                summaries.append(row)

                case_dir = args.output_dir / workload_key / model_key
                case_dir.mkdir(parents=True, exist_ok=True)
                write_csv(case_dir / f"{policy_key}_role_events.csv", sim.role_events)

                if should_save(
                    args.save_timeseries,
                    workload_key,
                    model_key,
                    args,
                ):
                    write_csv(case_dir / f"{policy_key}_timeseries.csv", sim.timeline)

                if should_save(
                    args.save_requests,
                    workload_key,
                    model_key,
                    args,
                ):
                    write_csv(
                        case_dir / f"{policy_key}_requests.csv",
                        sim.completed_records,
                    )

                warning = ""
                if summary["unfinished"]:
                    warning = f"  WARNING unfinished={summary['unfinished']}"

                print(
                    f"{workload['label']:<12} {profile['label']:<15} "
                    f"{policy_label:<16} "
                    f"complete@1h={100*summary['completion_by_cutoff_fraction']:6.2f}% "
                    f"drained={100*summary['drained_completion_fraction']:6.2f}% "
                    f"p95TTFT={summary['ttft_p95_s']:9.2f}s "
                    f"p95E2E={summary['e2e_p95_s']:9.2f}s "
                    f"moves={summary['role_changes']:3d}{warning}"
                )

    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "workloads.csv", workload_rows)

    metadata = {
        "trace_duration_s": args.trace_duration,
        "drain_timeout_s": args.drain_timeout,
        "persistent_workers": args.persistent_workers,
        "initial_prefill_workers": base_prefill,
        "initial_decode_workers": base_decode,
        "resource_trace": str(args.events),
        "resource_state_during_drain": (
            "The worker state at the end of the one-hour resource trace is "
            "retained while outstanding requests drain."
        ),
        "controller": {
            "mode": args.controller,
            "balance_interval_s": args.balance_interval,
            "metric_window_s": args.metric_window,
            "latency_half_life_s": args.latency_half_life,
            "latency_queue_quantile": args.latency_queue_quantile,
            "rebalance_cooldown_s": args.rebalance_cooldown,
            "max_moves_per_interval": args.max_moves,
            "min_role_dwell_s": args.min_role_dwell,
            "min_control_samples": args.min_control_samples,
            "demand_deadband": args.demand_deadband,
            "prompt_threshold_tokens": args.prompt_threshold,
            "share_rounding": args.share_rounding,
            "pressure_hysteresis": args.pressure_hysteresis,
            "ttft_target_s": args.ttft_target_s,
            "tpot_target_ms": args.tpot_target_ms,
            "slo_pressure_deadband": args.slo_pressure_deadband,
            "static_prefill_share": args.static_prefill_share,
            "kv_transfer_profile_s": list(DEFAULT_KV_TRANSFER_PROFILE),
            "kv_transfer_source": (
                "Mooncake vLLM P/D Disaggregation Performance, cross-node RDMA"
            ),
            "kv_handoff_extra_s": args.kv_handoff,
            "recovery_delay_extra_s": args.recovery_delay,
        },
        "workloads": workload_keys,
        "models": model_keys,
        "policies": policy_keys,
        "profiles_file": str(args.profiles),
        "latency_metrics": (
            "Computed over requests that complete during the trace plus drain "
            "period. Completion by the one-hour boundary is reported separately."
        ),
        "throughput_metric": (
            "Completed requests divided by a workload-defined common horizon. "
            "Conversation and Tool&Agent use the configured one-hour horizon; "
            "Synthetic uses its native final-arrival horizon. The same horizon "
            "is used for all policies within a workload."
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    unfinished_total = sum(int(row["unfinished"]) for row in summaries)
    if unfinished_total:
        print(
            f"\n{unfinished_total} requests remained unfinished across the matrix. "
            "Increase --drain-timeout if requests remain unfinished."
        )

    print(f"\nWrote matrix results to {args.output_dir}")


if __name__ == "__main__":
    main()
