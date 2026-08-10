#!/usr/bin/env python3
"""Small event-driven simulator used by the MalleServe paper experiments.

The goal is not to reproduce every detail of vLLM.  The simulator isolates the
mechanisms studied in the workshop paper:

* transient worker gain/loss and startup delay,
* prefill/decode worker pools,
* phase-specific queueing,
* worker draining and role reassignment,
* request retry when a transient worker disappears.

Prefill, startup, and decode behavior are supplied through explicit model
profiles. Prefill uses measured median TTFT and decode uses measured median
TPOT as a function of active sequence concurrency.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Deque, Dict, Iterable, Optional, Sequence, Tuple, Union

WorkerId = Union[int, str]

# Default to the measured Llama-3.1-8B profile. Evaluation runs normally
# override these values from profiles/model_profiles.json.
DEFAULT_PREFILL_PROFILE: Tuple[Tuple[int, float], ...] = (
    (512, 0.04344506352208555),
    (1024, 0.04571733414195478),
    (2048, 0.08216385007835925),
    (4096, 0.15842866199091077),
    (8192, 0.34635750111192465),
    (16384, 1.4316540483850986),
    (32767, 3.656333898426965),
)

DEFAULT_DECODE_PROFILE: Tuple[Tuple[int, float], ...] = (
    (1, 0.013715790311678074),
    (2, 0.014915688929385414),
    (4, 0.01655582557797578),
    (8, 0.027533262530706967),
    (16, 0.06117969190007916),
)


def interpolate_profile(
    x_value: float,
    points: Sequence[tuple[float, float]],
) -> tuple[float, bool]:
    """Linearly interpolate a positive-valued measurement profile."""
    if len(points) < 2:
        raise ValueError("a measurement profile requires at least two points")

    ordered = sorted((float(x), float(y)) for x, y in points)
    x = float(max(1.0, x_value))

    if x <= ordered[0][0]:
        (x0, y0), (x1, y1) = ordered[0], ordered[1]
        extrapolated = x < x0
    elif x >= ordered[-1][0]:
        (x0, y0), (x1, y1) = ordered[-2], ordered[-1]
        extrapolated = x > x1
    else:
        extrapolated = False
        x0 = y0 = x1 = y1 = 0.0
        for (xa, ya), (xb, yb) in zip(ordered[:-1], ordered[1:]):
            if xa <= x <= xb:
                x0, y0, x1, y1 = xa, ya, xb, yb
                break

    slope = (y1 - y0) / (x1 - x0)
    return max(0.0, y0 + slope * (x - x0)), extrapolated


def interp_prefill_s(
    input_tokens: int,
    profile: Sequence[tuple[int, float]] = DEFAULT_PREFILL_PROFILE,
) -> tuple[float, bool]:
    return interpolate_profile(float(input_tokens), profile)


def interp_decode_tpot_s(
    concurrency: int,
    profile: Sequence[tuple[int, float]] = DEFAULT_DECODE_PROFILE,
) -> tuple[float, bool]:
    return interpolate_profile(float(max(1, concurrency)), profile)


def decode_service_s(
    output_tokens: int,
    tpot_s: float,
) -> float:
    # output_len=1 is already included in the empirical prefill measurement.
    return max(0, int(output_tokens) - 1) * tpot_s


def phase_demand_share(
    input_tokens: int,
    output_tokens: int,
    decode_tpot_s: float,
    decode_max_concurrency: int,
    prefill_profile: Sequence[tuple[int, float]] = DEFAULT_PREFILL_PROFILE,
    decode_profile: Sequence[tuple[int, float]] = (),
) -> float:
    """Fraction of effective per-request worker demand attributable to prefill."""
    p = interp_prefill_s(input_tokens, prefill_profile)[0]
    if decode_profile:
        tpot = interp_decode_tpot_s(decode_max_concurrency, decode_profile)[0]
    else:
        tpot = decode_tpot_s
    d = decode_service_s(output_tokens, tpot) / max(1, decode_max_concurrency)
    return p / max(1e-12, p + d)


@dataclass
class RequestJob:
    request_id: int
    arrival_time: float
    input_length: int
    output_length: int
    regime: str = "trace"

    # Current queue-entry timestamps. These are reset after a retry so queueing
    # delay is not confused with lost service time.
    prefill_enqueued: float = 0.0
    prefill_start: Optional[float] = None
    first_token_time: Optional[float] = None
    decode_enqueued: Optional[float] = None
    decode_start: Optional[float] = None
    completion_time: Optional[float] = None

    prefill_wait_total_s: float = 0.0
    decode_wait_total_s: float = 0.0
    retries: int = 0
    prefill_retries: int = 0
    decode_retries: int = 0


@dataclass
class InFlightStage:
    request_id: int
    stage: str
    generation: int
    finish_time: float
    # Decode completion events are rescheduled whenever active concurrency
    # changes. remaining_tokens is unused for prefill.
    remaining_tokens: float = 0.0
    stage_start: float = 0.0


@dataclass
class Worker:
    worker_id: WorkerId
    role: str
    persistent: bool = False
    state: str = "active"  # active | starting | draining | absent
    generation: int = 0
    in_flight: Dict[int, InFlightStage] = field(default_factory=dict)
    pending_role: Optional[str] = None
    pending_since: Optional[float] = None
    last_role_change_time: float = -math.inf
    decode_last_update: Optional[float] = None
    decode_epoch: int = 0


@dataclass
class SimConfig:
    # Request arrivals and resource events are replayed over duration_s.  The
    # simulator then drains outstanding requests for at most drain_timeout_s.
    duration_s: float = 3600.0
    drain_timeout_s: float = 14400.0
    startup_s: float = 120.0

    balance_interval_s: float = 15.0
    metric_window_s: float = 120.0
    sample_interval_s: float = 5.0

    # Conservative defaults: one worker per action and enough time for a role
    # change to settle before another decision.
    rebalance_cooldown_s: float = 30.0
    max_moves_per_interval: int = 1
    min_role_dwell_s: float = 30.0
    min_control_samples: int = 16

    # demand_share uses measured service profiles to estimate recent offered
    # work and is the recommended simulator controller. latency_share retains
    # the earlier deployed-balancer approximation for sensitivity experiments.
    controller_mode: str = "demand_share"  # demand_share | latency_share | slo_pressure | pressure
    prompt_threshold_tokens: float = 2048.0
    share_rounding: float = 0.1
    demand_deadband: float = 0.05
    pressure_hysteresis: float = 3.0

    # Required only by slo_pressure. These are user-facing service targets,
    # not values inferred from the benchmark profiles.
    ttft_target_s: Optional[float] = None
    tpot_target_s: Optional[float] = None
    slo_pressure_deadband: float = 0.10

    prefill_profile: Tuple[Tuple[int, float], ...] = DEFAULT_PREFILL_PROFILE
    decode_profile: Tuple[Tuple[int, float], ...] = DEFAULT_DECODE_PROFILE
    decode_tpot_s: float = DEFAULT_DECODE_PROFILE[0][1]  # fallback when profile is empty
    decode_max_concurrency: int = 16

    # Static P/D target. 0.5 reproduces the even split used in the paper.
    static_prefill_share: float = 0.5

    # Optional prototype costs. They remain zero until measured.
    kv_handoff_s: float = 0.0
    recovery_delay_s: float = 0.0

    # Retained only for backwards compatibility with the older pressure-based
    # experiments.  The latency-share controller does not use neutralization.
    neutralize_when_idle: bool = False
    idle_neutral_delay_s: float = 180.0

    @property
    def hard_stop_s(self) -> float:
        return self.duration_s + max(0.0, self.drain_timeout_s)


class Simulator:
    """Discrete-event P/D serving simulator."""

    def __init__(
        self,
        policy: str,
        requests: Sequence[RequestJob],
        resource_events: Sequence[tuple[float, str, list[int]]],
        cfg: SimConfig,
        base_prefill: int,
        base_decode: int,
        use_resource_trace: bool,
    ) -> None:
        if policy not in {"fixed", "static", "adaptive"}:
            raise ValueError(f"unknown policy {policy}")
        if cfg.controller_mode == "slo_pressure":
            if (
                cfg.ttft_target_s is None
                or cfg.tpot_target_s is None
                or cfg.ttft_target_s <= 0
                or cfg.tpot_target_s <= 0
            ):
                raise ValueError(
                    "slo_pressure requires positive ttft_target_s and tpot_target_s"
                )
        self.policy = policy
        self.requests = {r.request_id: r for r in requests}
        self.resource_events = list(resource_events)
        self.cfg = cfg
        self.use_resource_trace = use_resource_trace and policy != "fixed"

        self.workers: Dict[WorkerId, Worker] = {}
        self.role_memory: Dict[WorkerId, str] = {}
        for i in range(base_prefill):
            self._add_base(f"base-p-{i}", "prefill")
        for i in range(base_decode):
            self._add_base(f"base-d-{i}", "decode")

        self.prefill_queue: Deque[int] = deque()
        self.decode_queue: Deque[int] = deque()

        # Recent normalized phase slowdowns.  A value of 1 means service with no
        # queueing; values >1 indicate waiting/contended phase latency.
        self.prefill_slowdowns: Deque[tuple[float, float]] = deque()
        self.decode_slowdowns: Deque[tuple[float, float]] = deque()
        self.prefill_phase_latencies: Deque[tuple[float, float]] = deque()
        self.decode_phase_latencies: Deque[tuple[float, float]] = deque()
        self.prompt_lengths_recent: Deque[tuple[float, float]] = deque()
        self.phase_demand_recent: Deque[tuple[float, float]] = deque()
        self.prefill_work_recent: Deque[tuple[float, float]] = deque()
        self.decode_work_recent: Deque[tuple[float, float]] = deque()
        self.arrival_markers_recent: Deque[tuple[float, float]] = deque()
        self.arrivals_since_balance = 0
        self.routing_table: Dict[tuple[str, str, float], float] = {}

        # User-visible recent metrics.
        self.ttft_recent: Deque[tuple[float, float]] = deque()
        self.tpot_recent: Deque[tuple[float, float]] = deque()
        self.prefill_wait_recent: Deque[tuple[float, float]] = deque()
        self.decode_wait_recent: Deque[tuple[float, float]] = deque()
        self.e2e_recent: Deque[tuple[float, float]] = deque()

        self.all_ttft: list[float] = []
        self.all_tpot: list[float] = []
        self.all_prefill_wait: list[float] = []
        self.all_decode_wait: list[float] = []
        self.all_e2e: list[float] = []
        self.completed_records: list[dict] = []

        self.arrived = 0
        self.completed = 0
        self.completed_by_cutoff = 0
        self.retries = 0
        self.prefill_retries = 0
        self.decode_retries = 0
        self.role_change_requests = 0
        self.role_changes = 0
        self.last_rebalance_action = -math.inf
        self.idle_since: Optional[float] = None
        self.last_target_prefill_share: Optional[float] = None
        self.last_prefill_signal: float = math.nan
        self.last_decode_signal: float = math.nan
        self.rebalance_cooldown_until: float = 0.0
        self.simulation_end_time: float = 0.0

        self.timeline: list[dict] = []
        self.role_events: list[dict] = []
        self._events: list[tuple[float, int, str, dict]] = []
        self._seq = 0

    def _add_base(self, wid: WorkerId, role: str) -> None:
        self.workers[wid] = Worker(wid, role, persistent=True)
        self.role_memory[wid] = role

    def _push(self, when: float, kind: str, payload: Optional[dict] = None) -> None:
        if when > self.cfg.hard_stop_s + 1e-9:
            return
        self._seq += 1
        heapq.heappush(self._events, (when, self._seq, kind, payload or {}))

    def initialize(self) -> None:
        for r in self.requests.values():
            self._push(r.arrival_time, "arrival", {"request_id": r.request_id})
        if self.use_resource_trace:
            for t, kind, ids in self.resource_events:
                self._push(t, "resource", {"event_type": kind, "worker_ids": ids})
        if self.policy == "adaptive":
            t = self.cfg.balance_interval_s
            while t <= self.cfg.hard_stop_s:
                self._push(t, "rebalance")
                t += self.cfg.balance_interval_s
        t = 0.0
        while t <= self.cfg.hard_stop_s:
            self._push(t, "sample")
            t += self.cfg.sample_interval_s

    def _active(self, role: Optional[str] = None) -> list[Worker]:
        ws = [w for w in self.workers.values() if w.state == "active"]
        if role is not None:
            ws = [w for w in ws if w.role == role]
        return ws

    def _capacity(self, w: Worker) -> int:
        return 1 if w.role == "prefill" else self.cfg.decode_max_concurrency

    def _slots(self, w: Worker) -> int:
        if w.state != "active" or w.pending_role is not None:
            return 0
        return max(0, self._capacity(w) - len(w.in_flight))

    def _decode_tpot(self, concurrency: int) -> float:
        if self.cfg.decode_profile:
            return interp_decode_tpot_s(
                max(1, concurrency),
                self.cfg.decode_profile,
            )[0]
        return self.cfg.decode_tpot_s

    def _update_decode_progress(self, worker: Worker, now: float) -> None:
        """Advance all active decode sequences using the old concurrency.

        vLLM continuous batching changes the per-sequence generation rate when
        sequences join or leave a batch. We therefore advance every active
        sequence to ``now`` before changing membership, then invalidate and
        reschedule its completion event at the new concurrency.
        """
        if worker.role != "decode":
            return
        stages = [s for s in worker.in_flight.values() if s.stage == "decode"]
        if not stages:
            worker.decode_last_update = now
            return
        if worker.decode_last_update is None:
            worker.decode_last_update = now
            return
        elapsed = max(0.0, now - worker.decode_last_update)
        if elapsed <= 0.0:
            return
        tpot = max(1e-12, self._decode_tpot(len(stages)))
        generated = elapsed / tpot
        for stage in stages:
            stage.remaining_tokens = max(0.0, stage.remaining_tokens - generated)
        worker.decode_last_update = now

    def _reschedule_decode(self, worker: Worker, now: float) -> None:
        if worker.role != "decode":
            return
        worker.decode_epoch += 1
        stages = [s for s in worker.in_flight.values() if s.stage == "decode"]
        worker.decode_last_update = now
        if not stages:
            return
        tpot = max(1e-12, self._decode_tpot(len(stages)))
        for stage in stages:
            finish = now + max(0.0, stage.remaining_tokens) * tpot
            self._push(
                finish,
                "decode_complete",
                {
                    "worker_id": worker.worker_id,
                    "generation": worker.generation,
                    "decode_epoch": worker.decode_epoch,
                    "request_id": stage.request_id,
                },
            )

    def _dispatch(self, now: float, stage: str) -> None:
        queue = self.prefill_queue if stage == "prefill" else self.decode_queue
        while queue:
            candidates = [w for w in self._active(stage) if self._slots(w) > 0]
            if not candidates:
                return
            worker = min(
                candidates,
                key=lambda w: (len(w.in_flight), str(w.worker_id)),
            )
            rid = queue.popleft()
            req = self.requests[rid]

            if stage == "prefill":
                wait = max(0.0, now - req.prefill_enqueued)
                req.prefill_wait_total_s += wait
                self.prefill_wait_recent.append((now, wait))
                self.all_prefill_wait.append(wait)
                req.prefill_start = now

                service = interp_prefill_s(
                    req.input_length,
                    self.cfg.prefill_profile,
                )[0]
                finish = now + service
                worker.in_flight[rid] = InFlightStage(
                    rid,
                    "prefill",
                    worker.generation,
                    finish,
                    stage_start=now,
                )
                self._push(
                    finish,
                    "stage_complete",
                    {
                        "worker_id": worker.worker_id,
                        "generation": worker.generation,
                        "request_id": rid,
                        "stage": "prefill",
                        "stage_start": now,
                    },
                )
                continue

            # Decode admission changes the batching concurrency. First advance
            # existing sequences at the old rate, then add this request and
            # reschedule all sequence completions at the new measured TPOT.
            self._update_decode_progress(worker, now)
            req.decode_start = now
            if req.decode_enqueued is not None:
                wait = max(0.0, now - req.decode_enqueued)
                req.decode_wait_total_s += wait
                self.decode_wait_recent.append((now, wait))
                self.all_decode_wait.append(wait)

            remaining = float(max(0, req.output_length - 1))
            if remaining <= 0.0:
                self._complete_request(now, req)
                continue

            worker.in_flight[rid] = InFlightStage(
                rid,
                "decode",
                worker.generation,
                math.inf,
                remaining_tokens=remaining,
                stage_start=now,
            )
            self._reschedule_decode(worker, now)

    def _complete_request(self, now: float, req: RequestJob) -> None:
        if req.completion_time is not None:
            return
        req.completion_time = now
        e2e = now - req.arrival_time
        self.e2e_recent.append((now, e2e))
        self.all_e2e.append(e2e)
        self.completed += 1
        if now <= self.cfg.duration_s + 1e-9:
            self.completed_by_cutoff += 1

        ttft = (
            req.first_token_time - req.arrival_time
            if req.first_token_time is not None
            else math.nan
        )
        self.completed_records.append(
            {
                "request_id": req.request_id,
                "regime": req.regime,
                "arrival_time": req.arrival_time,
                "completion_time": now,
                "input_length": req.input_length,
                "output_length": req.output_length,
                "prefill_wait_s": req.prefill_wait_total_s,
                "ttft_s": ttft,
                "decode_wait_s": req.decode_wait_total_s,
                "e2e_s": e2e,
                "retries": req.retries,
                "prefill_retries": req.prefill_retries,
                "decode_retries": req.decode_retries,
            }
        )

    def _arrival(self, now: float, rid: int) -> None:
        req = self.requests[rid]
        req.prefill_enqueued = now
        self.prefill_queue.append(rid)
        self.prompt_lengths_recent.append((now, float(req.input_length)))
        share = phase_demand_share(
            req.input_length,
            req.output_length,
            self.cfg.decode_tpot_s,
            self.cfg.decode_max_concurrency,
            self.cfg.prefill_profile,
            self.cfg.decode_profile,
        )
        self.phase_demand_recent.append((now, share))

        pwork = interp_prefill_s(
            req.input_length,
            self.cfg.prefill_profile,
        )[0]
        saturated_tpot = self._decode_tpot(self.cfg.decode_max_concurrency)
        dwork = (
            decode_service_s(req.output_length, saturated_tpot)
            / max(1, self.cfg.decode_max_concurrency)
        )
        self.prefill_work_recent.append((now, pwork))
        self.decode_work_recent.append((now, dwork))
        self.arrival_markers_recent.append((now, 1.0))
        self.arrived += 1
        self.arrivals_since_balance += 1
        self._dispatch(now, "prefill")

    def _finish_stage(self, now: float, payload: dict) -> None:
        """Handle prefill completion.

        Decode completion is handled separately because its finish time changes
        whenever the active decode concurrency changes.
        """
        worker = self.workers.get(payload["worker_id"])
        if (
            worker is None
            or worker.state not in {"active", "draining"}
            or worker.generation != payload["generation"]
        ):
            return
        rid = payload["request_id"]
        inflight = worker.in_flight.get(rid)
        if inflight is None or inflight.stage != "prefill":
            return
        del worker.in_flight[rid]
        req = self.requests[rid]
        service = max(1e-9, now - float(payload["stage_start"]))

        phase_latency = now - req.prefill_enqueued
        self.prefill_slowdowns.append((now, max(1.0, phase_latency / service)))
        self.prefill_phase_latencies.append((now, phase_latency))
        req.first_token_time = now
        ttft = now - req.arrival_time
        self.ttft_recent.append((now, ttft))
        self.all_ttft.append(ttft)

        if req.output_length <= 1:
            self._complete_request(now, req)
        else:
            # TTFT ends at prefill completion; a measured KV handoff can be
            # inserted here later without changing the controller interface.
            ready_time = now + max(0.0, self.cfg.kv_handoff_s)
            self._push(ready_time, "kv_ready", {"request_id": rid})

        self._apply_pending_if_idle(worker, now)
        self._dispatch(now, "prefill")

    def _kv_ready(self, now: float, rid: int) -> None:
        req = self.requests[rid]
        if req.completion_time is not None:
            return
        req.decode_enqueued = now
        self.decode_queue.append(rid)
        self._dispatch(now, "decode")

    def _finish_decode(self, now: float, payload: dict) -> None:
        worker = self.workers.get(payload["worker_id"])
        if (
            worker is None
            or worker.state not in {"active", "draining"}
            or worker.generation != payload["generation"]
            or worker.decode_epoch != payload["decode_epoch"]
        ):
            return

        rid = payload["request_id"]
        inflight = worker.in_flight.get(rid)
        if inflight is None or inflight.stage != "decode":
            return

        self._update_decode_progress(worker, now)
        inflight = worker.in_flight.get(rid)
        if inflight is None:
            return
        if inflight.remaining_tokens > 1e-7:
            self._reschedule_decode(worker, now)
            return

        del worker.in_flight[rid]
        req = self.requests[rid]
        if req.decode_enqueued is not None:
            phase_latency = now - req.decode_enqueued
            service = max(1e-9, now - inflight.stage_start)
            self.decode_slowdowns.append((now, max(1.0, phase_latency / service)))
            self.decode_phase_latencies.append((now, phase_latency))
            if req.output_length > 1:
                tpot = service / max(1, req.output_length - 1)
                self.tpot_recent.append((now, tpot))
                self.all_tpot.append(tpot)

        self._complete_request(now, req)

        # Removing one sequence changes the measured TPOT for all remaining
        # sequences, so reschedule them at the new concurrency.
        self._reschedule_decode(worker, now)
        self._apply_pending_if_idle(worker, now)
        self._dispatch(now, "decode")

    def _recent_demand_share(self, now: float) -> Optional[float]:
        self._trim(self.prefill_work_recent, now)
        self._trim(self.decode_work_recent, now)
        self._trim(self.arrival_markers_recent, now)
        if len(self.arrival_markers_recent) < self.cfg.min_control_samples:
            return None
        pwork = sum(v for _, v in self.prefill_work_recent)
        dwork = sum(v for _, v in self.decode_work_recent)
        if pwork + dwork <= 0:
            return None
        return pwork / (pwork + dwork)

    @staticmethod
    def _target_prefill_count(total_workers: int, share: float) -> int:
        if total_workers <= 1:
            return total_workers
        target = int(round(float(share) * total_workers))
        return max(1, min(total_workers - 1, target))

    def _initial_role(self, wid: WorkerId, now: float) -> str:
        """Assign a newly ready worker without changing existing workers."""
        total_after = len(self._active()) + 1
        current_prefill = len(self._active("prefill"))

        if self.policy == "static":
            target_share = self.cfg.static_prefill_share
        elif self.policy == "adaptive":
            target_share = self._recent_demand_share(now)
            if target_share is None:
                target_share = 0.5
        else:
            target_share = 0.5

        target_prefill = self._target_prefill_count(total_after, target_share)
        return "prefill" if current_prefill < target_prefill else "decode"

    def _gain(self, now: float, ids: Iterable[int]) -> None:
        for wid in ids:
            old = self.workers.get(wid)
            generation = 0 if old is None else old.generation + 1
            self.workers[wid] = Worker(
                worker_id=wid,
                role="unassigned",
                persistent=False,
                state="starting",
                generation=generation,
            )
            self._push(
                now + self.cfg.startup_s,
                "worker_ready",
                {"worker_id": wid, "generation": generation},
            )

    def _ready(self, now: float, payload: dict) -> None:
        worker = self.workers.get(payload["worker_id"])
        if (
            worker is None
            or worker.generation != payload["generation"]
            or worker.state != "starting"
        ):
            return
        worker.role = self._initial_role(worker.worker_id, now)
        worker.state = "active"
        worker.last_role_change_time = now
        self.role_memory[worker.worker_id] = worker.role
        self.role_events.append(
            {
                "time": now,
                "event": "ready",
                "worker_id": worker.worker_id,
                "role": worker.role,
            }
        )
        self._dispatch(now, worker.role)

    def _loss_warning(self, now: float, ids: Iterable[int]) -> None:
        """Stop routing new work to workers scheduled for reclamation.

        Existing in-flight work is allowed to drain until the actual loss event.
        Starting workers that receive a warning never enter the ready pool.
        """
        for wid in ids:
            worker = self.workers.get(wid)
            if worker is None or worker.state in {"absent", "draining"}:
                continue
            if worker.role == "decode":
                self._update_decode_progress(worker, now)
            worker.pending_role = None
            worker.pending_since = None
            worker.state = "draining"
            self.role_events.append(
                {
                    "time": now,
                    "event": "loss_warning",
                    "worker_id": wid,
                    "role": worker.role,
                    "in_flight": len(worker.in_flight),
                }
            )

        if self.policy == "adaptive":
            self._ensure_feasible(now)
        self._dispatch(now, "prefill")
        self._dispatch(now, "decode")

    def _loss(self, now: float, ids: Iterable[int]) -> None:
        for wid in ids:
            worker = self.workers.get(wid)
            if worker is None or worker.state == "absent":
                continue

            if worker.role == "decode":
                self._update_decode_progress(worker, now)

            for rid, stage in list(worker.in_flight.items()):
                req = self.requests[rid]
                req.retries += 1
                self.retries += 1

                if stage.stage == "prefill":
                    req.prefill_retries += 1
                    self.prefill_retries += 1
                    retry_at = now + max(0.0, self.cfg.recovery_delay_s)
                    req.prefill_enqueued = retry_at
                    self._push(
                        retry_at,
                        "retry_prefill",
                        {"request_id": rid},
                    )
                else:
                    # Prompt KV state is assumed to have been externalized at
                    # successful prefill. Decode therefore restarts without
                    # repeating prefill, but lost generation progress is not
                    # preserved.
                    req.decode_retries += 1
                    self.decode_retries += 1
                    retry_at = now + max(0.0, self.cfg.recovery_delay_s)
                    req.decode_enqueued = retry_at
                    self._push(
                        retry_at,
                        "retry_decode",
                        {"request_id": rid},
                    )

            worker.in_flight.clear()
            worker.pending_role = None
            worker.pending_since = None
            worker.generation += 1
            worker.decode_epoch += 1
            worker.state = "absent"
            self.role_events.append(
                {"time": now, "event": "loss", "worker_id": wid, "role": worker.role}
            )

        if self.policy == "adaptive":
            self.rebalance_cooldown_until = max(
                self.rebalance_cooldown_until,
                now + 2.0 * self.cfg.balance_interval_s,
            )
            self._ensure_feasible(now)
        self._dispatch(now, "prefill")
        self._dispatch(now, "decode")

    def _retry_prefill(self, now: float, rid: int) -> None:
        req = self.requests[rid]
        if req.completion_time is not None:
            return
        req.prefill_enqueued = now
        self.prefill_queue.appendleft(rid)
        self._dispatch(now, "prefill")

    def _retry_decode(self, now: float, rid: int) -> None:
        req = self.requests[rid]
        if req.completion_time is not None:
            return
        req.decode_enqueued = now
        self.decode_queue.appendleft(rid)
        self._dispatch(now, "decode")

    def _ensure_feasible(self, now: float) -> None:
        active = self._active()
        if len(active) < 2:
            return
        p = len(self._active("prefill"))
        d = len(self._active("decode"))
        if p and d:
            return
        target = "prefill" if p == 0 else "decode"
        source = "decode" if target == "prefill" else "prefill"
        candidates = [w for w in self._active(source) if w.pending_role is None]
        if not candidates:
            return
        worker = min(candidates, key=lambda w: (len(w.in_flight), str(w.worker_id)))
        worker.pending_role = target
        worker.pending_since = now
        self._apply_pending_if_idle(worker, now)

    def _apply_pending_if_idle(self, worker: Worker, now: float) -> None:
        if worker.state != "active" or worker.pending_role is None or worker.in_flight:
            return
        old = worker.role
        new = worker.pending_role
        worker.pending_role = None
        worker.pending_since = None
        worker.role = new
        worker.last_role_change_time = now
        if new == "decode":
            worker.decode_last_update = now
        self.role_memory[worker.worker_id] = new
        self.role_changes += 1
        self.role_events.append(
            {
                "time": now,
                "event": "role_change",
                "worker_id": worker.worker_id,
                "from_role": old,
                "role": new,
            }
        )
        self._dispatch(now, new)

    def _trim(self, dq: Deque[tuple[float, float]], now: float) -> None:
        cutoff = now - self.cfg.metric_window_s
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _recent_mean(
        self, dq: Deque[tuple[float, float]], now: float
    ) -> Optional[float]:
        self._trim(dq, now)
        return mean(v for _, v in dq) if dq else None

    def _pressure(self, now: float) -> Optional[tuple[float, float]]:
        pslow = self._recent_mean(self.prefill_slowdowns, now)
        dslow = self._recent_mean(self.decode_slowdowns, now)
        if pslow is None or dslow is None:
            return None
        pworkers = len(self._active("prefill"))
        dworkers = len(self._active("decode"))
        if pworkers < 1 or dworkers < 1:
            return None
        p_backlog = len(self.prefill_queue) / max(1, pworkers)
        d_backlog = len(self.decode_queue) / max(
            1, dworkers * self.cfg.decode_max_concurrency
        )
        return pslow * (1.0 + p_backlog), dslow * (1.0 + d_backlog)

    def _request_role_change(
        self,
        now: float,
        source: str,
        target: str,
        reason: str,
        pscore: float = math.nan,
        dscore: float = math.nan,
        count: Optional[int] = None,
    ) -> None:
        # The strengthened demand-share controller allows only one drain
        # transaction at a time. Legacy controller sensitivity runs may
        # explicitly permit multiple pending moves by raising max_moves.
        if (
            self.cfg.controller_mode == "demand_share"
            and any(w.pending_role is not None for w in self.workers.values())
        ):
            return

        candidates = [
            w
            for w in self._active(source)
            if w.pending_role is None
            and now - w.last_role_change_time >= self.cfg.min_role_dwell_s
        ]
        effective_source = len(self._active(source))
        if effective_source <= 1 or not candidates:
            return

        # Reclamation-aware selection. When adding decode capacity, prefer a
        # persistent donor. When adding prefill capacity, prefer a transient
        # donor. Current load is the tie-breaker.
        def candidate_key(worker: Worker) -> tuple[int, int, str]:
            if target == "decode":
                class_rank = 0 if worker.persistent else 1
            else:
                class_rank = 0 if not worker.persistent else 1
            return class_rank, len(worker.in_flight), str(worker.worker_id)

        candidates.sort(key=candidate_key)
        move_limit = self.cfg.max_moves_per_interval
        if count is not None:
            move_limit = min(move_limit, max(0, int(count)))
        move_limit = min(move_limit, effective_source - 1)
        if move_limit <= 0:
            return

        selected = candidates[:move_limit]
        for worker in selected:
            worker.pending_role = target
            worker.pending_since = now
            self.role_change_requests += 1
            self.role_events.append(
                {
                    "time": now,
                    "event": "role_change_requested",
                    "worker_id": worker.worker_id,
                    "from_role": source,
                    "role": target,
                    "reason": reason,
                    "prefill_pressure": pscore,
                    "decode_pressure": dscore,
                    "prefill_queue": len(self.prefill_queue),
                    "decode_queue": len(self.decode_queue),
                    "persistent": worker.persistent,
                    "in_flight": len(worker.in_flight),
                }
            )
            self._apply_pending_if_idle(worker, now)
        if selected:
            self.last_rebalance_action = now

    @staticmethod
    def _round_share(value: float, step: float) -> float:
        step = max(1e-9, step)
        return round(float(value) / step) * step

    @staticmethod
    def _rps_bucket(rps: float) -> str:
        if rps < 20.0:
            return "0-20"
        if rps < 40.0:
            return "20-40"
        if rps < 60.0:
            return "40-60"
        return "60+"

    @staticmethod
    def _prompt_bucket(tokens: float) -> str:
        return "2048-4096" if tokens < 4096.0 else "4096+"

    def _demand_share_rebalance(self, now: float) -> None:
        """Allocate workers from recent offered phase work.

        The target uses measured prefill time and measured decode throughput
        rather than queue-inflated latency. Queue presence gates changes: if
        neither phase is queued, role movement cannot improve queueing latency
        and only introduces drain overhead.
        """
        if not self.prefill_queue and not self.decode_queue:
            return

        target_share = self._recent_demand_share(now)
        if target_share is None:
            return

        active = self._active()
        total_workers = len(active)
        if total_workers < 2:
            return
        current_prefill = len(self._active("prefill"))
        current_share = current_prefill / total_workers
        target_prefill = self._target_prefill_count(total_workers, target_share)

        self._trim(self.prefill_work_recent, now)
        self._trim(self.decode_work_recent, now)
        pwork_rate = sum(v for _, v in self.prefill_work_recent) / max(
            1e-9, self.cfg.metric_window_s
        )
        dwork_rate = sum(v for _, v in self.decode_work_recent) / max(
            1e-9, self.cfg.metric_window_s
        )
        self.last_prefill_signal = pwork_rate
        self.last_decode_signal = dwork_rate
        self.last_target_prefill_share = target_share

        if target_prefill == current_prefill:
            return
        if abs(target_share - current_share) < self.cfg.demand_deadband:
            return

        if target_prefill > current_prefill:
            self._request_role_change(
                now,
                "decode",
                "prefill",
                "demand-share",
                pwork_rate,
                dwork_rate,
                count=1,
            )
        else:
            self._request_role_change(
                now,
                "prefill",
                "decode",
                "demand-share",
                pwork_rate,
                dwork_rate,
                count=1,
            )

    def _latency_share_rebalance(self, now: float) -> None:
        # The deployed router reports RPS since the previous controller tick.
        rps = self.arrivals_since_balance / max(1e-9, self.cfg.balance_interval_s)
        self.arrivals_since_balance = 0

        avg_prompt = self._recent_mean(self.prompt_lengths_recent, now)
        if avg_prompt is None or avg_prompt < self.cfg.prompt_threshold_tokens:
            return

        prefill_latency = self._recent_mean(self.prefill_phase_latencies, now)
        decode_latency = self._recent_mean(self.decode_phase_latencies, now)
        if prefill_latency is None or decode_latency is None:
            return

        total_latency = prefill_latency + decode_latency
        if total_latency <= 0:
            return

        active = self._active()
        total_workers = len(active)
        if total_workers < 2:
            return

        current_prefill = len(self._active("prefill"))
        current_share = self._round_share(
            current_prefill / total_workers,
            self.cfg.share_rounding,
        )
        rps_bucket = self._rps_bucket(rps)
        prompt_bucket = self._prompt_bucket(avg_prompt)
        current_perf = total_latency
        current_key = (rps_bucket, prompt_bucket, current_share)
        previous = self.routing_table.get(current_key)
        if previous is None or current_perf + 1e-3 < previous:
            self.routing_table[current_key] = current_perf

        raw_share = prefill_latency / total_latency
        target_share = self._round_share(raw_share, self.cfg.share_rounding)
        target_share = min(1.0, max(0.0, target_share))
        target_prefill = round(target_share * total_workers)
        target_prefill = min(total_workers - 1, max(1, target_prefill))

        delta = target_prefill - current_prefill
        if delta == 0:
            self.last_target_prefill_share = current_prefill / total_workers
            return

        move = min(self.cfg.max_moves_per_interval, abs(delta))
        candidate_prefill = current_prefill + (move if delta > 0 else -move)
        candidate_share = self._round_share(
            candidate_prefill / total_workers,
            self.cfg.share_rounding,
        )
        known_perf = self.routing_table.get(
            (rps_bucket, prompt_bucket, candidate_share)
        )
        if known_perf is not None and known_perf > current_perf + 1e-3:
            return

        self.last_target_prefill_share = candidate_prefill / total_workers
        self.last_prefill_signal = prefill_latency
        self.last_decode_signal = decode_latency
        if delta > 0:
            self._request_role_change(
                now,
                "decode",
                "prefill",
                "latency-share",
                prefill_latency,
                decode_latency,
                count=move,
            )
        else:
            self._request_role_change(
                now,
                "prefill",
                "decode",
                "latency-share",
                prefill_latency,
                decode_latency,
                count=move,
            )

    def _slo_pressure_rebalance(self, now: float) -> None:
        """Normalized p95 TTFT/TPOT pressure from the paper design."""
        self._trim(self.ttft_recent, now)
        self._trim(self.tpot_recent, now)
        ttft = [v for _, v in self.ttft_recent]
        tpot = [v for _, v in self.tpot_recent]
        if min(len(ttft), len(tpot)) < self.cfg.min_control_samples:
            return

        p95_ttft = self.percentile(ttft, 0.95)
        p95_tpot = self.percentile(tpot, 0.95)
        pscore = p95_ttft / float(self.cfg.ttft_target_s)
        dscore = p95_tpot / float(self.cfg.tpot_target_s)
        self.last_prefill_signal = pscore
        self.last_decode_signal = dscore

        if pscore <= 1.0 and dscore <= 1.0:
            return

        margin = 1.0 + max(0.0, self.cfg.slo_pressure_deadband)
        if pscore > max(1.0, dscore * margin):
            self._request_role_change(
                now,
                "decode",
                "prefill",
                "slo-pressure",
                pscore,
                dscore,
                count=1,
            )
        elif dscore > max(1.0, pscore * margin):
            self._request_role_change(
                now,
                "prefill",
                "decode",
                "slo-pressure",
                pscore,
                dscore,
                count=1,
            )

    def _pressure_rebalance(self, now: float) -> None:
        pworkers = len(self._active("prefill"))
        dworkers = len(self._active("decode"))

        recent_demand = self._recent_mean(self.phase_demand_recent, now)
        lightly_queued = (
            len(self.prefill_queue) <= max(1, pworkers)
            and len(self.decode_queue) <= max(1, dworkers)
        )
        balanced_recent_demand = (
            recent_demand is not None and 0.40 <= recent_demand <= 0.60
        )
        if self.cfg.neutralize_when_idle and lightly_queued and balanced_recent_demand:
            if self.idle_since is None:
                self.idle_since = now
            if now - self.idle_since >= self.cfg.idle_neutral_delay_s:
                if pworkers > dworkers + 1:
                    self._request_role_change(
                        now, "prefill", "decode", "balanced-workload-neutral"
                    )
                elif dworkers > pworkers + 1:
                    self._request_role_change(
                        now, "decode", "prefill", "balanced-workload-neutral"
                    )
            return

        self.idle_since = None
        pressure = self._pressure(now)
        if pressure is None:
            return
        pscore, dscore = pressure
        if pscore > dscore * self.cfg.pressure_hysteresis:
            self._request_role_change(
                now, "decode", "prefill", "prefill-pressure", pscore, dscore
            )
        elif dscore > pscore * self.cfg.pressure_hysteresis:
            self._request_role_change(
                now, "prefill", "decode", "decode-pressure", pscore, dscore
            )

    def _rebalance(self, now: float) -> None:
        if now < self.rebalance_cooldown_until:
            if self.cfg.controller_mode == "latency_share":
                self.arrivals_since_balance = 0
            return
        if now - self.last_rebalance_action < self.cfg.rebalance_cooldown_s:
            return
        if len(self._active()) < 2:
            return
        if self.cfg.controller_mode == "demand_share":
            self._demand_share_rebalance(now)
        elif self.cfg.controller_mode == "latency_share":
            self._latency_share_rebalance(now)
        elif self.cfg.controller_mode == "slo_pressure":
            self._slo_pressure_rebalance(now)
        elif self.cfg.controller_mode == "pressure":
            self._pressure_rebalance(now)
        else:
            raise ValueError(
                f"unknown controller_mode {self.cfg.controller_mode!r}"
            )

    @staticmethod
    def percentile(values: Sequence[float], q: float) -> float:
        vals = sorted(v for v in values if not math.isnan(v))
        if not vals:
            return math.nan
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return vals[lo]
        frac = pos - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    def _sample(self, now: float) -> None:
        for dq in (
            self.ttft_recent,
            self.tpot_recent,
            self.prefill_wait_recent,
            self.decode_wait_recent,
            self.e2e_recent,
            self.prefill_slowdowns,
            self.decode_slowdowns,
            self.prefill_phase_latencies,
            self.decode_phase_latencies,
            self.prompt_lengths_recent,
            self.phase_demand_recent,
            self.prefill_work_recent,
            self.decode_work_recent,
            self.arrival_markers_recent,
        ):
            self._trim(dq, now)

        ttft = [v for _, v in self.ttft_recent]
        tpot = [v for _, v in self.tpot_recent]
        pwait = [v for _, v in self.prefill_wait_recent]
        dwait = [v for _, v in self.decode_wait_recent]
        e2e = [v for _, v in self.e2e_recent]

        pwork_rate = sum(v for _, v in self.prefill_work_recent) / max(
            1e-9, self.cfg.metric_window_s
        )
        dwork_rate = sum(v for _, v in self.decode_work_recent) / max(
            1e-9, self.cfg.metric_window_s
        )
        demand_share = (
            pwork_rate / (pwork_rate + dwork_rate)
            if pwork_rate + dwork_rate > 0
            else math.nan
        )

        active = self._active()
        pworkers = len(self._active("prefill"))
        dworkers = len(self._active("decode"))
        current_share = pworkers / len(active) if active else math.nan

        self.timeline.append(
            {
                "time": now,
                "policy": self.policy,
                "controller": self.cfg.controller_mode,
                "active_workers": len(active),
                "starting_workers": sum(
                    w.state == "starting" for w in self.workers.values()
                ),
                "draining_workers": sum(
                    w.state == "draining" for w in self.workers.values()
                ),
                "available_workers": sum(
                    w.state != "absent" for w in self.workers.values()
                ),
                "prefill_workers": pworkers,
                "decode_workers": dworkers,
                "prefill_share": current_share,
                "target_prefill_share": self.last_target_prefill_share,
                "recent_demand_prefill_share": demand_share,
                "prefill_demand_worker_s_per_s": pwork_rate,
                "decode_demand_worker_s_per_s": dwork_rate,
                "prefill_control_signal": self.last_prefill_signal,
                "decode_control_signal": self.last_decode_signal,
                "pending_role_changes": sum(
                    w.pending_role is not None for w in self.workers.values()
                ),
                "prefill_queue": len(self.prefill_queue),
                "decode_queue": len(self.decode_queue),
                "ttft_p95_window": self.percentile(ttft, 0.95),
                "tpot_p95_window_ms": 1000.0 * self.percentile(tpot, 0.95),
                "prefill_wait_p95_window": self.percentile(pwait, 0.95),
                "decode_wait_p95_window": self.percentile(dwait, 0.95),
                "e2e_p95_window": self.percentile(e2e, 0.95),
                "arrived": self.arrived,
                "completed": self.completed,
                "completed_by_cutoff": self.completed_by_cutoff,
                "retries": self.retries,
                "role_change_requests": self.role_change_requests,
                "role_changes": self.role_changes,
            }
        )

    def _work_drained(self) -> bool:
        if self.arrived < len(self.requests):
            return False
        if self.prefill_queue or self.decode_queue:
            return False
        return not any(worker.in_flight for worker in self.workers.values())

    def run(self) -> dict:
        self.initialize()
        now = 0.0
        while self._events:
            now, _, kind, payload = heapq.heappop(self._events)
            if now > self.cfg.hard_stop_s:
                break
            if kind == "arrival":
                self._arrival(now, payload["request_id"])
            elif kind == "resource":
                event_type = payload["event_type"]
                if event_type == "gain":
                    self._gain(now, payload["worker_ids"])
                elif event_type == "loss_warning":
                    self._loss_warning(now, payload["worker_ids"])
                elif event_type == "loss":
                    self._loss(now, payload["worker_ids"])
                # gain_warning is informational only.
            elif kind == "worker_ready":
                self._ready(now, payload)
            elif kind == "stage_complete":
                self._finish_stage(now, payload)
            elif kind == "decode_complete":
                self._finish_decode(now, payload)
            elif kind == "kv_ready":
                self._kv_ready(now, payload["request_id"])
            elif kind == "retry_prefill":
                self._retry_prefill(now, payload["request_id"])
            elif kind == "retry_decode":
                self._retry_decode(now, payload["request_id"])
            elif kind == "rebalance":
                self._rebalance(now)
            elif kind == "sample":
                self._sample(now)
            else:
                raise RuntimeError(f"unknown event {kind}")

            if now >= self.cfg.duration_s and self._work_drained():
                break

        self.simulation_end_time = min(now, self.cfg.hard_stop_s)
        return self.summary()

    def summary(self) -> dict:
        arrived = self.arrived
        drained_fraction = self.completed / arrived if arrived else 0.0
        cutoff_fraction = self.completed_by_cutoff / arrived if arrived else 0.0
        return {
            "policy": self.policy,
            "arrived": arrived,
            "completed": self.completed,
            "completed_by_cutoff": self.completed_by_cutoff,
            # Backwards-compatible name: completion within the trace horizon.
            "completion_fraction": cutoff_fraction,
            "completion_by_cutoff_fraction": cutoff_fraction,
            "drained_completion_fraction": drained_fraction,
            "unfinished": max(0, arrived - self.completed),
            "mean_throughput_rps": self.completed_by_cutoff / self.cfg.duration_s,
            "drain_time_s": max(0.0, self.simulation_end_time - self.cfg.duration_s),
            "simulation_end_time_s": self.simulation_end_time,
            "ttft_p50_s": self.percentile(self.all_ttft, 0.50),
            "ttft_p95_s": self.percentile(self.all_ttft, 0.95),
            "tpot_p50_ms": 1000.0 * self.percentile(self.all_tpot, 0.50),
            "tpot_p95_ms": 1000.0 * self.percentile(self.all_tpot, 0.95),
            "prefill_wait_p50_s": self.percentile(self.all_prefill_wait, 0.50),
            "prefill_wait_p95_s": self.percentile(self.all_prefill_wait, 0.95),
            "decode_wait_p50_s": self.percentile(self.all_decode_wait, 0.50),
            "decode_wait_p95_s": self.percentile(self.all_decode_wait, 0.95),
            "e2e_p50_s": self.percentile(self.all_e2e, 0.50),
            "e2e_p95_s": self.percentile(self.all_e2e, 0.95),
            "retries": self.retries,
            "prefill_retries": self.prefill_retries,
            "decode_retries": self.decode_retries,
            "retries_per_1000_requests": 1000.0 * self.retries / max(1, arrived),
            "role_change_requests": self.role_change_requests,
            "role_changes": self.role_changes,
        }


def clone_requests(reqs: Sequence[RequestJob]) -> list[RequestJob]:
    return [
        RequestJob(
            request_id=r.request_id,
            arrival_time=r.arrival_time,
            input_length=r.input_length,
            output_length=r.output_length,
            regime=r.regime,
        )
        for r in reqs
    ]


def load_mooncake_shapes(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not rows:
        raise RuntimeError(f"no requests in {path}")
    return [
        {
            "input_length": int(row.get("input_length", 1)),
            "output_length": int(row.get("output_length", 1)),
        }
        for row in rows
    ]


def constant_rate_requests(
    pool: Sequence[dict],
    duration_s: float,
    rate_rps: float,
    seed: int,
    regime: str = "trace",
) -> list[RequestJob]:
    if rate_rps <= 0:
        raise ValueError("rate_rps must be positive")
    if not pool:
        raise ValueError("request pool is empty")
    rng = random.Random(seed)
    count = int(math.floor(duration_s * rate_rps))
    return [
        RequestJob(
            request_id=i,
            arrival_time=i / rate_rps,
            input_length=int((template := rng.choice(pool))["input_length"]),
            output_length=int(template["output_length"]),
            regime=regime,
        )
        for i in range(count)
    ]


def build_phase_pools(
    rows: Sequence[dict],
    decode_tpot_s: float,
    decode_concurrency: int,
    prefill_quantile: float = 0.90,
    decode_quantile: float = 0.05,
) -> dict[str, list[dict]]:
    """Construct real-shape pools for controlled phase-pressure regimes.

    The pool score is the ratio of effective prefill worker-seconds to decode
    worker-seconds.  Prefill-heavy uses the upper 10% by default; decode-heavy
    uses the lower 5% because strongly decode-dominated requests are rarer in
    this trace.  Balanced uses scores with a ratio in [0.8, 1.2].
    """
    enriched: list[tuple[dict, float]] = []
    for row in rows:
        p = interp_prefill_s(int(row["input_length"]))[0]
        d = decode_service_s(int(row["output_length"]), decode_tpot_s) / max(
            1, decode_concurrency
        )
        ratio = p / max(1e-12, d)
        enriched.append((row, ratio))
    ordered = sorted(enriched, key=lambda x: x[1])
    n = len(ordered)
    prefill_start = min(n - 1, max(0, int(math.floor(prefill_quantile * n))))
    decode_end = max(1, int(math.ceil(decode_quantile * n)))
    balanced = [row for row, ratio in enriched if 0.8 <= ratio <= 1.2]
    if len(balanced) < 100:
        lo = int(0.45 * n)
        hi = int(0.55 * n)
        balanced = [row for row, _ in ordered[lo:hi]]
    return {
        "prefill-heavy": [row for row, _ in ordered[prefill_start:]],
        "decode-heavy": [row for row, _ in ordered[:decode_end]],
        "balanced": balanced,
    }


def regime_sequence_requests(
    pools: dict[str, list[dict]],
    regimes: Sequence[tuple[str, float]],
    rate_rps: float,
    seed: int,
    heavy_mix: float,
    decode_tpot_s: float,
    decode_concurrency: int,
) -> tuple[list[RequestJob], list[dict]]:
    """Generate one continuous workload whose request *shapes* change by regime.

    Heavy regimes mix observed phase-heavy shapes with balanced shapes rather
    than using only the extreme tail.  ``heavy_mix=0.8`` means 80% phase-heavy
    and 20% balanced shapes.  Arrival rate remains identical in every regime.
    """
    if not 0.0 <= heavy_mix <= 1.0:
        raise ValueError("heavy_mix must be in [0, 1]")
    rng = random.Random(seed)
    reqs: list[RequestJob] = []
    workload_rows: list[dict] = []
    rid = 0
    start = 0.0
    for regime, duration_s in regimes:
        count = int(math.floor(duration_s * rate_rps))
        for i in range(count):
            arrival = start + i / rate_rps
            if regime == "balanced" or rng.random() < heavy_mix:
                pool = pools[regime]
            else:
                pool = pools["balanced"]
            template = rng.choice(pool)
            inp = int(template["input_length"])
            out = int(template["output_length"])
            reqs.append(RequestJob(rid, arrival, inp, out, regime))
            workload_rows.append(
                {
                    "request_id": rid,
                    "arrival_time": arrival,
                    "regime": regime,
                    "input_length": inp,
                    "output_length": out,
                    "prefill_demand_share": phase_demand_share(
                        inp, out, decode_tpot_s, decode_concurrency
                    ),
                }
            )
            rid += 1
        start += duration_s
    return reqs, workload_rows


def load_resource_events(
    path: Path, duration_s: float
) -> list[tuple[float, str, list[int]]]:
    out: list[tuple[float, str, list[int]]] = []
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        kind = str(row.get("event_type", ""))
        if kind not in {"gain_warning", "gain", "loss_warning", "loss"}:
            continue
        t = float(row["timestamp"])
        if t <= duration_s:
            out.append((t, kind, [int(x) for x in row["worker_ids"]]))
    return sorted(out)


def scale_resource_events(
    events: Sequence[tuple[float, str, list[int]]], replicas_per_worker: int
) -> list[tuple[float, str, list[int]]]:
    if replicas_per_worker < 1:
        raise ValueError("replicas_per_worker must be >= 1")
    out: list[tuple[float, str, list[int]]] = []
    for t, kind, ids in events:
        expanded: list[int] = []
        for wid in ids:
            expanded.extend(wid * 1000 + replica for replica in range(replicas_per_worker))
        out.append((t, kind, expanded))
    return out


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Create an empty file so downstream scripts fail predictably on schema,
        # not because a path silently disappeared.
        path.write_text("")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def count_slo_requests(records: Sequence[dict], ttft_slo_s: float) -> int:
    return sum(
        1
        for row in records
        if not math.isnan(float(row["ttft_s"])) and float(row["ttft_s"]) <= ttft_slo_s
    )


def load_mooncake_trace_requests(path: Path, duration_s: float, timestamp_scale: float = 1000.0) -> list[RequestJob]:
    """Load the Mooncake-derived trace while preserving its original timing.

    The bundled trace stores timestamps in milliseconds.  ``timestamp_scale``
    therefore defaults to 1000 so request arrival times are returned in seconds.
    Requests outside ``duration_s`` are omitted.  Unlike ``load_mooncake_shapes``,
    this loader preserves both arrival timing and sequence lengths.
    """
    if timestamp_scale <= 0:
        raise ValueError("timestamp_scale must be positive")
    out: list[RequestJob] = []
    for row_id, line in enumerate(path.open()):
        if not line.strip():
            continue
        row = json.loads(line)
        arrival = float(row.get("timestamp", 0.0)) / timestamp_scale
        if arrival > duration_s + 1e-9:
            continue
        out.append(
            RequestJob(
                request_id=len(out),
                arrival_time=arrival,
                input_length=int(row.get("input_length", 1)),
                output_length=int(row.get("output_length", 1)),
                regime="trace",
            )
        )
    if not out:
        raise RuntimeError(f"no requests in {path} within {duration_s}s")
    return out
