# OpServe trace-driven evaluation

The simulator is intentionally small and explicit. It is not intended to
reproduce every internal scheduling decision made by vLLM. It models the
mechanisms relaevan to the paper and parameterizes service time from the
measured model profiles.

## Workloads

The runner automatically uses the request traces that exist under `traces/`.

Expected names are:

```text
traces/
├── conversation_trace.jsonl
├── synthetic_trace.jsonl
└── toolagent_trace.jsonl
```
Each trace must use the same JSONL schema:

```json
{"timestamp": 10000, "input_length": 8192, "output_length": 256}
```

Timestamps are interpreted as milliseconds by the default loader.

## Models

Profiles are stored in:

```text
profiles/model_profiles.json
```

The bundled profiles contain measured:

- startup time in seconds;
- median isolated TTFT at prompt lengths
  512, 1024, 2048, 4096, 8192, 16384, and 32767 tokens;
- median TPOT at decode concurrency
  1, 2, 4, 8, and 16.

## Policies

The normal evaluation contains:

```text
fixed     Persistent-Only
static    Static P/D
adaptive  OpServe
```

`Persistent-Only` uses only the eight persistent workers.

`Static P/D` admits transient workers and targets an even prefill/decode split
by default. A worker retains its role until reclamation.

`OpServe` sees the same request and resource events and may drain and
reassign ready workers.

## Default adaptive controller

The default controller is:

```text
demand_share
```

For every recent request, the simulator estimates:

```text
prefill worker demand
    = measured TTFT(input length)

decode worker demand
    = (output tokens - 1)
      × TPOT(max decode concurrency)
      / max decode concurrency
```

These worker-second estimates are accumulated over the recent metric window.
The resulting prefill fraction is used as the target phase share.

The controller deliberately avoids a role move when both phase queues are
empty. When a change is needed, the default configuration moves at most one
worker and waits for the reassignment/drain to settle.

Default control parameters are:

```text
balance interval      15 s
metric window         120 s
rebalance cooldown     30 s
minimum role dwell     30 s
maximum moves/action    1
minimum samples        16
demand deadband      0.05
```

## Reclamation and recovery

The resource event trace may contain:

```text
gain_warning
gain
loss_warning
loss
```

`gain_warning` is informational.

At `gain`, model startup begins. The worker becomes ready after the measured
model startup delay.

At `loss_warning`, the worker enters `draining` and accepts no new work.

At `loss`, any unfinished work is interrupted:

- interrupted prefill is retried from prefill;
- interrupted decode restarts decode from the completed prompt state.

Optional measured costs can later be inserted with:

```bash
--kv-handoff SECONDS
--recovery-delay SECONDS
```
## Run the matrix

From the package directory:

```bash
python run_evaluation_matrix.py
```

## Output

The main output directory contains:

```text
summary.csv
workloads.csv
resource_capacity.csv

<workload>/<model>/
    fixed_role_events.csv
    static_role_events.csv
    adaptive_role_events.csv
    *_timeseries.csv
    *_requests.csv
```

`summary.csv` includes:

- completion by one hour;
- drained completion fraction;
- throughput by the one-hour boundary;
- p50/p95 TTFT;
- p50/p95 TPOT;
- p50/p95 prefill queueing delay;
- p50/p95 decode queueing delay;
- p50/p95 end-to-end latency;
- retry counts and retries per 1000 requests;
- role-change requests and completed role changes.