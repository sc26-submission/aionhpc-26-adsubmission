#!/usr/bin/env python3
"""
Measure isolated TTFT/prefill and decode TPOT for an already-running vLLM server.

The script uses `vllm bench serve` so that the reported TTFT and TPOT are the
same metrics used by vLLM's serving benchmark.

Two profiles are collected:

1. Prefill / TTFT profile
   - max concurrency = 1
   - output length = 1 token
   - prompt lengths configurable (default: 512..32767)
   - median and p95 TTFT are recorded

2. Decode profile
   - fixed prompt length (default: 8192)
   - fixed output length (default: 256)
   - concurrency configurable (default: 1,2,4,8,16)
   - median and p95 TPOT are recorded
   - TTFT is also recorded for reference

The script assumes vLLM is already running, for example:

    CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B \
        --port 8001 \
        --tensor-parallel-size 1

Then run:

    python3 measure-model-profile.py \
        --model meta-llama/Llama-3.1-8B \
        --port 8001 \
        --output-dir results/Llama-3.1-8B

Important:
- On HPC systems with HTTP/HTTPS proxies, localhost requests can accidentally
  be sent through the proxy. This script forces NO_PROXY for 127.0.0.1 and
  localhost and performs its own health check with requests trust_env=False.
- `vllm bench serve` must be available in the active environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import requests


DEFAULT_PREFILL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32767]
DEFAULT_CONCURRENCIES = [1, 2, 4, 8, 16]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure TTFT/prefill and decode TPOT using vLLM bench serve."
    )
    p.add_argument("--model", required=True, help="Model name exposed by vLLM.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for raw benchmark JSON and summary CSV files.",
    )
    p.add_argument(
        "--mode",
        choices=["all", "prefill", "decode"],
        default="all",
        help="Which profile to collect.",
    )

    # Prefill / TTFT profile.
    p.add_argument(
        "--prefill-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_PREFILL_LENGTHS,
        help="Prompt lengths for isolated TTFT measurements.",
    )
    p.add_argument(
        "--prefill-output-len",
        type=int,
        default=1,
        help="Output length for isolated TTFT/prefill runs. Keep at 1 for prefill profiling.",
    )
    p.add_argument(
        "--prefill-num-prompts",
        type=int,
        default=16,
        help="Measured prompts per prefill length.",
    )
    p.add_argument(
        "--prefill-warmups",
        type=int,
        default=2,
        help="Warmup requests before each prefill run.",
    )

    # Decode profile.
    p.add_argument(
        "--decode-input-len",
        type=int,
        default=8192,
        help="Prompt length used for decode profiling.",
    )
    p.add_argument(
        "--decode-output-len",
        type=int,
        default=256,
        help="Generated tokens per request for decode profiling.",
    )
    p.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=DEFAULT_CONCURRENCIES,
        help="Concurrent sequence counts for decode profiling.",
    )
    p.add_argument(
        "--decode-min-prompts",
        type=int,
        default=16,
        help="Minimum measured prompts for each decode run.",
    )
    p.add_argument(
        "--decode-waves",
        type=int,
        default=4,
        help=(
            "At least this many waves of requests are measured at each "
            "concurrency. num_prompts=max(decode_min_prompts, concurrency*decode_waves)."
        ),
    )
    p.add_argument(
        "--decode-warmups",
        type=int,
        default=4,
        help="Warmup requests before each decode run.",
    )

    p.add_argument(
        "--metric-percentiles",
        default="50,95",
        help="Percentiles requested from vLLM bench serve.",
    )
    p.add_argument(
        "--vllm-bin",
        default="vllm",
        help="vLLM CLI executable.",
    )
    p.add_argument(
        "--timeout-s",
        type=float,
        default=8.0,
        help="Health-check timeout.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an existing raw JSON result instead of rerunning it.",
    )
    return p.parse_args()


def ensure_no_proxy(env: dict[str, str]) -> None:
    """Ensure localhost traffic does not go through cluster proxy settings."""
    values = []
    for key in ("NO_PROXY", "no_proxy"):
        if env.get(key):
            values.extend(x.strip() for x in env[key].split(",") if x.strip())

    for value in ("127.0.0.1", "localhost"):
        if value not in values:
            values.append(value)

    joined = ",".join(dict.fromkeys(values))
    env["NO_PROXY"] = joined
    env["no_proxy"] = joined


def check_health(host: str, port: int, timeout_s: float) -> None:
    """
    Confirm that the server is reachable without using proxy environment vars.
    """
    session = requests.Session()
    session.trust_env = False
    url = f"http://{host}:{port}/health"

    try:
        response = session.get(url, timeout=timeout_s)
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach vLLM health endpoint {url}: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"vLLM health endpoint returned HTTP {response.status_code}: {url}"
        )


def safe_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def run_benchmark(
    *,
    args: argparse.Namespace,
    input_len: int,
    output_len: int,
    concurrency: int,
    num_prompts: int,
    num_warmups: int,
    result_path: Path,
) -> dict[str, Any]:
    """
    Run vllm bench serve and return its saved JSON result.
    """
    if args.skip_existing and result_path.exists():
        print(f"[reuse] {result_path}", flush=True)
        return load_result(result_path)

    result_path.parent.mkdir(parents=True, exist_ok=True)

    # vLLM writes --result-filename relative to --result-dir.
    result_dir = result_path.parent
    result_filename = result_path.name

    cmd = [
        args.vllm_bin,
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        "0",
        "--num-prompts",
        str(num_prompts),
        "--num-warmups",
        str(num_warmups),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,e2el",
        "--metric-percentiles",
        args.metric_percentiles,
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
    ]

    env = os.environ.copy()
    ensure_no_proxy(env)

    print("\n$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env, check=False)

    if proc.returncode != 0:
        raise RuntimeError(
            f"vllm bench serve failed with exit code {proc.returncode}"
        )

    if not result_path.exists():
        raise RuntimeError(
            f"Benchmark completed but expected result file was not created: "
            f"{result_path}"
        )

    return load_result(result_path)


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    # --append-result can produce a list. We do not use it, but tolerate it.
    if isinstance(obj, list):
        if not obj:
            raise RuntimeError(f"Empty JSON result: {path}")
        obj = obj[-1]

    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected JSON structure in {path}")

    return obj


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def percentile_metric(
    result: dict[str, Any],
    metric: str,
    percentile: float,
) -> float | None:
    """
    Read a percentile from vLLM benchmark JSON across several result formats.

    Examples handled:
      p95_ttft_ms
      percentiles_ttft_ms = [[50, ...], [95, ...]]
      percentiles_ttft_ms = {"50": ..., "95": ...}
    """
    p_int = int(percentile) if float(percentile).is_integer() else percentile

    direct_keys = [
        f"p{p_int}_{metric}_ms",
        f"p{str(p_int).replace('.', '_')}_{metric}_ms",
    ]
    for key in direct_keys:
        if key in result:
            return as_float(result[key])

    container = result.get(f"percentiles_{metric}_ms")

    if isinstance(container, dict):
        for key in (str(p_int), str(float(percentile))):
            if key in container:
                return as_float(container[key])

    if isinstance(container, list):
        for item in container:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                p, value = item[0], item[1]
                try:
                    if float(p) == float(percentile):
                        return as_float(value)
                except (TypeError, ValueError):
                    pass
            elif isinstance(item, dict):
                p = item.get("percentile", item.get("p"))
                value = item.get("value", item.get("latency_ms"))
                try:
                    if p is not None and float(p) == float(percentile):
                        return as_float(value)
                except (TypeError, ValueError):
                    pass

    return None


def median_metric(result: dict[str, Any], metric: str) -> float | None:
    key = f"median_{metric}_ms"
    value = as_float(result.get(key))
    if value is not None:
        return value
    return percentile_metric(result, metric, 50.0)


def mean_metric(result: dict[str, Any], metric: str) -> float | None:
    return as_float(result.get(f"mean_{metric}_ms"))


def completed_count(result: dict[str, Any]) -> int | None:
    value = result.get("completed")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def make_row(
    *,
    phase: str,
    args: argparse.Namespace,
    input_len: int,
    output_len: int,
    concurrency: int,
    num_prompts: int,
    raw_path: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": phase,
        "model": args.model,
        "input_len": input_len,
        "output_len": output_len,
        "concurrency": concurrency,
        "num_prompts": num_prompts,
        "completed": completed_count(result),
        "mean_ttft_ms": mean_metric(result, "ttft"),
        "median_ttft_ms": median_metric(result, "ttft"),
        "p95_ttft_ms": percentile_metric(result, "ttft", 95.0),
        "mean_tpot_ms": mean_metric(result, "tpot"),
        "median_tpot_ms": median_metric(result, "tpot"),
        "p95_tpot_ms": percentile_metric(result, "tpot", 95.0),
        "mean_e2el_ms": mean_metric(result, "e2el"),
        "median_e2el_ms": median_metric(result, "e2el"),
        "p95_e2el_ms": percentile_metric(result, "e2el", 95.0),
        "raw_json": str(raw_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "phase",
        "model",
        "input_len",
        "output_len",
        "concurrency",
        "num_prompts",
        "completed",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p95_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p95_tpot_ms",
        "mean_e2el_ms",
        "median_e2el_ms",
        "p95_e2el_ms",
        "raw_json",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "---"
    return f"{x:.{digits}f}"


def closest_prefill(
    rows: list[dict[str, Any]], target: int
) -> dict[str, Any] | None:
    candidates = [r for r in rows if r["phase"] == "prefill"]
    if not candidates:
        return None
    return min(candidates, key=lambda r: abs(int(r["input_len"]) - target))


def decode_at(
    rows: list[dict[str, Any]], concurrency: int
) -> dict[str, Any] | None:
    for row in rows:
        if row["phase"] == "decode" and int(row["concurrency"]) == concurrency:
            return row
    return None


def print_profile_summary(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("Profile summary")
    print("=" * 72)

    selected = []
    for target in (2048, 8192, 32767):
        row = closest_prefill(rows, target)
        if row is None:
            selected.append("---")
            continue
        ttft_ms = as_float(row.get("median_ttft_ms"))
        selected.append("---" if ttft_ms is None else f"{ttft_ms / 1000.0:.3f}")

    print("Prefill/TTFT median (s), 2K / 8K / 32K:")
    print("    " + " / ".join(selected))

    decode_values = []
    for concurrency in (1, 4, 16):
        row = decode_at(rows, concurrency)
        if row is None:
            decode_values.append("---")
            continue
        decode_values.append(fmt(as_float(row.get("median_tpot_ms")), 2))

    print("Decode median TPOT (ms/token), 1 / 4 / 16 seq.:")
    print("    " + " / ".join(decode_values))

    print("\nLaTeX-ready cells:")
    print("  Prefill: " + " / ".join(selected))
    print("  Decode:  " + " / ".join(decode_values))
    print("=" * 72)


def run_prefill(args: argparse.Namespace, raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    print("\n=== PREFILL / TTFT PROFILE ===", flush=True)
    for input_len in args.prefill_lengths:
        result_path = raw_dir / f"prefill_input{input_len}_c1.json"

        result = run_benchmark(
            args=args,
            input_len=input_len,
            output_len=args.prefill_output_len,
            concurrency=1,
            num_prompts=args.prefill_num_prompts,
            num_warmups=args.prefill_warmups,
            result_path=result_path,
        )

        row = make_row(
            phase="prefill",
            args=args,
            input_len=input_len,
            output_len=args.prefill_output_len,
            concurrency=1,
            num_prompts=args.prefill_num_prompts,
            raw_path=result_path,
            result=result,
        )
        rows.append(row)

        print(
            f"input={input_len:5d}: "
            f"median TTFT={fmt(row['median_ttft_ms'], 2)} ms, "
            f"p95 TTFT={fmt(row['p95_ttft_ms'], 2)} ms",
            flush=True,
        )

    return rows


def run_decode(args: argparse.Namespace, raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    print("\n=== DECODE PROFILE ===", flush=True)
    for concurrency in args.concurrencies:
        num_prompts = max(
            args.decode_min_prompts,
            concurrency * args.decode_waves,
        )

        result_path = raw_dir / (
            f"decode_input{args.decode_input_len}_"
            f"output{args.decode_output_len}_c{concurrency}.json"
        )

        result = run_benchmark(
            args=args,
            input_len=args.decode_input_len,
            output_len=args.decode_output_len,
            concurrency=concurrency,
            num_prompts=num_prompts,
            num_warmups=args.decode_warmups,
            result_path=result_path,
        )

        row = make_row(
            phase="decode",
            args=args,
            input_len=args.decode_input_len,
            output_len=args.decode_output_len,
            concurrency=concurrency,
            num_prompts=num_prompts,
            raw_path=result_path,
            result=result,
        )
        rows.append(row)

        print(
            f"concurrency={concurrency:2d}: "
            f"median TPOT={fmt(row['median_tpot_ms'], 2)} ms/token, "
            f"p95 TPOT={fmt(row['p95_tpot_ms'], 2)} ms/token, "
            f"median TTFT={fmt(row['median_ttft_ms'], 2)} ms",
            flush=True,
        )

    return rows


def main() -> int:
    args = parse_args()

    if any(x <= 0 for x in args.prefill_lengths):
        print("error: all prefill lengths must be positive", file=sys.stderr)
        return 2
    if any(x <= 0 for x in args.concurrencies):
        print("error: all concurrencies must be positive", file=sys.stderr)
        return 2
    if args.decode_output_len < 2:
        print(
            "error: --decode-output-len must be at least 2 to compute TPOT",
            file=sys.stderr,
        )
        return 2

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        check_health(args.host, args.port, args.timeout_s)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"vLLM health check passed at "
        f"http://{args.host}:{args.port}/health",
        flush=True,
    )

    rows: list[dict[str, Any]] = []

    try:
        if args.mode in ("all", "prefill"):
            rows.extend(run_prefill(args, raw_dir))

        if args.mode in ("all", "decode"):
            rows.extend(run_decode(args, raw_dir))

    except RuntimeError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        # Preserve already completed rows.
        if rows:
            write_csv(output_dir / "model_profile_summary.csv", rows)
        return 1

    summary_path = output_dir / "model_profile_summary.csv"
    write_csv(summary_path, rows)

    print(f"\nSummary written to {summary_path}")
    print_profile_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())