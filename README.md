# OPSERVE Artifact

This repository contains the analysis and simulation code for **OPSERVE: Opportunistic LLM Inference over Fragmented GPU Capacity in HPC Systems**.

The artifact has three main components:

- `polaris_2025_analysis/` reconstructs unallocated operational capacity from the 2025 Polaris workload and machine-status traces and analyzes per-node fragment lifetimes.
- `opserve-simulation-exps/` contains the trace-driven OPSERVE simulator, model profiles, request traces, and the one-hour worker-availability trace used by the evaluation.
- `scripts/` contains profiling and plotting utilities.

## Environment

The CPU-only analysis and simulation workflows were developed with Python 3.10.14 and the package versions listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A GPU is not required to reproduce the trace analysis or simulator results. Regenerating the model profiles requires a compatible NVIDIA GPU and vLLM 0.8.5. The paper profiles were collected on one NVIDIA A100-40GB GPU with tensor parallelism 1.

## Trace-driven evaluation

From the repository root:

```bash
cd opserve-simulation-exps
python run_evaluation.py \
  --models llama31_8b,qwen3_8b,qwen3_14b \
  --workloads all \
  --policies all \
  --output-dir logs/evaluation_matrix
```

The default adaptive controller is `latency_share`, the controller used for the paper evaluation. The runner writes:

- `summary.csv` with aggregate latency and throughput metrics,
- `workloads.csv` with request-trace statistics,
- `metadata.json` with the evaluation parameters,
- per-policy role-event, time-series, and request-level CSV files unless disabled with the save options.

Conversation and Tool&Agent report throughput over the configured one-hour horizon. Synthetic is a shorter source trace and reports throughput over its native final-arrival horizon. The same workload-specific reporting horizon is used for every policy.

### KV-transfer model

The simulator includes the published Mooncake cross-node RDMA KV-transfer measurements used in the paper parameterization:

| Prompt tokens | Transfer time |
| ---: | ---: |
| 2,048 | 2.51 ms |
| 4,096 | 4.75 ms |
| 8,192 | 8.84 ms |
| 16,384 | 16.43 ms |
| 32,768 | 31.65 ms |

Source: Mooncake, *vLLM P/D Disaggregation Performance*, https://kvcache-ai.github.io/Mooncake/performance/vllm/vllm-v1-pd-performance.html

Intermediate prompt lengths are linearly interpolated. Completed prefill KV is assumed to remain available in the shared tier after worker reclamation, and the same transfer profile is charged before decode resumes on another worker. The optional `--kv-handoff` and `--recovery-delay` arguments are additive sensitivity offsets and default to zero.

The current simulator uses `--prompt-threshold` to determine whether an inter-worker KV-transfer delay is charged. Requests below the threshold incur no transfer delay.

## Generate the evaluation figures

After running the evaluation matrix:

```bash
cd ..
python scripts/plot_throughput.py \
  --summary-csv opserve-simulation-exps/logs/evaluation_matrix/summary.csv \
  --output-prefix throughput_three_workloads

python scripts/plot_worker_availability_trace.py \
  --events opserve-simulation-exps/event-traces/one-hour-trace.jsonl \
  --output-prefix resource_trace_workers
```

Each command writes both PDF and PNG output. The PDF versions are intended for the paper.

## Polaris capacity analysis

The original 2025 Polaris traces are not bundled here. Download the workload and machine-status data from the ALCF public-data site:

https://reports.alcf.anl.gov/data/polaris.html

Then run the aggregate-capacity reconstruction:

```bash
python polaris_2025_analysis/analyze_traces.py \
  --jobs /path/to/ANL-ALCF-DJC-POLARIS_20250101_20251231.csv \
  --status /path/to/ANL-ALCF-MACHINESTATUS-POLARIS_20250101_20251231.csv \
  --output-dir polaris_2025_analysis/results
```

Generate the monthly-capacity and capacity-change plots with:

```bash
python polaris_2025_analysis/plot_fragmentation.py
```

For the conservative per-node fragment-lifetime analysis:

```bash
python polaris_2025_analysis/analyze_polaris_fragment_lifetimes.py \
  --jobs /path/to/ANL-ALCF-DJC-POLARIS_20250101_20251231.csv \
  --status /path/to/ANL-ALCF-MACHINESTATUS-POLARIS_20250101_20251231.csv \
  --output-dir polaris_fragment_lifetimes
```

Use `--save-details` only if the large per-node intermediate CSV files are needed.

## Regenerating model profiles

Start a vLLM server on the profiling GPU, for example:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B \
  --port 8001 \
  --tensor-parallel-size 1
```

Then run:

```bash
python scripts/measure_model_profile.py \
  --model meta-llama/Llama-3.1-8B \
  --port 8001 \
  --output-dir results/Llama-3.1-8B
```

The paper reports the median of three profiling runs for each prefill/decode configuration.

## Included request traces

The Conversation, Synthetic, and Tool&Agent request traces originate from the Mooncake FAST'25 release. They contain request timing and token-length metadata, not raw prompt or response text. See `THIRD_PARTY_SOURCES.md` for provenance.

## Release and privacy notes

This package contains no known credentials, private keys, API tokens, author email addresses, user home-directory paths, or raw conversation content. The archival ZIP is created without platform-specific ZIP extra fields.

Before a public DOI release, add the project or institution-approved software license. This refactor intentionally does not choose a license on behalf of the authors.

## Citation

See `CITATION.cff` for machine-readable citation metadata. Add the final artifact DOI to that file after it is minted.
