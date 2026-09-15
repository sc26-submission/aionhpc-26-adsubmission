# Third-Party Sources

## Mooncake request traces

The bundled Conversation, Synthetic, and Tool&Agent request traces are derived from the Mooncake FAST'25 trace release:

https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces

The bundled traces contain request timestamps and token lengths only. Review the upstream project's license and redistribution terms before public archival release.

## Mooncake KV-transfer measurements

The simulator encodes published cross-node RDMA KV-transfer measurements from:

Mooncake, *vLLM P/D Disaggregation Performance*  
https://kvcache-ai.github.io/Mooncake/performance/vllm/vllm-v1-pd-performance.html

The encoded transfer-time points are 2.51, 4.75, 8.84, 16.43, and 31.65 ms for 2K, 4K, 8K, 16K, and 32K-token prompts, respectively.

## Polaris traces

The raw Polaris workload and machine-status traces are not bundled. They are available from the Argonne Leadership Computing Facility public-data portal:

https://reports.alcf.anl.gov/data/polaris.html
