MalleServe model profile benchmark
===================================

Overview
--------

This benchmark collects the model-level measurements used to parameterize the
MalleServe simulator.

It measures two serving characteristics from an already-running vLLM server:

1. Prefill / TTFT profile
   Measures isolated time to first token (TTFT) across several prompt lengths.

2. Decode profile
   Measures time per output token (TPOT) across several decode concurrency
   levels.

The script also records mean, median, and p95 values in a CSV file and keeps
the raw JSON output produced by `vllm bench serve`.

The default profile is:

    Prefill prompt lengths:
        512, 1024, 2048, 4096, 8192, 16384, 32767 tokens

    Prefill output length:
        1 token

    Prefill concurrency:
        1 request

    Decode prompt length:
        8192 tokens

    Decode output length:
        256 tokens

    Decode concurrency:
        1, 2, 4, 8, 16 sequences


Requirements
------------

The benchmark assumes:

- vLLM is installed in the active Python environment.
- `vllm bench serve` is available.
- The model server is already running.
- The benchmark client is run on the same compute node as the vLLM server.
- The model uses the same GPU configuration intended for the MalleServe
  evaluation.

The script uses Python `requests` for its initial health check.

On Polaris, activate the environment used for the inference experiments before
running the benchmark. For example:

    conda activate /lus/grand/projects/BFTrainer/pwatters/envs/pw_env_lmcache_cu126


Starting vLLM
-------------

Start one model on one GPU.

Example for Llama-3.1-8B:

    CUDA_VISIBLE_DEVICES=0 \
    vllm serve meta-llama/Llama-3.1-8B \
        --port 8001 \
        --tensor-parallel-size 1

Wait until the server is ready.

From another shell on the same compute node:

    curl -i --max-time 3 http://127.0.0.1:8001/health or curl --noproxy '*' -i --max-time 3 http://127.0.0.1:8001/health

A ready server should return:

    HTTP/1.1 200 OK

If the model is gated on Hugging Face, make sure the shell running vLLM has
the required authentication environment loaded before starting the server.


Running the full model profile
------------------------------

Run both prefill/TTFT and decode profiling with:

    python3 measure-model-profile.py \
        --model meta-llama/Llama-3.1-8B \
        --port 8001 \
        --output-dir results/Llama-3.1-8B

The script first checks:

    http://127.0.0.1:8001/health

and then runs the prefill and decode experiments.


Running only the prefill / TTFT profile
---------------------------------------

    python3 measure-model-profile.py \
        --model meta-llama/Llama-3.1-8B \
        --port 8001 \
        --output-dir results/Llama-3.1-8B \
        --mode prefill

The default prompt lengths are:

    512
    1024
    2048
    4096
    8192
    16384
    32767

Each run:

- uses one request at a time;
- generates one output token;
- records TTFT;
- reports mean, median, and p95 TTFT.

The one-token output is intentional: the measurement is intended to represent
the isolated request cost through production of the first token rather than a
full generation request.

To use a different set of prompt lengths:

    python3 measure-model-profile.py \
        --model meta-llama/Llama-3.1-8B \
        --port 8001 \
        --output-dir results/Llama-3.1-8B \
        --mode prefill \
        --prefill-lengths 2048 8192 32767


Running only the decode profile
-------------------------------

    python3 measure-model-profile.py \
        --model meta-llama/Llama-3.1-8B \
        --port 8001 \
        --output-dir results/Llama-3.1-8B \
        --mode decode

By default, decode is measured using:

    input length:   8192 tokens
    output length:   256 tokens
    concurrency:       1, 2, 4, 8, 16

The script records:

- mean TPOT;
- median TPOT;
- p95 TPOT;
- mean TTFT;
- median TTFT;
- p95 TTFT;
- end-to-end latency statistics.

For the MalleServe decode profile, use median TPOT at each concurrency level.

To change the decode prompt or output length:

    python3 measure-model-profile.py \
        --model meta-llama/Llama-3.1-8B \
        --port 8001 \
        --output-dir results/Llama-3.1-8B \
        --mode decode \
        --decode-input-len 8192 \
        --decode-output-len 256

To change the measured concurrency levels:

    python3 measure-model-profile.py \
        --model meta-llama/Llama-3.1-8B \
        --port 8001 \
        --output-dir results/Llama-3.1-8B \
        --mode decode \
        --concurrencies 1 2 4 8 16


Output files
------------

For:

    --output-dir results/Llama-3.1-8B

the output structure is:

    results/Llama-3.1-8B/
        model_profile_summary.csv
        raw/
            prefill_input512_c1.json
            prefill_input1024_c1.json
            prefill_input2048_c1.json
            prefill_input4096_c1.json
            prefill_input8192_c1.json
            prefill_input16384_c1.json
            prefill_input32767_c1.json

            decode_input8192_output256_c1.json
            decode_input8192_output256_c2.json
            decode_input8192_output256_c4.json
            decode_input8192_output256_c8.json
            decode_input8192_output256_c16.json


Summary CSV
-----------

`model_profile_summary.csv` contains:

    phase
    model
    input_len
    output_len
    concurrency
    num_prompts
    completed
    mean_ttft_ms
    median_ttft_ms
    p95_ttft_ms
    mean_tpot_ms
    median_tpot_ms
    p95_tpot_ms
    mean_e2el_ms
    median_e2el_ms
    p95_e2el_ms
    raw_json

The raw JSON files are retained so that the measurements can be inspected or
reprocessed later.


Table 2 values
--------------

At the end of a successful run, the script prints a small Table 2 helper.

For prefill, it reports median TTFT in seconds at approximately:

    2K / 8K / 32K prompt tokens

For decode, it reports median TPOT in milliseconds per token at:

    1 / 4 / 16 concurrent sequences

These values correspond to a table layout such as:

    Model
    Params. (B)
    Startup (s)
    Prefill latency (s), 2K / 8K / 32K
    Decode TPOT (ms/token), 1 / 4 / 16 seq.

The simulator can still use the complete measured profiles:

    Prefill:
        512, 1024, 2048, 4096, 8192, 16384, 32767 tokens

    Decode:
        1, 2, 4, 8, 16 concurrent sequences


Recommended paper terminology
-----------------------------

The prefill benchmark is an isolated TTFT measurement rather than a pure
GPU-kernel timing measurement. A suitable description is:

    We profile isolated time to first token at prompt lengths from 512 to
    32,767 tokens using one request at a time. We profile decode using
    256-token generations at concurrency levels of 1, 2, 4, 8, and 16 and
    report time per output token (TPOT).

Use the same model configuration, GPU type, tensor-parallel degree, and vLLM
configuration for every model.


Proxy handling on Polaris
-------------------------

Polaris may define HTTP or HTTPS proxy environment variables.

A normal Python `requests` call to localhost can therefore be sent through the
proxy and return an error even when vLLM is healthy.

The benchmark handles this in two ways:

1. The initial Python health check uses:

       session.trust_env = False

2. The environment passed to `vllm bench serve` includes localhost in:

       NO_PROXY
       no_proxy

This prevents localhost benchmark traffic from being routed through the
cluster proxy.


Useful options
--------------

Run everything:

    --mode all

Run only TTFT/prefill:

    --mode prefill

Run only decode:

    --mode decode

Reuse raw JSON files that already exist:

    --skip-existing

Change the prefill prompt lengths:

    --prefill-lengths 512 1024 2048 4096 8192 16384 32767

Change decode concurrency:

    --concurrencies 1 2 4 8 16

Change decode prompt length:

    --decode-input-len 8192

Change decode output length:

    --decode-output-len 256


Example for Qwen3-8B
--------------------

Start the server:

    CUDA_VISIBLE_DEVICES=0 \
    vllm serve Qwen/Qwen3-8B \
        --port 8001 \
        --tensor-parallel-size 1

Then run:

    python3 measure-model-profile.py \
        --model Qwen/Qwen3-8B \
        --port 8001 \
        --output-dir results/Qwen3-8B


Example for Qwen3-14B
---------------------

Start the server:

    CUDA_VISIBLE_DEVICES=0 \
    vllm serve Qwen/Qwen3-14B \
        --port 8001 \
        --tensor-parallel-size 1

Then run:

    python3 measure-model-profile.py \
        --model Qwen/Qwen3-14B \
        --port 8001 \
        --output-dir results/Qwen3-14B

If the model cannot sustain one of the requested decode concurrency levels on
an A100-40GB, record that limitation rather than silently substituting a value.
The usable decode concurrency is itself part of the model configuration.


Suggested workflow
------------------

For each model:

1. Start vLLM on one GPU.
2. Confirm `/health` returns HTTP 200.
3. Run `measure-model-profile.py`.
4. Keep the raw JSON files.
5. Inspect `model_profile_summary.csv`.
6. Record median TTFT at 2K, 8K, and 32K for Table 2.
7. Record median TPOT at concurrency 1, 4, and 16 for Table 2.
8. Use the complete TTFT and TPOT profiles in the simulator.
9. Stop the vLLM server before profiling the next model.

Do not mix measurements collected with different GPU types, tensor-parallel
degrees, model settings, or serving configurations in the same Table 2.