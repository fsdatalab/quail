# Request backend confirmation

Date: 2026-09-02.

## Setup

This check ran LEP-1 at scale factor 0.1 with Qwen3 4B fp8. Each
backend used one H100 on Modal. The query evaluated 500 input
documents and generated one answer token per document.

The four configurations used the same logical query and the same
physical request interface:

- Quail used token-based admission and KV rewind.
- Stock vLLM submitted the filter as a stage-major request batch.
- Pipelined vLLM used per-document submission. It reused the vLLM
  model loaded by stock vLLM.
- Pipelined SGLang used per-document submission through SGLang.

Quail ran in its own Modal container. The two vLLM configurations ran
in a second container. SGLang ran in a third container with its own
Python package image. vLLM 0.26 captured one CUDA graph at size 8192.

## Prediction

Before the final run, the prediction was:

- All backends would complete through `QueryRequest` and return Arrow
  result relations.
- Quail and vLLM would no longer share GPU process state.
- Stock vLLM and pipelined vLLM would use one loaded vLLM model.
- The token IDs sent across the vLLM process boundary would be normal
  Python integers.

All four statements were confirmed.

The public compute provider was checked separately with three documents.
`Session` selected each backend from `EngineConfig.backend`, printed the Modal
function call ID, and returned a `QueryResult` backed by an Arrow table.
Stock vLLM and pipelined SGLang both returned document IDs `a` and `c`.

## Result

Figure: plots/request_backend_confirmation.png

| Backend | Query time (s) | Documents/s | $/query | Startup (s) | Output rows |
|---|---:|---:|---:|---:|---:|
| Quail | 1.27 | 393.7 | $0.00139 | 31.85 | 356 |
| Stock vLLM | 1.50 | 333.3 | $0.00165 | 81.45 | 377 |
| Pipelined vLLM | 1.40 | 357.1 | $0.00154 | 0.00, model reused | 377 |
| Pipelined SGLang | 1.68 | 297.6 | $0.00184 | 352.46 | 292 |

The primary time and cost exclude startup. The cost uses $3.9492 per
H100 hour. Throughput is 500 input documents divided by query time.

The final Quail and vLLM files are:

- `/results/benchmarks/quailb/runs/qb_20260902T065022Z_afca9ed1/20260902T065022Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families-quail.json`
- `/results/benchmarks/quailb/runs/qb_20260902T065022Z_3bd87741/20260902T065022Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families-stock_vllm.json`
- `/results/benchmarks/quailb/runs/qb_20260902T065022Z_6f846e04/20260902T065022Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families-pipelined_vllm.json`

The SGLang file is:

- `/results/benchmarks/quailb/runs/qb_20260902T064443Z_8fd34020/20260902T064443Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families-pipelined_sglang.json`

The public compute provider result summaries are:

- Stock vLLM, function call `fc-01M1GENBAJNPC8XR7MK3BFZEYG`:
  `/results/runs/run_1788332551993592379.json`
- Pipelined SGLang, function call `fc-01M1GETFC9MVBA4PDVM487R81Y`:
  `/results/runs/run_1788332923120856725.json`

## Failures found during the check

The first attempts found three interface bugs:

- Quail passed `False` where its executor expected an empty collection
  of retained document indexes.
- The benchmark tried to load vLLM after Quail in the same GPU process.
  Quail still held 57.9 GB, so vLLM could not allocate its KV.
- The memory-mapped token store returns NumPy integers. vLLM 0.26 does
  not serialize those integers across its worker process boundary.

The final code normalizes the empty retention value, runs Quail and
vLLM in separate containers, and converts request token IDs to Python
integers. CPU tests cover all three cases.

## What the numbers mean

- All four backends now use the new planning and execution interface.
- Stock vLLM and SGLang also work through the public
  `ModalComputeProvider.execute(QueryRequest)` path. The benchmark runner is
  not required to invoke them.
- Stock vLLM and pipelined vLLM returned the same 377 rows, processed
  the same 131,867 fresh tokens, and read the same 80 tokens from KV.
  LEP-1 has one filter stage, so it does not test a pipelining benefit.
- The four engines did not return identical rows. The ground truth was
  produced by Qwen3 32B, while this check used Qwen3 4B. The result is
  valid as an interface check, but it is not evidence that the engines
  have equal accuracy.
- The evaluator now reads filter and join answer relations from both
  Quail nodes and request backend nodes. A CPU test confirms that the
  same four predicate answers are scored for both physical plans.
