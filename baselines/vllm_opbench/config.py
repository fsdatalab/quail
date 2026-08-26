"""vLLM-opbench tunables. No GPU/Modal side effects on import - safe to
read from both the orchestrator and the worker.

Ported from /Users/adhariya/SQPE (a separate benchmarking project),
scoped down to Filter and Join only (no Classify/Extract - outside
this project's stated scope) and repointed at quail's own document
sets and predicates (quail/bench/quailb.py) instead of SQPE's
IMDb Spoiler dataset.

Two deliberate departures from SQPE, both requested when this port was
scoped:
  - Checkpoint: the base Qwen3 checkpoint, quantized to fp8 at runtime
    by vLLM (SQPE's original approach) - not quail's own pre-quantized
    -FP8 checkpoints.
  - Answer decoding: constrained TRUE/FALSE token ids (matching quail's
    own engine and the existing baselines/stock.py), not SQPE's
    original free-text "true"/"false" prefix parsing.
"""

import os

# GPU benchmark cells attach to the existing milestone app so they
# reuse its caches and warm state.
APP_NAME = "quail-milestone1"

# Base (non-FP8) checkpoints - fp8 quantization happens at vLLM load
# time via `quantization="fp8"`, not by loading a pre-quantized
# checkpoint. This is SQPE's original approach, kept deliberately: it
# isolates "generic vLLM usage" from quail's own choice to load
# pre-quantized weights.
MODEL_NAMES = {
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "qwen3-4b-stock": "Qwen/Qwen3-4B-FP8",
    "qwen3-32b-stock": "Qwen/Qwen3-32B-FP8",
}

# GPU x quantization cells to sweep. "H100!" pins the exact SKU so
# Modal's scheduler can't silently substitute a different card.
GRID = [("H100!", "fp8")]

# Which quail document set + predicate chain each query exercises.
# table/id_col/text_col match quail/bench/quailb.py's parquet
# schema exactly, so vLLM-opbench reads the same on-disk data quail's
# own engine does - same documents, same predicates, only the serving
# strategy differs. Absolute path: run.py's orchestrator runs on Modal
# now (not locally), with the "quail-results" volume mounted at
# /results - same path quail's own GPU cells use, so this baseline
# reuses whatever sf-scaled data quail's own runs already built.
DATA_DIR = "/results/quailb_data"
SF = 0.1

TENSOR_PARALLEL_SIZE = 1
MAX_NUM_BATCHED_TOKENS = 25_305     # matches the bf16-KV stock knobs
#                                     already committed in
#                                     tests/gpu/milestone1.py
# The tested stock join needs 4,096 so short suffix requests can fill
# max_num_batched_tokens. The filter submits only 200 requests here, so
# this upper bound does not change its effective admission.
MAX_NUM_SEQS_BY_GPU = {"H100!": 4096, "H100": 4096}
GPU_MEMORY_UTILIZATION = 0.92       # matches quail's own POOL_FRACTION
#                                     (budgets.py) - the fraction of
#                                     device memory the baseline may
#                                     claim, same as quail's engine
PREFIX_CACHE_BLOCK_SIZE = 16        # quail's own PAGE_TOKENS
ENABLE_CHUNKED_PREFILL = True
LONG_PREFILL_TOKEN_THRESHOLD = 4096
SCHEDULER_RESERVE_FULL_ISL = False
DISABLE_LOG_STATS = False           # must be False for llm.get_metrics()
#                                     to return anything at all

FILTER_MAX_TOKENS = 1               # one constrained token: the
#                                     TRUE/FALSE decision itself

# ---------------------------------------------------------- profiling
PROFILE_GPU = bool(int(os.environ.get("VLLM_OPBENCH_PROFILE_GPU", "0")))
PROFILE_ALL_OPERATORS = False
PROFILE_OPERATOR = None             # restrict nsys capture to one
#                                     operator name, or None for the
#                                     first eligible call only
NSYS_CAPTURE_WARMUP_S = 10.0
NSYS_CAPTURE_WINDOW_S = 15.0
NSYS_TRACE_FLAGS = "cuda,nvtx,osrt,cudnn,cublas"
NSYS_OUTPUT_DIR = "/root/nsys-traces"
NSYS_VOLUME_NAME = "vllm-opbench-nsys-traces"
NSYS_REEXEC_SENTINEL = "VLLM_OPBENCH_UNDER_NSYS"

RUN_DATE_DIR = None                 # set by run.py at start, one
#                                     folder per orchestrator run
