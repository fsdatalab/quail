# Untied lm_head moves to CPU memory; freed bytes go to the KV arena

## What changed

The engine answers filters and joins by scoring the final hidden
state against only the TRUE/FALSE rows of the output head (about a
dozen rows). That readout already ran on a sliced head in every
forward-pass path; the full 151,936-row head weight still sat on the
GPU with no reader. Now:

- `load_model` moves an untied head weight to CPU memory right after
  load (`move_untied_head_to_host` in `quail/executor/model.py`) and
  frees the GPU copy. Every boot path (the Modal worker, the
  multi-GPU child workers, the GPU cells, calibration, the
  ablations) goes through `load_model`, so all of them get it.
- `Answerer` and `_PayloadAnswerer` slice the TRUE/FALSE rows on
  whatever device the weight lives on, and keep only the slice on
  the GPU. Payloads with any TRUE/FALSE id spelling keep working:
  the full head stays available in CPU memory.
- `ModelSpec` gains `vocab` and `tied_head`, and a `head_mem_bytes`
  property (0 when tied). `budgets.arena_tokens` and
  `budgets.chunk_memory_bound` now subtract only resident weights
  (`W_resident = W_mem - head_mem_bytes`). `tensor_parallel` keeps
  the as-loaded footprint, because the head is on the GPU until the
  load finishes.

Qwen3 4B ties its head to the input embedding, so nothing moves and
no 4B number changes. Qwen3 32B has a separate bf16 head:
151,936 x 5,120 x 2 bytes = 1.556 GB freed.

## Why

The decision and the margin are exact on the sliced head: TRUE and
FALSE logits shift by the same softmax normalizer, so dropping the
other vocabulary rows changes neither. Keeping the full matrix on
the GPU therefore bought nothing, and on the 32B its bytes are worth
5,935 tokens of KV residency, which the join planner's KV retention
search can spend on kept documents.

## Before/after numbers

Computed from the spec constants (`tests/test_specs_budgets.py`
checks the arithmetic):

- Qwen3 32B on one H100: resident weights 34.37 GB before, 32.81 GB
  after. Admission budget 97,221 tokens before, 103,156 after
  (+5,935 tokens, +6.1%).
- Qwen3 4B: unchanged (tied head; the budget tests pin the same
  numbers as before).
- Chunk budgets unchanged for both models: the 32B chunk is capped
  by the int32 kernel index, not by memory.

The confirming cell is `tests/gpu/head_residency.py`. Prediction,
stated before the run: the 32B head lands on the CPU with allocated
memory within 0.3 GB of the 32.81 GB resident figure, the enlarged
arena allocates next to the weights, the 4B keeps its tied head on
the GPU, and 48 planted-flag filter answers per attention path
(unified, merge_quant, unpaged) are all correct on both models. The
cell writes its records to
`/results/ablations/head_residency_qwen3-4b-fp8.json` and
`/results/ablations/head_residency_qwen3-32b-fp8.json` on the
`quail-results` volume.
