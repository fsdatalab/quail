# Shared join prompts and vLLM baselines

## What changed

Quail, stock vLLM, and naive vLLM now build the exact same join prompt token
IDs. The prompt puts the anchor document and complete question in the shared
prefix. Each pair adds the partner document and answer cue.

The planner now counts the complete question once per anchor. It counts the
partner label, partner document, and answer cue once per pair. Tests compare
the exact prompts for either anchor choice and for a join with three inputs.

Both vLLM baselines submit the complete join in one batch. They use
`max_num_seqs=4096`, `max_num_batched_tokens=25305`, prefix caching, and the
exact Modal request `gpu="H100!"`.

## Why

The old prompt repeated the complete question once for every document pair.
The three systems also did not use the same prompt order, so the old timing
comparison was not controlled.

## Measured change

At sf=0.1, Quail processed 2,635,599 fresh tokens for BIO-2, which was 76
percent fewer than the old prompt. Quail processed 2,419,233 fresh tokens for
IMDB-2, which was 59 percent fewer.

Quail was faster than both vLLM baselines on all six join comparisons. At 4B,
BIO-2 took 31.12 seconds in Quail, compared with 187.17 seconds in naive vLLM
and 261.38 seconds in stock vLLM. At 32B, BIO-2 took 181.05 seconds in Quail,
compared with 302.96 seconds in naive vLLM and 265.80 seconds in stock vLLM.

The full setup and all six Modal volume paths are in
`reports/2026-08-25-vllm-opbench-baseline.md`.
