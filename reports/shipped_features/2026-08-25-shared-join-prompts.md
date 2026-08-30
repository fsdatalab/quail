# Shared join prompts and stock vLLM

## What changed

Quail and stock vLLM now build the exact same join prompt token IDs. The
prompt puts the anchor document and complete question in the shared prefix.
Each pair adds the partner document and answer cue.

The planner now counts the complete question once per anchor. It counts the
partner label, partner document, and answer cue once per pair. Tests compare
the exact prompts for either anchor choice and for a join with three inputs.

Stock vLLM submits the complete join in one batch. It uses
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

Quail was faster than stock vLLM on all six join comparisons. At 4B, BIO-2
took 31.12 seconds in Quail and 250.00 seconds in stock vLLM. At 32B, BIO-2
took 181.05 seconds in Quail and 283.96 seconds in stock vLLM.

The full setup and all source volume paths are in the deleted report
`2026-08-26-stock-vllm-joins.md` (in git history).
