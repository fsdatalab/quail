# Judge identity hashes only answer-affecting settings; BIO-6 joins the full terms table again

## What changed

Three things, in `quail/bench/judge_pass.py` and `quail/bench/quailb.py`:

1. The ground-truth judge runs at `gpu_memory_utilization=0.85` and
   `max_num_seqs=4096`, instead of 0.92 and 2,648.
2. `JUDGE_SPEC`, the dictionary hashed into `JUDGE_ID` and from there into
   every `label_set_id`, no longer contains `max_num_batched_tokens` or
   `max_num_seqs`.
3. `severe_terms` is gone. BIO-6's second join and the `REACTION_SEVERE`
   predicate both read the full `terms` table again, under a second alias
   (`m2`), the same shape IMDB-8 and FEV-7 already use.

## Why

The labeling job crashed with a CUDA out-of-memory error on BIO-6's second
join. It asked to allocate 5.41 GiB with 5.38 GiB free on an 80 GiB H100.

The judge's vLLM settings were copied from the 4B engine runs
(`baselines/vllm_opbench/config.py`, `tests/gpu/milestone1.py`), but the
judge loads Qwen3 32B FP8. At `gpu_memory_utilization=0.92` the KV
reservation takes almost everything the weights leave, so a full
25,305-token prefill chunk had nowhere to put its activations. Lowering the
reservation to 0.85 frees about 5.5 GiB more on the same GPU, roughly twice
the size of the allocation that failed.

`max_num_seqs` went from 2,648 to 4,096 because the judge never reaches
either number: `rows_per_call` caps a single submission at 256 prompts for
filters and 614 for the reports-by-terms joins. The knob is not what was
constraining memory.

The earlier fix (see `reports/old/2026-08-26-bio6-severe-terms-fix.md`) cut
BIO-6's second join from 614 terms to 64. That changed the benchmark to fit
the labeling job, and it cut the second stage 10x, which is the part of the
query that pipelining and token-based admission exist to speed up. It has
been reverted.

The identity change is what made the memory fix cheap. `JUDGE_SPEC` feeds
`JUDGE_ID`, `JUDGE_ID` feeds `label_set_id`, and `label_set_id` is both the
directory name and a stored column on every label row. With the capacity
knobs inside that hash, changing one to fix an out-of-memory crash
invalidated all 23 predicates' labels. Those knobs cannot change the answer:
the judge decodes one token at `temperature=0.0` with `allowed_token_ids`
restricted to TRUE and FALSE. They only set how many prompts run at once and
how much memory the KV reservation takes. So they no longer sit in the hash,
and future memory tuning is free.

## Numbers

| | before | after |
|---|---|---|
| judge `gpu_memory_utilization` | 0.92 | 0.85 |
| judge `max_num_seqs` | 2,648 | 4,096 |
| BIO-6 second join | 200 x 64 | 200 x 614 |
| `REACTION_SEVERE` labels | 12,800 | 122,800 |
| labels in the collection | 324,201 | 434,201 |
| fields in `JUDGE_SPEC` | 12 | 10 |

The 434,201 figure is 394,138 Qwen3 32B judgments plus 40,063 source labels
(40,000 from LePaRD passage ids, 63 from FEVER annotations), across 23
predicates at scale factor 0.1.

## Not measured yet

No labeling run has been made since this change, so the prediction that 0.85
clears the out-of-memory crash is a calculation, not a result. The
`quail-results` volume currently holds no `ground_truth/` tree at all, so
the full collection has to be regenerated rather than migrated in place.
