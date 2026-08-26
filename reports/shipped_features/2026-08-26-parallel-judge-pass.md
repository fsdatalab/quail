# Judge pass runs one container per workload

## What changed

`quail/bench/judge_pass.py` used to build the corpus and judge all 19
QUAIL-B predicates in a single GPU container. It now runs five Modal
functions:

- `prepare_corpus` builds the corpus once on a CPU container and writes
  it to the volume, so every judge reads identical rows.
- `judge_workload` runs four times in parallel, once per workload
  (IMDB, BioDEX, FEVER, LePaRD), each on its own H100.
- `finalize_collection` assembles the four partial results, checks that
  every predicate reported a label set, and points
  `corpora/<corpus_id>/active_collection.json` at the new collection.

The split is by workload rather than by predicate because a container
pays about 280 seconds to load the 32B checkpoint and then amortizes it
over everything it judges.

Two hardcoded tables went away with it. `FILTER_ROWS_PER_CALL` and
`JOIN_ANCHORS_PER_CALL` stated ten batch sizes with no rule behind them.
They are now derived: `rows_per_call(prompts_per_row)` divides a single
`PROMPTS_PER_CALL` target by however many prompts each left row
produces, which is the number of predicates in a filter group or the
number of partner rows in a join. One call is one Parquet part, so this
sets the resume granularity, not the GPU batch size — vLLM re-chunks
whatever it is handed against `max_num_batched_tokens` and
`max_num_seqs`.

## Why

Regenerating ground truth after a template change is a blocking step:
nothing downstream can be measured until it finishes. It was 43 minutes
of sitting still.

## Numbers

Measured at scale factor 0.1 on 2026-08-26, 245,557 labels:

| | before | after |
|---|---|---|
| wall time | 43.0 min (reconstructed) | 33.8 min |
| GPU-seconds | 2,580 | 3,848 |
| answer differences on rerun | 0 | 0 of 288 |

1.27x faster for 49% more GPU time. The ceiling for this split is 1.56x,
because BioDEX alone is 60% of the judging work.

Labels are unchanged by the split itself. Decoding is greedy and the
prompts are identical; only their grouping into calls differs.

Full report: `reports/2026-08-26-parallel-judge-pass.md`.
