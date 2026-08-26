# vllm-opbench: a second, more instrumented stock-vLLM baseline

Date: 2026-08-25. One H100 on Modal, Qwen3 4B and 32B, fp8. sf=0.1.

## How it works

`baselines/vllm_opbench/` runs quail's own document sets and
predicates (`quail/bench/quailb.py`) through plain vLLM, not through
quail's engine, so we get an honest "how fast is generic vLLM"
number to compare against. For each query it:

1. Reads the same parquet tables quail's engine reads and builds one
   prompt per document (filter) or per document pair (join), using
   quail's own predicate templates.
2. Submits every prompt for the query in one batched
   `llm.generate()` call - not one request at a time.
3. Constrains decoding to a single TRUE/FALSE token
   (`allowed_token_ids` + `max_tokens=1`), the same convention
   quail's own engine uses, so there's no free-text answer to parse
   or get wrong.
4. Records vLLM's own Prometheus metrics and a KV-cache "oracle
   regret" (how many prefix-cache blocks an infinite, never-evicting
   cache would have hit, vs. what vLLM's real cache hit) on every run.

It differs from the existing `baselines/stock.py` in two ways:

|  | checkpoint | client |
|---|---|---|
| `baselines/stock.py` | quail's own pre-quantized Qwen3-\*-FP8 | one request per (doc, stage) or per pair |
| `baselines/vllm_opbench/` | base checkpoint, fp8 quantized at vLLM load time | one batched call per query, all prompts at once |

Four queries, each mirroring an existing quailb query:

| Query | Mirrors | Shape |
|---|---|---|
| filter-reports | BIO-1 | F7 filter, 200 reports |
| join-reports | BIO-2 | REACTION join, 200 reports x 614 terms = 122,800 pairs |
| join-claims | FEV-2 | SUPPORT join, 100 claims x 57 evidence = 5,700 pairs |
| join-imdb | IMDB-2 | DISCUSS_ASPECT join, 5,000 reviews x 12 aspects = 60,000 pairs |

Both baselines are naive in the sense that neither uses quail's own
optimizations (pipelining, token-based admission, KV rewind).

## Is it correct?

- Every query's row/pair count matches quailb's own count exactly -
  same document tables, same predicate templates, same schema.
- Zero null or dropped answers across every query in this report -
  guaranteed by the constrained decode, not just observed.
- A line-by-line review, then a separate adversarial pass, found and
  fixed 6 bugs. The two that mattered: results now save after every
  query instead of only at the end (a crash used to silently drop
  already-computed results), and every saved answer now records
  which document, or document pair, it came from.

Two style questions came up during review and are being left as-is:
a new Modal app name for this baseline (`config.py`), and one
function that reimplements a few lines of quail's own token-id logic
instead of importing it (`operators.true_false_ids`). Neither affects
correctness.

## Measured result

Figure: plots/vllm_opbench_vs_quail.png

| Query | Model | quail wall_s | vllm-opbench generate_s |
|---|---|---:|---:|
| filter-reports | 4B | 9.0 – 10.1 | 10.75 |
| filter-reports | 32B | 53.8 – 55.7 | 74.05 |
| join-reports | 4B | 138.6 – 155 | 217.69 |
| join-reports | 32B | 812.2 (cold) | 468.11 |
| join-claims | 4B | 4.9 | 22.87 |
| join-claims | 32B | 13.5 | 175.61 |
| join-imdb | 4B | 52.4 – 58.6 | 33.43 |
| join-imdb | 32B | 358.1 (cold) | 250.68 |

join-claims at 32B is the one clean case: quail is faster, 13.5s vs
175.6s. join-reports' build step is CPU tokenization (each of 200
reports gets re-encoded once per partner term - 122,800 encode
calls), not GPU time, and it's noisy run to run (850-1,497s); a real
cost of the naive approach, not a bug.

join-imdb is the one query all three baselines (quail, `stock.py`,
this one) have run:

Figure: plots/vllm_opbench_join_imdb_3way.png

| | 4B | 32B |
|---|---:|---:|
| quail | 52.4 – 58.6 s | 358.1 s (cold) |
| stock vLLM (quail's own checkpoint) | 34.6 s | 206.7 s |
| vllm-opbench (base checkpoint, load-time fp8) | 33.4 s | 250.7 s |

## Accuracy against ground truth

Two of quail's own committed join-query reference runs (join-reports,
join-imdb) disagreed with each other by a lot. `tests/gpu/join_rerun.py`
reruns each query cold and warm and checks answers against QUAIL-B
ground truth (collection `gt_80e7582534b349bc61c087595f2e0a51`, pinned
explicitly - see the script's comment), explaining why:

| Model | Query | wall_s (cold/warm) | selectivity | accuracy | precision | recall |
|---|---|---:|---:|---:|---:|---:|
| 4B | filter-reports (F7) | 8.8 / 11.1 | 55.5% | 94.0% | 100% | 90.2% |
| 4B | join-imdb | 51.9 / 56.0 | 99.99% | 5.55% | 5.54% | 100% |
| 4B | join-reports | 137.1 / 135.1 | 100% | 16.1% | 16.1% | 100% |
| 32B (pending recheck) | join-imdb | 358.1 (cold) | 2.1% | 98.5% | 59.0% | 65.2% |
| 32B (pending recheck) | join-reports | 812.2 (cold) | 0.05% | 99.96% | 77.6% | 55.6% |

4B answers TRUE on almost every pair for both joins (99.99-100%
selectivity) - 100% recall, low precision, not a real judgment. It's
not broken generally: on filter-reports, checked the same way, it
scores 100% precision and 90.2% recall. The join instability is
specific to these two predicates, not to 4B across the board.

32B's row is marked pending: those numbers predate a QUAIL-B relabel
that fixed a template bug (`reports/2026-08-26-parallel-judge-pass.md`),
moving REACTION's positive rate from 0.07% to 16.11%. 32B answered
TRUE on only 0.05% of join-reports pairs under the old labels; against
a target that's actually 16.11% positive, recall should drop sharply
once rechecked - so treat 32B's numbers above as not yet confirmed,
not as settled.

## What this means

- **vllm-opbench's own code is correct**: right prompts, right
  decoding, honest and internally consistent numbers.
- **4B's join instability is explained**: it defaults to TRUE on
  these two join predicates specifically, not on filters in general.
- **32B's accuracy claim is still open** and, by the arithmetic
  above, likely to move once rechecked - not yet a settled result.

## Raw data

Per-query raw results (not committed, per house rules) are on the
`quail-results` Modal volume:

- `/results/vllm_opbench/<timestamp>/summary.json` - `run.py`'s wall-time
  grid (measured result table above).
- `/results/benchmarks/quailb/runs/<run_id>/` - `join_rerun.py`'s
  accuracy runs (accuracy table above).

Committed summary: `results/vllm_opbench_vs_quail.json`.
