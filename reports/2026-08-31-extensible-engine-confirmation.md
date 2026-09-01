# Typed engine confirmation

## Setup

- The main comparison uses QuailB at scale factor 0.1 with Qwen3 4B fp8 on
  one H100 80GB HBM3.
- BIO-2 is a large join with no preceding filter. It checks the direct join
  path where Quail has its largest measured advantage over stock vLLM.
- AGENT-1 is a filter over cumulative agent traces. Shared prefixes make it
  Quail's current worst case against stock and pipelined vLLM.
- Query time and cost exclude model startup. Cost uses $3.9492 per H100 hour.
- A small 4B join used two H100s in one Modal container. Each GPU held one
  model copy, so the query did not split one model across two GPUs.
- A small filter used Qwen3 32B fp8 on one H100.
- Every run used the existing `quail-milestone1` Modal app.

The main measurements came from:

- `/results/benchmarks/quailb/families/20260831T144252Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/biodex.json`
- `/results/benchmarks/quailb/runs/qb_20260831T062218Z_1192cd76/20260831T062218Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families.json`

The new measurements came from:

- `/results/ablations/extensible_engine_confirmation_4b.json`
- `/results/ablations/extensible_engine_confirmation_4b_agent1.json`
- `/results/ablations/extensible_engine_confirmation_4b_2gpu.json`
- `/results/ablations/extensible_engine_confirmation_32b.json`
- `/results/ablations/extensible_engine_confirmation_4b_output_check.json`
- `/results/ablations/extensible_engine_confirmation_4b_timing_check.json`

The successful Modal function calls were:

- BIO-2 performance: `fc-01M1DB65MSS76GTDT539JMYHGZ`
- AGENT-1 performance: `fc-01M1DF8SSYXS3WE2Q69CX1V94Y`
- 4B on two GPUs: `fc-01M1DB65QGRMK705B1R8MSJQB7`
- 32B on one GPU: `fc-01M1DB65VXR2VQPW0E4YKM1C30`
- Sorted BIO-2 output check: `fc-01M1DBQBMDW63H9BG8MFAPYYGW`
- BIO-2 timing check: `fc-01M1DCDDKPC67EA1EGJVABD13C`

## Prediction

- BIO-2 will evaluate 563,500 document pairs, report zero regret, and finish
  within 5% of the 129.64 second main result.
- AGENT-1 will evaluate 1,772 documents, process 17,389,113 fresh tokens,
  report zero KV regret, and finish within 10% of the 235.09 second main
  result.
- BIO-2 will return the same rows as main. AGENT-1 will return the same 344
  row count as main.
- The small 4B query on two GPUs and the small 32B query on one GPU will
  finish with tensor parallelism set to one.

## Result

Both main performance confirmations met the predictions.

| Query | Main time | Typed time | Change | Typed throughput | Typed cost | Evaluated work | Fresh tokens | KV regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BIO-2 | 129.64 s | 128.80 s | 0.6% faster | 4,375.0 pairs/s | $0.1413 | 563,500 pairs | 10,374,352 | 0 |
| AGENT-1 | 235.09 s | 238.13 s | 1.3% slower | 7.44 documents/s | $0.2612 | 1,772 documents | 17,389,113 | 0 |

The typed engine also matched the main row counts and KV measurements.

| Query | Rows | Join anchor | KV hits | KV misses | Retained prefixes | Evicted prefixes |
|---|---:|---|---:|---:|---:|---:|
| BIO-2 | 116,295 | `r` | 0 | 500 | 0 | 0 |
| AGENT-1 | 344 | none | 0 | 0 | 0 | 0 |

The earlier BIO-2 result did not contain a final output hash. The output
check rebuilt the old final Arrow result from its saved answer table, sorted
the old and new rows by every result column, and hashed both tables. Both
BIO-2 hashes were
`732e7cbe4bd13203e870be3cce02dba15d7274195c8e047a121dd03448713773`.

The AGENT-1 confirmation returned the same row count and processed the same
fresh tokens as main. Its sorted output hash was
`811902a8cfa300d23667bbd32820f8f80a7aa20fdead7019cd0c67a2ba7e45a1`.
The main AGENT-1 run did not save a final Arrow table, so the experiment did
not independently compare the exact document IDs.

The checks for multiple GPUs and both model sizes also passed.

| Query | Model | GPUs | Tensor parallel | Time | Throughput | Cost | Work | Fresh tokens | KV regret |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small join | Qwen3 4B fp8 | 2 | 1 | 2.77 s | 11.55 pairs/s | $0.0061 | 32 pairs | 752 | 0 |
| Small filter | Qwen3 32B fp8 | 1 | 1 | 0.04 s | 200.0 documents/s | $0.00004 | 8 documents | 240 | 0 |

The query on two GPUs created two model workers in one Modal container. Each
worker owned one 4B model copy on one H100. The 32B query used the same typed
plan and runner interfaces as the 4B queries.

## Runtime variance

Figure: plots/extensible_engine_confirmation.png

The BIO-2 output check took 397.13 seconds even though it processed the same
pairs and fresh tokens as the primary run. Sorting occurred after the engine
query timer, so sorting did not cause the slow query time.

A third BIO-2 run on an NVIDIA H100 80GB HBM3 took 129.38 seconds. The third
run was 0.2% faster than main and met the declared 5% range. The cause of the
slow output check is unknown. Two current BIO-2 measurements agree with main,
while one does not. The figure includes all three measurements.

## What the numbers mean

- The typed plan, generic runner, and Quail backend preserve the measured
  work and row counts for BIO-2 and AGENT-1.
- The primary BIO-2 and AGENT-1 times stayed within their declared ranges.
- The 4B and 32B models use the same interfaces.
- Multiple GPUs run as separate model workers in one container. They do not
  use tensor parallel weight splitting.
- The slow BIO-2 repeat shows that one Modal timing cannot rule out platform
  variance. The slow run did not coincide with a change in Quail's planned
  or measured model work.

## Rebuild

Run `reports/make_extensible_engine_confirmation_plots.py` from the repository
root. Its docstring contains every `modal volume get` command needed to
rebuild the figure.
