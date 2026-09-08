# quail-bench

quail-bench is a benchmark of AI filter and AI join queries over document sets. Every predicate is one
exact TRUE or FALSE question, answered by a language model about one document or one document pair.
Reference labels are saved, so accuracy is scored exactly rather than judged.

## Document sets

Counts are at scale factor 0.1, the default. One seed (`DATA_SEED`) and one pinned revision per source (`SOURCE_REVISIONS`), both in `quail_bench/data.py`, mean a scale factor names one exact corpus.

| Set | Source | Rows at sf=0.1 |
| --- | --- | --- |
| `reviews` | `stanfordnlp/imdb` | 5,000 |
| `aspects` | fixed movie-aspect list in `data.py` | 12 |
| `reports` | `BioDEX/BioDEX-Reactions` | 500 |
| `terms` | reaction terms of the sampled BioDEX reports | 1,127 |
| `claims` | `fever/fever` | 500 |
| `evidence` | Wikipedia pages cited by the sampled claims | 287 |
| `citation_contexts` | `rmahari/LePaRD` | 500 |
| `citation_passages` | `rmahari/LePaRD` | 433 |
| `agent_traces` | `TIGER-Lab/SWE-Next-SFT-Trajectories` | 1,772 |

## Queries

`QUERIES` in `quail_bench/queries.py` holds 32 queries in five families. A query has a base document
set, filters on each set, joins that each add one more set, and a projection. Filters and joins apply
in the order written. Two more queries, PRIV-1 and PRIV-2, need `register_privacy_sets()` first.

| Family | Ids | Shapes |
| --- | --- | --- |
| IMDB | IMDB-1 to IMDB-10 | filter only, single join, filter chains into a join, star join, three-join chains |
| BIO | BIO-1 to BIO-3 | filter only, single join, filter into a join, on long medical reports |
| FEV | FEV-1 to FEV-9 | filter only, single join, filter chains into a join, two-sided filters, star join, three-join chains |
| LEP | LEP-1 to LEP-8 | filter only, single join, filter chains up to five deep into a join, two-sided filters |
| AGENT | AGENT-1, AGENT-2 | filter only |

## Metrics

- Query time: seconds to run the query, excluding model startup.
- Documents per second, filter-only query: input document rows / query seconds.
- Document pairs per second, join query: evaluated pairs across all join stages / query seconds.
- `$/query`: query hours x GPU count x the H100 hourly price.
- Answer accuracy: agreement with the labels on the predicate answers the engine evaluated.
- Final rows: precision, recall, and F1 against the rows derived from the labels.

## Reference labels

Labels come from Qwen3 32B fp8 answering the exact prompt text in `quail_bench/rendering.py`, at
temperature 0 with the answer restricted to TRUE or FALSE. One label covers one predicate and one
document or document pair. A complete label set for every predicate over one corpus is a collection,
saved on the `quail-results` Modal volume. FEVER and LePaRD use source labels where the dataset gives
the exact answer. `evaluate()` in `quail_bench/scoring.py` scores a run against a collection.

## Running it

This repository defines the benchmark and runs no engine. An engine's runner turns each `QuerySpec`
into that engine's own query and hands the answers back as a `RunOutput` to score. Install it, then
run the benchmark from the Quail repository, `fsdatalab/quail-exploration`:

```bash
uv add "quail-bench @ git+https://github.com/fsdatalab/quail-bench.git"
uv run modal run --detach -m quail.bench.quailb_parallel \
  --sf 0.1 --model qwen3-4b-fp8 \
  --prediction "State the expected runtime and accuracy."
```

Add `--query IMDB-4` to run one query. Modal starts one H100 container per query family, running Quail, stock vLLM, and pipelined vLLM in it; SGLang runs in its own container.

## Adding a query or predicate

1. Add the query to `QUERIES` in `quail_bench/queries.py`. A query that reuses existing predicates needs no new labels, so stop here.
2. Add the new prompt constant to `quail_bench/prompts.py`.
3. Add one `PredicateSpec` to `PREDICATES` in `quail_bench/judge_pass.py`: a stable key, the prompt, the input table and column, and the left and right roles.
4. Use a source label only when the dataset gives the exact answer the prompt asks for. Otherwise keep the default Qwen3 32B fp8 source.
5. Update the predicate count test; add a prompt-rendering test if the input shape is new.
6. Run the labeling pass: `uv run modal run -m quail_bench.judge_pass`

A changed prompt or input role makes a new predicate version and needs a new label set. The labeling run resumes, skipping finished parts, then writes a new collection and makes it active.

## Layout

| Module | What it holds |
| --- | --- |
| `quail_bench/data.py` | document sets, pinned sources, sampling, corpus identity |
| `quail_bench/prompts.py` | the filter and join prompt templates |
| `quail_bench/queries.py` | the 32 queries as data, with selectivity estimates |
| `quail_bench/rendering.py` | the exact prompt text a predicate asks |
| `quail_bench/labels.py` | saved label sets and collections on the volume |
| `quail_bench/judge_pass.py` | the 21 predicates and the labeling pass |
| `quail_bench/scoring.py` | `RunOutput` and the scoring of one run |
