# QUAIL-B

QUAIL-B is a benchmark of 33 AI SQL queries over document tables.
AI SQL means SQL queries with predicates answered by a language model.

The current benchmark only supports two AI SQL operations:

- **AI FILTER:** ask a boolean question about one document and keep the
  documents answered `TRUE`.
- **AI JOIN:** ask a boolean question about two documents and keep the
  pairs answered `TRUE`.

QUAIL-B is not a SQL engine. It provides 33 engine-independent query
specifications, Arrow input tables, and reference answers. Your adapter
executes each specification with your engine and returns the result.
QUAIL-B then validates, scores, and saves the run.

For each query, the benchmark reports accuracy, query time, throughput,
GPU cost, and token work.

## Query plans

This diagram shows all 33 queries, grouped by dataset. Each small tree is
one query. Gray nodes are table scans, blue nodes are AI FILTER, and
orange nodes are AI JOIN. Percentages are the fixed planning selectivity
estimates from the 0.1-scale reference labels.

![All QUAIL-B query plans](figures/quailb_anatomy.png)

[Open the vector PDF.](figures/quailb_anatomy.pdf)

## Install

Python 3.12.

```sh
uv add "quail-b @ git+https://github.com/fsdatalab/quail-bench.git"
```

## What QUAIL-B measures

- **Accuracy:** agreement with saved predicate answers, plus precision and
  recall of the final rows.
- **Query performance:** query time, documents or pairs per second, and GPU
  cost when a price is supplied.
- **Token work:** how many input token positions the model computed, the
  minimum those requests required, and how many KV tokens were recomputed.

The three token counts are:

- `fresh_tokens`: all input token positions processed by model forward
  passes. Repeated computation is counted again. Your engine reports this.
- `minimum_tokens`: the fewest input token positions the same requests need
  if every shared prefix stays in KV. QUAIL-B computes this.
- `regret_tokens`: `fresh_tokens - minimum_tokens`. This is the reusable
  document or anchor KV that the engine computed again. QUAIL-B computes it.

To compute the last two counts, QUAIL-B needs the exact request prefixes.
Your adapter returns `fresh_tokens` and `prompt_pieces`. `prompt_pieces`
contains the tokenizer name and the token IDs placed around each document:

```python
prompt_pieces = {
    "tokenizer": "Qwen/Qwen3-4B-FP8",
    "preamble": [...],
    "filters": [
        {"alias": "r", "position": 0, "tail": [...]},
        {"alias": "r", "position": 1, "tail": [...]},
    ],
    "joins": [
        {
            "position": 0,
            "anchor": "r",
            "frame": [...],
            "label": [...],
            "tail": [...],
        },
    ],
}
```

All values represented by `[...]` are lists of token IDs. For a filter
request, the order is `preamble`, document, `tail`. For a join request,
the order is `preamble`, anchor document, `frame`, `label`, partner
document, `tail`. See `quail_b.minimum.validate_prompt_pieces`.

If an engine does not provide these values, accuracy and query performance
still score, but its token counts are unavailable.

## What your adapter receives

QUAIL-B calls your function as `run_query(query, tables)` once per query:

- `query` is a `QuerySpec`. It names the input tables, text columns,
  AI filter prompts, AI join prompts, and selected output columns.
- `tables` is a dictionary of Arrow tables. Every table has an `id`
  column and the text column named by the query.

For example, IMDB-4 means:

```sql
-- Pseudocode. Each engine has its own AI SQL syntax.
SELECT r.id, a.id
FROM reviews AS r
AI JOIN aspects AS a ON J1(r.body, a.aspect)
WHERE AI_FILTER(F1, r.body)
  AND AI_FILTER(F4, r.body);
```

The actual `QuerySpec` carries those operations without requiring one
engine's SQL syntax:

```python
query.id                 # "IMDB-4"
query.aliases[0].alias   # "r"
query.aliases[0].table   # "reviews"
query.aliases[0].column  # "body"
query.aliases[0].filters # (F1 prompt, F4 prompt)
query.aliases[1].table   # "aspects"
query.aliases[1].column  # "aspect"
query.joins[0].aliases   # ("r", "a")
query.joins[0].template  # J1 prompt
query.select             # ("r.id", "a.id")
```

For this query, `tables["reviews"]` has `id` and `body` columns.
`tables["aspects"]` has `id` and `aspect` columns.

## What your adapter returns

Your adapter translates the `QuerySpec` into your engine's AI SQL,
executes it, and returns a `RunOutput`:

```python
import pyarrow as pa
import quail_b


def run_query(query, tables):
    result = execute_with_your_engine(query, tables)

    return quail_b.RunOutput(
        filter_answers={
            ("r", 0): pa.table({
                "r": result.f1_ids,
                "answer": result.f1_answers,
            }),
            ("r", 1): pa.table({
                "r": result.f4_ids,
                "answer": result.f4_answers,
            }),
        },
        join_answers={
            0: pa.table({
                "r": result.j1_review_ids,
                "a": result.j1_aspect_ids,
                "answer": result.j1_answers,
            }),
        },
        rows=pa.table({
            "r": result.output_review_ids,
            "a": result.output_aspect_ids,
        }),
        runtime_s=result.query_seconds,
        measurements={"fresh_tokens": result.fresh_tokens},
        prompt_pieces=result.prompt_pieces,
    )
```

`execute_with_your_engine` is the function you supply. It is not part of
QUAIL-B.

The returned fields mean:

- `filter_answers[("r", 0)]` has every document evaluated by filter 0
  on alias `r`, plus the model's boolean answer.
- `filter_answers[("r", 1)]` has every document that reached filter 1,
  plus its answer.
- `join_answers[0]` has every pair evaluated by join 0, plus its answer.
- `rows` has the final query result. It has one ID column per alias in
  `query.select`.
- `runtime_s` is query execution time. It excludes model startup,
  result collection, scoring, and saving.
- `measurements["fresh_tokens"]` is the input work measured by the engine.
- `prompt_pieces` lets QUAIL-B compute the minimum and recomputed KV tokens.

Filter and join indices start at 0. Pass `None` for the answer dictionaries
if your engine did not record individual predicate answers. QUAIL-B can
still score final output precision and recall.

## Run the benchmark

Pass that adapter to `quail_b.run`:

```python
quail_b.run(
    run_query,
    queries=["IMDB-4"],
    scale_factor=0.1,
    output_dir="results/my-run",
    metadata={"engine": "my-engine", "model": "Qwen3-4B-FP8"},
)
```

### Data download and host memory

By default, `quail_b.run` reads the corpus and reference labels anonymously
from the public `s3://quail-bench` bucket. The first run downloads immutable
Parquet and manifest files to `~/.cache/quail-b`, or
`$XDG_CACHE_HOME/quail-b`. Later runs reuse those files. The small pointer
to the active reference collection is refreshed from S3.

Before it calls your adapter, the current loader puts the selected Arrow
input tables and the full reference-label collection in host memory. The
full label collection is loaded even when `queries` selects one query.
Budget:

| Scale factor | Reference answers | Loader peak RAM | Host RAM to use |
| ---: | ---: | ---: | ---: |
| 0.1 | 1.21 million | 0.84 GiB measured | 2 GiB or more |
| 0.5 | 17.62 million | 10–12 GiB estimated | 16 GiB or more |
| 1.0 | 51.80 million | 30–35 GiB estimated | 48 GiB or more |

The 0.5 and 1.0 estimates scale the measured 0.1 label-memory cost by the
published answer counts. They are planning values, not measured peaks.
They exclude your engine, model, and returned result tables. Use 64 GiB
for a full-scale run when the engine shares the same host.

- Scale factors `0.1`, `0.5`, and `1.0` are supported.
- Override the download cache with `cache_dir=` or `QUAIL_B_CACHE_DIR`.
- Pass `data_dir=` to use local input Parquet files. Reference labels still
  come from S3 unless you pass a local published-data mirror as `root=`.
- `gpu_count=` and `gpu_hourly_rate_usd=` add GPU cost.
- An existing `output_dir` is never overwritten.

## Read the results

`quail_b.run` writes `results/my-run/`:

```
results/my-run/run.json
results/my-run/IMDB-4/rows.parquet
results/my-run/IMDB-4/filters-0.parquet
results/my-run/IMDB-4/filters-1.parquet
results/my-run/IMDB-4/joins-0.parquet
results/my-run/report.md
results/my-run/measurements.parquet
```

`report.md` has three tables. Numbers below are from a two-review fixture,
not a published corpus.

Query time, cost, throughput:

| Query | Status | Seconds | $/query | Throughput | Unit |
| --- | --- | ---: | ---: | ---: | --- |
| IMDB-4 | complete | 2 | 0.004 | 1 | document pairs/second |

Accuracy against the labels:

| Query | Predicate accuracy | Evaluated answers | Output precision | Output recall |
| --- | ---: | ---: | ---: | ---: |
| IMDB-4 | 1 | 6 | 1 | 1 |

Input size and token counts (`unavailable` until you send token data):

| Query | Input rows by alias | Fresh tokens | Minimum tokens | Recomputed KV tokens |
| --- | --- | ---: | ---: | ---: |
| IMDB-4 | r: 2, a: 1 | unavailable | unavailable | unavailable |

Filter throughput is input documents per second. Join throughput is
evaluated pairs per second across all stages. Predicate accuracy is
agreement on evaluated answers; that count can differ between engines.
Labels are Qwen3 32B fp8; FEVER and LePaRD also use source labels.

Rebuild the report from saved answers:

```sh
quail-b report results/my-run
```

Query definitions: [`quail_b/queries.py`](quail_b/queries.py).
Tables: [`quail_b/data.py`](quail_b/data.py).
