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

## Install

Python 3.12.

```sh
uv add "quail-b @ git+https://github.com/fsdatalab/quail-bench.git"
```

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

- Scale factors `0.1`, `0.5`, and `1.0` are supported.
- Inputs and labels come from public S3. No AWS account is needed.
- Downloads use `~/.cache/quail-b`, or `$XDG_CACHE_HOME/quail-b`.
  Override with `cache_dir=` or `QUAIL_B_CACHE_DIR`.
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

## Add token metrics (optional)

To score KV reuse, also pass `prompt_pieces` (tokenizer name and the token
ids around each document; see `quail_b.minimum.validate_prompt_pieces`)
and `measurements={"fresh_tokens": ...}` (input token positions the model
computed rather than read from existing KV). Scoring fills
`minimum_tokens` and `regret_tokens`. Do not report those two.

Query definitions: [`quail_b/queries.py`](quail_b/queries.py).
Tables: [`quail_b/data.py`](quail_b/data.py).
