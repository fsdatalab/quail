# QUAIL-B

QUAIL-B is 33 filter and join queries over document tables. You write a
function that runs one query. This package loads the tables and labels,
calls your function, scores the answers, and writes a report.

## Install

Python 3.12.

```sh
uv add "quail-b @ git+https://github.com/fsdatalab/quail-bench.git"
```

## Use

You implement `run_query(query, tables)` and pass it to `quail_b.run`.

`query` is a `QuerySpec`. `tables` maps each table name to an Arrow table
with an `id` column and the text column named on the alias.

IMDB-4 is two filters on reviews, then a join to aspects:

```python
import pyarrow as pa
import quail_b

query = quail_b.get_query("IMDB-4")
# query.id == "IMDB-4"
# query.aliases[0]  -> alias "r", table "reviews", column "body", two filters
# query.aliases[1]  -> alias "a", table "aspects", column "aspect"
# query.joins[0]    -> aliases ("r", "a")
# query.select      -> ("r.id", "a.id")


def run_query(query, tables):
    reviews = tables["reviews"]   # columns include id, body
    aspects = tables["aspects"]   # columns include id, aspect
    # Ask the model. Return the table id values and the boolean answers.
    # The ids below are stand-ins for those column values.

    return quail_b.RunOutput(
        filter_answers={
            ("r", 0): pa.table({"r": ["r0", "r1"], "answer": [True, True]}),
            ("r", 1): pa.table({"r": ["r0", "r1"], "answer": [True, False]}),
        },
        join_answers={
            0: pa.table({"r": ["r0"], "a": ["a0"], "answer": [True]}),
        },
        rows=pa.table({"r": ["r0"], "a": ["a0"]}),
        runtime_s=12.4,
    )


quail_b.run(
    run_query,
    queries=["IMDB-4"],
    scale_factor=0.1,
    output_dir="results/my-run",
)
```

`filter_answers` is keyed by `(alias, filter index)` on that alias, starting
at 0. `join_answers` is keyed by join index, starting at 0. Each answer
table has alias id columns and a non-null boolean `answer` column. `rows`
has one id column per selected alias.

Pass `None` for `filter_answers` or `join_answers` if you did not record
predicates. Final-output precision and recall still score.

A [real runner](https://github.com/fsdatalab/quail-exploration/blob/main/quail/bench/quailb.py)
turns each `QuerySpec` into a Quail query. Your script owns model startup,
hardware, and any Modal resources.

- Scale factors `0.1`, `0.5`, and `1.0`.
- Inputs and labels come from public S3. No AWS account is needed.
- Downloads use `~/.cache/quail-b`, or `$XDG_CACHE_HOME/quail-b`.
  Override with `cache_dir=` or `QUAIL_B_CACHE_DIR`.
- `metadata=` records engine and model settings. `gpu_count=` and
  `gpu_hourly_rate_usd=` add GPU cost.
- Existing `output_dir` values are never overwritten.

## What you get

`quail_b.run` writes `results/my-run/`:

```
results/my-run/run.json
results/my-run/IMDB-4/rows.parquet
results/my-run/IMDB-4/filters-0.parquet
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

## Token counts (optional)

To score KV reuse, also pass `prompt_pieces` (tokenizer name and the token
ids around each document; see `quail_b.minimum.validate_prompt_pieces`)
and `measurements={"fresh_tokens": ...}` (input token positions the model
computed rather than read from existing KV). Scoring fills
`minimum_tokens` and `regret_tokens`. Do not report those two.

Query definitions: [`quail_b/queries.py`](quail_b/queries.py).
Tables: [`quail_b/data.py`](quail_b/data.py).
