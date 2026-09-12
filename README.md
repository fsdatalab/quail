# QUAIL-B

QUAIL-B contains 33 AI filter and join queries over movie reviews, medical
reports, factual claims, legal citations, and agent conversations.

Your engine runs the queries. QUAIL-B loads the inputs and reference labels,
saves the returned answers, computes metrics, and writes a report.

## Install

Use Python 3.12.

```sh
uv add "quail-b @ git+https://github.com/fsdatalab/quail-bench.git"
```

## Run

Supply a function `run_query(query, tables)` that returns a `quail_b.RunOutput`.
The query is a benchmark definition. The tables are Arrow tables with document IDs.

```python
import quail_b

quail_b.run(
    run_query,
    queries=["IMDB-4"],
    scale_factor=0.1,
    output_dir="results/my-run",
)
```

For example, [Quail's runner](https://github.com/fsdatalab/quail-exploration/blob/main/quail/bench/quailb.py)
translates each definition into a Quail query. Your calling script owns model
startup, engine configuration, hardware, and any Modal resources.

- Scale factors `0.1`, `0.5`, and `1.0` are supported for filters and joins.
- Inputs and labels come from public S3. No AWS account is needed.
- Downloads use `~/.cache/quail-b`, or `$XDG_CACHE_HOME/quail-b` when set.
  Override with `cache_dir=` or `QUAIL_B_CACHE_DIR`. Active label pointers
  are refreshed; each run records one exact collection.
- Use `metadata=` to record engine, model, configuration, warmup, and cache settings.
  Supply `gpu_count=` and `gpu_hourly_rate_usd=` to include GPU cost.
- Existing run directories are never overwritten.

## Return results

`run_query` returns a `RunOutput`:

- `rows`: final document IDs. Column names are the selected aliases.
- `runtime_s`: query time, excluding startup, result collection, scoring,
  and saving. Put those other durations in `measurements` if you have them.
- `filter_answers`: tables keyed by `(alias, predicate_position)`.
- `join_answers`: tables keyed by join position.
- `measurements`: optional extra numbers, such as `evaluated_document_pairs`.

Answer tables have alias ID columns and a non-null boolean `answer` column.
Pass `None` for predicates you did not evaluate. Final-output precision and
recall still score.

Token counts are optional. To include them, also return `prompt_pieces`
(tokenizer name and the token ids around each document; see
`quail_b.minimum.validate_prompt_pieces`) and set `measurements["fresh_tokens"]`
to the number of input token positions the model computed rather than read
from existing KV. Scoring then fills `minimum_tokens` and `regret_tokens`.
Do not report those two yourself.

## Report

A run writes `run.json`, per-query answer Parquet, `report.md`, and
`measurements.parquet`. Answers are saved before scoring. Failures stay
in the run record.

Regenerate from saved answers:

```sh
quail-b report results/my-run
```

The report has query time, throughput, GPU cost when supplied, and accuracy.
Filter throughput is input documents per second. Join throughput is
evaluated pairs per second across all stages. Token counts appear when you
returned `prompt_pieces` and `fresh_tokens`. Predicate accuracy is agreement
on evaluated answers; that count can differ between engines. Model-generated
labels use Qwen3 32B fp8; FEVER and LePaRD also use source labels.

See the [query definitions](quail_b/queries.py) and [data sources](quail_b/data.py).
