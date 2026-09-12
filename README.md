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

`RunOutput` contains:

- `rows`: an Arrow table of final document IDs, with columns named for selected aliases.
- `runtime_s`: completed query time, excluding startup, result collection,
  scoring, and saving. Report those other durations in `measurements`.
- `filter_answers`: tables keyed by `(alias, predicate_position)`.
- `join_answers`: tables keyed by join position.
- `measurements`: optional values such as `fresh_tokens` and
  `evaluated_document_pairs`. A fresh token is an input token position a
  model forward pass processed instead of reading it from existing KV.
- `prompt_pieces`: optional. The token ids the engine put around each
  document: `tokenizer` (the HuggingFace tokenizer name), `preamble`,
  one `{"alias", "position", "tail"}` per filter stage, and one
  `{"position", "anchor", "frame", "label", "tail"}` per join, where a
  join request is preamble, anchor document, frame, label, partner
  document, tail. See `quail_b.minimum.validate_prompt_pieces`.

Answer tables contain alias ID columns and a non-null boolean `answer` column.
Use `None` for unavailable predicate answers. QUAIL-B still scores final-output
precision and recall. Missing measurements are reported as unavailable.

With the answers, `fresh_tokens`, and `prompt_pieces`, scoring computes
`minimum_tokens`: the fewest input tokens the run's requests needed with
unlimited KV, where every distinct token prefix across the requests is
computed once. `regret_tokens` is `fresh_tokens` minus that minimum: the
KV the engine recomputed. Both are computed after the run, on the CPU,
from the saved answer tables and the corpus tokenized with the named
tokenizer, so they never count toward `runtime_s`.

## Report

A run saves `run.json`, per-query Parquet answers, `report.md`, and
`measurements.parquet`, one row per completed query with its runtime,
token counts, accuracy counts, and cost.
Completed answers are saved before scoring. Errors remain recorded in the run.

Regenerate the report from saved answers without executing queries:

```sh
quail-b report results/my-run
```

The report includes query time, throughput, GPU cost when supplied, and accuracy.
Filter throughput counts input documents; join throughput counts evaluated pairs
across all stages. Predicate accuracy is agreement on evaluated answers, whose
count can differ between engines. Model-generated reference labels use Qwen3 32B
fp8; FEVER and LePaRD also use source labels.

See the [query definitions](quail_b/queries.py) and [data sources](quail_b/data.py).
