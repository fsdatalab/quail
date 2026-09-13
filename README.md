# Quail

Quail is a query engine for language-model filters and joins over documents.
Write a query in SQL or Python. Quail plans all model calls together so they
can share work on the GPU.

[Quickstart](docs/content/docs/user-guide/quickstart.mdx) ·
[SQL reference](docs/content/docs/user-guide/sql.mdx) ·
[Python API](docs/content/docs/user-guide/python-api.mdx) ·
[Architecture](docs/content/docs/architecture/index.mdx) ·
[QUAIL-B](https://github.com/fsdatalab/quail-bench)

## What Quail does

AI functions in data warehouses usually send one model request for each row.
That approach cannot share model work across predicates or query stages.
Quail sees the complete query first and schedules its filters and joins as one
job.

```python
import quail

with quail.Session() as session:
    session.register(
        "reviews",
        quail.DocumentProvider.from_table(reviews, id_col="id"),
    )
    result = session.sql("""
        SELECT r.id
        FROM reviews r
        WHERE AI_FILTER(PROMPT(
            'Does the review in DOCUMENT {0} praise the movie?',
            r.body
        ))
    """).run()
    rows = result.collect()
```

The model answers each predicate with one token: `TRUE` or `FALSE`. Quail
supports filters over one table and AI joins across two or more tables. It
does not run open-ended generation, classification, or extraction.

## Why plan the whole query?

- **Pipelining:** a document starts its next predicate as soon as it passes
  the current one.
- **Token-based admission:** each forward pass is filled by token count and
  available KV, rather than by request count.
- **KV rewind:** Quail keeps a document's KV on the GPU and reuses it for the
  document's later predicates.
- **Packed joins:** one anchor document shares its KV across many join
  partners in the same forward pass.

KV is the attention key and value tensors stored for tokens the model has
already processed.

## Install and run

Quail currently installs from this repository. It requires Python 3.12.
Model execution requires a CUDA GPU. The supported models are Qwen3 4B fp8
and Qwen3 32B fp8, with one model copy per GPU.

```bash
git clone https://github.com/fsdatalab/quail-exploration.git
cd quail-exploration
uv sync
uv run python demos/quickstart.py
```

The quickstart runs QUAIL-B's IMDB-1 query over 100 reviews. To use a Modal
H100 instead of a local GPU:

```bash
uv run modal setup
mkdir -p results
uv run modal run demos/quickstart_modal.py \
  2>&1 | tee results/quickstart.log
```

See [Running on Modal](docs/content/docs/user-guide/compute.mdx) for the
complete setup.

## Documentation

| Guide | Use it to |
| --- | --- |
| [Quickstart](docs/content/docs/user-guide/quickstart.mdx) | Run a filter and inspect its rows, plan, and execution report. |
| [SQL reference](docs/content/docs/user-guide/sql.mdx) | See the supported `AI_FILTER`, join, and `EXISTS` syntax. |
| [Python API](docs/content/docs/user-guide/python-api.mdx) | Build the same queries without SQL. |
| [Data sources](docs/content/docs/user-guide/data-sources.mdx) | Register Arrow, Parquet, or Hugging Face tables. |
| [Results and explain](docs/content/docs/user-guide/results.mdx) | Read output tables, metrics, and physical plans. |
| [Architecture](docs/content/docs/architecture/index.mdx) | Follow a query from parsing through GPU execution. |
| [Extending Quail](docs/content/docs/extending/index.mdx) | Add providers, plan rules, backends, or observers. |

## Benchmark

[QUAIL-B](https://github.com/fsdatalab/quail-bench) provides 33 filter and
join queries, datasets, reference labels, and scoring. Quail's benchmark
runner compares the same queries with stock vLLM using operator-at-a-time
execution and with pipelined vLLM.

```bash
uv run modal run --detach -m quail.bench.quailb_parallel \
  --sf 0.1 --model qwen3-4b-fp8 --query IMDB-4 \
  --output-dir /results/benchmarks/quailb \
  2>&1 | tee results/quailb.log
```

See [Running QUAIL-B](docs/content/docs/user-guide/benchmark.mdx) for saved
results, report generation, and baseline options.

## Development

The test suite runs on the CPU with a fake model executor.

```bash
uv run ruff check quail tests experiments reports tools
uv run python tools/check_long_strings.py
uv run vulture
uv run pytest -q
```

Experiment code is under `experiments/`. Results and design notes are under
`reports/`.
