# Quail

Quail is a declarative, extensible query engine for AI SQL and LLM-powered
operators over unstructured data.

[Quickstart](docs/content/docs/user-guide/quickstart.mdx) ·
[SQL](docs/content/docs/user-guide/sql.mdx) ·
[Python](docs/content/docs/user-guide/python-api.mdx) ·
[Architecture](docs/content/docs/architecture/index.mdx) ·
[QUAIL-B](https://github.com/fsdatalab/quail-bench)

## What is Quail?

Quail lets you use natural-language predicates inside relational queries. You
can filter one document collection, join several collections, and compose
LLM-powered operators with ordinary SQL or the Python API.

- Write AI SQL with Snowflake-style `AI_FILTER` or BigQuery-style `AI.IF`.
- Build the same queries with a declarative Python API.
- Add models, backends, planning rules, physical operators, and observers
  through the extension registry.
- Receive Arrow tables with rows, query metrics, and execution details.

Without Quail, you submit each LLM call yourself and write the code that
connects model responses back to relational data.

## Example

This query filters toxic comments, then joins each matching comment with the
toxicity fields that apply to it:

```sql
SELECT c.comment_id, f.field
FROM comments c
JOIN fields f
  ON AI_FILTER(PROMPT(
       'Does the description in DOCUMENT {1} apply to '
       'the comment in DOCUMENT {0}?',
       c.text,
       f.statement
     ))
WHERE AI_FILTER(PROMPT(
  'Is the comment in DOCUMENT {0} hateful, threatening, or abusive?',
  c.text
))
```

The model answers each predicate with one constrained token: `TRUE` or
`FALSE`. The query returns an Arrow table of matching comment and field pairs,
along with query time, token counts, selectivity, KV use, and GPU cost.

The complete example is in
[`demos/civil_comments_join.py`](demos/civil_comments_join.py).

## How it works

Quail combines four execution techniques:

1. **Pipelining.** A comment that passes the toxicity filter can enter the join
   immediately. It does not wait for every other comment to finish the filter.
2. **Token-based admission.** Quail fills each forward pass by token count and
   available KV, rather than by request count.
3. **KV rewind.** When KV capacity allows, the comment prefix stays on the GPU
   while Quail asks its later questions. Quail can then compute only the new
   question tokens.
4. **Packed joins.** One anchor document shares its KV across many join
   partners in the same forward pass.

Pipelining and token-based admission reduce waiting and unused batch
capacity. KV rewind and packed joins reduce fresh input-token computation
when the needed KV remains available.

KV is the attention key and value tensors stored for tokens the model has
already processed.

## Install

Quail requires Python 3.12. The package distribution is named
`quail-engine`, and the Python package is `quail`. Until the first package
release, install from the repository:

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first,
then run:

```bash
git clone https://github.com/fsdatalab/quail.git
cd quail
uv sync --no-dev
```

Model execution needs one CUDA GPU per model copy (H100 SXM or RTX PRO 6000
Blackwell). Quail does not split one model across several GPUs.

The model runs in the process that creates the `Session`. Quail does not
depend on Modal or on any shared volume. Modal is one way to rent the GPU.

## Run the quickstart

On a machine with a supported GPU:

```bash
uv run python demos/quickstart.py
```

Without a local GPU, submit the same query to a Modal H100:

```bash
uv run --no-sync --with 'modal[api-proxy-support]==1.5.4' modal setup
mkdir -p results
uv run --no-sync --with 'modal[api-proxy-support]==1.5.4' \
  modal run demos/quickstart_modal.py \
  2>&1 | tee results/quickstart.log
```

The example runs QUAIL-B's IMDB-1 predicate over eight published reviews. See
the [quickstart](docs/content/docs/user-guide/quickstart.mdx) to inspect the
query, measured output, plan, and execution report.

## Current scope and roadmap

Quail currently runs true-or-false filters and joins through Snowflake-style
`AI_FILTER`, BigQuery-style `AI.IF`, or the Python builder. It reads Arrow,
Parquet, and Hugging Face tables and returns Arrow results.

The next operators expand what a model call can return:

- `AI.CLASSIFY` returns one label from a declared set.
- `AI.EXTRACT` returns typed fields from a document.
- `AI.MAP` returns a typed value for each input row.

These operators are planned, not implemented. Open-ended generation,
speculation, and request forking are not part of the current runtime.

## Evaluation

[QUAIL-B](https://github.com/fsdatalab/quail-bench) provides 33 filter and
join queries, their datasets, reference labels, and scoring. The runner
compares Quail with stock vLLM using operator-at-a-time execution and with
pipelined vLLM on the same query definitions.

In the September 12, 2026 sf=0.1 run with Qwen3 4B fp8 and one H100 per
configuration, Quail had lower query time than stock vLLM on 31 of the 33
queries. Its total query time was 1,542.13 seconds, compared with 3,249.04
seconds for stock vLLM. Because both used one H100, total GPU cost fell by
the same 52.5%.

On IMDB-10, Quail computed 5.48 million fresh input tokens, compared with
7.71 million for stock vLLM. Query time was 48.58 seconds, compared with
85.89 seconds. Each backend's answers can change the rows that reach later
stages, so the token difference includes both execution and answer-path
differences.

Query time excludes model startup. The saved run is
`/results/benchmarks/quailb/family-runs/20260912T225100Z-902686c5/` on the
`quail-results` Modal volume. QUAIL-B computes accuracy, output precision and
recall, fresh input tokens, throughput, and GPU cost from the saved results.

On a machine with a supported GPU, run one query and save the results to a
local directory. QUAIL-B downloads the inputs and reference labels from
public S3.

```bash
uv run python -m quail.bench.quailb \
  --sf 0.1 --only IMDB-4 \
  --model qwen3-4b-fp8 --device h100-sxm \
  --output-dir results/quailb/imdb-4
uv run quail-b report results/quailb/imdb-4
```

To run Quail and both stock vLLM baselines on Modal H100s:

```bash
uv run modal run --detach -m quail.bench.quailb_parallel \
  --sf 0.1 --model qwen3-4b-fp8 --query IMDB-4 \
  --output-dir /results/benchmarks/quailb \
  2>&1 | tee results/quailb.log
```

See [Benchmarks](docs/content/docs/contributing/benchmark.mdx) for report
generation and baseline options, and
[Metrics](docs/content/docs/user-guide/metrics.mdx) for the definitions.

## Documentation

| If you want to… | Start here |
| --- | --- |
| Run one query | [Quickstart](docs/content/docs/user-guide/quickstart.mdx) |
| Write filters and joins in SQL | [SQL reference](docs/content/docs/user-guide/sql.mdx) |
| Build queries in Python | [Python API](docs/content/docs/user-guide/python-api.mdx) |
| Load your own tables | [Running queries](docs/content/docs/user-guide/sessions.mdx) |
| Understand plans and metrics | [Results and explain](docs/content/docs/user-guide/results.mdx) |
| Pick a model or GPU, or run on Modal | [Supported models and GPUs](docs/content/docs/user-guide/models.mdx) |
| Understand the engine | [Architecture](docs/content/docs/architecture/index.mdx) |
| Add an extension | [Extending Quail](docs/content/docs/extending/index.mdx) |

## Development

The test suite uses a fake model executor and runs on the CPU.

```bash
uv run ruff check quail tests experiments tools
uv run python tools/check_long_strings.py
uv run vulture
uv run pytest -q
```

See [Contributing](docs/content/docs/contributing/index.mdx) for repository
layout, checks, and release instructions.
