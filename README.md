# Quail

Quail is a query engine for running language-model operations over document
collections. You write the analysis in SQL or Python. Quail plans the complete
query before it starts inference, then schedules the model work so repeated
document prefixes can share KV and each forward pass can use more of the
available token budget.

**Today:** `AI_FILTER`, AI joins, `EXISTS`, and `NOT EXISTS` with Qwen3 4B fp8
or Qwen3 32B fp8 on H100s.

**On the roadmap:** `AI.CLASSIFY`, `AI.EXTRACT`, and `AI.MAP`.

[Quickstart](docs/content/docs/user-guide/quickstart.mdx) ·
[SQL](docs/content/docs/user-guide/sql.mdx) ·
[Python](docs/content/docs/user-guide/python-api.mdx) ·
[Architecture](docs/content/docs/architecture/index.mdx) ·
[QUAIL-B](https://github.com/fsdatalab/quail-bench)

## Why Quail?

Suppose we want to analyze the 448,000 comments in the Jigsaw Civil Comments
dataset. First, we ask a model which comments are toxic. For each comment that
passes, we ask 31 more questions about toxicity type, identity references, and
moderator decisions.

A request-at-a-time system treats every question as separate work. It
processes the same comment tokens again for later questions, waits for the
whole filter to finish before starting the join, and groups requests by count
even when their documents have very different lengths.

Quail treats the analysis as one query. The planner sees the filter, the
user's selectivity estimates, the join against 31 field descriptions, each
document's token count, and the available GPU memory before execution begins.

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
uv sync
```

Model execution requires an H100. Each GPU holds one model copy; Quail does
not split one model across several GPUs.

## Run the quickstart

On an H100 machine:

```bash
uv run python demos/quickstart.py
```

From any machine, submit the same query to a Modal H100:

```bash
uv run modal setup
mkdir -p results
uv run modal run demos/quickstart_modal.py \
  2>&1 | tee results/quickstart.log
```

The example runs QUAIL-B's IMDB-1 query over 100 published reviews. See the
[quickstart](docs/content/docs/user-guide/quickstart.mdx) to inspect the query,
result, plan, and execution report.

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
queries. Query time excludes model startup. QUAIL-B stores the full result
tables and computes accuracy, output precision and recall, fresh input
tokens, throughput, and GPU cost.

```bash
uv run modal run --detach -m quail.bench.quailb_parallel \
  --sf 0.1 --model qwen3-4b-fp8 --query IMDB-4 \
  --output-dir /results/benchmarks/quailb \
  2>&1 | tee results/quailb.log
```

See [Running QUAIL-B](docs/content/docs/user-guide/benchmark.mdx) for report
generation, metric definitions, and baseline options.

## Documentation

| If you want to… | Start here |
| --- | --- |
| Run one query | [Quickstart](docs/content/docs/user-guide/quickstart.mdx) |
| Write filters and joins in SQL | [SQL reference](docs/content/docs/user-guide/sql.mdx) |
| Build queries in Python | [Python API](docs/content/docs/user-guide/python-api.mdx) |
| Load your own tables | [Data sources](docs/content/docs/user-guide/data-sources.mdx) |
| Understand plans and metrics | [Results and explain](docs/content/docs/user-guide/results.mdx) |
| Run on Modal | [Running on Modal](docs/content/docs/user-guide/compute.mdx) |
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
