# Quail

Quail (QUery-Aware Inference Layer) is an open-source execution engine
for AI-SQL. The [Full Stack Data Lab](https://fsdatalab.github.io/)
at CMU develops Quail.

With AI-SQL, you write instructions in natural language inside your SQL
queries. A language model evaluates those instructions for each row.

```sql
SELECT r.id
FROM reviews r
WHERE AI.IF(PROMPT('Does this review discuss the ending?\n\n{0}', r.body))
```

[Documentation](https://fsdatalab.github.io/quail) |
[Quickstart](https://fsdatalab.github.io/quail/docs/user-guide/quickstart) |
[QUAIL-B](https://github.com/fsdatalab/quail-bench)

## Install

Install [quail-engine from PyPI](https://pypi.org/project/quail-engine/).
The package is named `quail-engine`; in Python, import `quail`.

```bash
uv pip install quail-engine
```

You need Python 3.12 and a CUDA GPU.

## Example

```python
import pyarrow as pa
import quail

reviews = pa.table({
    "id": ["r1", "r2", "r3"],
    "body": [
        "A beautiful film with outstanding performances.",
        "Terrible pacing and a nonsensical plot.",
        "The cinematography was stunning, though the story dragged.",
    ],
})

config = quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")
with quail.Session(config=config) as session:
    session.register("reviews", quail.DocumentProvider.from_table(reviews, id_col="id"))
    result = session.sql("""
        SELECT r.id
        FROM reviews r
        WHERE AI.IF(PROMPT(
            'Does this review mention a positive aspect of the movie?\n\n{0}',
            r.body))
    """, dialect="bq").collect()
    print(result)
```

## Quail Server

You can use Quail Server to run queries on the machine with your GPU.
Start the HTTP server and set `endpoint` to its address. Your queries
continue if you disconnect. `submit()` returns after the server saves
your query. Use `get_run(query_id)` to reconnect later.

```bash
uv pip install "quail-engine[server]"
quail-server
```

```python
import pyarrow as pa
import quail

reviews = pa.table({
    "id": ["r1", "r2"],
    "body": ["The ending was excellent.", "I liked the soundtrack."],
})
sql = """
    SELECT r.id
    FROM reviews r
    WHERE AI.IF(PROMPT('Does this discuss the ending? {0}', r.body))
"""
config = quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")

with quail.Session(
    config=config,
    endpoint="http://127.0.0.1:8642",
) as session:
    session.register(
        "reviews",
        quail.DocumentProvider.from_table(reviews, id_col="id"),
    )
    run = session.sql(sql, dialect="bq").submit()
    for status in run.watch():
        print(status.phase["message"])
    table = run.result().collect()
    print(table)
```

See [Quail Server](https://fsdatalab.github.io/quail/docs/user-guide/server).

## Supported functions

You can use filters and joins evaluated by a language model,
`EXISTS` / `NOT EXISTS`, numeric scores (`AI.SCORE`), classification
(`AI.CLASSIFY`), and extracting text that answers a question (`AI.EXTRACT`).
`AI.MAP` is planned.

You can write queries in two AI-SQL dialects:
[Snowflake `AI_FILTER`](https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql)
and
[BigQuery `AI.IF`](https://cloud.google.com/blog/products/data-analytics/sql-reimagined-for-the-ai-era-with-bigquery-ai-functions),
or use the Python builder API.

## Supported models and GPUs

| Model | Device |
| --- | --- |
| Qwen3 4B fp8 | NVIDIA H100 SXM |
| Qwen3 32B fp8 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| DiffusionGemma 26B-A4B fp8 | NVIDIA H100 SXM |
| Decision 2.0 Kai 0.6B bf16 | NVIDIA H100 SXM |
| Qwen3.5 4B bf16 | NVIDIA H100 SXM |
| Qwen3.5 4B fp8 | NVIDIA H100 SXM |
| Qwen3.5 27B fp8 | NVIDIA H100 SXM |
| Qwen3.8 27B fp8 | NVIDIA H100 SXM |
| LFM2 2.6B bf16 | NVIDIA H100 SXM |

You can use 1, 2, 4, or 8 GPUs per query.

We are actively adding more hardware and models.

## Development

```bash
uv run ruff check quail tests experiments tools
uv run python tools/check_long_strings.py
uv run vulture
uv run pytest -q
```

## Contributing

See the [contributing guide](https://fsdatalab.github.io/quail/docs/contributing)
for how to propose and submit changes.
