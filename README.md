# QUAIL-B

QUAIL-B tests how an inference engine runs queries over text documents.
An AI filter asks a TRUE or FALSE question about one document. An AI join
asks a TRUE or FALSE question about a pair of documents.

The benchmark includes 32 queries, 9 document tables, and saved reference
labels. Published data is available at scale factors `0.1`, `0.5`, and
`1.0`. You can load data and score saved answers without a GPU or an AWS
account. Running the queries requires an inference engine.

## Install

Use Python 3.12. Install the revision that includes the loading API below:

```sh
uv add "quail-b @ git+https://github.com/fsdatalab/quail-bench.git@e30d8245d8e169c9df04ece224164d11409db7da"
```

The package is imported as `quail_b`. It does not install Quail or another
inference engine.

## Load data and choose a query

```python
import quail_b as benchmark

reviews = benchmark.load_table("reviews", limit=100)
query = benchmark.get_query("IMDB-1")

print(reviews.schema)
print(query.description)
```

IMDB-1 finds reviews that mention at least one positive aspect of a movie.
`reviews` is an Arrow table with `id` and `body` columns. `query` is a
definition of the query, not an executed result.

The default scale factor is `0.1`. A row limit selects the first rows of
that published table. Omit `limit` for a full benchmark run.

## Load the inputs for a join

IMDB-4 filters for reviews that mention a positive aspect and discuss the
ending. It then joins each surviving review with the movie aspects it discusses.

```python
import quail_b as benchmark

query = benchmark.get_query("IMDB-4")
scale_factor = 0.1

tables = {}
for name in sorted({alias.table for alias in query.aliases}):
    tables[name] = benchmark.load_table(name, scale_factor=scale_factor)

print({name: table.num_rows for name, table in tables.items()})
```

Load every input table at the same scale factor. If a query uses a table
more than once, load it once and reuse it for each alias. An alias is the
name given to one use of a table in a query.

Scale factors select the input documents, not the number of output pairs.
Join work depends on both input sizes and which rows pass the filters.

## Available data

The counts below are input rows, before any query filters. They come from
the published Parquet files.

| Table | Scale 0.1 | Scale 0.5 | Scale 1.0 |
| --- | ---: | ---: | ---: |
| `reviews` | 5,000 | 25,000 | 50,000 |
| `aspects` | 12 | 12 | 12 |
| `reports` | 500 | 2,500 | 5,000 |
| `terms` | 1,127 | 2,934 | 4,144 |
| `claims` | 500 | 2,500 | 5,000 |
| `evidence` | 287 | 1,037 | 1,478 |
| `citation_contexts` | 500 | 2,496 | 4,972 |
| `citation_passages` | 433 | 1,756 | 2,991 |
| `agent_traces` | 1,772 | 8,859 | 17,711 |

The document sources are:

- IMDB movie reviews and a fixed list of 12 movie aspects.
- BioDEX medical reports and the reaction terms in those reports.
- FEVER claims and their associated Wikipedia evidence.
- LePaRD citation contexts and cited passages.
- SWE-Next agent conversations, saved at selected turns.

Sources, revisions, sampling rules, and published dataset ids are in
[`quail_b/data.py`](quail_b/data.py). Table sizes do not all grow in direct
proportion to the scale factor. Some tables contain distinct values from
another table, and the movie aspects are fixed.

`load_table()` reads the requested table from the public S3 bucket.
To use a downloaded copy, pass `root="/path/to/data"` with the same
directory layout as the bucket. An `s3://` root is also accepted for
anonymous reads. Missing files raise an error.

For source-based dataset construction, use `build_sets()` in
[`quail_b/data.py`](quail_b/data.py). It first tries the published data
and checks the dataset id before rebuilding from the original sources.

## Queries

| Family | Query ids | What the queries ask about |
| --- | --- | --- |
| IMDB | IMDB-1 to IMDB-10 | Review content, discussed movie aspects, and sentiment |
| BioDEX | BIO-1 to BIO-3 | Patient descriptions and reported reactions |
| FEVER | FEV-1 to FEV-9 | Claim content and supporting or refuting evidence |
| LePaRD | LEP-1 to LEP-8 | Legal reasoning and links between citations and passages |
| SWE-Next | AGENT-1 and AGENT-2 | Recovery from failed approaches and plausible fixes |

The suite includes individual filters, sequences of filters, joins with
filters on either input, and queries with several joins.

A query definition lists its table aliases, exact question templates,
joins, and selected output columns. Read the definitions in
[`quail_b/queries.py`](quail_b/queries.py) and the question text in
[`quail_b/prompts.py`](quail_b/prompts.py).

The optional PRIV-1 and PRIV-2 definitions are separate from the 32-query
suite. Their privacy-policy data is not included in the published tables
supported by `load_table()`.

<details>
<summary>View all query plans</summary>

![QUAIL-B query plans](figures/quailb_anatomy.png)

Percentages are the fixed planning estimates from scale-factor 0.1 labels.
[Download the diagram as a PDF.](figures/quailb_anatomy.pdf)

</details>

## Run queries

An engine's runner translates a query definition into that engine's
operations. QUAIL-B itself does not execute models.

Quail's runner is in the separate
[Quail repository](https://github.com/fsdatalab/quail-exploration).
Its [benchmark module](https://github.com/fsdatalab/quail-exploration/blob/blogs/quail/bench/quailb.py)
translates the query definitions, executes them, and records the results.
Install and configure Quail there before running GPU queries.

## Score a run

Reference labels are saved TRUE or FALSE answers. Model-generated labels
use Qwen3 32B fp8. FEVER and LePaRD use source labels where those sources
provide the exact answer. Agreement with model-generated labels is not a
claim of human-verified accuracy.

Load the active label collection for the same published dataset:

```python
import quail_b as benchmark
from quail_b.data import PUBLISHED_CORPORA

scale_factor = 0.1
labels = benchmark.load_ground_truth(
    scale_factor=scale_factor,
    corpus_id=PUBLISHED_CORPORA[scale_factor],
)
print(labels.collection_id)
```

Record the collection id with your results. Pass it as `collection_id`
when loading labels for a repeat comparison.

The runner supplies a [`RunOutput`](quail_b/scoring.py) containing:

- Filter answers, identified by table alias and filter position.
- Join answers, identified by join position.
- Final output rows.

Answer tables use the original benchmark document ids and a boolean
`answer` column. Positions refer to the query's written order, starting at
zero, even if the engine executes operations in a different order.

`Evaluator(labels, tables).evaluate(query, output)` compares those answers
and final rows with the saved labels. Import `Evaluator` from
`quail_b.scoring`. Use the exact input tables supplied to the engine.

Report query time, throughput, GPU cost, and accuracy together:

| Metric | Definition |
| --- | --- |
| Query time | Seconds spent executing the query, excluding model startup and result collection |
| Filter throughput | Input document rows divided by query time |
| Join throughput | Evaluated document pairs across all join stages divided by query time |
| GPU cost per query | Query time in hours multiplied by GPU count and the H100 hourly price used |
| Answer accuracy | Agreement with reference labels on evaluated answers |
| Output precision and recall | Correct returned rows as a fraction of returned rows and expected rows, respectively |

Report startup time and cost separately if measured. Include the query id,
scale factor, input count for each alias, model, GPU configuration, and
label collection id with every comparison.

## Contribute

Query definitions and question templates belong in this repository.
Execution and label generation belong in the engine's repository.

A new query can reuse existing questions and labels. A changed question,
document role, or reference model requires new label identities. See
[`quail_b/predicates.py`](quail_b/predicates.py) and
[`quail_b/rendering.py`](quail_b/rendering.py) before changing them.

Run the checks from this repository:

```sh
uv sync
uv run ruff check quail_b tests tools
uv run python tools/check_long_strings.py
uv run vulture
uv run pytest -q
```
