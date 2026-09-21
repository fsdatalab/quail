# QUAIL-B

QUAIL-B is an academic benchmark of 30 AI SQL queries over document tables. AI SQL is SQL with LLM-powered operators.

This repository publishes the query plans, input tables, reference labels, and scoring harness. It does not include an execution engine. To benchmark your engine, you write an adapter function that translates each Substrait query plan into your engine's AI SQL dialect, executes it, and returns the execution results to QUAIL-B for scoring.

## Contents

- [Installation](#installation)
- [How to Adapt Your Engine](#how-to-adapt-your-engine)
- [Running the Benchmark](#running-the-benchmark)
- [Queries](#queries)
- [Scale Factors](#scale-factors)
- [Metrics](#metrics)
- [Results and CLI](#results-and-cli)
- [Source Files](#source-files)

## Installation

QUAIL-B requires Python 3.12.

```sh
uv add "quail-b @ git+https://github.com/fsdatalab/quail-bench.git"
```

For development:

```sh
git clone https://github.com/fsdatalab/quail-bench.git
cd quail-bench
uv sync
```

## How to Adapt Your Engine

To run QUAIL-B against your query engine, you write an adapter function:

```python
run_query(query: quail_b.QuerySpec, tables: dict[str, pyarrow.Table]) -> quail_b.RunOutput
```

Your adapter receives:
- `query`: A `QuerySpec` instance. Its `query.plan` property provides the parsed Substrait 0.103 `Plan`.
- `tables`: A dictionary mapping table names to input PyArrow tables.

You translate the Substrait plan into your engine's AI SQL dialect (for example, [BigQuery AI SQL](https://cloud.google.com/bigquery/docs/generative-ai-overview)), execute it, and return a `RunOutput`.

### The `RunOutput` Contract

Required fields:
- `rows`: A `pyarrow.Table` containing final output tuples, with one ID column per selected alias (such as `r` and `a`).
- `runtime_s`: Query execution time in seconds as a float. This excludes engine startup and model loading.

Optional fields (used for predicate accuracy and token accounting):
- `filter_answers`: A dictionary mapping filter operator ID (such as `"filter-1"`) to a `pyarrow.Table` of evaluated document IDs and boolean `answer` values.
- `join_answers`: A dictionary mapping join operator ID (such as `"join-1"`) to a `pyarrow.Table` of evaluated left/right ID pairs and boolean `answer` values.
- `measurements`: A dictionary for engine telemetry. Reporting `measurements["fresh_tokens"]` records the count of input tokens processed in model forward passes.
- `prompt_pieces`: Tokenized prompt IDs for prefix KV accounting. For a
  PDF relation, whose documents are rendered pages and not text, add
  `images`: table name to `{row id: prompt positions the document took}`.
  `quail_b.minimum.validate_prompt_pieces` documents every field.

If your engine does not record individual predicate answers, pass `filter_answers=None` and `join_answers=None`.

### Concrete Adapter Example

For query `IMDB-4` (see [Queries](#queries) below), your adapter translates the plan, executes the query against `tables["reviews"]` and `tables["aspects"]`, and returns the results:

```python
import pyarrow as pa
import quail_b


def run_query(query: quail_b.QuerySpec, tables: dict[str, pa.Table]) -> quail_b.RunOutput:
    sql = to_engine_sql(query.plan)
    result = execute(sql, tables)

    return quail_b.RunOutput(
        rows=pa.table({
            "r": result.output_review_ids,
            "a": result.output_aspect_ids,
        }),
        runtime_s=result.query_seconds,
        filter_answers={
            "filter-1": pa.table({"r": result.f1_ids, "answer": result.f1_answers}),
            "filter-2": pa.table({"r": result.f4_ids, "answer": result.f4_answers}),
        },
        join_answers={
            "join-1": pa.table({
                "r": result.join_review_ids,
                "a": result.join_aspect_ids,
                "answer": result.join_answers,
            }),
        },
        measurements={"fresh_tokens": result.fresh_tokens},
        prompt_pieces=result.prompt_pieces,
    )
```

`to_engine_sql` and `execute` are supplied by your adapter.

## Running the Benchmark

Pass your adapter function to `quail_b.run`:

```python
import quail_b

quail_b.run(
    run_query,
    queries=["IMDB-4"],  # Omit to run all 31 queries
    scale_factor=0.1,
    output_dir="results/my-run",  # Set your desired output directory path
    metadata={"engine": "my-engine", "model": "Qwen/Qwen3-4B-FP8"},
    gpu_count=1,
    gpu_hourly_rate_usd=3.9492,
)
```

- `output_dir`: Path to the directory where QUAIL-B writes run results (e.g. `"results/vllm-qwen3-4b"` or any custom path). Must be a new directory.
- `queries`: List of query IDs to run. Omit `queries=` (or pass `None`) to run all 30 benchmark queries.
- Data is downloaded from `s3://quail-bench` and cached locally in `~/.cache/quail-b`.
- Only the reference labels of the selected queries' predicates are loaded, as Arrow tables of about 25 bytes per answer. Loading the full published collection of 21 label sets at scale 0.1 (1.21 million answers) takes 1.9 s from cached files with a peak of 0.62 GiB, corpus tables included; at scale 1.0 (51.8 million answers) budget about 3 GiB.

### Inspecting Data and Queries Directly

Inspect benchmark queries and load input tables directly without running the benchmark harness:

```python
import quail_b

query = quail_b.get_query("IMDB-4")
plan = query.plan

reviews = quail_b.load_table("reviews", scale_factor=0.1)
```

## Queries

The benchmark defines 36 queries across 6 datasets:

| Dataset | Queries | Relations | Description |
| --- | ---: | --- | --- |
| IMDB | 10 | `reviews`, `aspects` | Movie review aspect extraction and sentiment analysis |
| BioDEX | 4 | `reports`, `terms` | Adverse drug reaction reporting from medical papers |
| FEVER | 10 | `claims`, `evidence` | Fact verification with two-sided selections and join chains |
| LePaRD | 5 | `citation_contexts`, `citation_passages` | Legal precedent retrieval and citation matching |
| SWE-Next | 2 | `agent_traces` | Software engineering agent trajectory evaluation |
| CUAD | 5 | `contracts`, `contract_pages` | Contract clause review over PDF pages rendered as images |

The original LEP-5, LEP-6, and LEP-8 have empty reference outputs at sf=0.1
with raw prompts as well as chat prompts. They are excluded. The original LEP-7
is now LEP-5; its plan and prompt text are unchanged.
Filter and join prompts use the raw document/question format ending in `ANSWER:`.
The benchmark uses the existing raw-prompt reference collections. Saved chat
results remain historical and must not be presented as raw-prompt results.

BIO-1 selects reports describing a serious or life-threatening adverse event.
BIO-3 applies that filter before joining reports to reaction terms.
BIO-4 finds serious reports with both neurological and cardiovascular reactions.
It filters the reports and two aliases of the reaction terms, then joins each
term alias to the same report. Its output is `(r.id, n.id, c.id)`, with one row
per matching reaction pair. A term may belong to both categories.

BIO-4 has Qwen3 32B fp8 reference labels for both term filters at all three
scale factors. The labels reuse the existing report and reaction-join answers
after checking their corpus and prompt identities. At sf=0.1, 505 of 1,127
terms pass the neurological filter and 394 pass the cardiovascular filter.
Those fractions are fixed planner estimates at every scale factor.
The public collections below include both filters. BIO-4 loads the matching
collection automatically when accuracy scoring is enabled.

| Scale factor | Corpus | Reference collection |
|---|---|---|
| 0.1 | `c_89e6d982644aad4553e7893a5a53c481` | `gt_a74b54853e14c80f5a7327f298b880ad` |
| 0.5 | `c_b5e46fdff06aabae95ba6905f08a31de` | `gt_a93ababf8933c23cf93f739e6d214d27` |
| 1.0 | `c_424dc3196ca47bd0ab131129ec8d48df` | `gt_c2872eec5eff63c1fcd64a3d103776a3` |

Adding the two CUAD tables gave every scale factor a new corpus id.
The collections above reuse all 23 earlier label sets unchanged (each
reused set's table manifest matched exactly) and add the 8 CUAD label
sets, which come from the CUAD annotation and needed no inference.

CUAD-1 to CUAD-5 read contract PDFs. The `document` column of both
relations is a file reference: `files/<id>.pdf` for a whole contract and
`files/<id>.pdf#page=<n>` for one page. The engine renders the referenced
pages and hands them to a model that takes images; there is no text
column to read. `load_benchmark` downloads the PDF files beside the
tables, checks each against the `pdf_sha256` column, and rewrites the
references to absolute paths. The reference labels are the CUAD lawyer
annotations, not a model's answers: a contract row is TRUE for a clause
category when the contract has an annotated span of that category, and a
page row when an annotated span touches the page. The predicates ask
about the harder categories (limitation of liability and its carve-outs,
change of control, exclusivity, non-compete, and license terms), not
about names or dates. CUAD-1 and CUAD-2 read the 9,348 pages one page per
row. CUAD-3 to CUAD-5 read whole contracts, restricted by an ordinary
`page_count <= 32` filter in the plan before the AI filters (430 of the
510 contracts; 32 pages is the most one DiffusionGemma prompt holds).
Scoring counts only the rows inside that bound as the query's input.
At sf=1.0 the annotation keeps 453 of the 9,348 pages for CUAD-1 and 124
for CUAD-2, and 78, 45, and 22 of the 430 bounded contracts for CUAD-3,
CUAD-4, and CUAD-5.

Queries use two LLM-powered relational operators:

- `ai_filter(prompt, document) -> boolean` (selection)
- `ai_join(prompt, left, right) -> boolean` (join)

All 36 queries are stored as standard Substrait 0.103 ProtoJSON plans in [`quail_b/plans/`](quail_b/plans/). Custom AI functions are declared in [`quail_b/substrait_extensions.yaml`](quail_b/substrait_extensions.yaml). [All 36 plans](figures/quailb_anatomy.pdf) are diagrammed in one figure.

### Example Query: IMDB-4

IMDB-4 applies two selection filters to movie reviews, followed by a join against movie aspects:

```sql
SELECT r.id, a.id
FROM reviews AS r
AI JOIN aspects AS a ON J1(r.body, a.aspect)
WHERE AI_FILTER(F1, r.body)
  AND AI_FILTER(F4, r.body);
```

Its relational operator tree is:

```text
Project [r.id, a.id]
└── AI Join J1                                      join-1
    ├── AI Selection F4                             filter-2
    │   └── AI Selection F1                         filter-1
    │       └── Scan reviews AS r
    └── Scan aspects AS a
```

In the Substrait plan, `F1`, `F4`, and `J1` are string prompt literals. Exact prompt texts and rendering logic are defined in [`quail_b/prompts.py`](quail_b/prompts.py) and [`quail_b/rendering.py`](quail_b/rendering.py).

Every filter and join instruction starts with "Evaluate TRUE or FALSE for the
following question:". It follows the document, so the document's KV can be reused
across questions.

## Scale Factors

QUAIL-B defines three scale factors: `0.1`, `0.5`, and `1.0`. They correspond to 10%, 50%, and 100% of each dataset's sampling target.

A scale factor changes the input table cardinalities and reference labels. It does not change the 36 query definitions.

| Dataset | Relation | 0.1 | 0.5 | 1.0 |
| --- | --- | ---: | ---: | ---: |
| IMDB | `reviews` | 5,000 | 25,000 | 50,000 |
| IMDB | `aspects` | 12 | 12 | 12 |
| BioDEX | `reports` | 500 | 2,500 | 5,000 |
| BioDEX | `terms` | 1,127 | 2,934 | 4,144 |
| FEVER | `claims` | 500 | 2,500 | 5,000 |
| FEVER | `evidence` | 287 | 1,037 | 1,478 |
| LePaRD | `citation_contexts` | 500 | 2,496 | 4,972 |
| LePaRD | `citation_passages` | 433 | 1,756 | 2,991 |
| SWE-Next | `agent_traces` | 1,772 | 8,859 | 17,711 |
| CUAD | `contracts` | 51 | 255 | 510 |
| CUAD | `contract_pages` | 1,131 | 4,522 | 9,348 |

CUAD samples contracts, so the page count at a scale factor follows from
the sampled contracts. The contract queries read the contracts of at most
32 pages: 40, 215, and 430 at the three scale factors.

Each scale factor deterministically samples upstream snapshots defined in [`quail_b/data.py`](quail_b/data.py).

## Metrics

| Metric | Definition |
| --- | --- |
| Query time / latency | Query execution time in seconds (`runtime_s`). Excludes engine startup. |
| Filter throughput | Input documents processed per second (`documents/second`). |
| Join throughput | Evaluated document pairs per second (`document pairs/second`). |
| GPU cost | Query cost in dollars (`$/query`), calculated as execution hours × GPU count × hourly price. |
| Cost per million input tokens | Query GPU cost divided by full input tokens, times 1,000,000 (`cost_usd_per_million_input_tokens`). Includes input tokens served from KV. |
| KV regret percentage | Recomputed tokens divided by fresh tokens, times 100 (`kv_regret_percent`). |
| Predicate accuracy | Agreement with reference labels on evaluated filter and join answers. |
| Output precision & recall | Precision and recall of final output rows compared to reference result rows. |
| Fresh tokens | Input token positions processed by model forward passes. |
| Input tokens | Full input lengths summed across evaluated prompts, including tokens served from KV. Each evaluated prompt counts its complete input once. Output tokens and repeated computation within a prompt are excluded. |
| Input token throughput | Input tokens divided by query execution time (`input_tokens_per_second`), excluding model startup and result collection. |
| Minimum tokens | Input token positions required under an unlimited KV prefix cache: each distinct prefix of the document requests once (a document's questions and frames share their common lead), and each pair's label, partner document, and answer cue once per pair. |
| Recomputed KV tokens | Fresh tokens minus minimum tokens (`regret_tokens`). Measures redundant KV computation. Tokens that must be computed once per request, such as a pair's suffix, are never regret. |

Input token throughput counts the prompts each method evaluated. Different filter
answers or plans can change those prompts and their total input length. It does not
measure how many token positions the GPU computed. If prompt pieces or an answer
table are missing, the input token count is unknown. If query time is zero,
input token throughput is unknown. Cost per million input tokens is unknown when
cost or the input count is missing, or the input count is zero. KV regret percentage
is unknown when regret or the fresh count is missing, or the fresh count is zero.
Both derived metrics appear in `run.json`, `measurements.parquet`, and `report.md`.

Reference labels are generated using `Qwen/Qwen3-32B-FP8`. FEVER and LePaRD also evaluate against published dataset ground truth.

## Results and CLI

Each benchmark run writes results to your configured `output_dir` (one subdirectory per evaluated query):

```text
results/my-run/
├── run.json
├── report.md
├── measurements.parquet
├── IMDB-1/
├── IMDB-2/
├── ...
└── IMDB-4/
    ├── plan.substrait
    ├── rows.parquet
    ├── filters-0.parquet
    ├── filters-1.parquet
    ├── joins-0.parquet
    └── prompt_pieces.json
```

- `run.json`: Complete execution record with parameters, configuration metadata, and per-query metrics.
- `report.md`: Markdown summary table with execution times, throughput, accuracy, and token counts.
- `measurements.parquet`: Parquet table containing one row of metrics per completed query.
- Per-query directories: Serialized Substrait plans, output rows, predicate answer tables, and token piece definitions.

Rebuild or rescore a report from saved results:

```sh
quail-b report results/my-run
```

## Source Files

| Path | Contents |
| --- | --- |
| [`quail_b/plans/`](quail_b/plans/) | Substrait 0.103 query plans and workload catalog |
| [`quail_b/queries.py`](quail_b/queries.py) | Query specifications and query loader |
| [`quail_b/substrait.py`](quail_b/substrait.py) | Substrait plan parser and operator extraction |
| [`quail_b/substrait_extensions.yaml`](quail_b/substrait_extensions.yaml) | Declarations for `ai_filter` and `ai_join` extensions |
| [`quail_b/prompts.py`](quail_b/prompts.py) | Prompt templates and prompt identifiers |
| [`quail_b/rendering.py`](quail_b/rendering.py) | Exact prompt text rendering logic |
| [`quail_b/data.py`](quail_b/data.py) | Dataset tables, sampling logic, and scale factors |
| [`quail_b/cuad.py`](quail_b/cuad.py) | CUAD contract PDFs as file-backed relations and their page annotations |
| [`quail_b/labels.py`](quail_b/labels.py) | Reference labels and ground truth loading |
| [`quail_b/run.py`](quail_b/run.py) | Benchmark runner and answer validator |
| [`quail_b/scoring.py`](quail_b/scoring.py) | Accuracy, precision, recall, and cost scoring |
| [`quail_b/minimum.py`](quail_b/minimum.py) | Prefix trie and minimum token accounting |
| [`quail_b/reporting.py`](quail_b/reporting.py) | Markdown reports and Parquet measurement export |
