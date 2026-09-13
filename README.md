# QUAIL-B

QUAIL-B is an academic benchmark of 33 AI SQL queries over document tables.
AI SQL means SQL queries with predicates answered by a language model.

## Contents

- [Scope](#scope)
- [Query plans](#query-plans)
- [Scale factors](#scale-factors)
- [Installation](#installation)
- [Metrics](#metrics)
- [Adapter interface](#adapter-interface)
  - [Input](#input)
  - [Output](#output)
- [Running the benchmark](#running-the-benchmark)
- [Data access and memory](#data-access-and-memory)
- [Results](#results)
- [Source definitions](#source-definitions)

## Scope

The benchmark supports two AI SQL operations:

- **AI FILTER:** ask a boolean question about one document and keep the
  documents answered `TRUE`.
- **AI JOIN:** ask a boolean question about two documents and keep the
  pairs answered `TRUE`.

QUAIL-B is not a SQL engine. It provides engine-independent query
specifications, Arrow input tables, and reference answers. An adapter is
the integration function that executes each specification with an engine
and returns the result. QUAIL-B validates, scores, and saves the run.

For each query, the benchmark reports accuracy, query time, throughput,
GPU cost, and token work.

## Query plans

Each `QuerySpec` uses a
[`substrait.Plan`](https://substrait.io/serialization/binary_serialization/).
This plan is the authoritative query representation. Substrait is a standard
protocol-buffer format for relational query plans. `query.plan` returns the
parsed plan. `query.plan_bytes` contains the deterministic binary serialization
produced by the pinned Substrait 0.103 bindings.

The query definitions are standard Substrait ProtoJSON files under
[`quail_b/plans/`](quail_b/plans/). The benchmark reads these files; it has
no other query model. Because one prompt appears in several plans, the files
are written by [`tools/make_substrait_plans.py`](tools/make_substrait_plans.py)
from the query shapes and the prompt strings in `quail_b/prompts.py`, and a
test checks that the checked-in files equal the script's output.

Protobuf does not define a canonical byte format across runtime versions.
The saved `definition_hash` therefore resolves function anchors and hashes the
validated relations, operators, prompts, and projection. It does not hash the
raw plan bytes.

QUAIL-B uses standard Substrait relations:

| Query operation | Substrait representation |
| --- | --- |
| Read a document table | `ReadRel` with a `NamedTable` |
| Apply an AI filter | `FilterRel` whose condition calls `ai_filter` |
| Apply an AI join | inner `JoinRel` whose condition calls `ai_join` |
| Apply an ordinary join condition | standard `equal`, combined with `and` |
| Select output IDs | `ProjectRel` and `RelRoot` |

The AI functions are defined in
[`quail_b/substrait_extensions.yaml`](quail_b/substrait_extensions.yaml):

- `ai_filter:str_str(prompt, document) -> boolean`
- `ai_join:str_str_str(prompt, left, right) -> boolean`

The extension URN is
`extension:org.fsdatalab.quail_b:functions_ai`. Ordinary equality and
boolean conjunction use the standard Substrait comparison and boolean
extension URNs.

See the [PDF of all 33 query plans](figures/quailb_anatomy.pdf). It groups
queries by dataset. Gray nodes are table scans, blue nodes are AI FILTER,
and orange nodes are AI JOIN. Percentages are the fixed planning
selectivity estimates from the 0.1-scale reference labels.

## Scale factors

The `scale_factor` parameter selects one published corpus. The supported
values are `0.1`, `0.5`, and `1.0`. They represent 10%, 50%, and 100% of
each dataset's full sampling target. The selected scale changes the input
tables and reference answers. It does not change the 33 query definitions,
prompts, or fixed planning selectivity estimates.

The fraction applies to the primary sampling unit for each dataset: IMDB
reviews, BioDEX reports, FEVER claims, LePaRD positive citation pairs, and
SWE-Next trace snapshots. Fixed and derived tables do not necessarily grow
in exact proportion. For example, IMDB has 12 fixed aspects, while BioDEX
terms and FEVER evidence pages are derived from the documents selected at
each scale.

The published table sizes are:

| Dataset | Input table | 0.1 rows | 0.5 rows | 1.0 rows |
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

Every value selects a fixed corpus built from pinned source revisions and a
fixed sampling seed. A run records both the scale factor and exact corpus ID.
Compare benchmark results only when these values match.

## Installation

Python 3.12.

```sh
uv add "quail-b @ git+https://github.com/fsdatalab/quail-bench.git"
```

## Metrics

- **Accuracy:** agreement with saved predicate answers, plus precision and
  recall of the final rows.
- **Query performance:** query time, documents or pairs per second, and GPU
  cost when a price is supplied.
- **Token work:** how many input token positions the model computed, the
  minimum those requests required, and how many KV tokens were recomputed.

The three token counts are:

- `fresh_tokens`: all input token positions processed by model forward
  passes. Repeated computation is counted again. Your engine reports this.
- `minimum_tokens`: the fewest input token positions the same requests need
  if every shared prefix stays in KV. QUAIL-B computes this.
- `regret_tokens`: `fresh_tokens - minimum_tokens`. This is the reusable
  document or anchor KV that the engine computed again. QUAIL-B computes it.

KV is the model's key-value cache for previously computed tokens.

To compute the last two counts, QUAIL-B needs the exact request prefixes.
The adapter returns `fresh_tokens` and `prompt_pieces`. `prompt_pieces`
contains the tokenizer name and the token IDs placed around each document:

```python
prompt_pieces = {
    "tokenizer": "Qwen/Qwen3-4B-FP8",
    "preamble": [...],
    "filters": [
        {"id": "filter-1", "tail": [...]},
        {"id": "filter-2", "tail": [...]},
    ],
    "joins": [
        {
            "id": "join-1",
            "anchor": "r",
            "frame": [...],
            "label": [...],
            "tail": [...],
        },
    ],
}
```

All values represented by `[...]` are lists of token IDs. For a filter
request, the order is `preamble`, document, `tail`. For a join request,
the order is `preamble`, anchor document, `frame`, `label`, partner
document, `tail`. See `quail_b.minimum.validate_prompt_pieces`.

If an engine does not provide these values, accuracy and query performance
still score, but its token counts are unavailable.

## Adapter interface

### Input

QUAIL-B calls `run_query(query, tables)` once per query:

- `query` is a `QuerySpec`. Its `plan` field is a Substrait `Plan`.
- `query.id` and `query.description` identify the benchmark query.
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

The IMDB-4 Substrait relation tree is:

```text
RelRoot [r, a]
└── ProjectRel [r.id, a.id]
    └── JoinRel INNER [ai_join(J1, r.body, a.aspect)]  hint.alias=join-1
        ├── FilterRel [ai_filter(F4, r.body)]           hint.alias=filter-2
        │   └── FilterRel [ai_filter(F1, r.body)]       hint.alias=filter-1
        │       └── ReadRel reviews [id, body]          hint.alias=r
        └── ReadRel aspects [id, aspect]                hint.alias=a
```

`F1`, `F4`, and `J1` stand for the full prompt string literals in the
plan. The exact strings are also defined in `quail_b/prompts.py`.

Substrait field references are numeric positions. The `ReadRel` schemas
define those positions. Every table has an `id` column. The column an AI
function reads is the one its field reference points at.

Relation aliases and operator IDs use the standard Substrait
`RelCommon.hint.alias` field: `r` on a `ReadRel`, `filter-1` on a
`FilterRel`, `join-1` on a `JoinRel`. The plans carry no other metadata.
Operator IDs number filters and joins in post-order (inputs before the
operator, left before right). They are the keys of the answer tables an
adapter returns.

An adapter reads the relation tree, resolves function anchors through the
plan's extension declarations, and translates the result to its engine.
Filters are `FilterRel` nodes over their input relation. They are not
properties of an alias.

For this query, `tables["reviews"]` has `id` and `body` columns.
`tables["aspects"]` has `id` and `aspect` columns.

### Output

The adapter translates the `QuerySpec` into the engine's AI SQL, executes
it, and returns a `RunOutput`:

```python
import pyarrow as pa
import quail_b


def run_query(query, tables):
    result = execute_with_your_engine(query, tables)

    return quail_b.RunOutput(
        filter_answers={
            "filter-1": pa.table({
                "r": result.f1_ids,
                "answer": result.f1_answers,
            }),
            "filter-2": pa.table({
                "r": result.f4_ids,
                "answer": result.f4_answers,
            }),
        },
        join_answers={
            "join-1": pa.table({
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
        measurements={"fresh_tokens": result.fresh_tokens},
        prompt_pieces=result.prompt_pieces,
    )
```

`execute_with_your_engine` is the function you supply. It is not part of
QUAIL-B.

The returned fields mean:

- `filter_answers["filter-1"]` has every document evaluated by the first
  filter, plus the model's boolean answer.
- `filter_answers["filter-2"]` has every document that reached the second
  filter, plus its answer.
- `join_answers["join-1"]` has every pair evaluated by the join, plus its
  answer.
- `rows` has the final query result. It has one ID column per alias in
  `query.select`.
- `runtime_s` is query execution time. It excludes model startup,
  result collection, scoring, and saving.
- `measurements["fresh_tokens"]` is the input work measured by the engine.
- `prompt_pieces` lets QUAIL-B compute the minimum and recomputed KV tokens.

Operator IDs are unique within a query and do not change when an engine
reorders execution. Pass `None` for the answer dictionaries if the engine
did not record individual predicate answers. QUAIL-B can still score final
output precision and recall.

## Running the benchmark

Pass the adapter to `quail_b.run`:

```python
quail_b.run(
    run_query,
    queries=["IMDB-4"],
    scale_factor=0.1,
    output_dir="results/my-run",
    metadata={"engine": "my-engine", "model": "Qwen3-4B-FP8"},
)
```

## Data access and memory

By default, `quail_b.run` reads the corpus and reference labels anonymously
from the public `s3://quail-bench` bucket. The first run downloads immutable
Parquet and manifest files to `~/.cache/quail-b`, or
`$XDG_CACHE_HOME/quail-b`. Later runs reuse those files. The small pointer
to the active reference collection is refreshed from S3.

Before it calls your adapter, the current loader puts the selected Arrow
input tables and the full reference-label collection in host memory. The
full label collection is loaded even when `queries` selects one query.
Budget:

| Scale factor | Reference answers | Loader peak RAM | Host RAM to use |
| ---: | ---: | ---: | ---: |
| 0.1 | 1.21 million | 0.84 GiB measured | 2 GiB or more |
| 0.5 | 17.62 million | 10–12 GiB estimated | 16 GiB or more |
| 1.0 | 51.80 million | 30–35 GiB estimated | 48 GiB or more |

The 0.5 and 1.0 estimates scale the measured 0.1 label-memory cost by the
published answer counts. They are planning values, not measured peaks.
They exclude your engine, model, and returned result tables. Use 64 GiB
for a full-scale run when the engine shares the same host.

- Override the download cache with `cache_dir=` or `QUAIL_B_CACHE_DIR`.
- Pass `data_dir=` to use local input Parquet files. Reference labels still
  come from S3 unless you pass a local published-data mirror as `root=`.
- `gpu_count=` and `gpu_hourly_rate_usd=` add GPU cost.
- An existing `output_dir` is never overwritten.

## Results

`quail_b.run` writes `results/my-run/`:

```
results/my-run/run.json
results/my-run/IMDB-4/plan.substrait
results/my-run/IMDB-4/rows.parquet
results/my-run/IMDB-4/filters-0.parquet
results/my-run/IMDB-4/filters-1.parquet
results/my-run/IMDB-4/joins-0.parquet
results/my-run/report.md
results/my-run/measurements.parquet
```

`report.md` includes query time, throughput, GPU cost, predicate accuracy,
final-output precision and recall, input rows, and token work.

For example, these are results from Quail on one H100 with Qwen3 4B fp8
at scale 0.1:

| Dataset | Query | Query time, seconds | Recomputed KV tokens |
| --- | --- | ---: | ---: |
| IMDB | IMDB-4 | 17.38 | 118,395 |
| BioDEX | BIO-3 | 79.96 | 2,275,033 |
| FEVER | FEV-10 | 1.65 | 2,927 |
| LePaRD | LEP-6 | 9.39 | 95,362 |
| SWE-Next | AGENT-1 | 237.65 | 11,886,152 |

Filter throughput is input documents per second. Join throughput is
evaluated pairs per second across all stages. Predicate accuracy is
agreement on evaluated answers; that count can differ between engines.
Labels are Qwen3 32B fp8; FEVER and LePaRD also use source labels.

Rebuild the report from saved answers:

```sh
quail-b report results/my-run
```

## Source definitions

Query plans: [`quail_b/plans/`](quail_b/plans/).
Plan generator: [`tools/make_substrait_plans.py`](tools/make_substrait_plans.py).
Catalog loader: [`quail_b/queries.py`](quail_b/queries.py).
Substrait extension: [`quail_b/substrait_extensions.yaml`](quail_b/substrait_extensions.yaml).
Tables: [`quail_b/data.py`](quail_b/data.py).
