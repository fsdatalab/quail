# Quail

A query engine for AI_FILTER and AI_JOIN over document collections.
Filter queries and joins only. The models are Qwen3 4B fp8 and Qwen3
32B fp8, one H100 per model copy, and workers run through Modal.

## Layout

- `quail/specs/` has the model and device structs. Every planner
  input comes from them.
- `quail/planner/` has the budget arithmetic, the physical plan,
  and the planner decisions (stage order, anchor choice,
  sharding). KV is always bf16.
- `quail/catalog.py`, `quail/logical.py`, `quail/sqlfront/`, and
  `quail/builder.py` are the providers, logical operators, and the
  two query entry points (AI SQL and the builder API).
- `quail/executor/` is the packed executor: chunk packing, admission,
  the paged KV arena, attention kernels, the overlapped loop, and
  weight loading. The GPU parts run only inside the Modal image.
- `quail/runtime/` is the run side. It contains the session, compute provider,
  multi-GPU coordinator, and Modal worker.
- `quail/bench/` has the QUAIL-B benchmark queries. Its
  [README](quail/bench/README.md) explains how to run the benchmark and
  label a new predicate.
- `ablations/` and `baselines/` have experiment entry points and the
  stock vLLM comparison code.
- `reports/` has experiment reports and plots. `results/` has the
  committed summaries that those reports read.
- `plans/` has the current extensible design and the archived original design.
- `docs/` is the documentation site (Fumadocs). See its README to run
  it locally.
- `tests/` has CPU tests. `tests/gpu/` has the Modal GPU cells -
  milestone gates, smokes, and benchmarks - which cost GPU time and
  run only when invoked explicitly.

## Setup and tests

```
uv sync
uv run pytest
uv run ruff check quail tests ablations \
  reports/make_sol_quailb.py reports/make_quailb_eval_plots.py
uv run vulture
```

GPU cells (Modal, H100). Tee output to a file per house rule:

```
uv run modal run tests/gpu/milestone1.py::run_probe 2>&1 | tee results/m1_probe.log
```

Committed result summaries (the JSON files reports cite) are in
`results/`; raw per-item run records live on the `quail-results`
Modal volume. Teed logs stay local and are not committed.

## Using Quail

Quail is the default model backend. A normal user does not select it:

```python
import quail

session = quail.Session()
```

The user selects a built in model through `EngineConfig` when the default
Qwen3 4B fp8 model is not the one they want:

```python
session = quail.Session(quail.EngineConfig(model="qwen3-32b-fp8"))
```

The benchmark backends use the same query and result interface:

```python
stock = quail.Session(quail.EngineConfig(backend="stock_vllm"))
pipelined = quail.Session(quail.EngineConfig(backend="pipelined_vllm"))
sglang = quail.Session(quail.EngineConfig(backend="pipelined_sglang"))
```

These are separate model backends. They do not call the Quail executor.

Snowflake SQL uses `AI_FILTER`. BigQuery SQL uses `AI.IF` and passes
`dialect="bq"` to `Session.sql`. Both forms compile to the same logical
filter node.

### Remote document sources

Quail reads remote Parquet and Hugging Face documents on Modal. The client
parses the SQL and sends the logical plan plus the source description. The
Modal worker reads bounded document batches and writes tokenized documents to
a temporary Arrow file. It plans and runs the query from that file, then
returns the projected rows.

```python
import modal
import quail

compute = quail.ModalComputeProvider(secrets=(
    modal.Secret.from_name("quail-s3"),
))
session = quail.Session(compute_provider=compute)
session.register(
    "documents",
    quail.DocumentProvider.from_parquet(
        "s3://my-bucket/documents.parquet",
        id_col="document_id",
    ),
)

result = session.sql("""
    SELECT d.document_id
    FROM documents d
    WHERE AI_FILTER(
        PROMPT('Does {0} describe an adverse event?', d.text)
    )
""").run()
```

`from_parquet` reads the schema and file list on the client so Quail can check
the SQL column names. It does not read the document rows. The named Modal
secret must provide any credentials the Modal worker needs to open the source.

An in memory Arrow table has no remote location. `ModalComputeProvider` reads
only the columns used by the query and sends those raw Arrow columns. The
Modal worker tokenizes the documents and creates the physical plan.

The worker does not keep the complete tokenized corpus in Python or Arrow heap
memory. It reads token values from a memory mapped Arrow file when documents
enter a model chunk. The same file stores the output columns, so final
projection does not scan a remote source a second time. The temporary file must
fit on the Modal worker's local disk.

## Adding an engine extension

An extension is an importable Python module with this function:

```python
def register_quail_extension(registry):
    registry.register_codec(...)
    registry.register_runtime(...)
    registry.register_physical_rule(...)
```

The function can register logical rules, physical planners, physical rules,
model backends, models, devices, physical node codecs, physical node runtimes,
and execution observers. Load the module before creating the session:

```python
import quail

registry = quail.ExtensionRegistry.with_built_ins()
registry.load_extension(
    "my_package.quail_extension",
    local_python_sources=("my_package",),
    pip_packages=("another-dependency==1.2.3",),
)
session = quail.Session(registry=registry)
```

The default `ModalComputeProvider` copies `local_python_sources` into the
existing `quail-engine` image and installs `pip_packages`. The physical plan
names the extension modules it needs. The Modal worker imports those modules
before it plans the query or selects a runtime.

`ModalComputeProvider` calls a Modal Function in the existing `quail-engine`
app. Quail does not run an application server. One Modal container receives
1, 2, 4, or 8 H100s, and it runs one model copy per H100.

[`quail_ext_examples/plan_trace.py`](quail_ext_examples/plan_trace.py) is a
complete execution observer. It records the input and output row counts for
each physical node. An observer can measure an existing plan without adding a
node that changes the plan.

A model backend decides whether it supports a model and device. It proposes a
physical plan, creates one model execution object per GPU, and executes the
physical plan inside the compute worker. Physical requests and responses are
internal to the worker.

Modal is the default compute provider. A compute provider implements one
method:

```python
class ComputeProvider(Protocol):
    def execute(self, request: QueryRequest) -> QueryResult:
        ...
```

`QueryRequest` contains the logical plan, table providers, model settings, and
registered extensions. `QueryResult` contains the final Arrow rows and the
execution report. A user can pass another provider through
`Session(compute_provider=provider)`.

`ModalComputeProvider` handles data location internally. It sends a remote
source description when the Modal worker can open the source. It sends only
the needed raw Arrow columns when the source exists only in the client
process. The Modal worker writes tokenized input batches to a temporary memory
mapped Arrow file, creates the physical plan, and runs the selected model
backend. The provider
keeps one function container running until `Session.close()` so several queries
can reuse the loaded model. Closing the session allows the function to scale to
zero.

A table provider does not need registry entry. It implements `TableProvider`
and is passed directly to `Session.register(name, provider)`.

## Adding a new model

There are two steps, and one constraint to know about.

First, add a spec struct in `quail/specs/` (about 15 lines). Every
field except `params` and `w_mem_bytes` comes from the model's HF
`config.json`. Register the spec in `quail/specs/__init__.py`.

Second, know that the executor's fused kernels assume the Qwen3
architecture: QK-norm before rope, gated SiLU MLP, and fp8
block-quantized weights. Another Qwen3-family fp8 checkpoint works
without changes. A different model family needs kernel-path changes
first.
