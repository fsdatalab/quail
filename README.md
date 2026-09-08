# Quail

A query engine for AI_FILTER and AI_JOIN over document collections.
Filter queries and joins only. The models are Qwen3 4B fp8 and Qwen3
32B fp8, one GPU per model copy. H100 execution is tested; RTX PRO 6000
Blackwell Server Edition has planning and kernel selection support, with
GPU validation pending. A query runs on the GPU in the
calling process, or on Modal.

## Layout

- `quail/` is the package. `logical.py`, `sqlfront/`, and `builder.py`
  are the front ends. `planner/` is planning and the cost model.
  `physical/` is the typed physical graph. `backends/` holds the backend
  interface, the Quail backend under `backends/quail/`, and the vLLM and
  SGLang request backends. `executor/` is the GPU code and runs where
  the model runs. `runtime/` is the session, compute providers,
  generic runner, token store, results, and the Modal worker. `bench/`
  is the QUAIL-B runner for Quail and its request backends.
- QUAIL-B, the benchmark, is its own repository:
  [fsdatalab/quail-b](https://github.com/fsdatalab/quail-b). It
  holds the document sets, prompts, queries as data, the predicates and
  their label identities, saved labels, and scoring, and runs no engine
  or model. Corpus and labels are public in the `quail-bench` S3
  bucket. It is installed here as the `quail_b` package, pinned in
  `pyproject.toml`. The pass that writes the labels is
  `quail/bench/labeling.py`, one GPU; `quail/bench/judge_pass.py` runs
  it on Modal.
- `demos/` has runnable examples for a machine with a GPU:
  `local_gpu_smoke.py` runs one filter over four short documents, and
  `imdb_ending_filter.py` filters all 100,000 IMDB reviews.
- `quail_ext_examples/` has example extensions.
- `tests/` is the CPU suite. It runs in seconds and needs no GPU.
- `experiments/` holds every Modal entry point that costs GPU time: the
  ablation and profiling scripts at the top level, and the smokes,
  probes, and gates under `experiments/cells/`. They run only when
  invoked.
- `reports/` has experiment reports, their plot scripts and PNGs,
  `shipped_features/`, and `engine-wiki.md`. Experiment data is not
  committed: reports cite it by its path on the `quail-results` volume.
- `docs/` is the documentation site, including the design decisions
  and the speed of light model under its architecture section.

## Setup and tests

```
uv sync
uv run pytest
uv run ruff check quail tests experiments reports
uv run vulture
```

GPU cells (Modal, H100). Tee output to a file per house rule:

```
uv run modal run experiments/cells/session_smoke.py 2>&1 | tee results/session_smoke.log
```

Run records live on the `quail-results` Modal volume. Teed logs go
under `results/`, which is local and not committed.

## Using Quail

Quail is the default model backend. A normal user does not select it:

```python
import quail

session = quail.Session()
```

`Session()` runs the model in the calling process. That process needs
a CUDA GPU and vLLM, which the package installs on Linux. On a machine
without a GPU, pass the Modal compute provider and the same query runs
in a Modal Function:

```python
session = quail.Session(compute_provider=quail.ModalComputeProvider())
```

The user selects a built in model through `EngineConfig` when the default
Qwen3 4B fp8 model is not the one they want:

```python
session = quail.Session(quail.EngineConfig(
    model="qwen3-32b-fp8", gpus=1, device="h100-sxm",
))
```

On an RTX PRO 6000 Blackwell Server Edition host, select its hardware
specification explicitly. For a host with eight GPUs:

```python
session = quail.Session(quail.EngineConfig(
    device="rtx-pro-6000-blackwell-server", gpus=8,
))
```

This runs one model copy per GPU on that host. The Modal provider supports
H100 configurations only.

Try the four-document filter on one RTX GPU:

```sh
uv run python demos/local_gpu_smoke.py --device rtx-pro-6000-blackwell-server 2>&1 | tee rtx-smoke.log
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

Register objects on the session's registry, the way a DataFusion
`SessionStateBuilder` takes rules and providers:

```python
import quail

from my_package.nodes import MyNode, MyNodeRuntime, PreferAllDocuments

registry = (
    quail.ExtensionRegistry.with_built_ins()
    .register_node(MyNode, runtime=MyNodeRuntime())
    .register_physical_rule(PreferAllDocuments())
)
session = quail.Session(
    registry=registry,
    compute_provider=quail.ModalComputeProvider(
        local_python_sources=("my_package",),
    ),
)
```

The registry takes logical rules, physical planners, physical rules, model
backends, models, devices, physical nodes, remote source readers, and execution
observers. Names come from the objects. `register_node` adds a node's codec
and runtime together. The lookup tables are read-only; add objects through
the registration methods.
A package can also expose one entry point, `register_quail_extension(registry)`,
and be loaded with `registry.load_extension(module)`. If loading fails, the
registry stays unchanged.

The query sends the registry to Modal as a Python object. The worker uses
that same registry for source preparation, planning, and execution.
`load_extension` runs when called; module registration functions are not
replayed during execution.

Set `local_python_sources` and `pip_packages` on `ModalComputeProvider` to
copy extension code and install its dependencies. Quail does not infer these
packages from registered objects. The worker already includes `quail`.

For registrations that create objects inside the worker, pass
`initialize_worker=callback` to the provider. Quail calls `callback(registry)`
once per query, before opening sources. The returned `QueryResult` contains
the materialized Arrow rows, executed plan, and node metrics.

The built-ins are Quail's default model and device specifications, backends,
node implementations, and source readers. Registering them does not load
model weights or start GPU processes.

`ModalComputeProvider` calls a Modal Function in the existing `quail-engine`
app. Quail does not run an application server. One Modal container receives
1, 2, 4, or 8 H100s, and it runs one model copy per H100.

[`quail_ext_examples/cost_ledger.py`](quail_ext_examples/cost_ledger.py)
charges each query's GPU seconds, tokens, and dollars to the physical nodes
that used them, for chargeback or a cost dashboard. It reads
`result.plan` and `result.node_metrics`, which every finished query carries,
and registers nothing; `result.explain(verbose=True)` prints the same numbers.

`query.explain()` prints logical and physical operator trees. Predicates in
physical filter nodes appear in execution order. `estimated_rows` means output
rows, after filtering or a limit. An unavailable output estimate is `unknown`;
join evaluation counts are labeled separately. Quail plans also show the model,
chunk and admission budgets in tokens, KV rewind, retention for joins, and
anchor KV reuse. A retained survivor percentage is a planning estimate.

Use `query.explain(verbose=True)` for node ids, ports, stage estimates, and KV
settings. After execution, `result.explain()` shows actual output rows and time
per node. `result.explain(verbose=True)` includes the full recorded metrics.

A model backend decides whether it supports a model and device. It proposes a
physical plan, creates one model execution object per GPU, and executes the
physical plan inside the compute worker. Physical requests and responses are
internal to the worker.

`InProcessComputeProvider` is the default compute provider. It runs the
model in the calling process, so that process needs a CUDA GPU and the
backend's runtime package. Without a GPU it raises a `RuntimeError` that
names `ModalComputeProvider`, which runs the same query on Modal. A
compute provider implements one method:

```python
class ComputeProvider(Protocol):
    def execute(self, request: QueryRequest) -> QueryResult:
        ...
```

`QueryRequest` contains the logical plan, table providers, model settings, and
the registry. `QueryResult` contains the final Arrow rows and the
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
