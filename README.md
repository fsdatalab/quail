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
- `quail/runtime/` is the run side: the session, compute provider,
  multi-GPU coordinator, and Modal worker.
- `quail/bench/` has the QUAIL-B benchmark queries. Its
  [README](quail/bench/README.md) explains how to run the benchmark and
  label a new predicate.
- `ablations/` and `baselines/` have experiment entry points and the
  stock vLLM comparison code.
- `reports/` has experiment reports and plots. `results/` has the
  committed summaries that those reports read.
- `plans/` has the original engine design.
- `tests/` has CPU tests. `tests/gpu/` has the Modal GPU cells -
  milestone gates, smokes, and benchmarks - which cost GPU time and
  run only when invoked explicitly.

## Setup and tests

```
uv sync
uv run pytest
```

GPU cells (Modal, H100). Tee output to a file per house rule:

```
uv run modal run tests/gpu/milestone1.py::run_probe 2>&1 | tee results/m1_probe.log
```

Committed result summaries (the JSON files reports cite) are in
`results/`; raw per-item run records live on the `quail-results`
Modal volume. Teed logs stay local and are not committed.

## Adding an engine extension

An extension is an importable Python module with this function:

```python
def register_quail_extension(registry):
    registry.register_codec(...)
    registry.register_runtime(...)
    registry.register_physical_rule(...)
```

The function can register logical rules, physical planners, physical rules,
model backends, physical node codecs, and physical node runtimes. Load the
module before creating the session:

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
before it decodes the plan or selects a runtime.

[`quail_ext_examples/count_documents.py`](quail_ext_examples/count_documents.py)
is a complete physical node, codec, runtime, and plan rule. Its node runs in
the remote graph and reports a document count without changing query results.

A model backend implements planning plus three process boundary methods.
`prepare` builds its request on the client. `execute_remote` runs inside the
compute process. `assemble` builds the public result on the client. The backend
owns its scheduler, model calls, and KV.

Modal is one compute provider. To use another provider, pass an object with
`execute(payload, gpu_count, extensions)` and `close()` methods as
`Session(compute=provider)`. Planning and backend code do not call Modal.

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
