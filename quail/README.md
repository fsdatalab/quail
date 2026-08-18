# Quail

A declarative query engine for AI_FILTER and AI_JOIN queries over
document collections. Filter queries and joins only; Qwen3 4B fp8 is
the first model, H100 workers on Modal the first device.

This directory is the clean engine build described in
`../plans/engine_design.md`. It copies logic out of `../exploration/`
but never imports it. It will move to its own repository later.

Layout:

- `quail/specs/` - model and device structs; every planner input
  derives from them.
- `quail/planner/` - budgets (spec arithmetic), calibration loading,
  the physical plan, and the decisions (order, anchor, sharding,
  access, KV dtype).
- `quail/calibration/` - the measured constants per (model, device):
  a, a2, q_kv, plus the host channel bandwidth table.
- `quail/catalog.py`, `quail/logical.py`, `quail/sqlfront/`,
  `quail/builder.py` - providers, the four logical operators, and the
  two query entry points (AI SQL and the builder).
- `quail/executor/` - the packed executor: chunk packing and
  admission, the paged KV arena, attention kernels, the overlapped
  loop, weight loading. GPU parts run only inside the Modal image.
- `tests/` - CPU tests. `tests/gpu/` holds the milestone 1 Modal
  cells; they cost GPU time and run only when invoked explicitly.

Setup and tests:

    cd quail
    uv sync
    uv run pytest

GPU cells (Modal, H100; tee output to a file per house rule):

    uv run modal run tests/gpu/milestone1.py::run_probe 2>&1 | tee results/m1_probe.log

Milestone 1 results live in results/ (JSON + teed logs) and in the
quail-results Modal volume under m1/.
