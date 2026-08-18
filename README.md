# Quail

Making AI SQL filter and join queries fast by changing how the
inference layer manages KV memory.

Two things live here:

- **`plans/engine_design.md`** — the settled design for the Quail
  engine: a declarative query engine for AI_FILTER and AI_JOIN
  (Snowflake AISQL syntax, one packed executor, Modal runtime).
  The engine itself will be built in its own repository from that
  document.
- **`exploration/`** — the exploration that produced the design:
  the `quail` package (planner, cost model, vLLM extensions,
  clients), the Modal experiments, the committed results, and the
  CPU test suite. Its own README explains the workload and the
  measured findings. Frozen as evidence; the design document
  cites its result files by path.

To run the exploration code, work from inside `exploration/`:

```bash
cd exploration
pip install -e ".[dev]"
python -m pytest tests/ -q     # CPU tests, no GPU
```

Engine experiments need Modal (one H100); see
`exploration/README.md`.
