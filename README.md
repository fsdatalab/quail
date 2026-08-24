# Quail

Making AI SQL filter and join queries fast by changing how the
inference layer manages KV memory.

Three things live here:

- **`plans/engine_design.md`** — the settled design for the Quail
  engine: a declarative query engine for AI_FILTER and AI_JOIN
  (Snowflake AISQL syntax, one packed executor, Modal runtime).
- **`quail/`** — the engine built from that document: the `quail`
  package (planner, executor, runtime, SQL front end and builder),
  the GPU test cells, the committed result summaries, and the
  reports. Its own README explains the layout;
  `quail/reports/engine-wiki.md` is the living design reference.
- **`old_exploration/`** — the exploration that produced the design.
  Frozen as evidence; the design document cites its result files by
  path.

Working conventions — writing style, experiments, reports, plots,
issue and PR descriptions — are in `AGENTS.md`.

To run the engine's CPU tests, work from inside `quail/`:

```bash
cd quail
uv run pytest tests/ -q     # CPU tests, no GPU
```

Engine runs go through Modal (Qwen3 4B or 32B fp8, one H100 per
model copy); see `quail/README.md`.
