# Extensible engine interfaces

Quail now plans queries through interfaces that a session can extend.
The change keeps the query language limited to model filters and joins.

The main changes are:

* SQL and Python use one `LogicalPlanBuilder` and the same registered
  logical node types.
* A session can register logical rules, physical rules, model backends,
  physical node codecs, physical node runtimes, and table providers.
* The planner returns an immutable typed `PhysicalGraph` with typed ports,
  schemas, execution locations, partitioning, and resource requirements.
* Quail represents runtime join planning with `AdaptiveJoinPlan`. Its
  expected child plan uses `AnchoredJoin` and `Exchange`. The typed plan has
  no barrier node.
* The session sends a versioned JSON plan envelope. The worker checks the
  plan version, backend, node codecs, model, and GPU count before a model
  call.
* Table providers return bounded Arrow batches and read only requested
  columns. Arrow datasets, Parquet, Hugging Face datasets, and in memory
  tables implement the same interface.
* Qwen3 4B fp8 and Qwen3 32B fp8 use the same plan interfaces. A query can
  use 1, 2, 4, or 8 H100s in one Modal container, with one model copy on
  each H100.

The CPU suite checks built in behavior and extension behavior. One extension
test registers a model backend, physical planner, physical rule, physical
node, codec, and runtime without changing generic planner or runner code.
Another test checks a custom logical node and logical rule.

The Modal confirmation preserved the row counts and measured model work.
BIO-2 took 128.80 seconds, compared with 129.64 seconds on main. AGENT-1 took
238.13 seconds, compared with 235.09 seconds on main. BIO-2's sorted final
result hash matched main. Both queries matched the main fresh token counts,
KV counts, and KV regret. A small 4B query on two GPUs and a small 32B query
on one GPU also finished through the same typed interfaces. The full
measurements are in
`reports/2026-08-31-extensible-engine-confirmation.md`.
