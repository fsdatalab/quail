# Extensible engine interfaces

Quail now has clear extension points for query planning, physical execution,
models, compute, and measurement. The query language remains limited to model
filters and joins.

The main changes are:

* `Session()` uses the Quail backend by default. Users only name a backend when
  they install a different one.
* Users select the model through `EngineConfig`. Extensions can register model
  and device specifications.
* SQL and Python use one `LogicalPlanBuilder` and the same logical node types.
* The planner returns an immutable `PhysicalGraph` with typed input and output
  ports.
* The generic `PhysicalPlan` contains backend-owned settings. Custom backends
  no longer fill Quail scheduler fields.
* `DocumentInput` identifies a named physical input. It does not contain a
  Quail table provider or column name.
* `QueryRequest` contains the logical plan, table providers, model settings,
  and registered extensions.
* A compute provider implements `execute(request)` and `close()`.
  `ModalComputeProvider` is the default provider and calls a Modal Function.
* The physical request and response stay inside the compute worker. Model
  outputs do not cross the compute provider boundary.
* A model backend checks support, proposes physical plans, creates model
  execution state, and runs the standard request in a compute process. It no
  longer prepares user inputs or assembles the public query result.
* A session can register logical rules, physical planners, physical rules,
  model backends, models, devices, physical node codecs,
  physical node runtimes, remote source readers, and execution observers.
  Table provider instances are passed directly to `Session.register`.
* `quail_ext_examples.plan_trace` shows how an extension observes the existing
  physical plan without adding a node that changes query behavior.
* Snowflake `AI_FILTER` and BigQuery `AI.IF` compile to the same logical
  operator.

The logical request and internal physical plan name the extension modules they
need. The worker rebuilds the registry and checks the physical node codecs,
backend, model, device, GPU count, and runtimes before model execution. Quail
does not keep separate request, response, and physical plan version globals.

Physical node codecs have one executable representation. Nodes expose display
fields separately for `explain()`.

The generic runner now finishes the complete physical graph. GPU model outputs
are supplied as completed node outputs. The same run then executes
`HashJoin`, `Project`, and `Limit`. Execution observers see every physical node
once.

A DataFusion integration can keep its scans and ordinary relational operators
in DataFusion. Its custom operator can wrap the needed Arrow columns in a Quail
table provider and submit a logical query. A Rust client would need a public
cross-language interface because the current Modal provider is a Python
interface.

The CPU tests cover built in planning and execution, registered source readers,
custom model lookup, custom backend dispatch, physical node
and rule registration, full graph observation, typed token requests, Arrow
responses, and both supported SQL spellings.
