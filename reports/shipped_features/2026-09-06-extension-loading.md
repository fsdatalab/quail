# Extensions as objects, metrics on the executed plan

Date: 2026-09-06.

## What changed

- **Registration takes objects.** `register_logical_rule(rule)`,
  `register_physical_planner(planner)`, `register_physical_rule(rule)`,
  `register_runtime(runtime, key=...)`, `register_source_reader(reader,
  source_type=...)`, and `register_observer(factory)` take the object
  and read its name from it; the backend, model, device, and codec
  methods already did. Each method returns the registry, so calls
  chain from `ExtensionRegistry.with_built_ins()`, the way a DataFusion
  `SessionStateBuilder` takes rules and providers.
- **Objects travel to the worker by pickle.** `registry.manifest()`
  builds an `ExtensionManifest`: every object registered after the
  built-ins, pickled with its kind and name; the entry point modules
  loaded with `load_extension`; the local Python sources, which now
  default to each registered object's top-level package; and the pip
  packages. The query request and the physical plan envelope carry the
  manifest instead of a list of module names, and `registry_from_manifest`
  rebuilds the worker's registry from it. An object that cannot pickle,
  such as a lambda, fails at manifest time with a message that names
  it. `register_quail_extension(registry)` entry points still work and
  are the way to declare pip packages.
- **Metrics live on the result.** A finished `QueryResult` carries the
  executed `PhysicalGraph` as `plan` and each node's `NodeMetrics` as
  `node_metrics`, and `explain_analyze()` prints them, like DataFusion's
  `ExecutionPlan::metrics()` and `EXPLAIN ANALYZE`. Observers stay for
  work that must happen during execution; `result.observer(cls)`
  returns one's report.
- **The example needs no registration.** `quail_ext_examples/cost_ledger.py`
  is a function, `charge(result)`, that turns the plan and metrics into
  per node rows and totals with GPU dollars. It replaces the observer
  version of the same day and `plan_trace.py` before it.

## Why

The documented way to add an observer and read its numbers was three
strings: the module path, the source directory, and the observer name
as a dict key, plus an entry point function, for a report the result
already had the data for. Only the module name had to be a string, and
only for entry point modules.

## Before and after

```python
registry.load_extension("quail_ext_examples.plan_trace",
                        local_python_sources=("quail_ext_examples",))
result.report["observers"]["example.plan_trace"]["nodes"]
```

```python
result = session.sql("...").run()
result.explain_analyze()
cost_ledger.charge(result)["totals"]["usd"]

registry = quail.ExtensionRegistry.with_built_ins().register_observer(Progress)
result.observer(Progress)
```
