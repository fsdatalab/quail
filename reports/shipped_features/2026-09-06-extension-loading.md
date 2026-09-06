# Load extensions by module and read observer reports by class

Date: 2026-09-06.

## What changed

- `ExtensionRegistry.load_extension` takes the extension module object
  or its importable name. The name is what travels to the compute
  worker, which imports the module by it. `local_python_sources` now
  defaults to the module's top-level package, so the common case is
  `registry.load_extension(my_extension_module)`.
- `QueryResult.observer(observer)` returns the report one execution
  observer attached, looked up by the class, an instance, or the name.
  A missing observer raises a `KeyError` that lists the observers the
  query ran with.
- The example extension is `quail_ext_examples/cost_ledger.py`, an
  observer that charges each query's GPU seconds, evaluated documents
  or pairs, tokens, and dollars to the physical nodes that used them.
  It replaces `plan_trace.py`, which recorded row counts nobody needed.

## Why

The documented way to use an observer was three strings: the module
path, the source directory, and the observer name as a dict key. Only
the module name has to be a string, and only because the worker
imports it in another process.

## Before and after

```python
registry.load_extension("quail_ext_examples.plan_trace",
                        local_python_sources=("quail_ext_examples",))
result.report["observers"]["example.plan_trace"]["nodes"]
```

```python
from quail_ext_examples import cost_ledger

registry.load_extension(cost_ledger)
result.observer(cost_ledger.CostLedger)["totals"]
```
