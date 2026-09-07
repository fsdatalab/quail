# Registry objects passed through execution

- `register_node(Node, runtime=runtime)` registers the standard node codec and
  runtime together. Registration methods specify their argument types. Lookup
  tables are read-only, and failed registrations leave no partial additions.
- The registry stores ordinary Python objects. The query sends that registry
  to Modal directly. There is no extension manifest, parser, or registration
  replay. `load_extension` calls its module's registration function once.
- Source preparation, planning, and execution reuse the same registry in the
  Modal process. Previously, the worker rebuilt it at each of those steps.
- The existing multi-GPU path sends the registry to each GPU child once per
  query. The single-GPU Quail path runs in the Modal process.
- Dependencies are explicit `local_python_sources` and `pip_packages` options
  on `ModalComputeProvider`. Module loading no longer accepts dependency
  arguments. Package inference from registered objects was removed.
- `initialize_worker=callback` calls `callback(registry)` once per query inside
  the Modal process, before opening sources. It can register objects that must
  be created there, such as objects holding connections or locks.
- Modal returns a materialized `QueryResult`, including the executed plan and
  node metrics. The client does not rebuild a registry to decode the result.

Validation covers Modal's Python serialization, a fresh Python process,
registration order, failed module loads, and worker initialization with a lock
that cannot be serialized. It also checks registry reuse during source
preparation, planning, and execution. There is no GPU benchmark in this change.

```sh
uv run --no-sync pytest -q
uv run --no-sync ruff check quail tests experiments reports
cd docs
node node_modules/next/dist/bin/next build
```
