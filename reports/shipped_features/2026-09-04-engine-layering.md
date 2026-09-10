# Engine layering cleanup

Five changes that make the engine's layers match its interfaces. None
changes the public API or the extension interfaces. No experiment was
run; the CPU suite passes at every step, and the GPU paths were not
re-measured.

## What changed

- **The Quail backend owns its runtime code.** `quail/backends/quail/`
  now holds the one GPU graph runtimes, the multi-GPU coordinator and
  distributed execution, and the model boot and GPU child protocol.
  `quail/runtime` keeps the session, in-process execution, the generic
  runner, the token store, results, and the shared Modal image. The
  generic worker calls `backend.execute_request` and nothing else;
  `BackendExecutionContext` lost its Quail-only `graph_executor`.
  Each built in backend supplies the runtimes for its own nodes.
- **A public planning interface.** `quail.planner` exports
  `collect_operators`, `plan_query`, `plan_quail`, `preamble_tokens`,
  `order_filters_indexed`, `join_specs`, and `balanced_shards`.
  Backends, the session, and the bench import
  those instead of private names. `built_in_registry` moved to
  `quail.builtins`, and `quail/runtime/__init__.py` no longer imports
  the session layer, which removed the import cycles. Every
  function-level import that only dodged a cycle is now a module
  import; the four that remain are commented.
- **One request backend.** Stock vLLM, pipelined vLLM, and pipelined
  SGLang are three instances of `RequestBackend` with an engine
  adapter (`VLLMEngine`, `SGLangEngine`). The scheduling loops live
  once in `quail.backends.request_scheduling`, and the baselines under
  `baselines/` import them instead of keeping copies. The admission
  cap now counts the answer token before rounding to KV blocks, as
  the measured baseline did; the pinned test case gives the same cap.
- **The bench uses the session's execution path.** The QUAIL-B runner
  uses ordinary sessions, so benchmark and application queries build
  the same result objects. The provider introduced in this change was
  removed in [the session update](2026-09-09-in-process-sessions.md).
- **One result path.** The unused Arrow IPC file result was removed.
  Every backend hands the runner Arrow tables keyed by port, and every
  query returns a `QueryResult`. The pipe protocol between
  Quail's GPU children stays as plain Python values; it is transport
  inside one backend and the CPU tests cover its merge logic.

## Numbers

| Check | Before | After |
| --- | ---: | ---: |
| Lines in `quail/runtime` | 4,450 | 2,268 |
| Lines in `quail/backends` | 1,909 | 4,076 |
| Copies of the filter chain loop | 3 | 1 |
| Function-level imports of quail modules outside bench and executor | 102 | 9 |
| CPU tests | 240 passed | 239 passed |

One test was removed with the IPC file path it covered.
