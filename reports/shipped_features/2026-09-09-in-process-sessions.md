# In-process sessions with whole-query Modal submission

- `Session` runs queries in its calling process. `Query.run()` passes the
  existing query directly to the executor and reuses its planning and token
  files. The compute provider interface and logical request wrapper are removed.
- Modal functions create ordinary sessions, register documents and extensions,
  run queries, and collect or save results before returning. The complete
  examples are `demos/quickstart.py` and `demos/quickstart_modal.py`.
  The Modal wrapper calls the ordinary example's `run_query()` function.
  Both use QUAIL-B's IMDB-1 query definition on 100 published reviews to find
  reviews that mention a positive aspect of a movie. QUAIL-B now exposes
  `load_table` and `get_query`; the demo has no S3 or Parquet code.
  QUAIL-B's file-store classes are removed. Label loaders accept root paths,
  and calling scripts choose where to save Quail's benchmark results.
  Both require a CUDA GPU. The separate `local_gpu_smoke.py` demo is removed.
  The IMDB documentation reproduction and extension smoke use the same pattern.
- The calling script defines the Modal image, GPU allocation, credentials,
  volume mounts, and report paths. There is no Modal integration module.
  Modal is a development dependency for demos and experiments, not an engine
  dependency. The quickstart mounts one volume for models, kernels, and reports,
  then commits it once when the function finishes or fails. Existing app names
  are preserved. Other experiments keep their existing cache volumes.
- Libraries use their own cache defaults unless the caller sets their cache
  environment variables. `QUAIL_CACHE_DIR` controls only Quail's warmup markers
  and defaults to `~/.cache/quail/kernels`.
- The execution entry point is `runtime/execute.py`. The runtime and model
  backends no longer import Modal, write run records, or commit volumes.
  The calling script writes reports and commits volumes explicitly.
  Kernel-volume commits are no longer included in model warmup timing.
- Table providers implement `scan` in the session's process. Source
  descriptions and the source-reader registry are removed. File and S3 reads
  continue through the same Arrow dataset interface.
- `Session.close()` releases token files and background tokenization. Backends
  keep loaded models in process state. Several queries inside one function can
  reuse a session; separate Modal invocations do not guarantee model reuse.
- Callers using the removed `compute_provider` argument must put their session
  inside a GPU process. No inference algorithm or benchmark measurement changes
  are included.

Tests cover planning reuse, table providers, extension registration, collected
results after the session closes, and volume persistence after a function fails.
An import check verifies that loading the engine does not load Modal.
Related test cases now share setup and assertions in 97 collected tests,
compared with 263 before consolidation. Obsolete registry-transfer tests are
removed. Failure cases still cover cancellation, invalid plans, failed
tokenization, and volume commits. All 97 tests passed with
`uv run --with torch --with ijson pytest -q`. The default environment passed
90 tests and skipped 7 because those optional dependencies were absent.
Ruff, the long-string check, Vulture, and the documentation build passed.
No GPU run was performed.
