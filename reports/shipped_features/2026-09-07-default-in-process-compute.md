# In-process compute is the default, and vLLM ships with the package

- `quail.Session()` now uses `InProcessComputeProvider` when no
  provider is passed. Before, `Query.run()` created a
  `ModalComputeProvider` on the first run when the session had none.
  The default is now chosen in `Session.__init__`, so
  `session.compute_provider` is set as soon as the session exists.
- `pyproject.toml` adds `vllm==0.26.0` as a dependency on Linux, the
  same version the Modal image installs. A plain install on a Linux
  machine with a CUDA GPU can therefore run a query without Modal.
  vLLM publishes Linux wheels only, so the dependency carries a
  `sys_platform == 'linux'` marker; on macOS and Windows the package
  installs without it and queries go through `ModalComputeProvider`.
- `InProcessComputeProvider.execute` checks that torch is installed
  and a CUDA GPU is visible before it plans the query. When neither
  holds it raises a `RuntimeError` that says which is missing and
  names `ModalComputeProvider` as the alternative. A fake
  `physical_executor` skips the check, so the CPU tests are unchanged.
- The Modal provider is unchanged. Pass it explicitly:
  `quail.Session(compute_provider=quail.ModalComputeProvider())`.
  The quickstart example and its docs page do that now, because that
  run happened on Modal.

Why: the package was only usable through Modal, though the in-process
path already existed for the benchmark runner inside Modal functions.
Anyone with their own GPU should be able to install the package and
run a query in their own process.

A local machine also needs a C compiler and the Python headers
(`python3.12-dev` on Ubuntu), because Triton compiles a small C helper
when vLLM starts. The install docs say so.

Costs: the locked Linux install grows by vLLM, torch 2.11 (CUDA 13),
and the NVIDIA libraries, about 7.8 GB on disk. CI on `ubuntu-latest`
installs them too.

Validation: `tests/test_default_compute.py` checks the default
provider type, the error message without a GPU, and that a fake
executor skips the check. Installed the locked dependencies on a Linux
machine without a GPU: `import vllm` works and `Session().sql(...).run()`
raises the new `RuntimeError`. No GPU run; the Modal path is unchanged.

```sh
uv run ruff check quail tests experiments reports tools
uv run python tools/check_long_strings.py
uv run vulture
uv run pytest -q
```
