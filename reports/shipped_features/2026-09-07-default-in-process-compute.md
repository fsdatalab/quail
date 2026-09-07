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

A local machine also needs a C compiler, because Triton compiles a
small C helper when vLLM starts. That helper also needs the Python C
headers, which Ubuntu's system Python does not ship, so `[tool.uv]`
now sets `python-preference = "only-managed"`: `uv sync` downloads its
own Python 3.12 build, which includes them. Found on a Nebius H100 VM
with Ubuntu 24.04, where the first run failed in `gcc` for the missing
`Python.h`.

The Quail backend committed the Modal kernel-cache and results volumes
after every warmup and query, and wrote a run record under `/results`.
On a machine outside Modal the commit raised `modal.exception.AuthError`
after the three minute kernel compile. `quail/runtime/volumes.py` now
has `commit_results`, `commit_kernel_cache`, and `run_record_path`; the
commits do nothing when `modal.is_local()` is true and the record is
skipped when `/results` is not a writable directory. Inside Modal the
behavior is unchanged.

Before the model loads, the in-process provider points the kernel
caches (`DG_CACHE_DIR`, `DG_JIT_CACHE_DIR`, `TRITON_CACHE_DIR`,
`VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`) at
`~/.cache/quail/kernels`, and sets `PYTORCH_CUDA_ALLOC_CONF` and
`VLLM_USE_FLASHINFER_SAMPLER` the way the Modal image does. Values
already set in the environment win. The warm-kernel marker derives its
path from `DG_CACHE_DIR`, so it lands next to the compiled kernels and
the second run on a machine does the touch pass instead of the compile.

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
