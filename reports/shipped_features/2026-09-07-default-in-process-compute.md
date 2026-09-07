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

`QueryRequest` carries the caller's planned `Query` as `planned_query`.
`execute_query_request` runs that Query when it is present instead of
building a second session, so documents tokenized for `explain()` are
not tokenized again by `run()`. Remote providers ignore the field; the
Modal worker still tokenizes on its own machine. Measured on 100,000
IMDB reviews on this CPU: tokenizing and planning takes 30.7 s, and a
second query on the same session reuses the token file in 0.3 s.

Costs: the locked Linux install grows by vLLM, torch 2.11 (CUDA 13),
and the NVIDIA libraries, about 7.8 GB on disk. CI on `ubuntu-latest`
installs them too.

Measured on a Nebius VM, one H100 SXM, Ubuntu 24.04, CUDA 13 driver,
running `demos/local_gpu_smoke.py` with the default provider
and Qwen3 4B fp8, first run on the machine:

| Number | Value | Compared with |
| --- | --- | --- |
| rows | r1, r4 | the two reports that name a female patient |
| `wall_s` (query only) | 0.28 s | 4 documents, 151 fresh tokens |
| `boot_s` | 218.9 s | includes the 189 s one-time kernel compile pass |
| `worker_total_s` | 225.6 s | boot plus query plus tokenizing |

Then `demos/imdb_ending_filter.py` with one filter over all 100,000
IMDB reviews (29.7 M document tokens), same machine, kernels cached:

| Number | Value | Compared with |
| --- | --- | --- |
| matching reviews | 28,296 of 100,000 | the 0.25 selectivity hint |
| `boot_s` | 10.87 s | 218.9 s on the first run, before the kernel cache |
| `wall_s` | 262.45 s | predicted 240 s from document tokens alone |
| `fresh_tokens` | 32,216,778 | 29.7 M document tokens plus 25 question tokens per review |
| documents/second | 381 | 351 on QUAIL-B IMDB-1, 5,000 reviews |
| $/query | $0.2879 | about $4.89 for the same 32.2 M input tokens on GPT-4o mini at $0.15 per 1 M input tokens and $0.60 per 1 M output tokens (list price, 2026-09-07), or $2.45 through its batch API |

The prediction missed by 8% because it counted document tokens only.
At 32.2 M fresh tokens, the IMDB-1 rate of 123,000 tokens/s gives
262 s, which is what was measured.

Progress lines: `quail/progress.py` prints one line, prefixed
`quail:`, when tokenizing starts, every five seconds while it runs,
when the plan is ready, when the model starts loading and is ready,
and every five seconds during a filter or join with the count of
finished documents or anchors. Before, a 100k review query printed
nothing between the plan and the result.

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
