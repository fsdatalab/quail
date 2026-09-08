# In-process compute is the default, and vLLM ships with the package

What changed:

- `quail.Session()` uses `InProcessComputeProvider`, chosen in the
  constructor. Modal is an explicit choice:
  `Session(compute_provider=quail.ModalComputeProvider())`.
- `vllm==0.26.0` is a dependency on Linux, the version the Modal image
  installs. vLLM has Linux wheels only, so macOS installs without it
  and runs on Modal. `[tool.uv]` uses uv's own Python, which carries
  the C headers Triton needs at startup.
- Without a CUDA GPU, `run()` raises a `RuntimeError` that names
  `ModalComputeProvider`.
- The Quail backend committed Modal volumes and wrote to `/results`
  on every run. Those are no-ops outside a Modal container now.
- The in-process provider points the kernel caches at
  `~/.cache/quail/kernels`, the Modal image's layout, so the compile
  pass runs once per machine.
- `QueryRequest.planned_query` lets the in-process provider run the
  caller's query, so `explain()` then `run()` tokenizes once.
- `quail/progress.py` prints `quail:` lines for tokenizing, planning,
  model boot, and every five seconds of a filter or join.
- Planning no longer waits for tokenization. The session reads the
  document column's byte lengths, tokenizes the first 256 documents for
  a tokens per byte ratio, and plans on the scaled lengths; the token
  file is written on a background thread, and the Quail backend boots
  from the plan (`prepare_request`) before that file is done. The
  report's `token_wait_s` is how long execution then waited for it.
  Once a column's token file exists, later queries plan on exact counts.

Why: the package only worked through Modal. Anyone with a GPU should
be able to install it and run a query in their own process.

Measured on a Nebius VM, one H100 SXM, Ubuntu 24.04, CUDA 13, Qwen3 4B
fp8, `demos/imdb_ending_filter.py` over all 100,000 IMDB reviews
(29.7 M document tokens; 30.7 s to tokenize and plan on the CPU):

| Query | Predicted | Measured |
| --- | --- | --- |
| one filter, `wall_s` | 240 s | 262.45 s |
| one filter, fresh tokens | 29.7 M | 32.2 M |
| one filter, $/query | | $0.2879 |
| two filters, `wall_s` | 270 s | 268.72 s |
| two filters, fresh tokens | 32.5 M | 32.50 M |
| two filters, $/query | $0.30 | $0.2948 |
| two filters, passing | 28,000 then unknown | 28,296 then 16,057 |

The one-filter prediction left out the 25 question tokens per review;
with them, the QUAIL-B IMDB-1 rate of 123,000 tokens/s gives 262 s.
The two-filter prediction scaled IMDB-6 against IMDB-1. The second
question cost 283,000 fresh tokens and 6.3 s because survivors' KV was
still on the GPU.

First boot on the machine was 218.9 s, of which 189 s was the kernel
compile pass. With the cache, boot is 10.6 s.

Length estimates on the 100,000 reviews, measured on this CPU: total
29,879,025 estimated against 29,716,778 exact, +0.55%; per document the
median error is 3.8% and the 90th percentile 9.8%. The byte-length scan
takes under 10 ms and `explain()` returns in 0.26 s, against 10 to
30 s for the full pass depending on the CPU. The sample uses the
transformers tokenizer, which loads in under a second; bpe-qwen builds
its tables for about 12 s at load, so it is loaded on the background
thread. Execution then waits only for the part of the pass that the
model boot did not cover.

GPT-5 nano for the same work, at its prices on 2026-09-07 ($0.05 per 1 M
input tokens, $0.40 per 1 M output tokens, half in batch), one output
token per request and no reasoning tokens: $1.65 list or $0.83 batch
for one filter, $2.11 or $1.06 for two, because the second pass resends
the 28,296 reviews. Cached input pricing does not apply: GPT-5 nano
caches prefixes of 2,048 tokens or more, and these reviews average 297.

Costs: the Linux install grows to about 7.8 GB (vLLM, torch 2.11 with
CUDA 13, NVIDIA libraries). CI installs them too.

Validation: `tests/test_default_compute.py` and
`tests/test_local_caches.py`; the two GPU runs above; the CPU suite.

```sh
uv run ruff check quail tests experiments reports tools
uv run python tools/check_long_strings.py
uv run vulture
uv run pytest -q
```
