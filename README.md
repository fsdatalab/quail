# Quail

Quail runs language-model filters and joins over document collections.
Write queries in SQL or Python. Quail plans and executes their model calls together.

Model execution requires a CUDA GPU. The supported models are Qwen3 4B fp8
and Qwen3 32B fp8, with one model copy per GPU.

## Quickstart

From this repository, install the dependencies with Python 3.12 and uv:

```bash
uv sync
```

For Modal, run `uv run modal setup` once to configure your account. Then:

```bash
mkdir -p results
uv run modal run demos/quickstart_modal.py \
  2>&1 | tee results/quickstart.log
```

The example runs QUAIL-B's IMDB-1 query on 100 reviews using an H100.
It prints the matching rows and execution report.

- [quickstart.py](demos/quickstart.py) contains the query and runs directly on a CUDA GPU.
- [quickstart_modal.py](demos/quickstart_modal.py) runs the same code on Modal.
  The wrapper owns the image, volume, cache settings, and saved report path.

## Benchmark

[QUAIL-B](https://github.com/fsdatalab/quail-bench) contains the query definitions,
datasets, reference labels, and scoring. Supported scale factors are 0.1, 0.5, and 1.0.
Reference labels load directly from QUAIL-B's public S3 bucket.

For example, run IMDB-4 with Quail:

```bash
uv run modal run --detach -m quail.bench.quailb_parallel \
  --query IMDB-4 --sf 0.1 --no-include-baselines \
  2>&1 | tee results/benchmark.log
```

Omit `--query` to run all 32 queries. Omit `--no-include-baselines` to include
stock vLLM with operator-at-a-time execution, pipelined vLLM, and pipelined SGLang.

Results are saved on the `quail-results` Modal volume under
`/results/benchmarks/quailb/<run-id>/`. Set `--output-dir` to change the parent
directory. The command prints the run directory and function call IDs.

After [downloading the run directory](docs/content/docs/user-guide/benchmark.mdx),
generate a report without rerunning inference:

```bash
uv run quail-b report /path/to/run-directory/quail
```

QUAIL-B writes `report.md` inside the backend's directory. Each query family
also has its own saved run and report.

## Development

The tests run on the CPU. Model execution tests use a fake executor.

```bash
uv run pytest -q
uv run ruff check quail tests experiments reports tools
uv run python tools/check_long_strings.py
uv run vulture
```

See the [user guide](docs/content/docs/user-guide/index.mdx),
[architecture](docs/content/docs/architecture/index.mdx), and
[extension guide](docs/content/docs/extending/index.mdx) for details.
Experiment scripts are in `experiments/`; their reports are in `reports/`.
