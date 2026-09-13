# Quail

Quail runs language-model filters and joins over document collections.
Write queries in SQL or Python. Quail plans and executes their model calls together.

Model execution requires a CUDA GPU. The supported models are Qwen3 4B fp8
and Qwen3 32B fp8, with one model copy per GPU.

Once published, install the distribution and import the `quail` package:

```bash
pip install quail-engine
```

## Quickstart

From this repository, install the dependencies with Python 3.12 and uv:

```bash
uv sync
```

Run the example on a machine with a CUDA GPU:

```bash
uv run python demos/quickstart.py
```

The [example](demos/quickstart.py) runs QUAIL-B's IMDB-1 query on 100 reviews.
It prints the matching rows and execution report.

## Benchmark

[QUAIL-B](https://github.com/fsdatalab/quail-bench) contains the query definitions,
datasets, reference labels, and scoring. Supported scale factors are 0.1, 0.5, and 1.0.
Reference labels load directly from QUAIL-B's public S3 bucket.

For example, run IMDB-4 with Quail:

```bash
uv run python -m quail.bench.quailb \
  --only IMDB-4 --sf 0.1 --output-dir results/imdb-4
```

Omit `--only` to run the full benchmark, or pass comma-separated query IDs.
Choose a new directory with `--output-dir` for each run.

The run directory contains `run.json`, saved query answers, and `report.md`
with timings, throughput, GPU cost, and accuracy.

Regenerate the report from saved answers without rerunning inference:

```bash
uv run quail-b report results/imdb-4
```

## Modal (optional)

To run the quickstart on Modal instead, configure your account and submit it:

```bash
uv run modal setup
mkdir -p results
uv run modal run demos/quickstart_modal.py \
  2>&1 | tee results/quickstart.log
```

The [Modal example](demos/quickstart_modal.py) calls the same query function on
an H100. You control its image, volume mounts, cache settings, and report path.
See the [benchmark guide](docs/content/docs/user-guide/benchmark.mdx) for running
Quail and the comparison engines on Modal.

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
