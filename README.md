# QUAIL-B

QUAIL-B is a benchmark of 32 AI filter and join queries over movie reviews,
medical reports, factual claims, legal citations, and agent conversations.
It provides the data, query definitions, and reference labels. Your inference
engine runs the queries.

## Install

Use Python 3.12.

```sh
uv add "quail-b @ git+https://github.com/fsdatalab/quail-bench.git"
```

## Use

Load a benchmark's queries, input tables, and reference labels:

```python
import quail_b as benchmark

suite = benchmark.load_benchmark(["IMDB-4"], scale_factor=0.1)
query = suite.queries[0]
tables = suite.tables
```

IMDB-4 filters reviews for positive aspects and discussion of the ending,
then joins them with the movie aspects they discuss.

- Scale factors `0.1`, `0.5`, and `1.0` are supported.
- Inputs and labels load from public S3. No AWS account or Modal volume is needed.
- QUAIL-B checks input contents and row order against the published corpus.
- For a small example, use `benchmark.load_table("reviews", limit=100)`.
  Full benchmark scoring requires the complete inputs.
- For execution, use your engine's runner. For example, the
  [Quail runner](https://github.com/fsdatalab/quail-exploration/blob/main/quail/bench/quailb.py)
  translates these definitions into Quail queries.

See the [query definitions](quail_b/queries.py),
[data sources](quail_b/data.py), and [query diagram](figures/quailb_anatomy.pdf).

## Generate a report

Your runner supplies a `RunOutput` with predicate answers and final document IDs,
plus measured runtime and token counts. [`suite.score()`](quail_b/benchmark.py)
computes accuracy, throughput, and GPU cost. `suite.summarize()` combines the
query records. QUAIL-B does not execute queries or save files.

Model-generated labels use Qwen3 32B fp8. FEVER and LePaRD also use source labels.

For example, Quail generates a report from its saved summary JSON with this
command, run from the [Quail repository](https://github.com/fsdatalab/quail-exploration):

```sh
uv run --with matplotlib python reports/make_quailb_eval_plots.py \
    --input /path/to/summary.json \
    --report /path/to/report.md
```

Quail's script writes a Markdown report and a PNG with timings, costs, token
counts, and accuracy. It reads saved results and does not rerun inference.
Other engines can use QUAIL-B's [scoring functions](quail_b/scoring.py)
to produce their own reports.
