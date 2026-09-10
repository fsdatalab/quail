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

Load a query and its input tables. The same code works for filters and joins:

```python
import quail_b as benchmark

query = benchmark.get_query("IMDB-4")
tables = {}
for name in sorted({alias.table for alias in query.aliases}):
    tables[name] = benchmark.load_table(name, scale_factor=0.1)
```

IMDB-4 filters reviews for positive aspects and discussion of the ending,
then joins them with the movie aspects they discuss.

- Scale factors `0.1`, `0.5`, and `1.0` are supported. Use the same scale
  for every input table.
- Tables are downloaded from public S3 as Arrow tables. No AWS account is needed.
- Add `limit=100` for a small example. Omit it for a full benchmark run.
- For execution, use your engine's runner. The
  [Quail runner](https://github.com/fsdatalab/quail-exploration/blob/blogs/quail/bench/quailb.py)
  translates these definitions into Quail queries.

See the [query definitions](quail_b/queries.py),
[data sources](quail_b/data.py), and [query diagram](figures/quailb_anatomy.pdf).

## Generate a report

The runner compares its answers with the saved reference labels using
[`Evaluator`](quail_b/scoring.py). Model-generated labels use Qwen3 32B fp8;
FEVER and LePaRD also use source labels. Keep the query id, scale factor,
and label collection id with the results.

For a Quail run, use its saved summary JSON. Run the following command from
the [Quail repository](https://github.com/fsdatalab/quail-exploration), not here:

```sh
uv run --with matplotlib python reports/make_quailb_eval_plots.py \
    --input /path/to/summary.json \
    --report /path/to/report.md
```

The script writes a Markdown report and a PNG with timings, costs, token
counts, and accuracy. It reads saved results and does not rerun inference.
Other engines can use QUAIL-B's [scoring functions](quail_b/scoring.py)
to produce their own reports.
