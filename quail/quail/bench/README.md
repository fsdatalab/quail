# QUAIL-B

QUAIL-B has 26 filter and join queries at scale factor 0.1. The saved
ground truth covers all 19 predicates used by those queries.

A predicate is one exact TRUE or FALSE question, including its prompt and
the input columns it reads. A query can use one predicate or combine several
predicates. Ground truth is saved per predicate, document, or document pair.

## Run the benchmark

Run all 26 queries with Qwen3 4B:

```bash
mkdir -p results/benchmark
run_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-quailb.log"

uv run python -m quail.bench.quailb \
  --sf 0.1 \
  --model qwen3-4b-fp8 \
  --gpus 1 \
  --prediction "State the expected runtime and accuracy." \
  2>&1 | tee "$run_log"
```

Do not pass `--only` when every query should run. To run one query, add its
ID:

```bash
uv run python -m quail.bench.quailb \
  --sf 0.1 \
  --model qwen3-4b-fp8 \
  --gpus 1 \
  --only IMDB-4 \
  --prediction "State the expected runtime and accuracy." \
  2>&1 | tee results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-IMDB-4.log
```

Each command runs a cold pass and a warm pass. It reads ground truth before
the timers start. It reports runtime, H100 cost, tokens, documents per second,
answer accuracy, and final-row precision, recall, and F1.

The local JSON and Markdown files use the same UTC timestamp prefix under
`results/benchmark/`. The PNG uses that prefix under
`reports/plots/benchmark/`. The JSON and raw query records are also saved on
the `quail-results` Modal volume under
`/results/benchmarks/quailb/runs/<run_id>/`.

## Add a query

First decide whether the query reuses existing predicates.

If it uses the same prompt constants with the same input roles, add the query
to `queries()` in `quailb.py`. No new labeling run is needed. The evaluator
will use the saved labels for those predicates.

If the query adds a predicate:

1. Add the prompt constant and query to `quailb.py`.
2. Add one `PredicateSpec` to `PREDICATES` in `judge_pass.py`. Give it a
   descriptive stable key, the prompt, the input table, the input column, and
   the left and right roles.
3. Add the predicate to the matching filter group or join call in
   `run_judge_pass()`. Filters label every row in their input table. Joins
   label every left and right row pair.
4. Use source labels only when the dataset gives the exact answer required by
   the prompt. Otherwise keep the default Qwen3 32B label source. BioDEX
   reactions do not count as source truth.
5. Update the predicate count test and add a prompt-rendering test when the
   new input shape differs from an existing one.
6. State the expected label count, runtime, cost, and repeat differences.
7. Run the label command below.

```bash
mkdir -p results/benchmark
label_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-label.log"

uv run modal run -m quail.bench.judge_pass \
  2>&1 | tee "$label_log"
```

The command prints the Modal function call ID into the log. If the call stops,
run the same command again. Completed Parquet parts have stable IDs and are
skipped. Existing predicate label sets are reused, so only missing predicate
rows are sent to Qwen3 32B.

When the run finishes, it writes a new complete collection and makes that
collection active for the corpus. The next benchmark run reads it by default.
Use `--ground-truth-collection <collection_id>` only when testing a specific
older collection.

Changing an existing prompt or its input roles creates a new predicate
version and a new label set. The old labels remain on the volume. Run the same
label command to create and activate the replacement collection.
