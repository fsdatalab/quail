# QUAIL-B

QUAIL-B has 32 filter and join queries at scale factor 0.1, plus
2 optional PrivacyPolicies queries (PRIV-1, PRIV-2) that run only
when `register_privacy_sets()` has been called. The saved ground
truth covers all 21 predicates used by the 32 default queries.

FEV-9 filters both claim inputs with F11 and both evidence inputs with F13,
then joins c1 with e1, e1 with c2, and c2 with e2. It has four filters and
three joins. These filters reuse the existing predicate labels and selectivity
estimates. Measurements made before this extension used only F11 on c1.

A predicate is one exact TRUE or FALSE question, including its prompt and
the input columns it reads. A query can use one predicate or combine several
predicates. Ground truth is saved per predicate, document, or document pair.

## Run the benchmark

Run all 32 queries with Qwen3 4B. Modal starts one H100! container for
each query family. Each container runs Quail, stock vLLM, and pipelined
vLLM on that family. SGLang runs in a separate container because its
Python package conflicts with the vLLM package:

```bash
mkdir -p results/benchmark
run_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-quailb.log"

uv run modal run --detach -m quail.bench.quailb_parallel \
  --sf 0.1 \
  --model qwen3-4b-fp8 \
  --prediction "State the expected runtime and accuracy." \
  2>&1 | tee "$run_log"
```

Do not pass `--query` when every query should run. To run one query, add its
ID:

```bash
uv run modal run --detach -m quail.bench.quailb_parallel \
  --sf 0.1 \
  --model qwen3-4b-fp8 \
  --query IMDB-4 \
  --prediction "State the expected runtime and accuracy." \
  2>&1 | tee results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-IMDB-4.log
```

Each command runs every selected query once. It reads ground truth before
the timers start. It reports runtime, H100 cost, tokens, documents per second,
answer accuracy, and final-row precision, recall, and F1.

To compare the methods, use one H100! container for each query family. The
container runs Quail in one process group. After the query results are saved,
the parent stops the complete process group and waits for GPU memory use to
fall below 1 GiB. The same container then runs stock vLLM and pipelined vLLM
in a second process group. Both vLLM configurations share one loaded model.
The runner records the physical GPU UUID and checks that both process groups
saw the same H100. SGLang uses the same child-process isolation and cleanup
in its separate container.

```bash
run_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-quailb-parallel.log"

uv run modal run --detach -m quail.bench.quailb_parallel \
  --sf 0.1 \
  --model qwen3-4b-fp8 \
  --prediction "State the expected runtime and accuracy." \
  2>&1 | tee "$run_log"
```

The command prints one Modal function call ID for each query family. Quail
clears KV before each query. vLLM resets its prefix cache before each stock or
pipelined query. SGLang has a separate function call ID and may run on another
physical H100. The command saves separate summaries and one manifest on the
`quail-results` volume.

By default, the command loads `SELECTIVITY_ESTIMATE_COLLECTION` from
`quailb.py`. It is the current active collection. Pass
`--ground-truth-collection <collection_id>` only when testing a specific
older collection.

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
3. `judge_workload()` derives the filter and join labeling work from
   `PREDICATES`. Filters label every row in their input table. Joins label
   every left and right row pair.
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
