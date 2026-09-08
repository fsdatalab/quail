# QUAIL-B in its own repository, with public data and Quail as one runner

Date: 2026-09-08.

The benchmark is its own repository,
[fsdatalab/quail-bench](https://github.com/fsdatalab/quail-bench),
installed here as the `quail_b` package (distribution `quail-b`) and
pinned to a commit in `pyproject.toml`. It runs no engine and no
model. `quail/bench/` is Quail's runner for it. Stored data keeps its
`quailb` identifiers, because predicate keys and the
`ground_truth/quailb/schema_v1` path are hashed into label-set ids.

## What changed

- The `quail_b` package holds the benchmark: `data.py` (document sets,
  pinned sources, sampling, corpus identity), `prompts.py`,
  `queries.py`, `predicates.py` (the 21 predicates and the identity of
  their labels), `rendering.py` (the exact prompt text a predicate
  asks), `store.py` (the public bucket and a local directory),
  `labels.py`, and `scoring.py`. Nothing in it imports `quail`.
- The 32 queries are data. `quail_b.queries.QUERIES` is a tuple of
  `QuerySpec` records: aliases with their tables, text columns, and
  filter templates in written order; binary joins with their aliases
  in placeholder order; and the select list. Before, the queries
  existed only as closures over a Quail session.
- Scoring takes a `RunOutput`: the engine's filter answers keyed by
  (alias, written position), its join answers keyed by written
  position, and its final rows, all in the benchmark's own ids. The
  expected rows come from the labels through pyarrow joins. A test in
  the runner checks that Quail sends the same prompt text as the
  labels answered, for every predicate.
- `quail/bench/quailb.py` is the runner. `build_query(session, spec)`
  turns one spec into a Quail query, `run_output(result, spec, corpus)`
  turns a `QueryResult` into a `RunOutput`, and `answer_oracle` wraps
  the labels as the callable `quail.speed_of_light_estimate` takes.
  `run_suite` and the same-GPU Modal runner measure and save what they
  did before. `rows_from_answers` derives final rows from saved
  answers; the shared KV retention scorer uses it.

- Corpus and labels are public: 1.07 GB at
  `s3://quail-bench/ground_truth/quailb/schema_v1/`, readable without
  an AWS account, in the same layout as the `quail-results` volume.
  `quail_b.store.S3Files` reads it over plain HTTPS with no AWS SDK.
- `build_sets` downloads the sf=0.1 corpus from the bucket, 13 MB in
  5 seconds, and checks that it hashes to the pinned corpus id. A new
  machine needs no Hugging Face downloads to get the documents.
- `load_ground_truth()` reads the bucket by default. Loading the full
  sf=0.1 collection, 21 predicates and 1,210,264 labels, took 15.5
  seconds from a remote container.
- `quail-b` runs no model and has no Modal dependency. It keeps the
  predicate table and the label identities in `quail_b/predicates.py`.
- The pass that writes the labels is `quail/bench/labeling.py` here.
  It runs each predicate as a Quail query with Qwen3 32B fp8 on one
  GPU: a filter over every document, a full join over every pair.
  `quail/bench/judge_pass.py` runs it on Modal, one H100 per workload,
  on this repository's volumes; `--publish` uploads a finished pass to
  the bucket. A different engine can flip a borderline answer, so
  labels from the Quail judge get their own judge id and label-set
  ids; the published collection, judged by vLLM, stays the reference
  until a full Quail-judged pass replaces it. That pass has not been
  run.
- `ModalVolumeFiles` moved to `quail.runtime.volumes` next to the
  volume it reads.
- The runner reads labels from the mounted volume inside a Modal
  container and from the bucket anywhere else, and writes run records
  to the volume on Modal or to `results/` locally. With main's
  in-process compute provider, `python -m quail.bench.quailb` runs the
  benchmark on a local GPU with no Modal account.

## Why

The benchmark repository should be usable by anyone with a GPU, the
way TPC-H or ClickBench are. Before this, reading the labels needed a
Modal account with access to our volume, the documents were rebuilt
from Hugging Face on every new machine, and the labeling pass could
only be launched as Modal functions.
