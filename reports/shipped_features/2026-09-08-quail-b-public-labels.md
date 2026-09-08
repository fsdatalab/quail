# QUAIL-B corpus and labels on a public bucket, no Modal in the benchmark

The benchmark package is `quail-b` (import `quail_b`), from
[fsdatalab/quail-b](https://github.com/fsdatalab/quail-b). Stored data
keeps its `quailb` identifiers, because predicate keys and the
`ground_truth/quailb/schema_v1` path are hashed into label-set ids.

## What changed

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
