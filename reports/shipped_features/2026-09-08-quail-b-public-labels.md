# QUAIL-B labels on a public bucket, labeling on a local GPU

The benchmark package is now `quail-b` (import `quail_b`), from
[fsdatalab/quail-b](https://github.com/fsdatalab/quail-b). Stored data
keeps its `quailb` identifiers, because predicate keys and the
`ground_truth/quailb/schema_v1` path are hashed into label-set ids.

## What changed

- The reference labels are public: 1.07 GB, 10,259 files, at
  `s3://quail-bench/ground_truth/quailb/schema_v1/`, readable without
  an AWS account. The layout is the same as on the `quail-results`
  volume, so one `GROUND_TRUTH_ROOT` serves the bucket, the volume,
  and a local directory.
- `quail_b.labels.S3Files` reads the bucket over plain HTTPS, no AWS
  SDK. `load_ground_truth()` uses it by default. Loading the full
  sf=0.1 collection, 21 predicates and 1,210,264 labels, took 15.5
  seconds from a remote container.
- The runner `quail/bench/quailb.py` reads labels from the mounted
  volume inside a Modal container and from the bucket anywhere else.
  Run records go to the `quail-results` volume on Modal and to the
  local `results/` directory otherwise. With main's in-process compute
  provider, `python -m quail.bench.quailb` runs the benchmark on a
  local GPU with no Modal account.
- The labeling pass is `quail_b.labeling`, which runs on one local GPU
  of at least 80 GB: `python -m quail_b.labeling --root DIR`.
  `quail_b.judge_pass` is the same pass wrapped in Modal functions,
  one H100 per workload. Both write the same files under the same
  ids. `modal` and `vllm` are optional extras of `quail-b`.

## Why

The benchmark repository should be usable by anyone with a GPU, the
way TPC-H or ClickBench are. Before this, reading the labels needed a
Modal account with access to our volume, and the labeling pass could
only be launched as Modal functions.
