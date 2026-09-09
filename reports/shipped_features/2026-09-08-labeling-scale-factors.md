# Labeling at scale factors 0.1, 0.5 and 1.0, with derived collections

Date: 2026-09-08.

## What changed

- `quail/bench/labeling.py` accepts scale factors 0.1, 0.5 and 1.0 and
  now holds the Modal functions and entrypoint that used to live in
  `judge_pass.py`. One module.
- `derive_collection(sf, source_collection_id)` builds a smaller scale
  factor's collection from a finished larger one on the CPU, matching
  documents by the content hash every label row stores. Only the
  LePaRD citation join is recomputed, because its source labels
  depend on which citation pairs the corpus sampled (2 of 216,500
  differ between sf=0.1 and sf=1.0). Entrypoint option
  `--derive-from-collection`.
- A join part holds 500,000 pairs instead of 4,096 prompts. At sf=1.0
  a 4,096-prompt part was 2 claims against 1,478 FEVER passages, and
  the planner recomputed all passage anchors in every part, about 31
  hours per join. Each label-set manifest records the part size, so
  older label sets keep their boundaries.
- Timeouts: 24 hours per workload container, 1 to 2 hours and 16 GB
  for the CPU steps.
- `--publish-collections <id>,<id>` uploads the files a reader needs
  for each collection from the volume to the public bucket: manifests,
  corpus tables, and each label set's compact `labels.parquet`. Part
  files stay on the volume. It runs in a small Modal container with
  the AWS credentials of the machine that launches it, resolved the
  way the aws CLI resolves them (profile, environment, or SSO).
- `quail/backends/quail/worker.py` creates the cuBLAS handle right
  after the weights load. Before, the first cuBLAS call was the answer
  head at the warm-up's activation peak, where the caching allocator
  held every free byte, and every 32B container that took the cached
  warm-up tier failed with `CUBLAS_STATUS_ALLOC_FAILED`. The boot also
  logs free GPU memory before loading.

## Why

The benchmark needs labels at sf=0.5 and sf=1.0, and one pass at
sf=1.0 is enough: a smaller scale factor samples a prefix of the
larger one's documents (checked against the pinned sources for every
table), so its labels copy over by content.

## Numbers

The sf=1.0 pass: 51,801,003 labels in 22.3 H100 hours, $88, wall time
7.7 hours. Report: `2026-09-09-quailb-sf1-labels.md`.
