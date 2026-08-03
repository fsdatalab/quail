# DocEngine

DocEngine runs an ordered conjunction of language model filters over a document
set on one GPU. A document stops after its first false answer.

The project separates three things:

1. The logical query defines the filters and their fixed order.
2. A physical implementation defines how documents, filter prompts, KV cache,
   and GPU batches are arranged.
3. The evaluator compares returned answers with synthetic ground truth after
   execution.

The runtime never receives future filter answers.

## Physical implementations

The repository contains these implementations and references:

1. Task first sends every live document through one filter stage before starting
   the next stage.
2. Document first keeps a document KV cache in HBM and advances each document as
   soon as its answer returns.
3. One request per document keeps one sequence alive and replaces each completed
   filter tail with the next tail.
4. Fused k speculation evaluates k independent filter tails over one shared
   document prefix with FlashInfer cascade attention.
5. Oracle schedules use synthetic answers before execution. They are analysis
   references and are not deployable runtimes.

## Custom runtime

The rebuild uses vLLM only for model loading, the GPU model runner, and existing
GPU kernels. DocEngine owns:

1. document and filter state;
2. continuous variable-length batch packing;
3. every scheduled token chunk;
4. FP8 KV page allocation and reference counts;
5. shared-prefix block tables;
6. explicit KV truncation and free operations;
7. finite-query planning;
8. engine-step traces and immutable run artifacts.

The runtime does not call the vLLM scheduler or KV cache manager.

## Local verification

```bash
python3 -m pytest -q
```

## Modal experiments

The custom runtime smoke test uses one H100:

```bash
modal run experiments/modal_rebuild.py \
  --phase custom-smoke \
  --n-docs 8 \
  --n-filters 2 \
  --k 1
```

The fused k=2 correctness test uses:

```bash
modal run experiments/modal_rebuild.py \
  --phase custom-smoke \
  --n-docs 8 \
  --n-filters 2 \
  --k 2
```

Every new run is written under `results/runs/<run-id>/`. The directory contains
metadata, a compressed result, and checksums. No default command overwrites an
earlier run.

Regenerate the run index with:

```bash
python3 scripts/build_run_index.py
```

See [notes/EXPERIMENTS.md](notes/EXPERIMENTS.md) for the current run list.

## Main modules

1. `docengine/runtime/protocol.py` separates runtime input from evaluator labels.
2. `docengine/runtime/kv.py` owns FP8 KV pages.
3. `docengine/runtime/batching.py` packs exact variable-length token chunks.
4. `docengine/runtime/custom.py` runs one finite filter query.
5. `docengine/runtime/vllm_runner.py` converts custom batches into pinned vLLM
   model-runner inputs.
6. `docengine/runtime/batch_cost.py` estimates exact batch time from measured
   primitive tables and analytical work.
7. `docengine/optimizer/finite_query.py` contains exact small-query planning and
   measured batch selection.
