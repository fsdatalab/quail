# Retain only TRUE/FALSE output weights

- Model loading now extracts the TRUE/FALSE output rows once and discards
  the full output head. No full CPU copy remains. Input embeddings remain
  available when they share weights with the output head.
- Both answerers reuse the small GPU matrix across queries. The worker
  supplies the answer token IDs when loading the model. Standalone loading
  derives them from the model's tokenizer. A later query with different
  answer token IDs is rejected before inference.
- Qwen3 32B's separate output matrix contains 151,936 x 5,120 bf16 values.
  Removing its CPU copy saves 1,555,824,640 bytes of tensor storage, computed
  from those dimensions. GPU weight and KV budgets are unchanged. Qwen3 4B
  still needs the full shared matrix for input embeddings.
- CPU tests use PyTorch tensors to verify that the untied matrix is
  released, shared embeddings remain usable, retained rows preserve
  scores, and repeated answerers share the same tensor. Backend lifecycle
  and memory-budget tests also pass. No GPU inference was rerun.

Validation:

```bash
uv run --with torch pytest tests/test_model.py tests/test_specs_budgets.py \
  tests/test_quail_backend.py tests/test_worker_release.py \
  tests/test_worker_regret.py -q
```

`experiments/cells/head_residency.py` is updated for a future H100 check
of both supported models. Its prediction is that the retained rows remain
on the GPU, no full output-head copy remains, and planted-flag answers
stay correct across all three attention paths.
