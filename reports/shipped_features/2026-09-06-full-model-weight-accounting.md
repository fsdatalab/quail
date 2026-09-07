# Keep the full model on the GPU

- Model loading caches the TRUE/FALSE output rows for reuse across queries
  and keeps the full output matrix on the GPU. It no longer removes that
  matrix or maintains separate accounting for shared output weights.
- Memory budgets use the full `W_mem` footprint. The calculated Qwen3 32B
  KV budget falls from 112,312 to 106,377 tokens on one H100. The Qwen3 4B
  budget is unchanged. These are calculated capacities, not benchmark results.
- Model tests check that the full output matrix and input embeddings remain
  available, cached rows preserve scores, and repeated queries reuse the
  same small tensor. Budget tests cover the full 32B weight footprint.
- The obsolete head-residency cell and report are removed. The load-profile
  cell and model documentation use the updated initialization function.
- The docs workspace includes the Next.js agent instructions and explicitly
  allows the required `esbuild` installation script.

No GPU inference was rerun for this change.
