# Faster cold model load: persisted vLLM cache and pinned hub revisions

What changed:

- Every vLLM GPU image now sets `VLLM_CACHE_ROOT` to
  `/root/.cache/kernels/vllm` on the kernel-cache volume. vLLM
  resolves the model architecture in a fresh Python subprocess at
  engine-config creation and caches the JSON under
  `VLLM_CACHE_ROOT/modelinfos`; the default location is ephemeral on
  Modal, so every container paid the ~13 s subprocess again. Now one
  container seeds it and every later container reads it back in
  about a second.
- `ModelSpec` has a `revision` field pinning each checkpoint to its
  hub commit hash, and `load_model` passes it into `EngineArgs`. A
  commit hash resolves from the HF cache volume without API round
  trips; a branch name pays them on every load. The worker, the
  calibration path, the ablation cells, and the stock vLLM baseline
  all pass `spec.revision`, so engine and baseline boots stay
  comparable.

Why: cold worker boots paid ~13 s of repeated architecture
inspection and hub-latency-dependent round trips on every fresh
container.

Before/after (cold `load_model`, Qwen3 4B fp8, one H100, measured by
`tests/gpu/load_profile.py`): 35.33 s before, 24.84 s after the
one-time seed - 10.5 s saved, 1.42x. Details in
`reports/2026-08-31-load-model-speed.md`.

Based on measurements and fixes by Charles Frye
(branch `charlesfrye/faster-boot`).
