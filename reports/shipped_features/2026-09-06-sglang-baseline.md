# SGLang uses the shared baseline scheduling rules

- SGLang filters advance on individual answers. Admission uses the same
  calculation as pipelined vLLM, with the engine's reported KV capacity,
  page size, and request limit.
- Joins submit all pairs in one call, with pairs sharing an anchor next to
  each other. The half-KV grouping rule and 16,384-request blocking batches
  are removed. The engine controls how many requests execute together.
- The benchmark runs SGLang in the existing child-process runner used by
  vLLM. Modal heartbeat messages no longer influence request batch sizes.
- The [FEV-9 report](../2026-09-06-sglang-baseline.md) records the confirming
  run. The main and dataset plots identify saved results that still use
  the earlier adapter.
- FEV-9 slowed from 138.26 to 304.92 seconds as fresh computation rose from
  4.10 million to 20.39 million tokens. Answer agreement remained 69.08%.
  The shared submission rules lost prefix reuse compared with the old policy.
