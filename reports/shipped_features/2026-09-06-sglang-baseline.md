# SGLang pair ordering without client batch limits

- SGLang filters advance on individual answers. Admission uses the same
  calculation as pipelined vLLM, with the engine's reported KV capacity,
  page size, and request limit.
- Joins submit all anchors for one partner before moving to the next partner.
  All pairs enter in one call. The half-KV grouping rule and 16,384-request
  blocking batches are removed. The engine controls active requests and KV.
- The benchmark runs SGLang in the existing child-process runner used by
  vLLM. Modal heartbeat messages do not influence request batch sizes.
- FEV-9 took 135.74 seconds, compared with 304.92 seconds with vLLM's
  pair order and 138.26 seconds under the earlier limits. Fresh computation
  was 4,095,266 tokens, compared with 20,388,882 and 4,096,274 respectively.
  Answer agreement remained 69.08%.
- The [FEV-9 report](../2026-09-06-sglang-baseline.md) records the confirming
  run. The main and dataset plots identify saved results that still use
  the earlier adapter. Larger anchor sets have not been rerun without tiling.
