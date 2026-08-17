# results/engine

Measured GPU runs. Every file here was produced on the same image
(CUDA 13.0.1 devel base, `vllm==0.26.0`) on one H100 SXM, which is the
image that reads ~97,000 prefill tokens/s. Older results on a slim
image read 80,556 and are not kept.

| file | produced by | holds |
|---|---|---|
| `filter_cells.json` | `modal_filters.py` | KV rewind against stock, 10k docs, 5 filters, 3 reps each |
| `filter_steptrace.jsonl.gz` | the scheduler's step trace | one record per scheduler step: tokens, prefill/decode split, queue depths, KV occupancy |
| `torchprof_4b_filter_rank0.pt.trace.json.gz` | `modal_profiling.py::torchprof` | raw chrome trace, stock arm (gitignored, 127 MB) |
| `torchprof_4b_filter_ours_rank0.pt.trace.json.gz` | same, rewind arm | (gitignored, 127 MB) |
| `torchprof_4b_filter_slim.json.gz` | the same run, summarized | kernel classes and busy fraction |
| `ncu_gemm_bench_summary.txt` | `modal_profiling.py::ncubench` | Nsight Compute speed-of-light per GEMM shape |
| `persist_store.json` | `modal_persist.py --stage store` | restore against recompute, 1k docs, now with the per-job transfer summary: loads copy at 23-28 GB/s while running but keep the channel busy only 25-29% of the restore wall, so the query sees 6.8 GB/s effective. The slowdown is between jobs, not in the DMA. Host caveat: this container's demonstrated copy rate is ~25 GB/s against the 55.4 the pinprobe host reached, so treat 25 as this host's floor, not the software ceiling |
| `persist_xfer_trace.jsonl.gz` | the sitecustomize hook in `modal_persist.py` | one record per offload transfer job at submit and at finish (5,773 jobs, both directions), the finish record carrying the CUDA-event-timed copy duration; the summary above is computed from these |
| `persist_store_quail.json` | `modal_persist.py --stage store --connector quail` | the same cell on the coalescing connector (`quail/engineext/offload.py`): 3,577 per-request load jobs become 41 per-step merged transfers (~87 requests each), restore wall 4.05 s to 2.83 s best, copy rate 25 to 38 GB/s because gigabyte transfers also amortize the DMA ramp, submit-to-finish 82 to 23 ms. Different container than the stock cell, so compare each restore against its own cold pass: stock 58%, quail 36% |
| `persist_xfer_trace_quail.jsonl.gz` | the same hook, quail-connector cell | per-job records for the merged transfers; merged load sizes run 6 MB to 1.4 GB (p50 40 MB) |
| `persist_store_quail_waves.json` | `modal_persist.py --stage store --connector quail --waves` | wave pre-loading on the same cell: restore walls 1.93 s and 1.83 s against 2.83 s on-demand and the 1.68 s warm-cache floor. 28.0 GB moves per restore for a 24.1 GB corpus plus per-stage question tails; waves carry 55.9 of the 56.1 GB, copies run 34-39 GB/s with the channel 39-42% busy. Getting here took three scheduling fixes, each measured in the commit history: wave documents are claimed so the vendor's zero-cost load deferral cannot duplicate them (1.6x duplication before), the seam is a one-time per-query budget because requests reach the scheduler over ~20 steps (77% leaked to the per-request path before), and waves load only blocks missing past the locally cached prefix because chain stages revisit documents (97 GB moved for this job before) |
| `persist_xfer_trace_quail_waves.jsonl.gz` | the same hook, waves cell | per-job records; wave jobs ride a dedicated transfer chain so their gates never wait on per-request traffic |
| `persist_store_quail_waves_choked22.json` | same cell, `--choke-util 0.22` | DEPRECATED: measured with full-memory admission settings on a quarter-size pool, the mis-sized regime; superseded by deriving admission from the actual pool. Kept as the record. The pool squeezed under the 327k-token working set, spill off: restores 3.04 and 2.88 s against 1.93/1.83 s unchoked, ~44 GB loaded per restore against 28 unchoked - documents evicted mid-query reload on later stages, and even the write query loads 4.1 GB it never loads on a full pool |
| `persist_store_quail_waves_spill_choked22.json` | same cell, `--spill` added | DEPRECATED: same mis-sized regime, and the spill machinery it measured is deleted from the code (admission sized to the pool leaves nothing to reorder). Kept as the record of what reload ordering was worth there. The same squeezed pool with preempted documents given first claim on wave capacity: restores 2.37 and 2.24 s, 22% better raw and ~17% after normalizing each restore by its own cold pass (different containers) - inside the predicted 10-25% band, recovering over half the choke penalty. Churn bytes stay ~equal (85.6 vs 88.0 GB); the win is ordering, not volume |
| `persist_xfer_trace_quail_waves_choked22.jsonl.gz`, `persist_xfer_trace_quail_waves_spill_choked22.jsonl.gz` | the same hook, choked cells | per-job records for both choked runs |
| `persist_split.json`, `persist_split_quail.json`, `persist_split_quail_waves.json` | `modal_persist.py --stage split --connector ...` | the realistic-reuse protocol: write the corpus through under filters 1-3, then restore under filters 4 and 5, which the store never saw, so only document KV can restore and each new question computes. Stock 2.50/2.49 s, per-step batching 2.05/2.07 s, waves 1.69/1.44 s (pre-run predictions 2.8-4.0, 1.8-2.6, and 1.0-1.5 s; the wave q2 landed 0.19 s above its band, everything else inside or better). Every rung moves the same 52.0 GB for the two restores with zero duplication; waves carry all of it, 2,000 document-visits in 23 batches at 35 GB/s - exact full coverage. Required waving the longest consecutive store-hit prefix instead of demanding a full-suffix hit |
| `persist_xfer_trace_split.jsonl.gz`, `persist_xfer_trace_split_quail.jsonl.gz`, `persist_xfer_trace_split_quail_waves.jsonl.gz` | the same hook, split cells | per-job records for the three split rungs |
| `pinprobe.json` | `modal_pinprobe.py` | host-to-GPU bandwidth for three pinned-memory paths |
| `speed_limit.json` | prefill throughput sweep | the 97,889 tok/s anchor `quail/plan/cost.py` calibrates PHI against |
| `calibrate_all.json` | `modal_calibrate.py --families all` | the calibration sweep the fits use: 100 step cells (alpha, c1, c2, c4, c5) plus the c6 transfer probes, one container, synchronous eager boot |
| `cost_model_fit.json` | `quail.plan.fit` | the fitted step, host, and prefill models, with t_read, eps, and the offload-vs-recompute crossover per transfer tier |
| `calibrate_alpha.json`, `calibrate_c1.json`, `calibrate_c2-c4-c5.json`, `calibrate_c6.json` | `modal_calibrate.py --families <fam>` | the staged gate-check runs, one container each; the fits read only `calibrate_all.json` because rows from different containers must not mix |
| `attnshare.json` | `modal_profiling.py::attnshare` | kernel-class shares of single-document prefills by length on the calibration boot, with each length's top kernels by time; says where attention overtakes the GEMMs |
| `c0_anchor.json` | `modal_filters.py::c0_anchor` | the filter comparison with an in-container rate probe; each cell carries c0 = wall - reads x corpus / rate, so host speed cancels |
| `makespan_check.json` | `quail.plan.validate` | predicted against measured query walls per arm, at designed and at effective (verdict-measured) selectivities, plus a container variant at the anchor's probed rate and c0 |
| `filter_cells_bf16.json` | `modal_filters.py::main --kv bf16` | the filter comparison on a 16-bit-KV boot, admission repriced at 2 bytes per element; accuracy identical to fp8, so the KV format does not cause the answer anomaly |
| `filter_cells_syncclient.json` | `modal_filters.py::main`, synchronous client | the validation run for the asyncio removal: every counter matches `filter_cells.json` exactly (rewind reads 1.197, stock 1.229, identical request and answer counts), so the sync client computes what the async client computed. Walls ran on a slower, noisier container (rewind 40.5-42.2 s, stock 41.7-45.3 s; within-container ratio brackets the banked 1.08x), so `filter_cells.json` stays the quotable comparison |

## One measurement caveat

`filter_cells.json` carries two read multipliers per cell:

- `client_reads` — `(prompt_tokens - cached_tokens) / corpus_tokens`,
  computed by the client. Correct for the stock arm, where each
  request is submitted once.
- `reads` — the same quantity from the scheduler's step trace, which
  sees every prefill.

They differ for chain mode. A rewind truncates and re-extends
`prompt_token_ids`, so the client's final snapshot shows only
`[document + last question]` and the intermediate question prefills
are invisible to it. The client counter reported an identical 1.143x
for three operators whose wall times spanned 39.8 to 52.4 seconds.
Use `reads`; `client_reads` is kept so the discrepancy stays visible.

Both multipliers divide by document tokens only while counting
question tokens in the numerator, so 1.00x is not the floor.

`filter_cells.json` was measured by `modal_opgrid.py`, which the scope
cut replaced with `modal_filters.py` — same workload, same image, same
hardware. Its `provenance` field says so. `modal_filters.py` now reads
the chain arm's number from the step trace directly.
