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
| `persist_split.json` | `modal_persist.py --stage split` | the realistic-reuse protocol on stock vLLM: write the corpus through under filters 1-3, then restore under filters 4 and 5, which the store never saw, so only document KV can restore and each new question computes. Stock 2.50/2.49 s restores. The intermediate bring-up cells that developed the transfer mechanisms on the per-stage client (coalescing: 3,577 per-request jobs to 41 per-step batches, copy rate 25 to 38 GB/s; waves: claims against 1.6x duplication, the one-time per-query seam against a 77% leak, missing-blocks-only against 97 GB moved for a 48 GB job) were removed with that mode; the mechanisms and their motivating measurements live in `quail/engineext/offload.py` and `quail/engineext/waves.py`, and the mechanisms are measured end-to-end in the chain cells below |
| `persist_xfer_trace_split.jsonl.gz` | the same hook, split stock cell | per-job records for the stock split rung |
| `persist_split_quail_waves_chain.json`, `persist_xfer_trace_split_quail_waves_chain.jsonl.gz` | `modal_persist.py --stage split --connector quail --waves --client chain` | the chain client at the split stage: restores 1.82/1.58 s. Superseded by the split7 protocol below, which gives each restore two stages so rewind has real work |
| `persist_split7.json`, `persist_split7_quail_waves_chain.json` | `modal_persist.py --stage split7 ...` | the seven-filter head-to-head: write under filters 1-3, restore under 4-5 and 6-7 (two stages per restore, so rewind has real work; no filter overlap, so the store never saw a restore question). Stock vLLM (per-stage requests, stock offloading): cold 5.37 s, restores 3.30/3.02 s. The full engine (chain client with KV rewind, one-token verdicts, write-through, wave pre-loading, overlapped scheduling): cold 4.75 s, restores 1.58/1.41 s on 45.2 GB/s copies - about 2.1x faster restores than stock with identical bytes moved (53.3 GB, zero duplication). Replaces the serial-scheduling measurement (4.55, 1.76/1.34 s at 45.8 GB/s): at this scale the two configurations measure the same within container noise. Runs on the pool-derived living-session sizing (1,016 slots); the depth-derived sizing that crashed the runner (No free indices) is recorded in the log history. Copy-rate caveat for every chain cell in this table: the container pool serves hosts from ~25 to ~50 GB/s pinned-copy speed, and restores are copy-bound, so each cell quotes its measured rate; draws at the ~25 floor (three hit it on rerun day) are committed in history but not quoted |
| `persist_xfer_trace_split7.jsonl.gz`, `persist_xfer_trace_split7_quail_waves_chain.jsonl.gz` | the same hook, split7 cells | per-job records for both rungs |
| `persist_split7_10k.json`, `persist_split7_quail_waves_chain_10k.json` | the split7 protocol at 10,000 documents | the at-scale head-to-head: 3.29M tokens, 242.4 GB of KV against the 71 GB pool. Stock: write 47.44 s, restores 32.35/30.42 s in 20,002 per-document copies. Quail (chain + waves, overlapped scheduling, 0.90 utilization the one remaining self-handicap): write 44.33 s - 3.1 s faster than stock - and restores 16.27/16.26 s on 41.8 GB/s copies, 1.9-2.0x faster, moving the identical 522.5 GB with zero duplication in planned copies. Replaces the serial-scheduling measurement (write 50.90, restores 16.63/16.78 s at 47.3 GB/s): overlap removed the 7% write tax and flipped its sign. 952 living sessions slid 10,000 documents through the pool, filling 949 of 952 slots at the tightest step with zero slot incidents - admission runs on an exact runner-published snapshot (quail/engineext/slots.py: free slots and slots taken, published per applied batch; the scheduler's emitted count minus the applied count is exactly the in-flight consumption, so over-admission is impossible by construction). The rest of the slot-management chain from the serial debugging stands: pool-derived session capacity, rewound sessions at the queue front, slack-gated fresh admissions, claimed documents budgeted while pending |
| `persist_split7_quail_waves_chain_3k.json`, `persist_xfer_trace_split7_quail_waves_chain_3k.jsonl.gz` | the 3k memory-pressure validation | 73.3 GB of KV against the 71 GB pool: write 13.45 s, restores 5.59/4.86 s on 36.0 GB/s copies, 157.8 GB loaded with ~8 percent reload overhead. Replaces the serial measurement (17.30, 6.24/5.69 s) - the overlap gain is largest here (write -22%) because per-step scheduler CPU, no longer serialized with the GPU, was the biggest share of this cell's wall. Also the cell that proved the exact slot gate: emitted equals applied at every query drain in two runs on different containers, walls reproducing within 0.4 s |
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
