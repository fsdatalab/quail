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
| `persist_store.json` | `modal_persist.py --stage store` | restore against recompute, 1k docs |
| `pinprobe.json` | `modal_pinprobe.py` | host-to-GPU bandwidth for three pinned-memory paths |
| `speed_limit.json` | prefill throughput sweep | the 97,889 tok/s anchor `quail/plan/cost.py` calibrates PHI against |
| `calibrate_all.json` | `modal_calibrate.py --families all` | the calibration sweep the fits use: 100 step cells (alpha, c1, c2, c4, c5) plus the c6 transfer probes, one container, synchronous eager boot |
| `cost_model_fit.json` | `quail.plan.fit` | the fitted step, host, and prefill models, with t_read, eps, and the offload-vs-recompute crossover per transfer tier |
| `calibrate_alpha.json`, `calibrate_c1.json`, `calibrate_c2-c4-c5.json`, `calibrate_c6.json` | `modal_calibrate.py --families <fam>` | the staged gate-check runs, one container each; the fits read only `calibrate_all.json` because rows from different containers must not mix |
| `attnshare.json` | `modal_profiling.py::attnshare` | kernel-class shares of single-document prefills by length on the calibration boot, with each length's top kernels by time; says where attention overtakes the GEMMs |
| `c0_anchor.json` | `modal_filters.py::c0_anchor` | the filter comparison with an in-container rate probe; each cell carries c0 = wall - reads x corpus / rate, so host speed cancels |
| `makespan_check.json` | `quail.plan.validate` | predicted against measured query walls per arm, at designed and at effective (verdict-measured) selectivities, plus a container variant at the anchor's probed rate and c0 |
| `filter_cells_bf16.json` | `modal_filters.py::main --kv bf16` | the filter comparison on a 16-bit-KV boot, admission repriced at 2 bytes per element; accuracy identical to fp8, so the KV format does not cause the answer anomaly |

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
