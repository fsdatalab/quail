# results/engine: what each file is and where it came from

This directory banks the measured GPU runs. One warning first, same
as the banner in notes/RESULTS.md: every file here was produced on
the old slim Docker image, except xengine.json's CUDA 13 arms and
the two fusegate files. The falsification flight showed the same
vLLM reads 97,220 tokens per second on a CUDA 13 devel image,
compared with the 80,556 these runs assume, so treat absolute
seconds as stale until the re-baseline flight lands
(notes/PROPOSAL.md).

KV below means the KV cache, the model's stored per-token reading
state. The producer map was verified against the experiment scripts
on 2026-08-04; experiments/modal_*.py is being refactored in
parallel, so phase names may drift after that date.

## Provenance table

| file | producer (script, phase) | what it banks | status |
|---|---|---|---|
| smoke.json | modal_engine.py, --smoke | tiny grid to prove the harness | scratch; never quoted |
| grid.json | modal_engine.py, full grid | the 2,000-document measured grid: 23 cold policy cells, 6 cold manifest cells, 4 warm cells | good |
| grid_analysis.csv | analyze_engine.py over grid.json | measured-against-ideal ratios for the grid | good |
| speed_limit.json | modal_scale.py, speed | 15 reading-rate cells, 74,000 to 81,000 tokens per second; the old 80,000 anchor | superseded: xengine.json shows the limit was the image |
| model_floor.json | modal_scale.py, floor | the floor split: matmul-only 186,000, bare model 48,000 bf16, projected 89,000 fp8 | good |
| overhead.json | modal_scale.py, overhead | per-request overhead, 2 by 2 (staged/streaming, raw/pre-tokenized): 1.12 down to 0.55 milliseconds | good |
| scale10k.json.gz | modal_scale.py, scale | the 10,000-document grid: task-first, naive, blocked | good |
| scale10k_analysis.csv | analyze_engine.py over scale10k.json.gz | measured-against-ideal for the 10k grid | good |
| client10k.json.gz | modal_scale.py, client | the client library on the 10k grid (48.9/52.5/57.9 seconds) | good |
| client10k_analysis.csv | analyze_engine.py over client10k.json.gz | measured-against-ideal for the client arm | good |
| pinned10k.json.gz | modal_scale.py, pinned | in-engine scheduler acceptance one, plus the gentle co-tenant, plus stock-engine arms | good |
| pinned10k_v3.json.gz | modal_scale.py, pinned3 | the closing contention demo: 52.17 seconds alone, 52.03 under the heavy neighbor | good; the stock-never-finished and 231-second equal-rank arms are log-only |
| strict10k.json.gz | modal_scale.py, strict | strict mode validated: 48.5/52.3/54.7 seconds, zero heuristic evictions, canary refused | good |
| chain_smoke.json | modal_scale.py, chain | truncation proof, 50 documents, 2 filters | good |
| chain4_smoke.json | modal_scale.py, chain4 | truncation proof, 50 documents, 4 filters | good |
| chain10k.json | modal_scale.py, chain10k | chain against request mode at 10k: the ACCURACY flight (carries the wrong-answer lists; walls 51.09/55.25) | good for accuracy; timing quoted from chainsteps10k.json |
| chainsteps10k.json | modal_scale.py, chainsteps | the same configuration with the step recorder on: the TIMING flight (chain 49.92 against request 52.00 seconds; the 49.9 headline) | good for timing |
| chaincore10k.json | modal_scale.py, chaincore | the same configuration under the in-process tracing profiler | MISLEADING for timing: walls are profiler-inflated (271.8 chain, 315.9 request, against the clean 49.9). Never quote it for timing |
| chainprof.json | none: ORPHAN, producer no longer in the repo | the chain-mode CPU halving at 4,000 documents: 7.06 seconds of client CPU against request mode's 16.84 | orphan; cited in notes/RESULTS.md; re-produce or retire during the re-baseline |
| chaincalib10k.json | modal_scale.py, chaincalib | the runtime-calibrated fp8 accuracy arm (14.7 percent wrong) | good |
| chainbf16_10k.json | modal_scale.py, chainbf16 | the bf16-KV accuracy arm (18.4 percent wrong, half the pool) | good |
| longdoc.json | modal_scale.py, longdoc | long-document validation: 30k and 100k token documents, the k=2 pathology, the quality cliff | good |
| longchain.json | modal_scale.py, longchain | chain mode at 100 documents of 30k tokens: 79.0 seconds, bit-identical to the one-in-flight plan | good |
| persist2000.json | modal_scale.py, persist | persisted-KV milestone one: offload 53 GB, restore 20.3 then 18.4 seconds against 10.5 recompute | good; 4B tier loses as predicted |
| model32_1k.json | modal_scale.py, model32 | the 32B tier, four arms: chain 32.0 seconds at 1.07 times the read floor | good |
| model32_probe.json | modal_scale.py, model32probe | the 32B answer-token probe behind the decisive-token gate | good |
| multigpu2.json | modal_scale.py, multigpu2 | 2 GPUs: 25.30 seconds on the 10k query | good |
| multigpu4.json | modal_scale.py, multigpu4 | 4 GPUs: 13.30 seconds | good |
| multigpu8.json | modal_scale.py, multigpu8 | 8 GPUs: 6.66 seconds, 7.50 times the single GPU | good |
| reason_grid.json | modal_scale.py, reason | the measured reasoning grid: four policies at thinking lengths 0/32/128/512 | good |
| width_scaling.json | modal_scale.py, width | admission-budget sweep at 128 thinking tokens: 95.07 seconds at 50k, flat 78.1 from 100k up | good |
| shared2000.json.gz | modal_shared.py, shared | shared scans, q = 1/2/4/8: speedups up to 5.09 | good, but the fidelity caveat blocks the claim (see notes/RESULTS.md) |
| fusegate.json | modal_fused.py, fusegate | the fp8-KV cascade gate verdict: confident flips 29/16/43 per 300 answers | good as evidence; predates the --kv naming scheme. A rerun banks fusegate_fp8.json |
| fusegate_auto.json | modal_fused.py, fusegate --kv auto | the bf16 isolation arm: confident flips 20/12, third cell fell back to unfused | good as evidence; its kv_cache_dtype field wrongly says "fp8" (the arm ran "auto", bf16). A recorder fix is in progress |
| xengine.json | modal_xengine.py, xengine | the falsification and attribution flight, six arms: control 80,556, SGLang 98,746, same vLLM on CUDA 13 devel 97,220 tokens per second | good; the anchor-moving evidence |

## Files that mislead if quoted blind

- chaincore10k.json holds profiler-inflated walls: 271.8 seconds for
  chain mode, compared with the clean 49.9 the same configuration
  measures without the profiler. It exists for its CPU tables only.
  Never quote it for timing.
- chain10k.json and chainsteps10k.json are two flights of one
  configuration. The ledger quotes accuracy (the wrong-answer lists)
  from chain10k.json and timing (49.92 against 52.00 seconds) from
  chainsteps10k.json.
- chainprof.json is an orphan: the code that produced it is no
  longer in the repo. It banks the chain-mode CPU halving, 7.06
  seconds against 16.84. Re-produce it during the re-baseline
  flight or retire the claim.
- fusegate.json predates modal_fused.py's current output naming
  (fusegate_<kv>.json). It is the fp8 run; a rerun today would write
  fusegate_fp8.json.
- fusegate_auto.json's kv_cache_dtype field says "fp8" but the arm
  ran KV dtype "auto", which is bf16. The field is wrong in the
  file; a parallel recorder fix is in progress.

## Phases that banked nothing

- pinned2 (modal_scale.py) would write pinned10k_hard.json.gz - the
  intermediate hostile co-tenant iteration (231 seconds, memory
  defended but compute lost). Never banked; those numbers are
  log-only.
- reason32 (modal_scale.py) would write reason_grid32.json - the
  32B reasoning arm. Never banked.
- run_smallN (now attic/experiments/run_smallN.py) writes no results
  file at all: it prints its tables to stdout, and its figure
  (results/plots/smallN_exact.png) is drawn by the atticked
  make_plots.py.
