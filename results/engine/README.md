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
| chain10k.json | modal_scale.py, chain10k | chain against request mode at 10k: the ACCURACY flight (carries the wrong-answer lists; walls 51.09/55.25) | good for accuracy; timing quoted from chainsteps10k.json |
| chainsteps10k.json | modal_scale.py, chainsteps | the same configuration with the step recorder on: the TIMING flight (chain 49.92 against request 52.00 seconds; the 49.9 headline) | good for timing |
| profile.json | modal_scale.py, profile | the CPU profile at 4,000 documents: the ranked table behind the request-toll section (deepcopy 22 percent, telemetry 10 percent) | good |
| chaincalib10k.json | modal_scale.py, chaincalib | the runtime-calibrated fp8 accuracy arm (14.7 percent wrong) | good |
| chainbf16_10k.json | modal_scale.py, chainbf16 | the bf16-KV accuracy arm (18.4 percent wrong, half the pool) | good |
| longdoc.json | modal_scale.py, longdoc | long-document validation: 30k and 100k token documents, the k=2 pathology, the quality cliff | good |
| longchain.json | modal_scale.py, longchain | chain mode at 100 documents of 30k tokens: 79.0 seconds, bit-identical to the one-in-flight plan | good |
| persist2000.json | modal_scale.py, persist | persisted-KV milestone one: offload 53 GB, restore 20.3 then 18.4 seconds against 10.5 recompute | good; 4B tier loses as predicted |
| model32_1k.json | modal_scale.py, model32 | the 32B tier, four arms: chain 32.0 seconds at 1.07 times the read floor | good |
| multigpu2.json | modal_scale.py, multigpu2 | 2 GPUs: 25.30 seconds on the 10k query | good |
| multigpu4.json | modal_scale.py, multigpu4 | 4 GPUs: 13.30 seconds | good |
| multigpu8.json | modal_scale.py, multigpu8 | 8 GPUs: 6.66 seconds, 7.50 times the single GPU | good |
| reason_race.json | modal_scale.py, reasonrace (a one-shot phase, removed with the fix; in git history) | the speculation race study at g=0, five arms with corpus read multipliers: spec_race (the old simultaneous launch) reads the corpus 2.41 times and takes 19.51 s; spec with the fix reads 1.33 and takes 10.15 against the pipelined arm's 9.27 | good; new image; the spec_race cell is the deliberate record of the deleted bug, not a quotable policy cost |
| width_scaling.json | modal_scale.py, width (removed with the reasoning-filter instruments, 2026-08-05; in git history) | admission-budget sweep at 128 thinking tokens, re-measured on the new image: 65.49 seconds at 50k, 58.69 at 100k, flat to 700k; saturation stays near 100k | good; new image (old-image run: 95.07 at 50k, flat 78.1 from 100k) |
| spec_smoke.json | modal_scale.py, specsmoke | the fork validation: 500 documents, 4 filters, selectivity 1; pipelined 2.84 s / forked speculation 3.32 s / sequential speculation 3.00 s; fork agrees with its sequential control on 1,997 of 2,000 answers; siblings share the document's physical blocks | good; new image; re-banked 2026-08-05 (the earlier three-run parity content is in git history) |
| spec_where.json | modal_scale.py, specwhere (a one-shot phase, removed after banking; in git history) | the 27-cell sweep: selectivity 0.5/0.8/0.95 x budget 10k/50k/700k x three policies; pipelined wins every cell; identical reads (1.14) for the two in-engine policies; turnover, not batch width, is the starved-regime knob | good; new image |
| underfill.json | modal_scale.py, underfill | the underfilled-round cell: hybrid (gate then fork survivors) beats pure gating 0.1374 against 0.1451 s at 20 docs x 6 filters; classifier speculation ties classifier pipelining; mid-chain switch no-loss at 500 docs | good; new image; walls only - re-check answer agreement across reps before quoting accuracy |
| shared2000.json.gz | modal_shared.py, shared | shared scans, q = 1/2/4/8: speedups up to 5.09 | good, but the fidelity caveat blocks the claim (see notes/RESULTS.md) |
| fusegate.json | modal_fused.py, fusegate | the fp8-KV cascade gate verdict: confident flips 29/16/43 per 300 answers | good as evidence; predates the --kv naming scheme. A rerun banks fusegate_fp8.json |
| fusegate_auto.json | modal_fused.py, fusegate --kv auto | the bf16 isolation arm: confident flips 20/12, third cell fell back to unfused | good as evidence; its kv_cache_dtype field wrongly says "fp8" (the arm ran "auto", bf16). A recorder fix is in progress |
| xengine.json | modal_xengine.py, xengine | the falsification and attribution flight, six arms: control 80,556, SGLang 98,746, same vLLM on CUDA 13 devel 97,220 tokens per second | good; the anchor-moving evidence |

## Files from the operator-grid second pass (2026-08-08)

| file | producer | what it banks | status |
|---|---|---|---|
| opgrid_maps_4b_pair.json (+ trace opgrid_maps_trace_new_4b_pair.jsonl.gz) | modal_opgrid.py, opgrid_maps, boots new,stock | the SAME-CONTAINER map pairs at caps 64 and 256: ours 139.1/500.5 against stock 130.8/480.0; the two banked losses | good; the only valid ours-against-stock map comparison (host variance falsified cross-container walls) |
| opgrid_maps_4b_new.json (+ traces) | modal_opgrid.py, opgrid_maps, boots new | the ours-only re-fly on a slow host: cap 16 70.9 s, cap 64 159.2, cap 256 637.7 with NO engine death | good for the survival claim and the host-variance evidence; do not quote its walls against stock |
| opgrid_filters_4b_permissive_classifier_stock.json | modal_opgrid.py, --classifier --stock-only | stock classifier baseline: 167.8-169.8 s, reads 4.94 | good for walls; wrong counts carry the 0.88 caveat |
| opgrid_filters_32b_permissive_classifier_stock.json | same, 32B, 1 rep | stock classifier baseline: 1,253.2 s, reads 5.72, wrong 138 (clean) | good |
| opgrid_filters_4b_cliff_stock.json | same, cliff | stock cliff baseline at 0.88: 48.8-50.9 s | good for walls; wrong 7,893 carries the 0.88 caveat, biased in stock's favor |
| pinprobe.json | modal_pinprobe.py | the channel calibration: 55.4 GB/s alloc-pinned, 9.7 unpinned, register works on anonymous mmaps and fails (304) on file-backed ones | good; the design doc's channel table |
| persist32_1k_cpu.json | modal_scale.py, persist32 (two containers, PLAIN CPU spec: pinned pool) | 32B restore 4.04-4.16 s against recompute 27.7 (6.8x win); channel 10.23 GB/s; write-through at ingest free; outcomes identical | good; the tiering spec's file-backed region is unpinnable here and its unpinned path died natively - use the plain spec for CPU-only stores |
| persist4b_pin_1k_cpu.json | modal_scale.py, persist4bpin | the 4B confirmation on the pinned channel: restore 2.28-2.39 s against recompute 4.54 (2x win, channel 10.22 GB/s) | good; REVERSES persist2000.json's "4B loses" - that conclusion was channel-bound. Caveat: outcomes_identical_within_store false (the checkpoint's borderline-call noise; 32B was fully identical) |
| opgrid_maps_32b.json (+ trace) | modal_opgrid.py, opgrid_maps 32b, cap 16, boots new,stock | the same-container 32B map pair: 356.3 against 356.1 s, reads 1.36 both - a tie at the compute floor | good; cap 64 dropped by decision, stated in notes/OPGRID2_PREDICTIONS.md |

## Files that mislead if quoted blind

- chain10k.json and chainsteps10k.json are two flights of one
  configuration. The ledger quotes accuracy (the wrong-answer lists)
  from chain10k.json and timing (49.92 against 52.00 seconds) from
  chainsteps10k.json.
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

## Files retired on 2026-08-04

smoke.json (scratch), chain_smoke.json and chain4_smoke.json (the
50-document truncation proofs; the ledger's chain-proof section
quotes no numbers from them), chaincore10k.json (profiler-inflated
walls, kept only for CPU tables that nothing cites),
chainprof.json (orphan; its CPU-halving claim was retired), and
model32_probe.json (the 32B answer-token probe; uncited). All are
in git history, last present at commit 8f4561f.

## Files retired on 2026-08-05

reason_grid.json: the measured reasoning grid, four policies at
thinking lengths 0/32/128/512, old image. Its speculation and
lookahead cells were an instrument artifact (the prefill race, see
reason_race.json), so the file was deleted rather than left
quotable. The pipeline and waves rows were valid old-image
measurements; they survive as prose in notes/RESULTS.md. Re-fly
the grid with the fixed client before citing any speculation cell.
