# The re-baseline flight

The banked read rate of 80,556 tokens per second was a property of
our Docker image, not of vLLM: the same vllm==0.26.0 on a CUDA 13
devel base reads 97,220 (results/engine/xengine.json, vllm_new arm).
The devel base carries nvcc, so FlashInfer can JIT its kernels; the
old slim image could not. Every headline number was measured on the
slow image, so every headline number is stale.

The images in modal_scale.py, modal_engine.py, and modal_shared.py
are now on the same CUDA devel base as the vllm_new arm. Same vllm
pin (0.26.0), same env vars, same code. The toolchain is the only
variable versus the banked runs. Every post-rebase JSON carries an
`image` field (`{"base": "nvidia/cuda:13.0.1-devel-ubuntu24.04",
"toolchain": "cuda13-devel"}`); a result file without that field is
pre-rebase. One knock-on fix rides along: modal_engine.py had vllm
unpinned and is now pinned to 0.26.0 like every other file.

Nothing below launches itself. Each run is one `modal run` command,
listed in order. Run 1 first: if it does not land near 97,000, the
rebase did not take, and everything after it would re-bank stale
physics.

## The runs, in order

| # | command | banks | replaces (stale) | expect |
|---|---------|-------|------------------|--------|
| 1 | `modal run experiments/modal_scale.py --phase speed` | results/engine/speed_limit.json | 80,556 tok/s best cell | ~97,000 tok/s |
| 2 | `modal run experiments/modal_scale.py --phase chainsteps --out results/engine/chainsteps10k_r1.json` (then `_r2`, `_r3`) | chainsteps10k_r{1,2,3}.json | 49.9 s chain | low 40s, x3 for spread |
| 3a | `modal run experiments/modal_scale.py --phase scale` | scale10k.json.gz | 122.8 s task-first, 111.3 s doc-first | ~102 / ~92 s |
| 3b | `modal run experiments/modal_scale.py --phase client` | client10k.json.gz | 52.5 s | ~44 s |
| 3c | `modal run experiments/modal_scale.py --phase strict` | strict10k.json.gz | 52.3 s | ~43 s |
| 3d | `modal run experiments/modal_scale.py --phase chain10k` | chain10k.json | 51.1 s + truncated wrong lists | ~42 s + full lists |
| 4 | `modal run experiments/modal_scale.py --phase model32` | model32_1k.json | 32.0 s; 10,800 tok/s prose-only | banked prefill_tok_s |
| 5 | `modal run experiments/modal_scale.py --phase multigpu4` | multigpu4.json | 13.30 s | ~11 s |
| 6 | `modal run experiments/modal_shared.py --phase shared` | shared2000.json.gz | 22.3 / 113.7 s (5.09x) | verdict on the 5.09x |
| 7 | `modal run experiments/modal_xengine.py --arm sglang_fp8kv,sglang_acc` | merged into xengine.json | nothing (new cells) | fp8-KV rate; parity check |
| 8a | `modal run experiments/modal_scale.py --phase pinned` | pinned10k.json.gz | 49.2-66.6 s rows | same rows, ~0.83x |
| 8b | `modal run experiments/modal_scale.py --phase pinned2` | pinned10k_hard.json.gz | "never finished" (log-only) | a banked timeout row |

## What each run settles

**1. Speed control.** The new anchor. Expect about 97,000 tokens per
second against the banked 80,556, matching the xengine vllm_new arm's
97,220 on this exact base. Every constant downstream reprices off
this number. If it lands near 80,000 instead, stop the flight and
check the image build.

**2. Chain, three repetitions.** The headline cell. Expect low 40s
against the banked 49.9 seconds: the read work that took about 47
seconds at 80,556 tok/s takes about 39 at 97,220, plus the ~3-second
software residue. The phase has no repetition support, so run it
three times with distinct `--out` names (r1, r2, r3); the paper
quotes the mean and the spread. Each run also banks request mode
(~52.0 stale) next to chain mode, so the chain-vs-request gap gets
repetitions for free.

**3. Ladder endpoints.** Four phases, one engine boot each. The
scale phase re-banks the naive cells at n=4, s=0.8: task-first waves
122.8 seconds and doc-first waves 111.3 (the 74.0 manifest cell
rides along). Client re-banks 52.5, strict 52.3. The chain endpoint
(49.9) comes from run 2; run 3d re-flies the chain10k file because
it now banks the FULL wrong-answer and flip lists - the banked file
cut them at 50 entries, so the paper's 12.3-percent wrong-answer
claim had no artifact. Expect every wall down by roughly the
toolchain ratio (80,556 / 97,220 = 0.83). The check that matters:
the ladder ORDER and the ratios between arms should survive; that
was an expectation until this flight measures it.

**4. The 32B tier.** Re-banks model32_1k.json. Each arm now carries
`prefill_tok_s` (prefilled tokens over wall), so the 32B rate -
10,800 tok/s, until now prose-only - becomes a banked field. Expect
the chain wall at or below the stale 32.0 seconds. Direction is
down but the size is unknown: the 32B is more compute-bound than
the 4B, so its toolchain share should be smaller than the 4B's 21
percent. Whatever it measures becomes the 32B anchor.

**5. One scaling point.** Re-banks multigpu4.json. Expect about 11
seconds against the stale 13.30 if per-shard reads scale with the
new rate. The claim to restate is efficiency against the NEW
one-GPU chain number from run 2, not against 49.9.

**6. Shared-scan fidelity discriminator.** The gate on the 5.09x
multi-query claim. Both arms now sample with logprobs=2; per-call
records (query, doc, stage, token id, top-2 gap in nats) are banked
for both arms, and a per-q classification table is printed and
banked: agreements, confident flips (tokens differ, BOTH gaps above
0.2 nats), near-ties (either gap at or under 0.2). Zero confident
flips: the 5.09x ships with a measured near-tie tolerance. Any
confident flips: there is a routing bug to find first, and the
number appears nowhere until it is found. Read the `nogap` column
before the verdict: if it is large, the chain rewind is dropping
logprob tables and the instrument needs a fix before its output
means anything. Expect both walls to drop and the ratio to
roughly hold; the ratio is same-image both times.

**7. Cross-engine completeness.** Two new arms merge into
xengine.json. `sglang_fp8kv` fills the missing speed cell: the
original sglang arm wrote KV at 'auto' (bf16) while the vllm
control wrote fp8 KV; expect at or slightly below sglang's 98,746,
since fp8 KV quantizes K and V on the write path. `sglang_acc`
answers the parity question: the standard planted-flags query
(2,000 docs, 4 filters at s=0.8, modal_scale's exact seeds and
prompt bytes) as plain per-stage requests, scored against planted
truth, wrong counts and survivors banked. Expect a wrong fraction
near vLLM's 12.3 percent if answer parity holds; a big gap either
way is a finding.

**8. Co-tenant re-bank.** 8a re-banks acceptance one plus the
gentle neighbor (25 requests per second of 800 junk tokens);
junk_stats now include offered counts and mean neighbor latency,
not just completions. 8b is the heavy neighbor (60 per second of
1,500 tokens) that produced the paper's "stock vLLM never finished
inside an hour" - a claim that today exists only in logs, because
the stock arm died with its container. Junk arms now run under a
900-second cap and bank `{finished: false, timeout_s: 900}` on
expiry, so the honest, bankable form - "did not finish within 15
minutes, while the pinned engine finished in ~52 seconds" - gets
its artifact. Expect the pinned rows to finish near their banked
walls times 0.83, and the stock heavy-junk row to time out.

## Constants to update after the flight

The planner's constants live in one place now: the calibration
table in docengine/plan/cost.py. Update it from the banked JSONs,
in the same commit as the JSONs:

- PHI, the serving rate over the 275,000 tok/s spec ceiling. Today
  80,000 / 275,000 = 0.291 on the stale anchor. Recompute from run
  1's best cell: about 97,000 / 275,000 = 0.353 if the expectation
  holds.
- ENGINE_OVERHEAD_S, the per-query software residue. Today 3.2
  seconds. Recompute from run 2: mean chain wall minus scheduled
  tokens at the new rate, over the three repetitions.
- The 32B anchor the cost model cites (10,800 tok/s in comments)
  becomes run 4's banked prefill_tok_s.

Then regenerate what renders numbers:

- plots/make_plot_story.py hardcodes every number; left alone it
  silently plots stale seconds. Update it (or point it at the new
  JSONs) and re-render.
- plots/make_plot_engine.py re-runs against the new result files.
- notes/RESULTS.md gets each run banked in the same commit as its
  JSON, stating which file backs which number.

modal_engine.py (the 2,000-document grid behind grid.json) is
rebased and pinned too, but none of its cells are headline numbers;
re-fly it only if the small grid gets cited again.

## Cost

Estimates from the banked walls plus boot, tokenize, and model-load
overhead, at 0.83x for the measured parts. One H100 per container.

| run | GPU minutes |
|-----|-------------|
| 1 speed | 30 |
| 2 chain x3 | 25 |
| 3a scale | 25 |
| 3b client | 12 |
| 3c strict | 10 |
| 3d chain10k | 8 |
| 4 model32 | 18 |
| 5 multigpu4 | 28 (4 GPUs x ~7 wall) |
| 6 shared | 12 |
| 7 xengine arms | 13 |
| 8a pinned | 32 (2 GPUs in parallel) |
| 8b pinned2 | 34 (2 GPUs; stock arm rides its 15-minute cap) |

Total: about 250 GPU minutes, call it 4 H100-hours. Sequential wall
time about 3 hours. First launch of each app also pays a one-time
image build of the CUDA devel base (about 10 minutes each, on
Modal's builders, not on the GPU meter); each phase's warmup
requests absorb the first-container FlashInfer JIT.
