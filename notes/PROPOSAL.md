# Proposal: the next steps, and what the refactor should be

Written 2026-08-04 after reading everything in the repo: both papers,
all nine notes, the 50-commit history, the result files, and every
Python module (audited module by module; all 68 tests pass in 2.2
seconds on CPU). This note answers two questions. What should we do
next? And do we need a massive refactor?

Short answers. Next: run the re-baseline flight before anything else,
because every headline number is calibrated against a reading rate
that moved 21 percent; then close the three claims that block the
paper (shared-scan fidelity, planner refusal path, SGLang
completeness); then revise the paper around the evidence that already
exists. Refactor: yes, but it is a week of surgery, not a rewrite.
About 46 percent of the package serves the superseded theory paper
and should move to an attic; the live half needs one cost model
instead of six, tests for the one untested load-bearing file, and
packaging so the repo installs like a normal Python project.

## 1. Where the project actually stands

One repo holds two papers and two generations of code.

- **Generation 1, the theory program.** `paper.md` ("Scheduling
  n-Stage AI Filters Under KV-Cache Constraints") plus the code that
  implements it: the exact solvers (`reference/`), the expected-flow
  linear program (`optimizer/`, `runtime/replay.py`), the schedule
  builders (`sched/blockwise.py`), the independent checker
  (`validator/check.py`), and the resource bounds (`lb.py`). This
  program is finished. Its three computed layers agree to 0.56
  percent, its review (`notes/REVIEW.md`) found and fixed four real
  errors, and its results fill the first half of `notes/RESULTS.md`.
- **Generation 2, the measured system.** `notes/PAPER.md`
  ("Plan-Governed KV State for Semantic Scans") plus the code that
  implements it: the vLLM scheduler subclass (`engineext/scheduler.py`),
  the client library (`runtime/engine_client.py`), the planner
  (`plan/planner.py`), the reasoning cost model (`reasoning/`), and
  the Modal experiment harness (`experiments/modal_*.py`). This is
  the SIGMOD/VLDB paper. Its measured ladder runs from 122.8 seconds
  (naive prompts through stock vLLM) down to 49.9 seconds (chain
  mode), against a 47-second work estimate; the 32B tier lands at
  1.07 times its read estimate; scaling holds 94 percent efficiency
  at 8 GPUs.

The confusion ("what are we doing?") is real and has a specific
cause: the repo's layout, module names, and half its notes still
describe Generation 1, while all fifty commits of the last two days
work on Generation 2. Nothing is broken. The story is just buried.

### What moved out from under the paper (why numbers cannot ship as-is)

1. **The calibration anchor moved 21 percent.** The whole measured
   story is priced against "vLLM reads 80,000 tokens per second."
   The falsification flight showed SGLang reading the same corpus at
   98,700, and the attribution arms proved the gap is the toolchain,
   not the architecture: the same vLLM on a CUDA 13 image where
   FlashInfer can compile its kernels reads at 97,220. So 80,556 was
   never a property of the hardware or the engine; it was a property
   of our Docker image. Every measured second, every "1.07x the
   floor" ratio, and the planner's hardcoded constant
   (`PHI = 80_000 / 275_000` in `plan/planner.py`) now sits on a
   stale anchor. The word "floor" is dead too: it was an achieved
   rate, and it was beaten.
2. **Fusion flipped from contribution to negative result.**
   `notes/PAPER.md` still lists the fused-fork operator as a headline
   contribution with three experiments (E5-E7) built around it. The
   gate said no: the cascade kernel flips answers confidently (past a
   0.2-nat probability gap) at every document length on fp8 KV — 29,
   16, and 43 flips per 300-answer cell at 300, 3,000, and 15,000
   tokens — and still flips on the bf16 isolation arm (20 and 12;
   the third cell silently fell back to unfused, so its zero proves
   nothing). The fused path was also far slower at the measured
   scale. Fusion on this stack is closed, with a two-format evidence
   chain that is effectively an upstream bug report. The paper must
   be restructured around that fact, and the gate itself
   (bit-identity checking against unfused execution) becomes part of
   the fidelity method.
3. **The shared-scan claim is not yet safe.** 8 queries in 22.3
   seconds against 113.7 separate is a 5.09x headline. But the
   banked run shows survivor sets diverging in 6 of 8 queries (up to
   30 documents per query) and 123 more wrong answers in shared mode
   (6,519 against 6,396 of 34,027 scored calls). That is too large to
   wave off as borderline noise without the confident-flip
   discriminator (rerun with logprobs, classify each disagreement by
   probability gap at 0.2 nats). Near-ties: the claim ships with a
   measured tolerance. Confident flips: there is a routing bug to
   find first.
4. **The paper skeleton is a day behind its own evidence.** E2's
   "planned" 8-GPU point is banked (6.66 seconds, 7.5x). E4's
   "planned" hardware run is substantially banked (the forced-length
   reasoning grid at four thinking lengths, plus a width sweep).
   Neither appears in `notes/PAPER.md`, and `notes/RESULTS.md` stops
   six campaigns ago: the 8-GPU point, the reasoning grid, the width
   sweep, the fusion verdict, shared scans, and the falsification
   flight exist only as commit messages and raw JSON.

A full cross-check of `notes/RESULTS.md` against the raw files is
reassuring about the ledger itself: essentially every quoted number
matched the banked data exactly — makespans, read multipliers, cache
hit fractions, survivor counts, per-request floors. The discipline
held. The cross-check also found the gaps that matter:

- **Some headline claims are log-only.** The 12.3-percent
  wrong-answer counts (the files truncate the wrong lists at 50
  entries), the hostile co-tenant result ("stock vLLM never finished
  inside an hour" — cited in the paper skeleton's engine
  contribution), the chain-mode token accounting (the 353,654-token
  rewind deficit), and the 32B 10,800 tokens-per-second prefill rate
  exist in prose and stdout only, with no banked artifact. They must
  be re-banked during the re-baseline flight or dropped from the
  paper.
- **Two files mislead and one is orphaned.**
  `results/engine/chaincore10k.json` holds profiler-inflated walls
  (271.8 seconds against the clean 49.9) that nothing labels as
  such; `chain10k.json` and `chainsteps10k.json` are two flights of
  the same configuration and RESULTS.md quotes timing from one and
  accuracy from the other without saying so; `chainprof.json` has no
  producer left in the repo — the code that made it is gone, and its
  numbers (chain mode halves client CPU) are cited in RESULTS.md.
- **Four small precision wobbles** to fix when RESULTS.md is brought
  current: the replay-vs-target bound is 0.56 percent, not "within
  0.5"; the 2,000-document grid has 23 cold cells, not "all 27"; the
  "100 versus 53 batches" pair is quoted across two different
  experiments; "16.56 to the digit" holds for the mean of two
  replicates (16.5887 and 16.5267), not for either run.

## 2. The refactor, right-sized

The instinct that something needs restructuring is correct. The size
estimate is not. The package is about 5,000 lines, the tests are
green, and imports flow in one clean direction. What is wrong is
duplication and dead weight, and both have sharp edges the audit
located exactly. A week of surgery, in six moves, most of it
CPU-only and safe to do while GPU flights run.

**Move 1: packaging before anything else (half a day).** There is no
`pyproject.toml`, no `requirements.txt`, no CI. Ten experiment files
carry the same `sys.path.insert` hack, and the test files use two
incompatible import styles, so plain `pytest` fails collection while
`python -m pytest` works. Add `pyproject.toml` (with pinned deps
matching the Modal image: `vllm==0.26.0` as an extra), a
`tests/conftest.py`, and a CI job that runs the 68 CPU tests. Every
later move gets safer once this exists.

**Move 2: attic the Generation 1 stack (one day).** Move to `attic/`
(out of the installed package, tests still collected and green):
`optimizer/` (735 lines), `reference/` (447), `runtime/replay.py`
(327), `manifest.py`, `reprice.py`, `cluster.py`, the five frozen
experiments that drive them (`run_lp`, `run_n10k`, `run_multi`,
`run_additive`, `run_smallN`), their tests, and the three plot
scripts that read their CSVs. Do not delete any of it: the new paper
still cites this machinery ("an independent validator that replays
every batch", the exact-solver evidence behind E8's small-N claim),
and `notes/RESULTS.md`'s first half is its evidence. Keep in the
main package: `configs.py` (imported by 19 modules), `costmodel.py`,
`instance.py`, `lb.py` (the live `analyze_engine.py` uses it),
`sched/blockwise.py` + `validator/check.py` as a pair (the
manifest-mode arm and measured-vs-ideal analysis still run them),
`reasoning/`, `plan/`, `runtime/engine_client.py`, `engineext/`.
Keep `engineext/fused.py` too, labeled as the banked negative: it is
the bug report.

**Move 3: one cost model (one to two days).** There are six pricing
rules today: `costmodel.tau`, the validator's deliberate independent
copy, a verbatim third copy inside `sched/blockwise.py`, the
no-overlap variant in `reprice.py`, the decode-aware model in
`reasoning/model.py`, and the planner's bare rate model. Meanwhile
Algorithm 1 in `notes/PAPER.md` — the cost estimator the paper
claims, with decode rate and thinking length as inputs — exists in
no single module. Write it once (say `plan/cost.py`), with the
calibrated constants taken from the re-baseline flight, and make
`plan_query` and `reasoning/model.py` consume it. Delete the copy
inside `blockwise` in favor of importing `costmodel`. Keep the
validator's copy: its independence is the point.

**Move 4: one import path, one dispatcher (half a day).** Delete the
4-line `docengine/planner.py` shim (two callers to fix). Merge
`exec.run_plan` and `engine_client.run_query` into one entry point,
and make `modal_scale.shard_worker` call it instead of reaching past
the planner to `run_filter_chain_engine` — today the executor facade
is tested but bypassed by the code that matters.

**Move 5: tests for the scheduler subclass (one day).**
`engineext/scheduler.py` is 430 lines, is referenced by ten call
sites in the live experiments, reaches into private vLLM internals
pinned at 0.26.0, and has zero tests, because it imports vLLM at
module top. Factor the pure logic (tag and directive parsing, pin
refcounting, the rewind boundary-block arithmetic) into a
vLLM-free module the subclass imports, and test it with stub
harnesses in the style of `tests/test_client.py` — the best test
file in the repo. This is the highest-risk untested code we own, and
the piece a vLLM version bump will break first.

**Move 6: make the documents match the project (one day).** Promote
`notes/PAPER.md` to `paper/PAPER.md` as the declared product; move
`paper.md` to `attic/theory-paper.md` with a header saying what
superseded it and what survives (its retention theory lives on as
Section 6; its solver semantics as the attic's validation layer).
Add a root `README.md`: the one-paragraph story, the module map, how
to run tests and how to launch a Modal flight. Bring
`notes/RESULTS.md` current (the six missing campaigns), and from
now on bank every flight there in the same commit as its JSON.
Archive `notes/PLAN.md` and `notes/ENGINE_PLAN.md` (both describe
finished phases) under `notes/archive/`. Do a terminology pass:
`RESULTS.md` and `SCHEDULER_PLAN.md` still call KV "notes", which
the repo's own style rule now bans. Fix
`plots/make_plot_story.py`, which has every number hardcoded — after
the re-baseline it would silently plot stale seconds. And clean the
results directory while at it: label `chaincore10k.json` as
profiler-inflated (or move it aside), state which of
`chain10k.json` and `chainsteps10k.json` backs which claim, delete
or re-produce the orphaned `chainprof.json`, and adopt one rule
going forward — no claim in a note or the paper without a named,
banked results file, banked in the same commit as the prose.

What NOT to do: rewrite the engine client or the scheduler (they are
measured, tested or about to be, and they are the paper); delete the
Generation 1 stack (banked claims cite it); or hold the GPU work
hostage to the refactor. The refactor and the flights can run in
parallel — the only ordering constraint is that Move 3's constants
come from the re-baseline flight.

## 3. The ordered plan

**Step 0 (now): pin the decisions.** The paper is `notes/PAPER.md`.
`paper.md` is its theory companion, archived. Fusion is a negative
result, in the paper as evidence, not contribution. Every rate is an
"achieved rate" against the spec-sheet bound; the word "floor" is
reserved for proven bounds (Proposition 1).

**Step 1 (first GPU session, ~1 hour): the re-baseline flight.**
Rebuild the experiment image on the CUDA 13 devel base so FlashInfer
compiles its kernels everywhere. The scope is wider than it first
looks: every banked measured number — including the brand-new
8-GPU, reasoning-grid, width-sweep, and shared-scan results — was
produced on the old slim image (`modal_scale.py`, `modal_engine.py`,
and `modal_shared.py` all build on `debian_slim`; only the xengine
attribution arms and the fusion gate used the new base). Ratios
between arms on the same image (the 5.09x, the 7.5x) should survive
better than absolute seconds, but that is an expectation, not a
measurement. Re-run the headline cells: the speed-limit control
(expect about 97,000 tokens per second), the 10k chain run (expect
low 40s from 49.9), the ladder endpoints, the 32B tier, one scaling
point — with three repetitions on the headline cells so the paper
can quote spread. Re-bank the log-only claims in the same flight:
the hostile co-tenant arm, the full wrong-answer lists, the 32B
prefill rate. Recalibrate `PHI` and the 3.2-second overhead constant
from the new runs. Keep the six-arm xengine table as the attribution
evidence. Everything downstream restates against these numbers, so
this flight goes first.

**Step 2 (same week): close the three blocking claims.**
- **Shared-scan fidelity.** Rerun shared-vs-separate at q=8 with the
  confident-flip discriminator. If flips are confident, find the
  routing bug before the 5.09x number appears anywhere.
- **Planner refusal path.** Implement Algorithm 2's lines 3 and 8 in
  `plan_query` (it currently proposes plans for hardware that does
  not exist and clamps a negative pool to zero). Small change,
  removes a stated code-paper divergence, unblocks E12.
- **SGLang completeness.** The fp8-KV speed cell and a truth-scored
  accuracy arm, closing the port question with parity known.

**Step 3: the surgical refactor (Moves 1-6 above), interleaved.**
Moves 1, 2, 4, 5, 6 need no GPU and can land while flights run.
Move 3 lands right after Step 1 delivers the constants.

**Step 4: the remaining experiment roster, smallest first.**
E6 (fork/no-sharing ladder, reframed without the fused arm), E8
(interactive boundary), E9 (retention against the clairvoyant
reference), E10 (planner end to end: every emitted plan beats the
alternatives it rejected), E11 (wrong-prior re-planning), E12
(device transfer to L40S, including the refusal). Re-number the
roster while revising the paper: E2 is done through 8 GPUs, E4's
hardware anchor is banked, E5 is now the fusion negative, and shared
scans need an experiment number.

**Step 5: the paper revision pass.** Fold in, in this order: the
re-baselined numbers with achieved-rate language; the falsification
and attribution story (it is a strength: we tried to break our own
baseline, broke it, and explained the break); fusion as a documented
negative with the gate as method; shared scans (pending Step 2's
verdict); the updated roster and the four stated code-paper
divergences (refusal path resolved by Step 2, block grouping and
mode-argmin kept as stated rules). Then the related-work reading
pass with receipts (Sarathi, Hydragen, SGLang, Parrot, LOTUS,
Palimpzest, DocETL, vAttention, CacheGen, CacheBlend). Consider one
added arm reviewers will ask for: a LOTUS-style operator layer
driving a stock engine on the reference query, as the external
baseline the E1 naive arms approximate.

**Parked, with triggers, unchanged from NEXT.md:** hybrid-attention
models (allocator work), fused speculation (awaits the upstream
kernel fix; our evidence chain is the bug report), joins (after the
scan story closes).

## 4. Risks worth naming

- **The shared-scan divergence may be a real bug.** Survivor drift
  that grows with query count looks like state, not noise. Treat the
  discriminator run as a bug hunt, not a formality.
- **Model accuracy is the reviewable soft spot.** 12.3 percent of
  calls disagree with planted truth (identically across scheduling
  modes), and long-context agreement falls to 0.63 at 100k tokens.
  The paper's posture — scheduling is bit-faithful to the one-call
  reference; accuracy is the model's — is right, but one arm with a
  stronger model would blunt the review.
- **The vLLM 0.26.0 pin is a wasting asset.** The scheduler subclass
  reaches into private internals. Move 5 (extract and test the pure
  logic) is the mitigation; an upgrade attempt is its own small
  flight, not a side effect.
- **Single-run cells.** Most measured cells are one run. The
  re-baseline flight should repeat the headline cells (three runs
  each is enough to quote spread) so the paper can state repetition
  statistics.

## 5. What this replaces

This note supersedes `notes/NEXT.md`'s ordering by wrapping it in
the refactor and the paper decisions; NEXT.md's seven items all
survive inside Steps 1-5. `notes/PLAN.md` and `notes/ENGINE_PLAN.md`
describe completed phases and move to the archive in Move 6.
