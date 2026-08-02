# The analytical model for reasoning filters

This note extends the paper's cost model to filters that think before
answering, and defines the three objects phase A computes before any
hardware run: the cost primitives, a Bellman value recurrence per
policy, and a throughput linear program per policy. Stage j's filter
now generates g_j tokens (thinking plus the final answer token), with
g_j = 1 recovering the answer-only filters measured so far.

## Assumptions

- Thinking length is modeled by its mean per stage. Randomness in
  length changes tail behavior, not the fluid quantities computed
  here, and is a later refinement.
- A filter call's thinking tokens are private to the call. The next
  filter forks from the document's notes; it never reads another
  call's thinking. Thinking notes are freed the moment the call ends
  (the phase C engine does exactly this).
- A wasted speculative branch wastes its full generation. Aborting a
  branch mid-thought when an earlier answer arrives saves nothing
  when stages think for similar lengths, so full waste is the honest
  default and an abort model is an optional refinement.
- Calibration multiplies the compute rate only. The measured 80,000
  tokens per second replaces the 275,000 ceiling through a factor
  phi; the bandwidth terms already matched measurement uncalibrated.

## Cost primitives

Reading (prefill) is priced as in the paper: U tokens cost
2 P U / (phi R_D) seconds of compute.

Generation is stepwise. A step that advances m concurrent calls by
one token each costs

    step(m, c) = max( 2 P m / (phi R_D),
                      (W_run + kappa c m) / BW )

seconds, where c is the mean context length (document plus question
plus half the thinking so far). The first argument is the compute
cost of m tokens; the second is the bandwidth cost of reading the
weights once plus each call's notes. Small cohorts are weight-read
bound, large cohorts become attention-read bound, and the crossover
sits near m = 330 calls for this model and card. Generating g tokens
for a cohort of m calls costs g times step(m, c).

Memory. A call in flight holds its prompt notes plus up to g
generated tokens: kappa (p + g) transient per call on top of whatever
the policy retains. This is the term that shrinks blocks: a document
block under lookahead k must fit d + k (p + g) tokens per document,
so thinking divides block sizes and with them the decode cohort.

## Bellman recurrences

Both recurrences report a makespan estimate as the maximum of three
certified components: total compute seconds, total bandwidth seconds,
and the critical (dependency) path. The first two are the paper's
resource bounds extended with generation; the third is the gate
structure that resources cannot hide.

Task-first is a chain over stages. Stage j serves N_j = N prod s
survivors; each re-reads its task prompt, document, and cue, then
generates g_j in cohorts of m_j = min(N_j, capacity / per-call
footprint). V_j = T_j + V_{j+1} with V after the last stage zero.

Document-first is a chain over stage compositions. A composition
partitions the n stages into consecutive blocks of lookahead sizes
(k = all ones is the pipeline, k = n is full speculation). Documents
move in memory-sized blocks of B(K) = capacity / (d + max_t k_t (p +
g)) documents; a block prefills its documents once, then per
composition block issues all k_t branch questions ungated (paying
for branches of documents that a within-block failure kills) and
generates with cohort b_t k_t. The Bellman recurrence over stage
index chooses the optimal composition:

    V[j] = min over k of  C(j, k) + V[j + k],   V[n + 1] = 0

where C(j, k) is the per-block cost of running stages j through
j + k - 1 at lookahead k, including the waste of branches issued for
documents that fail inside the block and the block shrinkage that k
and g cause. The three fixed policies are the three fixed
compositions; the recurrence also reports the optimal one.

## The throughput linear program

Per policy, steady state, documents flowing at rate lambda. Each
document consumes, in expectation over survival: prefill tokens,
generated tokens per stage, weight reads per generated token divided
by the decode cohort m, and note reads per generated token. Three
resource constraints bound lambda:

- compute: 2 P (prefill rate + decode rate) <= phi R_D
- bandwidth: W_run (decode rate / m) + kappa (context reads rate +
  prefill note traffic) <= BW
- memory: lambda times the per-document resident token-seconds
  (document notes for its span, transient prompt plus thinking per
  call) <= capacity in tokens

The cohort m and the residency span depend on the rates, so the
program is solved on a small grid of cohort sizes with a fixed point
on the span (two or three iterations suffice), and the reported
lambda is the best feasible. At g = 1 the program must agree with the
existing expected-flow program's ceiling, which is the consistency
anchor tested in tests/test_reasoning.py.

## What the model predicts before any run

Three structural effects, quantified by experiments/run_reasoning.py:

1. Generation grows linearly in g while reading is fixed, so above a
   modest thinking length the query is generation bound and the
   policies' reading differences compress.
2. Speculative waste now costs p + g per wasted branch instead of p,
   so the gate-or-speculate crossover moves sharply toward gating as
   g grows.
3. Thinking shrinks blocks. When d + k (p + g) approaches capacity
   per document times a small count, the decode cohort falls toward
   the weight-read bound and per-token generation cost rises, which
   punishes large k twice.
