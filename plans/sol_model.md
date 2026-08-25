# Speed of light: a floor on the wall time of one query

Status: formalization of the hand derivation for Qwen3-4B-fp8 on one
H100, written so each line can be checked. Sections 1 to 7 are the
derivation made precise and are implemented in `quail/sol.py`,
tested in `tests/test_sol.py`. Section 8 is a proposal for joins and
is not implemented.

Speed of light (SoL) is the smallest wall time the hardware allows
for a query. It counts the arithmetic and the memory traffic the
work cannot be done without, and counts nothing else. A run can
approach SoL and can never beat it, so the gap between a measured
wall and SoL is all of what the engine could still win.

SoL stands on its own. It is derived here from the datasheet, the
model dimensions, and a count of the query's work, and it reuses no
cost model the engine already carries: no roofline out of
`planner/budgets.py`, no calibration constant, no efficiency factor.
That independence is the point. A bound built on a fitted constant
is an estimate wearing a bound's name, and it cannot be used to
judge the thing the constant was fitted to. `quail/sol.py` imports
only `quail.specs`, which is how the rule is kept.

SoL is not a prediction and nothing in the planner reads it either.
`planner/plan.py` says no plan decision consumes an estimated wall,
and that is still true.

## 1. Constants that come from the two specs

| symbol | meaning | source | 4B / H100 |
|---|---|---|---|
| `P` | dense parameters used on every token | `sol.dense_params`, counted from the model dimensions | 3,633,511,936 |
| `L` | layers | `ModelSpec.layers` | 36 |
| `n_q` | query heads | `ModelSpec.n_q` | 32 |
| `n_kv` | KV heads | `ModelSpec.n_kv` | 8 |
| `d_head` | head dimension | `ModelSpec.d_head` | 128 |
| `f_pair` | attention FLOPs per scored pair per layer, `4 * n_q * d_head` | derived | 16,384 |
| `kappa` | KV bytes per token, `2 * L * n_kv * d_head * kv_bytes` | `ModelSpec.kappa` | 147,456 |
| `W` | resident weight bytes | `ModelSpec.W_mem` | 4.5e9 |
| `R_dense` | fp8 dense peak, FLOP/s | `DeviceSpec.peak_flops` | 1.979e15 |
| `R_attn` | bf16 dense peak, FLOP/s | `DeviceSpec.attn_flops` | 0.9895e15 |
| `BW` | HBM bandwidth, bytes/s | `DeviceSpec.hbm_bw` | 3.35e12 |
| `C` | tokens per forward pass (the batch size) | an input, stated at the call site | 110,376 |

Four of these need a sentence.

- `2` FLOPs per parameter per token: every weight does one multiply
  and one add.
- `P` is counted from the model dimensions rather than read off
  `ModelSpec.params`, which is a rounded stand-in: 3.6e9 against a
  real non-embedding count of 3,633,511,936, 0.93% low. That lands
  straight on the largest term of the bound. Embeddings and the
  lm_head are excluded: a token touches one embedding row, not 2
  FLOPs per parameter, and a filter reads logits at one position per
  evaluation.
- `f_pair = 4 * n_q * d_head`: for one (query token, key token) pair
  in one head, the QK dot product runs over `d_head` dimensions, so
  `2 * d_head` FLOPs, and multiplying the attention weight into V
  costs another `2 * d_head`. That is `4 * d_head` per head, times
  `n_q` heads. At 4B: `4 * 32 * 128 = 16,384`.
- The two peaks are the NVIDIA H100 SXM datasheet numbers halved,
  because the datasheet quotes them with 2:1 structured sparsity and
  we run dense: fp8 3,958 TFLOPS becomes 1.979e15 FLOP/s, bf16 1,979
  TFLOPS becomes 0.9895e15 FLOP/s.

The dense projections and attention price against different peaks
because they run in different dtypes. The projections are fp8
(DeepGEMM `fp8_gemm_nt`, `executor/attention.py`). Attention is bf16
(FlashAttention-3 over bf16 KV, `executor/attention.py:_fa`), and KV
is bf16, the only stored dtype. Pricing attention against the fp8
peak would halve the attention term and make the bound wrong in the
unsafe direction: too low, so a run could appear to beat it.

`C` is the one input here that is neither a datasheet figure nor a
model dimension. It is a batch size, it decides how many times the
weights are re-read, and what the engine picks for it is a planner
decision. So it is an input to the bound with no default, stated at
the call site, and the exercises below state 110,376 because that is
what the derivation used.

## 2. What a query contributes

A filter chain over one document set:

- `n` documents, document `i` holding `d_i` tokens.
- `p` preamble tokens: `SHARED_PRE`, the same string on every
  document, counted once per document.
- `S` stages. Stage `s` appends `q_s` question tokens per live
  document, and passes a fraction `sigma_s` of the documents that
  entered it.

Two derived quantities:

- `b_i = p + d_i`, the **retained prefix** of document `i`. Its KV is
  computed once and survives every rewind.
- `B = sum_i b_i`, the corpus prefix mass.

Write `Sigma_s = prod_{t < s} sigma_t` for the fraction of the corpus
still live entering stage `s`, so `Sigma_1 = 1`.

**The random-survivor assumption.** A stage's surviving documents are
treated as a uniform random sample of the ones that entered it, so
the surviving token mass entering stage `s` is `Sigma_s * B`. This is
the one assumption here that the data can violate: a filter that
prefers long documents makes the true mass larger. Nothing in the
engine enforces it.

## 3. What each stage computes

Stage 1 has nothing resident. For document `i` it computes the whole
sequence `[preamble | document | question 1]`, of length
`l_i = b_i + q_1`.

Every later stage rewinds: it drops the previous question's KV, keeps
the document prefix, and computes only its own `q_s` question tokens
against that prefix. Suffix KV is never cached
(`executor/pack.py`), so what survives a rewind is `b_i`, never
`b_i + q_1`.

This is the same shape attention already runs in. The engine's
`attention_merge_quant` issues two FlashAttention calls: one causal
call over the new tokens among themselves, and one non-causal call
of those tokens against the paged prefix KV. The pair count below
splits along the same line.

## 4. Tokens

Tokens pushed through the forward pass:

```
N = (B + n * q_1)                      stage 1: prefix and question
  + sum_{s >= 2} n * Sigma_s * q_s     later stages: question only
```

## 5. Attention pairs

Pairs are counted per layer; the `* L` is applied once in section 6.

```
Pi = sum_i l_i * (l_i + 1) / 2                        stage 1
   + sum_{s >= 2} [ q_s * Sigma_s * B                 stage s, prefix
                  + n * Sigma_s * q_s * (q_s + 1) / 2 ]   stage s, own
```

- Stage 1 is a full causal triangle per document: token `j` of the
  sequence attends to `j` keys including itself, so `l(l+1)/2` pairs.
- A later stage is a rectangle plus a small triangle: each of its
  `q_s` new tokens attends to all `b_i` prefix keys, and to itself and
  the question tokens before it.

The triangle term is quadratic in length, so the corpus cannot be
summarized by its total alone. `sum_i l_i^2` has to come from the
length distribution; the mean cannot produce it. `Corpus` in
`quail/sol.py` therefore carries `sum_i b_i` and `sum_i b_i^2`.

## 6. Seconds

Compute:

```
T_dense     = 2 * P * N / R_dense
T_attention = f_pair * L * Pi / R_attn
T_compute   = T_dense + T_attention
```

Memory. Weights are re-read once per forward pass; at 4.5e9 bytes
against a 50 MB L2 they never stay resident. Each new token writes
its KV once. Each stage after the first reads its live prefixes back
out of the arena once.

```
passes      = ceil(N / C)
bytes       = W * passes + kappa * (N + sum_{s >= 2} Sigma_s * B)
T_memory    = bytes / BW
```

And:

```
SoL = max(T_compute, T_memory)
```

`max`, not a sum, because the arithmetic units and the memory system
run at the same time and a floor is allowed to assume they overlap
perfectly. Inside `T_compute` the two terms are added, because the
dense and attention kernels are separate launches on the same SMs.

Taking `max` once at the top is the loosest honest choice. Taking
it per kernel and summing gives a larger, tighter floor, because it
stops a memory-bound kernel from hiding behind a compute-bound one.
Both are lower bounds, so both are safe; this one is the more
conservative. Tightening it later changes section 6 only and leaves
sections 4 and 5 untouched.

## 7. The two worked exercises, against the measured corpus

Qwen3-4B-fp8, one H100, the 5,000 reviews QUAIL-B builds at sf=0.1.
Nothing here is taken on trust from the derivation: the corpus was
rebuilt from the pinned IMDB revision at seed 20260818 and tokenized
with the Qwen3-4B-FP8 tokenizer, and the moments are committed in
`results/sol_imdb_corpus.json`.

| measured input | value |
|---|---|
| documents | 5,000 |
| `sum d_i` | 1,494,233 tokens, mean 298.8, max 2,135 |
| `p` (`SHARED_PRE`) | 2 tokens |
| `B = sum b_i` | 1,504,233 |
| `sum b_i^2` | 710,456,689 |
| F1 question | 54 tokens |
| F4 question | 48 tokens |

One input is not measured. F1's selectivity 0.8262 comes from a
labelled run and appears in no committed summary, so it is carried
over from the derivation as given.

**IMDB-1**, one filter:

| quantity | value |
|---|---|
| tokens | 1,774,233 |
| pairs | 444,634,043 |
| forward passes | 17 |
| bytes moved | 3.381e11 |
| `T_dense` | 6.5151 s |
| `T_attention` | 0.2650 s |
| `T_memory` | 0.1009 s |
| **SoL** | **6.7801 s**, compute bound |

**IMDB-6**, F1 then F4, keeping the derivation's choice to fold F1's
question into the retained prefix:

| quantity | value |
|---|---|
| tokens | 1,972,521 |
| pairs | 519,853,922 |
| KV read back | 1,465,871 tokens |
| forward passes | 18 |
| bytes moved | 5.880e11 |
| `T_dense` | 7.2432 s |
| `T_attention` | 0.3099 s |
| `T_memory` | 0.1755 s |
| **SoL** | **7.5531 s**, compute bound |

### What the check found

The derivation's stage-1 totals are exactly right. Tokenizing the
real corpus gives 1,774,233 tokens and 444,634,043 causal pairs, to
the digit, which also confirms the 2-token preamble and the 54-token
F1 question. Its final answers, 6.7798 s and 7.5366 s, are within
0.05% and 0.3% of the numbers above.

Three smaller things came out of the check.

1. **`P` is 3,633,511,936, not 3.6e9.** `T_dense` in the derivation
   is 0.93% above what `P = 3.6e9` gives, by the same factor in both
   exercises. That factor is exactly the ratio between Qwen3-4B's
   real non-embedding parameter count and the rounded 3.6e9 that
   `specs/qwen3_4b.py` carries: 3,633,511,936 / 3.6e9 = 1.00931. So
   the derivation used the real count and annotated it with the
   rounded one. Counting from the model dimensions reproduces its
   `T_dense` to five figures (6.5151 s against 6.5148 s, 7.2281 s
   against 7.2277 s). `quail/sol.py` therefore counts `P` with
   `dense_params()` and never reads `ModelSpec.params`, which is
   0.93% low at 4B and 0.02% low at 32B.
2. **F4's question is 48 tokens, not 47.** One token per surviving
   document, which moves IMDB-6 by 4,131 tokens. `bind_prompt`
   counts 48 for `Prompt.tail_tokens`.
3. **Folding F1's question into the prefix over-counts**, because
   the engine rewinds it. Dropping it gives 509,146,370 pairs,
   1,242,797 KV read tokens and SoL 7.5467 s, 0.08% below the folded
   number. Small at two stages and growing with chain depth, so
   `quail/sol.py` defaults to the rewound count and takes
   `carry_question_kv=True` to reproduce the derivation.

A fourth thing is worth writing down because it is easy to get
wrong. `bind_prompt` splits a filter prompt into a preamble, a frame
(the user's text before the placeholder) and a tail. The frame is
**not** paid by a filter: `runtime/session._question_ids` sends
`prompt.tail` only, and `stage_frames` is a `run_join` argument. So
F1 costs 54 tokens per document, not the 73 that tail plus frame
would suggest. Counting the frame would have inflated IMDB-1 by
95,000 tokens.

**A sanity anchor.** SoL 6.7801 s for 1,774,233 tokens is 262,000
tokens/s. The ceiling with attention set to zero is
`R_dense / 2 dense_params`, which is 272,325 tokens/s. SoL is 96% of
that, which is what it should be for a query whose attention term is
4% of compute. Both numbers come from the specs alone, so this
checks the arithmetic and nothing else. Checking the bound against
reality means measuring a wall on an H100 and comparing; that has
not been done yet.

## 8. Joins: proposed, not implemented

Everything above is filter chains, which is what the derivation
covered. A join needs one more shape, and this section is a proposal
to confirm before it is written down in code.

The engine already treats a filter chain as the degenerate join
(`executor/loop.py`): anchor equals the document, suffix equals the
question. A real join keeps the same two pieces and changes what goes
in the suffix.

For a join of an anchor table `A` against partner tables `p in Q`,
with `T` surviving tuples, question length `q_J`, and per-partner
block label length `lab_p`:

- Suffix length per tuple:
  `u = q_J + sum_p (mean_doc_tokens(p) + lab_p)`.
- Tokens: `Sigma_A * B_A` if this stage opens the anchor (the anchor
  prefix is computed here rather than inherited from a filter that
  ran before it), plus `T * u`.
- Pairs: `T * (u * anchor_prefix_len + u * (u + 1) / 2)`, plus the
  anchor's own causal triangle if this stage opens it. One suffix's
  tokens attend to each other but never to another suffix's, and a
  suffix is never split across chunks (`executor/pack.py`), so it is
  one rectangle and one triangle per tuple, as in section 5.
- KV read back: the anchor prefix once per chunk group its suffixes
  span, which is `ceil` of the tuple stream against the chunk budget,
  not once per stage. This is the part I am least sure of and the
  reason this section is a proposal.

The open questions are whether the anchor prefix is really read once
per chunk group rather than once per stage, and what the anchor side
should be when the query does not name one. Both are answerable from
the executor's packing behaviour, which is mechanism and countable;
neither needs a cost model.
