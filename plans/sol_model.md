# Speed of light: a floor on the wall time of one query

Status: formalization of the hand derivation for Qwen3-4B-fp8 on one
H100, written so each line can be checked. Sections 1 to 7 are the
derivation made precise. Section 8 adds joins, which the derivation
did not cover. All of it is implemented in `quail/sol.py` and tested
in `tests/test_sol.py`; section 9 points at the whole QUAIL-B
table.

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

**Survivors: measured, or assumed.** What a stage passes on is two
numbers - how many documents survive, and how much prefix mass they
carry. Given labels, both are counted and nothing is assumed. Given
only a selectivity `sigma_s`, both are scaled by it, which treats
the survivors as a uniform random sample of what entered.

That assumption is wrong on this data, measurably. Section 7.2 has
the numbers: scaling undercuts the true surviving mass by 3.5% after
one filter and 31% after three, because these predicates prefer long
reviews. It moves SoL by less than 0.1% anyway, for a reason worth
knowing - stage 1 computes every prefix, and later stages add about
50 tokens per surviving document, so the mass the assumption gets
wrong is nearly free. Expect it to matter where later stages carry
real work: a join streaming partners against an anchor, or a chain
run memory-bound.

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

## 7. The exercises, on measured inputs

Qwen3-4B-fp8, one H100, the 5,000 reviews QUAIL-B builds at sf=0.1.
Nothing below is taken on trust. The corpus was rebuilt from the
pinned IMDB revision at seed 20260818 and tokenized with the
Qwen3-4B-FP8 tokenizer; all 5,000 documents match the
`left_content_sha256` in the QUAIL-B label sets, so the token
lengths and the labels describe the same documents. Everything is
committed to `results/sol_imdb_corpus.json`; the raw label sets are
`/results/ground_truth/quailb/schema_v1/label_sets/imdb/` on
`quail-results`.

| measured input | value |
|---|---|
| documents | 5,000 |
| `sum d_i` | 1,494,233 tokens, mean 298.8, max 2,135 |
| `p` (`SHARED_PRE`) | 2 tokens |
| `B = sum b_i` | 1,504,233 |
| `sum b_i^2` | 710,456,689 |
| F1, F4, F5 questions | 54, 48, 52 tokens |

Survivors come from the ground truth - the QUAIL-B label sets, one
TRUE/FALSE per document - so each stage carries a counted document
count and a counted prefix mass, and no selectivity is scaled
anywhere:

| after | documents | prefix mass |
|---|---|---|
| F1 | 3,873 | 1,206,884 |
| F1, F4 | 963 | 376,886 |
| F1, F4, F5 | 628 | 272,514 |

**IMDB-1**, one filter. No survivors are involved: one stage
computes every document once.

| quantity | value |
|---|---|
| tokens | 1,774,233 |
| pairs | 444,634,043 |
| forward passes | 17 |
| `T_dense` | 6.5151 s |
| `T_attention` | 0.2650 s |
| `T_memory` | 0.1009 s |
| **SoL** | **6.7801 s**, compute bound |

**IMDB-6**, F1 then F4:

| quantity | value |
|---|---|
| tokens | 1,960,137 |
| pairs | 507,119,123 |
| KV read back | 1,206,884 tokens |
| forward passes | 18 |
| `T_dense` | 7.1978 s |
| `T_attention` | 0.3023 s |
| `T_memory` | 0.1636 s |
| **SoL** | **7.5000 s**, compute bound |

**IMDB-7**, F1 then F4 then F5: 2,010,213 tokens, 528,044,209 pairs,
19 passes, **SoL 7.6964 s**.

### 7.1 Against the derivation

The derivation's stage-1 totals are exactly right: tokenizing the
real corpus gives 1,774,233 tokens and 444,634,043 causal pairs, to
the digit, which also confirms the 2-token preamble and the 54-token
F1 question. Feeding it its own two inputs - selectivity 0.8262 and
F1's question folded into the retained prefix - reproduces 7.5531 s
against its stated 7.5366 s.

Four things came out of the check.

1. **`P` is 3,633,511,936, not 3.6e9.** `T_dense` in the derivation
   is 0.93% above what `P = 3.6e9` gives, by the same factor in both
   exercises, and that factor is exactly Qwen3-4B's real
   non-embedding parameter count over the rounded 3.6e9 that
   `specs/qwen3_4b.py` carries. So the derivation used the real
   count and annotated it with the rounded one; the stated 1.979e15
   divisor was right all along. Counting from the model dimensions
   reproduces its `T_dense` to five figures (6.5151 s against
   6.5148 s, 7.2281 s against 7.2277 s).
2. **0.8262 is the 4B run's own answer rate**, 4,131 of 5,000
   (`/results/sol_check_sf0.1_4b.json` on `quail-results`), not a
   ground-truth selectivity. The ground truth is the 32B model's
   labels, which put F1 at 3,873 of 5,000. The tables above use the
   ground truth.
3. **F4's question is 48 tokens, not 47.** `bind_prompt` counts 48
   for `Prompt.tail_tokens`.
4. **Folding F1's question into the prefix over-counts**, because
   the engine rewinds it. On ground truth it costs 0.08%: 7.5060 s
   folded against 7.5000 s rewound. `carry_question_kv=True`
   reproduces the folded form.

A fifth thing is worth writing down because it is easy to get wrong.
`bind_prompt` splits a filter prompt into a preamble, a frame (the
user's text before the placeholder) and a tail. The frame is **not**
paid by a filter: `runtime/session._question_ids` sends
`prompt.tail` only, and `stage_frames` is a `run_join` argument. So
F1 costs 54 tokens per document, not the 73 that tail plus frame
would suggest. Counting the frame would have inflated IMDB-1 by
95,000 tokens.

### 7.2 What the labels say about the assumption

With per-document labels the surviving mass is counted, so section
2's assumption can be tested rather than trusted. Scaling by
selectivity instead:

| after | counted | scaled by selectivity | error | mean length, survivors vs pool |
|---|---|---|---|---|
| F1 | 1,206,884 | 1,165,179 | -3.5% | 311.6 vs 300.8 |
| F1, F4 | 376,886 | 289,715 | -23.1% | 391.4 vs 311.6 |
| F1, F4, F5 | 272,514 | 188,932 | -30.7% | 433.9 vs 391.4 |

The survivors are consistently longer than the pool they came from,
and F4 is the worst: long reviews discuss endings, short ones do
not. The error compounds down a chain.

It barely moves SoL. IMDB-6 is 7.5000 s counted against 7.4988 s
scaled, IMDB-7 7.6964 s against 7.6925 s - under 0.1% both times,
against a mass 23% and 31% wrong. Stage 1 computes every prefix
once and later stages add about 50 tokens per surviving document,
so the mass the assumption gets wrong is nearly free here. Expect
it to bite where later stages carry real work: a join streaming
partners against an anchor, or a chain run memory-bound. Use the
counted pair wherever labels exist; it costs nothing.

### 7.3 Against a real wall

The 4B run measured IMDB-1 at 15.06 seconds of query time, boot
excluded. SoL says 6.7801 seconds. The engine is at 0.45 of the
floor, so a little over half the wall is loss this bound does not
price. That ratio is the number the whole document exists to
produce.

Two cross-checks on the arithmetic itself. SoL 6.7801 s for
1,774,233 tokens is 262,000 tokens/s, against 272,325 tokens/s for
`R_dense / 2 dense_params` with attention set to zero: SoL is 96% of
it, right for a query whose attention term is 4% of compute. And the
run's own recorded rate, 117,811 tokens/s, is 0.45 of 262,000 - the
same ratio arrived at from the other side.

`sol_check_sf0.1_4b.json` also carries a `sol_s` of 10.195 s for
this query, from a calculation that is not in this repository. It
does not agree with 6.7801 s and I could not find what produced it,
so it is flagged rather than reconciled.

## 8. Joins

A join keeps the same two pieces as a filter chain and changes what
goes in the suffix. The engine already treats a filter chain as the
degenerate join (`executor/loop.py`): anchor equals the document,
suffix equals the question.

One side **anchors**: its KV is held and every tuple attends to it.
The other side **streams**: a copy of each of its documents rides in
every tuple's suffix. Per tuple the suffix is that partner's block
label, its document, and the question tail. One suffix's tokens
attend to each other but never to another suffix's, and a suffix is
never split across chunks (`executor/pack.py`).

Write `c_a` for an anchor's prefix (`SHARED_PRE` plus its document),
`f` for the anchor naming line written into its kept KV once per
stage, and `u_p` for a partner's suffix length. Every live anchor
pairs with every live partner, so the tuple sums separate: a sum
over tuples of `u * c` is `sum_p u` times `sum_a c`. That is why
`JoinStage` takes two sides rather than a tuple list. A gated or
deduped tuple set does not separate and is not modelled.

```
context   = c_a + f                              per anchor

tokens    = sum_a context        (opening)       or   n_A * f
          + n_A * sum_p u_p

pairs     = sum_a context(context+1)/2  (opening)
          or f * sum_a c_a + n_A * f(f+1)/2
          + (sum_p u_p)(sum_a context)
          + n_A * sum_p u_p(u_p+1)/2

kv read   = sum_a context
```

**Opening.** A join opens the anchor when no filter on that side
already computed its prefixes. After a filter they are resident, so
the join adds only the naming line per anchor.

**Which side anchors.** The planner keeps whichever side is cheaper
and streams the other, and the bound has to make the same choice or
it is not a bound on what runs. Anchoring the long side costs one
prefix per document; anchoring the short side costs a full copy of
every long document in every tuple. On FEVER the difference is
5.5x - claims average 11 tokens and evidence 370, so evidence
anchors even though claims are the table the query is written
against. `reports/2026-08-25-sol-quailb.md` prices both ways per
query and keeps the smaller.

**KV read-back** counts each anchor's context once per stage. That
is the minimum for a tuple stream longer than one chunk, which every
join here is. It is an undercount when an anchor's suffixes straddle
a chunk boundary; at these sizes that is a handful of anchors out of
thousands, and every query in the suite is compute bound by more
than an order of magnitude, so it changes no answer.

## 9. Every QUAIL-B query

`reports/2026-08-25-sol-quailb.md` applies all of the above to the
26 queries at sf=0.1 on both models, with measured corpora and
ground-truth survivors. Three of the five queries with a measured
wall have a token count no model's answers can change, and the bound
reproduces all three exactly, including two joins on corpora of
opposite shape. The engine runs at 0.42 to 0.49 of the floor across
every query with a measured wall.
