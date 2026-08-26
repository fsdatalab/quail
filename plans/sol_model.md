# Speed of light: a floor on the wall time of one query

Speed of light (SoL) is the least time one query can take on one
GPU. It counts the arithmetic and the memory traffic the work cannot
be done without, and counts nothing else, so a run can approach it
and can never beat it.

Nothing in it is fitted or measured on a GPU. It uses the datasheet,
the model dimensions, and a count of the query's work, and it
borrows no cost model the engine already carries: no roofline out of
`planner/budgets.py`, no calibration constant, no efficiency factor.
A bound built on a fitted constant is an estimate wearing a bound's
name, and it cannot judge the thing the constant was fitted to.

## 1. Notation

| symbol | meaning | Qwen3-4B-fp8 | Qwen3-32B-fp8 |
|---|---|---|---|
| `p` | shared preamble tokens, on every document | 2 | 2 |
| `d_i` | tokens in document `i` | from the corpus | from the corpus |
| `q_s` | tokens in filter stage `s`'s question | 41 to 62 | 41 to 62 |
| `f_a` | join frame tokens after anchor `a` | 36 to 42 | 36 to 42 |
| `u_b` | one join tuple suffix for partner `b` | from the prompt and corpus | from the prompt and corpus |
| `A_s` | documents alive entering stage `s` | `A_1` is all | `A_1` is all |
| `T(n)` | `n(n+1)/2`, a causal sequence attending to itself | | |
| `P` | dense parameters per token | 3,633,511,936 | 31,206,298,624 |
| `L` | layers | 36 | 64 |
| `n_q`, `d_head` | query heads, head dimension | 32, 128 | 64, 128 |
| `kappa` | KV bytes per token, `2 L n_kv d_head 2` | 147,456 | 262,144 |
| `W` | resident weight bytes | 4.5e9 | 34.37e9 |
| `R_dense` | fp8 dense peak, FLOP/s | 1.979e15 | 1.979e15 |
| `R_attn` | bf16 dense peak, FLOP/s | 0.9895e15 | 0.9895e15 |
| `BW` | HBM bandwidth, bytes/s | 3.35e12 | 3.35e12 |
| `C` | tokens per forward pass | 110,376 | 41,943 |

`p`, `d_i` and `q_s` are the same for both models because every
Qwen3 model shares one tokenizer. The hardware rows are the same
because it is the same H100. Everything else differs.

`C` is a batch size the planner picks, not a property of the
hardware, so it is an input to the bound with no default. Both
values above are the fused kernels' 32-bit offset limit,
`(2^31 - 1)` over the widest projection: 19,456 columns at 4B and
51,200 at 32B. The 32B batch is a quarter of the 4B one, which is
why the same query takes more forward passes and re-reads a
7.6x larger weight set more often.

`P` is counted from the model dimensions rather than quoted, since a
rounded parameter count lands straight on the largest term of the
bound. `R_dense` and `R_attn` are the H100 datasheet
figures halved, because the datasheet quotes them with 2:1 sparsity
and we run dense. The projections are fp8 and attention is bf16
(FlashAttention-3 over bf16 KV), which is why there are two peaks.

## 2. The KV rule

Every document has a **prefix**, `p + d_i`: the shared preamble plus
the document text. Its KV is computed once and stays resident.
Anything attached after it is a **suffix**. A filter suffix is its
question. A join first attaches an anchor frame, which contains the
anchor note and complete question. Each join tuple then attaches the
partner label, partner document, and answer cue. Suffix KV is computed,
used for one evaluation, and dropped. The code in `executor/pack.py`
never caches suffix KV.

So a document is read once however many questions get asked about
it. In the equations below that shows up in one place: whether `d_i`
appears in the token count, or only inside a rectangle.

## 3. Counting the work

Four counts per query: tokens through the forward pass, scored
(query, key) pairs per layer, KV rows written, KV rows read back.

### A filter chain

```
STAGE 1 - every document scanned from nothing

  tokens   = sum over i in A_1 of (p + d_i + q_1)
  pairs    = sum over i in A_1 of T(p + d_i + q_1)
  kv write = tokens
  kv read  = 0


STAGE s > 1 - the document is already in the arena

  tokens   = |A_s| * q_s
  pairs    = sum over i in A_s of [ q_s * (p + d_i) + T(q_s) ]
  kv write = tokens
  kv read  = sum over i in A_s of (p + d_i)
```

**That is the whole of KV reuse.** Compare the two `tokens` lines:
`d_i` is in stage 1 and gone from stage `s > 1`. In the pairs line
it survives only as the width of a rectangle - the `q_s` new tokens
attending over the resident prefix - never as a triangle over
itself. Without reuse, stage `s > 1` would be another stage 1:
`p + d_i + q_s` tokens and `T(p + d_i + q_s)` pairs.

Over 5,000 IMDB reviews, one filter is 1,759,233 tokens. A second
filter adds 180,180 tokens. The added work is one 45 token question
for each of the 4,004 surviving documents. Rescanning would have added
another 1.2 million document and preamble tokens.

The active ground truth labels determine the exact document IDs in
`A_s`. The code does not estimate `A_s` from selectivity, because the
surviving document lengths affect the attention count.

### A join

One side **anchors**. Its prefixes are held and every tuple attends
to them. The other side **streams**. A copy of each partner document
appears in its tuple suffix. Write `f_a` for the anchor frame, which
contains the anchor note and complete question. Write
`u_b = label + d_b + answer cue` for one tuple suffix.

```
JOIN - anchors a, partners b, every a against every b

  per anchor, opening it when no filter ran on that side:
    tokens += p + d_a + f_a
    pairs  += T(p + d_a + f_a)

  per anchor, already resident when a filter ran on that side:
    tokens += f_a
    pairs  += f_a * (p + d_a) + T(f_a)

  per anchor, then per partner:
    tokens += u_b
    pairs  += u_b * (p + d_a + f_a) + T(u_b)

  kv read  = sum over a of (p + d_a + f_a)
```

The tuple lines are the `stage s > 1` shape again: a rectangle over
the resident context plus a triangle over the new tokens. Suffixes
are atomic and never attend to each other (`executor/pack.py`), so
it is one rectangle and one triangle per tuple rather than one big
triangle over the whole stream.

`kv read` counts each anchor's context once per stage. That is the
minimum for a tuple stream longer than one chunk, which every join
here is.

**Which side anchors.** Anchoring the long side costs one prefix per
document. Anchoring the short side puts a full copy of every long
document into every tuple. The planner keeps the cheaper side, so
the calculation prices both choices and keeps the cheaper one. The
join token counts differ by 14.9 times on FEV-2 and 193.7 times on
BIO-2.

## 4. Speed of light

```
T_dense     = 2 * P * tokens / R_dense
T_attention = 4 * n_q * d_head * L * pairs / R_attn
T_compute   = T_dense + T_attention

passes      = ceil(tokens / C)
bytes       = W * passes + kappa * (kv write + kv read)
T_memory    = bytes / BW

SoL         = max(T_compute, T_memory)
```

`2` FLOPs per parameter per token: one multiply and one add.
`4 * n_q * d_head` per pair per layer: the QK dot product runs over
`d_head` dimensions for `2 * d_head` FLOPs, the weight into V costs
another `2 * d_head`, times `n_q` heads. At 4B that is 16,384.

Weights are re-read once per forward pass: 4.5e9 bytes against a
50 MB L2 never stay resident.

`max`, not a sum, because the arithmetic units and the memory system
run at once and a floor may assume they overlap perfectly. Inside
`T_compute` the two terms add, because the dense and attention
kernels are separate launches on the same SMs. Taking `max` once at
the top is the loosest honest choice - per kernel it would be larger
and tighter, and both are lower bounds.

## 5. The code

`reports/make_sol_quailb.py` is these equations, plus the
measurement of the three inputs they need. Three functions carry
section 3:

| function | equation |
|---|---|
| `scan(prefix, suffix)` | stage 1: `prefix + suffix` tokens, `T(prefix + suffix)` pairs |
| `ask(prefix, suffix)` | stage `s > 1`: `suffix` tokens, `suffix * prefix + T(suffix)` pairs |
| `stream(prefix, suffixes)` | a join's tuples: `ask` per suffix, one prefix read |

`filter_chain`, `join` and `cheaper_anchor` compose them, and
`speed_of_light()` is section 4. Nothing in the engine imports any of it:
this is analysis, not a plan input.

## 6. What the bound assumes

- **KV read is a minimum**, one read per anchor per stage. An anchor
  whose tuples straddle a chunk boundary is read twice.
- **Compute and memory overlap perfectly**, per the `max` above.
- **Nothing is charged** for the tokenizer, the host, scheduling,
  kernel efficiency, or launch gaps. That is what makes it a floor.

## 7. Every QUAIL-B query

`reports/2026-08-26-sol-quailb.md` applies all of this to the 26
queries at sf=0.1 on Qwen3-4B-fp8 and Qwen3-32B-fp8, from measured
document lengths, measured prompt lengths, and exact active ground
truth labels. `reports/make_sol_quailb.py` produces the report data.
The mounted volume path is `/results/sol/sol_quailb_sf0.1.json`.
