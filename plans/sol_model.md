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
| `q_s` | tokens in stage `s`'s question | 40 to 62 | 40 to 62 |
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
Anything attached after it is a **suffix** - a filter's question, or
a join's partner block plus question - and suffix KV is computed,
used for that one evaluation, and dropped (`executor/pack.py`:
suffix KV is never cached).

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

Over 5,000 IMDB reviews, one filter is 1,774,233 tokens. A second
filter adds 185,904 - one 48-token question per surviving document.
Rescanning would have added another 1.5 million.

`|A_s|` comes from the selectivities: `A_1` is the whole corpus and
`|A_{s+1}| = sigma_s * |A_s|`. Which documents survive matters as
well, because `d_i` is in the pairs line; see section 6.

### A join

One side **anchors**: its prefixes are held and every tuple attends
to them. The other **streams**: a copy of each of its documents
rides in every tuple's suffix. Write `note` for the anchor naming
line, written into each anchor's kept KV once for the stage, and
`u_b = label + d_b + question` for one tuple's suffix.

```
JOIN - anchors a, partners b, every a against every b

  per anchor, opening it (no filter ran on that side):
    tokens += p + d_a + note
    pairs  += T(p + d_a + note)

  per anchor, already resident (a filter ran on that side):
    tokens += note
    pairs  += note * (p + d_a) + T(note)

  per anchor, then per partner:
    tokens += u_b
    pairs  += u_b * (p + d_a + note) + T(u_b)

  kv read  = sum over a of (p + d_a + note)
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
document; anchoring the short side puts a full copy of every long
document into every tuple. The planner keeps the cheaper side, so
the bound prices both orientations and keeps the cheaper one. The
choice is not a detail: the two orientations differ by 5.2x on
FEVER and 46x on BioDEX.

## 4. Seconds

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
`seconds()` is section 4. Nothing in the engine imports any of it:
this is analysis, not a plan input.

## 6. What the bound assumes

- **Selectivity fixes how many documents survive, not which.**
  `survivors()` keeps an evenly spaced slice of the length-sorted
  pool, so the survivors carry the pool's length distribution. A
  predicate correlated with document length breaks that.
- **KV read is a minimum**, one read per anchor per stage. An anchor
  whose tuples straddle a chunk boundary is read twice.
- **Compute and memory overlap perfectly**, per the `max` above.
- **Nothing is charged** for the tokenizer, the host, scheduling,
  kernel efficiency, or launch gaps. That is what makes it a floor.

## 7. Every QUAIL-B query

`reports/2026-08-25-sol-quailb.md` applies all of this to the 26
queries at sf=0.1 on Qwen3-4B-fp8 and Qwen3-32B-fp8, from measured
document lengths, measured prompt lengths, and ground-truth
selectivities. `reports/make_sol_quailb.py` is the one script that
produces it, and it writes both those inputs and the answers to
`/sol/sol_quailb_sf0.1.json` on the `quail-results` volume.
