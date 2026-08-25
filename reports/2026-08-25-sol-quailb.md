# Speed of light for all 26 QUAIL-B queries, 4B and 32B

## What this is

The least time each QUAIL-B query at sf=0.1 can take on one H100,
for Qwen3-4B-fp8 and Qwen3-32B-fp8. Only three things are counted:
the dense projection FLOPs, the attention pair FLOPs, and the bytes
moved. Everything a real run also pays - kernel efficiency, launch
gaps, scheduling, the host - is left out, so a run can approach
these numbers and can never beat them.

Nothing here is fitted or measured on a GPU. `quail/sol.py` imports
only `quail.specs`, and the whole file is arithmetic small enough to
check by hand. The equations are `plans/sol_model.md`.

## The three inputs

Every number below comes from exactly three measured things.
`reports/make_sol_quailb.py` measures them and computes the table in
one pass; `results/sol_quailb_sf0.1.json` holds both.

1. **Document lengths.** Each corpus tokenized with the Qwen3
   tokenizer, kept as a length histogram so nothing is lost and no
   per-document record is committed.

   | column | documents | tokens | mean |
   |---|---|---|---|
   | reviews.body | 5,000 | 1,494,233 | 298.8 |
   | reports.report | 200 | 829,199 | 4,146.0 |
   | claims.claim | 100 | 1,142 | 11.4 |
   | evidence.text | 57 | 21,099 | 370.2 |
   | citations.destination_context | 200 | 46,619 | 233.1 |
   | citations.passage_text | 200 | 16,589 | 82.9 |
   | terms.term | 614 | 2,848 | 4.6 |
   | aspects.aspect | 12 | 27 | 2.25 |

2. **Prompt lengths.** The shared preamble is 2 tokens. Each
   filter's question is 40 to 62 tokens. Each join carries a partner
   block label, an anchor naming line, and a question.

3. **Selectivities.** From the QUAIL-B ground truth labels - the 32B
   model's TRUE/FALSE per document - taken per stage and conditional
   on the stages before it. F1 passes 0.7746 of the reviews; F4 then
   passes 0.2486 of what F1 left.

One thing is set rather than measured: the batch size the forward
pass runs at, which decides how many times the weights are re-read.
It is the fused kernels' 32-bit offset limit, `(2^31 - 1)` over the
widest projection, so **110,376 tokens at 4B and 41,943 at 32B** -
the 32B batch is a quarter of the 4B one, over a weight set 7.6x
larger.

## How KV reuse enters the equations

A document has a **prefix**, `p + d_i`: the shared preamble plus the
document text. Its KV is computed once and stays resident. A
**suffix** is whatever gets attached after it - a filter's question,
or a join's partner block plus question - and suffix KV is computed,
used once, and dropped (`executor/pack.py`).

The whole mechanism is the difference between these two blocks. Let
`T(n) = n(n+1)/2`, a causal sequence attending to itself.

```
STAGE 1 - every document scanned from nothing

  tokens = sum over i of (p + d_i + q_1)
  pairs  = sum over i of T(p + d_i + q_1)


STAGE s > 1 - the document is already in the arena

  tokens  = |A_s| * q_s                                <- no d_i
  pairs   = sum over i in A_s of [ q_s * (p + d_i)     <- d_i only
                                   + T(q_s) ]             as context
  kv read = sum over i in A_s of (p + d_i)
```

`d_i` is in stage 1's token count and gone from stage `s > 1`'s. In
the pairs it survives only as the width of a rectangle - the `q_s`
new tokens attending over the resident prefix - never as a triangle
over itself. Without reuse, stage `s > 1` would be another stage 1.

Read straight off the table below: IMDB-1 is one filter over 5,000
reviews at 1,774,233 tokens. IMDB-6 adds a second filter and comes
to 1,960,137. The difference is 185,904, which is exactly 3,873
surviving reviews times F4's 48-token question. The reviews are not
scanned again.

A join is the same shape with a longer suffix. One side anchors -
its prefixes held, every tuple attending to them - and the other
streams, a copy of each of its documents riding in every tuple's
suffix. Per tuple that is one rectangle over the anchor's context
plus one triangle over the tuple's own tokens, because suffixes are
atomic and never attend to each other.

The equations in full, joins included, are `plans/sol_model.md`
section 3.

## Result

Figure: plots/sol_quailb_per_query.png

| query | shape | tokens | pairs | tuples | anchor | 4B SoL | 32B SoL | 32B/4B |
|---|---|---|---|---|---|---|---|---|
| IMDB-1 | 1F | 1,774,233 | 4.45e8 | - | - | 6.78 s | 56.90 s | 8.39 |
| IMDB-2 | 1J | 6,124,233 | 1.96e9 | 60,000 | reviews | 23.66 s | 197.31 s | 8.34 |
| IMDB-3 | 1F+1J | 5,352,885 | 1.69e9 | 46,476 | reviews | 20.66 s | 172.40 s | 8.34 |
| IMDB-4 | 2F+1J | 2,849,949 | 8.14e8 | 11,556 | reviews | 10.95 s | 91.61 s | 8.37 |
| IMDB-5 | 3F+1J | 2,590,485 | 7.22e8 | 7,536 | reviews | 9.94 s | 83.23 s | 8.37 |
| IMDB-6 | 2F | 1,960,137 | 5.05e8 | - | - | 7.50 s | 62.89 s | 8.39 |
| IMDB-7 | 3F | 2,010,213 | 5.21e8 | - | - | 7.69 s | 64.50 s | 8.39 |
| BIO-1 | 1F | 839,599 | 2.38e9 | - | - | 4.50 s | 31.53 s | 7.00 |
| BIO-2 | 1J | 11,347,799 | 4.65e10 | 122,800 | reports | 69.40 s | 456.48 s | 6.58 |
| BIO-3 | 1F+1J | 7,097,928 | 2.83e10 | 73,066 | reports | 42.93 s | 283.82 s | 6.61 |
| BIO-4 | 2F+1J | 5,157,297 | 1.99e10 | 50,348 | reports | 30.81 s | 204.84 s | 6.65 |
| BIO-5 | 3F+1J | 3,794,195 | 1.41e10 | 34,384 | reports | 22.33 s | 149.50 s | 6.70 |
| FEV-1 | 1F | 6,942 | 2.45e5 | - | - | 0.026 s | 0.22 s | 8.56 |
| FEV-2 | 1J | 485,820 | 2.02e8 | 5,700 | evidence | 1.90 s | 15.75 s | 8.27 |
| FEV-3 | 1F+1J | 301,926 | 1.22e8 | 3,363 | evidence | 1.18 s | 9.78 s | 8.28 |
| FEV-4 | 2F+1J | 77,499 | 2.57e7 | 570 | evidence | 0.30 s | 2.50 s | 8.33 |
| FEV-5 | 2F+1J two-sided | 209,571 | 7.94e7 | 2,183 | evidence | 0.82 s | 6.78 s | 8.30 |
| FEV-6 | 3F+1J two-sided | 64,884 | 1.98e7 | 370 | evidence | 0.25 s | 2.09 s | 8.35 |
| LEP-1 | 1F | 58,419 | 1.16e7 | - | - | 0.22 s | 1.87 s | 8.43 |
| LEP-2 | 1J | 6,086,619 | 1.97e9 | 40,000 | excerpts | 23.53 s | 196.13 s | 8.34 |
| LEP-3 | 1F+1J | 179,211 | 4.29e7 | 800 | excerpts | 0.68 s | 5.74 s | 8.40 |
| LEP-4 | 2F+1J | 149,213 | 2.85e7 | 600 | excerpts | 0.57 s | 4.77 s | 8.44 |
| LEP-5 | 3F+1J | 58,769 | 1.17e7 | 0 | excerpts | 0.22 s | 1.88 s | 8.43 |
| LEP-6 | 5F+1J | 58,769 | 1.17e7 | 0 | excerpts | 0.22 s | 1.88 s | 8.43 |
| LEP-7 | 2F+1J two-sided | 175,402 | 3.04e7 | 600 | excerpts | 0.66 s | 5.60 s | 8.45 |
| LEP-8 | 5F | 58,769 | 1.17e7 | - | - | 0.22 s | 1.88 s | 8.43 |

Every query on both models is compute bound. The largest
`T_memory` in the suite is FEV-1's at 6.4% of its `T_compute`, and
FEV-1 is the smallest query here - 100 claims of 11 tokens.
Everything else pushes enough tokens per pass that the weight reads
amortize away.

## Which side of a join is held

A join has to pick a side to anchor. Anchoring the long side costs
one prefix per document; anchoring the short side copies every long
document into every tuple. The bound prices both and keeps the
cheaper, because that is what the engine does.

| join | side held | side streamed | tokens as run | tokens the other way | cost of choosing wrong |
|---|---|---|---|---|---|
| IMDB-2 | reviews, 299 tokens | aspects, 2 | 6,124,233 | 22,370,955 | 3.7x |
| BIO-2 | reports, 4,146 | terms, 5 | 11,347,799 | 519,084,588 | 45.7x |
| FEV-2 | evidence, 370 | claims, 11 | 485,820 | 2,511,142 | 5.2x |
| FEV-5 | evidence, 370 | claims, 11 | 177,711 | 922,878 | 5.2x |
| LEP-2 | excerpts, 233 | passages, 83 | 6,086,619 | 12,062,589 | 2.0x |

FEVER is the one that inverts. The query is written against claims,
but claims average 11 tokens and evidence 370, so the join holds
evidence and streams the claims past it - the opposite of how it
reads. BioDEX is the extreme, because a 4,146-token report copied
into each of 122,800 tuples is 46 times the cost of holding 200 of
them.

## Why 32B is not a flat 8.6x

Figure: plots/sol_quailb_attention_share.png

The dense term scales with parameters - 31,206,298,624 over
3,633,511,936, or 8.59x. The attention term scales with
`4 n_q d_head L` - 2,097,152 at 32B against 589,824 at 4B, or 3.56x.
Both attention terms price against the same bf16 peak.

So each query's 32B multiplier sits between 3.56 and 8.59 according
to how much of its compute is attention. IMDB, FEVER and LePaRD are
0.6-6.3% attention and land at 8.3-8.6. BioDEX is 32-40% attention
and lands at 6.6-7.0. The cause is document length: BioDEX reports
average 4,146 tokens against IMDB's 299, and pairs are quadratic in
length.

Long documents therefore make the big model relatively cheaper, not
dearer.

## What this does not settle

- **The floor is loose.** `max(T_compute, T_memory)` taken once at
  the top is weaker than taking it per kernel and summing. Both are
  lower bounds; the per-kernel one would be larger and tighter.
- **Selectivity fixes how many documents survive, not which.**
  `survivors()` keeps an evenly spaced slice of the length-sorted
  pool, so the survivors carry the pool's length distribution. A
  predicate correlated with document length breaks that.
- **The KV read count is a minimum**, one read per anchor per stage.
  An anchor whose tuples straddle a chunk boundary is read twice.
  Nothing here is memory bound, so it changes no answer.
- **LEP-5, LEP-6 and LEP-8 are the same number.** LEP1 leaves 4
  documents of 200 and LEP3 leaves none, so their later stages and
  their joins cost nothing. Those three queries carry no signal at
  sf=0.1.
- **Nothing here is compared with a measured wall.** These are
  floors. What fraction of them the engine reaches is a separate
  question and a separate run.
