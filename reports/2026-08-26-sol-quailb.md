# Speed of light for all 26 QUAIL-B queries, 4B and 32B

## What this is

The least time each QUAIL-B query at sf=0.1 can take on one H100,
for Qwen3-4B-fp8 and Qwen3-32B-fp8. Only three things are counted:
the dense projection FLOPs, the attention pair FLOPs, and the bytes
moved. Everything a real run also pays - kernel efficiency, launch
gaps, scheduling, the host - is left out, so a run can approach
these numbers and can never beat them.

Nothing here is fitted or measured on a GPU. The equations are
`plans/sol_model.md`, and `reports/make_sol_quailb.py` is those
equations plus the measurement of their inputs - one file, small
enough to check by hand.

## The three inputs

Every number below comes from exactly three measured things.
`reports/make_sol_quailb.py` measures them and computes the table in
one pass, and writes both to `/sol/sol_quailb_sf0.1.json` on the
`quail-results` volume. The scale factor is the script's second
argument and defaults to 0.1; everything below is sf=0.1. Each scale
factor has its own corpus and its own label collection, and the script
stops if the two it is given disagree.

1. **Document lengths.** Each corpus tokenized with the Qwen3
   tokenizer, kept as a length histogram so nothing is lost and no
   per-document record is committed.

   | set | side | documents | tokens | mean |
   |---|---|---|---|---|
   | IMDB | documents | 5,000 | 1,494,233 | 298.8 |
   | IMDB | partner | 12 | 27 | 2.25 |
   | BioDEX | documents | 200 | 829,199 | 4,146.0 |
   | BioDEX | partner | 614 | 2,848 | 4.6 |
   | FEVER | documents | 100 | 1,142 | 11.4 |
   | FEVER | partner | 57 | 21,099 | 370.2 |
   | LePaRD | documents | 200 | 46,619 | 233.1 |
   | LePaRD | partner | 200 | 16,589 | 82.9 |

   "Documents" is the set a query is written against; "partner" is
   the set its join runs against.

2. **Prompt lengths.** The shared preamble is 2 tokens. Each
   filter's question is 41 to 62 tokens. Each join carries a partner
   block label, an anchor naming line, and a question.

3. **Selectivities.** From the QUAIL-B ground truth labels - the 32B
   model's TRUE/FALSE per document - taken per stage and conditional
   on the stages before it. F1 passes 0.7746 of IMDB's documents; F4 then
   passes 0.2486 of what F1 left.

   These come from collection `gt_76bc95a8a8dfee478bc8546e5b6154da`,
   the labels active for corpus `c_df45ef585738f42e4a7a731306f1b9fc`.
   **These labels are known to be wrong for 19 of their 23
   predicates** - see "The labels this was computed against" below.
   A predicate keeps one label set per template it has been judged
   under, and superseded ones stay on the volume, so the script reads
   the corpus's `active_collection.json` and takes only the label
   sets that collection names.

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
IMDB documents at 1,759,233 tokens. IMDB-6 adds a second filter and
comes to 1,933,518. The difference is 174,285, which is exactly the
3,873 surviving documents times F4's 45-token question. Nothing is
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
| IMDB-1 | 1F | 1,759,233 | 4.39e8 | - | - | 6.72 s | 56.41 s | 8.39 |
| IMDB-2 | 1J | 3,669,233 | 1.07e9 | 60,000 | documents | 14.11 s | 117.98 s | 8.36 |
| IMDB-3 | 1F+1J | 3,436,242 | 9.90e8 | 46,476 | documents | 13.21 s | 110.47 s | 8.36 |
| IMDB-4 | 2F+1J | 2,350,497 | 6.32e8 | 11,556 | documents | 9.01 s | 75.47 s | 8.38 |
| IMDB-5 | 3F+1J | 2,252,629 | 6.00e8 | 7,536 | documents | 8.63 s | 72.31 s | 8.38 |
| IMDB-6 | 2F | 1,933,518 | 4.96e8 | - | - | 7.40 s | 62.03 s | 8.39 |
| IMDB-7 | 3F | 1,980,705 | 5.11e8 | - | - | 7.58 s | 63.55 s | 8.39 |
| BIO-1 | 1F | 838,999 | 2.38e9 | - | - | 4.50 s | 31.50 s | 7.00 |
| BIO-2 | 1J | 6,190,399 | 2.47e10 | 122,800 | documents | 37.48 s | 247.68 s | 6.61 |
| BIO-3 | 1F+1J | 4,028,675 | 1.55e10 | 73,066 | documents | 24.05 s | 159.95 s | 6.65 |
| BIO-4 | 2F+1J | 3,041,806 | 1.13e10 | 50,348 | documents | 17.89 s | 119.83 s | 6.70 |
| BIO-5 | 3F+1J | 2,348,920 | 8.33e9 | 34,384 | documents | 13.59 s | 91.73 s | 6.75 |
| FEV-1 | 1F | 6,642 | 2.25e5 | - | - | 0.025 s | 0.21 s | 8.56 |
| FEV-2 | 1J | 274,977 | 1.09e8 | 5,700 | partner | 1.07 s | 8.90 s | 8.29 |
| FEV-3 | 1F+1J | 177,252 | 6.65e7 | 3,363 | partner | 0.69 s | 5.73 s | 8.30 |
| FEV-4 | 2F+1J | 55,989 | 1.64e7 | 570 | partner | 0.22 s | 1.80 s | 8.36 |
| FEV-5 | 2F+1J two-sided | 128,366 | 4.50e7 | 2,183 | partner | 0.50 s | 4.14 s | 8.32 |
| FEV-6 | 3F+1J two-sided | 50,583 | 1.39e7 | 370 | partner | 0.19 s | 1.62 s | 8.37 |
| LEP-1 | 1F | 57,819 | 1.14e7 | - | - | 0.22 s | 1.85 s | 8.43 |
| LEP-2 | 1J | 4,726,819 | 1.46e9 | 40,000 | documents | 18.23 s | 152.17 s | 8.35 |
| LEP-3 | 1F+1J | 151,415 | 3.43e7 | 800 | documents | 0.58 s | 4.85 s | 8.41 |
| LEP-4 | 2F+1J | 128,204 | 2.35e7 | 600 | documents | 0.48 s | 4.09 s | 8.44 |
| LEP-5 | 3F+1J | 58,148 | 1.15e7 | 0 | documents | 0.22 s | 1.86 s | 8.43 |
| LEP-6 | 5F+1J | 58,148 | 1.15e7 | 0 | documents | 0.22 s | 1.86 s | 8.43 |
| LEP-7 | 3F+1J two-sided | 153,793 | 2.53e7 | 600 | documents | 0.58 s | 4.90 s | 8.46 |
| LEP-8 | 5F | 58,148 | 1.15e7 | - | - | 0.22 s | 1.86 s | 8.43 |

Every query on both models is compute bound. The largest
`T_memory` in the suite is FEV-1's at 6.7% of its `T_compute`, and
FEV-1 is the smallest query here - 100 documents of 11 tokens.
Everything else pushes enough tokens per pass that the weight reads
amortize away.

## Which side of a join is held

A join has to pick a side to anchor. Anchoring the long side costs
one prefix per document; anchoring the short side copies every long
document into every tuple. The bound prices both and keeps the
cheaper, because that is what the engine does.

| join | side held | side streamed | tokens as run | tokens the other way | cost of choosing wrong |
|---|---|---|---|---|---|
| IMDB-2 | documents, 299 tokens | partner, 2 | 3,669,233 | 19,910,967 | 5.4x |
| BIO-2 | documents, 4,146 | partner, 5 | 6,190,399 | 513,927,602 | 83.0x |
| FEV-2 | partner, 370 | documents, 11 | 274,977 | 2,300,342 | 8.4x |
| FEV-5 | partner, 370 | documents, 11 | 96,977 | 842,166 | 8.7x |
| LEP-2 | documents, 233 | partner, 83 | 4,726,819 | 10,702,789 | 2.3x |

FEVER is the one that inverts. Its documents average 11 tokens and
its partner set 370, so the join holds the partner and streams the
documents past it - the opposite of how the query
reads. BioDEX is the extreme, because a 4,146-token report copied
into each of 122,800 tuples is 83 times the cost of holding 200 of
them.

## Document length decides the mix, and the mix decides the model gap

Figure: plots/sol_quailb_attention_share.png

The two compute terms grow differently with **context length** -
how many earlier tokens each new token attends over. `T_dense` does
not care: it is 2 P FLOPs per token whatever the context.
`T_attention` is linear in it, since a token attending over `m`
earlier tokens scores `m` pairs, which makes it quadratic in
document length once a whole document is scanned.

So a query's mix is set by what it holds in context. A token
attending over a 4,146-token BioDEX document does 377 times the
attention work of one attending over an 11-token FEVER document,
while both do identical dense work. FEVER sits at both ends of the
figure for exactly this reason: FEV-1 is a filter, so its context is
an 11-token document, while the FEVER joins hold the 370-token
partner in context and stream the documents past it.

The two terms also scale differently with model size. `T_dense`
scales with the parameter count, 31,206,298,624 over 3,633,511,936
or 8.59x. `T_attention` scales with `4 n_q d_head L`, 2,097,152
against 589,824 or 3.56x, and both models price it against the same
bf16 peak.

Put together: a query's cost multiplier from 4B to 32B lies between
3.56 and 8.59, at whatever its mix is. The five BioDEX queries hold
4,146-token documents and spend 31 to 39% of their compute on
attention, so they multiply by 6.6 to 7.0. Every other query holds
documents of 11 to 370 tokens, spends under 7% on attention, and
multiplies by 8.3 to 8.6.

Long documents therefore make the big model relatively cheaper, not
dearer.

## The labels this was computed against

The selectivities above come from collection
`gt_76bc95a8a8dfee478bc8546e5b6154da`, which is what
`active_collection.json` points at for this corpus. For 19 of its 23
predicates it names the same label files as the run that predates
PR #57 - identical label set ids, not re-judged. Those labels were
produced with templates that carried a duplicated `ANSWER=` cue, which
PR #57 measured as pushing the model toward `FALSE` on 14 of 19
predicates. BioDEX's reaction join reads 81 true pairs out of 122,800
here; without the cue it is 19,777.

So the filter and join arithmetic below is current, and the
selectivities feeding it are not. Concretely:

- **Token and pair counts for a single-stage query are unaffected.**
  IMDB-1, BIO-1, FEV-1 and LEP-1 scan every document whatever the
  labels say.
- **Every multi-stage query is affected**, because how many documents
  reach stage two is exactly what the labels decide.
- **The join costs are affected twice over**: once through how many
  documents survive to be joined, and once through the prompt change
  in PR #52, which is real and is the larger of the two.

What would settle it is one judge-pass run against current `main`, so
all 23 predicates get labels from the cue-free templates. That is about
34 minutes on four H100s. Until then, read the join arithmetic and the
anchor choice as current, and every absolute second as provisional.

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
  documents of 200, LEP2 leaves 3 and LEP3 leaves none, so their
  later stages and their joins cost nothing. Those three carry no
  signal at sf=0.1.
- **Nothing here is compared with a measured wall.** These are
  floors. What fraction of them the engine reaches is a separate
  question and a separate run.
