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

## Prediction

The corrected join accounting was expected to match Quail's measured
fresh token counts exactly: 2,635,599 for BIO-2, 145,359 for FEV-2,
and 2,419,233 for IMDB-2. It did. The earlier SoL code counted the
complete question once per pair, so it was expected to overstate all
three join floors.

The ground truth pass predicted 284,138 Qwen3 32B labels, 40,063 source
labels, and 0 answer differences in a 352 judgment deterministic check.
All three predictions matched. The 30 to 45 minute and $4 to $6 run
prediction cannot be compared directly because the first command stopped
on duplicate part validation and the corrected command resumed its saved
parts.

## Ground truth setup and result

The labels were generated on Modal with four `H100!` requests and one
Qwen3 32B fp8 model copy per GPU. The four workloads were IMDB, BioDEX,
FEVER, and LePaRD. LePaRD's citation join uses source passage IDs. FEVER
uses its source annotation for 63 support pairs. Qwen judged the remaining
rows.

The completed collection has 324,201 labels across 23 predicates. Its
summary is at
`/results/ground_truth/quailb/schema_v1/collections/gt_306dac4fc83883c7a5bcc86f4d103f32/summary.json`
on the `quail-results` volume. The finalizer function call was
`fc-01M0YKBGBF09HSDAF5FSA533AX`.

## The three inputs

Every number below comes from exactly three measured things.
`reports/make_sol_quailb.py` measures them and computes the table in
one pass, and writes both to `/results/sol/sol_quailb_sf0.1.json` on the
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
   filter's question is 41 to 62 tokens. Each join writes the anchor
   note and complete question once per anchor. Each tuple carries the
   partner label, partner document, and answer cue.

3. **Ground truth labels.** The calculation uses each saved TRUE or FALSE
   answer to determine the exact document IDs that reach the next stage.
   The selectivity is only a summary of those answers. F1 passes 0.8008 of
   IMDB's documents. F4 then passes 0.2567 of what F1 left.

   These come from collection `gt_306dac4fc83883c7a5bcc86f4d103f32`,
   the labels active for corpus `c_df45ef585738f42e4a7a731306f1b9fc`.
   The collection has 324,201 labels. Its deterministic check repeated
   352 Qwen judgments with 0 answer differences.
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
**suffix** is whatever gets attached after it. A filter suffix is its
question. A join tuple suffix is the partner label, partner document,
and answer cue. Suffix KV is computed, used once, and dropped
(`executor/pack.py`).

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
comes to 1,939,413. The difference is 180,180, which is exactly the
4,004 surviving documents times F4's 45-token question. Nothing is
scanned again.

A join writes the complete anchor frame once after each anchor
document. The other side streams. Each tuple carries only the partner
label, partner document, and answer cue. Per tuple that is one
rectangle over the anchor document and frame, plus one triangle over
the tuple's own tokens. Suffixes are atomic and never attend to each
other.

The equations in full, joins included, are `plans/sol_model.md`
section 3.

## Result

Figure: plots/sol_quailb_per_query.png

| query | shape | tokens | pairs | tuples | anchor | 4B SoL | 32B SoL | 32B/4B |
|---|---|---|---|---|---|---|---|---|
| IMDB-1 | 1F | 1,759,233 | 4.39e8 | - | - | 6.72 s | 56.41 s | 8.39 |
| IMDB-2 | 1J | 2,419,233 | 6.66e8 | 60,000 | documents | 9.28 s | 77.71 s | 8.37 |
| IMDB-3 | 1F+1J | 2,491,965 | 6.95e8 | 48,048 | documents | 9.56 s | 80.06 s | 8.37 |
| IMDB-4 | 2F+1J | 2,127,537 | 5.80e8 | 12,336 | documents | 8.16 s | 68.33 s | 8.38 |
| IMDB-5 | 3F+1J | 2,112,212 | 5.78e8 | 8,028 | documents | 8.10 s | 67.84 s | 8.37 |
| IMDB-6 | 2F | 1,939,413 | 4.99e8 | - | - | 7.42 s | 62.22 s | 8.39 |
| IMDB-7 | 3F | 1,989,785 | 5.20e8 | - | - | 7.62 s | 63.86 s | 8.38 |
| BIO-1 | 1F | 838,999 | 2.38e9 | - | - | 4.50 s | 31.50 s | 7.00 |
| BIO-2 | 1J | 2,635,599 | 9.92e9 | 122,800 | documents | 15.59 s | 104.15 s | 6.68 |
| BIO-3 | 1F+1J | 1,949,689 | 7.25e9 | 75,522 | documents | 11.48 s | 76.86 s | 6.69 |
| BIO-4 | 2F+1J | 1,647,712 | 5.99e9 | 54,646 | documents | 9.62 s | 64.66 s | 6.72 |
| BIO-5 | 3F+1J | 1,390,203 | 4.65e9 | 36,840 | documents | 7.88 s | 53.71 s | 6.82 |
| FEV-1 | 1F | 6,642 | 2.25e5 | - | - | 0.025 s | 0.21 s | 8.56 |
| FEV-2 | 1J | 145,359 | 5.78e7 | 5,700 | partner | 0.57 s | 4.71 s | 8.28 |
| FEV-3 | 1F+1J | 107,313 | 3.93e7 | 3,648 | partner | 0.42 s | 3.47 s | 8.31 |
| FEV-4 | 2F+1J | 47,949 | 1.33e7 | 570 | partner | 0.18 s | 1.54 s | 8.37 |
| FEV-5 | 2F+1J two-sided | 82,967 | 2.93e7 | 2,368 | partner | 0.32 s | 2.68 s | 8.32 |
| FEV-6 | 3F+1J two-sided | 45,443 | 1.22e7 | 370 | partner | 0.17 s | 1.46 s | 8.38 |
| LEP-1 | 1F | 57,819 | 1.14e7 | - | - | 0.22 s | 1.85 s | 8.43 |
| LEP-2 | 1J | 3,772,219 | 1.23e9 | 40,000 | documents | 14.58 s | 121.56 s | 8.34 |
| LEP-3 | 1F+1J | 132,323 | 4.90e7 | 800 | documents | 0.52 s | 4.28 s | 8.30 |
| LEP-4 | 2F+1J | 113,885 | 3.93e7 | 600 | documents | 0.44 s | 3.67 s | 8.32 |
| LEP-5 | 3F+1J | 76,774 | 2.26e7 | 200 | documents | 0.30 s | 2.47 s | 8.36 |
| LEP-6 | 5F+1J | 58,209 | 1.16e7 | - | documents | 0.22 s | 1.86 s | 8.43 |
| LEP-7 | 3F+1J two-sided | 139,474 | 4.11e7 | 600 | documents | 0.54 s | 4.49 s | 8.36 |
| LEP-8 | 5F | 58,209 | 1.16e7 | - | - | 0.22 s | 1.86 s | 8.43 |

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

| join | side held | side streamed | join tokens as run | join tokens the other way | cost of choosing wrong |
|---|---|---|---|---|---|
| IMDB-2 | documents, 299 tokens | partner, 2 | 2,419,233 | 18,531,279 | 7.7x |
| BIO-2 | documents, 4,146 | partner, 5 | 2,635,599 | 510,386,050 | 193.7x |
| FEV-2 | partner, 370 | documents, 11 | 145,359 | 2,171,842 | 14.9x |
| FEV-5 | partner, 370 | documents, 11 | 51,578 | 914,560 | 17.7x |
| LEP-2 | documents, 233 | partner, 83 | 3,772,219 | 9,748,189 | 2.6x |

FEVER is the one that inverts. Its documents average 11 tokens and
its partner set 370, so the join holds the partner and streams the
documents past it - the opposite of how the query
reads. BioDEX is the extreme, because a 4,146-token report copied
into each of 122,800 tuples is 194 times the cost of holding 200 of
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
4,146-token documents and spend 35 to 38% of their compute on
attention, so they multiply by 6.7 to 7.0. Every other query holds
documents of 11 to 370 tokens, spends under 7% on attention, and
multiplies by 8.3 to 8.6.

Long documents therefore make the big model relatively cheaper, not
dearer.

## Ground truth labels

The selectivities above come from active collection
`gt_306dac4fc83883c7a5bcc86f4d103f32`. All 23 predicates use the current
prompt layout. All 8 joins use renderer
`join_anchor_question_then_partners_v2`. The compact files contain
284,138 Qwen3 32B labels and 40,063 source labels.

The calculation uses the exact document IDs that pass each filter. It
does not infer survivor lengths from selectivity. This matters when a
filter is correlated with document length.

## What this does not settle

- **The floor is loose.** `max(T_compute, T_memory)` taken once at
  the top is weaker than taking it per kernel and summing. Both are
  lower bounds; the per-kernel one would be larger and tighter.
- **The KV read count is a minimum**, one read per anchor per stage.
  An anchor whose tuples straddle a chunk boundary is read twice.
  Nothing here is memory bound, so it changes no answer.
- **LEP-6 and LEP-8 are nearly the same number.** LEP1 leaves 4
  documents, LEP2 leaves 3, LEP3 leaves 1, and LEP4 leaves none.
- **The separate baseline report compares six of these floors with
  measured wall time.** See `reports/2026-08-25-vllm-opbench-baseline.md`.
