# Speed of light for all 35 QUAIL-B queries, 4B and 32B

## What this is

The least time each QUAIL-B query at sf=0.1 can take on one H100! request,
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

The replacement ground truth pass predicted 394,138 Qwen3 32B labels,
40,063 source labels, and 0 answer differences in a 352 judgment check.
All three label predictions matched. The pass predicted 27 to 35 minutes
for the slowest workload and measured 50.6 minutes. It predicted $5 to $9
and measured about $7.

The model specific anchor correction predicted that all 35 current queries
would choose the same orientation on 4B and 32B. The new calculation matched
the prediction. The output still stores the plan and orientation separately
for each model.

Before the regeneration, I predicted that BIO-6 would increase the most
because its second join now uses all 614 terms instead of 64 terms. BIO-6
increased from 135,088 to 240,074 evaluated pairs. Its SoL increased from
16.663 to 26.033 seconds for 4B and from 111.222 to 173.157 seconds for 32B.
BIO-C and BIO-D changed by less than 0.3 percent because the new REACTION
labels removed a slightly different set of rows before their later joins.

## Ground truth setup and result

The labels were generated on Modal with four `H100!` requests and one
Qwen3 32B fp8 model copy per GPU. The four workloads were IMDB, BioDEX,
FEVER, and LePaRD. LePaRD's citation join uses source passage IDs. FEVER
uses its source annotation for 63 support pairs. Qwen judged the remaining
rows.

The completed collection has 434,201 labels across 23 predicates. The
collection includes all eight join predicates used by the 35 queries. Its
summary is at
`/results/ground_truth/quailb/schema_v1/collections/gt_04231c5de83cdf9e7e68fc03849959d6/summary.json`
on the `quail-results` volume. The collection is active for corpus
`c_df45ef585738f42e4a7a731306f1b9fc`.

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

   These come from collection `gt_04231c5de83cdf9e7e68fc03849959d6`,
   the labels active for corpus `c_df45ef585738f42e4a7a731306f1b9fc`.
   The collection has 434,201 labels. Its deterministic check repeated
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

## How queries with several joins are counted

The calculation follows the physical plan stage by stage. It uses the
saved TRUE and FALSE pair labels to determine which documents reach each
later join. Consecutive stages with the same anchor reuse the document KV
and write a new complete question once per surviving anchor.

An anchor change creates a barrier. The calculation uses the passing pair
relations at the barrier to remove documents that cannot appear in the
final result. It then computes the new anchor prefix once and continues
with the smaller document sets.

The calculation builds the physical plan separately for 4B and 32B. A free
join chooses its runtime anchor with that model's chunk limit, which is
110,376 tokens for 4B and 41,943 tokens for 32B. Work counts, stage details,
and anchor choices are stored separately for each model. The current 35
queries choose the same orientations on both models.

For example, IMDB-9 evaluates 60,000 review and aspect pairs in its first
join. The first join leaves all 12 aspect values live for the next stage,
so the next two joins evaluate 144 pairs each. The total is 60,288 pair
evaluations. BIO-C evaluates 122,800 pairs, then 107,800 pairs after its
first barrier, then 117,274 pairs after the next gate. The total is
347,874 pair evaluations.

The SoL work is the sum of all filter and join stages. For a query with a
join, document pairs per second at SoL is the sum of evaluated pairs across
its join stages divided by the complete SoL time. For a filter only query,
documents per second at SoL is the number of input document rows divided by
the complete SoL time. Dollars per query at SoL uses $3.9492 per GPU hour for
an H100! request, which
is the same price used by the benchmark evaluation code. Modal lists the
same price as $0.001097 per second on its
[pricing page](https://modal.com/pricing).

## Result

Figure: plots/sol_quailb_per_query.png

The cost and throughput columns apply the benchmark metric formulas to SoL
time. They are not measured Quail metrics. Measured Quail metrics require the
wall time from an engine run. The cost at SoL includes GPU time only, so it
does not include CPU or memory charges.

| Query | Stages | Work units | 4B SoL | 4B $/query at SoL | 4B docs/s or pairs/s at SoL | 32B SoL | 32B $/query at SoL | 32B docs/s or pairs/s at SoL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IMDB-1 | 1F | 5,000 documents | 6.722 s | $0.00737 | 743.8 | 56.413 s | $0.06188 | 88.6 |
| IMDB-2 | 1J | 60,000 pairs | 9.281 s | $0.01018 | 6,465.1 | 77.708 s | $0.08525 | 772.1 |
| IMDB-3 | 1F + 1J | 48,048 pairs | 9.565 s | $0.01049 | 5,023.4 | 80.063 s | $0.08783 | 600.1 |
| IMDB-4 | 2F + 1J | 12,336 pairs | 8.158 s | $0.00895 | 1,512.1 | 68.327 s | $0.07495 | 180.5 |
| IMDB-5 | 3F + 1J | 8,028 pairs | 8.101 s | $0.00889 | 991.0 | 67.840 s | $0.07442 | 118.3 |
| IMDB-6 | 2F | 5,000 documents | 7.419 s | $0.00814 | 673.9 | 62.223 s | $0.06826 | 80.4 |
| IMDB-7 | 3F | 5,000 documents | 7.617 s | $0.00836 | 656.4 | 63.856 s | $0.07005 | 78.3 |
| IMDB-9 | 3J | 60,288 pairs | 9.297 s | $0.01020 | 6,484.4 | 77.852 s | $0.08540 | 774.4 |
| IMDB-11 | 1F + 3J | 48,336 pairs | 9.582 s | $0.01051 | 5,044.6 | 80.207 s | $0.08799 | 602.6 |
| IMDB-8 | 2J | 115,224 pairs | 12.602 s | $0.01382 | 9,143.3 | 105.337 s | $0.11555 | 1,093.9 |
| BIO-1 | 1F | 200 documents | 4.499 s | $0.00494 | 44.5 | 31.502 s | $0.03456 | 6.3 |
| BIO-2 | 1J | 122,800 pairs | 15.592 s | $0.01710 | 7,875.9 | 104.147 s | $0.11425 | 1,179.1 |
| BIO-3 | 1F + 1J | 75,522 pairs | 11.482 s | $0.01260 | 6,577.5 | 76.857 s | $0.08431 | 982.6 |
| BIO-4 | 2F + 1J | 54,646 pairs | 9.620 s | $0.01055 | 5,680.5 | 64.656 s | $0.07093 | 845.2 |
| BIO-5 | 3F + 1J | 36,840 pairs | 7.879 s | $0.00864 | 4,675.7 | 53.707 s | $0.05892 | 685.9 |
| BIO-C | 3J | 347,874 pairs | 40.195 s | $0.04409 | 8,654.7 | 267.945 s | $0.29394 | 1,298.3 |
| BIO-D | 1F + 3J | 289,596 pairs | 35.021 s | $0.03842 | 8,269.2 | 233.692 s | $0.25636 | 1,239.2 |
| BIO-6 | 2J | 240,074 pairs | 26.033 s | $0.02856 | 9,221.9 | 173.157 s | $0.18995 | 1,386.5 |
| FEV-1 | 1F | 100 documents | 0.025 s | $0.00003 | 4,077.7 | 0.210 s | $0.00023 | 476.3 |
| FEV-2 | 1J | 5,700 pairs | 0.568 s | $0.00062 | 10,031.2 | 4.707 s | $0.00516 | 1,211.0 |
| FEV-3 | 1F + 1J | 3,648 pairs | 0.417 s | $0.00046 | 8,738.5 | 3.468 s | $0.00380 | 1,052.0 |
| FEV-4 | 2F + 1J | 570 pairs | 0.184 s | $0.00020 | 3,098.0 | 1.540 s | $0.00169 | 370.0 |
| FEV-5 | 2F + 1J | 2,368 pairs | 0.322 s | $0.00035 | 7,351.5 | 2.679 s | $0.00294 | 884.0 |
| FEV-6 | 3F + 1J | 370 pairs | 0.174 s | $0.00019 | 2,124.4 | 1.459 s | $0.00160 | 253.6 |
| FEV-C | 3J | 9,896 pairs | 1.011 s | $0.00111 | 9,784.4 | 8.384 s | $0.00920 | 1,180.4 |
| FEV-D | 1F + 3J | 6,431 pairs | 0.739 s | $0.00081 | 8,700.3 | 6.140 s | $0.00674 | 1,047.4 |
| FEV-7 | 2J | 7,296 pairs | 0.791 s | $0.00087 | 9,228.1 | 6.554 s | $0.00719 | 1,113.2 |
| LEP-1 | 1F | 200 documents | 0.219 s | $0.00024 | 912.7 | 1.848 s | $0.00203 | 108.2 |
| LEP-2 | 1J | 40,000 pairs | 14.582 s | $0.01600 | 2,743.1 | 121.563 s | $0.13335 | 329.0 |
| LEP-3 | 1F + 1J | 800 pairs | 0.515 s | $0.00057 | 1,553.0 | 4.277 s | $0.00469 | 187.0 |
| LEP-4 | 2F + 1J | 600 pairs | 0.442 s | $0.00048 | 1,358.7 | 3.675 s | $0.00403 | 163.3 |
| LEP-5 | 3F + 1J | 200 pairs | 0.295 s | $0.00032 | 677.1 | 2.469 s | $0.00271 | 81.0 |
| LEP-6 | 5F + 1J | 0 pairs | 0.221 s | $0.00024 | 0.0 | 1.860 s | $0.00204 | 0.0 |
| LEP-7 | 3F + 1J | 600 pairs | 0.537 s | $0.00059 | 1,118.1 | 4.486 s | $0.00492 | 133.8 |
| LEP-8 | 5F | 200 documents | 0.221 s | $0.00024 | 906.3 | 1.860 s | $0.00204 | 107.5 |

Every query on both models is compute bound. The largest
`T_memory` in the suite is FEV-1's at 6.7% of its `T_compute`, and
FEV-1 is the smallest query here - 100 documents of 11 tokens.
Everything else pushes enough tokens per pass that the weight reads
amortize away.

## Which side of a join is held

A join has to pick a side to anchor. Anchoring the long side costs
one prefix per document; anchoring the short side copies every long
document into every tuple. The bound prices both feasible choices for
each model and keeps the cheaper one, because that is what the engine
does. Both models make the same choices for every join in the current
suite, so one orientation is shown below.

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

The attention share figure contains the 26 queries with no more than one
join. The nine queries with several joins can use more than one anchor
context, so one context length would not describe their work correctly.
Each plotted point uses the anchor context chosen for that model.

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
3.56 and 8.59, at whatever its mix is. The five plotted BioDEX queries hold
4,146-token documents and spend 35 to 38% of their compute on
attention, so they multiply by 6.7 to 7.0. Every other plotted query holds
documents of 11 to 370 tokens, spends under 7% on attention, and
multiplies by 8.3 to 8.6.

Long documents therefore make the big model relatively cheaper, not
dearer.

## Ground truth labels

The selectivities above come from active collection
`gt_04231c5de83cdf9e7e68fc03849959d6`. All 23 predicates use the current
prompt layout. All 8 joins use renderer
`join_anchor_question_then_partners_v2`. The compact files contain
394,138 Qwen3 32B labels and 40,063 source labels.

The calculation uses the exact document IDs that pass each filter. It
does not infer survivor lengths from selectivity. This matters when a
filter is correlated with document length.

## What this does not settle

- **The nine queries with several joins have not been checked against a
  measured Quail run.** Their token counts follow the physical plan and the
  exact saved labels, but this report does not yet compare those counts with
  the engine's measured fresh token count.
- **The floor is loose.** `max(T_compute, T_memory)` taken once at
  the top is weaker than taking it per kernel and summing. Both are
  lower bounds; the per-kernel one would be larger and tighter.
- **The KV read count is a minimum**, one read per anchor per stage.
  An anchor whose tuples straddle a chunk boundary is read twice.
  Nothing here is memory bound, so it changes no answer.
- **LEP-6 and LEP-8 are nearly the same number.** LEP1 leaves 4
  documents, LEP2 leaves 3, LEP3 leaves 1, and LEP4 leaves none.
- **The separate baseline report compares six of these floors with
  measured wall time.** See `reports/2026-08-26-stock-vllm-joins.md`.
