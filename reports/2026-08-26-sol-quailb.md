# Speed of light for all 35 QUAIL-B queries, 4B and 32B

## What this is

The least time each QUAIL-B query at sf=0.1 can take on one H100! request,
for Qwen3-4B-fp8 and Qwen3-32B-fp8. The model has three components:
attention projections, the MLP, and attention over KV. Each component
counts FLOPs and bytes, takes the larger of compute time and memory time,
and adds its time to the query total. The calculation leaves out kernel
efficiency, launch gaps, scheduling, and host work.

For joins, the calculation checks every feasible left deep relation order,
every order for predicates that connect the new relation, and every anchor
choice. A left deep plan adds one relation to the current result at each
step. The search runs separately for 4B and 32B on one H100! because the two
models have different batch limits and different work costs.

Nothing here is fitted or measured on a GPU. The equations are in
`plans/sol_model.md`. Work counting, Qwen3 component definitions, and the
generic component calculation are separate modules under `quail/planner/`.
`reports/make_sol_quailb.py` measures their inputs and applies the shared
calculator.

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

Before adding the join optimizer, I predicted three results:

1. Filter only queries and queries with one join would not change.
2. Some queries with several joins would improve because the current planner
   does not check every left deep order using exact ground truth survivors.
3. The subset DP pruning rule would match complete enumeration in a small
   controlled test.

All three predictions matched. The 26 queries with at most one join are
unchanged. Seven of the nine queries with several joins improve. Two are
unchanged. The largest improvement is FEV-8. Its 4B estimate decreases from
0.900 seconds with the current planner to 0.787 seconds with the best left
deep plan. Its 32B estimate decreases from 7.460 seconds to 6.520 seconds.

The unit test for the DP matches complete enumeration. The production SoL run
does not repeat complete enumeration. The output stores the best plan
separately for 4B and 32B. The planner comparison above was computed during
development. It is not part of the SoL output.

The filter order change predicted two more results:

1. Fixed selectivity estimates would move cheaper, more selective filters
   earlier and reduce total work.
2. Adding the dense and attention limits separately would only raise a result
   when one part is compute bound and the other is memory bound.

Both predictions matched. The sum across all 35 queries decreases from
327.10 to 324.65 seconds for 4B, and from 2,490.56 to 2,470.11 seconds for
32B. Ten queries change filter order. The four affected IMDB filter chains
decrease by 7.0% to 8.2%. FEV-1 is the only mixed query. Its 4B estimate
increases by 0.65% because its attention work is memory bound.

The component refactor predicted that every chosen plan would remain the
same. All dense components were already compute bound, so reducing their
weight bytes would not change their time. Matching the component model also
omits norms from dense FLOPs, which predicted a very small decrease. The
prediction matched. Every filter order, join order, and anchor stayed the
same. The suite totals decreased from 324.6505 to 324.6364 seconds for 4B,
and from 2,470.1103 to 2,470.0616 seconds for 32B. The decreases are 0.0043%
and 0.0020%.

## Ground truth setup and result

The labels were generated on Modal with four `H100!` requests and one
Qwen3 32B fp8 model copy per GPU. The four workloads were IMDB, BioDEX,
FEVER, and LePaRD. LePaRD's citation join uses source passage IDs. FEVER
uses its source annotation for 63 support pairs. Qwen judged the remaining
rows.

The SoL input combines two matching sources. IMDB, BioDEX, and FEVER use
collection `gt_04231c5de83cdf9e7e68fc03849959d6`. LePaRD uses the revised
labels for the 500 document corpus `c_350e4ae7332a3dcf6d1292b96fe05a0a`.
Together they contain 612,434 labels across 23 predicates. The current 35
queries use 22 of those predicates. The remaining predicate is the retired
`ASPECT_RELATED` join. The non-LePaRD collection is at
`/results/ground_truth/quailb/schema_v1/collections/gt_04231c5de83cdf9e7e68fc03849959d6/summary.json`
on the `quail-results` volume. The revised LePaRD label sets are under
`/results/ground_truth/quailb/schema_v1/label_sets/lepard/` on the same
volume.

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
   | LePaRD | documents | 500 | 103,947 | 207.9 |
   | LePaRD | partner | 433 | 25,621 | 59.2 |

   "Documents" is the set a query is written against; "partner" is
   the set its join runs against.

2. **Prompt lengths.** The shared preamble is 2 tokens. Each
   filter's question is 41 to 62 tokens. Each join writes the anchor
   note and complete question once per anchor. Each tuple carries the
   partner label, partner document, and answer cue.

3. **Ground truth labels.** The planner gets one fixed marginal selectivity
   estimate per predicate. It is the TRUE count divided by the complete label
   count at sf=0.1. For example, F4 has 1,218 TRUE labels out of 5,000, or
   0.2436. This makes F4 run before F1 in the affected IMDB chains.

   After the order is selected, the calculation uses each saved TRUE or FALSE
   answer to determine the exact document IDs that reach the next stage. It
   does not multiply row counts by the marginal estimate. IMDB, BioDEX, and
   FEVER use collection `gt_04231c5de83cdf9e7e68fc03849959d6`. LePaRD uses
   the revised labels for corpus `c_350e4ae7332a3dcf6d1292b96fe05a0a`.

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

Read straight off the saved work: IMDB-1 is one filter over 5,000 IMDB
documents at 1,759,233 tokens. IMDB-6 now runs F4 first because its fixed
selectivity is lower. The complete chain is 1,791,351 tokens. The first scan
is 1,729,233 tokens. F4 passes 1,218 documents, so the later F1 question adds
exactly `1,218 * 51 = 62,118` fresh tokens. Nothing is scanned again.

A join writes the complete anchor frame once after each anchor
document. The other side streams. Each tuple carries only the partner
label, partner document, and answer cue. Per tuple that is one
rectangle over the anchor document and frame, plus one triangle over
the tuple's own tokens. Suffixes are atomic and never attend to each
other.

The equations in full, joins included, are `plans/sol_model.md`
section 3.

## How queries with several joins are counted

All filters run before the join search. The first filter on an alias computes
its document prefix KV. Later filters on the same alias reuse that KV. Every
surviving document from a filtered alias is therefore already available in
KV when joins start.

The join search uses a subset DP. A DP state contains two sets:

1. The aliases already joined.
2. The aliases whose document prefix KV is available.

The second set is needed because anchor choices change future work. The model
assumes unlimited KV capacity. Once an alias enters the second set, it stays
there. Only document prefix KV is reusable. Partner suffix KV is not reusable.
An alias with no filter enters the set the first time it is used as an anchor.

For each state, the DP tries every relation that has a join predicate to the
current subset. It tries every order for predicates that connect the new
relation. It tries both ends of every binary predicate as the anchor. Each
choice includes the exact anchor marker, partner marker, join question,
partner document, and answer cue token counts.

The saved TRUE and FALSE labels give the exact rows that survive each logical
intermediate. A join graph without a cycle uses repeated exact filtering over
its edges. A join graph with a cycle uses exact assignment checks because
pairwise filtering alone can keep rows that do not occur in any complete
result.

Several work records can reach the same state. A record is removed only when
another record is no larger in fresh tokens, attention pairs, KV writes, and
KV reads. Compute time and memory time are combined only after the final plan
is known.

The search runs separately with the 4B and 32B batch limits. A small unit test
compares the DP result with complete enumeration. The QUAIL-B calculation runs
only the DP.

For example, the best IMDB-9 plan evaluates 60,000 pairs, then 44,412 pairs,
then 60,000 pairs. The total is 164,412 pair evaluations, compared with
175,224 for the current planner. The best BIO-7 plan evaluates 122,800 pairs,
then 92,400 pairs, then 110,520 pairs. The total is 325,720 pairs, compared
with 341,120 for the current planner.

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

For queries with several joins, the table reports the best left deep plan.
The JSON file contains only the unlimited KV SoL result.

| Query | Stages | Work units | 4B SoL | 4B $/query at SoL | 4B docs/s or pairs/s at SoL | 32B SoL | 32B $/query at SoL | 32B docs/s or pairs/s at SoL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IMDB-1 | 1F | 5,000 documents | 6.722 s | $0.00737 | 743.9 | 56.412 s | $0.06188 | 88.6 |
| IMDB-2 | 1J | 60,000 pairs | 9.280 s | $0.01018 | 6,465.5 | 77.706 s | $0.08524 | 772.1 |
| IMDB-3 | 1F + 1J | 48,048 pairs | 9.564 s | $0.01049 | 5,023.7 | 80.061 s | $0.08783 | 600.1 |
| IMDB-4 | 2F + 1J | 12,336 pairs | 7.588 s | $0.00832 | 1,625.8 | 63.561 s | $0.06973 | 194.1 |
| IMDB-5 | 3F + 1J | 8,028 pairs | 7.475 s | $0.00820 | 1,074.0 | 62.615 s | $0.06869 | 128.2 |
| IMDB-6 | 2F | 5,000 documents | 6.849 s | $0.00751 | 730.1 | 57.457 s | $0.06303 | 87.0 |
| IMDB-7 | 3F | 5,000 documents | 6.991 s | $0.00767 | 715.2 | 58.631 s | $0.06432 | 85.3 |
| IMDB-9 | 3J | 164,412 pairs | 21.252 s | $0.02331 | 7,736.4 | 177.777 s | $0.19502 | 924.8 |
| IMDB-10 | 1F + 3J | 152,460 pairs | 21.536 s | $0.02363 | 7,079.3 | 180.132 s | $0.19760 | 846.4 |
| IMDB-8 | 2J | 104,412 pairs | 11.972 s | $0.01313 | 8,721.5 | 100.071 s | $0.10978 | 1,043.4 |
| BIO-1 | 1F | 200 documents | 4.499 s | $0.00494 | 44.5 | 31.501 s | $0.03456 | 6.3 |
| BIO-2 | 1J | 122,800 pairs | 15.591 s | $0.01710 | 7,876.2 | 104.145 s | $0.11425 | 1,179.1 |
| BIO-3 | 1F + 1J | 75,522 pairs | 11.481 s | $0.01260 | 6,577.8 | 76.855 s | $0.08431 | 982.7 |
| BIO-4 | 2F + 1J | 54,646 pairs | 9.620 s | $0.01055 | 5,680.7 | 64.655 s | $0.07093 | 845.2 |
| BIO-5 | 3F + 1J | 36,840 pairs | 7.876 s | $0.00864 | 4,677.3 | 53.692 s | $0.05890 | 686.1 |
| BIO-7 | 3J | 325,720 pairs | 38.260 s | $0.04197 | 8,513.3 | 255.347 s | $0.28012 | 1,275.6 |
| BIO-8 | 1F + 3J | 282,228 pairs | 34.345 s | $0.03768 | 8,217.4 | 229.298 s | $0.25154 | 1,230.8 |
| BIO-6 | 2J | 233,320 pairs | 25.394 s | $0.02786 | 9,188.0 | 169.049 s | $0.18545 | 1,380.2 |
| FEV-1 | 1F | 100 documents | 0.025 s | $0.00003 | 4,051.7 | 0.210 s | $0.00023 | 476.2 |
| FEV-2 | 1J | 5,700 pairs | 0.568 s | $0.00062 | 10,031.7 | 4.707 s | $0.00516 | 1,211.0 |
| FEV-3 | 1F + 1J | 3,648 pairs | 0.417 s | $0.00046 | 8,738.9 | 3.468 s | $0.00380 | 1,052.0 |
| FEV-4 | 2F + 1J | 570 pairs | 0.174 s | $0.00019 | 3,277.6 | 1.454 s | $0.00160 | 392.0 |
| FEV-5 | 2F + 1J | 2,368 pairs | 0.322 s | $0.00035 | 7,351.9 | 2.679 s | $0.00294 | 884.1 |
| FEV-6 | 3F + 1J | 370 pairs | 0.164 s | $0.00018 | 2,255.0 | 1.373 s | $0.00151 | 269.5 |
| FEV-8 | 3J | 7,211 pairs | 0.787 s | $0.00086 | 9,160.9 | 6.520 s | $0.00715 | 1,105.9 |
| FEV-9 | 1F + 3J | 5,576 pairs | 0.672 s | $0.00074 | 8,297.3 | 5.584 s | $0.00613 | 998.5 |
| FEV-7 | 2J | 7,011 pairs | 0.769 s | $0.00084 | 9,116.9 | 6.375 s | $0.00699 | 1,099.7 |
| LEP-1 | 1F | 500 documents | 0.499 s | $0.00055 | 1,002.4 | 4.212 s | $0.00462 | 118.7 |
| LEP-2 | 1J | 216,500 pairs | 58.091 s | $0.06373 | 3,726.9 | 485.593 s | $0.53270 | 445.8 |
| LEP-3 | 1F + 1J | 6,062 pairs | 2.151 s | $0.00236 | 2,818.9 | 17.844 s | $0.01957 | 339.7 |
| LEP-4 | 2F + 1J | 2,165 pairs | 1.092 s | $0.00120 | 1,983.3 | 9.103 s | $0.00999 | 237.8 |
| LEP-5 | 3F + 1J | 0 pairs | 0.496 s | $0.00054 | 0.0 | 4.190 s | $0.00460 | 0.0 |
| LEP-6 | 5F + 1J | 0 pairs | 0.488 s | $0.00054 | 0.0 | 4.124 s | $0.00452 | 0.0 |
| LEP-7 | 3F + 1J | 1,755 pairs | 1.138 s | $0.00125 | 1,542.1 | 9.536 s | $0.01046 | 184.0 |
| LEP-8 | 5F | 500 documents | 0.488 s | $0.00054 | 1,023.8 | 4.124 s | $0.00452 | 121.2 |

The attention projection and MLP components are compute bound for every
query. The attention component is also compute bound for every query except
FEV-1. For the 4B FEV-1 result, the two dense components take 0.024389
seconds and attention memory takes 0.000292 seconds. The total is 0.024681
seconds. Every other query is compute bound in all three components.

## Which side of a join is held

A join has to pick a side to anchor. Anchoring the long side costs
one prefix per document; anchoring the short side copies every long
document into every tuple. The search prices both feasible choices for
each model and keeps the one that contributes to the best complete plan.
Both models make the same choices for every join in the current suite, so
one orientation is shown below.

| join | side held | side streamed | join tokens as run | join tokens the other way | cost of choosing wrong |
|---|---|---|---|---|---|
| IMDB-2 | documents, 299 tokens | partner, 2 | 2,419,233 | 18,531,279 | 7.7x |
| BIO-2 | documents, 4,146 | partner, 5 | 2,635,599 | 510,386,050 | 193.7x |
| FEV-2 | partner, 370 | documents, 11 | 145,359 | 2,171,842 | 14.9x |
| FEV-5 | partner, 370 | documents, 11 | 51,578 | 914,560 | 17.7x |
| LEP-2 | documents, 208 | partner, 59 | 15,098,947 | 47,216,559 | 3.1x |

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

The dense and attention components grow differently with **context length**,
which is how many earlier tokens each new token attends over. The dense
components do not depend on context length. They use 2 FLOPs per modeled
projection parameter per token.
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

The components also scale differently with model size. The two dense
components scale with their combined parameter count, 31,205,621,760 over
3,633,315,840 or 8.59x. `T_attention` scales with `4 n_q d_head L`, 2,097,152
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

The IMDB, BioDEX, and FEVER selectivities come from collection
`gt_04231c5de83cdf9e7e68fc03849959d6`. The LePaRD selectivities come from the
revised label sets for corpus `c_350e4ae7332a3dcf6d1292b96fe05a0a`. The 22
predicates used by the current queries have the current prompt layout. The
combined input also contains the retired `ASPECT_RELATED` labels as its 23rd
predicate. All join labels use renderer
`join_anchor_question_then_partners_v2`. The combined input has 612,434
labels.

The calculation uses the exact document IDs that pass each filter. It does
not infer survivor lengths from selectivity. This matters when a filter is
correlated with document length.

## What this does not settle

- **The nine queries with several joins have not been checked against a
  measured Quail run.** Their token counts describe the best left deep plan
  under the stated unlimited KV rules. The current engine may use a different
  plan. This report does not compare those counts with measured fresh tokens.
- **The KV read count is a minimum**, one read per anchor per stage.
  An anchor whose tuples straddle a chunk boundary is read twice.
  FEV-1 attention is memory bound. Extra KV reads could raise that one
  estimate. Every other attention stage is compute bound.
- **LEP-6 and LEP-8 have the same work.** The chosen filter order is LEP5,
  LEP3, LEP1, LEP4, then LEP2. The first two filters leave 2 documents and
  LEP1 leaves none. The join in LEP-6 therefore evaluates no pairs.
- **The separate baseline report compares six of these floors with
  measured wall time.** See `reports/2026-08-26-stock-vllm-joins.md`.

## Rebuild

Run `reports/make_sol_quailb_plots.py` from the repository root. Its docstring
contains the `modal volume get` command and the plotting command needed to
rebuild both figures from `/results/sol/sol_quailb_sf0.1.json`.
