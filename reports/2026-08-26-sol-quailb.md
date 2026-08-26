# Speed of light for all 35 QUAIL-B queries, 4B and 32B

## What this is

The least time each QUAIL-B query at sf=0.1 can take on one H100! request,
for Qwen3-4B-fp8 and Qwen3-32B-fp8. Only three things are counted:
the dense projection FLOPs, the attention pair FLOPs, and the bytes
moved. The calculation leaves out kernel efficiency, launch gaps,
scheduling, and host work.

For joins, the calculation checks every feasible left deep relation order,
every order for predicates that connect the new relation, and every anchor
choice. A left deep plan adds one relation to the current result at each
step. The search runs separately for 4B and 32B on one H100! because the two
models have different batch limits and different work costs.

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
separately for 4B and 32B. It also stores the current planner result so the
difference remains visible.

## Ground truth setup and result

The labels were generated on Modal with four `H100!` requests and one
Qwen3 32B fp8 model copy per GPU. The four workloads were IMDB, BioDEX,
FEVER, and LePaRD. LePaRD's citation join uses source passage IDs. FEVER
uses its source annotation for 63 support pairs. Qwen judged the remaining
rows.

The completed collection has 434,201 labels across 23 predicates. The current
35 queries use 22 of those predicates, including seven join predicates. The
remaining predicate is the retired `ASPECT_RELATED` join. The summary is at
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
The JSON file also contains the current planner result and its ratio to this
minimum.

| Query | Stages | Work units | 4B SoL | 4B $/query at SoL | 4B docs/s or pairs/s at SoL | 32B SoL | 32B $/query at SoL | 32B docs/s or pairs/s at SoL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IMDB-1 | 1F | 5,000 documents | 6.722 s | $0.00737 | 743.8 | 56.413 s | $0.06188 | 88.6 |
| IMDB-2 | 1J | 60,000 pairs | 9.281 s | $0.01018 | 6,465.1 | 77.708 s | $0.08525 | 772.1 |
| IMDB-3 | 1F + 1J | 48,048 pairs | 9.565 s | $0.01049 | 5,023.4 | 80.063 s | $0.08783 | 600.1 |
| IMDB-4 | 2F + 1J | 12,336 pairs | 8.158 s | $0.00895 | 1,512.1 | 68.327 s | $0.07495 | 180.5 |
| IMDB-5 | 3F + 1J | 8,028 pairs | 8.101 s | $0.00889 | 991.0 | 67.840 s | $0.07442 | 118.3 |
| IMDB-6 | 2F | 5,000 documents | 7.419 s | $0.00814 | 673.9 | 62.223 s | $0.06826 | 80.4 |
| IMDB-7 | 3F | 5,000 documents | 7.617 s | $0.00836 | 656.4 | 63.856 s | $0.07005 | 78.3 |
| IMDB-9 | 3J | 164,412 pairs | 21.253 s | $0.02331 | 7,736.0 | 177.781 s | $0.19503 | 924.8 |
| IMDB-10 | 1F + 3J | 152,460 pairs | 21.537 s | $0.02363 | 7,078.9 | 180.136 s | $0.19761 | 846.4 |
| IMDB-8 | 2J | 104,412 pairs | 11.972 s | $0.01313 | 8,721.1 | 100.073 s | $0.10978 | 1,043.4 |
| BIO-1 | 1F | 200 documents | 4.499 s | $0.00494 | 44.5 | 31.502 s | $0.03456 | 6.3 |
| BIO-2 | 1J | 122,800 pairs | 15.592 s | $0.01710 | 7,875.9 | 104.147 s | $0.11425 | 1,179.1 |
| BIO-3 | 1F + 1J | 75,522 pairs | 11.482 s | $0.01260 | 6,577.5 | 76.857 s | $0.08431 | 982.6 |
| BIO-4 | 2F + 1J | 54,646 pairs | 9.620 s | $0.01055 | 5,680.5 | 64.656 s | $0.07093 | 845.2 |
| BIO-5 | 3F + 1J | 36,840 pairs | 7.879 s | $0.00864 | 4,675.7 | 53.707 s | $0.05892 | 685.9 |
| BIO-7 | 3J | 325,720 pairs | 38.261 s | $0.04197 | 8,513.0 | 255.351 s | $0.28012 | 1,275.6 |
| BIO-8 | 1F + 3J | 282,228 pairs | 34.346 s | $0.03768 | 8,217.1 | 229.302 s | $0.25154 | 1,230.8 |
| BIO-6 | 2J | 233,320 pairs | 25.395 s | $0.02786 | 9,187.7 | 169.052 s | $0.18545 | 1,380.2 |
| FEV-1 | 1F | 100 documents | 0.025 s | $0.00003 | 4,077.7 | 0.210 s | $0.00023 | 476.3 |
| FEV-2 | 1J | 5,700 pairs | 0.568 s | $0.00062 | 10,031.2 | 4.707 s | $0.00516 | 1,211.0 |
| FEV-3 | 1F + 1J | 3,648 pairs | 0.417 s | $0.00046 | 8,738.5 | 3.468 s | $0.00380 | 1,052.0 |
| FEV-4 | 2F + 1J | 570 pairs | 0.184 s | $0.00020 | 3,098.0 | 1.540 s | $0.00169 | 370.0 |
| FEV-5 | 2F + 1J | 2,368 pairs | 0.322 s | $0.00035 | 7,351.5 | 2.679 s | $0.00294 | 884.0 |
| FEV-6 | 3F + 1J | 370 pairs | 0.174 s | $0.00019 | 2,124.4 | 1.459 s | $0.00160 | 253.6 |
| FEV-8 | 3J | 7,211 pairs | 0.787 s | $0.00086 | 9,160.4 | 6.520 s | $0.00715 | 1,105.9 |
| FEV-9 | 1F + 3J | 5,576 pairs | 0.672 s | $0.00074 | 8,296.9 | 5.585 s | $0.00613 | 998.5 |
| FEV-7 | 2J | 7,011 pairs | 0.769 s | $0.00084 | 9,116.5 | 6.375 s | $0.00699 | 1,099.7 |
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
`gt_04231c5de83cdf9e7e68fc03849959d6`. The 22 predicates used by the current
queries have the current prompt layout. The active collection also contains
the retired `ASPECT_RELATED` labels as its 23rd predicate. All join labels use
renderer `join_anchor_question_then_partners_v2`. The compact files contain
394,138 Qwen3 32B labels and 40,063 source labels.

The calculation uses the exact document IDs that pass each filter. It does
not infer survivor lengths from selectivity. This matters when a filter is
correlated with document length.

## What this does not settle

- **The nine queries with several joins have not been checked against a
  measured Quail run.** Their token counts describe the best left deep plan
  under the stated unlimited KV rules. The current engine may use a different
  plan. This report does not compare those counts with measured fresh tokens.
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
