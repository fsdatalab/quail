# QUAIL-B reference labels at sf=1.0, judged by Quail

Date: 2026-09-09.

The first labeling pass at scale factor 1.0, and the first where Quail
itself is the judge. It produced collection
`gt_3792d237611fe2fe9c8eca67002f4213` for corpus
`c_81a95887a650aaa1a343e0d688b81bef`; the sf=0.5 and sf=0.1
collections were derived from it on the CPU. The vLLM-judged sf=0.1
collection `gt_77bb8b128743a79aedddaa24c808c3f8` is no longer the
reference.

## Setup

- Judge: Qwen3 32B fp8 through Quail's executor, argmax over the TRUE
  and FALSE logits after one forward pass. 21 predicates.
- One H100 per workload on Modal, five containers side by side. A join
  part holds 500,000 pairs, a filter part 4,096 prompts; parts commit
  to the `quail-results` volume as they finish.
- Corpus at sf=1.0: reviews 50,000, reports 5,000, terms 4,144, claims
  5,000, evidence 1,478, citation contexts 4,972, citation passages
  2,991, agent traces 17,711 (every eligible SWE-Next snapshot; the
  17,718 base count was 7 more than the source has).

## Prediction

51,801,003 labels: 36,926,465 model judgments and 14,874,538 source
labels. About 23 H100 hours in total, no workload over 8 hours: IMDB
2.3, BioDEX 7.0, FEVER 6.5, LePaRD 0.2, agent 6.9. The rerun of 16
saved answers per predicate shows no differences.

## Result

Label counts matched the prediction exactly. Hours per workload, in
the run that finished each one:

| Workload | Predicted | Measured | Ratio |
|---|---:|---:|---:|
| IMDB | 2.3 | 1.41 | 0.61 |
| BioDEX | 7.0 | 7.19 | 1.03 |
| FEVER | 6.5 | 5.96 | 0.92 |
| LePaRD | 0.2 | 0.04 | |
| Agent | 6.9 | 7.72 | 1.12 |
| Sum | 22.9 | 22.33 | 0.98 |

- The measured hours exclude parts a cancelled earlier run had written
  (all of LePaRD, the BioDEX and FEVER filters, a third of IMDB's
  filters), about 1.2 hours. GPU cost of the finishing run: $88.19 at
  $3.9492 per H100 hour. Total spend with the failed launches: about
  $105.
- Wall time 7.72 hours, set by the agent traces, which ran at about
  9,500 fresh tokens per second against the 14,000 assumed; each
  document is about 10,000 tokens. BioDEX ran at 13,600 and FEVER at
  16,500 fresh tokens per second, with the planner anchoring FEVER on
  the passages at about 22 fresh tokens per pair, as predicted.
- The rerun prediction did not hold: 18 of 320 answers changed, all
  in IMDB. A one-off check showed the three IMDB filters reproduce
  exactly, and the two IMDB joins reproduce exactly when asked as a
  batch but flip when asked one pair at a time (3 of 16 and 15 of 16),
  because a one-pair query lets the planner anchor on the aspect
  instead of the review, which changes the prompt order.

## The prompt order follows the planner's anchor

No predicate names an anchor, so the planner picks the anchor table
by cost and writes that document first. In this pass BioDEX anchored
on reports, IMDB on reviews, and both FEVER joins on the passages; the
vLLM-judged labels were made claim-first. Comparing the derived
Quail-judged sf=0.1 collection with the vLLM-judged one on the same
rows:

| Predicate | Rows | Differ % | TRUE % vLLM | TRUE % Quail |
|---|---:|---:|---:|---:|
| fever.passage.refutes_claim | 143,500 | 17.10 | 0.3 | 17.2 |
| fever.passage.supports_claim | 143,500 | 5.12 | 0.2 | 5.3 |
| imdb.review.discusses_aspect | 60,000 | 5.32 | 29.5 | 32.8 |
| imdb.review.positive_sentiment_about_aspect | 60,000 | 4.00 | 16.1 | 18.3 |
| biodex.report.experienced_reaction | 563,500 | 1.38 | 3.4 | 4.0 |
| the other 16 predicates | 239,764 | 1.3 | | |
| all 21 | 1,210,264 | 3.76 | | |

Predicates with the same prompt order differ by 0.2 to 6%, the engine
difference on borderline answers. The two FEVER joins, whose order
flipped, differ far more: passage-first, the judge calls 17.2% of all
claim-passage pairs refutations, against 0.3% claim-first.

Decision: keep what we have. No anchor is declared for any predicate;
the labeling pass and the benchmark queries both let the planner
choose, so the reference labels and the measured runs send the same
prompt order.

## Derived collections

- sf=0.5: `gt_245ed902a5db326c55235b17f72f2695` for corpus
  `c_6773c85b3754908434661c1dadfad0fa`, 17,618,467 labels, the
  predicted count, 1,607 seconds on a CPU container.
- sf=0.1: `gt_83ef3a3c60ae670a2d2e3192d5adff69` for the existing
  corpus `c_1aa2c4f0d0b6c816fd37aa5748c33341`, 1,210,264 labels, 344
  seconds. Two of its 216,500 LePaRD citation labels differ from the
  sf=1.0 corpus's, as predicted.

## Data

Under `/results/ground_truth/quailb/schema_v1/` on `quail-results`:
`collections/<id>/summary.json`, `corpora/<id>/manifest.json`, and
`label_sets/<workload>/<slug>/<label_set_id>/labels.parquet`. Modal
function calls of the finishing run: judge_workload imdb
`fc-01M21FNWBB18V78B2HZ02HQ1TP`, biodex `fc-01M21FNWFJ77AEAT4F2AZ6BB70`,
fever `fc-01M21FNWJF85YYH1DTDM8H9EAR`, lepard
`fc-01M21FNWNCBXXRSANTKTVCRK2P`, agent `fc-01M21FNWRGD8TESC3G9NDHM2XN`;
finalize `fc-01M22A5Y5EVS3E7HV9MN49WBRA`; derive sf=0.5
`fc-01M22A7S81YMEPJ5QDXKY1TVT7`, sf=0.1 `fc-01M22A7S9DV9NGX02F9HDSJYB3`.

## What is left

- Done: the three collections and both new corpora are in the bucket.
  From a machine without Modal, `load_ground_truth` reads the sf=0.1
  collection in 9 seconds and the sf=0.5 collection (17,618,467
  labels) in 103 seconds, and `build_sets` fetches the sf=0.5 and
  sf=1.0 corpora in 29 and 39 seconds. quail-bench main lists the
  corpus ids in `PUBLISHED_CORPORA`.
- `quail_b.labels.load_ground_truth` holds a whole collection in one
  Python dictionary: 51.8 million entries, about 10 GB, at sf=1.0.
  Scoring an sf=1.0 run needs a lighter loader first.
