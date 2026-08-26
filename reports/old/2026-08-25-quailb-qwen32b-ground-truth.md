> Superseded by `reports/2026-08-26-parallel-judge-pass.md`. The
> selectivities below were measured with templates that carried a
> duplicated `ANSWER=` cue, removed in PR #56. Relabelling moved 14
> of the 19 predicates, the BioDEX reaction join by a factor of 244.
> The collection described here, `gt_42674891c824e01c6d966eb48c9cf8c7`,
> is no longer active.

# QUAIL-B Qwen3 32B ground truth at scale factor 0.1

## Result

The run finished and saved 245,557 labels for 19 predicates.

- Qwen3 32B produced 205,494 labels.
- Existing source data supplied 40,063 labels.
- BioDEX reactions were relabeled by Qwen. The source reaction field was not used as truth.
- The raw collection is 46.90 MiB after adding compact reader files. At Modal's $0.09 per GiB per month volume rate, it costs about $0.0041 per month to store.
- The two H100 calls cost about $4.57 before the small CPU charge.

The committed aggregate summary is `results/benchmark/20260825T082145Z-quailb-qwen32b-ground-truth-sf0.1.json`. The raw per-row data is on the `quail-results` Modal volume.

## Setup

- Scale factor: 0.1.
- Corpus seed: 20260818.
- Model: `Qwen/Qwen3-32B-FP8` at revision `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`.
- Hardware: one H100 with 96 GiB of host memory.
- Modal app: `quail-milestone1`.
- Cell: `quailb_judge_pass`.
- Generation: temperature 0, one output token, and only `TRUE` or `FALSE` allowed.
- Source datasets and the model are pinned to exact revisions.

One GPU was enough. More GPUs could process independent predicates at the same time, with one full model copy on each GPU. That would shorten elapsed time, but each copy would pay its own model load. At scale factor 0.1, one GPU kept the total cost lower and avoided extra recovery work.

## Prediction

Before the run, the prediction was:

- 205,457 Qwen judgments.
- 40,100 source labels.
- 245,557 total labels.
- 30 to 45 minutes on one H100, including model load.
- $3 to $5 at current Modal prices.
- 0 answer differences in the repeated deterministic sample.

## Measured

The total label count and cost matched the prediction.

- The first call stopped after about 19.1 minutes. Its completed Parquet parts stayed on the volume.
- The resumed call finished in 39.1 minutes. This included 7.1 minutes to load the model and 27.9 minutes inside model generation.
- Total elapsed compute across both calls was about 58.2 minutes.
- At $0.001097 per H100 second and $0.00000222 per GiB-second for 96 GiB of host memory, that is about $4.57 before the small CPU charge. The rate comes from [Modal's pricing page](https://modal.com/pricing).
- The repeated sample had 0 answer differences across 48 labels. It submitted the same prompts in reverse order.

The predicted source split was off by 37 labels. Only 63 of the 100 sampled FEVER claims had their annotated page in the 57-page benchmark evidence pool. Those 63 matching pairs used FEVER labels. Qwen labeled the other claim and passage pairs. This produced 205,494 Qwen labels and 40,063 source labels.

The 48-label repeat is smaller than intended because the first call stopped. The resumed call sampled only the Qwen predicates that it completed. The code now rebuilds the repeat sample from all saved parts, so a future resumed run covers 16 saved rows from every Qwen predicate.

Function calls:

- `fc-01M0VX1TV57EKVMN5JEN283F72`: stopped after partial progress.
- `fc-01M0VY5HEYV3SKKKQEYA5403Y8`: resumed and completed the collection.

## Label sources

Existing source labels were used only where the benchmark has a direct answer.

- LePaRD passage IDs supplied all 40,000 citation join labels.
- FEVER supplied 63 labels for the matching annotated claim and page pairs.
- Qwen supplied every other label.
- BioDEX supplied reports and reaction terms, but none of its reaction pairs were accepted as truth.

Each predicate has its own stable key and label set ID. Names such as `F1` are kept only as old display names. The stable ID includes the full predicate text, argument roles, corpus identity, and label source. This keeps different predicates and future prompt changes separate even if an old name is reused.

## Query coverage

All 26 queries have complete ground truth for every predicate they use.

| Query | Predicates used | Complete ground truth | Label source |
|---|---|---|---|
| IMDB-1 | F1 | Yes | Qwen3 32B |
| IMDB-2 | DISCUSS_ASPECT | Yes | Qwen3 32B |
| IMDB-3 | F1, DISCUSS_ASPECT | Yes | Qwen3 32B |
| IMDB-4 | F1, F4, DISCUSS_ASPECT | Yes | Qwen3 32B |
| IMDB-5 | F1, F4, F5, DISCUSS_ASPECT | Yes | Qwen3 32B |
| IMDB-6 | F1, F4 | Yes | Qwen3 32B |
| IMDB-7 | F1, F4, F5 | Yes | Qwen3 32B |
| BIO-1 | F7 | Yes | Qwen3 32B |
| BIO-2 | REACTION | Yes | Qwen3 32B |
| BIO-3 | F7, REACTION | Yes | Qwen3 32B |
| BIO-4 | F7, F8, REACTION | Yes | Qwen3 32B |
| BIO-5 | F7, F8, F9, REACTION | Yes | Qwen3 32B |
| FEV-1 | F11 | Yes | Qwen3 32B |
| FEV-2 | SUPPORT | Yes | FEVER annotations and Qwen3 32B |
| FEV-3 | F11, SUPPORT | Yes | FEVER annotations and Qwen3 32B |
| FEV-4 | F11, F12, SUPPORT | Yes | FEVER annotations and Qwen3 32B |
| FEV-5 | F11, F13, SUPPORT | Yes | FEVER annotations and Qwen3 32B |
| FEV-6 | F11, F12, F13, SUPPORT | Yes | FEVER annotations and Qwen3 32B |
| LEP-1 | LEP1 | Yes | Qwen3 32B |
| LEP-2 | LEPJOIN | Yes | LePaRD passage IDs |
| LEP-3 | LEP1, LEPJOIN | Yes | Qwen3 32B and LePaRD passage IDs |
| LEP-4 | LEP1, LEP2, LEPJOIN | Yes | Qwen3 32B and LePaRD passage IDs |
| LEP-5 | LEP1, LEP2, LEP3, LEPJOIN | Yes | Qwen3 32B and LePaRD passage IDs |
| LEP-6 | LEP1 through LEP5, LEPJOIN | Yes | Qwen3 32B and LePaRD passage IDs |
| LEP-7 | LEP1, LEP2, LEPS1, LEPJOIN | Yes | Qwen3 32B and LePaRD passage IDs |
| LEP-8 | LEP1 through LEP5 | Yes | Qwen3 32B |

## Selectivity

Figure: plots/benchmark/20260825T082145Z-quailb-qwen32b-ground-truth-sf0.1-selectivity.png

Two results need a small human audit before these labels are treated as final ground truth.

- The BioDEX reaction join has 81 true pairs out of 122,800, or 0.066%. The old reactions were known to be wrong, so this may be correct, but the very low positive count should be checked.
- The LePaRD general-rule predicate has 200 true rows out of 200. It has no negative examples in this sample, so it does not help measure filtering quality as written.

These are model-created labels. The deterministic repeat shows that the run is repeatable. It does not prove that each answer is correct. A small review of positives and negatives for each predicate would measure that.

## Storage

The raw data is stored by corpus, predicate, and label set.

- Collection: `/results/ground_truth/quailb/schema_v1/collections/gt_42674891c824e01c6d966eb48c9cf8c7`
- Corpus: `/results/ground_truth/quailb/schema_v1/corpora/c_df45ef585738f42e4a7a731306f1b9fc`
- Label sets: `/results/ground_truth/quailb/schema_v1/label_sets/<workload>/<predicate>/<label_set_id>/parts`

Each Parquet row stores the answer, label source, stable judgment ID, stable example ID, predicate version, corpus row IDs, and content hashes. This supports per-document filters and per-pair joins without mixing results from predicates that share an old name.

Each label set now also has one `labels.parquet` file beside its `parts` directory. The compact files contain the same 245,557 rows. They reduce a complete reader load from more than 90 seconds to 6.93 seconds.

## Benchmark evaluation

The QUAIL-B runner now reads this collection by default. It checks that the local benchmark corpus ID matches the collection before a GPU query starts.

For every query, the summary includes:

- Query runtime and boot time.
- H100 inference cost and H100 cost including boot.
- Tokens processed and H100 cost per token.
- Input documents per second.
- Accuracy across the model calls that the query evaluated.
- Precision, recall, F1, and exact match for the final returned rows.

The runner saves raw returned rows, answer rows, and a timestamped aggregate JSON summary under `/results/benchmarks/quailb/runs/<run_id>/` on the `quail-results` volume. It also writes the aggregate JSON summary and Markdown report under `results/benchmark/`. It writes the PNG plot under `reports/plots/benchmark/`.
