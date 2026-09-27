# BIO-4 across Quail's history

- BIO-4 at sf=0.5 took 1,027.8 seconds end to end (958.1 query plus
  69.7 startup) on Quail's first engine, Aug 18, and 709.8 seconds (678.5
  plus 31.3) on today's code: 1.45 times less. Every configuration
  answers the same prompts.
- The query-time gains came from two changes: the fused Triton kernels
  (#3, 15.8% less) and streaming filter survivors into the join with
  their KV pinned (#92, 10.3% less). Pinned copies (#12) saved 4.4%.
  Every other change moved BIO-4's query time by less than the 1.7%
  run-to-run spread.
- Startup fell from 69.7 to 31.3 seconds in three steps: kernels
  compiled once (#47), vLLM's cache kept on a volume (#78), and the
  Gigatoken tokenizer, which loads in 1.5 seconds instead of 10 to 17
  (#103).
- The vLLM baselines took 4,496 to 5,764 seconds end to end. Quail's
  first engine was already 4.4 times faster than pipelined vLLM on the
  same date; today Quail takes 7.4 times less query time than pipelined
  vLLM today (678.5 against 5,039.0 seconds), at $0.744 per query
  against $5.528.
- Two side results change answers, not speed: how the prompt is
  tokenized (Quail joins separately tokenized pieces; vLLM today
  tokenizes whole prompts and agrees more with the reference, 86.9%
  against 79.5%), and where the join question sits (before #52 the
  question followed the term, and agreement was 94.5%).

Figures (vector PDFs):

- [Query and startup seconds on Quail's code at each merge date](plots/bio4_history_quail.pdf)
- [Startup + query seconds for the vLLM baselines and Quail by date](plots/bio4_history_versus_vllm.pdf)
- [Fresh input tokens, recomputed KV tokens, and answer agreement by date](plots/bio4_history_tokens.pdf)

## What was run

- One query, BIO-4 at sf=0.5: a filter on 2,500 adverse event reports,
  filters on two aliases of the 2,934-term table (neurological and
  cardiovascular reactions), and two joins of the surviving reports
  against each term list. Qwen3 4B FP8, bf16 KV, one H100 per
  configuration.
- Every configuration ran the current code on
  `claude/compassionate-maxwell-ux26bx` with the features merged after
  its date switched off. Each switch in `quail/ablation.py` recreates
  what the engine did before one pull request. Checking out old
  commits instead was not possible: BIO-4 was added on Sep 20, `main`
  was squashed on Sep 7, and the latency definition changed on Sep 22.
- Every configuration answers the same prompts, with the join question
  written before the term, so answers match across configurations up to
  kernel rounding. #52 (Aug 26) moved the question to that position; the
  saving it brought comes from the prompt, and the engine already
  computed text in that position once per report, so it is not a step.
- The vLLM configurations are the baselines as they stood on their
  dates. Before Sep 22 they received prompt token ids from Quail; since
  then they tokenize the prompt text with Gigatoken.
- Each configuration ran in its own fresh container, all at once.
  Cell: `experiments/bio4_history.py`. Run directories on
  `quail-results`: the configurations dated before Aug 26 in
  `/results/ablations/bio4-history-sf0.5-20260926T223002Z` (commit
  `6537700`), with the three vLLM baselines dated before Sep 22; the
  rest in
  `/results/ablations/bio4-history-sf0.5-20260926T213610Z` (commit
  `e38bfeb`). The logs with every Modal function call id are
  `results/benchmark/20260926T222959Z-bio4-history-sf0.5-early.log` and
  `results/benchmark/20260926T213606Z-bio4-history-sf0.5.log`.
- Reference labels: collection `gt_68f9ce9439bd7615de92b33d576dff9e`
  (Qwen3 32B FP8 and the benchmark's annotations).

### Timing

- Query time is QUAIL-B's submission-to-answer time: planning,
  tokenization, execution, and answer preparation. Model startup is
  excluded, as in every QUAIL-B report.
- Startup time is the seconds from starting a fresh process until the
  engine is ready: imports and tokenizer load until the session is
  ready, then the engine boot (model load, KV arena, kernel compile or
  touch pass for Quail; `LLM()` construction for vLLM). It excludes
  Modal's container provisioning and QUAIL-B's data and label loading.
- Startup varies from machine to machine: the model load took 9 to 33
  seconds for the same configuration on different containers. Each
  configuration was therefore booted three times, once in its query run
  and twice in startup-only containers, and the figures use the median.
- Throughput is requested input tokens per second, as QUAIL-B reports
  it. $/query is query seconds times $3.9492 per H100 hour.

### The switches

| Switch | Merged | PR | Affects | With the feature | Without it |
|---|---|---|---|---|---|
| `triton_kernels` | Aug 16 | #3 | query | fused Triton kernels for norm and quantize, q and k norm with rotary, and SiLU with quantize | vLLM's unfused kernels for those three steps |
| `pinned_staging` | Aug 19 | #12 | query | chunk inputs copied to the GPU through pinned memory without blocking | pageable, blocking copies |
| `attention_paths` | Aug 23 | #35 | query | filters use the unified attention path and joins use merge_quant | filters use the join's two-call attention path |
| `skip_arena_writes` | Aug 23 | #36 | query | a one-stage filter whose KV nothing reads skips the KV arena | every filter writes its KV into the arena |
| `join_search` | Aug 24 | #42 | query | one search over join order and anchor choice | joins in written order; each join anchors on the input with the most tokens |
| `compile_once` | Aug 25 | #47 | startup | the kernel compile pass runs once; later boots only touch kernels | every boot runs the compile pass |
| `filter_kv_reuse` | Aug 29 | #70 | query | filter survivors keep their KV for the joins; filters ordered by cost | joins recompute every anchor prefix; filters in written order |
| `scan_ring` | Aug 30 | #72 | query | retained KV is capped to leave two chunks of pages for admission | retained KV may fill the arena |
| `boot_cache` | Sep 1 | #78 | startup | vLLM's cache on the kernel volume and pinned model revisions | a fresh vLLM cache per boot and the model's main branch |
| `shared_retention` | Sep 7 | #79 | query | one KV retention pool shared by every planned anchor input | only the first join anchor's filter survivors are retained |
| `join_continuous_batching` | Sep 7 | #81 | query | join anchors admitted continuously, stages mixed in one chunk | each arena-sized group of anchors runs one stage at a time and waits for every answer |
| `projection_pushdown` | Sep 7 | #82 | query | scans keep only the columns the query reads | scans keep every source column |
| `plan_on_estimates` | Sep 8 | #85 | query | planning uses estimated token counts while tokenization runs | planning waits for exact token counts |
| `filter_join_streaming` | Sep 13 | #92 | query | filter survivors stream into the join with their KV pinned | the join starts after its filter finishes |
| `gigatoken` | Sep 16 | #103 | both | Gigatoken tokenizes documents and prompts | bpe-qwen tokenizes documents and prompts |
| `vllm_gigatoken` | Sep 22 | #167 | query | vLLM receives prompt text and tokenizes it with Gigatoken | vLLM receives prompt token ids built from each document's tokens |

The two Quail configurations dated Aug 18 are the first engine commit
(`44eaf81`, no pull request) with vLLM's kernels and with the fused
kernels from #3.

### What the history could not recreate

- BIO-4 has two joins. The engine could first run two joins on Aug 24
  (#42). The configurations dated earlier run them the way the Aug 24
  code does: one join node, one anchor, its KV shared by both stages.
- fp8 KV (removed in #17, Aug 20) is not a switch; every configuration
  uses bf16 KV.
- The memory-mapped token store (in #79) and the Sep 18 engine
  refactors have no switch.
- The Aug 2 to Aug 12 prototype, which added KV rewind inside a patched
  vLLM, is not runnable; KV rewind and token-based admission are part
  of the Aug 18 base engine.
- Changes that do not touch BIO-4 are not steps: LIMIT early stop
  (#29), filter-chain queries, equality joins, AI.SCORE, and the
  DiffusionGemma work.

## Prediction

Stated in the run logs before launch, from the sf=0.1 check:

- 2,500 reports and 2,934 terms give about 3.2 million evaluated pairs,
  13 times sf=0.1. Measured: 2.9 million.
- Quail today about 700 seconds of query time. Measured: 678.5 and
  667.0 seconds.
- Configurations before filter-join streaming about 900 seconds, the
  Aug 18 engine with vLLM's kernels up to a quarter more. Measured: 744
  to 807 seconds, and 958.1 seconds for the Aug 18 engine.
- Pipelined vLLM today 5,000 to 10,000 seconds.
- Startup does not depend on the scale factor.

## What each step did

Changes smaller than 1.7% are within noise: Quail today ran twice, in
two containers, at 678.5 and 667.0 seconds.

- **Quail engine with vLLM's kernels, Aug 18.** 958.1 seconds of query
  time and 69.7 seconds of startup.
- **Fused kernels, #3.** Query time fell 15.8%, to 806.9 seconds. The
  new kernels round differently, so a few answers changed: answer
  agreement moved from 80.5% to 79.4%, and 2.96 million pairs were
  evaluated instead of 2.91 million.
- **Pinned copies, #12.** 4.4% less query time, same answers.
- **Attention paths (#35), skip KV writes (#36), join search (#42).**
  Between 747 and 760 seconds, within noise of each other. The join
  search runs the cardiovascular join first; both orders evaluate about
  2.9 million pairs, because every surviving report matches at least
  one cardiovascular term.
- **Compile once, #47.** Startup fell from 71.6 to 50.7 seconds (median
  of three). The compile pass took about 25 seconds; the touch pass that
  replaced it takes about 5.
- **Filter KV reuse, #70.** No change in query time. The reports average
  about 4,100 tokens, so the retention pool holds only a few dozen of
  the 1,894 reports that pass the filter: 26 joins found their report's
  KV, 1,868 recomputed it.
- **Scan ring, #72.** No change. The ring fixed filters that stalled
  on tiny chunks of short documents (IMDB-3); a chunk of 4,100-token
  reports stays large.
- **Boot cache, #78.** Startup fell from 48.9 to 37.6 seconds.
- **Shared retention (#79), join batching (#81), projection pushdown
  (#82), planning on estimates (#85).** Each within noise on BIO-4.
- **Filter-join streaming, #92.** Query time fell 10.3%, from 743.7 to
  667.4 seconds. Each passing report now keeps its KV pinned until its
  pairs are answered: all 1,894 joins found their report's KV, and
  recomputed KV tokens fell from 13.6 million to 6.1 million.
- **Gigatoken, #103.** Session startup fell from 10 to 17 seconds to
  1.5 seconds, and document tokenization inside the query from about 4
  seconds to 1.3 seconds. Query time moved within noise.

Over the whole history, startup fell from 69.7 to 31.3 seconds and
query time from 958.1 to 678.5 seconds.

## The vLLM baselines

- vLLM at default settings took 5,579.9 seconds of query time and 184.3
  seconds of startup. Tuned settings (#2) took 5,330.1, and pipelining
  the filters took 4,391.0. All three received Quail's prompt token ids
  and returned the same answers.
- Pipelined vLLM today (#167) receives prompt text and tokenizes it
  with Gigatoken: 5,039.0 seconds, 648 more than with token ids, since
  it now tokenizes about 12 billion tokens of prompt text. Its startup
  fell to 49.2 seconds because its cache is kept on a volume.
- vLLM computes 70 to 71 million fresh tokens, of which 22 to 23 million
  recompute KV it already had, against 53.5 million and 6.2 million for
  Quail today.

## Side result: tokenizing prompt pieces changes the answers

- Quail tokenizes each document, question, and label separately and
  joins the token ids. vLLM before #167 received those same token ids.
  vLLM since #167 receives the prompt text and tokenizes each whole
  prompt with Gigatoken. The text is identical; the token ids differ
  where a piece boundary would merge into one token in the whole text.
- At sf=0.5 the answers follow the tokenization, not the engine:

| | Quail today | Pipelined vLLM, Aug 18 (token ids) | Pipelined vLLM today (whole text) |
|---|---:|---:|---:|
| Answer agreement | 79.5% | 78.2% | 86.9% |
| Rows returned (reference 6.46 million) | 86.5 million | 99.8 million | 44.5 million |

- Whole-prompt tokenization also costs time: pipelined vLLM took
  4,391 seconds with token ids and 5,039 seconds tokenizing about 12
  billion tokens of prompt text itself.
- To settle how much of the accuracy gap this explains, run Quail with
  the join pieces tokenized as whole prompts on a sample of pairs.

## Side result: where the question sits changes the answers

This is not part of the comparison above. An earlier run of this
history moved the join question after the term for the configurations
dated before Aug 26, as the prompt read before #52. At sf=0.5, on the
Aug 25 engine (`/results/ablations/bio4-history-sf0.5-20260926T213610Z`,
configurations `compile_once` and `shared_join_prompts`):

| | Question after term | Question before term |
|---|---:|---:|
| Join pairs answered TRUE (reference 4.4%) | 4.8% | 23% |
| Join answer precision | 38% | 14.5% |
| Join answer recall | 42% | 77% |
| Answer agreement, all predicates | 94.5% | 79.6% |
| Rows returned (reference 6.46 million) | 6.2 million | 86.3 million |
| Fresh tokens | 150.0 million | 61.1 million |
| Query seconds | 1,823.5 | 764.1 |

- The filters' answers are the same in both layouts; only the join
  answers move.
- The reference labels were made by Qwen3 32B with the question-before-term
  layout. The 4B model answers TRUE five times as often as the
  reference with that layout. At sf=0.1, pipelined vLLM given the
  question-after-term token ids agreed 92.6%, close to Quail's 93.1%
  with the same prompts, so this is a property of the prompt, not of
  the engine.
- To settle which layout to keep, read a sample of the pairs the two
  layouts answer differently, and repeat the swap on BIO-2 and BIO-3.

## Tables

| Configuration | Merged | Query s | Startup s (median, range) | Input tokens/s | $/query |
|---|---|---:|---:|---:|---:|
| vLLM defaults | Aug 1 | 5,579.9 | 184.3 (171.1 to 297.9, n=3) | 2,186,057 | 6.1211 |
| vLLM tuned | Aug 13 #2 | 5,330.1 | 104.7 (82.9 to 138.0, n=3) | 2,288,518 | 5.8471 |
| vLLM pipelined | Aug 18 | 4,391.0 | 105.1 (86.6 to 105.4, n=3) | 2,777,959 | 4.8169 |
| vLLM + Gigatoken | Sep 22 #167 | 5,039.0 | 49.2 (44.1 to 78.2, n=3) | 2,384,270 | 5.5278 |
| Quail engine | Aug 18 | 958.1 | 69.7 (61.2 to 74.9, n=3) | 12,408,716 | 1.0510 |
| fused kernels | Aug 18 #3 | 806.9 | 63.4 (61.7 to 73.7, n=3) | 15,022,765 | 0.8852 |
| pinned copies | Aug 19 #12 | 771.7 | 70.6 (62.1 to 80.8, n=3) | 15,709,007 | 0.8465 |
| attention paths | Aug 23 #35 | 749.8 | 70.5 (59.9 to 73.0, n=3) | 15,770,636 | 0.8225 |
| skip KV writes | Aug 23 #36 | 759.5 | 71.1 (69.0 to 77.8, n=3) | 15,569,245 | 0.8331 |
| join search | Aug 24 #42 | 747.4 | 71.6 (62.7 to 78.2, n=3) | 15,811,710 | 0.8199 |
| compile once | Aug 25 #47 | 764.4 | 50.7 (42.0 to 63.9, n=3) | 15,459,188 | 0.8386 |
| filter KV reuse | Aug 29 #70 | 763.4 | 55.1 (40.0 to 64.1, n=3) | 15,479,908 | 0.8375 |
| scan ring | Aug 30 #72 | 764.0 | 48.9 (40.6 to 64.6, n=3) | 15,468,237 | 0.8381 |
| boot cache | Sep 1 #78 | 747.9 | 37.6 (24.8 to 54.4, n=3) | 15,800,363 | 0.8205 |
| shared retention | Sep 7 #79 | 747.0 | 37.5 (29.1 to 325.9, n=3) | 15,820,077 | 0.8194 |
| join batching | Sep 7 #81 | 751.2 | 37.6 (30.3 to 45.9, n=3) | 15,730,956 | 0.8241 |
| projection pushdown | Sep 7 #82 | 759.3 | 37.0 (27.9 to 41.4, n=3) | 15,563,953 | 0.8329 |
| plan on estimates | Sep 8 #85 | 743.7 | 33.4 (28.6 to 48.0, n=3) | 15,890,436 | 0.8158 |
| filter-join streaming | Sep 13 #92 | 667.4 | 35.7 (33.8 to 50.5, n=3) | 17,705,415 | 0.7322 |
| Gigatoken | Sep 16 #103 | 678.5 | 31.3 (29.5 to 33.3, n=3) | 17,451,851 | 0.7443 |

| Configuration | Fresh tokens | Recomputed KV tokens | KV regret % | Evaluated pairs |
|---|---:|---:|---:|---:|
| vLLM defaults | 70,972,089 | 22,754,343 | 32.1 | 2,976,294 |
| vLLM tuned | 70,972,089 | 22,754,343 | 32.1 | 2,976,294 |
| vLLM pipelined | 70,972,089 | 22,754,343 | 32.1 | 2,976,294 |
| vLLM + Gigatoken | 70,014,999 | 22,094,124 | 31.6 | 2,953,628 |
| Quail engine | 61,197,654 | 13,820,021 | 22.6 | 2,907,980 |
| fused kernels | 62,050,071 | 14,044,155 | 22.6 | 2,962,456 |
| pinned copies | 62,050,071 | 14,044,155 | 22.6 | 2,962,456 |
| attention paths | 61,090,091 | 13,765,816 | 22.5 | 2,898,385 |
| skip KV writes | 61,090,091 | 13,765,816 | 22.5 | 2,898,385 |
| join search | 61,083,840 | 13,765,816 | 22.5 | 2,897,888 |
| compile once | 61,083,840 | 13,765,816 | 22.5 | 2,897,888 |
| filter KV reuse | 60,727,937 | 13,409,913 | 22.1 | 2,897,888 |
| scan ring | 60,945,119 | 13,627,095 | 22.4 | 2,897,888 |
| boot cache | 60,945,119 | 13,627,095 | 22.4 | 2,897,888 |
| shared retention | 60,945,119 | 13,627,095 | 22.4 | 2,897,888 |
| join batching | 60,945,119 | 13,627,095 | 22.4 | 2,897,888 |
| projection pushdown | 60,945,119 | 13,627,095 | 22.4 | 2,897,888 |
| plan on estimates | 60,945,119 | 13,627,095 | 22.4 | 2,897,888 |
| filter-join streaming | 53,466,829 | 6,148,805 | 11.5 | 2,897,888 |
| Gigatoken | 53,512,247 | 6,155,353 | 11.5 | 2,900,954 |

| Configuration | Answer agreement % | Returned rows | Output precision % | Output recall % |
|---|---:|---:|---:|---:|
| vLLM defaults | 78.2 | 99,846,219 | 1.62 | 25.09 |
| vLLM tuned | 78.2 | 99,846,219 | 1.62 | 25.09 |
| vLLM pipelined | 78.2 | 99,846,215 | 1.62 | 25.09 |
| vLLM + Gigatoken | 86.9 | 44,534,403 | 2.59 | 17.88 |
| Quail engine | 80.5 | 81,765,858 | 1.87 | 23.63 |
| fused kernels | 79.4 | 90,648,935 | 1.72 | 24.08 |
| pinned copies | 79.4 | 90,648,935 | 1.72 | 24.08 |
| attention paths | 79.6 | 86,300,301 | 1.81 | 24.21 |
| skip KV writes | 79.6 | 86,300,301 | 1.81 | 24.21 |
| join search | 79.6 | 86,300,301 | 1.81 | 24.21 |
| compile once | 79.6 | 86,300,301 | 1.81 | 24.21 |
| filter KV reuse | 79.6 | 86,300,301 | 1.81 | 24.21 |
| scan ring | 79.6 | 86,300,301 | 1.81 | 24.21 |
| boot cache | 79.6 | 86,300,301 | 1.81 | 24.21 |
| shared retention | 79.6 | 86,300,301 | 1.81 | 24.21 |
| join batching | 79.6 | 86,300,301 | 1.81 | 24.21 |
| projection pushdown | 79.6 | 86,300,301 | 1.81 | 24.21 |
| plan on estimates | 79.6 | 86,300,301 | 1.81 | 24.21 |
| filter-join streaming | 79.6 | 86,300,301 | 1.81 | 24.21 |
| Gigatoken | 79.5 | 86,516,625 | 1.81 | 24.22 |

## Reproduce

- Rerun every configuration: `uv run modal run --detach
  experiments/bio4_history.py::history --sf 0.5 --startup-samples 2`
  on `claude/compassionate-maxwell-ux26bx`.
- Figures and tables: pull `configurations/` and `startup/` from both
  run directories, the earlier one first, then run
  `uv run --with matplotlib python reports/make_bio4_history_plots.py
  "$W1" "$W2"`, and the same with `--table`. The script's docstring
  has the `modal volume get` commands.

  - `$W1`: `/results/ablations/bio4-history-sf0.5-20260926T213610Z`
  - `$W2`: `/results/ablations/bio4-history-sf0.5-20260926T223002Z`
