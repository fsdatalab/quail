# KV retention that does not reduce filter admission

## Setup

The IMDB 3 discrepancy report found that retained survivor KV consumed nearly
the full arena. The planner treated 358,861 of the 362,250 arena tokens as
retained KV and left only 3,389 tokens for active filter work. The filter can
have two chunks in progress, which require up to 220,752 tokens.

After six chunks, every new admission had to evict retained KV. The filter ran
2,922 forward passes with a mean of 602 tokens per pass and took 50.28 seconds.
The same filter takes about 15 seconds when retention does not reduce its chunk
size.

The current implementation has three parts.

* The filter reserves pages for two chunk budgets before it starts. The
  retained pool can use only the pages outside this scan reserve.

* A passing survivor stays in its existing pages. The retained pool maximizes
  reusable prefix tokens under its page limit. When the pool is full, residents
  with the fewest prefix tokens per page are considered first. A candidate
  replaces complete resident documents only when it contains more prefix tokens
  than all victims combined. The policy does not estimate runtime.

* The planner reserves the same two chunk budgets. Because page rounding can
  favor documents of any length, the planner credits a fraction of filter
  survivors with the corpus length distribution instead of assuming that the
  longest survivors remain resident.

The old admission eviction path remains as a safety check. The scan reserve
should make the path unreachable during a normal filter.

The confirming query was IMDB 3 at scale factor 0.1 with Qwen3 4B fp8 on one
H100. `ablations/profile_quail.py` recorded an unprofiled pass for query time
and a profiled pass for the GPU trace. The current run used the prefix
`ringfix_tokens_head`.

## Prediction

The following prediction was recorded before the prefix token run.

* The filter would remain at 17 near full forward passes and take 14 to 17
  seconds because the scan reserve did not change.

* The retained pool would fill its 8,843 page limit and hold about 141,000
  prefix tokens.

* The token policy would retain more than the 171 documents retained by the
  saved seconds policy because it would no longer prefer long documents.

* The join would remain near 18 seconds. Total engine time would remain between
  32 and 35 seconds.

* KV regret would remain near 1.22 million tokens because the retained token
  mass would remain nearly unchanged.

BIO 2 was not rerun for the score change. Its plan retains no filter KV, so it
does not enter the changed pool. The earlier scan reserve run remains the
control for the shared filter path.

## Result

Every prediction held. The prefix token policy took 32.79 seconds, compared
with 32.68 seconds for the saved seconds policy. The 0.11 second difference is
0.3 percent of query time.

### IMDB 3

| Metric | Unbounded retention | Saved seconds score | Prefix token score |
|---|---:|---:|---:|
| Engine time, seconds | 68.93 | 32.68 | 32.79 |
| Filter time, seconds | 50.28 | 14.54 | 14.59 |
| Filter forward passes | 2,922 | 17 | 17 |
| Mean filter pass, tokens | 602 | 103,484 | 103,484 |
| Admission eviction calls | 2,625 | 0 | 0 |
| Join time, seconds | 18.53 | 18.05 | 18.15 |
| Retained at join, documents | 120 | 171 | 309 |
| Retained at join, pages | 8,631 | 8,843 | 8,843 |
| Join hit tokens | 137,397 | 140,458 | 140,212 |
| KV regret, tokens | 1,220,547 | 1,217,486 | 1,217,732 |
| Filter survivors | 4,380 | 4,380 | 4,380 |

Figure: plots/kv_ring_fix_timeline.png

Figure: plots/kv_ring_fix_walls.png

* The scan reserve remains the source of the speedup. The current filter took
  14.59 seconds in 17 passes, compared with 50.28 seconds in 2,922 passes before
  the reserve. The current filter was 3.4 times faster.

* The new score changed which documents remained resident without changing the
  retained token mass. The pool held 309 documents with a mean prefix length of
  454 tokens. The saved seconds policy held 171 documents with a mean prefix
  length of 821 tokens. Both used 8,843 pages and supplied about 140,000 hit
  tokens to the join.

* The current statistics recorded 285 pool replacement evictions covering
  4,211 pages and 64,470 prefix tokens. The replacements happened while filter
  answers were handled. They did not block admission or create another forward
  pass.

* The token score increased regret by 246 tokens compared with the saved
  seconds score. The difference is 0.02 percent of total regret. Join time
  changed from 18.05 to 18.15 seconds.

* IMDB 3 evaluates 52,560 document pairs. The current result is 1,603 document
  pairs per second. Query cost is $0.0360 at the H100 rate of $3.9492 per hour,
  compared with $0.0823 for unbounded retention and $0.0578 for stock vLLM's
  recorded 52.65 seconds. Query cost excludes model startup.

### BIO 2

The earlier scan reserve control took 128.22 seconds, compared with 130.35
seconds before the reserve. Both runs used 98 forward passes, and KV regret was
zero. BIO 2 retains no filter KV, so the prefix token selection code is not
called for this query.

## Data

The measured files are on the `quail-results` volume.

* Current prefix token run:
  `/results/ablations/ringfix_tokens_head_imdb3.json`

* Current traces:
  `/results/ablations/ringfix_tokens_head_traces/`

* Saved seconds comparison:
  `/results/ablations/ringfix_imdb3.json`

* BIO 2 control:
  `/results/ablations/ringfix_bio2.json`

* Unbounded retention comparison:
  `/results/ablations/discrepancy_imdb3.json`

The current Modal function call was `fc-01M1A2VX8Z1K1F07JP271R4DFQ`. The saved
seconds run was `fc-01M18J895155R1VS59VRPTGWEN`.

## Meaning

The scan reserve solves the admission problem independently of the retention
score. The saved seconds score and prefix token score produced the same filter
and join shape within normal run variation.

The retention objective is now exact and limited. Quail maximizes reusable
prefix tokens under the retained page limit. It does not claim to predict join
time. The report measures the resulting join time instead.

The QuailB headline table still needs a full benchmark rerun. The confirming
cell in this report covers IMDB 3, which is the query that previously exposed
the admission failure.

## Rebuild

Use a new output prefix so the recorded result remains unchanged.

    uv run modal run ablations/profile_quail.py::run --queries IMDB-3 --out-prefix rerun

Rebuild the figures with `reports/make_kv_ring_fix_plots.py`. Its docstring
contains every `modal volume get` command needed to pull the source files.
