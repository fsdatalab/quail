# Joins: plan for data, estimates, and block sizes

Status: plan only. Nothing here is implemented. Every wall-clock
number below is a prediction from the measured constants in
`quail/plan/cost.py`, to be checked against one confirming run
before any full run is bought.

The query: two document collections L and R, and an AI-if predicate —
for a pair (l, r), ask the model one yes/no question about both
documents. Same rules as the filters: Qwen3 4B fp8, one H100, answer
constrained to YES/NO, `max_tokens=1`, zero decode. A join is a
filter whose input is a pair.

Datasets come from the featurized-decomposition join paper
(Zeighami, Shankar, Parameswaran, arXiv 2512.05399, VLDB 2026;
"FDJ" below).

---

## 1. The one structural fact

A pair prompt is `[preamble p | outer doc | inner doc | question q]`.
Attention is causal, so:

- The outer document's KV depends only on the preamble. It can be
  computed once and reused by every pair that contains that document.
- The inner document's KV depends on the outer document in front of
  it. It can never be reused across pairs. Every pair recomputes its
  inner document, no matter what the schedule does.

So the minimum fresh tokens any schedule can reach is

    N_out * (p + T_out)  +  pairs * (T_in + q)

with T_out and T_in mean document tokens per side. The second term
dominates. All planning follows from this.

There is a second, smaller cost the filters never made large: each
pair's suffix must attend to its cached outer prefix. The calibration
priced that read at 82–141 ns per cached token for suffixes of 16–64
tokens. Join suffixes are 54–4,000 tokens, outside the measured
range — this is exactly calibration family C3, which we skipped on
purpose because filters could not identify it. Joins can, and need
it. Until C3 runs, read-cost estimates below carry a range: 0.49 ns
per (cached token x suffix token) at the low end (the a2 attention
price inside a fresh prefill) to 2.2 ns at the high end (the measured
141 ns / 64 at the widest calibrated suffix).

## 2. The decision procedure

**Which side is outer.** The outer side is the one with longer
documents. Inner tokens are paid once per pair; outer tokens once per
document plus a cheap cached read per pair. Reading a cached token
costs 1/160th of computing one, so swapping a long side into the
inner position multiplies the dominant term. This is the opposite of
disk block-nested-loop join, where the smaller relation goes outer.
Note the trap: the side that fits entirely in the KV pool is usually
the short side, and pinning it anyway is the wrong plan. BioDEX below:
all 3,718 reaction terms fit in the pool at once, and putting them
outer still costs 91 hours against 6.

**Outer block size W** (how many outer prefixes are resident at
once, each as one living chain). W is not a compute knob. Unlike disk
BNLJ, a bigger outer block saves nothing, because there is no
inner-scan cost to amortize — inner tokens are per-pair no matter
what. W only needs to keep the GPU fed:

    W_min = ceil(2 * step_budget / (T_in + q))     # saturation
    W_max = floor(0.8 * pool / (p + T_out + T_in + q + 1))  # admission

Prediction: wall time is flat in W across [W_min, W_max]. Past
saturation, extra resident chains buy nothing and charge host
block-table time (1.12 us per resident 16-token block per step). So
pick W near W_min with margin, not at the memory limit. This flatness
claim is the one thing worth a small sweep, because the claim is the
point.

**Inner block size.** In the engine design there is none: each chain
holds one inner suffix at a time, rewinds to the outer-document
boundary, appends the next. (Several suffixes at once over one prefix
is forking, which is out of scope on purpose.) The classic inner
block reappears only in the packed design (Section 5, design C),
where it is the number of inner documents that fill the rest of a
25,305-token chunk.

**Memory per relation.** Falls out of W rather than being chosen:
outer gets W * (p + T_out) tokens of the pool, inner gets
W * (T_in + q + 1). Per dataset:

| dataset | per-chain split outer/inner | W_min | W_max | pool used at chosen W |
|---|---|---|---|---|
| Products | 47% / 53% | 461 | 3,589 (all 973 fit) | 13% (whole outer side resident) |
| BioDEX | **95% / 5%** | 873 | 689 | 100% — memory-bound, see note |
| Categorize | 72% / 28% | 817 | 3,396 | 30% at W=1,024 |
| Police | 50% / 50% | 33 | 245 | 26% at W=64 |

BioDEX note: W_max < W_min — the admission budget caps in-flight
suffix tokens at 1.6 step budgets instead of 2. Fix by dropping the
step budget toward 16k (throughput is flat above ~4,096) or trimming
notes slightly. This is the one dataset where admission genuinely
binds, which is what makes it interesting.

## 3. Getting the data

FDJ Table 1, with pair counts added and token lengths **assumed** —
measuring real token lengths is step one and moves every estimate:

| dataset | L x R | ordered pairs | T_out (assumed) | T_in (assumed) | true pairs n+ |
|---|---|---|---|---|---|
| Products | 973 x 956 | 0.93M | 60 | 60 | 616 |
| BioDEX | 8,103 notes x 3,718 terms | 30.1M | 1,000 | 8 | 20,000 |
| Citations | 16,161 x 16,161 self | 261M (130.6M unordered) | 200 | 200 | 20,000 |
| Movies | 4,043 x 4,983 | 20.1M | 4,000 | 4,000 | 10,475 |
| Police Records | 2,093 x 2,093 self | 4.38M (2.19M unordered) | 1,500 | 1,500 | 56,531 |
| Categorize | 19,101 products x 3,144 categories | 60.1M | 120 | 12 | 20,000 |

Sources and steps:

- **Products** — classic entity-matching benchmark (Köpcke 2010;
  DeepMatcher, Mudgal 2018). Public. The paper's 973 x 956 subset is
  the records appearing in the train/test pairs; get the exact split
  from the authors or rebuild from the labeled pairs.
- **BioDEX** — HuggingFace, D'Oosterlinck 2023; LOTUS also ships a
  prepared join version. Paper sampled 20,000 ground-truth pairs and
  kept only those records; do the same.
- **Citations** — LePaRD (Mahari 2024), judicial citations to
  precedent. Public. Same 20k-pair sampling. Check whether the
  predicate is directional ("cites") or symmetric ("cite each
  other") before halving the pair count.
- **Categorize** — Amazon product data (McAuley 2013) + the Extreme
  Classification repository (Varma). Public; same sampling.
- **Movies** — the authors' own Wikipedia crawl (movie pages x actor
  pages). Ask for the artifact; re-crawling is the fallback.
- **Police Records** — Police Records Access Project, private. Ask;
  do not plan on it. Its *shape* (symmetric self-join, long
  documents) is reproducible with a planted stand-in, below.
- One email to Sepanta covers: exact subsets, join prompts
  (their Appendix I), and Movies/Police availability.

**Planted stand-in, zero acquisition.** Reuse the 10k IMDB reviews:
plant a `[GROUP] G=nnnn` line in each document, predicate "do these
two reviews name the same group?", selectivity set by the group-size
distribution. Same philosophy as the flag filters: the answer is in
the text, so runs measure execution, not reasoning (and inherit the
same ~27% flag-misread caveat — no accuracy claims from this corpus).

## 4. Predicted runtimes

Constants: fresh token 10.4 us (96,180 tok/s sustained); requests
50.3 us each; steps 2.94 ms at the 25,305-token budget; pool 946,800
tokens; p = 40, q = 50 assumed for join prompts. Read term carries
the C3 range from Section 1. All numbers move with measured token
lengths; treat magnitudes and ratios as the content.

The four plans priced:

- **A1, stock vLLM, one request per pair, arbitrary order** — what a
  LOTUS-style client does today. No reuse: every pair prefills
  p + T_out + T_in + q. Where the outer side is several times the
  pool, add the measured thrash risk: on the filter workload at 3.4x
  pool pressure, default admission re-prefilled 2.40x the corpus and
  ran 1.87x slower; treat that as the multiplier's scale.
- **A2, stock vLLM, pairs grouped by outer document, admission
  matched to the pool** — the fair strong baseline the house rules
  require. Gets prefix reuse from the automatic prefix cache.
  Predicted at design B plus 2–5% (16-token boundary blocks, one
  request per pair instead of one chain per outer document).
- **B, chain mode / KV rewind** — one living chain per outer
  document; rewind to the outer-document boundary, append the next
  [inner doc + question]. This is the existing rewind arithmetic
  (`chainlogic.rewind_target`) with a fatter continuation; W from
  Section 2.
- **C, packed forward pass** — the giant-forward-pass path: chunk =
  [one outer prefix | as many inner suffixes as fit], shared-prefix
  attention inside the chunk, our three kernels, no engine, no KV
  pool. Inner block k = floor((25,305 − p − T_out)/(T_in + q)); each
  outer document is re-prefilled once per chunk, i.e. ceil(N_in/k)
  times. Rate assumed 121,045 tok/s as measured on the packed filter;
  the shared-prefix attention kernel is new work and could move this.

| dataset (pairs priced) | A1 stock, arbitrary order | A2 stock, ordered | B chain mode | C packed |
|---|---|---|---|---|
| Products (0.93M) | 34 min (no thrash: fits pool; likely ≈ B in practice) | ~20 min | 19 min | 14 min |
| BioDEX (30.1M) | 96 h, thrash risk to ~180 h | ~6–7 h | 5.8–6.7 h | 4.2 h |
| Categorize (60.1M) | 38 h, thrash risk to ~72 h | ~12 h | 11.8–12.1 h | 8.6 h |
| Police full (4.38M) | 39 h, thrash risk to ~73 h | ~22–27 h | 21–26 h | 16.6 h |
| Police unordered (2.19M) | 20 h / ~37 h | ~11–13 h | 10.7–13.2 h | 8.3 h |
| Citations 4k x 4k unordered (8.0M) | 11.3 h | ~6 h | 6.0–6.3 h | 4.7 h |
| Citations full unordered (130.6M) | 187 h | — | 98–112 h | ~73 h |
| Movies full (20.1M) | 470 h | — | ~240 h | — |

Orientation is worth more than any mechanism: BioDEX with terms
outer instead of notes costs 91 h in design B against 5.8–6.7 —
14–16x from choosing which side is which. Categorize flipped: 29.5 h
against ~12. The wide B ranges on long-outer datasets are the C3
uncertainty: at T_out beyond ~2k tokens, the cached read of the
outer prefix stops being negligible next to the fresh inner (the two
meet when T_out is roughly 4,700–21,000 tokens, the span of the C3
range), which is also exactly why C can beat B — its in-chunk
attention to the prefix runs at the 0.49 ns fresh-prefill price, not
the paged-cache read price.

What each mechanism is worth here, honestly: orientation up to 16x;
admission sized to the pool up to ~2x (it prevents the thrash
multiplier); rewind over ordered stock the same 2–5% as filters; the
packed pass another ~1.3–1.4x over B. The planner is the star; the
engine work is the same two small, real terms it was for filters.

## 5. What runs, in order

Each phase states its prediction first; a full run is bought only
after the cheap check lands within ~10%.

- **Phase 0a — C3 calibration.** Add family C3 to
  `quail/plan/calib.py` / `modal_calibrate.py`: suffix c in {64,
  256, 1,024, 4,096} x cached h in {512, 2,048, 8,192}, matched pair
  counts, same validity gates. Refit. This collapses the read-price
  range to a number, and unlocks the pair-vs-byte split Prop. 4 says
  filter cells cannot identify. Cheapest highest-value item on the
  list.
- **Phase 0b — confirming cell, planted join.** 1,000 x 1,000
  planted IMDB self-join, design B at W=256. Predicted 1.1–1.2 h
  (fresh 370M tokens at 10.4 us + read term 65–293 s + 50 s of
  request overhead). One container, teed to a file, A2 alongside at
  matched admission. Pass gate: measured wall within 10% of the
  estimator once C3 constants are in.
- **Phase 1 — Products.** First real data end to end; every design
  fits under 35 min, so it is the harness shakeout plus an accuracy
  read against the 616 gold pairs. No performance claim expected:
  the corpus is 0.1x the pool, so even arbitrary-order stock should
  cache well. Say so before running it.
- **Phase 2 — BioDEX, the headline.** Run B full (~6 h predicted).
  Do not run A1 full at 96 h; run A1 on a 5% pair sample, verify its
  tokens/s, extrapolate, and report it as an extrapolation. This is
  the asymmetric case: 95/5 memory split, admission binding,
  orientation worth 15x.
- **Phase 3 — symmetric long self-join.** Police if the authors
  share it; otherwise the planted stand-in scaled to its shape
  (2,093 x 2,093, ~1,500-token documents). This is the
  memory-pressure case (outer side 3.4x pool — same ratio as the
  filter corpus) and the W-flatness sweep lives here: W in {33, 64,
  128, 245}, predicted flat within a few percent.
- **Deferred.** Movies and full Citations — hundreds of GPU-hours as
  full cross products; only worth it subsampled, or as candidate
  sets (below).

## 6. Relation to FDJ, and what we do not do yet

FDJ attacks the pair *count*: per-side feature extraction, cheap
matching, LLM calls only on candidates (their cost ratios put the
verified set at a few percent of L x R). We attack the cost per
evaluated pair. They compose: candidate-list mode is the same
executor fed a pair list instead of the full cross product, with the
same orientation rule, and grouping candidates by outer document
recovers most prefix reuse at high candidate density. Not in scope
now: implementing FDJ's featurization, batching several inner
documents into one suffix (changes what the model is asked; accuracy
question first), forking, speculation, the 32B model.

## 7. Risks, stated

- Token lengths are assumed until tokenized; BioDEX notes and Movies
  pages have heavy tails, so truncation policy changes both cost and
  accuracy. Measure first.
- The C3 read price is the widest open uncertainty (Section 1); it
  moves long-outer estimates by up to ~5 h on Police and decides how
  close A2/B can get to C.
- The 121,045 tok/s packed rate was measured on the filter pass; the
  join needs a shared-prefix attention path that does not exist yet.
  Treat C's column as a design target until a chunk-level probe runs.
- At W near the memory cap, host block-table time reaches ~20% of a
  step (53 ms against 263 ms at BioDEX's W=689). Production overlaps
  host with GPU, so the prediction is that it hides; if it does not,
  that argues for smaller W and the packed design.
- The thrash multiplier for A1 is borrowed from the filter
  measurement at 3.4x pool pressure; BioDEX sits at 8.9x, where it
  could be worse. The A1 sample run in Phase 2 measures it.
- Police data may never arrive; the stand-in reproduces shape, not
  content, and carries the planted corpus's no-accuracy-claims
  caveat.
