# Results from the solved schedules

## What was solved and how to read the numbers

The paper in this repo (paper.md) studies a database query that runs a chain
of yes or no language model filters over documents on one graphics card. A
document that fails a filter skips the rest, so the work depends on how many
documents pass each stage. The paper compares three scheduling policies.
Task-first reprocesses every surviving document at each filter stage under a
shared prompt. Pipeline reads each document once, keeps the model's stored
memory of it (the KV cache) on the card, and runs each filter as a prompt of
about 50 tokens on top, waiting for each answer. Speculation runs future
filter prompts without waiting, wasting them when an early filter fails.

Every number below comes from the paper's ideal cost model, not from
hardware. The ideal model charges each batch the larger of its compute time
and its weight loading time, plus the larger of its attention compute time
and its memory traffic time, using the card's advertised peak speeds. The
conventions from notes/PLAN.md (C1 to C12) apply, including the corrected
attention width, a 2 gigabyte memory reserve, 1 byte per stored element, and
50 token prompts. The workload is 10,000 real IMDb reviews with 2,966,000
document tokens in total, tokenized for the Qwen3 models. The pass rate of a
filter is the fraction of documents that pass it. A lower bound here means a
finish time that no schedule can beat given the card's speeds, computed per
policy from the smallest possible work totals.

Raw data lives in results/n10k_two_stage.csv, per batch schedule records in
results/manifests/, and the small instance study in
experiments/run_smallN.py. An independent checker replayed every schedule
cited here and confirmed its cost and its legality.

The results split into three regimes, and each claim below names its
regime.

## At 10,000 documents, token counts decide everything

For every combination of two models (Qwen3-4B, Qwen3-32B), two cards (H100,
L40S), and first filter pass rates from 0.1 to 0.9, the schedules we built
finish within 0.07 percent of the lower bound, and most land exactly on it.
Landing on the bound proves no schedule can do better under the ideal model.
Finish times in seconds, averaged over two random outcome tables, with the
best policy per column in bold:

| model and card | policy | pass rate 0.10 | 0.50 | 0.90 |
|---|---|---|---|---|
| 4B on H100 | task-first | **12.12** | 16.56 | 20.98 |
| | pipeline | 13.05 | **13.80** | **14.54** |
| | full speculation | 14.73 | 14.73 | 14.73 |
| 32B on L40S | task-first | **280.8** | 383.5 | 485.1 |
| | pipeline | 301.8 | **319.3** | **336.3** |
| | full speculation | 340.5 | 340.5 | 340.5 |

The near zero gaps do not mean the schedule builders are clever. They mean
the problem is easy at that scale. Each batch holds hundreds of thousands of
tokens, so the card's compute speed is the only binding limit, and any
sensible packing reaches the bound. The choice between policies then reduces
to counting tokens, with the following consequences.

- Task-first beats pipeline only when the first filter's pass rate is below
  about 0.20, on every model and card. The exact crossing point is the
  number of documents times the prompt length, divided by the total document
  tokens minus that same product, which is 500,000 over 2,466,000, or 0.203.
  Below it, rereading the few survivors costs less than giving every
  document its own prompt tokens.
- Full speculation never wins at this scale. Its extra prompts on documents
  that fail early are never paid back, because waiting for answers costs
  nothing when there are always other documents to work on. Even on the
  tightest pair, the 32B model on the L40S, where only 3.3 percent of the
  corpus fits on the card, the pipeline never has to recompute anything,
  because it keeps each document's stored state for only one batch.
- Attention is at most 2.5 percent of the finish time at these document
  lengths, and weight loading is at most 1.2 seconds even when a run needs
  70 batches. Runs that scale document lengths up would change both, since
  attention grows with the square of length.

## At small document counts, the exact optimizer finds better schedules

Small queries are the regime where scheduling is a live problem, because
batches are small, loading the model weights costs as much as computing,
and waiting for a filter answer can waste a batch. The exact optimizer runs
here. On real Qwen3-4B and H100 numbers with real review lengths, whole
document chunks, and a 0.5 pass rate, finish times in milliseconds:

| documents | task-first, exact | pipeline, exact | full speculation, exact | pipeline, builder |
|---|---|---|---|---|
| 2 | 2.44 | 2.17 | **2.09** | 2.80 |
| 3 | 3.92 | **3.12** | 3.30 | not run |
| 4 | 5.98 | **4.52** | 4.70 | 5.05 |

The optimizer found a schedule shape we did not anticipate. At 3 documents,
the best pipeline schedule puts documents 0 and 1 with their filter 1
prompts in batch 1, and puts document 2's reading, document 2's filter 1
prompt, and the filter 2 prompts of batch 1's survivors together in batch 2.
Holding document 2 back gives the scheduler useful work to run while the
filter 1 answers arrive, so waiting costs nothing, and the pipeline gets
speculation's batch count without its wasted prompts. A simple prediction
that ignores holding documents back says speculation should win up to about
6 documents. In fact it wins only at 2, when there is nothing left to hold
back. Speculation should also win at the tail end of any query, for the
same reason.

The schedule builders used for the 10,000 document runs do not hold
documents back, and they finish 12 to 24 percent behind the exact optimizer
at 8 documents or fewer (2.80 versus 2.17 milliseconds at 2 documents, 5.05
versus 4.52 at 4). The gap fades by about 16 documents and is invisible at
10,000. Any claim about small queries should come from the exact optimizer,
or from a builder taught to hold documents back, which is on the plan.

The exact optimizer's practical limit, recorded per the paper's section
10.5: with single token chunks and deletion choices it reaches 4 documents
of about 7 tokens in roughly 146 seconds, and 3 documents for the online
case. With whole document chunks it reaches 4 documents at real lengths in
seconds. Past that, the number of states blows up, which matches the
paper's hardness claim, so large runs rely on built schedules plus lower
bounds instead.

## With four filters, pipeline still wins at full scale

A run with four filters, each with the same per stage pass rate, on the
tightest and loosest model and card pairs, from one outcome table, finish
times in seconds:

| pass rate per stage | model and card | task-first | lookahead 1 | lookahead 2 | lookahead 4 |
|---|---|---|---|---|---|
| 0.50 | 32B on L40S | 473.8 | **334.8** | 361.6 | 426.5 |
| 0.95 | 32B on L40S | 946.8 | **414.4** | 418.4 | 426.5 |
| 0.95 | 4B on H100 | 41.0 | **17.9** | 18.1 | 18.5 |

Lookahead 1 is the pipeline and lookahead 4 is full speculation. The
pipeline wins everywhere, and task-first falls far behind because it rereads
every surviving document at each of the four stages. Speculation's one
advantage is batch count. At a 0.95 pass rate on the 32B model and L40S,
the pipeline needs 138 batches and full speculation needs 53. The ideal
model charges nothing per batch beyond the work inside it, but real serving
software pays some fixed cost per batch. If that fixed cost exceeds about
140 milliseconds (the 12.1 second gap divided by the 85 extra batches),
speculation starts winning. Measuring that fixed cost is the first thing
the calibration phase should do.

## Caveats

- The ideal model omits per batch software overhead on purpose, so every
  claim that a policy never wins is a claim about the ideal model at the
  stated scale, following the paper's rule of labeling each number as a
  bound, a model result, or a measurement.
- The weight sizes are estimates from parameter counts, not measurements
  from the released model files. They enter only through the weight loading
  term, which stays under 1.2 seconds in these runs, and through the batch
  size at which computing overtakes loading.
- The small instance numbers come from one random outcome table each, so
  the exact document count where speculation stops winning moves from run
  to run.
- Attention uses the corrected width from convention C1. With the paper's
  equation 20 as written, every attention term would be 1.6 times smaller.
- The break-even pass rate of 0.203 is bracketed by the grid points 0.10
  and 0.25 in the actual runs, per the paper's rule against reporting a
  crossing point more precisely than the grid supports.
