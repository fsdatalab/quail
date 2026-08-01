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

The revised paper organizes every claim into four layers, and the repo now
computes the first three: a resource lower bound that no schedule can beat,
an asymptotic latency target from a linear program over steady batch rates,
a constructed finite schedule replayed from the program's rates and checked
by an independent validator, and a measured engine number that is a later
phase and deliberately not started. Every number below comes from the
paper's ideal cost model with its corrected attention width and its
write-through storage rule, on 10,000 real IMDb reviews (2,966,000 document
tokens) tokenized for the Qwen3 models. The pass rate of a filter is the
fraction of documents that pass it.

Raw data: results/lp_two_stage.csv (the program and its replays),
results/n10k_two_stage.csv (direct schedule builders), manifests in
results/manifests/, and the small instance study in
experiments/run_smallN.py.

## The three computed layers agree, which is the main check

For every combination of two models (Qwen3-4B, Qwen3-32B), two cards (H100,
L40S), pass rates 0.1, 0.5, 0.9, and the three policies, all of the
following hold:

- The lower bound never exceeds the replayed finite latency, in all 36
  cells.
- The replayed finite latency lands within 0.5 percent of the program's
  asymptotic target. The signed difference runs from minus 0.07 to plus 2.0
  seconds and is legitimately either sign, because the asymptotic target is
  not a bound for a finite run.
- Where the direct schedule builders had already produced certified optima,
  the program reproduces them to the digit. For example, task-first on the
  4B model and H100 at pass rate 0.5 gives 16.56 seconds by ledger, by
  built schedule, and by program target, and the replay gives 16.62.

Selected finite latencies in seconds (replayed, validated):

| model and card | policy | pass rate 0.10 | 0.50 | 0.90 |
|---|---|---|---|---|
| 4B on H100 | task-first | **12.14** | 16.62 | 21.04 |
| | pipeline | 13.05 | **13.80** | **14.54** |
| | full speculation | 14.74 | 14.74 | 14.74 |
| 32B on L40S | task-first | **281.9** | 384.3 | 486.1 |
| | pipeline | 301.7 | **319.1** | **336.2** |
| | full speculation | 340.9 | 340.9 | 340.9 |

The conclusions from the earlier direct builders stand. Task-first wins
below a pass rate of about 0.20 (the crossing point is 500,000 over
2,466,000, or 0.203). Pipeline wins above it. Full speculation never wins
at this scale under the ideal cost model, because there is always other
work to hide filter waits behind, and even at 3.3 percent KV residency
(32B on L40S) the pipeline never recomputes anything.

## What the program layer adds

The linear program chooses long-run rates for state-conditioned batch
actions, subject to queue balance, cache-state balance, and one GPU second
per second, and its optimum is a certified throughput ceiling for its
supplied state and action set. Details of this implementation, all
recorded in the result files:

- Task-first and full speculation keep no per-document KV between batches,
  so their cache state collapses to a single state and the program runs
  over batch templates with exact empirical length types (1,037 distinct
  lengths, no averaging).
- The pipeline uses a one-dimensional quantized count of resident
  survivors. Under the paper's assumption that pass rates do not depend on
  document length, the length mix of survivors equals the admission mix
  exactly, so per-survivor costs use the exact mix and no per-length cache
  state is needed. Survivor counts above the capacity share overflow to a
  recompute queue, which is the model's eviction path.
- Costs of fluid actions are expectations over the mix, which the paper
  permits when declared. Real feasibility is enforced during replay, and
  the independent validator replays every batch of every reported schedule.

The replay converts rates into actual batches with real documents and
tags every batch as fill, core, repair, or drain. Repair batches were never
needed in these runs. One structural observation: the task-first replay
executes many batches far below their template size (its queues hold at
most a few documents of each exact length), which costs nothing under the
ideal model because those batches stay compute-bound, but it would cost
real time under a calibrated model with a fixed charge per batch. The
pipeline replay packs 7 to 100 large batches instead.

## Small documents counts and the exact reference solver

The exact solver (offline Dijkstra plus online value iteration) is now the
validation reference the revised paper assigns it. On real Qwen3-4B and
H100 numbers it solves up to 4 documents in seconds, and it found the one
schedule shape the other layers miss: holding a document back so that the
survivors' next filter prompts run inside the held document's prefill
batch, which makes filter waits free. Speculation wins only when nothing is
left to hold back (2 documents, or the tail of a query). The direct
builders run 12 to 24 percent behind the exact optimum below 8 documents,
and the program layer inherits a milder version of the same limitation
through its restricted action menu.

## Caveats

- Every "X never wins" statement is about the ideal cost model at the
  stated scale. The batch-count differences (for example 100 pipeline
  batches versus 53 full-speculation batches at high pass rates on the
  32B model and L40S) mean a fixed per-batch overhead of roughly 140
  milliseconds would start reversing the speculation conclusion; that is
  the first target for the calibrated phase.
- The program's value is the optimum of a restricted state and action set,
  labeled per the paper: a fuller action search could raise it, and the
  achievability of the rate is established here only by the successful
  replays, not by the program itself.
- Weight sizes are still estimates from parameter counts, not measured
  from the released files.
- The small instance numbers come from single outcome draws.
