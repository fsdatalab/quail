# DocEngine

DocEngine accelerates semantic filters, classifiers, and joins over
a document corpus. Each one asks a large language model a fixed
natural-language question about every document. A filter is gated,
so a document that fails one question skips the rest. A classifier
set needs every answer for every document. A join asks its question
about pairs of documents.

The speedup comes from treating KV as an object the query plan
owns. KV is the key-value cache, the per-token attention state a
transformer stores so it can extend a prefix instead of recomputing
it. A serving engine keeps or drops KV by recency. DocEngine's
planner decides, per query, what KV is built, kept, reused, and
dropped, and the engine enforces that plan. It is built inside
vLLM. The paper is paper/PAPER.md.

## Operators

Four physical operators, from two semantics crossed with one rule.
Filter means gated (a failed prompt skips the rest); map means
binary classification, every prompt answered for every document.
The rule forks a document's remaining prompts at the first stage
whose surviving work would underfill a compute round.

- pipelined_filter: gate, one prompt per round.
- hybrid_filter: gate, then fork the survivors at the computed
  switch stage. Measured to dominate pure gating.
- pipelined_map: every prompt, one per round (large corpora).
- hybrid_map: every prompt, forked after the first (small corpora).

Three entry points ride these operators:

    from docengine.api import query, classify, map

- query(engine, docs, filters, est_selectivities): gated filters.
  The estimates reorder execution (cheapest rejection first), place
  the fork switch, and price the plan; answers return under the
  caller's filter indices. Single-token answers, no decode steps.
- classify(engine, docs, prompts, classes): every prompt on every
  document, each answer exactly one of the class labels. Sampling
  is constrained to the classes' first tokens, so this is the same
  no-decode tier as filters, generalized past yes/no.
- map(engine, docs, prompts, max_output_tokens): open-ended
  generation per prompt per document - the one tier where decode
  exists. One request per (document, prompt); the document reads
  once and later prompts reuse its KV, the first prompt committing
  it before the rest launch. The plan's predicted makespan does not
  yet price generation time.

## Map of the code

- docengine/api.py — query(): the string entry point, plus
  configure() for deployment facts (the KV store) and
  plan_engine_kwargs() to boot the engine the plan's way.
- docengine/plan — the planner: describe the query, get a plan or
  a refusal naming the violated constraint.
- docengine/reasoning — the analytical cost layer.
- docengine/runtime/engine_client.py — the client library.
- docengine/engineext — the in-engine scheduler (rewind, forks,
  pins, strict memory), the fork connector that copies partial
  boundary blocks, and the fusion path its fidelity gate closed
  (kept as the bug report).
- docengine/sched, validator, costmodel.py, lb.py — the schedule
  builders and the independent checker that replays them.

## Tests

    pip install -e ".[dev]"
    pytest

## Flights

GPU experiments run on Modal (`modal run experiments/...`). The
re-baseline flight landed 2026-08-04 (experiments/REBASELINE.md is
its record); every current number sits on the CUDA 13 image at the
97,889 tokens-per-second anchor.

## Evidence rule

No claim ships without a named result file banked in results/, in
the same commit as the prose that cites it. The ledger is
notes/RESULTS.md.
