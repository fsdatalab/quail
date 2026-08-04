# DocEngine

DocEngine runs semantic scans — natural-language predicates applied
to every document in a corpus — by treating KV as an object the
query plan owns. KV is the key-value cache: the per-token attention
state a transformer stores so it can extend a prefix instead of
recomputing it. Serving engines manage KV with a recency rule;
DocEngine's planner decides, per query, what KV is built, kept,
reused, and dropped, and the engine enforces that plan. It is built
inside vLLM. The paper is paper/PAPER.md.

## Map

- docengine/plan — the planner: describe the query, get a plan.
- docengine/reasoning — the analytical cost layer.
- docengine/runtime/engine_client.py — the client library.
- docengine/engineext — the in-engine scheduler, and the fusion
  path its fidelity gate closed (kept as the bug report).
- docengine/sched, validator, costmodel.py, lb.py — the schedule
  builders and the independent checker that replays them.
- attic/ — the superseded theory program, kept runnable.

## Tests

    pip install -e ".[dev]"
    pytest

## Flights

GPU experiments run on Modal (`modal run experiments/...`). The
next one is the re-baseline flight: experiments/REBASELINE.md.

## Evidence rule

No claim ships without a named result file banked in results/, in
the same commit as the prose that cites it. The ledger is
notes/RESULTS.md.
