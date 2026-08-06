# Next (the live list; reasoning in notes/PROPOSAL.md)

1. In-engine forks: landed 2026-08-05, bit-clean against the
   sequential control (spec_smoke.json; the ledger's fork section).
   What remains of this item: measure the fork's predicted win
   region (few live documents, where pipelined's rounds run half
   empty), then teach the planner when to pick forked against
   sequential speculation. Original design, kept for the record: The client still sends one
   request per document. When the document's blocks commit (stage
   one's prefill), the scheduler fabricates n-1 sibling sequences
   whose prompts are document + question j. The siblings hit the
   document's cached blocks, so each computes only its ~30 question
   tokens plus the partial boundary block, and all of them can sit
   in the same engine step. Each samples its one answer from its
   own prefill; there are no decode steps. The parent is parked
   (held out of the schedule) until the siblings finish; the
   scheduler splices their answers onto the parent's stream in
   stage order and finishes it, so the client contract is
   unchanged. Named risks: fabricated Request objects and output
   splicing are pinned to vLLM 0.26 internals; a sibling's free
   must not strip the shared document blocks' cache entries while
   its siblings are still in flight; parking must not busy-
   schedule. Decisions go in chainlogic (unit-tested with no
   engine), glue in scheduler.py, validation by the spec_smoke
   parity pattern plus a step-count check: a forked document must
   finish in two engine steps, not n. Answers are constrained to
   the yes/no token ids (allowed_token_ids), so every tier keeps
   the one-token contract and the 32B chatter problem is masked at
   the sampler. When this lands, delete the client-side
   speculation (the lookahead blocks in run_filter_chain) and move
   the composition executor in-engine with it.
2. Re-baseline flight (experiments/REBASELINE.md): done, closed
   2026-08-04; kept here until the paper's numbers restate.
3. Close the three blocking claims: shared-scan fidelity (the E13
   discriminator run), the planner refusal path (landed this
   session; E12's expected outcome corrected to match), SGLang
   completeness (fp8-KV speed cell plus a truth-scored accuracy
   arm).
4. Remaining roster, smallest first: E6, then E8-E12, numbered as
   in paper/PAPER.md.
5. Finish the paper revision: re-baselined numbers, then the
   related-work reading pass with receipts.
6. Parked, with triggers (unchanged): reasoning filters (the grid
   re-fly waits until the fork milestone lands, so the speculation
   arms measure the shipped mechanism); hybrid-attention models
   (allocator work); fused speculation (awaits the upstream kernel
   fix; our evidence chain is the bug report); joins (after the
   scan story closes).
