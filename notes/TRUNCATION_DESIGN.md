# Sequence truncation: design against vLLM 0.26 internals

The build: one living engine sequence per document runs the whole
filter chain. Append a question's tokens, decode the answer token,
roll the sequence back to the document boundary, append the next
question, or finish. One request per document instead of n; the
per-request toll (profiled: 22 percent deepcopy, 5 percent input
processing, 4 percent event machinery, all times n today) is paid
once, the long-document double-read race disappears by construction,
and prompts are unchanged: every question still sees exactly
[document + itself].

## The mechanisms, with their upstream precedents

Everything needed exists in vLLM 0.26 in some form; line references
are to the extracted wheel source in the session scratchpad.

- Appending input to a live request: the streaming session mechanism
  (`_update_request_as_session`, scheduler.py 1286). It extends
  prompt_token_ids and _all_token_ids, calls
  request.update_block_hashes(), resets status to WAITING, and
  re-enqueues. Our advance step is this function plus a truncation
  first.
- Rewinding computed state: a WAITING request is sent to the worker
  with full state when scheduled (scheduled_new_reqs, scheduler.py
  1016), so the worker needs no incremental consistency with the
  request's past; preemption already re-runs requests with a lower
  computed count through this exact path. Rewind is therefore: set
  num_computed_tokens to the document token count, and the normal
  WAITING scheduling path re-syncs everything.
- Truncating the token record: mirror `_update_request_as_session`,
  but delete back to the document boundary d instead of keeping
  computed output: truncate _all_token_ids and prompt_token_ids to d,
  clear _output_token_ids, set num_prompt_tokens, and truncate
  request.block_hashes to d // 16 entries (hashes exist only for full
  blocks, so the partial boundary block has none to remove).
- Freeing the tail notes: truncate the coordinator's req_to_blocks
  for the request to ceil(d / 16) blocks and free_blocks the removed
  tail (strip hashes first, as the strict-mode discipline requires).
  Positions past d inside the retained partial boundary block are
  simply overwritten when the next question computes at those
  positions; the block is private to the request, so this is safe.
- Detecting the answer: each stage runs with max_tokens=1, so the
  engine's own stop check fires on the answer token. The scheduler
  judges the token inside that stop check
  (`_update_request_with_output`). The subtlety, found by the first
  smoke run: the engine captures the finish reason from the request's
  status BEFORE the stopped-request hook runs, and sends it to the
  client even when the hook keeps the request alive — which ends the
  client's stream mid-chain. So when a stage passes and questions
  remain, the scheduler erases the stop status right there in the
  stop check. The captured finish reason is then empty, the client's
  stream stays open, and the stop path still routes into
  `_handle_stopped_request`, which rewinds and appends the next
  question. The final stage (or a failed one) keeps its real finish
  and closes the stream normally.
- The gate: the client registers the yes token ids and each stage's
  question token list once per run, via a registration request whose
  prompt tokens carry the question lists separated by a sentinel and
  whose tag carries the yes ids; the scheduler stores them and aborts
  the registration request. From then on pass or fail is decided
  in-engine by comparing the sampled token id.
- Answer delivery: outputs stream per token, so the client's one
  generator per document receives each stage's answer token as a
  delta and parses per-stage answers; the request finishes after the
  last stage or the first failure.

## Milestones

1. DONE. Rewind proof at fifty documents, two filters: identical
   survivors and answers, one request per document, clean notes
   accounting, zero heuristic evictions.
2. DONE. Gate, registration, kill on failure, four-filter chains:
   identical outcomes, three rewinds per surviving document.
3. DONE, both halves. 10k grid: 49.9 seconds against request mode's
   52.0 (target was high forties against 52.3), after the rewind
   learned to keep the questions' 33-token shared preamble - erasing
   it made every continuation recompute it, which was the entire
   first-flight deficit of 4.4 seconds (353,654 extra tokens at
   80,000 per second). Long documents, 100 at 30,000 tokens: chain
   mode 79.0 seconds, identical answers and survivors to the
   one-in-flight plan's 79.0, with 100 requests against 150; the
   pathological two-in-flight plan (156.9 seconds, half-percent hit
   rate, corpus read twice) is obsolete - a document that is one
   living request cannot race itself.

## Risks, named (and how milestone 1 settled them)

- The stop path finishing a chain request before our intercept: this
  was real, and it was the first flight's bug — not an ordering race
  but the finish reason being captured before the hook and sent to
  the client unconditionally. Fixed by judging in the stop check and
  erasing the stop status there (see "Detecting the answer" above).
- Async scheduling overlap: does not apply. vLLM disables async
  scheduling whenever the scheduler is a subclass of the plain
  Scheduler class, which ours is (the engine logs a warning saying
  exactly this). Sampled tokens are processed at step boundaries
  where the request is not scheduled.
- Two smaller findings from the same flight. First, the block at the
  document boundary was cached with question-one content that the
  next question overwrites; the rewind now strips that cache entry.
  Second, a run tag starting with "r" (or "p", "d") made every
  request id suffix look like a directive and silently swallowed all
  releases; the parser now never reads the last id field as a
  directive.
- Spec decode, LoRA, and multimodal interactions are out of scope and
  asserted absent for chain requests.
