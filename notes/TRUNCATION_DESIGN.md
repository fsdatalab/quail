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
- Detecting the answer: our scheduler subclass sees sampled tokens in
  update_from_output. A chain request that has produced its stage's
  answer token is intercepted there, before stop processing can
  finish it (max_tokens=2 on the request means the engine's own stop
  check never fires between stages, because each rewind clears the
  output count; the final stage is finished explicitly by the
  scheduler).
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

1. Rewind proof: chain mode for two filters at fifty documents,
   answers bit-identical to request mode, requests exactly one per
   document, notes accounting clean (strict mode's zero heuristic
   eviction invariant must keep holding).
2. The gate and registration protocol; chains of four filters; kill
   on failure.
3. The 10k grid: target high forties against 52.3 seconds, and the
   30k-document k=1 cell unchanged while chain mode replaces the
   pathological k=2.

## Risks, named

- The stop path (request.resumable handling near scheduler.py 1755)
  may finish a chain request before our intercept in some orderings;
  milestone 1 exists to find this.
- Async scheduling overlap means the sampled token for step t is
  processed while step t+1 may already include the request; the
  rewind must only be applied at a step boundary where the request is
  not scheduled, or the request must be held out of scheduling for
  one step (the session mechanism already tolerates this by parking
  the request in WAITING).
- Spec decode, LoRA, and multimodal interactions are out of scope and
  asserted absent for chain requests.
