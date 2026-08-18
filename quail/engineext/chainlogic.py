"""Pure decision logic for the Quail scheduler.

This module must import no vLLM, torch, or numpy. The scheduler
subclass (scheduler.py) cannot even be imported without vLLM 0.26.0,
so every decision it makes is computed here on plain ints, strings,
lists, and sets, and scheduler.py only applies the results to engine
state. tests/test_engineext_logic.py covers this module with no engine
installed; a vLLM version bump can break the adapter, not these rules.

The request-id protocol is documented in scheduler.py's module
docstring.
"""


# ---- request-id protocol ----------------------------------------------

def is_planned(request_id):
    """A planned request: carries the de1| wire-format prefix. In
    single-tenant mode everything else is refused."""
    return request_id.startswith("de1|")


def doc_key(request_id):
    """The document key (the d part) a request works for; None when
    the id carries none (registrations, foreign traffic).

    The last field is the free-text suffix and is never read as a
    directive: a suffix that happens to start with d must not shadow
    a real directive."""
    if not request_id.startswith("de1|"):
        return None
    for part in request_id.split("|")[1:-1]:
        if part.startswith("d") and len(part) > 1:
            return part[1:]
    return None


def parse_qid(request_id):
    """The query id a registration or chain request belongs to; "0"
    when the id carries no Q part (the single-query protocol)."""
    for part in request_id.split("|")[1:-1]:
        if part.startswith("Q") and len(part) > 1:
            return part[1:]
    return "0"


def parse_yes_no(request_id):
    """The yes and no token id sets a registration id carries in its Y
    and N parts; empty sets when absent. Like every directive, the
    last field (the suffix) is never read."""
    yes, no = set(), set()
    for part in request_id.split("|")[1:-1]:
        if part.startswith("Y") and len(part) > 1:
            yes = {int(x) for x in part[1:].split(",")}
        elif part.startswith("N") and len(part) > 1:
            no = {int(x) for x in part[1:].split(",")}
    return yes, no


def is_registration(request_id):
    """A chain registration request: its prompt carries the question
    token lists, its id the gate token ids."""
    return request_id.startswith("de1|") and "|reg|" in request_id


def is_chain(request_id):
    """A chain request: one living request runs a document's whole
    filter chain."""
    return request_id.startswith("de1|") and "|c|" in request_id


# ---- registration payload ---------------------------------------------

def parse_registration(toks):
    """Decode a registration prompt into the question token lists.
    Layout: [question count, then per question its length followed by
    its tokens]."""
    nq, i, qs = toks[0], 1, []
    for _ in range(nq):
        ln = toks[i]
        i += 1
        qs.append(toks[i:i + ln])
        i += ln
    return qs


def shared_preamble(qs):
    """Token count of the prefix every question shares; 0 for a single
    question (a one-stage chain never continues, so nothing reuses a
    preamble). The preamble's KV must survive every rewind; see
    rewind_target."""
    qc = 0
    for col in zip(*qs):
        if len(set(col)) != 1:
            break
        qc += 1
    return qc if len(qs) > 1 else 0


# ---- rewind arithmetic ------------------------------------------------

def document_boundary(prompt_len, first_question_len):
    """Where the document ends inside a chain request's first prompt.
    The prompt is [document + first question], so the boundary is the
    prompt length minus the first question's length."""
    return prompt_len - first_question_len


def rewind_target(d, qc):
    """The position a chain rewinds to between stages: the document
    boundary d plus the questions' shared preamble qc.

    The rewind must keep the preamble, not just the document. In the
    reference query the preamble is 33 tokens; erasing it made every
    continuation recompute it, and that recomputation - 353,654 extra
    tokens across the continuations of a 10,000-document run - was a
    4.4-second deficit against separate requests."""
    return d + qc


def full_blocks(tokens, block_size):
    """How many blocks the first `tokens` tokens fill completely.
    Block hashes exist only for full blocks, so a rewound request's
    hash record is cut here; a pin of `tokens` tokens also covers
    exactly this many whole blocks."""
    return tokens // block_size


def retained_blocks(tokens, block_size):
    """How many blocks a rewound request keeps: every block holding
    any of the first `tokens` tokens, so a partly filled boundary
    block stays."""
    return -(-tokens // block_size)


def partial_boundary(tokens, block_size):
    """True when the boundary falls inside a block. That block is kept
    (see retained_blocks), but it was cached with content past the
    boundary that the next continuation overwrites, so its cache entry
    must be stripped or it would serve stale hits."""
    return bool(tokens % block_size)


def continuation_tail(question, qc):
    """The tokens a continuation appends after a rewind: the next
    question minus the shared preamble the rewind kept. Kept preamble
    plus this tail equals the whole question, so every stage still
    sees exactly [document + its question]."""
    return list(question[qc:])


# ---- the decisive-token gate ------------------------------------------

KEEP_DECODING = "keep-decoding"
PASSED = "passed"
FAILED = "failed"
UNDECIDED = "undecided"


def gate_decision(tok, stopped, yes_ids, no_ids):
    """Judge a stage from `tok`, its latest sampled token id (None
    when nothing was sampled). `stopped` says the engine's own stop
    fired, which ends the stage's token window.

    With a no set registered, the stage may run several tokens - the
    model may restate the line before answering - but it ends at the
    first token that decides, so trailing chatter never decodes:
    PASSED or FAILED with `stopped` False tells the caller to stop the
    request now. Without a no set the window is one token and the
    engine's stop always decides.

    Returns KEEP_DECODING (window open, nothing decisive yet), PASSED
    (a yes token), FAILED (a no token; or any non-yes stop token when
    no no set is registered), or UNDECIDED (window exhausted without a
    decisive token)."""
    if not stopped:
        if no_ids and tok is not None and (tok in yes_ids or tok in no_ids):
            return PASSED if tok in yes_ids else FAILED
        return KEEP_DECODING
    if tok is not None and tok in yes_ids:
        return PASSED
    if not no_ids:
        return FAILED
    if tok is not None and tok in no_ids:
        return FAILED
    return UNDECIDED


def stage_advances(decision, stage, n_stages):
    """A chain advances only past a passed stage that is not the last.
    A fail, an undecided window, or the final stage keeps the engine's
    real finish, which closes the client's stream."""
    return decision == PASSED and stage < n_stages


# ---- the step trace -----------------------------------------------------

def step_record(now_s, sched_cpu_s, tokens_by_request, doc_keys,
                free_blocks, total_blocks, block_size, decoding=None,
                shapes=None):
    """One scheduler step reduced to plain numbers for the trace file:
    when it was scheduled and how long the packing decision took; how
    many tokens moved, over how many sequences; the prefill and decode
    split; how many distinct documents those sequences serve (sequences
    over documents is the filters-in-flight-per-document ratio); and
    the KV pool's occupancy in tokens.

    `decoding` is the set of request ids that are generating - they
    already hold sampled output, so the split is exact: a one-token
    final prefill chunk counts as prefill. A one-token filter run must
    therefore trace zero decode (each stage's answer is sampled from
    its own last prefill chunk's forward pass, never a decode step);
    any decode band on a filter trace is a misconfiguration. Without
    `decoding` the split falls back to the one-token heuristic.

    `shapes` (calibration only) is a list of [new, cached] token pairs,
    one per scheduled request, recorded so a measured step can be
    checked against the composition it was designed to have."""
    counts = list(tokens_by_request.values())
    if decoding is None:
        decode_seqs = sum(1 for c in counts if c == 1)
        prefill = sum(c for c in counts if c > 1)
    else:
        decode_seqs = sum(1 for r in tokens_by_request if r in decoding)
        prefill = sum(c for r, c in tokens_by_request.items()
                      if r not in decoding)
    rec = dict(
        t=round(now_s, 6),
        sched_ms=round(sched_cpu_s * 1e3, 3),
        tokens=int(sum(counts)),
        seqs=len(counts),
        decode_seqs=decode_seqs,
        prefill_tokens=int(prefill),
        unique_docs=len({d for d in doc_keys if d is not None}),
        kv_used_tokens=(total_blocks - free_blocks) * block_size,
        kv_total_tokens=total_blocks * block_size)
    if shapes is not None:
        rec["shapes"] = [[int(new), int(cached)] for new, cached in shapes]
    return rec


def request_shape(scheduled_tokens, computed_tokens_after):
    """The [new, cached] pair of one scheduled request. The scheduler
    advances a request's computed-token count by its scheduled tokens
    inside schedule(), so at schedule exit the cached context is the
    computed count minus this step's share."""
    return [int(scheduled_tokens),
            int(computed_tokens_after) - int(scheduled_tokens)]


def attach_step_timing(rec, exec_s, update_s):
    """Amend a step record with the execution window (schedule exit to
    output processing, which is the GPU wait in the synchronous
    engine) and the output-processing time itself. Amended in place
    after the step runs, because both spans end after the record is
    built."""
    rec["exec_ms"] = round(exec_s * 1e3, 3)
    rec["update_ms"] = round(update_s * 1e3, 3)
    return rec


# ---- the strict-mode invariant ----------------------------------------

STRICT_VIOLATION = (
    "Quail single-tenant invariant violated: the recency rule "
    "evicted a cached block, so some memory escaped the plan's "
    "accounting")


def is_heuristic_eviction(evicted, plan_evicting):
    """An eviction the plan did not initiate. The plan accounts for
    every block it admits, so a heuristic eviction means memory
    escaped that accounting."""
    return bool(evicted and not plan_evicting)


def strict_violation(evicted, plan_evicting, strict):
    """Whether an eviction must raise: in strict (single-tenant) mode
    any heuristic eviction is an error, never a policy."""
    return bool(strict and is_heuristic_eviction(evicted, plan_evicting))
