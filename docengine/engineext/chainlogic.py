"""Pure decision logic for the DocEngine scheduler.

This module must import no vLLM, torch, or numpy. The scheduler
subclass (scheduler.py) cannot even be imported without vLLM 0.26.0,
so every decision it makes is computed here on plain ints, strings,
lists, and sets, and scheduler.py only applies the results to engine
state. tests/test_engineext_logic.py covers this module with no engine
installed; a vLLM version bump can break the adapter, not these rules.

The request-id protocol is documented in scheduler.py's module
docstring; the rewind mechanics in paper/PAPER.md section 4 and the
ledger's chain-proof section (notes/RESULTS.md).
"""


# ---- request-id protocol ----------------------------------------------

def parse_tag(request_id):
    """Return (pin_tokens, doc_key, releases, uses) for a tagged id,
    else None. `uses` is the number of release mentions the pin waits
    for before freeing; one when absent.

    The last field is the free-text suffix and is never read as a
    directive: a suffix that happens to start with p, d, or r must not
    shadow a real directive (a tag like "rm" once swallowed every
    release, including the end-of-run flush)."""
    if not request_id.startswith("de1|"):
        return None
    pin, doc, rel, uses = 0, None, [], 1
    for part in request_id.split("|")[1:-1]:
        if part.startswith("p") and part[1:].isdigit():
            pin = int(part[1:])
        elif part.startswith("d") and len(part) > 1:
            doc = part[1:]
        elif part.startswith("r") and len(part) > 1:
            rel = ["*"] if part[1:] == "*" else part[1:].split(",")
        elif part.startswith("u") and part[1:].isdigit():
            uses = int(part[1:])
    return pin, doc, rel, uses


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


def is_spec(request_id):
    """A speculative chain: the same living request, but the engine
    advances through every stage regardless of the gate, so a query
    that needs all answers (a classifier set) gets them from one
    request per document. The "s" part rides alongside "c"; the
    machinery is identical except the advance rule."""
    return request_id.startswith("de1|") and "|s|" in request_id


def is_seq(request_id):
    """The sq part forces a speculative chain onto the sequential
    stage-by-stage path instead of forked siblings: the per-request
    control for fork validation."""
    return request_id.startswith("de1|") and "|sq|" in request_id


def is_compose(request_id):
    """A composed chain: stage k+1 reads stage k's output, so the
    advance APPENDS the next instruction at the current record end
    instead of rewinding to the document boundary. The co part rides
    alongside c; gating does not apply (stages end at their natural
    stop) and forks never fire (stages are dependent by
    construction)."""
    return request_id.startswith("de1|") and "|co|" in request_id


def compose_advances(stage, n_stages):
    """A composed chain advances until its last stage; there is no
    gate to fail and no speculation to cut short."""
    return stage < n_stages


def parse_switch(request_id):
    """The hybrid policy's switch stage from the sw<j> part: a gated
    chain that passes stage j forks all remaining filters instead of
    appending the next one, because too few documents survive past j
    to fill a round with gated work. None when absent (pure gating).
    Like every directive, the suffix is never read."""
    for part in request_id.split("|")[1:-1]:
        if part.startswith("sw") and part[2:].isdigit():
            return int(part[2:])
    return None


def hybrid_switch_stage(n_docs, selectivities, question_tail_tokens,
                        round_tokens):
    """The first stage j after which the expected survivors' gated
    question work no longer fills a round: survivors(j) x tail <
    round_tokens. Returns 0 when gating fills rounds through the
    whole chain (never switch), and 1 when even stage two would run
    underfilled (fork everything after the first answer)."""
    surviving = float(n_docs)
    for j, s in enumerate(selectivities[:-1], start=1):
        surviving *= s
        if surviving * question_tail_tokens < round_tokens:
            return j
    return 0


def is_release_all(releases):
    """The "*" release frees every pin outright, regardless of
    counts."""
    return releases == ["*"]


def plan_priority(tag):
    """Scheduling rank from the verified tag; lower runs first.
    Consumers of resident KV run first (0), new document reads next
    (1), unplanned traffic - only possible outside strict mode - last
    (2)."""
    if tag is None:
        return 2
    pin = tag[0]
    return 1 if pin > 0 else 0


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
    tokens across the continuations of a 10,000-document run - was the
    entire 4.4-second first-flight deficit (the ledger's chain-proof
    section in notes/RESULTS.md)."""
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


def spec_advances(stage, n_stages):
    """A speculative chain advances past every stage but the last,
    whatever the gate said: the query needs all answers, so a failed
    stage never ends the document early. Spec mode assumes the
    one-token answer contract (no decisive-token window): every stage
    emits exactly one token, so the client's stage record stays
    aligned."""
    return stage < n_stages


# ---- forked speculation -------------------------------------------------

def fork_rid(parent_rid, stage):
    """The request id of a scheduler-fabricated sibling: the parent's
    id with a fk<stage> part spliced before the suffix. Siblings never
    pass through add_request's directive parsing (the scheduler
    creates them directly), so the part is for logs and uniqueness,
    not dispatch."""
    head, _, suffix = parent_rid.rpartition("|")
    return f"{head}|fk{stage}|{suffix}"


class ForkBoard:
    """Answer collection for one forked document. The parent request
    samples stage 1 itself and parks; each fabricated sibling owns one
    of stages 2..n and reports its single answer token here. When the
    board is full, the parent's stream is finished with the collected
    tokens in stage order, so the client sees the same n-token record
    a sequential speculative chain produces."""

    def __init__(self, stages):
        self.want = frozenset(stages)
        self.got = {}
        self.diag = {}   # stage -> opaque per-sibling diagnostics

    def record(self, stage, toks):
        """Store a sibling's output (a token list: one token for
        filters and classify, a whole generation for maps); returns
        True when the board just became complete (exactly once)."""
        was_done = self.done
        self.got[stage] = list(toks)
        return self.done and not was_done

    @property
    def done(self):
        return set(self.got) >= self.want

    def in_stage_order(self):
        """The collected outputs sorted by stage."""
        return [self.got[s] for s in sorted(self.got)]


def splice(outputs, sep):
    """Flatten per-stage outputs for the parent's stream. With no
    separator (single-token stages), plain concatenation - the client
    reads one token per stage. With a separator (generative stages),
    one sep token lands between stages; generations stop at the
    end-of-sequence token and so never contain it, which is what
    makes the separator unambiguous."""
    flat = []
    for k, toks in enumerate(outputs):
        if sep is not None and k > 0:
            flat.append(sep)
        flat.extend(toks)
    return flat


def parse_sep(request_id):
    """The separator token id from a registration's E part; None when
    absent (single-token stages need no separator)."""
    for part in request_id.split("|")[1:-1]:
        if part.startswith("E") and part[1:].isdigit():
            return int(part[1:])
    return None


# ---- the step trace -----------------------------------------------------

def doc_key(request_id):
    """The document key (the d part) a request works for; None when
    the id carries none (registrations, flushes, foreign traffic)."""
    tag = parse_tag(request_id)
    return tag[1] if tag else None


def step_record(now_s, sched_cpu_s, tokens_by_request, doc_keys,
                free_blocks, total_blocks, block_size, decoding=None):
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
    `decoding` the split falls back to the one-token heuristic."""
    counts = list(tokens_by_request.values())
    if decoding is None:
        decode_seqs = sum(1 for c in counts if c == 1)
        prefill = sum(c for c in counts if c > 1)
    else:
        decode_seqs = sum(1 for r in tokens_by_request if r in decoding)
        prefill = sum(c for r, c in tokens_by_request.items()
                      if r not in decoding)
    return dict(
        t=round(now_s, 6),
        sched_ms=round(sched_cpu_s * 1e3, 3),
        tokens=int(sum(counts)),
        seqs=len(counts),
        decode_seqs=decode_seqs,
        prefill_tokens=int(prefill),
        unique_docs=len({d for d in doc_keys if d is not None}),
        kv_used_tokens=(total_blocks - free_blocks) * block_size,
        kv_total_tokens=total_blocks * block_size)


# ---- the strict-mode invariant ----------------------------------------

STRICT_VIOLATION = (
    "DocEngine single-tenant invariant violated: the recency rule "
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


# ---- pin refcount bookkeeping -----------------------------------------

def new_pin_intent(pin_tokens, doc, pins):
    """Record a pin intent only for a real pin: a positive token
    count, a document key, and a document not already pinned."""
    return pin_tokens > 0 and doc is not None and doc not in pins


def pin_ready(intent, computed_tokens, pins):
    """Take the pin when its request's blocks are freed, but only if
    the request actually computed past the pin span (a preempted or
    aborted request may not have) and no sibling pinned the document
    first."""
    return (intent is not None
            and computed_tokens >= intent[0]
            and intent[1] not in pins)


class PinLedger:
    """Pin refcount bookkeeping for shared scans: a pin waits for
    `uses` release mentions - one per query reading the document - and
    frees exactly once, when the mentions run out. Payloads (the
    scheduler stores its pinned block lists here) are opaque to this
    module."""

    def __init__(self):
        self.pins = {}   # doc key -> payload
        self.refs = {}   # doc key -> release mentions left

    def __contains__(self, key):
        return key in self.pins

    def add(self, key, payload, uses):
        self.pins[key] = payload
        self.refs[key] = uses

    def pop(self, key):
        """Drop the pin and its mention count; returns the payload,
        None when the key holds no pin."""
        self.refs.pop(key, None)
        return self.pins.pop(key, None)

    def to_free(self, releases):
        """Apply release mentions; drop and return the (key, payload)
        pairs whose pins free now, so each pin is returned exactly
        once. "*" frees every pin outright, regardless of counts.
        Unknown keys are ignored: a release can arrive for a pin that
        was already freed or was never taken."""
        if is_release_all(releases):
            return [(key, self.pop(key)) for key in list(self.pins)]
        freed = []
        for key in releases:
            if key not in self.pins:
                continue
            left = self.refs.get(key, 1) - 1
            if left > 0:
                self.refs[key] = left
            else:
                freed.append((key, self.pop(key)))
        return freed
