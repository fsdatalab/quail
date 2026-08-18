"""Checks for the scheduler's pure decision logic (chainlogic.py),
driven from planted flags and hand-built request ids, with no vLLM
installed. Numbers mirror the measured run shape where it helps:
block size 16, a 345-token document (21 full blocks plus 9 tokens),
and the questions' 33-token shared preamble."""

import re

from quail.engineext import chainlogic
from quail.engineext.chainlogic import (
    FAILED, KEEP_DECODING, PASSED, UNDECIDED, continuation_tail,
    doc_key, document_boundary, full_blocks, gate_decision, is_chain,
    is_heuristic_eviction, is_planned, is_registration, parse_qid,
    parse_registration, parse_yes_no, partial_boundary,
    retained_blocks, rewind_target, shared_preamble, stage_advances,
    strict_violation)

PAGE = 16
BODY = 345          # 21 full blocks of 16, plus 9 remainder tokens
Q_COMMON = 33       # question tokens shared by every stage
YES_TOK, NO_TOK = 111, 222


# ---- request-id protocol ----------------------------------------------

def test_doc_key_and_suffix_trap():
    """The document key comes from the d part; the free-text suffix is
    the last field and must never be read as a directive, even when it
    starts with d."""
    assert doc_key("de1|c|d7|q-7-0") == "7"
    assert doc_key("de1|c|Q1|d7|s-7-1") == "7"
    assert doc_key("de1|reg|Y111|s-reg") is None    # no document
    assert doc_key("foreign-42") is None
    assert doc_key("de1|c|d7|d9") == "7"            # suffix "d9" is text
    assert doc_key("de1|c|dog-run") is None         # suffix never read


def test_is_planned_is_the_prefix():
    """Single-tenant mode admits exactly the de1|-prefixed ids."""
    assert is_planned("de1|c|d7|q-7-0")
    assert is_planned("de1|reg|Y111|s-reg")
    assert not is_planned("probe-3")
    assert not is_planned("q-7-0")


def test_qid_defaults_and_ignores_suffix():
    assert parse_qid("de1|c|Q1|d7|s-7-1") == "1"
    assert parse_qid("de1|c|d7|q-7-0") == "0"       # single-query default
    assert parse_qid("de1|c|d1|Q9-x") == "0"        # suffix never read


def test_yes_no_sets_and_suffix_trap():
    yes, no = parse_yes_no("de1|reg|Q0|Y111,112|N222|run-a")
    assert yes == {111, 112} and no == {222}
    yes, no = parse_yes_no("de1|reg|Y111|s")
    assert yes == {111} and no == set()             # no set optional
    _, no = parse_yes_no("de1|reg|Y111|N9")         # suffix "N9" is text
    assert no == set()


def test_request_classification():
    assert is_registration("de1|reg|Q1|Y111|N222|s-reg")
    assert is_chain("de1|c|d7|q-7-0")
    assert is_chain("de1|c|Q1|d7|s-7-1")
    assert not is_chain("de1|d7|q-7-0")             # no c part
    assert not is_registration("de1|c|d7|q-7-0")
    assert not is_chain("foreign-42")


# ---- registration payload ---------------------------------------------

def _questions():
    """Three questions sharing a 33-token preamble, distinct tails of
    different lengths (the client's real shape)."""
    pre = list(range(1000, 1000 + Q_COMMON))
    return [pre + [2000 + j] * (3 + j) for j in range(3)]


def test_registration_prompt_roundtrip():
    """Decode the [count, len, tokens, ...] layout the client builds."""
    qs = _questions()
    reg = [len(qs)]
    for q in qs:
        reg += [len(q)] + list(q)
    assert parse_registration(reg) == qs
    assert parse_registration([1, 2, 5, 6]) == [[5, 6]]


def test_shared_preamble_is_33_tokens():
    qs = _questions()
    assert shared_preamble(qs) == Q_COMMON
    # kept preamble plus appended tail is exactly the question again
    for q in qs:
        assert q[:Q_COMMON] + continuation_tail(q, Q_COMMON) == q


def test_shared_preamble_edges():
    """A single question has no continuation, so no preamble; a
    question that is a prefix of another shares its full length."""
    assert shared_preamble([[1, 2, 3, 4]]) == 0
    assert shared_preamble([[1, 2, 3], [1, 2, 3, 4]]) == 3
    assert shared_preamble([[1, 2], [3, 4]]) == 0


# ---- rewind arithmetic ------------------------------------------------

def test_rewind_target_keeps_document_plus_preamble():
    """The 4.4-second trap: rewinding to the document boundary alone
    erases the shared preamble and every continuation recomputes it.
    The target must sit exactly one preamble past the boundary."""
    d = BODY
    assert rewind_target(d, Q_COMMON) == d + Q_COMMON == 378
    assert rewind_target(d, Q_COMMON) > d
    assert rewind_target(d, 0) == d          # single-question chain


def test_boundary_block_arithmetic():
    """Hashes exist only for full blocks; the partly filled boundary
    block is retained but its cache entry must be stripped."""
    assert full_blocks(BODY, PAGE) == 21          # 336 of 345 tokens
    assert retained_blocks(BODY, PAGE) == 22      # the 9-token remainder
    assert partial_boundary(BODY, PAGE)
    target = rewind_target(BODY, Q_COMMON)        # 378 tokens
    assert full_blocks(target, PAGE) == 23
    assert retained_blocks(target, PAGE) == 24
    assert partial_boundary(target, PAGE)
    # an exact multiple has no partial block and nothing to strip
    assert full_blocks(352, PAGE) == retained_blocks(352, PAGE) == 22
    assert not partial_boundary(352, PAGE)


def test_block_counts_consistent_at_every_length():
    """The hash cut can never pass the retained count, and the two
    differ by exactly the partial boundary block."""
    for d in range(0, 3 * PAGE + 1):
        fb, rb = full_blocks(d, PAGE), retained_blocks(d, PAGE)
        assert rb - fb == (1 if partial_boundary(d, PAGE) else 0)
        assert fb * PAGE <= d <= rb * PAGE


def test_rewind_reconstructs_each_stage_prompt():
    """Walk one document's token record through three passing stages
    the way the scheduler does: rewind to the target, append the next
    tail. Every stage must see exactly [document + its question]."""
    doc = [10] * BODY
    qs = _questions()
    qc = shared_preamble(qs)
    record = doc + list(qs[0])
    d = document_boundary(len(record), len(qs[0]))
    assert d == BODY
    for stage in (2, 3):
        record = (record[:rewind_target(d, qc)]
                  + continuation_tail(qs[stage - 1], qc))
        assert record == doc + qs[stage - 1]


# ---- the decisive-token gate ------------------------------------------

def test_decisive_token_in_later_position():
    """A fused "=YES"-style token may arrive after restated chatter.
    With a no set registered, the chatter keeps decoding and the first
    decisive token ends the stage before the window is exhausted."""
    yes, no = {YES_TOK}, {NO_TOK}
    window = [900, 901, YES_TOK]
    seen = [gate_decision(t, False, yes, no) for t in window]
    assert seen == [KEEP_DECODING, KEEP_DECODING, PASSED]
    assert stage_advances(seen[-1], 1, 3)
    # a decisive no ends the stage the same way, without advancing
    assert gate_decision(NO_TOK, False, yes, no) == FAILED
    assert not stage_advances(FAILED, 1, 3)


def test_gate_without_no_set_waits_for_the_stop():
    """Single-token mode (no no set): nothing ends the window early,
    and the engine's own stop decides - any non-yes token fails."""
    yes = {YES_TOK}
    assert gate_decision(YES_TOK, False, yes, set()) == KEEP_DECODING
    assert gate_decision(YES_TOK, True, yes, set()) == PASSED
    assert gate_decision(777, True, yes, set()) == FAILED
    assert gate_decision(None, True, yes, set()) == FAILED


def test_gate_window_exhausted_without_decision():
    """With a no set, a stop on a non-decisive token is undecided; a
    decisive token at the cap still counts."""
    yes, no = {YES_TOK}, {NO_TOK}
    assert gate_decision(900, True, yes, no) == UNDECIDED
    assert gate_decision(None, True, yes, no) == UNDECIDED
    assert gate_decision(YES_TOK, True, yes, no) == PASSED
    assert gate_decision(NO_TOK, True, yes, no) == FAILED
    for verdict in (UNDECIDED, FAILED, KEEP_DECODING):
        assert not stage_advances(verdict, 1, 3)


def test_final_stage_never_advances():
    """The last stage keeps its real finish so the client's stream
    closes; only a pass with stages remaining rewinds."""
    assert stage_advances(PASSED, 1, 3)
    assert stage_advances(PASSED, 2, 3)
    assert not stage_advances(PASSED, 3, 3)


def _walk_chain(doc, qs, flags_row):
    """Drive one document's chain from planted flags: judge each
    stage's token, rewind, append the next tail. Returns the per-stage
    answers and the rewind count."""
    qc = shared_preamble(qs)
    record = doc + list(qs[0])
    d = document_boundary(len(record), len(qs[0]))
    stage, answers, rewinds = 1, [], 0
    while True:
        assert record == doc + qs[stage - 1]
        tok = YES_TOK if flags_row[stage - 1] else NO_TOK
        verdict = gate_decision(tok, True, {YES_TOK}, set())
        answers.append(verdict == PASSED)
        if not stage_advances(verdict, stage, len(qs)):
            return answers, rewinds
        record = (record[:rewind_target(d, qc)]
                  + continuation_tail(qs[stage], qc))
        rewinds += 1
        stage += 1


def test_chain_walk_from_planted_flags():
    """The pieces together reproduce chain semantics: stop at the
    first failure, survive only on all-yes, one rewind per advance
    (n-1 rewinds for a surviving n-stage chain)."""
    doc = [10] * BODY
    qs = _questions()
    flags = [[1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]]
    want_stages = [3, 2, 1, 3]
    for row, stages in zip(flags, want_stages):
        answers, rewinds = _walk_chain(doc, qs, row)
        assert len(answers) == stages       # never asked past a failure
        assert answers == [bool(f) for f in row[:stages]]
        assert rewinds == stages - 1
        survived = all(answers) and len(answers) == len(qs)
        assert survived == all(row)


# ---- the strict-mode invariant ----------------------------------------

def test_strict_mode_predicates():
    """Any eviction the plan did not initiate is heuristic; in strict
    mode it must raise, and plan-initiated evictions never count."""
    assert is_heuristic_eviction(True, False)
    assert not is_heuristic_eviction(True, True)    # plan-initiated
    assert not is_heuristic_eviction(False, False)  # nothing evicted
    assert strict_violation(True, False, True)
    assert not strict_violation(True, True, True)
    assert not strict_violation(True, False, False)  # shared card mode
    assert not strict_violation(False, False, True)


# ---- the step trace ---------------------------------------------------

def test_step_record_reduces_a_step_to_plain_numbers():
    """The trace record: token totals, the prefill/decode split,
    unique documents (registrations carry no d part and do not
    count), pool occupancy in tokens, and the packing CPU in
    milliseconds. Without the generating set the split is the
    one-token heuristic."""
    toks = {"de1|c|d5|q-5-0": 340,      # prefill chunk of doc 5
            "de1|c|d7|q-7-0": 1,        # doc 7: one scheduled token
            "de1|reg|Y9|q-reg": 8}      # registration: no document
    rec = chainlogic.step_record(
        10.5, 0.0012, toks, [chainlogic.doc_key(r) for r in toks],
        free_blocks=900, total_blocks=1000, block_size=16)
    assert rec["tokens"] == 349 and rec["seqs"] == 3
    assert rec["decode_seqs"] == 1 and rec["prefill_tokens"] == 348
    assert rec["unique_docs"] == 2
    assert rec["kv_used_tokens"] == 1600
    assert rec["kv_total_tokens"] == 16000
    assert rec["sched_ms"] == 1.2


def test_step_record_decode_split_is_exact_with_the_generating_set():
    """With the generating set the split stops guessing: a request is
    decode only if it already holds sampled output, so a one-token
    final prefill chunk counts as prefill and a one-token filter run
    traces zero decode."""
    toks = {"de1|c|d5|q-5-0": 1,        # one-token prefill remainder
            "de1|c|d7|q-7-0": 1}        # a decisive-token decode
    keys = [chainlogic.doc_key(r) for r in toks]
    rec = chainlogic.step_record(0.0, 0.0, toks, keys, 0, 10, 16,
                                 decoding={"de1|c|d7|q-7-0"})
    assert rec["decode_seqs"] == 1
    assert rec["prefill_tokens"] == 1   # the remainder chunk
    # a pure filter step: nothing generating yet -> zero decode
    rec = chainlogic.step_record(0.0, 0.0, toks, keys, 0, 10, 16,
                                 decoding=set())
    assert rec["decode_seqs"] == 0 and rec["prefill_tokens"] == 2


def test_step_record_shapes_are_per_request_pairs():
    """Calibration records each request's [new, cached] pair so a
    measured step can be checked against its designed composition;
    without the flag the record carries no shapes key at all."""
    toks = {"a": 32, "b": 512}
    keys = [None, None]
    rec = chainlogic.step_record(0.0, 0.0, toks, keys, 0, 10, 16,
                                 shapes=[[32, 4096], [512, 0]])
    assert rec["shapes"] == [[32, 4096], [512, 0]]
    rec = chainlogic.step_record(0.0, 0.0, toks, keys, 0, 10, 16)
    assert "shapes" not in rec


def test_request_shape_recovers_cached_from_advanced_count():
    """schedule() advances the computed count by the scheduled share
    before the record is built, so the cached context is the advanced
    count minus this step's tokens: a 32-token suffix over a 4,096
    cached document reads (32, 4128) there."""
    assert chainlogic.request_shape(32, 4128) == [32, 4096]
    assert chainlogic.request_shape(512, 512) == [512, 0]


def test_attach_step_timing_amends_in_place():
    rec = dict(t=1.0, sched_ms=0.3)
    out = chainlogic.attach_step_timing(rec, 0.0421, 0.0007)
    assert out is rec
    assert rec["exec_ms"] == 42.1
    assert rec["update_ms"] == 0.7


# ---- module purity ----------------------------------------------------

def test_chainlogic_imports_no_engine():
    """The whole point of the module: it must stay importable and
    testable with no vLLM, torch, or numpy installed."""
    with open(chainlogic.__file__) as f:
        src = f.read()
    assert re.search(r"^\s*(?:import|from)\s+(?:vllm|torch|numpy)\b",
                     src, re.M) is None
