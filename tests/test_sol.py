"""The speed-of-light bound, checked by hand where possible.

Most of these compute a small case on paper and assert the code
agrees. The point of `quail/sol.py` is that this is possible.
"""

import json
import math
from pathlib import Path

import pytest

from quail.sol import (Work, ask, cheaper_anchor, dense_params,
                       filter_chain, flops_per_pair, join,
                       kv_bytes_per_token, scan, seconds, stream,
                       survivors, triangle)
from quail.specs import H100_SXM, QWEN3_4B_FP8, QWEN3_32B_FP8

ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((ROOT / "results" / "sol_quailb_sf0.1.json").read_text())
INPUTS = RESULTS["measured_inputs"]
CHUNK_4B = 110_376


# ---- the model constants

def test_dense_params_matches_the_published_counts():
    """Qwen3-4B is 4.02B total and 3.63B non-embedding; Qwen3-32B is
    32.8B total and 31.2B non-embedding. Vocab is 151,936."""
    assert dense_params(QWEN3_4B_FP8) == 3_633_511_936
    assert (dense_params(QWEN3_4B_FP8) + 151_936 * QWEN3_4B_FP8.hidden
            == 4_022_468_096)
    assert dense_params(QWEN3_32B_FP8) == 31_206_298_624


def test_spec_params_is_rounded_and_the_bound_does_not_use_it():
    """3.6e9 is 0.93% below the real count, and that lands straight on
    T_dense, the largest term."""
    assert dense_params(QWEN3_4B_FP8) / QWEN3_4B_FP8.params == (
        pytest.approx(1.0093, abs=5e-5))


def test_flops_per_pair_and_kv_bytes():
    assert flops_per_pair(QWEN3_4B_FP8) == 4 * 32 * 128 == 16_384
    assert kv_bytes_per_token(QWEN3_4B_FP8) == 2 * 36 * 8 * 128 * 2
    assert kv_bytes_per_token(QWEN3_4B_FP8) == 147_456


# ---- the three operations, on paper

def test_triangle():
    assert triangle(4) == 1 + 2 + 3 + 4


def test_scan_is_one_causal_triangle_and_pays_for_the_document():
    """A 100-token prefix with a 10-token question: 110 tokens
    computed, 110 * 111 / 2 pairs, nothing read back."""
    w = scan(100, 10)
    assert w.tokens == 110
    assert w.pairs == triangle(110)
    assert w.kv_written == 110
    assert w.kv_read == 0


def test_ask_does_not_pay_for_the_document_again():
    """The whole point of KV rewind. A second question over the same
    100-token prefix computes 10 tokens, not 110."""
    w = ask(100, 10)
    assert w.tokens == 10
    assert w.pairs == 10 * 100 + triangle(10)
    assert w.kv_written == 10
    assert w.kv_read == 100


def test_ask_is_cheaper_than_scan_by_exactly_the_document():
    assert scan(100, 10).tokens - ask(100, 10).tokens == 100


def test_stream_is_ask_repeated_but_reads_the_prefix_once():
    """Three tuples against one anchor: three rectangles and three
    triangles, one prefix read."""
    w = stream(100, [7, 7, 7])
    assert w.tokens == 21
    assert w.pairs == 3 * (7 * 100 + triangle(7))
    assert w.kv_read == 100
    assert w.kv_read == ask(100, 7).kv_read      # once, not three times


def test_work_adds_and_scales():
    assert (Work(1, 2, 3, 4) + Work(10, 20, 30, 40)).pairs == 22
    assert (Work(1, 2, 3, 4) * 3).tokens == 3


# ---- survivors

def test_survivors_keeps_the_right_count():
    assert len(survivors(list(range(100)), 0.25)) == 25
    assert survivors([1, 2, 3], 0.0) == []
    assert sorted(survivors([1, 2, 3], 1.0)) == [1, 2, 3]


def test_survivors_preserve_the_length_distribution():
    """The assumption, made visible: an even stride over the sorted
    lengths, so the survivors are neither longer nor shorter than the
    pool they came from."""
    pool = list(range(1, 1001))
    kept = survivors(pool, 0.5)
    assert abs(sum(kept) / len(kept) - sum(pool) / len(pool)) < 5


# ---- filter chains

def test_one_filter_over_two_documents():
    """By hand: prefixes 2 + 10 and 2 + 20, question 5.
    Tokens 17 + 27 = 44. Pairs triangle(17) + triangle(27)."""
    w = filter_chain([10, 20], preamble=2, questions=[5],
                     selectivities=[1.0])
    assert w.tokens == 44
    assert w.pairs == triangle(17) + triangle(27)
    assert w.kv_read == 0


def test_a_second_filter_costs_only_its_question():
    """Two documents, both surviving. Stage 2 adds 2 * 3 tokens and
    nothing else - the documents are not rescanned."""
    one = filter_chain([10, 20], 2, [5], [1.0])
    two = filter_chain([10, 20], 2, [5, 3], [1.0, 1.0])
    assert two.tokens - one.tokens == 2 * 3
    assert two.kv_read == 12 + 22          # both prefixes read once


def test_selectivity_thins_the_later_stages():
    half = filter_chain([10, 20, 30, 40], 2, [5, 3], [0.5, 1.0])
    full = filter_chain([10, 20, 30, 40], 2, [5, 3], [1.0, 1.0])
    assert full.tokens - half.tokens == 2 * 3   # two fewer documents


def test_five_filters_cost_barely_more_than_one():
    """The claim KV rewind is for. Five questions of 50 tokens over
    300-token documents add well under half again, where rescanning
    would be five times."""
    docs = [300] * 100
    one = filter_chain(docs, 2, [50], [1.0])
    five = filter_chain(docs, 2, [50] * 5, [1.0] * 5)
    assert five.tokens / one.tokens < 1.7
    rescan = sum(scan(302, 50).tokens for _ in docs) * 5
    assert rescan / one.tokens == pytest.approx(5.0)


# ---- joins

def test_join_streams_every_partner_past_every_anchor():
    """Two anchors, three partners, so six tuples. Each tuple's
    suffix is label + partner + question."""
    w = join([100, 100], [7, 7, 7], preamble=2, note=4, label=3,
             question=9, anchor_resident=True)
    suffix = 3 + 7 + 9
    assert w.tokens == 2 * (4 + 3 * suffix)


def test_a_resident_anchor_only_pays_the_naming_line():
    fresh = join([100], [7], 2, note=4, label=3, question=9,
                 anchor_resident=False)
    resident = join([100], [7], 2, note=4, label=3, question=9,
                    anchor_resident=True)
    assert fresh.tokens - resident.tokens == 102     # preamble + doc


def test_the_cheaper_anchor_holds_the_long_side():
    """Anchoring the short side copies every long document into every
    tuple. With two long and three short documents, holding the long
    side wins."""
    _, pick, both = cheaper_anchor([1000, 1000], [5, 5, 5], preamble=2,
                                   note=4, label=3, question=9,
                                   left_resident=False,
                                   right_resident=False)
    assert pick == "left"
    # 2 * (1006 + 3 * 17) = 2114 holding the long side, against
    # 3 * (11 + 2 * 1012) = 6105 holding the short one
    assert both["left"] == 2114 and both["right"] == 6105


def test_anchor_choice_flips_when_the_sides_swap():
    _, pick, _ = cheaper_anchor([5, 5, 5], [1000, 1000], 2, 4, 3, 9,
                                False, False)
    assert pick == "right"


def test_empty_side_is_no_work():
    w, _, _ = cheaper_anchor([], [1, 2], 2, 4, 3, 9, False, False)
    assert w == Work()


# ---- seconds

def test_seconds_is_the_hand_formula():
    w = Work(tokens=1_000_000, pairs=1e9, kv_written=1_000_000,
             kv_read=0)
    s = seconds(w, QWEN3_4B_FP8, H100_SXM, CHUNK_4B)
    assert s.dense == pytest.approx(2 * 3_633_511_936 * 1e6 / 1.979e15)
    assert s.attention == pytest.approx(16_384 * 1e9 * 36 / 0.9895e15)
    assert s.passes == math.ceil(1e6 / CHUNK_4B)
    assert s.bytes_moved == pytest.approx(4.5e9 * s.passes + 147_456 * 1e6)
    assert s.sol == max(s.compute, s.memory)


def test_attention_prices_against_the_bf16_peak():
    assert H100_SXM.attn_flops == 0.9895e15 == H100_SXM.peak_flops / 2


def test_chunk_tokens_must_be_given_and_positive():
    w = Work(tokens=10)
    with pytest.raises(TypeError):
        seconds(w, QWEN3_4B_FP8, H100_SXM)
    with pytest.raises(ValueError):
        seconds(w, QWEN3_4B_FP8, H100_SXM, 0)


def test_more_passes_move_more_weight_bytes():
    w = Work(tokens=1e6, kv_written=1e6)
    few = seconds(w, QWEN3_4B_FP8, H100_SXM, 10 ** 9)
    many = seconds(w, QWEN3_4B_FP8, H100_SXM, 1000)
    assert many.bytes_moved > few.bytes_moved
    assert many.dense == pytest.approx(few.dense)   # compute unchanged


# ---- the committed QUAIL-B table

def test_inputs_carry_all_three_measured_things():
    assert INPUTS["preamble_tokens"] == 2
    assert len(INPUTS["selectivities"]) == 26
    assert sum(INPUTS["document_lengths"]["reviews.body"].values()) == 5000
    for q in INPUTS["selectivities"].values():
        for st in q["filters"]:
            assert 0.0 <= st["selectivity"] <= 1.0
            assert st["question_tokens"] > 0


def test_document_lengths_reproduce_the_corpus_totals():
    """The histogram is an aggregate but a lossless one."""
    reviews = INPUTS["document_lengths"]["reviews.body"]
    assert sum(int(n) * k for n, k in reviews.items()) == 1_494_233
    reports = INPUTS["document_lengths"]["reports.report"]
    assert sum(int(n) * k for n, k in reports.items()) == 829_199


def test_every_query_has_a_bound_on_both_models():
    assert len(RESULTS["queries"]) == 26
    for qid, r in RESULTS["queries"].items():
        for m in ("qwen3-4b-fp8", "qwen3-32b-fp8"):
            assert r["models"][m]["sol_s"] > 0, qid
            assert r["models"][m]["bound_by"] == "compute", qid


def test_the_single_filter_query_pushes_the_whole_corpus_once():
    """IMDB-1 scans 5,000 reviews: 1,494,233 document tokens plus 2
    preamble and 54 question tokens each."""
    assert RESULTS["queries"]["IMDB-1"]["tokens"] == (
        1_494_233 + 5000 * (2 + 54))


def test_a_second_filter_only_adds_its_question():
    """KV reuse, read straight off the committed table. IMDB-6 is
    IMDB-1 plus one 48-token question per surviving review; the
    5,000 documents are not scanned again."""
    added = (RESULTS["queries"]["IMDB-6"]["tokens"]
             - RESULTS["queries"]["IMDB-1"]["tokens"])
    q_f4 = INPUTS["filter_question_tokens"]["F4"]
    survivors_after_f1 = (
        INPUTS["selectivities"]["IMDB-6"]["filters"][1]["evaluated"])
    assert added == survivors_after_f1 * q_f4 == 3873 * 48


def test_32b_costs_more_but_not_uniformly():
    """The dense term scales 8.59x with the parameter counts, the
    attention term only 3.56x with 4 n_q d_head L. A query whose
    attention share is large therefore scales by less, and BioDEX's
    4,146-token reports are that query."""
    def ratio(q):
        m = RESULTS["queries"][q]["models"]
        return m["qwen3-32b-fp8"]["sol_s"] / m["qwen3-4b-fp8"]["sol_s"]

    def attention_share(q):
        m = RESULTS["queries"][q]["models"]["qwen3-4b-fp8"]
        return m["t_attention"] / m["t_compute"]

    assert attention_share("BIO-2") > 0.35
    assert attention_share("IMDB-2") < 0.10
    assert ratio("BIO-2") < ratio("IMDB-2") < 8.59
    assert dense_params(QWEN3_32B_FP8) / dense_params(QWEN3_4B_FP8) == (
        pytest.approx(8.59, abs=0.01))


def test_fever_anchors_on_evidence_not_claims():
    """Claims average 11 tokens and evidence 370, so the join holds
    evidence and streams claims - the opposite of how the query is
    written."""
    for qid in ("FEV-2", "FEV-3", "FEV-5"):
        assert RESULTS["queries"][qid]["anchor"] == "right", qid
    for qid in ("IMDB-2", "BIO-2", "LEP-2"):
        assert RESULTS["queries"][qid]["anchor"] == "left", qid


def test_choosing_the_wrong_anchor_would_cost_multiples():
    both = RESULTS["queries"]["BIO-2"]["anchor_tokens_both_ways"]
    assert both["right"] / both["left"] > 40
    both = RESULTS["queries"]["FEV-5"]["anchor_tokens_both_ways"]
    assert both["left"] / both["right"] > 5
