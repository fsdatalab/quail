"""The SoL bound, checked against the measured corpus.

The two exercises are IMDB-1 and IMDB-6 over the 5,000 reviews
QUAIL-B builds at sf=0.1. The corpus numbers are measured, not taken
from the derivation: `results/sol_imdb_corpus.json` holds the token
moments produced by tokenizing the pinned IMDB revision with the
Qwen3-4B-FP8 tokenizer, so the token and pair totals below are a
real check rather than a restatement. The derivation is
`plans/sol_model.md` section 7.
"""

import json
import math
from pathlib import Path

import pytest

from quail.sol import (Corpus, FilterStage, JoinSide, JoinStage, Workload,
                       bound, dense_params, filter_chain_sol,
                       filter_chain_workload, join_workload)
from quail.specs import H100_SXM, QWEN3_4B_FP8, QWEN3_32B_FP8

# The batch size the derivation ran the forward pass at. An input to
# the bound, not something it derives: it sets how many times the
# weights are re-read.
CHUNK = 110_376

_GT = json.loads(
    (Path(__file__).resolve().parents[1]
     / "results" / "sol_imdb_corpus.json").read_text())

N_DOCS = _GT["corpus"]["n_docs"]
Q_F1 = _GT["question_tokens"]["F1"]
Q_F4 = _GT["question_tokens"]["F4"]
Q_F5 = _GT["question_tokens"]["F5"]
_S = _GT["survivors"]
IMDB = Corpus(n_docs=N_DOCS,
              sum_prefix=float(_GT["prefix_tokens"]["sum"]),
              sum_prefix_sq=float(_GT["prefix_tokens"]["sum_sq"]))

# Ground truth: the QUAIL-B label sets, counted per document. Each
# stage carries what actually survived it, so no selectivity scaling
# happens anywhere in the headline numbers.
F1 = FilterStage(question_tokens=Q_F1,
                 surviving_docs=_S["F1"]["surviving_docs"],
                 surviving_prefix=_S["F1"]["surviving_prefix"])
F4 = FilterStage(question_tokens=Q_F4,
                 surviving_docs=_S["F4_given_F1"]["surviving_docs"],
                 surviving_prefix=_S["F4_given_F1"]["surviving_prefix"])
F5 = FilterStage(question_tokens=Q_F5)
# The 4B engine run's own observed selectivity, kept only to
# reproduce the derivation: a 4B run's workload is set by the 4B
# model's answers, and 0.8262 is 4131 of 5000. Ground truth is the
# 32B model's 0.7746.
SEL_F1_4B_RUN = _GT["engine_run_4b"]["IMDB-1"]["observed_selectivity"]


def test_dense_params_matches_the_published_counts():
    """Qwen3-4B is 4.02B total and 3.63B non-embedding; Qwen3-32B is
    32.8B total and 31.2B non-embedding. The embedding is
    vocab x hidden at a 151,936 vocab."""
    emb = 151_936 * QWEN3_4B_FP8.hidden
    assert dense_params(QWEN3_4B_FP8) == 3_633_511_936
    assert dense_params(QWEN3_4B_FP8) + emb == 4_022_468_096
    assert dense_params(QWEN3_32B_FP8) == 31_206_298_624


def test_spec_params_is_the_rounded_stand_in():
    """The bound must not use it: at 4B it is 0.93% low, which lands
    straight on T_dense, the largest term."""
    assert dense_params(QWEN3_4B_FP8) / QWEN3_4B_FP8.params == (
        pytest.approx(1.0093, abs=5e-5))


def test_corpus_from_doc_tokens_matches_moments():
    c = Corpus.from_doc_tokens([10, 20, 30], preamble_tokens=3)
    assert c.n_docs == 3
    assert c.sum_prefix == 13 + 23 + 33
    assert c.sum_prefix_sq == 13 ** 2 + 23 ** 2 + 33 ** 2


def test_single_filter_workload():
    """IMDB-1: one filter, nothing rewound, nothing read back. Both
    totals come out of the measured length distribution."""
    w = filter_chain_workload(IMDB, [F1])
    assert w.tokens == _GT["stage1_totals"]["tokens"] == 1_774_233
    assert w.pairs == _GT["stage1_totals"]["causal_pairs"] == 444_634_043
    assert w.kv_read_tokens == 0.0


def test_single_filter_bound():
    b = filter_chain_sol(QWEN3_4B_FP8, H100_SXM, IMDB, [F1], CHUNK)
    assert b.passes == 17
    assert b.t_dense == pytest.approx(6.5151, abs=5e-4)
    assert b.t_attention == pytest.approx(0.2650, abs=5e-4)
    assert b.t_memory == pytest.approx(0.1009, abs=5e-4)
    assert b.seconds == pytest.approx(6.7801, abs=5e-4)
    assert b.bound_by == "compute"


def test_two_filter_workload():
    """IMDB-6: F1 then F4 on ground truth, with the engine's rewind.
    3,873 documents carrying 1,206,884 prefix tokens reach F4."""
    w = filter_chain_workload(IMDB, [F1, F4])
    assert w.tokens == pytest.approx(1_960_137, abs=1)
    assert w.pairs == pytest.approx(507_119_123, abs=1)
    assert w.kv_read_tokens == _S["F1"]["surviving_prefix"] == 1_206_884


def test_two_filter_bound():
    b = filter_chain_sol(QWEN3_4B_FP8, H100_SXM, IMDB, [F1, F4], CHUNK)
    assert b.passes == 18
    assert b.t_dense == pytest.approx(7.1978, abs=5e-4)
    assert b.t_attention == pytest.approx(0.3023, abs=5e-4)
    assert b.t_memory == pytest.approx(0.1636, abs=5e-4)
    assert b.seconds == pytest.approx(7.5000, abs=5e-4)


def test_three_filter_bound():
    """IMDB-7: F1 then F4 then F5, every survivor count counted."""
    b = filter_chain_sol(QWEN3_4B_FP8, H100_SXM, IMDB, [F1, F4, F5], CHUNK)
    assert b.passes == 19
    assert b.workload.tokens == pytest.approx(2_010_213, abs=1)
    assert b.workload.pairs == pytest.approx(528_044_209, abs=1)
    assert b.seconds == pytest.approx(7.6964, abs=5e-4)


def test_reproduces_the_derivation_with_its_own_inputs():
    """The derivation used the 4B run's selectivity and folded F1's
    question into the prefix. Both together give 7.5531 s, against
    its own 7.5366 s; the gap is F4's 48th token."""
    b = filter_chain_sol(
        QWEN3_4B_FP8, H100_SXM, IMDB,
        [FilterStage(Q_F1, SEL_F1_4B_RUN), FilterStage(Q_F4)], CHUNK,
        carry_question_kv=True)
    assert b.workload.tokens == pytest.approx(1_972_521, abs=1)
    assert b.seconds == pytest.approx(7.5531, abs=5e-4)


def test_rewinding_the_question_kv_is_cheaper():
    """What the engine actually does: F1's question KV is dropped, so
    F4 reads back and attends over 54 fewer tokens per survivor."""
    kept = filter_chain_workload(IMDB, [F1, F4], carry_question_kv=True)
    rewound = filter_chain_workload(IMDB, [F1, F4])
    assert rewound.tokens == kept.tokens          # same tokens computed
    assert rewound.kv_read_tokens == pytest.approx(
        kept.kv_read_tokens - _S["F1"]["surviving_docs"] * Q_F1)
    assert rewound.pairs < kept.pairs


def test_stage_one_pair_count_is_the_causal_triangle():
    """One document, one stage: the pair count is exactly l(l+1)/2."""
    c = Corpus(n_docs=1, sum_prefix=100.0, sum_prefix_sq=10_000.0)
    w = filter_chain_workload(c, [FilterStage(question_tokens=10)])
    assert w.tokens == 110
    assert w.pairs == 110 * 111 / 2


def test_later_stage_pairs_are_rectangle_plus_triangle():
    c = Corpus(n_docs=1, sum_prefix=100.0, sum_prefix_sq=10_000.0)
    stages = [FilterStage(question_tokens=10, selectivity=1.0),
              FilterStage(question_tokens=7)]
    w = filter_chain_workload(c, stages)
    stage2 = w.per_stage[1]
    assert stage2[0] == 7
    assert stage2[1] == 7 * 100 + 7 * 8 / 2
    assert stage2[2] == 100


def test_selectivity_thins_every_later_stage():
    c = Corpus(n_docs=1000, sum_prefix=100_000.0,
               sum_prefix_sq=10_000_000.0)
    half = [FilterStage(question_tokens=10, selectivity=0.5),
            FilterStage(question_tokens=10, selectivity=0.5),
            FilterStage(question_tokens=10)]
    w = filter_chain_workload(c, half)
    assert w.per_stage[1][0] == pytest.approx(0.5 * 1000 * 10)
    assert w.per_stage[2][0] == pytest.approx(0.25 * 1000 * 10)


def test_memory_bound_when_the_work_is_one_token_per_document():
    """Many short documents drive the ratio the other way: the weight
    reads per pass stop being amortized long before the FLOPs do."""
    c = Corpus.from_doc_tokens([1] * 8)
    b = filter_chain_sol(QWEN3_4B_FP8, H100_SXM, c,
                         [FilterStage(question_tokens=1)], CHUNK)
    assert b.bound_by == "memory"
    assert b.passes == 1


def test_bound_is_linear_in_a_pure_workload_scale():
    w1 = filter_chain_workload(IMDB, [F1])
    b1 = bound(QWEN3_4B_FP8, H100_SXM, w1, 10 ** 9)
    w2 = Workload(tokens=2 * w1.tokens, pairs=2 * w1.pairs,
                  kv_read_tokens=2 * w1.kv_read_tokens)
    b2 = bound(QWEN3_4B_FP8, H100_SXM, w2, 10 ** 9)
    assert b2.t_dense == pytest.approx(2 * b1.t_dense)
    assert b2.t_attention == pytest.approx(2 * b1.t_attention)


def test_passes_round_up():
    w = filter_chain_workload(IMDB, [F1])
    b = bound(QWEN3_4B_FP8, H100_SXM, w, CHUNK)
    assert b.passes == math.ceil(w.tokens / CHUNK)


def test_chunk_tokens_has_no_default_and_must_be_positive():
    w = filter_chain_workload(IMDB, [F1])
    with pytest.raises(TypeError):
        bound(QWEN3_4B_FP8, H100_SXM, w)
    with pytest.raises(ValueError):
        bound(QWEN3_4B_FP8, H100_SXM, w, 0)


def test_attention_prices_against_the_bf16_peak():
    assert H100_SXM.attn_flops == 0.9895e15
    assert H100_SXM.attn_flops == pytest.approx(H100_SXM.peak_flops / 2)


def test_empty_chain_refuses():
    with pytest.raises(ValueError):
        filter_chain_workload(IMDB, [])


def test_measured_survivors_override_the_selectivity_scaling():
    """With ground-truth labels a stage carries its surviving document
    count and prefix mass, and nothing is assumed."""
    gt = _GT["survivors"]["F1"]
    assumed = FilterStage(Q_F1, selectivity=gt["selectivity"])
    assert F1.measured and not assumed.measured
    q4 = FilterStage(Q_F4)
    wa = filter_chain_workload(IMDB, [assumed, q4])
    wm = filter_chain_workload(IMDB, [F1, q4])
    assert wa.kv_read_tokens == pytest.approx(
        gt["random_survivor_prefix"], abs=1)
    assert wm.kv_read_tokens == gt["surviving_prefix"]
    # same documents survive either way, so the token count matches;
    # only the mass they carry differs
    assert wa.tokens == pytest.approx(wm.tokens)
    assert wm.pairs > wa.pairs


def test_the_random_survivor_assumption_undercounts_but_barely_moves_sol():
    """Scaling by selectivity misses 3.5% of F1's surviving mass, and
    still lands within 0.1% on SoL: stage 1 owns the wall."""
    gt = _GT["survivors"]["F1"]
    assert gt["random_survivor_prefix"] / gt["surviving_prefix"] == (
        pytest.approx(0.9654, abs=5e-4))
    q4 = FilterStage(Q_F4)
    a = filter_chain_sol(QWEN3_4B_FP8, H100_SXM, IMDB,
                         [FilterStage(Q_F1, gt["selectivity"]), q4], CHUNK)
    m = filter_chain_sol(QWEN3_4B_FP8, H100_SXM, IMDB, [F1, q4], CHUNK)
    assert abs(m.seconds / a.seconds - 1) < 1e-3


def test_survivors_are_longer_than_the_pool_they_came_from():
    """The reason the assumption undercounts: these predicates prefer
    long reviews, and F4 prefers them hard."""
    for key in ("F1", "F4_given_F1", "F5_given_F1_F4"):
        gt = _GT["survivors"][key]
        assert gt["mean_prefix_survivors"] > gt["mean_prefix_pool"]
    f4 = _GT["survivors"]["F4_given_F1"]
    assert f4["mean_prefix_survivors"] / f4["mean_prefix_pool"] > 1.2


# ---- joins, and the whole QUAIL-B table

_QB = json.loads(
    (Path(__file__).resolve().parents[1]
     / "results" / "sol_quailb_sf0.1.json").read_text())


def test_join_tuple_count_is_the_cross_product():
    j = JoinStage(anchor=JoinSide.from_lengths([100, 200]),
                  suffix=JoinSide.from_lengths([10, 10, 10]))
    assert j.tuples == 6


def test_opening_join_pairs_are_the_anchor_triangle_plus_tuples():
    """One anchor of 100 tokens, no naming line, two partners of 10:
    the anchor is a full triangle, each suffix a rectangle over 100
    plus its own small triangle."""
    j = JoinStage(anchor=JoinSide.from_lengths([100]),
                  suffix=JoinSide.from_lengths([10, 10]),
                  note_tokens=0, opens_anchor=True)
    w = join_workload(j)
    assert w.tokens == 100 + 20
    assert w.pairs == 100 * 101 / 2 + 2 * (10 * 100 + 10 * 11 / 2)
    assert w.kv_read_tokens == 100


def test_resident_anchor_only_pays_the_naming_line():
    """After a filter the anchor prefixes are already computed, so the
    join adds the naming line and the tuple suffixes, nothing else."""
    opened = join_workload(JoinStage(
        anchor=JoinSide.from_lengths([100]),
        suffix=JoinSide.from_lengths([10]), note_tokens=4,
        opens_anchor=True))
    resident = join_workload(JoinStage(
        anchor=JoinSide.from_lengths([100]),
        suffix=JoinSide.from_lengths([10]), note_tokens=4,
        opens_anchor=False))
    assert opened.tokens - resident.tokens == 100
    assert resident.tokens == 4 + 10


def test_workloads_add():
    a = Workload(1.0, 2.0, 3.0)
    b = Workload(10.0, 20.0, 30.0)
    assert (a + b).tokens == 11.0
    assert (a + b).pairs == 22.0
    assert (a + b).kv_read_tokens == 33.0


def test_every_quailb_query_has_a_bound_for_both_models():
    assert len(_QB["queries"]) == 26
    for qid, r in _QB["queries"].items():
        for m in ("qwen3-4b-fp8", "qwen3-32b-fp8"):
            assert r["models"][m]["sol_s"] > 0, qid


def test_token_counts_match_the_engine_where_answers_cannot_differ():
    """IMDB-1, IMDB-2 and BIO-2 push a token count no model's answers
    can change - one filter over the whole corpus, or a join with no
    filter in front of it. Those must match the engine exactly. The
    other two ran on 4B's own answers, not the 32B ground truth."""
    v = _QB["validation"]["queries"]
    for qid in ("IMDB-1", "IMDB-2", "BIO-2"):
        assert v[qid]["ratio"] == 1.0, qid
    for qid in ("IMDB-5", "FEV-5"):
        assert 0.9 < v[qid]["ratio"] < 1.1, qid


def test_the_engine_runs_at_a_steady_fraction_of_the_floor():
    """Five measured walls, three corpora, filters and joins, 1.8 to
    140 seconds: all of them land between 0.41 and 0.50 of SoL."""
    f = [q["fraction_of_sol"] for q in _QB["validation"]["queries"].values()]
    assert min(f) > 0.41 and max(f) < 0.50


def test_32b_costs_more_but_not_uniformly():
    """The dense term scales 8.6x (parameter counts) and the attention
    term 3.6x (f_pair x L), so a query whose attention share is large
    scales by less. BIO's 4,146-token reports are that query."""
    def ratio(q):
        m = _QB["queries"][q]["models"]
        return m["qwen3-32b-fp8"]["sol_s"] / m["qwen3-4b-fp8"]["sol_s"]

    def attn_share(q):
        m = _QB["queries"][q]["models"]["qwen3-4b-fp8"]
        return m["t_attention"] / m["t_compute"]

    assert attn_share("BIO-2") > 0.35 and attn_share("IMDB-2") < 0.10
    assert ratio("BIO-2") < ratio("IMDB-2")
    assert 6.5 < ratio("BIO-2") < 6.7
    assert 8.3 < ratio("IMDB-2") < 8.4
