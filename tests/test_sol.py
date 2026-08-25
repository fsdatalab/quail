"""The SoL bound, checked against the hand derivation.

The two exercises are IMDB-1 and IMDB-6 over 5,000 IMDB reviews at
Qwen3-4B-fp8 on one H100. The derivation is `plans/sol_model.md`
section 7.
"""

import math

import pytest

from quail.sol import (Corpus, FilterStage, Workload, bound,
                       filter_chain_sol, filter_chain_workload)
from quail.specs import H100_SXM, QWEN3_4B_FP8

# The batch size the derivation ran the forward pass at. An input to
# the bound, not something it derives: it sets how many times the
# weights are re-read.
CHUNK = 110_376

# The corpus the hand derivation used, recovered from the two numbers
# it reports for stage 1: 1,774,233 sequence tokens and 444,634,043
# causal pairs over 5,000 documents, with F1's 54 question tokens
# already inside both. Inverting sum l(l+1)/2 gives the second moment,
# and shifting by q1 moves both moments off the sequence and onto the
# prefix (preamble + document), which is what Corpus holds.
N_DOCS = 5000
Q_F1 = 54
Q_F4 = 47
SEL_F1 = 0.8262
_SUM_L = 1_774_233
_SUM_L_SQ = 2 * 444_634_043 - _SUM_L
_SUM_B = _SUM_L - N_DOCS * Q_F1
IMDB = Corpus(n_docs=N_DOCS, sum_prefix=float(_SUM_B),
              sum_prefix_sq=float(_SUM_L_SQ - 2 * Q_F1 * _SUM_B
                                  - N_DOCS * Q_F1 ** 2))

F1 = FilterStage(question_tokens=Q_F1, selectivity=SEL_F1)
F4 = FilterStage(question_tokens=Q_F4)


def test_corpus_from_doc_tokens_matches_moments():
    c = Corpus.from_doc_tokens([10, 20, 30], preamble_tokens=3)
    assert c.n_docs == 3
    assert c.sum_prefix == 13 + 23 + 33
    assert c.sum_prefix_sq == 13 ** 2 + 23 ** 2 + 33 ** 2


def test_single_filter_workload():
    """IMDB-1: one filter, nothing rewound, nothing read back."""
    w = filter_chain_workload(IMDB, [F1])
    assert w.tokens == pytest.approx(1_774_233)
    assert w.pairs == pytest.approx(444_634_043)
    assert w.kv_read_tokens == 0.0


def test_single_filter_bound():
    b = filter_chain_sol(QWEN3_4B_FP8, H100_SXM, IMDB, [F1], CHUNK)
    assert b.passes == 17
    assert b.t_dense == pytest.approx(6.4550, abs=5e-4)
    assert b.t_attention == pytest.approx(0.2650, abs=5e-4)
    assert b.t_memory == pytest.approx(0.1009, abs=5e-4)
    assert b.seconds == pytest.approx(6.7200, abs=5e-4)
    assert b.bound_by == "compute"


def test_two_filter_workload_matches_derivation():
    """IMDB-6: F1 then F4. The derivation folded F1's question into
    the retained prefix, so this reproduces it with the same flag."""
    w = filter_chain_workload(IMDB, [F1, F4], carry_question_kv=True)
    assert w.tokens == pytest.approx(1_968_390)
    # 444,634,043 stage 1 + 68,895,951 streaming + 4,659,768 within
    # F4's own question. The derivation wrote 68,895,937 because it
    # rounded the surviving token count to a whole token first.
    assert w.pairs == pytest.approx(518_189_762, abs=1)
    assert w.kv_read_tokens == pytest.approx(1_465_871, abs=1)


def test_two_filter_bound():
    b = filter_chain_sol(QWEN3_4B_FP8, H100_SXM, IMDB, [F1, F4], CHUNK,
                         carry_question_kv=True)
    assert b.passes == 18
    assert b.t_dense == pytest.approx(7.1614, abs=5e-4)
    assert b.t_attention == pytest.approx(0.3089, abs=5e-4)
    assert b.t_memory == pytest.approx(0.1753, abs=5e-4)
    assert b.seconds == pytest.approx(7.4703, abs=5e-4)


def test_rewinding_the_question_kv_is_cheaper():
    """What the engine actually does: F1's question KV is dropped, so
    F4 reads back and attends over 270,000 fewer prefix tokens."""
    kept = filter_chain_workload(IMDB, [F1, F4], carry_question_kv=True)
    rewound = filter_chain_workload(IMDB, [F1, F4])
    assert rewound.tokens == kept.tokens          # same tokens computed
    assert rewound.kv_read_tokens == pytest.approx(
        kept.kv_read_tokens - SEL_F1 * N_DOCS * Q_F1)
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
