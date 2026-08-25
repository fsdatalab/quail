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

from quail.sol import (Corpus, FilterStage, Workload, bound, dense_params,
                       filter_chain_sol, filter_chain_workload)
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
# Not measurable from anything committed here: it comes from a
# labelled run and is carried over from the derivation.
SEL_F1 = 0.8262
IMDB = Corpus(n_docs=N_DOCS,
              sum_prefix=float(_GT["prefix_tokens"]["sum"]),
              sum_prefix_sq=float(_GT["prefix_tokens"]["sum_sq"]))

F1 = FilterStage(question_tokens=Q_F1, selectivity=SEL_F1)
F4 = FilterStage(question_tokens=Q_F4)


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
    """IMDB-6: F1 then F4. F4's question is 48 tokens measured, where
    the derivation used 47, so the totals sit just above its
    1,968,390 tokens and 518,189,748 pairs."""
    w = filter_chain_workload(IMDB, [F1, F4], carry_question_kv=True)
    assert w.tokens == pytest.approx(1_972_521, abs=1)
    assert w.pairs == pytest.approx(519_853_922, abs=1)
    assert w.kv_read_tokens == pytest.approx(1_465_871, abs=1)


def test_two_filter_bound():
    b = filter_chain_sol(QWEN3_4B_FP8, H100_SXM, IMDB, [F1, F4], CHUNK,
                         carry_question_kv=True)
    assert b.passes == 18
    assert b.t_dense == pytest.approx(7.2432, abs=5e-4)
    assert b.t_attention == pytest.approx(0.3099, abs=5e-4)
    assert b.t_memory == pytest.approx(0.1755, abs=5e-4)
    assert b.seconds == pytest.approx(7.5531, abs=5e-4)


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
