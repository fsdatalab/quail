"""Tests for reading Boolean answers from public request outputs."""

from types import SimpleNamespace

import pytest

from quail.backends.request_scheduling import canvas_answer, text_answer, true_bit


def _output(entries=None, *, tokens=(7,), text="TRUE"):
    return SimpleNamespace(outputs=[SimpleNamespace(
        token_ids=list(tokens), text=text,
        logprobs=None if entries is None else [entries])])


def _entry(score, rank=None, decoded=None):
    return SimpleNamespace(logprob=score, rank=rank, decoded_token=decoded)


def _canvas(entries, *, top_k=3):
    return canvas_answer(_output(entries), true_ids={7, 9}, false_ids={8, 10},
                         top_k=top_k)


def test_token_and_text_answers():
    assert true_bit(_output(tokens=(7,)), {7}) == 1
    assert true_bit(_output(tokens=(8,)), {7}) == 0
    with pytest.raises(ValueError, match="no answer token"):
        true_bit(_output(tokens=()), {7})
    assert text_answer(_output(text="The answer is TRUE.")) == 1
    assert text_answer(_output(text="TRUEISH **FALSE**")) == 0
    with pytest.raises(ValueError, match="no TRUE/FALSE"):
        text_answer(_output(text=""))


def test_canvas_uses_token_ids_and_scores_not_sampled_text():
    assert _canvas({7: _entry(-4, decoded="FALSE"),
                    8: _entry(-2, decoded="TRUE")}) == 0
    assert _canvas({7: _entry(-4), 9: _entry(-1), 8: _entry(-2)}) == 1


@pytest.mark.parametrize("tokens", [(7, 8), (8, 7)])
def test_canvas_ties_are_false_in_either_order(tokens):
    assert _canvas({t: _entry(-2) for t in tokens}) == 0


def test_canvas_can_rule_out_an_omitted_class():
    assert _canvas({7: _entry(-1, 1), 1: _entry(-2, 2),
                    2: _entry(-3, 3)}) == 1
    assert _canvas({1: _entry(-1, 1), 2: _entry(-2, 2),
                    8: _entry(-3, 3)}) == 0


@pytest.mark.parametrize("entries", [
    None,
    {},
    {1: _entry(-1, 1), 2: _entry(-2, 2), 3: _entry(-3, 3)},
    {1: _entry(-1, 1), 2: _entry(-2, 2), 7: _entry(-3, 3)},
    {1: _entry(-1, 1), 2: _entry(-2, 2), 7: _entry(-4, 100)},
    {7: _entry(-1, 1)},
    {11: _entry(-1, 1, "TRUE")},
])
def test_canvas_missing_scores_do_not_fall_back_to_true_text(entries):
    with pytest.raises(ValueError, match="insufficient TRUE/FALSE scores"):
        _canvas(entries)
