import itertools
import random

import pytest

from quail.executor.retention import Retained, minimum_loss_victims
from quail.planner.sol import prefix_recompute_seconds, triangle
from quail.specs import H100_SXM, QWEN3_4B_FP8, QWEN3_32B_FP8


def _complete(entries, need):
    choices = []
    for count in range(len(entries) + 1):
        for subset in itertools.combinations(entries, count):
            pages = sum(entry.pages for entry in subset)
            if pages >= need:
                choices.append((sum(entry.value for entry in subset),
                                pages, tuple(entry.key for entry in subset)))
    return min(choices, default=None)


def test_minimum_loss_victims_matches_complete_enumeration():
    rng = random.Random(7)
    for n in range(1, 9):
        for _ in range(30):
            entries = tuple(
                Retained(i, rng.randrange(1, 8), rng.randrange(1, 80))
                for i in range(n)
            )
            need = rng.randrange(1, 12)
            expected = _complete(entries, need)
            actual = minimum_loss_victims(entries, need)
            if expected is None:
                assert actual is None
            else:
                assert actual is not None
                assert actual.value == expected[0]
                assert actual.pages >= need


def test_minimum_loss_can_keep_two_smaller_documents():
    residents = (
        Retained("seven", 7, 49),
        Retained("four-a", 4, 30),
        Retained("four-b", 4, 30),
    )

    victims = minimum_loss_victims(residents, 7)

    assert victims is not None
    assert victims.keys == ("seven",)
    assert victims.value == 49


def test_prefix_value_uses_dense_tokens_and_attention_pairs():
    for model in (QWEN3_4B_FP8, QWEN3_32B_FP8):
        one = prefix_recompute_seconds(1, model, H100_SXM)
        two = prefix_recompute_seconds(2, model, H100_SXM)
        assert one > 0
        assert two > 2 * one
        assert triangle(2) == 3

    with pytest.raises(ValueError):
        prefix_recompute_seconds(-1, QWEN3_4B_FP8, H100_SXM)
