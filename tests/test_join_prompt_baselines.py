import itertools

import pytest

from baselines.stock import build_join_grouped_inputs
from baselines.old_stock.operators import Join
from quail.logical import (ColumnRef, bind_join_prompt,
                           render_join_prompt_ids)


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]


def _tok(tokenizer):
    return lambda text: tokenizer.encode(text, add_special_tokens=False)


@pytest.mark.parametrize("anchor", [0, 1])
def test_stock_and_naive_vllm_join_prompts_match_quail(anchor):
    tokenizer = CharTokenizer()
    tok = _tok(tokenizer)
    template = ("Does DOCUMENT {0} support the statement in "
                "DOCUMENT {1}?")
    left = ["left a", "left b"]
    right = ["right a", "right b", "right c"]
    args = (ColumnRef("left", "left", "document"),
            ColumnRef("right", "right", "document"))
    bound = bind_join_prompt(template, args, tok)
    documents = ([tok(text) for text in left],
                 [tok(text) for text in right])

    prefixes, suffixes, _ = build_join_grouped_inputs(
        bound, documents, anchor, tok)
    stock_prompts = [prefix + suffix
                     for prefix in prefixes for suffix in suffixes]
    naive_prompts, pairs = Join(
        "test", template, anchor=anchor).build_prompts(
            left, right, tokenizer)

    assert naive_prompts == stock_prompts
    assert len(pairs) == len(stock_prompts)
    for prompt_ids, (left_idx, right_idx) in zip(naive_prompts, pairs):
        expected = render_join_prompt_ids(
            bound,
            (documents[0][left_idx], documents[1][right_idx]),
            anchor,
            tok,
        )
        assert prompt_ids == expected


def test_three_way_stock_join_prompts_match_quail():
    tokenizer = CharTokenizer()
    tok = _tok(tokenizer)
    template = "Do DOCUMENT {0}, DOCUMENT {1}, and DOCUMENT {2} agree?"
    args = tuple(ColumnRef(f"d{i}", f"d{i}", "document")
                 for i in range(3))
    bound = bind_join_prompt(template, args, tok)
    documents = (
        [tok("a0"), tok("a1")],
        [tok("b0")],
        [tok("c0"), tok("c1")],
    )

    prefixes, suffixes, members = build_join_grouped_inputs(
        bound, documents, anchor=1, tokenizer=tok)
    partner_slots = [0, 2]
    for anchor_idx, prefix in enumerate(prefixes):
        for suffix, member in zip(suffixes, members):
            indices = [None, anchor_idx, None]
            for slot, document_idx in zip(partner_slots, member):
                indices[slot] = document_idx
            tuple_documents = tuple(
                documents[slot][indices[slot]] for slot in range(3))
            assert prefix + suffix == render_join_prompt_ids(
                bound, tuple_documents, anchor=1, tokenizer=tok)

    assert members == list(itertools.product(range(2), range(2)))
