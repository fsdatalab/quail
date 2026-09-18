"""CPU checks for the exact prompt text."""

import pytest

from quail_b.prompts import DISCUSS_ASPECT, F1
from quail_b.rendering import (
    ANSWER_CUE,
    SHARED_PRE,
    render_filter_prompt,
    render_join_prompt,
    true_false_ids,
)


def test_filter_prompt_puts_the_document_first_and_the_question_after():
    text = render_filter_prompt(F1, "the review")
    lead, question = F1.split("{0}")
    assert text.startswith(SHARED_PRE + "the review")
    assert "You are performing a data processing task." in text
    assert "Evaluate TRUE or FALSE for the following question: " in text
    assert question.strip() in text
    # the template text before the placeholder moves after the document
    assert text.index("the review") < text.index(lead.strip())
    assert text.endswith(ANSWER_CUE)


def test_join_prompt_frames_the_anchor_and_labels_the_partners():
    text = render_join_prompt(DISCUSS_ASPECT, ("review", "aspect"), anchor=0)
    assert text.startswith(SHARED_PRE + "review")
    assert "You are performing a data processing task." in text
    assert "(The document above is DOCUMENT {0}.)" in text
    assert "\n\nDOCUMENT {1}:\naspect" in text
    assert text.endswith(ANSWER_CUE)
    assert render_join_prompt(DISCUSS_ASPECT, ("review", "aspect"), anchor=1
                              ).startswith(SHARED_PRE + "aspect")


def test_rendering_rejects_the_wrong_placeholder_count():
    with pytest.raises(ValueError, match="placeholders"):
        render_filter_prompt(DISCUSS_ASPECT, "one document")
    with pytest.raises(ValueError, match="placeholders"):
        render_join_prompt(F1, ("a", "b"))


def test_true_false_ids_take_the_first_token_of_each_spelling():
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(text.strip()[0]), 0]}

    assert true_false_ids(Tokenizer()) == ({ord("T")}, {ord("F")})
