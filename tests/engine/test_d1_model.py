"""Liquid's d1 decision models: prompt layout, answer ids, spec, and support."""

from types import SimpleNamespace

from quail.baselines.request import RequestBackend
from quail.logical import ColumnRef, bind_prompt
from quail.logical.prompts import bind_join_prompt, render_join_prompt_text
from quail.planner.prompt_lowering import answer_ids
from quail.specs import D1_2B_BF16, H100_SXM

REVIEW = ColumnRef(alias="r", provider="reviews", column="body")
MOVIE = ColumnRef(alias="m", provider="movies", column="plot")
# d1-2B's prompt.render for a text state and a yes/no question, with the
# chat template's empty thinking block as the lead
D1_PROMPT = ("<|startoftext|><|im_start|>user\n{state}\n\n\nQUESTION:\n"
             "{question}\n\nReply with yes or no only.<|im_end|>\n"
             "<|im_start|>assistant\n<think></think>")
# text_config.layer_types of LiquidAI/d1-2B at the spec's revision
CHECKPOINT_LAYER_TYPES = (
    "conv", "conv", "full_attention", "conv", "sliding_attention", "conv",
    "full_attention", "conv", "sliding_attention", "conv", "full_attention",
    "conv", "sliding_attention", "conv", "full_attention", "conv",
    "sliding_attention", "conv", "full_attention", "conv",
    "sliding_attention", "conv", "full_attention", "conv")


def test_a_d1_filter_prompt_is_the_text_d1_was_trained_on():
    prompt = bind_prompt("Does the review praise the acting? {0}", (REVIEW,),
                         turn=D1_2B_BF16.turn, layout=D1_2B_BF16.prompt_layout)
    document = "The lead is superb."

    text = prompt.preamble + document + prompt.tail[len("{0}"):]

    assert text == D1_PROMPT.format(
        state=document, question="Does the review praise the acting?")


def test_a_d1_join_asks_its_question_after_both_documents():
    prompt = bind_join_prompt("Is {0} a review of the movie in {1}?",
                              (REVIEW, MOVIE), turn=D1_2B_BF16.turn,
                              layout=D1_2B_BF16.prompt_layout)

    text = render_join_prompt_text(prompt, ["Great film.", "A heist."], 0)

    assert prompt.frame == ""
    assert text == D1_PROMPT.format(
        state=("Great film.\n\n(The document above is DOCUMENT {0}.)"
               "\n\nDOCUMENT {1}:\nA heist."),
        question="Is {0} a review of the movie in {1}?")


def test_d1_answers_compare_its_one_token_yes_spellings_with_its_no_spellings():
    vocabulary = {"yes": [1], "Yes": [2], "YES": [3, 4], "no": [5],
                  "No": [6], "NO": [7]}

    true_ids, false_ids = answer_ids(D1_2B_BF16, vocabulary.__getitem__)

    assert (true_ids, false_ids) == ([1, 2], [5, 6, 7])


def test_the_d1_2b_spec_has_the_checkpoints_layer_kinds():
    expected = tuple("short_conv" if kind == "conv" else kind
                     for kind in CHECKPOINT_LAYER_TYPES)

    assert D1_2B_BF16.layer_kinds == expected
    assert D1_2B_BF16.sliding_layer_set == {4, 8, 12, 16, 20}


def test_the_vllm_baseline_refuses_d1_2b():
    engine = SimpleNamespace(label="stock vLLM", kind="vllm")
    backend = RequestBackend(name="stock", engine=engine,
                             filter_submission="operator")

    support = backend.supports(D1_2B_BF16, H100_SXM, 1)

    assert not support.supported
    assert "lfm3" in support.reason


def test_d1_2b_refuses_classify_and_extract_with_a_reason():
    from quail.planner.validation import (
        classification_refusal,
        extraction_refusal,
    )

    context = SimpleNamespace(model=D1_2B_BF16, tokenizer=lambda text: [1])

    classify = classification_refusal(context)
    extract = extraction_refusal(context)

    assert "option-code prompt" in classify.reasons[0]
    assert "document positions" in extract.reasons[0]


def test_a_hybrid_model_with_sliding_layers_gets_a_sliding_pool():
    from quail.cost import budgets

    full, sliding = budgets.arena_pages(D1_2B_BF16, H100_SXM)

    assert full > 0
    assert sliding == full
