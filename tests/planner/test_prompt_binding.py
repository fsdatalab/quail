"""The Rust prompt binder writes the same prompt text as Python's, for every layout.

tests/conftest.py binds every prompt a second time in Rust and fails
where the two differ; these calls reach each layout and refusal.
"""

import pytest

from quail.common.expressions import ColumnRef, CompileError, GetField
from quail.frontend.prompts import PromptBinder
from quail.specs import MODELS

pytest.importorskip("_quail_native")

REVIEW = ColumnRef("r", "reviews", "body")
ASPECT = ColumnRef("a", "aspects", "aspect")
FIELD = GetField(ColumnRef("d", "docs", "details"), "body")
LABELS = ("positive", "negative", "mixed")

# (kind, template, args, keyword arguments) per call shape
CALLS = {
    "filter": ("filter", "Is this about pacing? {0}", (REVIEW,), {}),
    "filter with a frame": ("filter", "q1: {0}\nIs it long?", (FIELD,), {}),
    "join": ("join", "Does {0} discuss {1}?", (REVIEW, ASPECT), {}),
    "score": ("score", "{0} is positive", (REVIEW,), {}),
    "classify": ("label", "Sentiment of {0}?", (REVIEW,),
                 {"labels": LABELS, "descriptions": ("good", "", "both")}),
    "classify with a task": ("label", "{0}", (REVIEW,),
                             {"labels": ("a", "b"), "task_description": "Be brief."}),
    "classify a pair": ("label", "How does {0} treat {1}?", (REVIEW, ASPECT),
                        {"labels": LABELS}),
    "extract": ("extract", "When does it end?", (REVIEW,), {}),
    "map": ("map", "Summarize.", (REVIEW,), {}),
    "features": ("features", "{0} What is this about?", (REVIEW,), {}),
}

REFUSED = {
    "a missing placeholder": ("filter", "q1: {1}", (REVIEW,), {}),
    "a stray brace": ("filter", "a {b} {0}", (REVIEW,), {}),
    "a join over one table": ("join", "{0} {1}", (REVIEW, REVIEW), {}),
    "an empty question": ("extract", "  ", (REVIEW,), {}),
    "a map instruction with a brace": ("map", "Use {x}", (REVIEW,), {}),
}

LAYOUT_MODELS = ("qwen3.8-27b-fp8", "decision-2.0-kai-0.6b-bf16", "d1-2b-fp8")


@pytest.mark.parametrize("model", LAYOUT_MODELS)
@pytest.mark.parametrize("name", CALLS)
def test_each_prompt_binds_the_same_in_rust(model, name):
    kind, template, args, options = CALLS[name]
    prompt = PromptBinder(model=MODELS[model]).bind(kind, template, args, **options)
    assert prompt.args == args


@pytest.mark.parametrize("model", LAYOUT_MODELS)
@pytest.mark.parametrize("name", REFUSED)
def test_each_prompt_refusal_reads_the_same_in_rust(model, name):
    kind, template, args, options = REFUSED[name]
    with pytest.raises(CompileError):
        PromptBinder(model=MODELS[model]).bind(kind, template, args, **options)
