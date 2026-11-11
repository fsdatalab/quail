"""Each prompt binder binds its operator's prompt in every layout, or refuses."""

import pytest

from quail.api.prompts import (
    bind_classify_prompt,
    bind_extract_prompt,
    bind_join_prompt,
    bind_map_prompt,
    bind_prompt,
    bind_sae_features_prompt,
    bind_score_prompt,
)
from quail.common.expressions import ColumnRef, CompileError, GetField
from quail.specs import MODELS

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

def _bind(model, kind, template, args, *, labels=(), descriptions=(),
          task_description=""):
    turn, layout = model.turn, model.prompt_layout
    if kind == "extract":
        return bind_extract_prompt(args, template, turn, layout)
    if kind == "map":
        return bind_map_prompt(args, template, turn)
    if kind == "features":
        return bind_sae_features_prompt(template, args, turn)
    if kind == "label":
        return bind_classify_prompt(template, args, labels, descriptions, turn,
                                    task_description=task_description, layout=layout)
    if kind == "score":
        return bind_score_prompt(template, args)
    binder = {"filter": bind_prompt, "join": bind_join_prompt}[kind]
    return binder(template, args, turn=turn, layout=layout)


LAYOUT_MODELS = ("qwen3.8-27b-fp8", "decision-2.0-kai-0.6b-bf16", "d1-2b-fp8")


def test_each_prompt_binds_in_each_layout():
    for model in LAYOUT_MODELS:
        for name, (kind, template, args, options) in CALLS.items():
            prompt = _bind(MODELS[model], kind, template, args, **options)
            assert prompt.args == args, (model, name)


def test_a_malformed_prompt_is_refused_in_each_layout():
    for model in LAYOUT_MODELS:
        for name, (kind, template, args, options) in REFUSED.items():
            with pytest.raises(CompileError):
                _bind(MODELS[model], kind, template, args, **options)
                pytest.fail(f"{name} was bound on {model}")
