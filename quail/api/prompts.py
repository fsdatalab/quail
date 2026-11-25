"""Bind an AI function's prompt with Rust's binder, as planning does.

Each function returns the ``Prompt`` that
``crates/quail-planner/src/prompts.rs`` binds for one call: the
preamble, the document labels, the question, and the answer cue in the
model's prompt layout.

For example, imagine a filter over a table of reviews:

    from quail.common.expressions import ColumnRef

    review = ColumnRef("r", "reviews", "text")
    prompt = bind_prompt("Is {0} negative?", (review,))
"""

import json

import _quail_native

from quail.common.expressions import CompileError, Prompt
from quail.physical import decode_plan_value

Turn = tuple[str, str]


def _bind(kind: str, template: str, args, *, turn: Turn = ("", ""),
          layout: str = "ai-if", labels=(), descriptions=(),
          task_description: str = "") -> Prompt:
    found, message = _quail_native.bind_prompt(
        kind, template, json.dumps([json.loads(arg.to_json()) for arg in args]),
        list(labels), list(descriptions), task_description, tuple(turn), layout)
    if message is not None:
        raise CompileError(message)
    return decode_plan_value(found)


def bind_prompt(template: str, args: tuple, turn: Turn = ("", ""),
                layout: str = "ai-if") -> Prompt:
    """Bind an AI.IF filter prompt.

    Args:
        template: The question, with {0}, {1}, ... for the arguments.
        args: The column each placeholder reads, in placeholder order.
        turn: The model's chat-turn text before the preamble and after
            the answer cue.
        layout: The name of the model's prompt layout.

    Raises:
        CompileError: The template and arguments do not fit.
    """
    return _bind("filter", template, args, turn=turn, layout=layout)


def bind_join_prompt(template: str, args: tuple, turn: Turn = ("", ""),
                     layout: str = "ai-if") -> Prompt:
    """Bind an AI join prompt over two or more tables' documents."""
    return _bind("join", template, args, turn=turn, layout=layout)


def bind_classify_prompt(template: str, args: tuple, labels, descriptions=(),
                         turn: Turn = ("", ""), task_description: str = "",
                         layout: str = "ai-if") -> Prompt:
    """Bind an AI.CLASSIFY prompt that answers with one of `labels`."""
    return _bind("label", template, args, turn=turn, layout=layout,
                 labels=labels, descriptions=descriptions,
                 task_description=task_description)


def bind_extract_prompt(args: tuple, question: str, turn: Turn = ("", ""),
                        layout: str = "ai-if") -> Prompt:
    """Bind an AI.EXTRACT prompt that answers `question` from the document."""
    return _bind("extract", question, args, turn=turn, layout=layout)


def bind_map_prompt(args: tuple, instruction: str,
                    turn: Turn = ("", "")) -> Prompt:
    """Bind an AI.MAP prompt that writes an answer to `instruction`."""
    return _bind("map", instruction, args, turn=turn)


def bind_score_prompt(template: str, args: tuple) -> Prompt:
    """Bind an AI.SCORE prompt."""
    return _bind("score", template, args)


def bind_sae_features_prompt(template: str, args: tuple,
                             turn: Turn = ("", "")) -> Prompt:
    """Bind an AI.SAE_FEATURES prompt."""
    return _bind("features", template, args, turn=turn)
