"""Model-aware prompt binding for SQL and Python queries."""

from collections.abc import Callable
from dataclasses import dataclass

from quail.logical.expressions import Prompt
from quail.logical.prompts import (
    bind_classify_prompt,
    bind_extract_prompt,
    bind_join_prompt,
    bind_prompt,
    bind_score_prompt,
)
from quail.specs import ModelSpec


@dataclass(frozen=True)
class PromptBinder:
    """Bind operator prompts using one model and tokenizer.

    Without a model, turn and layout configure standalone frontend calls.
    """

    tokenizer: Callable | None = None
    model: ModelSpec | None = None
    turn: tuple[str, str] = ("", "")
    layout: str = "ai-if"

    def bind(self, kind: str, template: str, args: tuple, *, labels=(),
             descriptions=(), task_description: str = "") -> Prompt:
        """Bind and tokenize the prompt for an operator.

        Args:
            kind: Filter, join, score, label, or extract operator kind.
            template: Prompt template, or the extraction question.
            args: Document column references in template order.
            labels: Classification labels.
            descriptions: Optional descriptions in label order.
            task_description: Optional classification instruction.

        Returns:
            The bound prompt with its text and token IDs.

        Raises:
            ValueError: The operator kind is unknown.
        """
        turn = self.turn if self.model is None else self.model.turn
        layout = self.layout if self.model is None else self.model.prompt_layout
        if kind == "extract":
            if self.model is not None:
                turn = self.model.extract_turn
            return bind_extract_prompt(args, template, self.tokenizer, turn)
        if kind == "label":
            return bind_classify_prompt(
                template, args, labels, descriptions, self.tokenizer, turn,
                task_description=task_description, layout=layout)
        if kind == "filter":
            binder = bind_prompt
        elif kind == "join":
            binder = bind_join_prompt
        elif kind == "score":
            binder = bind_score_prompt
        else:
            raise ValueError(f"unknown prompt kind {kind!r}")
        return binder(template, args, self.tokenizer, turn=turn, layout=layout)
