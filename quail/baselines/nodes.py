"""The physical node of the request backends: one engine run per query.

A request backend submits one prompt per document or pair to vLLM or
SGLang. Its whole model work is one ``RequestExecution`` node holding
the filter, join, classification, and extraction specs, in the order
the engine runs them. The Quail engine plans one node per operator
instead; see ``quail.physical.nodes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from quail.physical.base import (
    AnswerOutput,
    ExecutionLocation,
    OutputPort,
    PhysicalNode,
    Token,
    ValueType,
    validate_integer_setting,
)


@dataclass(frozen=True)
class RequestFilterSpec:
    """One filter sequence submitted as model requests."""

    alias: str
    written_positions: tuple[int, ...]
    question_token_ids: tuple[tuple[Token, ...], ...]
    question_texts: tuple[str, ...] = ()

@dataclass(frozen=True)
class RequestClassifySpec:
    """Specification for one decoded classification answer per document or row.

    The tail_token_ids follow the document and contain the question, the
    label list, and the answer cue. Tests contains (written position,
    accepted labels) pairs applied in order after classification.
    """

    alias: str
    output: str
    tail_token_ids: tuple[Token, ...]
    labels: tuple[str, ...]
    label_token_ids: tuple[tuple[Token, ...], ...]
    tests: tuple[tuple[int, tuple[str, ...]], ...] = ()
    # a classification of joined rows: the partner alias, the anchor
    # note and partner label token ids, and the join whose rows it labels
    partner: str | None = None
    join_layout_token_ids: tuple[tuple[Token, ...], ...] = ()
    join_written_position: int | None = None
    # a decision model's scored options: distances before the prompt's
    # last row of each option block's end, then 0
    option_offsets: tuple[int, ...] = ()

@dataclass(frozen=True)
class RequestJoinSpec:
    """One join predicate submitted over its complete cross product."""

    written_position: int
    aliases: tuple[str, ...]
    outer_aliases: tuple[str, ...]
    anchor: str | None
    semantics: str
    selectivity: float | None
    label_token_ids: tuple[tuple[str, tuple[Token, ...]], ...]
    frame_token_ids: tuple[tuple[str, tuple[Token, ...]], ...]
    tail_token_ids: tuple[Token, ...]
    label_texts: tuple[tuple[str, str], ...] = ()
    frame_texts: tuple[tuple[str, str], ...] = ()
    tail_text: str = ""

@dataclass(frozen=True)
class RequestAnswerSpec:
    """Specification for one decoded answer per document.

    The preamble precedes the document: the model's turn and the
    document label, which differ from the request's shared preamble on
    a model that answers AI.IF raw. The tail_token_ids follow the
    document and end in the cue that opens the answer.
    """

    alias: str
    output: str
    tail_token_ids: tuple[Token, ...]
    # the tail's text, for an engine that tokenizes prompts itself
    tail_text: str = ""
    preamble_token_ids: tuple[Token, ...] = ()
    preamble_text: str = ""
    max_answer_tokens: int = 64

    def __post_init__(self):
        validate_integer_setting("max_answer_tokens", self.max_answer_tokens)


@dataclass(frozen=True)
class RequestExtractSpec(RequestAnswerSpec):
    """An answer copied from the document, located in its text.

    The tail holds the question, the instruction, the answer template,
    and the phrase cue. The answer is the text the model writes up to
    the closing quote, located in the document's text.
    """

    question: str = ""

    @property
    def span_output(self) -> str:
        """Return the name of the span column."""
        return self.output + "_span"

def _short_text(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass(frozen=True)
class RequestMapSpec(RequestAnswerSpec):
    """An answer the model writes for an instruction: its text, up to its turn end."""


@dataclass(frozen=True)
class RequestExecution(PhysicalNode):
    """Run filters and joins through an independent request engine."""

    backend_name: str = ""
    aliases: tuple[str, ...] = ()
    preamble_token_ids: tuple[Token, ...] = ()
    preamble_text: str = ""
    filters: tuple[RequestFilterSpec, ...] = ()
    joins: tuple[RequestJoinSpec, ...] = ()
    classifies: tuple[RequestClassifySpec, ...] = ()
    extracts: tuple[RequestExtractSpec, ...] = ()
    maps: tuple[RequestMapSpec, ...] = ()

    type_name: ClassVar[str] = "quail.request_execution"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR

    @property
    def backend(self) -> str:
        """Return the backend selected for this model node."""
        return self.backend_name

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        outputs = [
            OutputPort(
                f"rows:{alias}",
                ValueType.ROWS,
                schema=(alias,),
            )
            for alias in self.aliases
        ]
        outputs.extend(
            OutputPort(
                f"filter_answers:{spec.alias}",
                ValueType.FILTER_ANSWERS,
                schema=(spec.alias, "predicate", "answer"),
            )
            for spec in self.filters
        )
        outputs.extend(
            OutputPort(
                f"join_answers:{spec.written_position}",
                ValueType.JOIN_ANSWERS,
                schema=(*spec.aliases, "answer"),
            )
            for spec in self.joins
        )
        outputs.extend(
            OutputPort(
                f"label_answers:{spec.output}",
                ValueType.LABEL_ANSWERS,
                schema=(*(alias for alias in (spec.alias, spec.partner) if alias),
                        spec.output),
                answer=AnswerOutput("classifies", spec.output,
                    (*(alias for alias in (spec.alias, spec.partner) if alias),
                     spec.output)),
            )
            for spec in self.classifies
        )
        # the answers of the filters on a label, one relation per tested classification
        outputs.extend(
            OutputPort(
                f"label_in_answers:{spec.output}",
                ValueType.FILTER_ANSWERS,
                schema=(spec.alias, "predicate", "answer"),
            )
            for spec in self.classifies if spec.tests
        )
        outputs.extend(
            OutputPort(
                f"extract_answers:{spec.output}",
                ValueType.EXTRACT_ANSWERS,
                schema=(spec.alias, spec.output, spec.span_output),
                answer=AnswerOutput("extracts", spec.output,
                                    (spec.alias, spec.output, spec.span_output)),
            )
            for spec in self.extracts
        )
        outputs.extend(
            OutputPort(
                f"map_answers:{spec.output}",
                ValueType.ROWS,
                schema=(spec.alias, spec.output),
                answer=AnswerOutput("maps", spec.output, (spec.alias, spec.output)),
            )
            for spec in self.maps
        )
        return tuple(outputs)

    def explain_fields(self) -> Mapping[str, Any]:
        fields: dict[str, Any] = {
            "backend": self.backend_name,
            "aliases": list(self.aliases),
        }
        for spec in self.filters:
            texts = spec.question_texts or tuple(
                f"predicate {position + 1}" for position in spec.written_positions)
            fields[f"filter {spec.alias}"] = [
                _short_text(text) for text in texts]
        for spec in self.joins:
            fields[f"join {spec.written_position + 1}"] = (
                f"{spec.semantics}: {_short_text(spec.tail_text)}")
        for spec in self.classifies:
            fields[f"classify {spec.output}"] = list(spec.labels)
        for spec in self.extracts:
            fields[f"extract {spec.output}"] = f"question={spec.question!r}"
        for spec in self.maps:
            fields[f"map {spec.output}"] = _short_text(spec.tail_text)
        return fields
