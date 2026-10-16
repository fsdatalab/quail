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
    ValueType,
    validate_integer_setting,
)


@dataclass(frozen=True)
class RequestFilterSpec:
    """One filter sequence submitted as model requests."""

    alias: str
    written_positions: tuple[int, ...]
    question_token_ids: tuple[tuple[Any, ...], ...]
    question_texts: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RequestFilterSpec":
        return cls(
            alias=str(value["alias"]),
            written_positions=tuple(
                int(position) for position in value["written_positions"]
            ),
            question_token_ids=tuple(
                tuple(question) for question in value["question_token_ids"]
            ),
            question_texts=tuple(value.get("question_texts", ())),
        )

    def to_dict(self) -> dict:
        return {
            "alias": self.alias,
            "written_positions": list(self.written_positions),
            "question_token_ids": [
                list(question) for question in self.question_token_ids
            ],
            "question_texts": list(self.question_texts),
        }


@dataclass(frozen=True)
class RequestClassifySpec:
    """Specification for one decoded classification answer per document or row.

    The tail_token_ids follow the document and contain the question, the
    label list, and the answer cue. Tests contains (written position,
    accepted labels) pairs applied in order after classification.
    """

    alias: str
    output: str
    tail_token_ids: tuple[Any, ...]
    labels: tuple[str, ...]
    label_token_ids: tuple[tuple[Any, ...], ...]
    tests: tuple[tuple[int, tuple[str, ...]], ...] = ()
    # a classification of joined rows: the partner alias, the anchor
    # note and partner label token ids, and the join whose rows it labels
    partner: str | None = None
    join_layout_token_ids: tuple[tuple[Any, ...], ...] = ()
    join_written_pos: int | None = None
    # a decision model's scored options: distances before the prompt's
    # last row of each option block's end, then 0
    option_offsets: tuple[int, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RequestClassifySpec":
        partner = value.get("partner")
        join_written_pos = value.get("join_written_pos")
        return cls(
            alias=str(value["alias"]),
            output=str(value["output"]),
            tail_token_ids=tuple(value["tail_token_ids"]),
            labels=tuple(str(label) for label in value["labels"]),
            label_token_ids=tuple(tuple(ids) for ids in value["label_token_ids"]),
            tests=tuple((int(position), tuple(str(label) for label in accepted))
                        for position, accepted in value.get("tests", ())),
            partner=None if partner is None else str(partner),
            join_layout_token_ids=tuple(tuple(ids)
                                 for ids in value.get("join_layout_token_ids", ())),
            join_written_pos=(None if join_written_pos is None
                              else int(join_written_pos)),
            option_offsets=tuple(int(offset)
                                 for offset in value.get("option_offsets", ())),
        )

    def to_dict(self) -> dict:
        return {
            "alias": self.alias,
            "output": self.output,
            "tail_token_ids": list(self.tail_token_ids),
            "labels": list(self.labels),
            "label_token_ids": [list(ids) for ids in self.label_token_ids],
            "tests": [[position, list(accepted)]
                      for position, accepted in self.tests],
            "partner": self.partner,
            "join_layout_token_ids": [list(ids) for ids in self.join_layout_token_ids],
            "join_written_pos": self.join_written_pos,
            "option_offsets": list(self.option_offsets),
        }


@dataclass(frozen=True)
class RequestJoinSpec:
    """One join predicate submitted over its complete cross product."""

    written_pos: int
    aliases: tuple[str, ...]
    outer_aliases: tuple[str, ...]
    anchor: str | None
    semantics: str
    selectivity: float | None
    label_token_ids: tuple[tuple[str, tuple[Any, ...]], ...]
    frame_token_ids: tuple[tuple[str, tuple[Any, ...]], ...]
    tail_token_ids: tuple[Any, ...]
    label_texts: tuple[tuple[str, str], ...] = ()
    frame_texts: tuple[tuple[str, str], ...] = ()
    tail_text: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RequestJoinSpec":
        return cls(
            written_pos=int(value["written_pos"]),
            aliases=tuple(value["aliases"]),
            outer_aliases=tuple(value["outer_aliases"]),
            anchor=value["anchor"],
            semantics=str(value["semantics"]),
            selectivity=value["selectivity"],
            label_token_ids=tuple(
                (str(alias), tuple(tokens))
                for alias, tokens in value["label_token_ids"]
            ),
            frame_token_ids=tuple(
                (str(alias), tuple(tokens))
                for alias, tokens in value["frame_token_ids"]
            ),
            tail_token_ids=tuple(value["tail_token_ids"]),
            label_texts=tuple(
                (str(alias), str(text))
                for alias, text in value.get("label_texts", ())
            ),
            frame_texts=tuple(
                (str(alias), str(text))
                for alias, text in value.get("frame_texts", ())
            ),
            tail_text=str(value.get("tail_text", "")),
        )

    def to_dict(self) -> dict:
        return {
            "written_pos": self.written_pos,
            "aliases": list(self.aliases),
            "outer_aliases": list(self.outer_aliases),
            "anchor": self.anchor,
            "semantics": self.semantics,
            "selectivity": self.selectivity,
            "label_token_ids": [
                [alias, list(tokens)] for alias, tokens in self.label_token_ids
            ],
            "frame_token_ids": [
                [alias, list(tokens)] for alias, tokens in self.frame_token_ids
            ],
            "tail_token_ids": list(self.tail_token_ids),
            "label_texts": [list(item) for item in self.label_texts],
            "frame_texts": [list(item) for item in self.frame_texts],
            "tail_text": self.tail_text,
        }



@dataclass(frozen=True)
class RequestExtractSpec:
    """Specification for one decoded answer per document, located in its text.

    The tail_token_ids follow the document and contain the question,
    the instruction, the answer template, and the phrase cue. The
    answer is the text the model writes up to the closing quote,
    located in the document's text. The preamble precedes the document: the model's
    extract turn and the document label, which differ from the
    request's shared preamble on a model that answers AI.IF raw.
    """

    alias: str
    output: str
    question: str
    tail_token_ids: tuple[Any, ...]
    # the tail's text, for an engine that tokenizes prompts itself
    tail_text: str = ""
    preamble_token_ids: tuple[Any, ...] = ()
    preamble_text: str = ""
    max_answer_tokens: int = 64

    def __post_init__(self):
        validate_integer_setting("max_answer_tokens", self.max_answer_tokens)

    @property
    def span_output(self) -> str:
        """Return the name of the span column."""
        return self.output + "_span"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RequestExtractSpec":
        return cls(
            alias=str(value["alias"]),
            output=str(value["output"]),
            question=str(value["question"]),
            tail_token_ids=tuple(value["tail_token_ids"]),
            tail_text=str(value.get("tail_text", "")),
            preamble_token_ids=tuple(value.get("preamble_token_ids", ())),
            preamble_text=str(value.get("preamble_text", "")),
            max_answer_tokens=value.get("max_answer_tokens",
                                        cls.max_answer_tokens),
        )

    def to_dict(self) -> dict:
        return {
            "alias": self.alias,
            "output": self.output,
            "question": self.question,
            "tail_token_ids": list(self.tail_token_ids),
            "tail_text": self.tail_text,
            "preamble_token_ids": list(self.preamble_token_ids),
            "preamble_text": self.preamble_text,
            "max_answer_tokens": self.max_answer_tokens,
        }


def _short_text(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass(frozen=True)
class RequestExecution(PhysicalNode):
    """Run filters and joins through an independent request engine."""

    backend_name: str = ""
    aliases: tuple[str, ...] = ()
    preamble_token_ids: tuple[Any, ...] = ()
    preamble_text: str = ""
    filters: tuple[RequestFilterSpec, ...] = ()
    joins: tuple[RequestJoinSpec, ...] = ()
    classifies: tuple[RequestClassifySpec, ...] = ()
    extracts: tuple[RequestExtractSpec, ...] = ()

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
                f"ids:{alias}",
                ValueType.DOCUMENT_IDS,
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
                f"join_answers:{spec.written_pos}",
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
        return tuple(outputs)

    def attributes(self) -> dict:
        return {
            "backend_name": self.backend_name,
            "aliases": list(self.aliases),
            "preamble_token_ids": list(self.preamble_token_ids),
            "preamble_text": self.preamble_text,
            "filters": [spec.to_dict() for spec in self.filters],
            "joins": [spec.to_dict() for spec in self.joins],
            "classifies": [spec.to_dict() for spec in self.classifies],
            "extracts": [spec.to_dict() for spec in self.extracts],
        }

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
            fields[f"join {spec.written_pos + 1}"] = (
                f"{spec.semantics}: {_short_text(spec.tail_text)}")
        for spec in self.classifies:
            fields[f"classify {spec.output}"] = list(spec.labels)
        for spec in self.extracts:
            fields[f"extract {spec.output}"] = f"question={spec.question!r}"
        return fields

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            backend_name=str(attributes["backend_name"]),
            aliases=tuple(attributes["aliases"]),
            preamble_token_ids=tuple(attributes["preamble_token_ids"]),
            preamble_text=str(attributes.get("preamble_text", "")),
            filters=tuple(
                RequestFilterSpec.from_mapping(spec)
                for spec in attributes["filters"]
            ),
            joins=tuple(
                RequestJoinSpec.from_mapping(spec)
                for spec in attributes["joins"]
            ),
            classifies=tuple(
                RequestClassifySpec.from_mapping(spec)
                for spec in attributes.get("classifies", ())
            ),
            extracts=tuple(
                RequestExtractSpec.from_mapping(spec)
                for spec in attributes.get("extracts", ())
            ),
        )
