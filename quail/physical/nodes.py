"""Built in physical node definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from .base import (
    ExecutionLocation,
    OutputPort,
    PhysicalNode,
    ValueType,
)


@dataclass(frozen=True)
class FilterStage:
    """One predicate inside a packed filter."""

    written_pos: int
    question_tokens: int
    preamble_tokens: int
    selectivity: float | None
    expected_docs: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FilterStage":
        return cls(
            written_pos=int(value["written_pos"]),
            question_tokens=int(value["question_tokens"]),
            preamble_tokens=int(value["preamble_tokens"]),
            selectivity=value["selectivity"],
            expected_docs=float(value["expected_docs"]),
        )

    def to_dict(self) -> dict:
        return {
            "written_pos": self.written_pos,
            "question_tokens": self.question_tokens,
            "preamble_tokens": self.preamble_tokens,
            "selectivity": self.selectivity,
            "expected_docs": self.expected_docs,
        }


@dataclass(frozen=True)
class JoinStage:
    """One predicate inside an anchored join."""

    written_pos: int
    exec_idx: int
    anchor: str
    partners: tuple[str, ...]
    semantics: str
    selectivity: float | None
    expected_tuples: float
    anchor_frame_tokens: int
    pair_tail_tokens: int
    anchor_resident: str
    tuple_tokens: float
    # empty for a cross join
    pairs_from: str = ""
    frame_token_ids: tuple[int, ...] = ()
    label_token_ids: tuple[tuple[str, tuple[int, ...]], ...] = ()
    tail_token_ids: tuple[int, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "JoinStage":
        return cls(
            written_pos=int(value["written_pos"]),
            exec_idx=int(value["exec_idx"]),
            anchor=str(value["anchor"]),
            partners=tuple(value["partners"]),
            semantics=str(value["semantics"]),
            selectivity=value["selectivity"],
            expected_tuples=float(value["expected_tuples"]),
            anchor_frame_tokens=int(value["anchor_frame_tokens"]),
            pair_tail_tokens=int(value["pair_tail_tokens"]),
            anchor_resident=str(value["anchor_resident"]),
            tuple_tokens=float(value["tuple_tokens"]),
            pairs_from=str(value.get("pairs_from", "")),
            frame_token_ids=tuple(value.get("frame_token_ids", ())),
            label_token_ids=tuple(
                (alias, tuple(tokens))
                for alias, tokens in value.get("label_token_ids", ())
            ),
            tail_token_ids=tuple(value.get("tail_token_ids", ())),
        )

    def explain_fields(self) -> dict:
        return {
            "written_pos": self.written_pos,
            "exec_idx": self.exec_idx,
            "anchor": self.anchor,
            "partners": list(self.partners),
            "semantics": self.semantics,
            "selectivity": self.selectivity,
            "expected_tuples": self.expected_tuples,
            "anchor_frame_tokens": self.anchor_frame_tokens,
            "pair_tail_tokens": self.pair_tail_tokens,
            "anchor_resident": self.anchor_resident,
            "tuple_tokens": self.tuple_tokens,
            "pairs_from": self.pairs_from,
        }

    def to_dict(self) -> dict:
        return {
            **self.explain_fields(),
            "frame_token_ids": list(self.frame_token_ids),
            "label_token_ids": [[alias, list(tokens)]
                                for alias, tokens in self.label_token_ids],
            "tail_token_ids": list(self.tail_token_ids),
        }

    def runtime_spec(self) -> dict:
        """Return the bound prompt and predicate for the join driver."""
        return {
            "anchor": self.anchor,
            "partners": list(self.partners),
            "semantics": self.semantics,
            "selectivity": self.selectivity,
            "written_pos": self.written_pos,
            "pairs_from": self.pairs_from,
            "frame": self.frame_token_ids,
            "labels": dict(self.label_token_ids),
            "tail": self.tail_token_ids,
        }


@dataclass(frozen=True)
class RequestFilterSpec:
    """One filter chain submitted as model requests."""

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
    """One AI.CLASSIFY call submitted as one decoded answer per document.

    ``tail_token_ids`` follow each document: the question, the
    category list, and the answer cue. ``tests`` are the filters on its label
    on this call, (written position, accepted labels), applied in
    order after the documents are labeled.
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
class Scan(PhysicalNode):
    """Read one tokenized document input supplied by the coordinator."""

    alias: str = ""
    input_id: str = ""
    n_docs: int = 0
    total_tokens: int = 0
    shard_ranges: tuple[tuple[int, int], ...] = ()
    shard_token_loads: tuple[int, ...] = ()

    type_name: ClassVar[str] = "quail.scan"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    def __post_init__(self) -> None:
        if len(self.shard_ranges) != len(self.shard_token_loads):
            raise ValueError(
                "document shard ranges and token loads must have equal size"
            )
        expected = 0
        for start, stop in self.shard_ranges:
            if start != expected or stop < start:
                raise ValueError(
                    "document shard ranges must be ordered and contiguous"
                )
            expected = stop
        if self.shard_ranges and expected != self.n_docs:
            raise ValueError(
                "document shard ranges must cover every document"
            )

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            f"ids:{self.alias}",
            ValueType.DOCUMENT_IDS,
            schema=(self.alias,),
        ),)

    def attributes(self) -> dict:
        return {
            "alias": self.alias,
            "input_id": self.input_id,
            "n_docs": self.n_docs,
            "total_tokens": self.total_tokens,
            "shard_token_loads": list(self.shard_token_loads),
            "shard_ranges": [list(bounds) for bounds in self.shard_ranges],
        }

    @property
    def shards(self) -> tuple[range, ...]:
        """Return each worker's contiguous document range."""
        return tuple(range(start, stop) for start, stop in self.shard_ranges)

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            "alias": self.alias,
            "input_id": self.input_id,
            "n_docs": self.n_docs,
            "total_tokens": self.total_tokens,
            "shard_token_loads": list(self.shard_token_loads),
            "shard_docs": [stop - start
                           for start, stop in self.shard_ranges],
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            alias=attributes["alias"],
            input_id=attributes["input_id"],
            n_docs=int(attributes["n_docs"]),
            total_tokens=int(attributes["total_tokens"]),
            shard_ranges=tuple(
                (int(start), int(stop))
                for start, stop in attributes["shard_ranges"]
            ),
            shard_token_loads=tuple(attributes["shard_token_loads"]),
        )


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
                schema=(spec.alias, spec.output),
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
        }

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            "backend": self.backend_name,
            "aliases": list(self.aliases),
            "filter_chains": len(self.filters),
            "joins": len(self.joins),
            "classifications": len(self.classifies),
        }

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
        )


@dataclass(frozen=True)
class ScoreSpec:
    """One batched AI.SCORE computation."""

    name: str
    aliases: tuple[str, ...]
    query_template: str
    arguments: tuple[tuple[str, str], ...]
    expected_inputs: float
    estimated_seconds: float
    pair_fraction: float = 1.0
    prompt_token_parts: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScoreSpec":
        return cls(
            name=str(value["name"]),
            aliases=tuple(value["aliases"]),
            query_template=str(value["query_template"]),
            arguments=tuple(
                (str(alias), str(column))
                for alias, column in value["arguments"]
            ),
            expected_inputs=float(value["expected_inputs"]),
            estimated_seconds=float(value["estimated_seconds"]),
            pair_fraction=float(value.get("pair_fraction", 1.0)),
            prompt_token_parts=tuple(
                tuple(part) for part in value.get("prompt_token_parts", ())
            ),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "query_template": self.query_template,
            "arguments": [list(argument) for argument in self.arguments],
            "expected_inputs": self.expected_inputs,
            "estimated_seconds": self.estimated_seconds,
            "pair_fraction": self.pair_fraction,
            "prompt_token_parts": [list(part) for part in self.prompt_token_parts],
        }


@dataclass(frozen=True)
class AiScore(PhysicalNode):
    """Append one FLOAT64 reranker score to candidate rows."""

    backend_name: str = ""
    model: str = ""
    spec: ScoreSpec | None = None

    type_name: ClassVar[str] = "quail.ai_score"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR

    @property
    def backend(self) -> str:
        """Return the backend selected for this model node."""
        return self.backend_name

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        if self.spec is None:
            return ()
        return (OutputPort(
            "scores",
            ValueType.SCORES,
            schema=(*self.spec.aliases, self.spec.name),
        ),)

    def attributes(self) -> dict:
        return {
            "backend_name": self.backend_name,
            "model": self.model,
            "spec": None if self.spec is None else self.spec.to_dict(),
        }

    def explain_fields(self) -> Mapping[str, Any]:
        spec = self.spec
        return {
            "backend": self.backend_name,
            "model": self.model,
            "batching": "token_based_admission",
            "output": None if spec is None else spec.name,
            "aliases": [] if spec is None else list(spec.aliases),
            "expected_inputs": 0 if spec is None else spec.expected_inputs,
            "estimated_seconds": (
                0 if spec is None else spec.estimated_seconds
            ),
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        value = attributes["spec"]
        return cls(
            node_id=node_id,
            inputs=inputs,
            backend_name=str(attributes["backend_name"]),
            model=str(attributes["model"]),
            spec=None if value is None else ScoreSpec.from_mapping(value),
        )


@dataclass(frozen=True)
class ClassifySpec(ScoreSpec):
    """One batched AI.CLASSIFY computation over one table's documents.

    ``prompt_token_parts`` is (preamble ids, tail ids): the document
    goes between them. ``label_token_ids`` holds each label's ids as
    scored after the tail, in label order: under the ``letters`` rule
    the letter standing for each label, one token; under ``trie_tree``
    and ``trie_decode`` the label's own text. ``scoring`` names the
    rule the executor runs.
    ``share_prefixes`` lets a document borrow the KV pages of a
    document sharing its token prefix (the prefix_sharing rule).
    ``stages`` are later classifications of the same documents, run
    while each document's KV is still resident; a stage runs on the
    documents whose previous label its gate accepts. A classification
    of joined rows has two aliases, the anchor and its partner, and
    ``join_layout`` holds (the anchor note written after the anchor
    document, the partner label written before each partner
    document); its tail follows the partner.
    """

    labels: tuple[str, ...] = ()
    label_token_ids: tuple[tuple[int, ...], ...] = ()
    scoring: str = "letters"
    share_prefixes: bool = False
    stages: tuple["ClassifyStage", ...] = ()
    join_layout: tuple[tuple[int, ...], tuple[int, ...]] | None = None

    @property
    def anchor(self) -> str:
        """The alias whose documents the classification is anchored on."""
        return self.aliases[0]

    @property
    def partner(self) -> str | None:
        """The partner alias of a classification of joined rows, else None."""
        return self.aliases[1] if len(self.aliases) == 2 else None

    @property
    def chain(self) -> tuple["ClassifySpec", ...]:
        """This classification and every later stage's, in order."""
        return (self,) + tuple(stage.spec for stage in self.stages)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClassifySpec":
        base = ScoreSpec.from_mapping(value)
        return cls(
            **{name: getattr(base, name) for name in (
                "name", "aliases", "query_template", "arguments",
                "expected_inputs", "estimated_seconds", "pair_fraction",
                "prompt_token_parts")},
            labels=tuple(str(label) for label in value["labels"]),
            label_token_ids=tuple(
                tuple(int(token) for token in ids)
                for ids in value["label_token_ids"]),
            scoring=str(value.get("scoring", "letters")),
            share_prefixes=bool(value.get("share_prefixes", False)),
            stages=tuple(ClassifyStage.from_mapping(stage)
                         for stage in value.get("stages", ())),
            join_layout=(None if value.get("join_layout") is None else tuple(
                tuple(int(token) for token in part)
                for part in value["join_layout"])),
        )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "labels": list(self.labels),
            "label_token_ids": [list(ids) for ids in self.label_token_ids],
            "scoring": self.scoring,
            "share_prefixes": self.share_prefixes,
            "stages": [stage.to_dict() for stage in self.stages],
            "join_layout": (None if self.join_layout is None
                            else [list(part) for part in self.join_layout]),
        }


@dataclass(frozen=True)
class ClassifyStage:
    """A later classification in a chain and the gate before it.

    ``accepted`` names the previous stage's labels that let a document
    through; None lets every document through.
    """

    spec: ClassifySpec
    accepted: tuple[str, ...] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClassifyStage":
        accepted = value.get("accepted")
        return cls(
            spec=ClassifySpec.from_mapping(value["spec"]),
            accepted=None if accepted is None
            else tuple(str(label) for label in accepted),
        )

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "accepted": None if self.accepted is None else list(self.accepted),
        }


@dataclass(frozen=True)
class AiClassify(AiScore):
    """Append one STRING label, chosen from a fixed list, to each document row.

    Runs through the AI.SCORE runtime; the spec's labels make the
    appended column a label instead of a score.
    """

    spec: ClassifySpec | None = None

    type_name: ClassVar[str] = "quail.ai_classify"

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        """The label rows, and the ids of the documents that got a label."""
        ports = super().outputs
        if self.spec is not None and len(self.spec.aliases) == 1:
            (alias,) = self.spec.aliases
            ports += (OutputPort(f"ids:{alias}", ValueType.DOCUMENT_IDS,
                                 schema=(alias,)),)
        return ports

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            **super().explain_fields(),
            "labels": [] if self.spec is None else list(self.spec.labels),
            "scoring": None if self.spec is None else self.spec.scoring,
            "share_prefixes": (False if self.spec is None
                               else self.spec.share_prefixes),
            "stages": [] if self.spec is None else [
                {"accepted": None if stage.accepted is None
                 else list(stage.accepted), "output": stage.spec.name}
                for stage in self.spec.stages],
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        value = attributes["spec"]
        return cls(
            node_id=node_id,
            inputs=inputs,
            backend_name=str(attributes["backend_name"]),
            model=str(attributes["model"]),
            spec=None if value is None else ClassifySpec.from_mapping(value),
        )


@dataclass(frozen=True)
class InList:
    """The predicate ``column IN (values)`` over a label column."""

    column: str
    values: tuple[str, ...]

    def accepts(self, value) -> bool:
        """Whether one value passes."""
        return value in self.values

    def to_dict(self) -> dict:
        return {"kind": "in_list", "column": self.column,
                "values": list(self.values)}

    def describe(self) -> str:
        return f"{self.column} IN {list(self.values)}"


@dataclass(frozen=True)
class Comparison:
    """The predicate ``column <op> value`` over a score column."""

    column: str
    op: str
    value: float

    def accepts(self, value) -> bool:
        """Whether one value passes."""
        import operator

        compare = {"<": operator.lt, "<=": operator.le,
                   ">": operator.gt, ">=": operator.ge}[self.op]
        return bool(compare(value, self.value))

    def to_dict(self) -> dict:
        return {"kind": "comparison", "column": self.column, "op": self.op,
                "value": self.value}

    def describe(self) -> str:
        return f"{self.column} {self.op} {self.value}"


def predicate_from_mapping(value: Mapping[str, Any]) -> "InList | Comparison":
    """The filter predicate a mapping encodes."""
    if value["kind"] == "in_list":
        return InList(str(value["column"]),
                      tuple(str(item) for item in value["values"]))
    if value["kind"] == "comparison":
        return Comparison(str(value["column"]), str(value["op"]),
                          float(value["value"]))
    raise ValueError(f"unknown filter predicate {value['kind']!r}")


@dataclass(frozen=True)
class Filter(PhysicalNode):
    """Keep the rows of a score or label table that pass a predicate.

    The predicate reads a column a model call produced: ``InList`` for
    an AI.CLASSIFY label tested with ``=`` or ``IN``, ``Comparison``
    for an AI.SCORE score against a threshold. Over one table's rows
    the filter also yields the kept documents' ids, so a join or a
    later operator can take them; over a join's rows it yields the
    join's answers.
    """

    predicate: InList | Comparison | None = None
    aliases: tuple[str, ...] = ()
    selectivity: float | None = None
    written_pos: int = 0

    type_name: ClassVar[str] = "quail.filter"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR

    @property
    def column(self) -> str:
        """The score or label column the predicate reads."""
        return self.predicate.column

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        if len(self.aliases) == 1:
            (alias,) = self.aliases
            return (
                OutputPort("scores", ValueType.SCORES),
                OutputPort(f"filter_answers:{alias}", ValueType.FILTER_ANSWERS),
                OutputPort(f"ids:{alias}", ValueType.DOCUMENT_IDS,
                           schema=(alias,)),
            )
        return (
            OutputPort("scores", ValueType.SCORES),
            OutputPort(f"join_answers:{self.written_pos}",
                       ValueType.JOIN_ANSWERS),
        )

    def attributes(self) -> dict:
        return {
            "predicate": self.predicate.to_dict(),
            "aliases": list(self.aliases),
            "selectivity": self.selectivity,
            "written_pos": self.written_pos,
        }

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            "predicate": self.predicate.describe(),
            "selectivity": self.selectivity,
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            predicate=predicate_from_mapping(attributes["predicate"]),
            aliases=tuple(attributes["aliases"]),
            selectivity=attributes["selectivity"],
            written_pos=int(attributes["written_pos"]),
        )


@dataclass(frozen=True)
class AiFilter(PhysicalNode):
    """Evaluate ordered predicates on one document input."""

    alias: str = ""
    arena_writes: bool = False
    keep_kv: bool = False
    stages: tuple[FilterStage, ...] = ()
    question_token_ids: tuple[tuple[Any, ...], ...] = ()
    # documents borrow the KV pages of a document that shares their
    # token prefix (the prefix_sharing rule)
    share_prefixes: bool = False
    # "unified" or "tree": the attention path the tree_attention rule
    # chose; empty leaves the executor's default
    attention: str = ""

    type_name: ClassVar[str] = "quail.ai_filter"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR
    backend: ClassVar[str] = "quail"

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (
            OutputPort(
                f"ids:{self.alias}",
                ValueType.DOCUMENT_IDS,
                schema=(self.alias,),
            ),
            OutputPort(
                f"filter_answers:{self.alias}",
                ValueType.FILTER_ANSWERS,
                schema=(self.alias, "predicate", "answer"),
            ),
        )

    def attributes(self) -> dict:
        return {
            "alias": self.alias,
            "arena_writes": self.arena_writes,
            "keep_kv": self.keep_kv,
            "stages": [stage.to_dict() for stage in self.stages],
            "question_token_ids": [
                list(question) for question in self.question_token_ids
            ],
            "share_prefixes": self.share_prefixes,
            "attention": self.attention,
        }

    def explain_fields(self) -> Mapping[str, Any]:
        value = self.attributes()
        value.pop("question_token_ids")
        return value

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            alias=attributes["alias"],
            arena_writes=bool(attributes["arena_writes"]),
            keep_kv=bool(attributes["keep_kv"]),
            stages=tuple(
                FilterStage.from_mapping(stage)
                for stage in attributes["stages"]
            ),
            question_token_ids=tuple(
                tuple(question)
                for question in attributes["question_token_ids"]
            ),
            share_prefixes=bool(attributes.get("share_prefixes", False)),
            attention=str(attributes.get("attention", "")),
        )


@dataclass(frozen=True)
class Barrier(PhysicalNode):
    """Move or repartition survivor ids between join steps."""

    next_anchor: str = ""
    aliases: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.barrier"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return tuple(
            OutputPort(
                f"ids:{alias}",
                ValueType.DOCUMENT_IDS,
                schema=(alias,),
            )
            for alias in self.aliases
        )

    def attributes(self) -> dict:
        return {"next_anchor": self.next_anchor, "aliases": list(self.aliases)}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            next_anchor=attributes["next_anchor"],
            aliases=tuple(attributes["aliases"]),
        )


@dataclass(frozen=True)
class AiJoin(PhysicalNode):
    """Evaluate join predicates that use one anchor alias."""

    anchor: str = ""
    anchor_resident: str = "none"
    keep_anchor_kv: bool = False
    stages: tuple[JoinStage, ...] = ()
    # "unified" or "tree": the attention path the tree_attention rule
    # chose; empty leaves the model pipeline's default
    attention: str = ""
    # anchors borrow the KV pages of an anchor that shares their token
    # prefix (the prefix_sharing rule)
    share_prefixes: bool = False

    type_name: ClassVar[str] = "quail.ai_join"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR
    backend: ClassVar[str] = "quail"

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        outputs = [OutputPort(
            f"ids:{self.anchor}",
            ValueType.DOCUMENT_IDS,
            schema=(self.anchor,),
        )]
        outputs.extend(OutputPort(
            f"join_answers:{stage.written_pos}",
            ValueType.JOIN_ANSWERS,
            schema=(stage.anchor, *stage.partners, "answer"),
        ) for stage in self.stages)
        return tuple(outputs)

    def attributes(self) -> dict:
        return {
            "anchor": self.anchor,
            "anchor_resident": self.anchor_resident,
            "keep_anchor_kv": self.keep_anchor_kv,
            "stages": [stage.to_dict() for stage in self.stages],
            "attention": self.attention,
            "share_prefixes": self.share_prefixes,
        }

    def explain_fields(self) -> dict:
        return {**self.attributes(),
                "stages": [stage.explain_fields() for stage in self.stages]}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            anchor=attributes["anchor"],
            anchor_resident=attributes["anchor_resident"],
            keep_anchor_kv=bool(attributes["keep_anchor_kv"]),
            stages=tuple(
                JoinStage.from_mapping(stage)
                for stage in attributes["stages"]
            ),
            attention=str(attributes.get("attention", "")),
            share_prefixes=bool(attributes.get("share_prefixes", False)),
        )


@dataclass(frozen=True)
class Foreign(PhysicalNode):
    """Call a user function between two operators.

    ``kind`` is ``per_batch`` (called on each document as it reaches
    the function inside its table's pipeline, or once over a
    materialized input) or ``barrier`` (called once over every
    survivor, ending the pipeline before it). ``ids`` is
    ``preserve``, ``drop``, or ``pairs``; the function never invents an
    id. ``columns`` are (alias, column) pairs read as values.
    """

    function: str = ""
    kind: str = "per_batch"
    ids: str = "drop"
    columns: tuple[tuple[str, str], ...] = ()
    aliases: tuple[str, ...] = ()
    written_pos: int = -1

    type_name: ClassVar[str] = "quail.foreign"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        if self.ids == "pairs":
            return (OutputPort(
                f"pairs:{self.written_pos}", ValueType.PAIRS,
                schema=tuple(self.aliases)),)
        (alias,) = self.aliases
        return (OutputPort(f"ids:{alias}", ValueType.DOCUMENT_IDS,
                           schema=(alias,)),)

    def attributes(self) -> dict:
        return {
            "function": self.function,
            "kind": self.kind,
            "ids": self.ids,
            "columns": [list(column) for column in self.columns],
            "aliases": list(self.aliases),
            "written_pos": self.written_pos,
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            function=str(attributes["function"]),
            kind=str(attributes["kind"]),
            ids=str(attributes["ids"]),
            columns=tuple(
                (str(alias), str(column))
                for alias, column in attributes["columns"]
            ),
            aliases=tuple(attributes["aliases"]),
            written_pos=int(attributes["written_pos"]),
        )


@dataclass(frozen=True)
class HashJoin(PhysicalNode):
    """Pair the rows of two tables whose key columns are equal.

    ``on`` lists (left column, right column) pairs; a row pair is kept
    when every listed pair of values is equal. The node reads the key
    columns as values and writes the pairs the join at ``written_pos``
    asks the model about. ``pair_fraction`` is the planner's estimate
    of the pairs kept over the cross product.
    """

    left: str = ""
    right: str = ""
    on: tuple[tuple[str, str], ...] = ()
    written_pos: int = -1
    pair_fraction: float = 1.0

    type_name: ClassVar[str] = "quail.hash_join"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(f"pairs:{self.written_pos}", ValueType.PAIRS,
                           schema=(self.left, self.right)),)

    def attributes(self) -> dict:
        return {
            "left": self.left,
            "right": self.right,
            "on": [list(condition) for condition in self.on],
            "written_pos": self.written_pos,
            "pair_fraction": self.pair_fraction,
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            left=str(attributes["left"]),
            right=str(attributes["right"]),
            on=tuple((str(left), str(right))
                     for left, right in attributes["on"]),
            written_pos=int(attributes["written_pos"]),
            pair_fraction=float(attributes.get("pair_fraction", 1.0)),
        )


@dataclass(frozen=True)
class Exchange(PhysicalNode):
    """Route one alias's documents to the GPU that holds their KV.

    Present only in plans for several GPUs. On one GPU it passes the
    ids through.
    """

    anchor: str = ""

    type_name: ClassVar[str] = "quail.exchange"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            f"ids:{self.anchor}",
            ValueType.DOCUMENT_IDS,
            schema=(self.anchor,),
        ),)

    def attributes(self) -> dict:
        return {"anchor": self.anchor}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(node_id=node_id, inputs=inputs, anchor=attributes["anchor"])


@dataclass(frozen=True)
class Recombine(PhysicalNode):
    """Combine answer relations by exact document ids."""

    alias_order: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.recombine"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            "tuples",
            ValueType.ROWS,
            schema=self.alias_order,
        ),)

    def attributes(self) -> dict:
        return {"alias_order": list(self.alias_order)}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            alias_order=tuple(attributes["alias_order"]),
        )


@dataclass(frozen=True)
class Project(PhysicalNode):
    """Select the requested result columns."""

    columns: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.project"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            "rows",
            ValueType.ROWS,
            schema=self.columns,
        ),)

    def attributes(self) -> dict:
        return {"columns": list(self.columns)}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            columns=tuple(attributes["columns"]),
        )


@dataclass(frozen=True)
class Limit(PhysicalNode):
    """Stop result output after a fixed number of rows."""

    count: int = 0

    type_name: ClassVar[str] = "quail.limit"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort("rows", ValueType.ROWS),)

    def attributes(self) -> dict:
        return {"count": self.count}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(node_id=node_id, inputs=inputs, count=int(attributes["count"]))
