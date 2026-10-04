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
        """Return the bound prompt and predicate for the join driver.

        The frame includes the first partner's label, which every pair
        of the anchor shares.
        """
        labels = dict(self.label_token_ids)
        frame = tuple(self.frame_token_ids)
        if self.partners and self.partners[0] in labels:
            frame += tuple(labels[self.partners[0]])
            labels[self.partners[0]] = ()
        return {
            "anchor": self.anchor,
            "partners": list(self.partners),
            "semantics": self.semantics,
            "selectivity": self.selectivity,
            "written_pos": self.written_pos,
            "pairs_from": self.pairs_from,
            "frame": frame,
            "labels": labels,
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
    """Specification for a batched AI.SCORE computation.

    draws is the most noise draws a diffusion model averages per answer;
    an answer whose first draw is certain enough uses one. An
    autoregressive model always uses one.
    """

    name: str
    aliases: tuple[str, ...]
    query_template: str
    arguments: tuple[tuple[str, str], ...]
    expected_inputs: float
    estimated_seconds: float
    pair_fraction: float = 1.0
    prompt_token_parts: tuple[tuple[int, ...], ...] = ()
    draws: int = 1

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
            draws=int(value.get("draws", 1)),
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
            "draws": self.draws,
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
    """Specification for classifying individual documents or joined pairs.

    Attributes:
        labels: Labels in query order.
        label_token_ids: Token ids for each label: its letter for the
            letters rule, its text for the trie rules.
        scoring: The label scoring rule: letters, trie_tree, or trie_decode.
            The planner leaves it empty until the label_scoring rule
            picks one.
        share_prefixes: Whether documents may reuse shared prefix KV pages.
        probabilities: Whether to add a name + "_probabilities" map column.
        join_layout: Anchor note and partner label token sequences for pair
            classification, or None for individual documents. The question
            follows the partner document.
        frame_tokens: For the decision_choice rule, the tokens of the
            prompt tail before the first option block; the option
            blocks and the closing line follow.
    """

    labels: tuple[str, ...] = ()
    label_token_ids: tuple[tuple[int, ...], ...] = ()
    scoring: str = "letters"
    share_prefixes: bool = False
    probabilities: bool = False
    join_layout: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    frame_tokens: int = 0

    @property
    def anchor(self) -> str:
        """Return the anchor table alias."""
        return self.aliases[0]

    @property
    def partner(self) -> str | None:
        """Return the partner table alias, or None for individual documents."""
        return self.aliases[1] if len(self.aliases) == 2 else None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClassifySpec":
        if value.get("stages"):
            raise ValueError(
                "nested classification stages are unsupported; replan query")
        base = ScoreSpec.from_mapping(value)
        return cls(
            **{name: getattr(base, name) for name in (
                "name", "aliases", "query_template", "arguments",
                "expected_inputs", "estimated_seconds", "pair_fraction",
                "prompt_token_parts", "draws")},
            labels=tuple(str(label) for label in value["labels"]),
            label_token_ids=tuple(
                tuple(int(token) for token in ids)
                for ids in value["label_token_ids"]),
            scoring=str(value.get("scoring", "letters")),
            share_prefixes=bool(value.get("share_prefixes", False)),
            probabilities=bool(value.get("probabilities", False)),
            join_layout=(None if value.get("join_layout") is None else tuple(
                tuple(int(token) for token in part)
                for part in value["join_layout"])),
            frame_tokens=int(value.get("frame_tokens", 0)),
        )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "labels": list(self.labels),
            "label_token_ids": [list(ids) for ids in self.label_token_ids],
            "scoring": self.scoring,
            "share_prefixes": self.share_prefixes,
            "probabilities": self.probabilities,
            "join_layout": (None if self.join_layout is None
                            else [list(part) for part in self.join_layout]),
            "frame_tokens": self.frame_tokens,
        }


@dataclass(frozen=True)
class AiClassify(AiScore):
    """Physical operator that appends a label to each input row.

    The spec names the label scoring rule. The backend runs any decode
    rounds within this node and returns the labels through its scores port.
    """

    spec: ClassifySpec | None = None

    type_name: ClassVar[str] = "quail.ai_classify"

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        """Return ports for label rows and the IDs of labeled documents."""
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
            "draws": 1 if self.spec is None else self.spec.draws,
            "probabilities": (False if self.spec is None
                              else self.spec.probabilities),
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
        """Return whether the value is one of the accepted labels."""
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
        """Return whether the value satisfies the score comparison."""
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
    """Decode a filter predicate from its serialized mapping."""
    if value["kind"] == "in_list":
        return InList(str(value["column"]),
                      tuple(str(item) for item in value["values"]))
    if value["kind"] == "comparison":
        return Comparison(str(value["column"]), str(value["op"]),
                          float(value["value"]))
    raise ValueError(f"unknown filter predicate {value['kind']!r}")


@dataclass(frozen=True)
class Filter(PhysicalNode):
    """Physical operator that keeps rows matching a score or label condition.

    InList tests a label produced by AI.CLASSIFY. Comparison tests a score
    produced by AI.SCORE. Single-document filters also return the kept
    document IDs; filters over joined pairs return join answers.
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
        """Return the column tested by the predicate."""
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
    """Physical operator that calls a user function on document IDs or pairs.

    Per-batch functions run as documents reach them in a pipeline, or once
    over a materialized input. Barrier functions run once over all surviving
    rows and end the preceding pipeline. The function may preserve or remove
    input IDs, or return pairs of input IDs; it cannot create new IDs.
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
class Sort(PhysicalNode):
    """Order, deduplicate, and bound the result rows on the coordinator.

    ``keys`` are (column, descending, nulls_first) triples over the
    input columns. ``columns`` are the output columns; an input column
    not among them is read by a key only. Distinct runs before the
    sort, and ``offset`` and ``fetch`` after it.
    """

    keys: tuple[tuple[str, bool, bool], ...] = ()
    columns: tuple[str, ...] = ()
    distinct: bool = False
    offset: int = 0
    fetch: int | None = None

    type_name: ClassVar[str] = "quail.sort"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort("rows", ValueType.ROWS, schema=self.columns),)

    def attributes(self) -> dict:
        return {
            "keys": [list(key) for key in self.keys],
            "columns": list(self.columns),
            "distinct": self.distinct,
            "offset": self.offset,
            "fetch": self.fetch,
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            keys=tuple((str(column), bool(descending), bool(nulls_first))
                       for column, descending, nulls_first
                       in attributes["keys"]),
            columns=tuple(attributes["columns"]),
            distinct=bool(attributes.get("distinct", False)),
            offset=int(attributes.get("offset", 0)),
            fetch=(None if attributes.get("fetch") is None
                   else int(attributes["fetch"])),
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
