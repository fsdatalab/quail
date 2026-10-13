"""Built in physical node definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from .base import (
    AnswerOutput,
    ExecutionLocation,
    OutputPort,
    PhysicalNode,
    ValueType,
    validate_integer_setting,
)


@dataclass(frozen=True)
class Scan(PhysicalNode):
    """Read one tokenized document input supplied by the coordinator.

    ``predicates`` are the regular predicates, as (column, comparison, value)
    triples; the runtime keeps the documents that pass every one and
    reads nothing else for the rest. ``expected_docs`` is the planner's
    estimate of how many pass, from a sample; None without tests.
    """

    alias: str = ""
    input_id: str = ""
    n_docs: int = 0
    total_tokens: int = 0
    shard_ranges: tuple[tuple[int, int], ...] = ()
    shard_token_loads: tuple[int, ...] = ()
    predicates: tuple[tuple[str, str, Any], ...] = ()
    expected_docs: float | None = None

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
            "predicates": [
                [column, comparison,
                 list(value) if isinstance(value, tuple) else value]
                for column, comparison, value in self.predicates],
            "expected_docs": self.expected_docs,
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
            "predicates": [
                f"{column} {comparison}" + ("" if value is None else f" {value!r}")
                for column, comparison, value in self.predicates],
            "expected_docs": self.expected_docs,
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
            predicates=tuple(
                (str(column), str(comparison),
                 tuple(value) if isinstance(value, list) else value)
                for column, comparison, value
                in attributes.get("predicates", ())),
            expected_docs=(None if attributes.get("expected_docs") is None
                           else float(attributes["expected_docs"])),
        )



@dataclass(frozen=True)
class InferenceSpec:
    """Bound inputs, output columns, and cost of one model call."""

    name: str
    aliases: tuple[str, ...]
    query_template: str
    arguments: tuple[tuple[str, str], ...]
    expected_inputs: float
    estimated_seconds: float
    prompt_token_parts: tuple[tuple[int, ...], ...] = ()
    share_prefixes: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InferenceSpec":
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
            prompt_token_parts=tuple(
                tuple(part) for part in value.get("prompt_token_parts", ())
            ),
            share_prefixes=bool(value.get("share_prefixes", False)),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "query_template": self.query_template,
            "arguments": [list(argument) for argument in self.arguments],
            "expected_inputs": self.expected_inputs,
            "estimated_seconds": self.estimated_seconds,
            "prompt_token_parts": [list(part) for part in self.prompt_token_parts],
            "share_prefixes": self.share_prefixes,
        }

    @property
    def output_columns(self) -> tuple[str, ...]:
        """Return the document IDs and computed columns."""
        return (*self.aliases, self.name)


@dataclass(frozen=True)
class ScoreSpec(InferenceSpec):
    """Settings for scoring documents or pairs, including diffusion draws."""

    pair_fraction: float = 1.0
    draws: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScoreSpec":
        return cls(
            **vars(InferenceSpec.from_mapping(value)),
            pair_fraction=float(value.get("pair_fraction", 1.0)),
            draws=int(value.get("draws", 1)),
        )

    def to_dict(self) -> dict:
        return {**super().to_dict(), "pair_fraction": self.pair_fraction,
                "draws": self.draws}


@dataclass(frozen=True)
class AiInference(PhysicalNode):
    """Append model outputs to candidate rows."""

    backend_name: str = ""
    model: str = ""
    spec: InferenceSpec | None = None
    keep_kv: bool = False

    answer_kind: ClassVar[str] = "scores"
    spec_type: ClassVar[type[InferenceSpec]] = InferenceSpec
    document_ids: ClassVar[bool] = False
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR

    @property
    def document_head(self) -> tuple[int, ...] | None:
        return (self.spec.prompt_token_parts[0]
                if self.spec and self.spec.prompt_token_parts else ())

    @property
    def backend(self) -> str:
        """Return the backend selected for this model node."""
        return self.backend_name

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        if self.spec is None:
            return ()
        columns = (self.spec.output_columns if self.answer_kind == "extracts"
                   else (*self.spec.aliases, self.spec.name))
        ports = (OutputPort("rows", ValueType.ROWS,
                            schema=self.spec.output_columns,
                            answer=AnswerOutput(self.answer_kind, self.spec.name,
                                                columns)),)
        if self.document_ids and len(self.spec.aliases) == 1:
            (alias,) = self.spec.aliases
            ports += (OutputPort(f"ids:{alias}", ValueType.DOCUMENT_IDS,
                                 schema=(alias,)),)
        return ports

    def attributes(self) -> dict:
        return {
            "backend_name": self.backend_name,
            "keep_kv": self.keep_kv,
            "model": self.model,
            "spec": None if self.spec is None else self.spec.to_dict(),
        }

    def explain_fields(self) -> Mapping[str, Any]:
        spec = self.spec
        return {
            "backend": self.backend_name,
            "keep_kv": self.keep_kv,
            "model": self.model,
            "batching": "token_based_admission",
            "output": None if spec is None else spec.name,
            "aliases": [] if spec is None else list(spec.aliases),
            "expected_inputs": 0 if spec is None else spec.expected_inputs,
            "estimated_seconds": (
                0 if spec is None else spec.estimated_seconds
            ),
            "share_prefixes": False if spec is None else spec.share_prefixes,
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        value = attributes["spec"]
        return cls(
            node_id=node_id,
            inputs=inputs,
            backend_name=str(attributes["backend_name"]),
            keep_kv=bool(attributes.get("keep_kv", False)),
            model=str(attributes["model"]),
            spec=None if value is None else cls.spec_type.from_mapping(value),
        )


@dataclass(frozen=True)
class AiScore(AiInference):
    """Append one FLOAT64 reranker score to candidate rows."""

    spec: ScoreSpec | None = None
    spec_type: ClassVar[type[InferenceSpec]] = ScoreSpec
    type_name: ClassVar[str] = "quail.ai_score"
    runtime_key: ClassVar[str] = type_name


@dataclass(frozen=True)
class ClassifySpec(InferenceSpec):
    """Specification for classifying individual documents or joined pairs.

    Attributes:
        labels: Labels in query order.
        label_token_ids: Token ids for each label: its letter for the
            letters rule, its text for the trie rules.
        label_word_starts: Per label, which of its tokens begin a new
            word, from ``quail.labels.word_starts``; empty marks none.
            The trie_tree rule reads a label through the end of the
            word that tells it apart.
        scoring: The label scoring rule: letters, trie_tree, or trie_decode.
            The planner leaves it empty until the label_scoring rule
            picks one.
        probabilities: Whether to add a name + "_probabilities" map column.
        join_layout: Anchor note and partner label token sequences for pair
            classification, or None for individual documents. The question
            follows the partner document.
        frame_tokens: For the decision_choice rule, the tokens of the
            prompt tail before the first option block; the option
            blocks and the closing line follow.
    """

    pair_fraction: float = 1.0
    draws: int = 1
    labels: tuple[str, ...] = ()
    label_token_ids: tuple[tuple[int, ...], ...] = ()
    label_word_starts: tuple[tuple[bool, ...], ...] = ()
    scoring: str = "letters"
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
        return cls(
            **vars(InferenceSpec.from_mapping(value)),
            pair_fraction=float(value.get("pair_fraction", 1.0)),
            draws=int(value.get("draws", 1)),
            labels=tuple(str(label) for label in value["labels"]),
            label_token_ids=tuple(
                tuple(int(token) for token in ids)
                for ids in value["label_token_ids"]),
            label_word_starts=tuple(
                tuple(bool(flag) for flag in flags)
                for flags in value.get("label_word_starts", ())),
            scoring=str(value.get("scoring", "letters")),
            probabilities=bool(value.get("probabilities", False)),
            join_layout=(None if value.get("join_layout") is None else tuple(
                tuple(int(token) for token in part)
                for part in value["join_layout"])),
            frame_tokens=int(value.get("frame_tokens", 0)),
        )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "pair_fraction": self.pair_fraction,
            "draws": self.draws,
            "labels": list(self.labels),
            "label_token_ids": [list(ids) for ids in self.label_token_ids],
            "label_word_starts": [list(flags) for flags in self.label_word_starts],
            "scoring": self.scoring,
            "probabilities": self.probabilities,
            "join_layout": (None if self.join_layout is None
                            else [list(part) for part in self.join_layout]),
            "frame_tokens": self.frame_tokens,
        }


@dataclass(frozen=True)
class AiClassify(AiInference):
    """Physical operator that appends a label to each input row.

    The spec names the label scoring rule. The backend runs any decode
    rounds within this node and returns the labels through its rows port.
    """

    spec: ClassifySpec | None = None

    answer_kind: ClassVar[str] = "classifies"
    spec_type: ClassVar[type[InferenceSpec]] = ClassifySpec
    document_ids: ClassVar[bool] = True
    type_name: ClassVar[str] = "quail.ai_classify"
    runtime_key: ClassVar[str] = type_name

    @property
    def document_aliases(self) -> tuple[str, ...]:
        return () if self.spec is None else (self.spec.anchor,)

    def document_outputs(self, alias: str) -> tuple[str, ...]:
        return ("rows", f"ids:{alias}")

    def can_follow(self, previous: PhysicalNode, first: PhysicalNode) -> bool:
        if isinstance(previous, AiJoin):
            return self.classifies_join(previous)
        if isinstance(first, AiClassify):
            return (self.spec.partner is None
                    and self.document_head == first.document_head)
        return self.spec.partner is None

    def classifies_join(self, join: "AiJoin") -> bool:
        """Return whether this classification reads the join's surviving pairs."""
        return (self.spec is not None and self.spec.partner is not None
                and self.spec.anchor == join.anchor
                and sorted(self.spec.aliases) == sorted((join.anchor, *join.partners)))

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            **super().explain_fields(),
            "labels": [] if self.spec is None else list(self.spec.labels),
            "scoring": None if self.spec is None else self.spec.scoring,
            "draws": 1 if self.spec is None else self.spec.draws,
            "probabilities": (False if self.spec is None
                              else self.spec.probabilities),
        }


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
    def document_aliases(self) -> tuple[str, ...]:
        return self.aliases if len(self.aliases) == 1 else ()

    def document_outputs(self, alias: str) -> tuple[str, ...]:
        return ("rows", f"ids:{alias}")

    @property
    def column(self) -> str:
        """Return the column tested by the predicate."""
        return self.predicate.column

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        if len(self.aliases) == 1:
            (alias,) = self.aliases
            return (
                OutputPort("rows", ValueType.ROWS),
                OutputPort(f"filter_answers:{alias}", ValueType.FILTER_ANSWERS),
                OutputPort(f"ids:{alias}", ValueType.DOCUMENT_IDS,
                           schema=(alias,)),
            )
        return (
            OutputPort("rows", ValueType.ROWS),
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
    """Evaluate one predicate on a document input."""

    alias: str = ""
    arena_writes: bool = False
    keep_kv: bool = False
    written_pos: int = 0
    question_tokens: int = 0
    preamble_tokens: int = 0
    selectivity: float | None = None
    expected_docs: float = 0.0
    question_token_ids: tuple[Any, ...] = ()
    # documents borrow the KV pages of a document that shares their
    # token prefix (the prefix_sharing rule)
    share_prefixes: bool = False
    # "unified" or "tree": the attention path the tree_attention rule
    # chose; empty leaves the executor's default
    attention: str = ""
    # value columns of the alias; the executor reads one document of
    # each key value at a time and skips the rest once one survives
    stop_key: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.ai_filter"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR
    backend: ClassVar[str] = "quail"

    worker_pipelining: ClassVar[bool] = True

    @property
    def document_aliases(self) -> tuple[str, ...]:
        return (self.alias,)

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
            "written_pos": self.written_pos,
            "question_tokens": self.question_tokens,
            "preamble_tokens": self.preamble_tokens,
            "selectivity": self.selectivity,
            "expected_docs": self.expected_docs,
            "question_token_ids": list(self.question_token_ids),
            "share_prefixes": self.share_prefixes,
            "attention": self.attention,
            "stop_key": list(self.stop_key),
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
            written_pos=int(attributes["written_pos"]),
            question_tokens=int(attributes["question_tokens"]),
            preamble_tokens=int(attributes["preamble_tokens"]),
            selectivity=attributes["selectivity"],
            expected_docs=float(attributes["expected_docs"]),
            question_token_ids=tuple(attributes["question_token_ids"]),
            share_prefixes=bool(attributes.get("share_prefixes", False)),
            attention=str(attributes.get("attention", "")),
            stop_key=tuple(str(column)
                           for column in attributes.get("stop_key", ())),
        )


@dataclass(frozen=True)
class ExtractSpec(InferenceSpec):
    """Specification for copying each document's answer to a question.

    The prompt token parts are the head before the document and the
    tail after it, ending in the phrase cue.

    Attributes:
        question: The question the answer is copied for.
        max_answer_tokens: Maximum tokens in an extracted answer.
        draft_tokens: Source tokens to verify after each chosen token.
        top_tokens: Vocabulary candidates returned by each GPU readout.
        first_window_chars: Source characters tokenized in the first window.
        window_chars: Source characters tokenized in later windows.
        trim: Whether the answer is the fewest words that answer the
            question, or the whole lines that hold them.
        shares_head: Whether the prompt's head is the query's shared
            preamble, so the extraction reads the document KV a filter
            wrote and continues its pipeline.
    """

    question: str = ""
    trim: bool = True
    shares_head: bool = False
    max_answer_tokens: int = 64
    draft_tokens: int = 16
    top_tokens: int = 1024
    first_window_chars: int = 64
    window_chars: int = 512

    def __post_init__(self):
        for name in ("max_answer_tokens", "draft_tokens", "top_tokens",
                     "first_window_chars", "window_chars"):
            validate_integer_setting(name, getattr(self, name),
                                      minimum=0 if name == "draft_tokens" else 1)

    def same_decoding(self, other: "ExtractSpec") -> bool:
        """Return whether two extractions can use the same decoder."""
        return all(getattr(self, name) == getattr(other, name) for name in (
            "max_answer_tokens", "draft_tokens", "top_tokens",
            "first_window_chars", "window_chars"))

    @property
    def alias(self) -> str:
        """Return the table alias the answer columns belong to."""
        return self.aliases[0]

    @property
    def span_name(self) -> str:
        """Return the name of the span column."""
        return self.name + "_span"

    @property
    def output_columns(self) -> tuple[str, ...]:
        """Return the document ID, answer, and source span columns."""
        return (*super().output_columns, self.span_name)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExtractSpec":
        return cls(
            **vars(InferenceSpec.from_mapping(value)),
            question=str(value["question"]),
            trim=bool(value.get("trim", True)),
            shares_head=bool(value.get("shares_head", False)),
            **{name: value.get(name, getattr(cls, name))
               for name in ("max_answer_tokens", "draft_tokens", "top_tokens",
                            "first_window_chars", "window_chars")},
        )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "question": self.question,
            "trim": self.trim,
            "shares_head": self.shares_head,
            "max_answer_tokens": self.max_answer_tokens,
            "draft_tokens": self.draft_tokens,
            "top_tokens": self.top_tokens,
            "first_window_chars": self.first_window_chars,
            "window_chars": self.window_chars,
        }


@dataclass(frozen=True)
class AiExtract(AiInference):
    """Physical operator that appends each row's answer and its span.

    The rows port carries one row per input document: its id, the
    answer text, and the answer's (start, end) character span in the
    document; both are null when the document does not answer the
    question. The ids port carries the same documents.
    """

    spec: ExtractSpec | None = None

    answer_kind: ClassVar[str] = "extracts"
    spec_type: ClassVar[type[InferenceSpec]] = ExtractSpec
    document_ids: ClassVar[bool] = True
    type_name: ClassVar[str] = "quail.ai_extract"
    runtime_key: ClassVar[str] = type_name

    worker_pipelining: ClassVar[bool] = True

    @property
    def document_head(self) -> tuple[int, ...] | None:
        return (None if self.spec and self.spec.shares_head
                else super().document_head)

    @property
    def document_aliases(self) -> tuple[str, ...]:
        return () if self.spec is None else (self.spec.alias,)

    def document_outputs(self, alias: str) -> tuple[str, ...]:
        return ("rows", f"ids:{alias}")

    def can_follow(self, previous: PhysicalNode, first: PhysicalNode) -> bool:
        if isinstance(previous, AiExtract):
            return (previous.spec.same_decoding(self.spec)
                    and previous.document_head == self.document_head)
        if not self.spec.shares_head:
            return False
        if isinstance(first, AiClassify):
            return (first.spec.partner is None
                    and first.spec.prompt_token_parts[0]
                    == self.spec.prompt_token_parts[0])
        return isinstance(first, AiFilter)

    def can_precede(self, consumer: PhysicalNode) -> bool:
        return isinstance(consumer, AiExtract)

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            **super().explain_fields(),
            "question": None if self.spec is None else self.spec.question,
            "trim": True if self.spec is None else self.spec.trim,
        }


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
    """Evaluate one join predicate using an anchor document."""

    anchor: str = ""
    anchor_resident: str = "none"
    keep_kv: bool = False
    written_pos: int = 0
    exec_idx: int = 0
    partners: tuple[str, ...] = ()
    semantics: str = "full"
    selectivity: float | None = None
    expected_tuples: float = 0.0
    anchor_frame_tokens: int = 0
    pair_tail_tokens: int = 0
    tuple_tokens: float = 0.0
    pairs_from: str = ""
    frame_token_ids: tuple[int, ...] = ()
    label_token_ids: tuple[tuple[str, tuple[int, ...]], ...] = ()
    tail_token_ids: tuple[int, ...] = ()
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

    worker_pipelining: ClassVar[bool] = True

    @property
    def document_aliases(self) -> tuple[str, ...]:
        return (self.anchor,)

    def document_outputs(self, alias: str) -> tuple[str, ...]:
        return (f"join_answers:{self.written_pos}", f"ids:{alias}")

    def can_precede(self, consumer: PhysicalNode) -> bool:
        if isinstance(consumer, AiJoin):
            return (self.anchor == consumer.anchor
                    and self.semantics == consumer.semantics == "full")
        return isinstance(consumer, AiClassify) and consumer.classifies_join(self)

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        outputs = [OutputPort(
            f"ids:{self.anchor}",
            ValueType.DOCUMENT_IDS,
            schema=(self.anchor,),
        )]
        outputs.append(OutputPort(
            f"join_answers:{self.written_pos}",
            ValueType.JOIN_ANSWERS,
            schema=(self.anchor, *self.partners, "answer"),
        ))
        return tuple(outputs)

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
            "keep_kv": self.keep_kv,
            "attention": self.attention,
            "share_prefixes": self.share_prefixes,
        }

    def attributes(self) -> dict:
        return {
            **self.explain_fields(),
            "frame_token_ids": list(self.frame_token_ids),
            "label_token_ids": [[alias, list(tokens)]
                                for alias, tokens in self.label_token_ids],
            "tail_token_ids": list(self.tail_token_ids),
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, value):
        return cls(
            node_id=node_id, inputs=inputs,
            keep_kv=bool(value["keep_kv"]),
            attention=str(value["attention"]),
            share_prefixes=bool(value["share_prefixes"]),
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
    def document_aliases(self) -> tuple[str, ...]:
        if self.kind != "per_batch":
            return ()
        return self.aliases if self.ids == "pairs" else self.aliases[:1]

    def document_outputs(self, alias: str) -> tuple[str, ...]:
        if self.ids == "pairs":
            return (f"pairs:{self.written_pos}",)
        return super().document_outputs(alias)

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
class Aggregate(PhysicalNode):
    """Group the result rows and compute aggregates on the coordinator.

    ``keys`` name input columns. ``aggregates`` are (name, function,
    argument) triples, the argument an input column or None for
    ``count(*)``. ``having`` are (name, comparison, value) tests over
    the aggregates. ``columns`` are the output columns, keys and
    aggregate names in result order.
    """

    keys: tuple[str, ...] = ()
    aggregates: tuple[tuple[str, str, str | None], ...] = ()
    having: tuple[tuple[str, str, float], ...] = ()
    columns: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.aggregate"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort("rows", ValueType.ROWS, schema=self.columns),)

    def attributes(self) -> dict:
        return {
            "keys": list(self.keys),
            "aggregates": [list(aggregate) for aggregate in self.aggregates],
            "having": [list(test) for test in self.having],
            "columns": list(self.columns),
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            keys=tuple(attributes["keys"]),
            aggregates=tuple(
                (str(name), str(function),
                 None if argument is None else str(argument))
                for name, function, argument in attributes["aggregates"]),
            having=tuple((str(name), str(comparison), value)
                         for name, comparison, value
                         in attributes.get("having", ())),
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
