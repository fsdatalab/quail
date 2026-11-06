"""The speed of light estimate for one query.

The estimate prices the ideal execution of a query on one device: every
distinct document prefix computed once and kept in KV without limit,
forward passes packed across the whole query, and no host, scheduling,
or kernel overhead. On each alias the AI.IF filters run first in the
planner's order, then its classifications, each followed by the label
filters that test it. A filter under a DISTINCT over its table's
columns, which the executor stops at each key's first survivor, reads
only each key's shortest survivor, or every document of a key with
none. Joins are searched over every feasible eager binary full left
deep plan and anchor choice, with exact survivors from caller supplied
answer and label oracles at every step. A classification of joined
rows follows its join on the join's anchor.

A classification is priced as one question per live document: the
reference prompt's tail (the question, the labels by name, and the
answer cue) over the document's prefix, and one readout row at the cue
that confirms the reference label. An extraction's reference answer is
verified in one pass after the cue, each of its tokens a readout row.

Every operator in the plan is priced or refused: an AI.SCORE
predicate or column, a LIMIT, or a join without full semantics raises
NotImplementedError.

docs/content/docs/architecture/optimizer.mdx describes the model. The
Rust planner in ``_quail_native`` runs the search, calling back into
the oracles for each answer; the work counting and component pricing
are the planner's own, and the search does not call or simulate the
production planner.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping

import _quail_native
import numpy as np
import pyarrow as pa

from quail.common import budgets
from quail.common.expressions import (
    accepts,
    column_name,
    condition_columns,
    evaluate_expression,
    table_alias,
)
from quail.common.pairs import pair_table, partner_map
from quail.common.tokens import prefix_tree
from quail.common.work import Work
from quail.physical import decode_plan_value
from quail.physical.lowering import Vocabulary
from quail.specs import DeviceSpec, ModelSpec

# answer(prompt, assignment) -> bool, where assignment maps each alias
# in the prompt to a row index of that alias's corpus table; for an
# AI.EXTRACT prompt, the reference answer text, or None for no answer;
# for an AI.MAP prompt, the answer the map writes
AnswerOracle = Callable[[object, Mapping[str, int]], object]
# label(prompt, assignment) -> str, the reference label of an
# AI.CLASSIFY prompt for one row assignment
LabelOracle = Callable[[object, Mapping[str, int]], str]

# the operators of one alias, as the stage records name them
FILTER = "AI.IF"
CLASSIFY = "AI.CLASSIFY"
EXTRACT = "AI.EXTRACT"

# the exceptions a refused estimate raises, by the name Rust gives
_ERRORS = {"NotImplementedError": NotImplementedError, "ValueError": ValueError}


@dataclass(frozen=True)
class ComponentLatency:
    """One model component's share of the estimate, as Rust priced it."""

    name: str
    precision: str
    flops: float
    bytes_moved: float
    compute_seconds: float
    memory_seconds: float
    seconds: float
    bound_by: str


@dataclass(frozen=True)
class SpeedOfLight:
    """The estimate's work, forward passes, and component breakdown."""

    work: Work
    passes: int
    components: tuple[ComponentLatency, ...]

    def component(self, name: str) -> ComponentLatency:
        """Return one named component result."""
        return next(component for component in self.components
                    if component.name == name)

    @property
    def bytes_moved(self) -> float:
        return sum(component.bytes_moved for component in self.components)

    @property
    def compute(self) -> float:
        return sum(component.compute_seconds for component in self.components)

    @property
    def memory(self) -> float:
        return sum(component.memory_seconds for component in self.components)

    @property
    def seconds(self) -> float:
        return sum(component.seconds for component in self.components)

    @property
    def bound_by(self) -> str:
        limits = {component.bound_by for component in self.components}
        return limits.pop() if len(limits) == 1 else "mixed"


@dataclass(frozen=True)
class SpeedOfLightEstimate:
    """The ideal time of one query and how it was reached.

    seconds is a lower bound on any schedule of the work: the larger of
    its summed memory time, with the weights read once per pass, and its
    summed compute time, at the device's peak rates. latency prices the
    same work by model component.
    """

    model: str
    device: str
    chunk_tokens: int
    credit_shared_prefixes: bool
    work: Work
    latency: SpeedOfLight
    seconds: float
    usd_per_hour: float
    alias_columns: dict[str, str]
    documents_by_alias: dict[str, int]
    post_filter_counts: dict[str, int]
    filter_stages: tuple[dict, ...]
    join_stages: tuple[dict, ...]
    relation_order: tuple[str, ...]
    cached_prefixes: tuple[str, ...]
    search: dict = field(default_factory=dict)

    @property
    def fresh_tokens(self) -> float:
        return self.work.tokens

    @property
    def usd_per_query(self) -> float:
        return self.seconds * self.usd_per_hour / 3600

    @property
    def input_document_rows(self) -> int:
        return sum(self.documents_by_alias.values())

    @property
    def filter_evaluations(self) -> int:
        return sum(stage["evaluated"] for stage in self.filter_stages
                   if stage["operator"] == FILTER)

    @property
    def classification_evaluations(self) -> int:
        """Return the number of documents and joined rows classified."""
        return sum(stage["evaluated"] for stage in self.filter_stages
                   if stage["operator"] == CLASSIFY) + sum(
            classification["pairs"]
            for stage in self.join_stages
            for classification in stage["classifications"])

    @property
    def extraction_evaluations(self) -> int:
        """Return the number of documents an answer was copied from."""
        return sum(stage["evaluated"] for stage in self.filter_stages
                   if stage["operator"] == EXTRACT)

    @property
    def join_pair_evaluations(self) -> int:
        return sum(stage["evaluated_pairs"] for stage in self.join_stages)

    def assumptions(self) -> dict:
        """Return what the estimate takes as given, for a report."""
        return {
            "persistent_kv_capacity": "unlimited",
            "gpus": 1,
            "survivors": "exact, from the caller's answers and labels",
            "stop_key": (
                "a filter under a DISTINCT over its table's columns reads "
                "only each key's shortest survivor, or every document of "
                "a key with none"),
            "filter_order": (
                "the planner's rule for this query; an alias's "
                "classifications follow its AI.IF filters"),
            "classification": (
                "the reference prompt's tail per document or joined row, "
                "and one readout row at the cue"),
            "extraction": (
                "the plain prompt's tail per document, then the reference "
                "answer verified in one pass after the cue, each token a "
                "readout row"),
            "plan_space": "all feasible eager binary full left deep plans",
            "cached_values": (
                "document prefixes used by filters, classifications, "
                "or as anchors"),
            "cross_alias_prefix_reuse": (
                "an anchor row whose column another alias filtered, or "
                "whose row is live under another cached alias of the "
                "column, pays the frame only"
                if self.credit_shared_prefixes else
                "none; each alias computes its own documents"),
            "shared_prefixes_across_documents": (
                "each distinct token prefix computed once"
                if self.credit_shared_prefixes else
                "each document computed once"),
            "streamed_partner_kv": "not reusable",
            "chunk_tokens": self.chunk_tokens,
        }

    def as_dict(self) -> dict:
        """Return a JSON friendly record of the estimate."""
        latency = self.latency
        return {
            "model": self.model,
            "device": self.device,
            "chunk_tokens": self.chunk_tokens,
            "credit_shared_prefixes": self.credit_shared_prefixes,
            "sol_s": self.seconds,
            "bound_by": latency.bound_by,
            "t_compute": latency.compute,
            "t_memory": latency.memory,
            "passes": latency.passes,
            "bytes_moved": latency.bytes_moved,
            "tokens": self.work.tokens,
            "pairs": self.work.pairs,
            "sliding_pairs": self.work.sliding_pairs,
            "sliding_kv_read": self.work.sliding_kv_read,
            "kv_written": self.work.kv_written,
            "kv_read": self.work.kv_read,
            "components": [
                {
                    "name": component.name,
                    "precision": component.precision,
                    "flops": component.flops,
                    "bytes_moved": component.bytes_moved,
                    "t_compute": component.compute_seconds,
                    "t_memory": component.memory_seconds,
                    "seconds": component.seconds,
                    "bound_by": component.bound_by,
                }
                for component in latency.components
            ],
            "usd_per_hour": self.usd_per_hour,
            "usd_per_query": self.usd_per_query,
            "alias_columns": dict(self.alias_columns),
            "documents_by_alias": dict(self.documents_by_alias),
            "input_document_rows": self.input_document_rows,
            "post_filter_counts": dict(self.post_filter_counts),
            "filter_evaluations": self.filter_evaluations,
            "classification_evaluations": self.classification_evaluations,
            "extraction_evaluations": self.extraction_evaluations,
            "join_pair_evaluations": self.join_pair_evaluations,
            "filter_stages": [dict(stage) for stage in self.filter_stages],
            "join_stages": [dict(stage) for stage in self.join_stages],
            "relation_order": list(self.relation_order),
            "cached_prefixes": list(self.cached_prefixes),
            "search": dict(self.search),
            "assumptions": self.assumptions(),
        }


def _plain(value):
    """A value with its NumPy scalars as Python ones."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _shared_tokens(store, rows=None) -> list[int]:
    """Per document, the prefix tokens an earlier computed document also has.

    With rows, only those documents compute: a document outside rows
    shares nothing, and one in rows shares only with earlier ones in rows.
    """
    if rows is None:
        return [int(tokens) for tokens in prefix_tree(store, 1).shared]
    rows = list(rows)
    tree = prefix_tree([store[row] for row in rows], 1)
    shared = [0] * len(store)
    for row, tokens in zip(rows, tree.shared):
        shared[row] = int(tokens)
    return shared


class _Oracles:
    """The estimate's oracles, as the native module calls them."""

    def __init__(self, answer, label, stores):
        self.has_label = label is not None
        self._answer, self._label, self._stores = answer, label, stores
        self._prompts = {}

    def _prompt(self, text):
        if text not in self._prompts:
            self._prompts[text] = decode_plan_value(text)
        return self._prompts[text]

    def answer(self, prompt, assignment):
        found = self._answer(self._prompt(prompt), dict(json.loads(assignment)))
        return json.dumps(_plain(found))

    def label(self, prompt, assignment):
        return str(self._label(self._prompt(prompt), dict(json.loads(assignment))))

    def accepts(self, condition, values):
        condition = decode_plan_value(condition)
        (column,) = condition_columns(condition)
        kept = accepts(condition, pa.table({column: pa.array(json.loads(values),
                                                             pa.string())}))
        return json.dumps(kept.to_pylist())

    def subset_shared(self, alias, rows):
        return json.dumps(_shared_tokens(self._stores[alias].tokens, json.loads(rows)))


def _estimate_inputs(query, stores, chunk: int, credit: bool) -> dict:
    """What the estimate reads from the query's tables.

    Returns:
        Each scanned table's column, token lengths, and shared prefix
        tokens; each stop key's values per row; and each join's allowed
        pairs, by the rows of each side.
    """
    facts = query.facts()
    aliases = []
    for scan in facts["scans"]:
        store = stores[scan["alias"]]
        tokens = [int(length) for length in store.lengths]
        column = column_name(decode_plan_value(json.dumps(scan["column"])),
                             qualified=False)
        aliases.append({
            "alias": scan["alias"],
            "column": f"{scan['provider']}.{column}",
            "tokens": tokens,
            "shared": _shared_tokens(store.tokens) if credit else [0] * len(tokens)})
    stop_groups = {}
    for alias, key in facts["stop_keys"].items():
        columns = [stores[alias].column(name).to_pylist() for name in key]
        stop_groups[alias] = [_plain(list(values)) for values in zip(*columns)]
    allowed_pairs = {}
    for join in facts["join_conditions"]:
        position = join["position"]
        left_alias, right_alias = join["left_alias"], join["right_alias"]
        conditions = [(decode_plan_value(json.dumps(left)),
                       decode_plan_value(json.dumps(right)))
                      for left, right in join["pairs"]]
        pairs = pair_table(
            left_alias, [evaluate_expression(left, stores[table_alias(left)])
                         for left, _ in conditions],
            right_alias, [evaluate_expression(right, stores[table_alias(right)])
                          for _, right in conditions])
        allowed_pairs[str(position)] = {
            alias: [[int(key), [int(row) for row in groups.rows[
                groups.offsets[index]:groups.offsets[index + 1]]]]
                    for index, key in enumerate(groups.keys)]
            for alias, groups in (
                (left_alias, partner_map(pairs, left_alias, right_alias)),
                (right_alias, partner_map(pairs, right_alias, left_alias)))}
    return {"aliases": aliases, "stop_groups": stop_groups,
            "allowed_pairs": allowed_pairs, "order": query.order,
            "credit_shared": credit, "chunk_tokens": chunk}


def speed_of_light_estimate(
    query,
    answer: AnswerOracle,
    *,
    label: LabelOracle | None = None,
    model: ModelSpec | None = None,
    device: DeviceSpec | None = None,
    chunk_tokens: int | None = None,
    credit_shared_prefixes: bool = True,
) -> SpeedOfLightEstimate:
    """Return the ideal time of one query on one device.

    Args:
        query: A query built through a session. Its session tokenizes
            the scanned columns; the query is not planned or run.
        answer: Callable (prompt, assignment) -> bool giving the exact
            answer of a filter or join prompt for one row assignment,
            where assignment maps each alias in the prompt to a row
            index. Survivors at every stage come from it.
        label: Callable (prompt, assignment) -> str giving the reference
            label of an AI.CLASSIFY prompt for one row assignment. A
            label filter keeps the rows whose label it accepts. Required
            when the query classifies.
        model: Model to price; the session's model by default.
        device: Device to price; the session's device by default.
        chunk_tokens: Forward pass size; the planner's chunk budget for
            the model and device by default.
        credit_shared_prefixes: True computes each distinct token
            prefix once across documents and aliases of one column.
            False computes each alias's document once and reuses it
            only across that document's own questions.

    Raises:
        NotImplementedError: The plan holds an operator the estimate
            does not price, named in the message.
        ValueError: The query classifies and no label oracle was given.
    """
    model = model or query.session.model
    device = device or query.session.device
    chunk = (budgets.chunk_budget(model, device) if chunk_tokens is None
             else chunk_tokens)
    stores = query.token_inputs()
    try:
        inputs = _estimate_inputs(query, stores, chunk, credit_shared_prefixes)
    except (NotImplementedError, ValueError):
        # an operator the estimate cannot price; Rust names it
        inputs = {"aliases": [], "stop_groups": {}, "allowed_pairs": {},
                  "order": query.order, "credit_shared": credit_shared_prefixes,
                  "chunk_tokens": chunk}
    vocabulary = Vocabulary(query.session.tokenizer)
    fields = {"model": asdict(model), "device": asdict(device),
              "gpus": 1, "canvas_draws": 4,
              "document_tokens": {data["alias"]: data["tokens"]
                                  for data in inputs["aliases"]},
              "plan": None, "attention": None}
    found, message = _quail_native.plan_operator(
        "speed_of_light", json.dumps(fields),
        json.dumps({"plan": json.loads(query.logical), "inputs": inputs}),
        vocabulary.tokens, vocabulary.decoder(),
        _Oracles(answer, label, stores))
    if message is not None:
        name, _, text = message.partition(": ")
        raise _ERRORS.get(name, ValueError)(text if name in _ERRORS else message)
    found = json.loads(found)
    work = Work(**found["work"])
    return SpeedOfLightEstimate(
        model=found["model"], device=found["device"],
        chunk_tokens=found["chunk_tokens"],
        credit_shared_prefixes=found["credit_shared_prefixes"],
        work=work,
        latency=SpeedOfLight(work, found["passes"], tuple(
            ComponentLatency(
                name=component["name"], precision=component["precision"],
                flops=component["flops"], bytes_moved=component["bytes_moved"],
                compute_seconds=component["t_compute"],
                memory_seconds=component["t_memory"],
                seconds=component["seconds"], bound_by=component["bound_by"])
            for component in found["components"])),
        seconds=found["sol_s"], usd_per_hour=found["usd_per_hour"],
        alias_columns=found["alias_columns"],
        documents_by_alias=found["documents_by_alias"],
        post_filter_counts=found["post_filter_counts"],
        filter_stages=tuple(found["filter_stages"]),
        join_stages=tuple(found["join_stages"]),
        relation_order=tuple(found["relation_order"]),
        cached_prefixes=tuple(found["cached_prefixes"]),
        search=found["search"])
