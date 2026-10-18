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

Every operator in the plan is priced or refused: an Apply, an AI.SCORE
predicate or column, a LIMIT, or a join without full semantics raises
NotImplementedError.

docs/content/docs/architecture/optimizer.mdx describes the model. The
work counting and component pricing are the planner's own
(quail.cost.work, quail.cost.sol); the search does not call or
simulate the production planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quail.cost import budgets
from quail.cost.sol import SpeedOfLight
from quail.cost.work import Work
from quail.logical import (
    SPAN_SUFFIX,
    Aggregate,
    Filter,
    Join,
    ModelCall,
    Project,
    Result,
    Scan,
    SemanticClassify,
    SemanticExtract,
    SemanticFilter,
    SemanticJoin,
)
from quail.planner.sol.search import (
    ANSWER_TEST,
    CLASSIFY,
    EXTRACT,
    FILTER,
    LABEL_IN,
    AnswerOracle,
    LabelOracle,
    Search,
)
from quail.specs import DeviceSpec, ModelSpec

_PRICED_NODES = (Scan, SemanticFilter, SemanticClassify, SemanticExtract,
                 Filter, Join, SemanticJoin, Project, Aggregate, Result)


@dataclass(frozen=True)
class SpeedOfLightEstimate:
    """The ideal time of one query and how it was reached."""

    model: str
    device: str
    chunk_tokens: int
    credit_shared_prefixes: bool
    work: Work
    latency: SpeedOfLight
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
    def seconds(self) -> float:
        return self.latency.seconds

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
            "sol_s": latency.seconds,
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


def _refuse_unpriced(logical) -> None:
    """Raise NotImplementedError for any operator the estimate cannot price."""
    for node in logical.walk():
        if not isinstance(node, _PRICED_NODES):
            raise NotImplementedError(
                f"the speed of light estimate does not price {node.type_name}")
        if isinstance(node, Result) and node.limit is not None \
                and not (node.order or node.distinct or node.offset
                         or isinstance(node.input, Aggregate)):
            # a sorted, offset, or grouped result runs every document,
            # and a distinct one stops each key at its first survivor;
            # the estimate prices both. A bare LIMIT stops early.
            raise NotImplementedError(
                "the speed of light estimate does not price LIMIT")
    operators = logical.operators()
    for alias, predicates in operators.filters.items():
        for predicate in predicates:
            expression = predicate.expression
            if not (isinstance(expression, ModelCall)
                    and expression.kind == "boolean"):
                raise NotImplementedError(
                    f"the speed of light estimate does not price the filter "
                    f"{expression} on {alias!r}; it prices AI.IF and label "
                    f"filters")
    for tests in operators.extract_filters.values():
        for test in tests:
            if test.condition.column.column.endswith(SPAN_SUFFIX):
                raise NotImplementedError(
                    f"the speed of light estimate does not price the test "
                    f"{test.condition} of an answer's span; the answer "
                    f"oracle gives the answer's text only")
    for join in operators.joins:
        if not (isinstance(join.predicate, ModelCall)
                and join.predicate.kind == "boolean"):
            raise NotImplementedError(
                f"the speed of light estimate does not price a join on "
                f"{type(join.predicate).__name__}; it prices AI.IF joins")
        if join.semantics != "full":
            raise NotImplementedError(
                f"the speed of light estimate does not price "
                f"{join.semantics!r} joins; it prices full joins")
        if len({arg.alias for arg in join.prompt.args}) != 2:
            raise NotImplementedError(
                "the speed of light estimate prices binary join predicates")
    for column in operators.projections:
        if column.expression.kind not in ("label", "extract"):
            raise NotImplementedError(
                f"the speed of light estimate does not price the "
                f"{column.expression.kind!r} column {column.name!r}; it "
                f"prices AI.CLASSIFY and AI.EXTRACT columns")
    joined = [frozenset(arg.alias for arg in join.prompt.args)
              for join in operators.joins]
    for call, _ in operators.labels.calls:
        aliases = call.aliases()
        if len(aliases) == 1:
            continue
        name = operators.labels.names[call]
        if len(aliases) != 2 or frozenset(aliases) not in joined:
            raise NotImplementedError(
                f"the classification {name!r} reads {list(aliases)}; the "
                f"speed of light estimate prices a classification of one "
                f"document or of one join's rows")
        if operators.labels.tests.get(call):
            raise NotImplementedError(
                f"the speed of light estimate does not price a label "
                f"filter on {name!r}, a classification of joined rows")


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
    if chunk_tokens is None:
        chunk_tokens = budgets.chunk_budget(model, device)
    _refuse_unpriced(query.logical)
    search = Search(query, answer, label, model, device, chunk_tokens,
                    credit_shared_prefixes)
    survivors, filter_work, filter_stages = search.run_filters()
    resident = frozenset(stage["alias"] for stage in filter_stages
                         if stage["operator"] not in (LABEL_IN, ANSWER_TEST))
    base_rows = {alias: tuple(rows) for alias, rows in survivors.items()}
    best, result = search.search_joins_sol(base_rows, resident, filter_work)
    return SpeedOfLightEstimate(
        model=model.name,
        device=device.name,
        chunk_tokens=chunk_tokens,
        credit_shared_prefixes=credit_shared_prefixes,
        work=best.work,
        latency=search.latency(best),
        usd_per_hour=device.usd_per_hour,
        alias_columns={
            alias: data.column for alias, data in search.aliases.items()},
        documents_by_alias={
            alias: len(data.tokens) for alias, data in search.aliases.items()},
        post_filter_counts={
            alias: len(rows) for alias, rows in survivors.items()},
        filter_stages=tuple(filter_stages),
        join_stages=tuple(best.steps),
        relation_order=tuple(best.relation_order),
        cached_prefixes=tuple(sorted(best.state_property)),
        search={
            "dp_states": result.num_states,
            "dp_records_generated": result.num_generated,
            "dp_final_records": len(result.candidates),
        },
    )
