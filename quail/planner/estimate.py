"""The speed of light estimate for one query.

The estimate prices the ideal execution of a query on one device: every
distinct document prefix computed once and kept in KV without limit,
forward passes packed across the whole query, and no host, scheduling,
or kernel overhead. On each alias the AI.IF filters run first in the
planner's order, then its classifications, each followed by the label
filters that test it. Joins are searched over every feasible eager
binary full left deep plan and anchor choice, with exact survivors from
caller supplied answer and label oracles at every step. A
classification of joined rows follows its join on the join's anchor.

A classification is priced as one question per live document: the
reference prompt's tail (the question, the labels by name, and the
answer cue) over the document's prefix, read at the cue's row as a
filter's answer is. The rows a scoring rule appends to read the label
(a letter, a trie, or decode rounds) are an execution choice and are
not counted.

Every operator in the plan is priced or refused: an Apply, an AI.SCORE
predicate or column, a LIMIT, or a join without full semantics raises
NotImplementedError.

docs/content/docs/architecture/optimizer.mdx describes the model. The
work counting and component pricing are the planner's own
(quail.cost.work, quail.cost.sol); the search does not call or
simulate the production planner.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

from quail.cost import budgets
from quail.cost.sol import SpeedOfLight, speed_of_light
from quail.cost.work import Work, ask, scan, triangle
from quail.execution.pairs import pair_table
from quail.logical import (
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
    oriented_join_conditions,
)
from quail.planner.filter_order import default_order_rule, order_filters_indexed
from quail.planner.leftdeep import Extension, optimize_left_deep
from quail.planner.live_rows import PairRelation, exact_live_rows
from quail.planner.prefixes import document_shared_tokens
from quail.planner.statistics import prepare_filter_costs
from quail.specs import DeviceSpec, ModelSpec

# answer(prompt, assignment) -> bool, where assignment maps each alias
# in the prompt to a row index of that alias's corpus table
AnswerOracle = Callable[[object, Mapping[str, int]], bool]
# label(prompt, assignment) -> str, the reference label of an
# AI.CLASSIFY prompt for one row assignment
LabelOracle = Callable[[object, Mapping[str, int]], str]

# the operators of one alias, as the stage records name them
FILTER = "AI.IF"
CLASSIFY = "AI.CLASSIFY"
EXTRACT = "AI.EXTRACT"
LABEL_IN = "IN"

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
            "gpu_count": 1,
            "survivors": "exact, from the caller's answers and labels",
            "filter_order": (
                "the planner's rule for this query; an alias's "
                "classifications follow its AI.IF filters"),
            "classification": (
                "the reference prompt's tail per document or joined row, "
                "read at the answer cue; a scoring rule's extra rows are "
                "not counted"),
            "extraction": (
                "the plain prompt's tail per document, read once at the "
                "cue; the lines answer, the start step, and the span "
                "passes are not counted"),
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


@dataclass
class _AliasData:
    column: str
    tokens: list[int]
    shared: list[int]


def _prompt_token_counts(prompt):
    """Return each alias's partner label and anchor frame tokens, and the tail.

    A join prompt carries the counts; a classification of joined rows
    carries the token ids.
    """
    if prompt.labels:
        labels_by_alias = {
            alias: {"label": label_tokens, "frame": frame_tokens}
            for alias, label_tokens, frame_tokens in prompt.labels}
    else:
        labels_by_alias = {
            alias: {"label": len(label_ids), "frame": len(frame_ids)}
            for alias, label_ids, frame_ids in prompt.label_token_ids}
    if not labels_by_alias or prompt.tail_tokens is None:
        raise ValueError(
            "prompts were bound without a tokenizer; the estimate needs "
            "token counts")
    return labels_by_alias, prompt.tail_tokens


def _refuse_unpriced(logical) -> None:
    """Raise NotImplementedError for any operator the estimate cannot price."""
    for node in logical.walk():
        if not isinstance(node, _PRICED_NODES):
            raise NotImplementedError(
                f"the speed of light estimate does not price {node.type_name}")
        if isinstance(node, Result) and node.limit is not None \
                and not (node.order or node.distinct or node.offset
                         or isinstance(node.input, Aggregate)):
            # a sorted, distinct, offset, or grouped result runs every
            # document, which the estimate prices; a bare LIMIT stops early
            raise NotImplementedError(
                "the speed of light estimate does not price LIMIT")
    operators = logical.operators()
    for alias, predicates in operators.filters.items():
        for predicate in predicates:
            expression = predicate.expression
            if not (isinstance(expression, ModelCall)
                    and expression.kind == "boolean"):
                raise NotImplementedError(
                    f"the speed of light estimate does not price the "
                    f"{type(expression).__name__} filter on {alias!r}; "
                    f"it prices AI.IF and label filters")
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


class _Search:
    """One estimate's state: corpus tokens, the oracles, and the credit.

    The credit is the shared prefix counted as read from KV rather
    than computed, when credit_shared is on.
    """

    def __init__(self, query, answer: AnswerOracle, label: LabelOracle | None,
                 model: ModelSpec, device: DeviceSpec, chunk_tokens: int,
                 credit_shared: bool):
        _refuse_unpriced(query.logical)
        self.answer = answer
        self.label = label
        self.model = model
        self.device = device
        self.chunk_tokens = chunk_tokens
        self.credit_shared = credit_shared
        # a diffusion model answers on canvas rows appended to every
        # evaluation's suffix; a decoder answers on the suffix's last row
        self.canvas = model.canvas_tokens
        self.window = model.sliding_window
        stores = query.token_inputs()
        operators = query.logical.operators()
        self.scans, self.filters, self.joins = (
            operators.scans, operators.filters, operators.joins
        )
        self.labels = operators.labels
        self.label_filters = operators.label_filters
        self.extracts = operators.extracts
        if self.labels.calls and label is None:
            raise ValueError(
                "the query classifies documents; pass label=(prompt, "
                "assignment) -> str giving each row's reference label")
        # classifications of joined rows, by the aliases they read
        self.joined_classifications: dict[frozenset, list] = {}
        for call, _ in self.labels.calls:
            if len(call.aliases()) == 2:
                self.joined_classifications.setdefault(
                    frozenset(call.aliases()), []).append(call)
        self.order = query.order
        self.pre = next((prompt.preamble_tokens for prompt in operators.prompts
                         if prompt.preamble_tokens is not None), 0)
        self.aliases: dict[str, _AliasData] = {}
        for scan_node in self.scans:
            store = stores[scan_node.alias]
            tokens = [int(length) for length in store.lengths]
            shared = (document_shared_tokens(store) if credit_shared
                      else [0] * len(tokens))
            self.aliases[scan_node.alias] = _AliasData(
                column=f"{scan_node.provider}.{scan_node.column}",
                tokens=tokens,
                shared=shared,
            )
        # column -> rows every filtered alias of that column computed
        # as prefixes; another alias of the column may reuse them
        self.computed_rows_by_column: dict[str, set[int]] = {}
        # joins without conditions are absent: every pair
        self.allowed_pairs: dict[int, dict[str, dict[int, set[int]]]] = {}
        for position, join in enumerate(self.joins):
            oriented = oriented_join_conditions(join)
            if oriented is None:
                continue
            left_alias, right_alias, conditions = oriented
            left_keys = [
                stores[left.alias].column(left.column)
                for left, _ in conditions
            ]
            right_keys = [
                stores[right.alias].column(right.column)
                for _, right in conditions
            ]
            pairs = pair_table(left_alias, left_keys, right_alias, right_keys)
            by_side = {left_alias: {}, right_alias: {}}
            for left_row, right_row in zip(
                    pairs.column(left_alias).to_pylist(),
                    pairs.column(right_alias).to_pylist()):
                by_side[left_alias].setdefault(left_row, set()).add(right_row)
                by_side[right_alias].setdefault(right_row, set()).add(left_row)
            self.allowed_pairs[position] = by_side

    # ---- work counting --------------------------------------------

    def first_use(self, alias: str, row: int, suffix: int) -> Work:
        """Compute one document's prefix for the first time in a query.

        The tokens an earlier document already computed are resident,
        so the document pays only for the rest of its prefix and the
        suffix.
        """
        data = self.aliases[alias]
        prefix = self.pre + data.tokens[row]
        shared_tokens = data.shared[row]
        if shared_tokens == 0:
            return scan(prefix, suffix, window=self.window)
        resident = self.pre + shared_tokens
        return ask(resident, prefix - resident + suffix, window=self.window)

    def question_work(self, alias: str, live, tail: int, first: bool) -> Work:
        """Return the work of one question asked of every live document.

        The first question on an alias computes each document's prefix,
        unless credit_shared is on and another alias of the column already
        did. Later questions attach the tail to the resident prefix.
        """
        tokens = self.aliases[alias].tokens
        computed = self.computed_rows_by_column.setdefault(
            self.aliases[alias].column, set())
        suffix = tail + self.canvas
        work = Work()
        for row in live:
            if first and not (self.credit_shared and row in computed):
                work = work + self.first_use(alias, row, suffix)
            else:
                prefix = self.pre + tokens[row]
                work = work + ask(prefix, suffix, window=self.window)
        if first:
            computed.update(live)
        return work

    def join_stage_work(self, anchor, partners, survivors, prompt,
                        resident_rows, cross_resident_rows=(),
                        allowed=None) -> Work:
        """Return one join stage's work.

        Anchor rows in resident_rows have their prefix KV in the arena
        and pay the frame only; the rest scan. cross_resident_rows are
        anchor rows whose document prefix another alias of the same
        column computed; with the shared prefix credit on they pay the
        frame only too. allowed maps an anchor row to the partner rows
        its equality conditions keep; None streams every partner.
        """
        labels_by_alias, tail = _prompt_token_counts(prompt)
        partner_rows = list(itertools.product(
            *[survivors[alias] for alias in partners]))
        # matches JoinStage.runtime_spec: the first label is in the frame
        suffixes = [
            tail + self.canvas
            + sum(self.aliases[alias].tokens[row]
                  + (labels_by_alias[alias]["label"] if index else 0)
                  for index, (alias, row) in enumerate(zip(partners, partner_row)))
            for partner_row in partner_rows
        ]
        all_tokens = sum(suffixes)
        all_triangles = sum(triangle(suffix) for suffix in suffixes)
        frame = labels_by_alias[anchor]["frame"] + (
            labels_by_alias[partners[0]]["label"] if partners else 0)
        resident_rows = set(resident_rows)
        if self.credit_shared:
            resident_rows |= set(cross_resident_rows)
        work = Work()
        for row in survivors[anchor]:
            if allowed is None:
                streamed = suffixes
                suffix_tokens, suffix_triangles = all_tokens, all_triangles
            else:
                mine = allowed.get(row, ())
                streamed = [suffix for suffix, partner_row
                            in zip(suffixes, partner_rows)
                            if partner_row[0] in mine]
                suffix_tokens = sum(streamed)
                suffix_triangles = sum(triangle(suffix) for suffix in streamed)
            prefix = self.pre + self.aliases[anchor].tokens[row]
            work = work + (ask(prefix, frame, window=self.window)
                           if row in resident_rows
                           else self.first_use(anchor, row, frame))
            anchor_prefix = prefix + frame
            sliding_pairs = 0.0
            if self.window:
                sliding_pairs = (
                    self.window * suffix_tokens if anchor_prefix >= self.window - 1
                    else sum(triangle(anchor_prefix + suffix, self.window)
                             - triangle(anchor_prefix, self.window)
                             for suffix in streamed))
            work = work + Work(
                tokens=suffix_tokens,
                pairs=anchor_prefix * suffix_tokens + suffix_triangles,
                kv_written=suffix_tokens,
                kv_read=anchor_prefix,
                sliding_pairs=sliding_pairs,
                sliding_kv_read=(min(anchor_prefix, self.window - 1)
                                 if self.window else 0.0),
            )
        return work

    # ---- filters and classifications --------------------------------

    def run_filters(self):
        """Apply each alias's filters and classifications with exact answers.

        The AI.IF filters run first in the planner's order. Then each
        classification runs once over the live documents, followed by
        the label filters that test it, in written order; the
        classifications no label filter tests follow.

        Returns:
            (survivors by alias, work, stage records).
        """
        rule = (self.order if self.order is not None else
                default_order_rule(self.filters, self.joins)[0])
        survivors = {
            alias: list(range(len(data.tokens)))
            for alias, data in self.aliases.items()}
        work = Work()
        stages = []
        for scan_node in self.scans:
            alias = scan_node.alias
            predicates = self.filters.get(alias, ())
            asks = list(range(len(predicates)))
            calls = [call for call, owner in self.labels.calls
                     if owner == alias and len(call.aliases()) == 1]
            extracts = [node for node in self.extracts if node.alias == alias]
            if not asks and not calls and not extracts:
                continue
            tokens = self.aliases[alias].tokens
            mean_tokens = sum(tokens) / len(tokens) if tokens else 0
            order = [asks[index] for index in order_filters_indexed(
                prepare_filter_costs(
                    [predicates[position] for position in asks],
                    prefix_tokens=self.pre + mean_tokens,
                    model=self.model, device=self.device,
                    chunk_tokens=self.chunk_tokens), rule)] if asks else []
            live = survivors[alias]
            first = True
            for written_pos in order:
                predicate = predicates[written_pos]
                stage_work = self.question_work(
                    alias, live, predicate.prompt.tail_tokens, first)
                first = False
                work = work + stage_work
                passed = [
                    row for row in live
                    if self.answer(predicate.prompt, {alias: row})
                ]
                stages.append(_stage(
                    FILTER, alias, predicate.prompt.template, written_pos,
                    predicate.prompt.tail_tokens, predicate.selectivity,
                    live, passed, stage_work))
                live = passed
            steps = [(test.call, test)
                     for test in self.label_filters.get(alias, ())]
            steps.extend((call, None) for call in calls
                         if call not in self.labels.tests)
            classified = set()
            for call, test in steps:
                if call not in classified:
                    stage_work = self.question_work(
                        alias, live, call.prompt.tail_tokens, first)
                    first = False
                    work = work + stage_work
                    stages.append(_stage(
                        CLASSIFY, alias, call.prompt.template, None,
                        call.prompt.tail_tokens, None, live, live,
                        stage_work, name=self.labels.names[call]))
                    classified.add(call)
                if test is not None:
                    accepted = set(test.values)
                    passed = [
                        row for row in live
                        if self.label(call.prompt, {alias: row}) in accepted
                    ]
                    stages.append(_stage(
                        LABEL_IN, alias, call.prompt.template, test.position,
                        0, test.selectivity, live, passed, Work(),
                        name=self.labels.names[call],
                        accepted=list(test.values)))
                    live = passed
            # an extraction follows the labels; it keeps every document
            for node in extracts:
                stage_work = self.question_work(
                    alias, live, node.call.prompt.tail_tokens, first)
                first = False
                work = work + stage_work
                stages.append(_stage(
                    EXTRACT, alias, node.call.prompt.template, None,
                    node.call.prompt.tail_tokens, None, live, live,
                    stage_work, name=node.name))
            survivors[alias] = live
        return survivors, work, stages

    # ---- joins -----------------------------------------------------

    def anchor_fits(self, prompt, anchor, stage_aliases, live) -> bool:
        """Return whether the longest tuple fits one forward pass."""
        labels_by_alias, tail = _prompt_token_counts(prompt)
        anchor_max = max(
            (self.aliases[anchor].tokens[row] for row in live[anchor]),
            default=0,
        )
        partners = [alias for alias in stage_aliases if alias != anchor]
        prefix = self.pre + anchor_max + labels_by_alias[anchor]["frame"]
        if not live[anchor] or any(not live[partner] for partner in partners):
            return prefix <= self.chunk_tokens
        return (
            prefix
            + tail
            + sum(
                labels_by_alias[partner]["label"]
                + max(
                    (self.aliases[partner].tokens[row]
                     for row in live[partner]),
                    default=0,
                )
                for partner in partners
            )
            <= self.chunk_tokens
        )

    def cross_resident(self, anchor, live, live_cache) -> set[int]:
        """Return anchor rows whose prefix another alias of its column holds.

        A filtered alias computed every row of its column. An alias in
        live_cache holds the prefixes of its live rows; it computed at
        least those, so the credit is conservative.
        """
        column = self.aliases[anchor].column
        rows = set(self.computed_rows_by_column.get(column, ()))
        for other in live_cache:
            if other != anchor and self.aliases[other].column == column:
                rows.update(live[other])
        return rows

    def joined_classification_work(self, call, anchor, partner, live,
                                   relation) -> tuple[Work, dict]:
        """Price one classification of the pairs a join kept, on its anchor.

        Each anchor row with a kept pair attaches the classification's
        anchor note and partner label to its resident prefix, then
        streams one suffix per kept pair: the partner document and the
        tail.
        """
        kept = {}
        for left_row, right_row in relation.pairs:
            anchor_row, partner_row = ((left_row, right_row)
                                       if relation.left == anchor
                                       else (right_row, left_row))
            kept.setdefault(anchor_row, set()).add(partner_row)
        live_partners = set(live[partner])
        anchor_rows = [row for row in live[anchor]
                       if kept.get(row, set()) & live_partners]
        work = self.join_stage_work(
            anchor, [partner], {anchor: anchor_rows, partner: live[partner]},
            call.prompt, anchor_rows, allowed=kept)
        pairs = sum(len(kept[row] & live_partners) for row in anchor_rows)
        return work, {
            "name": self.labels.names[call],
            "template": call.prompt.template,
            "pairs": pairs,
            "fresh_tokens": work.tokens,
        }

    def run_joins(self, base_rows, resident, base_work):
        """Search every feasible left deep plan over the filtered rows.

        Returns:
            The best candidate and the search result it came from.
        """
        joins = self.joins
        edge_relations = []
        edge_aliases = []
        for join in joins:
            left, right = tuple(dict.fromkeys(
                arg.alias for arg in join.prompt.args))
            allowed = self.allowed_pairs.get(len(edge_relations), {}).get(
                left)
            passing = frozenset(
                (left_row, right_row)
                for left_row in base_rows[left]
                for right_row in base_rows[right]
                if (allowed is None or right_row in allowed.get(left_row, ()))
                and self.answer(
                    join.prompt, {left: left_row, right: right_row})
            )
            edge_relations.append(PairRelation(left, right, passing))
            edge_aliases.append(frozenset((left, right)))

        live_cache = {}

        def live_for(active_edges: frozenset[int]):
            if active_edges not in live_cache:
                live_cache[active_edges] = exact_live_rows(
                    base_rows,
                    tuple(edge_relations[index]
                          for index in sorted(active_edges)),
                )
            return live_cache[active_edges]

        def active_inside(relations: frozenset[str]) -> frozenset[int]:
            return frozenset(
                index for index, endpoints in enumerate(edge_aliases)
                if endpoints <= relations
            )

        extension_cache = {}

        def extend(relations, cached, added):
            cache_key = (relations, cached, added)
            if cache_key in extension_cache:
                return extension_cache[cache_key]
            crossing = tuple(
                index for index, endpoints in enumerate(edge_aliases)
                if added in endpoints and endpoints & relations
            )
            if not crossing:
                extension_cache[cache_key] = ()
                return ()
            initial_edges = active_inside(relations)
            extensions = []

            def visit_edges(order, position, active_edges, live, cached_now,
                            work, steps):
                if position == len(order):
                    extensions.append(Extension(
                        work=work, state_property=cached_now,
                        steps=tuple(steps)))
                    return
                edge_index = order[position]
                join = joins[edge_index]
                relation = edge_relations[edge_index]
                stage_aliases = (relation.left, relation.right)
                classifications = self.joined_classifications.get(
                    frozenset(stage_aliases), ())
                for anchor in stage_aliases:
                    if not all(
                            self.anchor_fits(prompt, anchor, stage_aliases, live)
                            for prompt in [join.prompt] + [
                                call.prompt for call in classifications]):
                        continue
                    partners = [alias for alias in stage_aliases
                                if alias != anchor]
                    allowed = self.allowed_pairs.get(edge_index, {}).get(
                        anchor)
                    stage_work = self.join_stage_work(
                        anchor, partners, live, join.prompt,
                        live[anchor] if anchor in cached_now else (),
                        self.cross_resident(anchor, live, cached_now),
                        allowed=allowed,
                    )
                    classified = []
                    for call in classifications:
                        call_work, record = self.joined_classification_work(
                            call, anchor, partners[0], live, relation)
                        stage_work = stage_work + call_work
                        classified.append(record)
                    next_edges = active_edges | {edge_index}
                    next_cache = cached_now | {anchor}
                    passing = sum(
                        1
                        for left_row in live[relation.left]
                        for right_row in live[relation.right]
                        if (left_row, right_row) in relation.pairs
                    )
                    step = {
                        "template": join.prompt.template,
                        "written_pos": edge_index,
                        "added_alias": added,
                        "aliases": list(stage_aliases),
                        "anchor": anchor,
                        "partners": partners,
                        "evaluated_pairs": (
                            math.prod(len(live[alias])
                                      for alias in stage_aliases)
                            if allowed is None else sum(
                                sum(1 for partner_row in live[partners[0]]
                                    if partner_row in allowed.get(row, ()))
                                for row in live[anchor])),
                        "passing_pairs": passing,
                        "fresh_tokens": stage_work.tokens,
                        "classifications": classified,
                        "cached_prefixes_after": sorted(next_cache),
                    }
                    visit_edges(
                        order, position + 1, next_edges,
                        live_for(next_edges), next_cache,
                        work + stage_work, steps + [step],
                    )

            for order in itertools.permutations(crossing):
                visit_edges(order, 0, initial_edges, live_for(initial_edges),
                            cached, Work(), [])
            extension_cache[cache_key] = tuple(extensions)
            return extension_cache[cache_key]

        search = optimize_left_deep(
            tuple(scan_node.alias for scan_node in self.scans),
            frozenset(resident),
            base_work,
            extend,
        )
        if not search.candidates:
            raise ValueError("query join graph has no connected left deep plan")

        def candidate_seconds(candidate):
            return speed_of_light(
                candidate.work, self.model, self.device,
                self.chunk_tokens).seconds

        best = min(
            search.candidates,
            key=lambda candidate: (
                candidate_seconds(candidate),
                candidate.work.tokens,
                candidate.work.pairs,
                candidate.work.kv_written,
                candidate.work.kv_read,
                candidate.relation_order,
            ),
        )
        return best, search


def _stage(operator, alias, template, written_pos, question_tokens,
           selectivity, live, passed, work, **extra) -> dict:
    """Return one alias stage's record."""
    return {
        "operator": operator,
        "alias": alias,
        "template": template,
        "written_pos": written_pos,
        "question_tokens": question_tokens,
        "provided_selectivity": selectivity,
        "evaluated": len(live),
        "passed": len(passed),
        "selectivity": (round(len(passed) / len(live), 6) if live else 0.0),
        "fresh_tokens": work.tokens,
        **extra,
    }


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
    search = _Search(query, answer, label, model, device, chunk_tokens,
                     credit_shared_prefixes)
    survivors, filter_work, filter_stages = search.run_filters()
    resident = frozenset(stage["alias"] for stage in filter_stages
                         if stage["operator"] != LABEL_IN)
    base_rows = {alias: tuple(rows) for alias, rows in survivors.items()}
    best, result = search.run_joins(base_rows, resident, filter_work)
    return SpeedOfLightEstimate(
        model=model.name,
        device=device.name,
        chunk_tokens=chunk_tokens,
        credit_shared_prefixes=credit_shared_prefixes,
        work=best.work,
        latency=speed_of_light(best.work, model, device, chunk_tokens),
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
            "dp_states": result.state_count,
            "dp_records_generated": result.generated_count,
            "dp_final_records": len(result.candidates),
        },
    )
