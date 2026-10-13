"""The plan search of the speed of light estimate, over exact survivors."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Mapping

from quail.catalog.pairs import RowGroups, pair_table, partner_map
from quail.cost.joins import longest_join_input, partner_stream_work
from quail.cost.sol import speed_of_light
from quail.cost.work import Work, ask, scan
from quail.logical import oriented_join_conditions
from quail.planner.filter_order import default_order_rule, order_filters_indexed
from quail.planner.leftdeep import search_left_deep
from quail.planner.live_rows import PairRelation, exact_live_rows
from quail.planner.prefixes import (
    document_shared_tokens,
    document_token_order,
    subset_shared_tokens,
)
from quail.planner.prompt_lowering import lower_prompt
from quail.planner.statistics import filter_stop_keys, prepare_filter_costs
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


@dataclass
class _AliasData:
    column: str
    tokens: list[int]
    shared: list[int]


def _prompt_token_counts(lowered):
    """Return each alias's partner label and anchor frame tokens, and the tail."""
    labels_by_alias = {
        alias: {"label": label_tokens, "frame": frame_tokens}
        for alias, label_tokens, frame_tokens in lowered.labels}
    if not labels_by_alias or lowered.tail_tokens is None:
        raise ValueError("a join prompt is needed to count its pieces")
    return labels_by_alias, lowered.tail_tokens


class Search:
    """One estimate's state: corpus tokens, the oracles, and the credit.

    The credit is the shared prefix counted as read from KV rather
    than computed, when credit_shared is on.
    """

    def __init__(self, query, answer: AnswerOracle, label: LabelOracle | None,
                 model: ModelSpec, device: DeviceSpec, chunk_tokens: int,
                 credit_shared: bool):
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
        tokenizer = query.session.tokenizer
        self._lowered = {}

        def lower(prompt):
            if prompt not in self._lowered:
                self._lowered[prompt] = lower_prompt(prompt, tokenizer)
            return self._lowered[prompt]

        self.lower = lower
        self.pre = next((lowered.preamble_tokens
                         for lowered in map(lower, operators.prompts)
                         if lowered.preamble_tokens is not None), 0)
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
        self.stores = stores
        # alias -> per row, its stop key value, and per row, its rank in
        # sorted token order, for the aliases whose filters stop a key
        # at its first survivor
        self.stop_groups: dict[str, list[tuple]] = {}
        self.token_rank: dict[str, dict[int, int]] = {}
        for alias, key in filter_stop_keys(query.logical).items():
            columns = [stores[alias].column(name).to_pylist() for name in key]
            self.stop_groups[alias] = list(zip(*columns))
            self.token_rank[alias] = {
                row: rank
                for rank, row in enumerate(document_token_order(stores[alias]))}
        # column -> rows every filtered alias of that column computed
        # as prefixes; another alias of the column may reuse them
        self.computed_rows_by_column: dict[str, set[int]] = {}
        # joins without conditions are absent: every pair
        self.allowed_pairs: dict[int, dict[str, RowGroups]] = {}
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
            self.allowed_pairs[position] = {
                left_alias: partner_map(pairs, left_alias, right_alias),
                right_alias: partner_map(pairs, right_alias, left_alias)}

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
        # matches join_request: the first label is in the frame
        suffixes = [
            tail + self.canvas
            + sum(self.aliases[alias].tokens[row]
                  + (labels_by_alias[alias]["label"] if index else 0)
                  for index, (alias, row) in enumerate(zip(partners, partner_row)))
            for partner_row in partner_rows
        ]
        frame = labels_by_alias[anchor]["frame"] + (
            labels_by_alias[partners[0]]["label"] if partners else 0)
        resident_rows = set(resident_rows)
        if self.credit_shared:
            resident_rows |= set(cross_resident_rows)
        work = Work()
        for row in survivors[anchor]:
            if allowed is None:
                streamed = suffixes
            else:
                mine = allowed.get(row, ())
                streamed = [suffix for suffix, partner_row
                            in zip(suffixes, partner_rows)
                            if partner_row[0] in mine]
            prefix = self.pre + self.aliases[anchor].tokens[row]
            work = work + (ask(prefix, frame, window=self.window)
                           if row in resident_rows
                           else self.first_use(anchor, row, frame))
            work = work + partner_stream_work(
                prefix + frame, streamed, window=self.window)
        return work

    # ---- filters and classifications --------------------------------

    def stop_key_rows(self, alias, live, order, predicates) -> list[list[int]]:
        """Return per stage the rows read when each key stops at a survivor.

        A key with a row that survives every stage reads only its
        survivor with the fewest tokens, the earliest in sorted token
        order on a tie: no execution of the stop key reads less. A key
        with no survivor reads each row until the first stage it fails.
        """
        rank = self.token_rank[alias]
        tokens = self.aliases[alias].tokens
        groups: dict[tuple, list[int]] = {}
        for row in live:
            groups.setdefault(self.stop_groups[alias][row], []).append(row)
        reached: list[list[int]] = [[] for _ in order]
        for rows in groups.values():
            survivors = [
                row for row in rows
                if all(self.answer(predicates[written_pos].prompt, {alias: row})
                       for written_pos in order)]
            if survivors:
                shortest = min(survivors, key=lambda row: (tokens[row], rank[row]))
                for stage in reached:
                    stage.append(shortest)
                continue
            for row in rows:
                for index, written_pos in enumerate(order):
                    reached[index].append(row)
                    if not self.answer(predicates[written_pos].prompt,
                                       {alias: row}):
                        break
        return [sorted(rows) for rows in reached]

    def run_filters(self):
        """Apply each alias's filters and classifications with exact answers.

        The AI.IF filters run first in the planner's order. Then each
        classification runs once over the live documents, followed by
        the label filters that test it, in written order; the
        classifications no label filter tests follow. An alias whose
        filters carry a stop key reads only each key's shortest
        survivor, or every document of a key with none.

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
                    chunk_tokens=self.chunk_tokens, lower=self.lower),
                rule)] if asks else []
            live = survivors[alias]
            reached = None
            if order and alias in self.stop_groups:
                reached = self.stop_key_rows(alias, live, order, predicates)
                if self.credit_shared:
                    # a skipped document computes no prefix for another
                    self.aliases[alias].shared = subset_shared_tokens(
                        self.stores[alias], reached[0])
            first = True
            for index, written_pos in enumerate(order):
                if reached is not None:
                    live = reached[index]
                predicate = predicates[written_pos]
                stage_work = self.question_work(
                    alias, live, self.lower(predicate.prompt).tail_tokens, first)
                first = False
                work = work + stage_work
                passed = [
                    row for row in live
                    if self.answer(predicate.prompt, {alias: row})
                ]
                stages.append(_stage(
                    FILTER, alias, predicate.prompt.template, written_pos,
                    self.lower(predicate.prompt).tail_tokens, predicate.selectivity,
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
                        alias, live, self.lower(call.prompt).tail_tokens, first)
                    first = False
                    work = work + stage_work
                    stages.append(_stage(
                        CLASSIFY, alias, call.prompt.template, None,
                        self.lower(call.prompt).tail_tokens, None, live, live,
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
                    alias, live, self.lower(node.call.prompt).tail_tokens, first)
                first = False
                work = work + stage_work
                stages.append(_stage(
                    EXTRACT, alias, node.call.prompt.template, None,
                    self.lower(node.call.prompt).tail_tokens, None, live, live,
                    stage_work, name=node.name))
            survivors[alias] = live
        return survivors, work, stages

    # ---- joins -----------------------------------------------------

    def anchor_fits(self, prompt, anchor, stage_aliases, live) -> bool:
        """Return whether the longest tuple fits one forward pass."""
        labels_by_alias, tail = _prompt_token_counts(prompt)

        def longest(alias):
            return max((self.aliases[alias].tokens[row] for row in live[alias]),
                       default=0)

        partners = [alias for alias in stage_aliases if alias != anchor]
        frame = labels_by_alias[anchor]["frame"]
        if not live[anchor] or any(not live[partner] for partner in partners):
            return self.pre + longest(anchor) + frame <= self.chunk_tokens
        return longest_join_input(
            self.pre, longest(anchor), frame, tail,
            [(labels_by_alias[partner]["label"], longest(partner))
             for partner in partners]) <= self.chunk_tokens

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
            self.lower(call.prompt), anchor_rows, allowed=kept)
        pairs = sum(len(kept[row] & live_partners) for row in anchor_rows)
        return work, {
            "name": self.labels.names[call],
            "template": call.prompt.template,
            "pairs": pairs,
            "fresh_tokens": work.tokens,
        }

    def search_joins_sol(self, base_rows, resident, base_work):
        """Search every feasible left deep plan over the exact filtered rows.

        The planner's search_joins prices expected survivors; this
        prices the rows the oracles kept, on the same search core.

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

        def price_step(edge_index, relations, applied, cached):
            join = joins[edge_index]
            relation = edge_relations[edge_index]
            stage_aliases = (relation.left, relation.right)
            added = edge_aliases[edge_index] - relations
            live = live_for(applied)
            classifications = self.joined_classifications.get(
                frozenset(stage_aliases), ())
            for anchor in stage_aliases:
                if not all(
                        self.anchor_fits(prompt, anchor, stage_aliases, live)
                        for prompt in map(self.lower, [join.prompt] + [
                            call.prompt for call in classifications])):
                    continue
                partners = [alias for alias in stage_aliases if alias != anchor]
                allowed = self.allowed_pairs.get(edge_index, {}).get(anchor)
                stage_work = self.join_stage_work(
                    anchor, partners, live, self.lower(join.prompt),
                    live[anchor] if anchor in cached else (),
                    self.cross_resident(anchor, live, cached),
                    allowed=allowed,
                )
                classified = []
                for call in classifications:
                    call_work, record = self.joined_classification_work(
                        call, anchor, partners[0], live, relation)
                    stage_work = stage_work + call_work
                    classified.append(record)
                next_cache = cached | {anchor}
                passing = sum(
                    1
                    for left_row in live[relation.left]
                    for right_row in live[relation.right]
                    if (left_row, right_row) in relation.pairs
                )
                step = {
                    "template": join.prompt.template,
                    "written_pos": edge_index,
                    "added_alias": next(iter(added), None),
                    "aliases": list(stage_aliases),
                    "anchor": anchor,
                    "partners": partners,
                    "evaluated_pairs": (
                        math.prod(len(live[alias]) for alias in stage_aliases)
                        if allowed is None else sum(
                            sum(1 for partner_row in live[partners[0]]
                                if partner_row in allowed.get(row, ()))
                            for row in live[anchor])),
                    "passing_pairs": passing,
                    "fresh_tokens": stage_work.tokens,
                    "classifications": classified,
                    "cached_prefixes_after": sorted(next_cache),
                }
                yield stage_work, next_cache, step

        search = search_left_deep(
            tuple(scan_node.alias for scan_node in self.scans),
            edge_aliases, frozenset(resident), base_work, price_step)
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
