"""Price a physical plan: each node on its own, and the plan as a whole.

A node's price is the work its forward passes do, or the seconds its
own estimator gives. The plan's seconds pack every node's work into
forward passes together and add the seconds-priced nodes, so node
prices do not add up to the plan's. A node that borrows its documents'
shared prefix (share_prefixes) is priced over their fresh tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

from quail.cost import budgets
from quail.cost.extract import estimate as extract_estimate
from quail.cost.score import score_fixed_tokens
from quail.cost.sol import speed_of_light
from quail.cost.work import Work, ask, scan
from quail.physical import (
    AiClassify,
    AiExtract,
    AiFilter,
    AiInference,
    AiJoin,
    AiScore,
    PhysicalGraph,
)
from quail.physical.pipelines import build_pipelines
from quail.planner.operators.classify import classify_scoring
from quail.planner.statistics.prefixes import page_tree
from quail.planner.statistics.tables import summarize_alias


@dataclass(frozen=True)
class NodePrice:
    """One node's price: its packed work, or its own estimated seconds."""

    work: Work | None = None
    seconds: float = 0.0


@dataclass(frozen=True)
class PlanPricer:
    """Price the nodes of a graph from the plan's statistics.

    Attributes:
        context: The planning context, with the model, device, and
            document tokens.
        statistics: The plan's PlanStatistics.
        stage_works: Join written position -> the Work its stage does.
        expected_rows: alias -> documents expected after its filters.
    """

    context: object
    statistics: object
    stage_works: dict
    expected_rows: dict

    def __call__(self, graph: PhysicalGraph) -> tuple[float, dict]:
        """Price a graph: its seconds, and node id -> its explain entry.

        An entry holds the node's seconds and, for the last filter of
        a table a join anchors on, what releasing its KV would cost.
        """
        work, seconds, entries = Work(), 0.0, {}
        for node_id, (price, entry) in self._priced(graph).items():
            entries[node_id] = entry
            if price.work is not None:
                work += price.work
            else:
                seconds += price.seconds
        return self._seconds(work) + seconds, entries

    def _priced(self, graph) -> dict:
        pipelines = build_pipelines(graph)
        anchored = {node.anchor for node in graph.nodes if isinstance(node, AiJoin)}
        last_filters = {node.alias: node.node_id for node in graph.nodes
                        if isinstance(node, AiFilter)}
        # alias -> the document head of the KV a finished pipeline kept
        retained = {}
        out = {}
        for node in graph.topological_nodes():
            entry = {}
            resident = (node.node_id in pipelines
                        and pipelines[node.node_id][0] is not node)
            if node.node_id in pipelines and not resident:
                (alias,) = node.document_aliases
                resident = retained.get(alias) == node.document_head
                members = pipelines[node.node_id]
                if any(isinstance(member, (AiFilter, AiJoin, AiInference))
                       and member.keep_kv for member in members):
                    retained[alias] = node.document_head
                else:
                    retained.pop(alias, None)
            if isinstance(node, AiFilter):
                price = NodePrice(work=self._filter_work(node, resident) * (
                    1 / budgets.parallel_copies(self.context.gpus,
                                                node.expected_rows)))
                entry["seconds"] = self._seconds(price.work)
                if node.alias in anchored and last_filters[node.alias] == node.node_id:
                    entry.update(self._release_recompute(node))
            elif isinstance(node, AiJoin):
                price = NodePrice(
                    work=self.stage_works.get(node.written_position, Work())
                )
                entry["seconds"] = self._seconds(price.work)
            elif isinstance(node, (AiScore, AiClassify, AiExtract)) \
                    and node.spec is not None:
                price = NodePrice(seconds=self._inference_seconds(node, resident))
                entry["seconds"] = price.seconds
            else:
                price = NodePrice()
            out[node.node_id] = (price, entry)
        return out

    def _seconds(self, work: Work) -> float:
        return speed_of_light(work, self.context.model, self.context.device,
                              self.statistics.chunk).seconds

    def _filter_work(self, node: AiFilter, resident: bool) -> Work:
        """A filter's work: its question over resident KV, else a scan of each."""
        window = self.context.model.sliding_window
        question = node.question_tokens
        mean_prefix = self._mean_prefix(node)
        if resident:
            return ask(mean_prefix, question, window=window) * node.expected_rows
        shared = self._shared(node, node.alias)
        if shared is None:
            return scan(mean_prefix, question, window=window) * node.expected_rows
        lengths = self.context.document_tokens[node.alias]
        if not len(lengths):
            return Work()
        work = Work()
        for length, shared in zip(lengths, shared):
            prefix = node.preamble_tokens + int(length)
            # a borrowed prefix includes the prompt head the documents share
            borrowed = shared + node.preamble_tokens if shared else 0
            work += (ask(borrowed, prefix + question - borrowed, window=window)
                     if shared else scan(prefix, question, window=window))
        return work * (node.expected_rows / len(lengths))

    def _mean_prefix(self, node: AiFilter) -> float:
        return (
            node.preamble_tokens
            + self.statistics.stats[node.alias].mean_document_tokens
        )

    def _release_recompute(self, node: AiFilter) -> dict:
        """What a later join pays if the sequence's KV is released, not pinned."""
        prefix = self._mean_prefix(node)
        pages = budgets.pages(prefix)
        fits = self.statistics.cap_pages // max(1, pages)
        excess = max(0.0, self.expected_rows.get(node.alias, 0.0) - fits)
        return {
            "release_recompute_tokens": round(excess * prefix),
            "release_recompute_seconds": self._seconds(
                scan(prefix, 0, window=self.context.model.sliding_window) * excess),
        }

    def _inference_seconds(self, node, resident: bool) -> float:
        """An inference node's seconds: its spec's, scaled by the share it computes.

        A node that borrows its documents' shared prefix keeps its
        spec's estimate as the price of computing everything, and pays
        the fraction its estimator gives for the fresh tokens.
        """
        spec = node.spec
        alias = spec.aliases[0] if len(spec.aliases) == 1 else None
        shared = None if alias is None else self._shared(node, alias)
        if shared is None or not spec.estimated_seconds:
            return spec.estimated_seconds
        return spec.estimated_seconds * self._fresh_fraction(node, alias, shared,
                                                             resident)

    def _fresh_fraction(self, node, alias: str, shared, resident: bool) -> float:
        context, spec = self.context, node.spec
        if isinstance(node, AiExtract):
            head, tail = spec.prompt_token_parts

            def seconds(shared):
                return extract_estimate(
                    spec.expected_rows, context.document_tokens[alias], len(head),
                    len(tail), chunk=self.statistics.chunk, model=context.model,
                    device=context.device, resident=resident, shared=shared,
                    max_answer_tokens=spec.max_answer_tokens,
                    draft_tokens=spec.draft_tokens, draft_policy=spec.draft_policy,
                    round_cost_per_token=spec.round_cost_per_token,
                    gpus=context.gpus, memo=context.memo).seconds

            whole = seconds(None)
            return seconds(shared) / whole if whole else 1.0
        if isinstance(node, AiClassify):
            def seconds(shared):
                return classify_scoring(context, alias, shared=shared).estimate_spec(
                    spec, resident).seconds

            whole = seconds(())
            return seconds(shared) / whole if whole else 1.0
        # a score's time is taken as proportional to the tokens it
        # computes, its documents borrowing the table's share of theirs
        summary = summarize_alias(context.document_tokens[alias],
                                  context.model.sliding_window, memo=context.memo)
        fixed = score_fixed_tokens(tuple(map(len, spec.prompt_token_parts)),
                                   context.model.canvas_tokens, spec.draws)
        total = summary.total + summary.count * fixed
        return max(0.0, 1.0 - sum(shared) / total) if total else 1.0

    def _store(self, alias: str):
        lengths = self.context.document_tokens.get(alias)
        return None if lengths is None else lengths.store

    def _shared(self, node, alias: str):
        """Per document, the prefix tokens a node borrows, or None without sharing."""
        store = self._store(alias)
        if not node.share_prefixes or store is None:
            return None
        return page_tree(store).shared

