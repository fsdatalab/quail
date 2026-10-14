"""Rules that propose alternatives or choose implementations by estimated cost."""

from __future__ import annotations

from dataclasses import replace

from quail.cost.budgets import choose_attention_path, tree_attention_allowed
from quail.cost.extract import estimate as extract_estimate
from quail.cost.score import score_fixed_tokens
from quail.physical import (
    AiClassify,
    AiExtract,
    AiFilter,
    AiJoin,
    AiScore,
    PhysicalGraph,
)
from quail.physical.pipelines import build_pipelines
from quail.planner.operators.classify import classify_scoring
from quail.planner.rules.tree_transforms import lift_maps
from quail.planner.sol.prefixes import document_shared_tokens, page_tree
from quail.planner.statistics.tables import summarize_alias


class ShouldLiftMaps:
    """Offer classification and extraction placement above joins on Quail."""

    name = "should_lift_maps"
    target = "logical"
    cost_based = True

    def rewrite(self, root, context):
        # Request backends do not implement map placement above joins.
        if context.backend != "quail":
            return None
        return lift_maps(root)


class ChooseClassifyImplementation:
    """Choose each classification's implementation by estimated time.

    Fires for every AiClassify whose spec has no scoring rule yet and
    leaves chosen ones alone, so a later pass keeps the first choice.
    The candidates are letters, when the prompt has a one-token letter
    per label; trie_tree, under the tree attention path; and
    trie_decode, when no label starts another.
    The documents count as resident, with their KV in the arena, when
    the classification continues its table's filter sequence on one GPU
    before any join, or follows a classification with the same prompt
    head. A classification of joined rows always uses letters and
    arrives chosen.

    Graph refinement updates the plan cost from the chosen specs.
    The rule needs the logical plan on the context for the calls' prompts.
    """

    name = "choose_classify_implementation"
    target = "physical"
    cost_based = True

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None:
            return None
        pending = [node for node in graph.nodes
                   if isinstance(node, AiClassify) and node.spec is not None
                   and not node.spec.scoring]
        if not pending:
            return None
        if context.logical_plan is None:
            raise ValueError(
                "choose_classify_implementation needs the logical plan "
                "on the planning context "
                "to score a classification without a scoring rule")

        logical = context.logical_plan
        calls = {node.name: node.call for node in logical.operators().classifies}
        paths = build_pipelines(graph, context.gpu_count)
        chosen = {}
        for node in pending:
            table = classify_scoring(context, node.spec.anchor)
            resident = paths[node.node_id][0] is not node
            chosen[node.node_id] = replace(node, spec=table.choose_scoring(
                node.spec, calls[node.spec.name], resident))
            if "estimated_fresh_tokens" in context.settings:
                work = table.estimate_spec(chosen[node.node_id].spec, resident).work
                context.settings["estimated_fresh_tokens"] += work.tokens
                context.settings["estimated_attention_pairs"] += work.pairs
        return PhysicalGraph(
            tuple(chosen.get(node.node_id, node) for node in graph.nodes),
            graph.root)


def _token_store(document_tokens):
    """The token store behind an alias's document lengths, or None."""
    return getattr(document_tokens, "_store", None)


def page_aligned_shared_tokens(store) -> int:
    """Tokens a prefix tree over the store saves in whole KV pages."""
    return page_tree(store).shared_tokens


def filter_attention(model, device, store, lengths, question_tokens: int) -> str:
    """The attention path for a filter whose documents share prefixes.

    Documents borrowing the same pages of one parent are that node's
    readers; each reads with its own tokens plus the question. The
    path that wins on the most borrowed tokens wins the filter.
    """
    tree = page_tree(store)
    votes = {"tree": 0, "unified": 0}
    for shared, readers, rows in zip(*tree.reader_groups(lengths)):
        path = choose_attention_path(
            model, device, readers=readers,
            reader_rows=rows / readers + question_tokens, node_tokens=shared)
        votes[path] += readers * shared
    return "tree" if votes["tree"] > votes["unified"] else "unified"


def sharing_pays(model, device, *, shared_tokens: int, total_tokens: int,
                 writes_pages: bool) -> bool:
    """Whether borrowing shared prefixes saves more than it costs.

    Borrowing saves the forward pass over the shared tokens. For a
    filter that did not write KV pages, it costs writing them for every
    token of the corpus.
    """
    if shared_tokens <= 0:
        return False
    saved = shared_tokens * 2.0 * model.params / device.peak_flops
    cost = (0.0 if writes_pages
            else total_tokens * model.kappa / device.hbm_bw)
    return saved > cost


class ShouldSharePrefixes:
    """Let documents borrow the KV pages of a document sharing their prefix.

    Fires for a filter whose documents share whole pages of prefix
    worth more forward-pass time than the page writes the filter takes
    on. The filter then writes pages even when it has one stage, since
    borrowed pages must exist. A join always writes its anchors' KV,
    so it fires for any join whose anchors share a whole page; a
    classification or extraction writes its documents' KV the same way.
    A score of
    one table is priced like a filter that writes no pages, since
    without sharing its documents follow one shared prompt head.
    """

    name = "should_share_prefixes"
    target = "physical"
    cost_based = True

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None:
            return None
        nodes = []
        changed = False
        pipelines = build_pipelines(graph, context.gpu_count)
        summaries = {
            alias: summarize_alias(lengths, context.model.sliding_window,
                                   memo=context.memo)
            for alias, lengths in context.document_tokens.items()
        }
        shared_by_alias = {}
        for node in graph.nodes:
            settings = getattr(node, "spec", node)
            if (not isinstance(node, (AiFilter, AiJoin, AiClassify, AiExtract, AiScore))
                    or settings is None or settings.share_prefixes
                    or (isinstance(node, AiScore) and len(settings.aliases) != 1)):
                nodes.append(node)
                continue
            alias = (settings.aliases[0] if isinstance(node, AiScore)
                     else node.document_aliases[0])
            store = _token_store(context.document_tokens.get(alias))
            if store is None:
                nodes.append(node)
                continue
            if alias not in shared_by_alias:
                shared_by_alias[alias] = page_aligned_shared_tokens(store)
            shared = shared_by_alias[alias]
            writes_pages = (node.arena_writes if isinstance(node, AiFilter)
                            else not isinstance(node, AiScore))
            if not sharing_pays(
                    context.model, context.device, shared_tokens=shared,
                    total_tokens=summaries[alias].total, writes_pages=writes_pages):
                nodes.append(node)
                continue
            if isinstance(node, AiFilter):
                node = replace(node, share_prefixes=True, arena_writes=True)
            elif isinstance(node, AiJoin):
                node = replace(node, share_prefixes=True)
            elif isinstance(node, AiScore):
                node = replace(node, spec=_shared_score_spec(
                    node.spec, shared, summaries[alias], context.model.canvas_tokens))
            else:
                pipeline = pipelines.get(node.node_id)
                resident = pipeline is not None and pipeline[0].node_id != node.node_id
                prepare = (_shared_classify_spec if isinstance(node, AiClassify)
                           else _shared_extract_spec)
                node = replace(node, spec=prepare(
                    node, context, store, resident=resident))
            nodes.append(node)
            changed = True
        if not changed:
            return None
        return PhysicalGraph(tuple(nodes), graph.root)


def _shared_classify_spec(node, context, store, *, resident=False):
    """Enable prefix sharing and update the classification cost estimate."""
    table = classify_scoring(context, node.spec.aliases[0],
                           shared=document_shared_tokens(store))
    return table.reestimate(replace(node.spec, share_prefixes=True), resident=resident)


def _shared_extract_spec(node, context, store, *, resident=False):
    """Enable extraction prefix sharing and price the borrowed document KV."""
    from quail.cost.budgets import chunk_budget

    spec = node.spec
    head, tail = spec.prompt_token_parts
    cost = extract_estimate(
        spec.expected_inputs, context.document_tokens[spec.alias],
        len(head), len(tail), chunk=chunk_budget(context.model, context.device),
        model=context.model, device=context.device, resident=resident,
        shared=page_tree(store).shared, max_answer_tokens=spec.max_answer_tokens,
        draft_tokens=spec.draft_tokens, workers=context.gpu_count)
    return replace(spec, share_prefixes=True, estimated_seconds=cost.seconds)


def _shared_score_spec(spec, shared_tokens, summary, canvas_tokens):
    """Enable prefix sharing and scale the score's estimate by its fresh tokens.

    Assumes time is proportional to the tokens the score computes, and
    that the scored documents borrow the same share of their tokens as
    the whole table.
    """
    fixed = score_fixed_tokens(tuple(map(len, spec.prompt_token_parts)),
                               canvas_tokens, spec.draws)
    total = summary.total + summary.count * fixed
    fresh = max(0.0, 1.0 - shared_tokens / total) if total else 1.0
    return replace(spec, share_prefixes=True,
                   estimated_seconds=spec.estimated_seconds * fresh)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


class ChooseAttentionAlgorithm:
    """Annotate each join with the attention path the cost model prefers.

    A join stage has every partner of an anchor reading the anchor's
    KV: under tree attention those reads are stacked into one, under
    unified attention each partner reads the anchor itself. The choice
    is by roofline (choose_attention_path). A diffusion model and a
    model under 2B dense parameters stay unified
    (tree_attention_allowed).

    A filter sharing prefixes is annotated from its tree: the
    documents borrowing one parent's pages are that node's readers
    (filter_attention). A filter that does not share has one reader
    per node and stays on the unified path.

    The choice is recomputed on every pass, so a plan first made on
    estimated lengths gets the path its exact tokens call for.
    """

    name = "choose_attention_algorithm"
    target = "physical"
    cost_based = True

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None:
            return None
        model, device = context.model, context.device
        forced = context.attention
        if forced is not None and forced not in ("tree", "unified"):
            raise ValueError(
                f"attention must be 'tree' or 'unified', got {forced!r}")
        tree_available = tree_attention_allowed(model)
        nodes = []
        changed = False
        for node in graph.nodes:
            path = None
            if isinstance(node, AiFilter):
                lengths = context.document_tokens.get(node.alias, ())
                store = _token_store(lengths)
                path = "unified"
                if tree_available and node.share_prefixes and store is not None:
                    path = filter_attention(
                        model, device, store, lengths,
                        node.question_tokens)
            elif isinstance(node, AiJoin):
                lengths = context.document_tokens.get(node.anchor, ())
                path = "unified"
                if tree_available:
                    stage = node
                    readers = (stage.expected_tuples / len(lengths)
                               if len(lengths) else 1.0)
                    # a partner's rows: its document and the question tail
                    rows = stage.pair_tail_tokens + sum(
                        _mean(context.document_tokens.get(partner, ()))
                        for partner in stage.partners)
                    path = choose_attention_path(
                        model, device, readers=readers, reader_rows=rows,
                        node_tokens=_mean(lengths))
            if path is not None and forced is not None:
                path = forced
            if path is not None and path != node.attention:
                node = replace(node, attention=path)
                changed = True
            nodes.append(node)
        if not changed:
            return None
        return PhysicalGraph(tuple(nodes), graph.root)
