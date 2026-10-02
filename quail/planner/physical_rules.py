"""Built in physical optimizer rules.

- kv_retention: the KV later operators read stays resident, within
  the retention pool's page cap: a filter chain keeps its survivors'
  KV for a join anchored on its table, and a join keeps its anchor's
  KV for a later group on the same anchor.
- label_scoring: each classification gets the label scoring rule
  (letters, trie_tree, or trie_decode) whose simulated time is lowest.
- prefix_sharing: documents that share a token prefix with another
  document borrow its KV pages for the shared part instead of
  computing it again.
- tree_attention: each join is annotated with the attention path the
  cost model prefers for its partners' reads of the anchor's KV.
"""

from __future__ import annotations

from dataclasses import replace

from quail.cost.budgets import choose_attention_path
from quail.cost.retention import coefficients
from quail.execution.pipelines import build_pipelines
from quail.logical import classified_above_joins
from quail.physical import (
    AiClassify,
    AiFilter,
    AiJoin,
    Filter,
    Foreign,
    PhysicalGraph,
)
from quail.planner import retention
from quail.planner.prefixes import document_shared_tokens, page_tree
from quail.planner.statistics import cached_statistics, live_after_filters


class KvRetention:
    """Keep the KV later operators read resident, within the page cap.

    The schedule (quail.planner.retention.schedule) records, at each
    execution boundary, which aliases a later stage anchors on and how
    likely each document is to survive to that use; the executor's
    retention pool pins their KV by that priority, up to the pool's
    page cap. A filter chain keeps its survivors' KV (keep_kv) when a
    later join anchors on its table, unless the chain streams into
    that join in one pipeline on one GPU; a join keeps its anchor's KV
    (keep_anchor_kv) when a later group anchors on the same table. A
    chain writes KV pages only when something reads them: a later
    stage, the next operator of its pipeline, or the retention pool.

    Fires once, when the plan is first made: it reads the logical plan
    on the context and leaves a plan that has a schedule alone, so a
    later pass over exact tokens keeps the schedule the plan was made
    with. The schedule, the retention coefficients, and the cap go in
    the plan's ``retention`` setting.
    """

    name = "kv_retention"

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None or context.logical_plan is None \
                or "retention" in context.settings:
            return None
        logical = context.logical_plan
        statistics = cached_statistics(
            logical, context.memo, model=context.model, device=context.device,
            doc_tokens=context.document_tokens,
            pair_fractions=context.pair_fractions)
        joins = [node for node in graph.nodes if isinstance(node, AiJoin)]
        groups = [[(statistics.specs[stage.written_pos], stage.anchor)
                   for stage in node.stages] for node in joins]
        schedule = retention.schedule(
            [stage for group in groups for stage in group],
            live_after_filters(logical, statistics),
            [node.node_id for node in joins])
        ask_aliases = {node.alias for node in graph.nodes
                       if isinstance(node, AiFilter)}
        barriers = [node for node in graph.nodes
                    if isinstance(node, Foreign) and node.kind == "barrier"]
        barrier_aliases = {node.aliases[0] for node in barriers
                           if node.ids != "pairs"}
        for node in joins:
            positions = {stage.written_pos for stage in node.stages}
            if any(barrier.ids == "pairs" and barrier.written_pos in positions
                   for barrier in barriers):
                barrier_aliases.add(node.anchor)
        chained = retention.chained_aliases(
            groups, ask_aliases, barrier_aliases, context.gpu_count)
        schedule["initial"] = {
            alias: use for alias, use in schedule["initial"].items()
            if alias not in chained
        }
        schedule.update(**coefficients(context.model, context.device),
                        cap_pages=statistics.cap_pages)
        nodes = []
        for node in graph.nodes:
            if isinstance(node, AiFilter):
                keep = node.alias in schedule["initial"]
                writes = node.arena_writes or keep
                node = replace(node, keep_kv=keep, arena_writes=writes)
            elif isinstance(node, AiJoin):
                node = replace(node, keep_anchor_kv=(
                    node.anchor in schedule["after"][node.node_id]))
            nodes.append(node)
        context.settings["retention"] = schedule
        return PhysicalGraph(tuple(nodes), graph.root)


class LabelScoring:
    """Pick each classification's label scoring rule by simulated time.

    Fires for every AiClassify whose spec has no scoring rule yet, as
    the Quail planner emits them, and leaves chosen ones alone, so a
    later pass over exact tokens keeps the rule the plan was made
    with. The candidates are the letters rule when the prompt has a
    one-token letter per label, the packed trie under tree attention,
    and a greedy decode over fresh documents (_Table.choose). The
    documents count as resident, with their KV in the arena, when the
    classification continues its table's filter chain on one GPU
    before any join, or follows a classification with the same prompt
    head. A classification of joined rows always uses letters and
    arrives chosen.

    The chosen rule's seconds complete the plan's estimate: the rule
    adds them to the ``search_seconds`` setting, the planner's
    estimate of the plan as planned. The classify_placement logical
    rule prices each candidate plan with this rule applied
    (quail.planner.pricing), so plans are compared with their scoring
    counted and a classification no rule can run refuses the plan. It
    needs the logical plan on the context for the calls' prompts.
    """

    name = "label_scoring"

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
                "label_scoring needs the logical plan on the planning context "
                "to score a classification without a scoring rule")
        from quail.planner.classify import classify_table

        logical = context.logical_plan
        calls = {node.name: node.call for node in logical.operators().classifies}
        after_joins = classified_above_joins(logical.root)
        chosen = {}
        for node in pending:
            table = classify_table(context, node.spec.anchor, node.backend_name)
            resident = _resident(node, graph, context, calls, after_joins, table)
            chosen[node.node_id] = replace(node, spec=table.choose_scoring(
                node.spec, calls[node.spec.name], resident))
        context.settings["search_seconds"] = (
            context.settings.get("search_seconds", 0.0)
            + sum(node.spec.estimated_seconds for node in chosen.values())
            - sum(node.spec.estimated_seconds for node in pending))
        return PhysicalGraph(
            tuple(chosen.get(node.node_id, node) for node in graph.nodes),
            graph.root)


def _resident(node, graph, context, calls, after_joins, table) -> bool:
    """Whether a classification's documents have their KV in the arena.

    True on one GPU when the classification continues its table's
    AI.IF filter chain before any join, through any filters on labels
    and applies between them, or follows a classification whose prompt
    head is the same.
    """
    if context.gpu_count != 1:
        return False
    source = graph.node(node.inputs[0].source.node_id)
    while isinstance(source, (Filter, Foreign)):
        source = graph.node(source.inputs[0].source.node_id)
    if isinstance(source, AiClassify) and source.spec is not None:
        return (table.head(calls[source.spec.name])
                == table.head(calls[node.spec.name]))
    return isinstance(source, AiFilter) and node.spec.anchor not in after_joins


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
    groups = {}
    for doc, parent in enumerate(tree.parent):
        if parent is None:
            continue
        readers, rows = groups.get((parent, tree.shared[doc]), (0, 0))
        groups[(parent, tree.shared[doc])] = (
            readers + 1, rows + lengths[doc] - tree.shared[doc])
    votes = {"tree": 0, "unified": 0}
    for (_, shared), (readers, rows) in groups.items():
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


class PrefixSharing:
    """Let documents borrow the KV pages of a document sharing their prefix.

    Fires for a filter whose documents share whole pages of prefix
    worth more forward-pass time than the page writes the filter takes
    on. The filter then writes pages even when it has one stage, since
    borrowed pages must exist. A join always writes its anchors' KV,
    so it fires for any join whose anchors share a whole page; a
    classification writes its documents' KV the same way.
    """

    name = "prefix_sharing"

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None:
            return None
        nodes = []
        changed = False
        pipelines = build_pipelines(graph) if context.gpu_count == 1 else {}
        for node in graph.nodes:
            if isinstance(node, AiFilter) and not node.share_prefixes:
                lengths = context.document_tokens.get(node.alias)
                store = _token_store(lengths)
                if store is not None and sharing_pays(
                        context.model, context.device,
                        shared_tokens=page_aligned_shared_tokens(store),
                        total_tokens=sum(lengths),
                        writes_pages=node.arena_writes):
                    node = replace(node, share_prefixes=True,
                                   arena_writes=True)
                    changed = True
            elif isinstance(node, AiJoin) and not node.share_prefixes:
                lengths = context.document_tokens.get(node.anchor)
                store = _token_store(lengths)
                if store is not None and sharing_pays(
                        context.model, context.device,
                        shared_tokens=page_aligned_shared_tokens(store),
                        total_tokens=sum(lengths), writes_pages=True):
                    node = replace(node, share_prefixes=True)
                    changed = True
            elif (isinstance(node, AiClassify) and node.spec is not None
                  and not node.spec.share_prefixes):
                lengths = context.document_tokens.get(node.spec.aliases[0])
                store = _token_store(lengths)
                if store is not None and sharing_pays(
                        context.model, context.device,
                        shared_tokens=page_aligned_shared_tokens(store),
                        total_tokens=sum(lengths), writes_pages=True):
                    pipeline = pipelines.get(node.node_id)
                    resident = (pipeline is not None
                                and pipeline.members[0].node_id != node.node_id)
                    node = replace(node, spec=_shared_classify_spec(
                        node, context, store, resident=resident))
                    changed = True
            nodes.append(node)
        if not changed:
            return None
        return PhysicalGraph(tuple(nodes), graph.root)


def _shared_classify_spec(node, context, store, *, resident=False):
    """Enable prefix sharing and update the classification cost estimate."""
    from quail.planner.classify import classify_table

    table = classify_table(context, node.spec.aliases[0], node.backend_name,
                           shared=document_shared_tokens(store))
    return table.reestimate(replace(node.spec, share_prefixes=True), resident=resident)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


class TreeAttention:
    """Annotate each join with the attention path the cost model prefers.

    A join stage has every partner of an anchor reading the anchor's
    KV: under tree attention those reads are stacked into one, under
    unified attention each partner reads the anchor itself. The choice
    is by roofline (choose_attention_path). Tree attention needs the
    fp8 merge kernel and packs no canvas rows, so other models stay
    unified.

    A filter sharing prefixes is annotated from its tree: the
    documents borrowing one parent's pages are that node's readers
    (filter_attention). A filter that does not share has one reader
    per node and stays on the unified path.

    The choice is recomputed on every pass, so a plan first made on
    estimated lengths gets the path its exact tokens call for.
    """

    name = "tree_attention"

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None:
            return None
        model, device = context.model, context.device
        forced = context.attention
        if forced is not None and forced not in ("tree", "unified"):
            raise ValueError(
                f"attention must be 'tree' or 'unified', got {forced!r}")
        tree_available = (model.weight_precision == "fp8"
                          and not model.canvas_tokens)
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
                        node.stages[0].question_tokens if node.stages else 0)
            elif isinstance(node, AiJoin):
                lengths = context.document_tokens.get(node.anchor, ())
                path = "unified"
                if tree_available and node.stages:
                    stage = node.stages[0]
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


def built_in_physical_rules() -> tuple:
    """Return the physical rules registered with the built in registry.

    In order: kv_retention, label_scoring, prefix_sharing, and
    tree_attention. prefix_sharing follows kv_retention because it
    weighs the page writes a chain already makes, and tree_attention
    follows prefix_sharing because a filter's attention path depends
    on the prefixes it shares.
    """
    return (KvRetention(), LabelScoring(), PrefixSharing(), TreeAttention())
