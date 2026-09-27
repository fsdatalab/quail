"""Built in physical optimizer rules.

- prefix_sharing: documents that share a token prefix with another
  document borrow its KV pages for the shared part instead of
  computing it again.
- tree_attention: each join is annotated with the attention path the
  cost model prefers for its partners' reads of the anchor's KV.
"""

from __future__ import annotations

from dataclasses import replace

from quail.cost.budgets import PAGE_TOKENS, choose_attention_path
from quail.physical import AiFilter, AiJoin, PhysicalGraph
from quail.planner.prefixes import prefix_credits


def _token_store(document_tokens):
    """The token store behind an alias's document lengths, or None."""
    return getattr(document_tokens, "_store", None)


def page_aligned_shared_tokens(store, page_tokens: int = PAGE_TOKENS) -> int:
    """Tokens a prefix tree over the store saves in whole KV pages."""
    return sum(credit // page_tokens * page_tokens
               for credit in prefix_credits(store))


# CPU seconds per document to sort a corpus and take prefixes, measured
# on the worker (13 us per document on AGENT); the GPU idles meanwhile
TREE_SECONDS_PER_DOCUMENT = 20e-6


def sharing_pays(model, device, *, shared_tokens: int, total_tokens: int,
                 n_docs: int, writes_pages: bool) -> bool:
    """Whether borrowing shared prefixes saves more than it costs.

    Borrowing saves the forward pass over the shared tokens. It costs
    the tree build on the worker, and, for a filter that did not write
    KV pages, writing them for every token of the corpus.
    """
    if shared_tokens <= 0:
        return False
    saved = shared_tokens * 2.0 * model.params / device.peak_flops
    cost = n_docs * TREE_SECONDS_PER_DOCUMENT
    if not writes_pages:
        cost += total_tokens * model.kappa / device.hbm_bw
    return saved > cost


class PrefixSharing:
    """Let documents borrow the KV pages of a document sharing their prefix.

    Fires for a filter whose documents share whole pages of prefix
    worth more forward-pass time than the page writes the filter takes
    on. The filter then writes pages even when it has one stage, since
    borrowed pages must exist.
    """

    name = "prefix_sharing"

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None:
            return None
        nodes = []
        changed = False
        for node in graph.nodes:
            if isinstance(node, AiFilter) and not node.share_prefixes:
                lengths = context.document_tokens.get(node.alias)
                store = _token_store(lengths)
                if store is not None and sharing_pays(
                        context.model, context.device,
                        shared_tokens=page_aligned_shared_tokens(store),
                        total_tokens=sum(lengths), n_docs=len(lengths),
                        writes_pages=node.arena_writes):
                    node = replace(node, share_prefixes=True,
                                   arena_writes=True)
                    changed = True
            nodes.append(node)
        if not changed:
            return None
        return PhysicalGraph(tuple(nodes), graph.root)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


class TreeAttention:
    """Annotate each join with the attention path the cost model prefers.

    A join stage has every partner of an anchor reading the anchor's
    KV: under tree attention those reads are stacked into one, under
    unified attention each partner reads the anchor itself. The choice
    is by roofline (choose_attention_path). Tree attention needs the
    two-call path, which the fp8 models without canvas rows run.

    A filter is not annotated: its first stage packs fresh documents,
    whose rows must read any borrowed pages in the same call, and its
    later stages have one question tail per document, one reader per
    node, for which unified always wins.

    Measured on LEP-4 and FEV-4 (experiments/join_attention_paths.py):
    unified is 2 to 3 percent faster where the rule picks it and tied
    elsewhere, with equal agreement against vLLM's answers.
    """

    name = "tree_attention"

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None:
            return None
        model, device = context.model, context.device
        two_call = model.weight_precision == "fp8" and not model.canvas_tokens
        nodes = []
        changed = False
        for node in graph.nodes:
            if isinstance(node, AiJoin) and not node.attention:
                lengths = context.document_tokens.get(node.anchor, ())
                path = "unified"
                if two_call and node.stages:
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
                node = replace(node, attention=path)
                changed = True
            nodes.append(node)
        if not changed:
            return None
        return PhysicalGraph(tuple(nodes), graph.root)


def built_in_physical_rules() -> tuple:
    """Return the physical rules registered with the built in registry."""
    return (PrefixSharing(), TreeAttention())
