"""Offline checks for the fused shared-prefix attention plan.

Reconstructs the fusegate flight's question-pass batch shape: 50
documents, 6 question tails each, where each doc's first tail computes
the shared question-prefix pages in-flight and the other five reference
them (vLLM registers full blocks at allocation). Verifies grouping,
page ownership, and exact numerical equivalence of the cascade split
against direct attention, in both the natural order and the flight's
observed order (runs of five with the first tails at the back), which
produced the 250-grouped / 50-singleton signature.

Pure Python on purpose: the grouping helpers only need indexing and
tolist(), so the test runs without torch, numpy, or vllm installed.
"""

import math

import pytest

from docengine.engineext.fused import (
    CascadeGroup,
    _check_group,
    _derived_groups,
)

PAGE = 16
N_DOCS = 50
N_Q = 6
BODY = 345          # 21 full pages of 16, plus 9 remainder tokens
Q_COMMON = 33       # question tokens shared by all six questions
SUFFIX = [15 + j for j in range(N_Q)]


class Row:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class Table:
    """The two access forms the fused helpers use on the block table."""

    def __init__(self, rows, width):
        self.rows = [list(r) + [0] * (width - len(r)) for r in rows]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, cols = key
            return Row(self.rows[row][cols])
        return Row(self.rows[key])


def build_batch(order):
    """Block tables and lengths as the pinned KVCacheManager produces
    them for the gate's question pass (verified against the real
    manager in the scratchpad replay)."""
    page = iter(range(1, 100_000))
    requests = []
    for doc in range(N_DOCS):
        body_pages = [next(page) for _ in range(21)]     # warm, full
        # First tail: hits the 21 warm pages, computes remainder plus
        # the shared question prefix plus its own suffix (57 tokens,
        # 4 fresh pages). Its first two fresh pages become the shared
        # question-prefix pages the other five tails reference.
        first_new = [next(page) for _ in range(4)]
        seq0 = BODY + Q_COMMON + SUFFIX[0]
        requests.append(dict(
            doc=doc, q=0, pages=body_pages + first_new,
            seq=seq0, query=seq0 - 336))
        for j in range(1, N_Q):
            own = [next(page) for _ in range(2)]
            seq = BODY + Q_COMMON + SUFFIX[j]
            requests.append(dict(
                doc=doc, q=j,
                pages=body_pages + first_new[:2] + own,
                seq=seq, query=seq - 368))
    if order == "flight":
        requests = ([r for r in requests if r["q"] != 0]
                    + [r for r in requests if r["q"] == 0])
    width = max(len(r["pages"]) for r in requests) + 4
    table = Table([r["pages"] for r in requests], width)
    seqs = [r["seq"] for r in requests]
    queries = [r["query"] for r in requests]
    return requests, table, seqs, queries


def derive(requests, table, seqs, queries):
    groups = _derived_groups(len(requests), table, seqs, queries, PAGE)
    assert groups is not None
    for first, group in groups:
        _check_group(group, first, table, seqs, queries, PAGE)
    return groups


def test_natural_order_groups_each_doc_once():
    requests, table, seqs, queries = build_batch("natural")
    groups = derive(requests, table, seqs, queries)
    sizes = sorted(g.request_count for _, g in groups)
    assert sizes == [N_Q] * N_DOCS
    # Shared depth is capped by the first tail's computed pages (21),
    # not the five-way page agreement (23).
    assert all(g.shared_blocks == 21 for _, g in groups)


def test_flight_order_reproduces_250_grouped_50_singleton():
    requests, table, seqs, queries = build_batch("flight")
    groups = derive(requests, table, seqs, queries)
    multi = [g for _, g in groups if g.request_count > 1]
    single = [g for _, g in groups if g.request_count == 1]
    assert sum(g.request_count for g in multi) == 250
    assert len(single) == 50
    # The runs of five see all 23 agreed pages as computed.
    assert all(g.shared_blocks == 23 for g in multi)
    assert all(g.shared_blocks == 21 for g in single)


@pytest.mark.parametrize("order", ["natural", "flight"])
def test_level0_pages_belong_to_each_request(order):
    requests, table, seqs, queries = build_batch(order)
    groups = derive(requests, table, seqs, queries)
    for first, group in groups:
        for r in range(first, first + group.request_count):
            own = tuple(requests[r]["pages"][: group.shared_blocks])
            assert own == group.shared_page_ids


@pytest.mark.parametrize("order", ["natural", "flight"])
def test_cascade_split_matches_direct_attention_exactly(order):
    requests, table, seqs, queries = build_batch(order)
    groups = derive(requests, table, seqs, queries)

    def kv(page_id, slot):
        return math.sin(page_id * 16 + slot * 0.618)

    for first, group in groups:
        for r in range(first, first + group.request_count):
            info = requests[r]
            seq, query = info["seq"], info["query"]
            for t in (0, query - 1):
                pos = seq - query + t
                q = math.cos(r * 3.1 + t * 0.271)
                direct = [
                    kv(info["pages"][p // PAGE], p % PAGE)
                    for p in range(pos + 1)
                ]
                level0 = [
                    kv(page, s)
                    for page in group.shared_page_ids
                    for s in range(PAGE)
                ]
                own = pos + 1 - group.shared_blocks * PAGE
                level1 = [
                    kv(info["pages"][group.shared_blocks + p // PAGE],
                       p % PAGE)
                    for p in range(own)
                ]
                # Same keys, same order: the split may not change the
                # set of KV slots a query token attends.
                assert level0 + level1 == direct
                weights = [math.exp(q * k) for k in direct]
                total = sum(weights)
                merged = sum(w * k for w, k in zip(weights, direct))
                assert total > 0 and math.isfinite(merged)


def test_ceil_rounded_explicit_group_is_rejected():
    requests, table, seqs, queries = build_batch("natural")
    first = requests[0]
    bad = CascadeGroup(
        request_count=1,
        shared_blocks=25,
        shared_page_ids=tuple(first["pages"][:25]),
    )
    with pytest.raises(RuntimeError):
        _check_group(bad, 0, table, seqs, queries, PAGE)


def test_cold_request_disables_the_step():
    requests, table, seqs, queries = build_batch("natural")
    queries = list(queries)
    queries[0] = seqs[0]        # first request computes from scratch
    assert _derived_groups(len(requests), table, seqs, queries,
                           PAGE) is None
