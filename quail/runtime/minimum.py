"""The fewest input tokens a query's requests need with unlimited KV.

Every request the engine made is a token sequence: a document prefix
followed by a filter question, or an anchor prefix and its frame
followed by a partner label, the partner document, and the answer cue.
With unlimited KV every distinct prefix across those sequences is
computed once, so the query needs one forward pass position per node
of their prefix trie. What an engine computed beyond that is its
regret, whatever the cause: an evicted anchor computed again, a set
scanned twice under two aliases, or a prompt prefix the documents
share computed once per document.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quail.planner import collect_operators


@dataclass
class _Document:
    alias: str
    row: int
    suffixes: set = field(default_factory=set)
    groups: dict = field(default_factory=dict)


def _tokens(document) -> np.ndarray:
    """Return a token sequence as an unsigned 32 bit array."""
    if isinstance(document, np.ndarray):
        return document.astype(np.uint32, copy=False)
    numpy = getattr(document, "numpy", None)
    if numpy is not None:
        try:
            return np.asarray(numpy(), dtype=np.uint32)
        except (TypeError, ValueError):
            pass
    return np.fromiter(document, dtype=np.uint32, count=len(document))


def prefix_trie_size(sequences) -> int:
    """Return the distinct prefix positions across token sequences.

    Sorted, each sequence sits next to the one it shares the longest
    prefix with, so the trie holds the total length minus the shared
    prefix of every neighbouring pair. Sequences compare as big endian
    bytes, whose order is the token order.
    """
    keys = sorted(_tokens(sequence).astype(">u4").tobytes()
                  for sequence in sequences)
    total = sum(len(key) for key in keys) // 4
    shared = 0
    for earlier, later in zip(keys, keys[1:]):
        length = min(len(earlier), len(later))
        differs = (np.frombuffer(earlier, np.uint8, length)
                   != np.frombuffer(later, np.uint8, length))
        first = int(differs.argmax()) if differs.any() else length
        shared += first // 4
    return total - shared


def _documents(store):
    return getattr(store, "tokens", store)


def _preamble(filters, joins) -> np.ndarray:
    prompts = [predicate.prompt for predicates in filters.values()
               for predicate in predicates]
    prompts += [join.predicate for join in joins]
    for prompt in prompts:
        if prompt.preamble_token_ids:
            return _tokens(tuple(prompt.preamble_token_ids))
    return np.zeros(0, dtype=np.uint32)


def _anchor(table, anchors, written_pos) -> str:
    if anchors is not None and written_pos in anchors:
        return anchors[written_pos]
    metadata = table.schema.metadata or {}
    return metadata[b"quail.anchor"].decode("utf-8")


def minimum_input_tokens(logical, token_inputs, filter_answers,
                         join_answers, anchors=None) -> int:
    """Return the fewest input tokens the query's requests need.

    Args:
        logical: The query's logical plan, with bound prompts.
        token_inputs: Alias to token store, one per scan.
        filter_answers: (alias, written position) -> table with the
            alias's row positions and answers, one per filter stage.
        join_answers: Written position -> table with one row position
            column per alias and answers. The anchor alias comes from
            the table metadata unless anchors names it.
        anchors: Written position -> anchor alias, for tables saved
            without metadata.
    """
    scans, filters, joins = collect_operators(logical)
    sets = {scan.alias: (scan.provider, scan.column) for scan in scans}
    stores = {alias: _documents(store)
              for alias, store in token_inputs.items()}
    pre = _preamble(filters, joins)
    records: dict = {}

    def record(alias, row) -> _Document:
        key = (sets[alias], int(row))
        if key not in records:
            records[key] = _Document(alias=alias, row=int(row))
        return records[key]

    for (alias, written_pos), table in filter_answers.items():
        question = tuple(filters[alias][written_pos].prompt.tail_token_ids)
        for row in table.column(alias).to_pylist():
            record(alias, row).suffixes.add(question)
    for written_pos, table in join_answers.items():
        prompt = joins[written_pos].predicate
        anchor = _anchor(table, anchors, written_pos)
        partners = [name for name in table.column_names
                    if name not in (anchor, "answer")]
        if len(partners) != 1:
            raise NotImplementedError("the minimum needs binary joins")
        (partner,) = partners
        parts = {alias: (tuple(label), tuple(frame))
                 for alias, label, frame in prompt.label_token_ids}
        group = (parts[anchor][1], parts[partner][0],
                 tuple(prompt.tail_token_ids))
        partner_set = sets[partner]
        for anchor_row, partner_row in zip(
                table.column(anchor).to_pylist(),
                table.column(partner).to_pylist()):
            document = record(anchor, anchor_row)
            document.suffixes.add(group[0])
            document.groups.setdefault(group, set()).add(
                (partner_set, int(partner_row)))

    alias_of = {sets[alias]: alias for alias in sets}
    total = prefix_trie_size(
        np.concatenate((pre, _tokens(stores[document.alias][document.row])))
        for document in records.values())
    suffix_sizes: dict = {}
    partner_sizes: dict = {}
    for document in records.values():
        suffixes = frozenset(document.suffixes)
        if suffixes not in suffix_sizes:
            suffix_sizes[suffixes] = prefix_trie_size(suffixes)
        total += suffix_sizes[suffixes]
        for (_, label, tail), partners in document.groups.items():
            members = frozenset(partners)
            if members not in partner_sizes:
                partner_sizes[members] = prefix_trie_size(
                    stores[alias_of[member_set]][row]
                    for member_set, row in members)
            total += (len(label) + partner_sizes[members]
                      + len(tail) * len(members))
    return total
