"""The fewest input tokens a run's requests need with unlimited KV.

Every request an engine made is a token sequence: a document prefix
followed by a filter question, or an anchor prefix and its frame
followed by a partner label, the partner document, and the answer cue.
With unlimited KV every distinct prefix across those sequences is
computed once, so the run needs one forward pass position per node of
their prefix trie. What an engine computed beyond that is its regret,
whatever the cause: an evicted anchor computed again, a set scanned
twice under two aliases, or a prompt prefix the documents share
computed once per document.

The engine reports the prompt pieces it used as token ids (see
`validate_prompt_pieces`); the documents are tokenized here with the
tokenizer the pieces name, after the run, so nothing is tracked while
the query runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from quail_b.data import _ids


@dataclass
class _Document:
    alias: str
    row_id: str
    suffixes: set = field(default_factory=set)
    groups: dict = field(default_factory=dict)


def _tokens(sequence) -> np.ndarray:
    return np.asarray(sequence, dtype=np.uint32)


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


def _token_list(value, name):
    if not isinstance(value, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value):
        raise ValueError(f"prompt pieces: {name} must be a list of token ids")
    return [int(item) for item in value]


def validate_prompt_pieces(spec, pieces) -> dict:
    """Check and normalize the prompt pieces an engine reports.

    Args:
        spec: The query.
        pieces: A dict with `tokenizer` (a HuggingFace tokenizer name,
            the one that tokenized the documents), `preamble` (token
            ids before every document), `filters` (a list of
            `{"id", "tail"}`: the operator ID and ids after the
            document of that filter) and `joins` (a list of
            `{"id", "anchor", "frame", "label", "tail"}`: the
            anchor alias, the ids after the anchor document, the ids
            before the partner document, and the ids after it).

    Returns:
        The pieces as plain lists, with every stage of the query named.
    """
    if not isinstance(pieces, dict) or not isinstance(
            pieces.get("tokenizer"), str) or not pieces["tokenizer"]:
        raise ValueError("prompt pieces need a tokenizer name")
    checked = {"tokenizer": pieces["tokenizer"],
               "preamble": _token_list(pieces.get("preamble", ()), "preamble"),
               "filters": [], "joins": []}
    stages = {filter_spec.id for filter_spec in spec._info.filters}
    for item in pieces.get("filters", ()):
        operator_id = item.get("id")
        if operator_id not in stages:
            raise ValueError(
                f"prompt pieces: unknown filter operator {operator_id!r}"
            )
        stages.remove(operator_id)
        checked["filters"].append({
            "id": operator_id,
            "tail": _token_list(item.get("tail", ()), "filter tail")})
    if stages:
        raise ValueError(
            f"prompt pieces: missing filter operators {sorted(stages)}"
        )
    joins = {join.id: join for join in spec._info.joins}
    operator_ids = set(joins)
    for item in pieces.get("joins", ()):
        operator_id = item.get("id")
        if operator_id not in operator_ids:
            raise ValueError(
                f"prompt pieces: unknown join operator {operator_id!r}"
            )
        operator_ids.remove(operator_id)
        if item.get("anchor") not in joins[operator_id].relations:
            raise ValueError(
                f"prompt pieces: join {operator_id!r} anchors on an alias it "
                f"does not join: {item.get('anchor')!r}")
        checked["joins"].append({
            "id": operator_id, "anchor": item["anchor"],
            **{name: _token_list(item.get(name, ()), f"join {name}")
               for name in ("frame", "label", "tail")}})
    if operator_ids:
        raise ValueError(
            f"prompt pieces: missing join operators {sorted(operator_ids)}"
        )
    return checked


@lru_cache(maxsize=4)
def load_tokenizer(name: str):
    """Return a callable tokenizing a list of texts, from HuggingFace."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)

    def encode(texts):
        return tokenizer(list(texts), add_special_tokens=False)["input_ids"]

    return encode


class DocumentTokens:
    """Token ids of documents, tokenized on first use and kept.

    Args:
        corpus_rows: Table name to its rows.
        tokenizer: Callable(list of texts) -> list of token id lists.
    """

    def __init__(self, corpus_rows, tokenizer):
        self.corpus_rows = corpus_rows
        self.tokenizer = tokenizer
        self._texts = {}
        self._tokens = {}

    def _text(self, table, column, row_id):
        key = (table, column)
        if key not in self._texts:
            rows = self.corpus_rows[table]
            values = (rows.column(column).to_pylist()
                      if hasattr(rows, "column") else [row[column] for row in rows])
            self._texts[key] = dict(zip(map(str, _ids(rows)), values))
        return self._texts[key][row_id]

    def fetch(self, documents) -> None:
        """Tokenize the (table, column, id) documents not seen yet."""
        missing = [document for document in dict.fromkeys(documents)
                   if document not in self._tokens]
        if not missing:
            return
        encoded = self.tokenizer([self._text(*document) for document in missing])
        for document, ids in zip(missing, encoded):
            self._tokens[document] = _tokens(ids)

    def __getitem__(self, document) -> np.ndarray:
        return self._tokens[document]


def _encoded(column) -> tuple[np.ndarray, pa.Array]:
    """Return a string id column as dictionary indices and the dictionary."""
    column = pc.cast(column, pa.string())
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    encoded = pc.dictionary_encode(column)
    return np.asarray(encoded.indices), encoded.dictionary


def minimum_input_tokens(spec, pieces, filter_answers, join_answers,
                         documents: DocumentTokens) -> int:
    """Return the fewest input tokens the run's requests need.

    A join can hold tens of millions of pairs, so the pairs are
    grouped by anchor in numpy; Python visits documents and anchors.

    Args:
        spec: The query.
        pieces: Validated prompt pieces (`validate_prompt_pieces`).
        filter_answers: Filter operator ID to a table with the relation's
            alias and answers, one row per document asked.
        join_answers: Join operator ID to a table with one ID column per
            relation and answers, one row per evaluated pair.
        documents: The document tokens.
    """
    sets = {
        relation.alias: (relation.table, relation.text_column)
        for relation in spec._info.relations
    }
    pre = _tokens(pieces["preamble"])
    records: dict = {}

    def record(alias, row_id) -> _Document:
        key = (sets[alias], str(row_id))
        if key not in records:
            records[key] = _Document(alias=alias, row_id=str(row_id))
        return records[key]

    tails = {item["id"]: tuple(item["tail"]) for item in pieces["filters"]}
    filters = {
        filter_spec.id: filter_spec for filter_spec in spec._info.filters
    }
    for operator_id, table in filter_answers.items():
        alias = filters[operator_id].relation
        question = tails[operator_id]
        ids = pc.unique(pc.cast(table.column(alias), pa.string()))
        for row_id in ids.to_pylist():
            record(alias, row_id).suffixes.add(question)
    joins = {item["id"]: item for item in pieces["joins"]}
    join_specs = {join.id: join for join in spec._info.joins}
    # a member key names one distinct partner set of one join: the
    # sorted partner indices of that join's answer table
    members_of: dict = {}
    for operator_id, table in join_answers.items():
        piece = joins[operator_id]
        anchor = piece["anchor"]
        (partner,) = [alias for alias in join_specs[operator_id].relations
                      if alias != anchor]
        group = (tuple(piece["frame"]), tuple(piece["label"]), tuple(piece["tail"]))
        partner_set = sets[partner]
        anchors, anchor_ids = _encoded(table.column(anchor))
        partners, partner_ids = _encoded(table.column(partner))
        if not len(anchors):
            continue
        order = np.lexsort((partners, anchors))
        anchors, partners = anchors[order], partners[order]
        starts = np.flatnonzero(np.r_[True, anchors[1:] != anchors[:-1]])
        ends = np.r_[starts[1:], len(anchors)]
        for start, end in zip(starts.tolist(), ends.tolist()):
            members = np.unique(partners[start:end])
            key = (operator_id, members.tobytes())
            if key not in members_of:
                members_of[key] = frozenset(
                    (partner_set, row_id) for row_id in pc.take(
                        partner_ids, pa.array(members)).to_pylist())
            document = record(anchor, anchor_ids[int(anchors[start])].as_py())
            document.suffixes.add(group[0])
            document.groups.setdefault(group, set()).add(key)

    documents.fetch([(*table_set, row_id) for table_set, row_id in records]
                    + [(*member_set, row_id)
                       for members in members_of.values()
                       for member_set, row_id in members])
    total = prefix_trie_size(
        np.concatenate((pre, documents[(*table_set, row_id)]))
        for table_set, row_id in records)
    suffix_sizes: dict = {}
    partner_sizes: dict = {}
    for document in records.values():
        suffixes = frozenset(document.suffixes)
        if suffixes not in suffix_sizes:
            suffix_sizes[suffixes] = prefix_trie_size(suffixes)
        total += suffix_sizes[suffixes]
        for (_, label, tail), keys in document.groups.items():
            keys = frozenset(keys)
            if keys not in partner_sizes:
                members = frozenset().union(*(members_of[key] for key in keys))
                partner_sizes[keys] = (len(members), prefix_trie_size(
                    documents[(*member_set, row_id)]
                    for member_set, row_id in members))
            count, size = partner_sizes[keys]
            total += len(label) + size + len(tail) * count
    return total


def _fresh_tokens(measurements) -> int | None:
    if "fresh_tokens" not in measurements:
        return None
    fresh = measurements["fresh_tokens"]
    if isinstance(fresh, bool) or not isinstance(fresh, int) or fresh < 0:
        raise ValueError("fresh_tokens must be a nonnegative integer")
    return fresh


def token_metrics(spec, output, corpus_rows, stores=None) -> dict:
    """Return the run's fresh, minimum, and regret token counts.

    `fresh_tokens` is engine-reported and is required when the output
    includes `prompt_pieces`. `minimum_tokens` and `regret_tokens` are
    None when there are no prompt pieces; they are never omitted.

    Args:
        spec: The query.
        output: The run output.
        corpus_rows: Table name to its rows.
        stores: Tokenizer name -> DocumentTokens, kept across the
            queries of one run so each document is tokenized once.
    """
    fresh = _fresh_tokens(output.measurements)
    if output.prompt_pieces is None:
        return {"fresh_tokens": fresh, "minimum_tokens": None, "regret_tokens": None}
    if fresh is None:
        raise ValueError("prompt pieces need a fresh_tokens measurement")
    if output.filter_answers is None or output.join_answers is None:
        raise ValueError("prompt pieces need filter and join answers")
    pieces = validate_prompt_pieces(spec, output.prompt_pieces)
    stores = {} if stores is None else stores
    name = pieces["tokenizer"]
    if name not in stores:
        stores[name] = DocumentTokens(corpus_rows, load_tokenizer(name))
    minimum = minimum_input_tokens(
        spec, pieces, output.filter_answers, output.join_answers, stores[name])
    if fresh < minimum:
        raise ValueError(
            f"fresh_tokens ({fresh}) is below the minimum the requests need "
            f"({minimum})")
    return {
        "fresh_tokens": fresh, "minimum_tokens": minimum,
        "regret_tokens": fresh - minimum,
    }
