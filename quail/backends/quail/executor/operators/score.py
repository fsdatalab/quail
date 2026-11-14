"""AI.SCORE over its candidate rows, apart from a pipeline.

A score reads documents of one table, or pairs of two. Its rows come
from its inputs: one table's documents, a cross product of two, or the
pairs a hash join chose. The prompt layout follows the rows: pairs
share each anchor document's KV across its partners, and documents of
one table share the prompt head, or each document's whole pages
through the prefix tree.
"""

from dataclasses import dataclass, replace

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.backends.quail.executor.model_operator import (
    SingleNodeOperator,
    shared_prefix_tree,
)
from quail.backends.quail.executor.readout import AsyncScores
from quail.backends.quail.executor.stages import Stage
from quail.backends.quail.reranker import (
    classify_outputs,
    probability_column,
    score_rows,
    scored_batch,
)
from quail.common.expressions import PROBABILITIES_SUFFIX
from quail.common.tokens import DocumentPrefixes, concat_tokens
from quail.execution.finished import (
    finished_rows,
    finished_table,
    hinting,
    keeps,
)
from quail.execution.ids import id_positions
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import rows_of
from quail.physical import ClassifySpec, ValueType
from quail.progress import answer_sink
from quail.specs.base import CANVAS_ENTROPY_NATS, CANVAS_SEED


@dataclass(frozen=True)
class ScoreRows:
    """Document indices with an optional unexpanded Cartesian product."""

    columns: tuple[np.ndarray, ...]
    product: bool = False

    def __len__(self):
        if self.product:
            return len(self.columns[0]) * len(self.columns[1])
        return len(self.columns[0])

    def array(self) -> np.ndarray:
        """Every row as one array, one column per alias, in input order.

        A product's rows run through the second alias for each row of the
        first.
        """
        if self.product:
            left, right = self.columns
            return np.column_stack((np.repeat(left, len(right)),
                                    np.tile(right, len(left))))
        return np.column_stack(self.columns)


def _ids(value, alias: str) -> np.ndarray:
    if isinstance(value, pa.Table):
        value = value.column(alias)
    return id_positions(value)


def candidate_rows(node, inputs):
    """Return the rows a score or classification reads and its priors by alias.

    Args:
        node: The AiScore or AiClassify.
        inputs: Its input values by port name.

    Returns:
        The ScoreRows, and alias -> the Rows whose values its rows keep.
    """
    spec = node.spec
    by_alias = {}
    priors = {}
    pairs = None
    for port in node.inputs:
        value = inputs[port.name]
        if port.value_type is ValueType.PAIRS:
            pairs = value
            continue
        if port.value_type is ValueType.ROWS:
            if port.source.port.startswith("rows:"):
                value = rows_of(value, port.source.port.split(":", 1)[1])
            for alias in spec.aliases:
                if alias in value.ids:
                    by_alias[alias] = id_positions(value.ids[alias])
                    priors[alias] = value
    if len(spec.aliases) == 1:
        return ScoreRows((by_alias[spec.aliases[0]],)), priors
    left, right = spec.aliases
    if pairs is None:
        return ScoreRows((by_alias[left], by_alias[right]), product=True), priors
    selected = pairs.filter(pc.and_(
        pc.is_in(pairs.column(left), value_set=pa.array(by_alias[left])),
        pc.is_in(pairs.column(right), value_set=pa.array(by_alias[right])),
    ))
    return ScoreRows((_ids(selected, left), _ids(selected, right))), priors


def hinted_rows(node, rows: ScoreRows):
    """Callable(row position) -> whether the region's hint drops the row, or None.

    A row is dropped when the hint drops the document of any of its
    tables.
    """
    if not hinting():
        return None
    aliases = node.spec.aliases
    width = len(rows.columns[-1])

    def skip(position):
        places = ((position // width, position % width) if rows.product
                  else (position,) * len(aliases))
        return not all(keeps(alias, int(documents[place])) for alias, documents, place
                       in zip(aliases, rows.columns, places))

    return skip


def value_type(spec) -> pa.DataType:
    """Return the Arrow type for a label or a numeric score."""
    return pa.string() if isinstance(spec, ClassifySpec) else pa.float64()


def scored_outputs(node, scored, priors) -> dict:
    """The node's outputs for scored rows: their priors' values, labels kept."""
    for alias, prior in priors.items():
        scored = scored.joined(prior, alias)
    if isinstance(node.spec, ClassifySpec):
        # a document whose decoded answer names no label leaves the query
        return classify_outputs(node, scored.filter(
            pc.is_valid(scored.column(node.spec.name))))
    return {node.rows_port: scored}


def finished_scores(node, rows, values, probabilities=None) -> list:
    """The finished rows of a score or classification: ids and answers.

    Args:
        node: The AiScore or AiClassify.
        rows: The rows answered, one column per alias.
        values: Their scores or labels.
        probabilities: A classification's label probabilities per row,
            or None.
    """
    if not len(rows) or finished_rows() is None:
        return []
    scored = scored_rows(node, rows, values, probabilities)
    if isinstance(node.spec, ClassifySpec):
        # a document whose decoded answer names no label leaves the query
        scored = scored.filter(pc.is_valid(scored.column(node.spec.name)))
    return [finished_table(node.node_id, node.rows_port, scored.table())]


def scored_rows(node, rows, values, probabilities=None):
    """The Rows of a score or classification, in row order, without priors."""
    spec = node.spec
    scored = score_rows(rows, spec.aliases, spec.name, values, value_type(spec))
    if isinstance(spec, ClassifySpec) and spec.probabilities:
        scored = scored.with_column(
            spec.name + PROBABILITIES_SUFFIX, probability_column(
                spec.labels,
                np.full((len(rows), len(spec.labels)), np.nan)
                if probabilities is None else probabilities))
    return scored


def scored_result(node, rows, values, priors, *, fresh_tokens, cached_tokens=0,
                  suffix_tokens=0, borrowed_tokens=0, pack_s=0.0, gpu_s=0.0,
                  chunks=0, probabilities=None) -> NodeResult:
    """The NodeResult of a score or classification of rows, in row order.

    Args:
        node: The AiScore or AiClassify.
        rows: The rows asked, one column per alias.
        values: Their scores or labels.
        priors: alias -> the Rows whose values the rows keep.
        fresh_tokens: Token positions the model computed.
        cached_tokens: Token positions read from KV.
        suffix_tokens: A classification's tokens after the documents.
        borrowed_tokens: Prefix tokens read from another document's KV.
        pack_s: Host seconds spent building chunks.
        gpu_s: Forward-pass GPU seconds.
        chunks: Timed forward passes.
        probabilities: A classification's label probabilities per row,
            or None.
    """
    spec = node.spec
    count = len(rows)
    outputs = scored_outputs(node, scored_rows(node, rows, values, probabilities),
                             priors)
    output = outputs[node.rows_port]
    return NodeResult(outputs, NodeMetrics(
        input_rows=count,
        output_rows=(output.num_rows if isinstance(spec, ClassifySpec) else count),
        evaluated_documents=count if len(spec.aliases) == 1 else 0,
        evaluated_document_pairs=count if len(spec.aliases) == 2 else 0,
        fresh_tokens=fresh_tokens, cached_tokens=cached_tokens,
        gpu_s=gpu_s, chunks=chunks,
        extension={
            "output": spec.name, "aliases": list(spec.aliases),
            "input_rows": len(output),
            "borrowed_prefix_tokens": borrowed_tokens,
            **({"suffix_tokens": suffix_tokens, "pack_s": pack_s}
               if isinstance(spec, ClassifySpec) else {}),
        }))


def score_readout(state) -> AsyncScores:
    """Return the score readout for the current model."""
    if state.score_readout is not None:
        return state.score_readout
    answer_rows = state.answer_rows
    async_scores = state.async_scores
    if async_scores is None or async_scores.rows is not answer_rows:
        async_scores = AsyncScores(
            state.torch, answer_rows,
            state.loaded_model.model_spec.answer_temperature)
        state.async_scores = async_scores
    return async_scores


class ScoreOperator(SingleNodeOperator):
    """AI.SCORE over its candidate rows, with the prompt layout they need.

    Args:
        state: The query's GPU state.
        node: The AiScore.
        rows: Its ScoreRows.
        priors: alias -> the Rows whose values its rows keep.
        documents: Tokenized documents by alias.
    """

    @classmethod
    def build(cls, region, nodes, chain):
        node = nodes[0]
        if node.spec is None:
            raise ValueError("AI.SCORE needs a score specification")
        rows, priors = candidate_rows(node, region.node_inputs(node))
        return cls(region.query, node, rows, priors, region.documents)

    @classmethod
    def alone(cls, region, operators, table, settings):
        """A score reads no limit; it skips the rows the region's hint drops."""
        (operator,) = operators
        return replace(settings, label=f"score {operator.node.spec.name}",
                       limit=None, attention=None, skip=operator.skip)

    def __init__(self, state, node, rows: ScoreRows, priors, documents):
        spec = node.spec
        if len(spec.arguments) != len(spec.aliases):
            raise ValueError("AI.SCORE needs one prompt argument per document relation")
        self.state = state
        self.node = node
        self.priors = priors
        self.rows = np.asarray(rows.array(), dtype=np.int32).reshape(
            -1, len(spec.aliases))
        self.scores = np.full(len(self.rows), np.nan, dtype=np.float32)
        self.skipped = np.zeros(len(self.rows), dtype=bool)
        self.sent = np.zeros(len(self.rows), dtype=bool)
        self.finished = []
        self.order = self.offsets = None   # a grouped layout's rows by anchor
        hinted = hinted_rows(node, rows)
        self.skip = None
        if hinted is not None:
            def skip(position):
                self.skipped[position] = hinted(position)
                return bool(self.skipped[position])

            self.skip = skip
        parts = spec.prompt_token_parts
        if not len(self.rows):
            # no rows: nothing to run
            self.stages, self.total = (), 0
            self.table = _table([], [])
            return
        if len(parts) != len(spec.aliases) + 1:
            raise ValueError("AI.SCORE needs tokenized prompt parts")
        if (len(spec.aliases) == 1 and spec.draws > 1
                and state.loaded_model.pipeline.canvas_ids):
            self._drawn(documents)
        elif len(spec.aliases) == 1 and (spec.share_prefixes
                                         or (self.skip is not None and parts[1])):
            # admission skips whole documents, so a score that skips rows
            # gives each document its own prefix, sharing the prompt
            # head's pages; with no prompt after the document, it asks
            # every row
            self._shared(documents)
        else:
            self._grouped(documents)

    def layout(self, region):
        return self.table

    def _keys(self, prefixes) -> list:
        return [("score", self.node.spec.name, index) for index in range(len(prefixes))]

    def _grouped(self, documents) -> None:
        """Ask each anchor's rows after one shared prefix.

        A score of one table asks every document after the prompt head;
        a score of pairs asks each left document's partners after the
        left document.
        """
        spec = self.node.spec
        parts = spec.prompt_token_parts
        rows = self.rows
        if len(spec.aliases) == 1:
            prefixes = [parts[0]]
            suffixes = [concat_tokens(documents[spec.aliases[0]][document_id],
                                      parts[1])
                        for document_id in rows[:, 0]]
            partners = None
            self.order = np.arange(len(rows))
            self.offsets = np.array([0, len(rows)])
            self.total = len(rows) * len(parts[0]) + sum(map(len, suffixes))
        else:
            left, right = spec.aliases
            anchors, anchor_index = np.unique(rows[:, 0], return_inverse=True)
            candidates, candidate_index = np.unique(rows[:, 1], return_inverse=True)
            prefixes = [concat_tokens(parts[0], documents[left][document_id],
                                      parts[1])
                        for document_id in anchors]
            suffixes = [concat_tokens(documents[right][document_id], parts[2])
                        for document_id in candidates]
            self.order = order = np.argsort(anchor_index, kind="stable")
            self.offsets = offsets = np.concatenate(
                ([0], np.cumsum(np.bincount(anchor_index))))
            partners = [candidate_index[order[start:end]]
                        for start, end in zip(offsets[:-1], offsets[1:])]
            prefix_lengths = np.asarray([len(prefix) for prefix in prefixes])
            suffix_lengths = np.asarray([len(suffix) for suffix in suffixes])
            self.total = int(prefix_lengths[anchor_index].sum()
                             + suffix_lengths[candidate_index].sum())
        self.stages = [Stage(suffixes=suffixes, readout=score_readout(self.state),
                             requests=(None if partners is None else
                                       lambda key: partners[key[2]]))]
        self.table = _table(prefixes, self._keys(prefixes))

    def _positions(self, anchor) -> np.ndarray:
        """The row positions an anchor asks."""
        if self.order is None:
            return np.asarray([anchor])
        return self.order[self.offsets[anchor]:self.offsets[anchor + 1]]

    def _shared(self, documents) -> None:
        """Score documents that borrow the KV pages of shared prefixes.

        Each document's prompt head and tokens are its prefix; a
        document whose prefix shares whole pages with another's reads
        those pages instead of computing them. The prompt tail follows
        each document, its last token read for the score.

        Raises:
            ValueError: The prompt has no tokens after the document.
        """
        spec = self.node.spec
        (alias,) = spec.aliases
        head, tail = spec.prompt_token_parts
        if not tail:
            raise ValueError("AI.SCORE needs a question after the document")
        document_ids = self.rows[:, 0]
        scores = self.scores

        def keep(anchor, row):
            scores[anchor] = float(row[0])
            return True

        prefixes = DocumentPrefixes(head, documents[alias], document_ids)
        self.stages = [Stage(suffixes=[[tail[-1]]], readout=score_readout(self.state),
                             frame=list(tail[:-1]), decide=keep, single=True,
                             label=f"score {spec.name}")]
        self.total = sum(len(head) + len(documents[alias][document_id]) + len(tail)
                         for document_id in document_ids)
        self.table = _table(prefixes, self._keys(prefixes), shared_prefix_tree(
            spec.name, prefixes, self.state.loaded_model.arena))

    def _drawn(self, documents) -> None:
        """Score documents with reproducible diffusion draws.

        A low-entropy first answer uses one draw. Other documents run the
        remaining draws and use the mean TRUE probability. Draws share
        each document's KV.
        """
        state = self.state
        spec = self.node.spec
        pipeline = state.loaded_model.pipeline
        vocab = state.loaded_model.model_spec.vocab
        (alias,) = spec.aliases
        head, tail = spec.prompt_token_parts
        document_ids = self.rows[:, 0]
        width = len(pipeline.canvas_ids)
        more = spec.draws - 1
        scores = self.scores
        first = np.zeros(len(document_ids), dtype=np.float64)
        extended = [0]

        def canvas(anchor, draw):
            rng = np.random.default_rng((CANVAS_SEED, int(document_ids[anchor]), draw))
            return rng.integers(0, vocab, width)

        def settle(anchor, row):
            p = float(row[0])
            first[anchor] = p
            q = min(max(p, 1e-12), 1 - 1e-12)
            if -(q * np.log(q) + (1 - q) * np.log(1 - q)) <= CANVAS_ENTROPY_NATS:
                scores[anchor] = p
            return True

        def average(anchor, row):
            extended[0] += 1
            scores[anchor] = (first[anchor] + float(np.sum(row))) / spec.draws
            return True

        cue = [tail[-1]]
        readout = score_readout(state)
        self.stages = [
            Stage(suffixes=[cue], readout=readout, frame=list(tail[:-1]),
                  decide=settle, single=True, label=f"score {spec.name}",
                  canvas=lambda anchor: canvas(anchor, 0), canvas_rows=width),
            Stage(suffixes=[cue] * more, readout=readout,
                  frame=list(tail[:-1]), decide=average,
                  requests=lambda key: (None if np.isnan(scores[key[2]])
                                        else Stage.SKIP),
                  label=f"score {spec.name} draws",
                  canvas=lambda anchor: np.stack(
                      [canvas(anchor, draw) for draw in range(1, spec.draws)]),
                  canvas_rows=width),
        ]
        prefixes = DocumentPrefixes(head, documents[alias], document_ids)
        tree = (shared_prefix_tree(spec.name, prefixes, state.loaded_model.arena)
                if spec.share_prefixes else None)
        read = len(cue) + width
        base = sum(
            len(head) + len(documents[alias][document_id])
            for document_id in document_ids
        ) + len(document_ids) * (len(tail) - 1 + read)
        self.extended = extended
        self.total = lambda: base + extended[0] * more * read
        self.table = _table(prefixes, self._keys(prefixes), tree)

    def _total(self) -> int:
        return self.total() if callable(self.total) else self.total

    def _fill(self, anchor, row) -> None:
        """Store a grouped layout's scores from an anchor's answer row."""
        if self.order is not None and len(row):
            self.scores[self._positions(anchor)] = np.asarray(row, dtype=np.float32)

    def settle(self, document, survived, row):
        self._fill(document, row)
        self.finished.extend(self._positions(document).tolist())

    def answered(self, anchor, stage, start, end, answers):
        """Score an anchor's rows as their requests' answers arrive.

        A grouped layout asks each anchor's rows as its requests: one
        anchor of a shared prompt head asks every document, and a left
        document asks its partners; each row is final once answered.
        """
        if self.order is None or stage:
            return False
        positions = self._positions(anchor)[start:end]
        self.scores[positions] = np.asarray(answers, dtype=np.float32)
        self.finished.extend(positions.tolist())
        return True

    def take_finished(self):
        """Send newly scored rows to the answer sink; return their rows."""
        positions = sorted({position for position in self.finished
                            if not self.sent[position] and not self.skipped[position]
                            and not np.isnan(self.scores[position])})
        self.finished = []
        if not positions:
            return []
        self.sent[positions] = True
        rows, values = self.rows[positions], self.scores[positions].tolist()
        sink = answer_sink()
        if sink is not None:
            sink(scored_batch(self.node, rows, values))
        return finished_scores(self.node, rows, values)

    def read_answers(self, every) -> int:
        """Send the scores not yet sent; return the full prompt token count."""
        for answers in every[:1]:
            for anchor, row in answers.items():
                if np.isnan(self.scores[self._positions(anchor)]).any():
                    self._fill(anchor, row)
        rest = np.flatnonzero(~self.sent & ~self.skipped & ~np.isnan(self.scores))
        if len(rest):
            self.sent[rest] = True
            sink = answer_sink()
            if sink is not None:
                sink(scored_batch(self.node, self.rows[rest],
                                  self.scores[rest].tolist()))
        return self._total()

    def result(self, fresh_tokens, gpu_s, chunks, stats) -> NodeResult:
        asked = ~self.skipped
        return scored_result(
            self.node, self.rows[asked], self.scores[asked], self.priors,
            fresh_tokens=fresh_tokens, cached_tokens=self._total() - fresh_tokens,
            borrowed_tokens=stats.get("borrowed_tokens", 0),
            pack_s=stats.get("pack_s", 0.0), gpu_s=gpu_s, chunks=chunks)


def _table(prefixes, keys, tree=None):
    from quail.backends.quail.executor.region import RegionDocuments

    return RegionDocuments(prefixes, keys, tree)
