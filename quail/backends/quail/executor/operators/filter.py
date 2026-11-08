"""Run AI filters and conditions on computed columns."""

from dataclasses import replace

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.backends.quail.executor.model_operator import SingleNodeOperator
from quail.backends.quail.executor.stages import Stage, filter_stages
from quail.common.expressions import accepts
from quail.common.pairs import COPIES, columns_key
from quail.common.selection import equality_keys
from quail.execution.finished import finished_rows, finished_table
from quail.execution.ids import document_ids as arrow_ids
from quail.execution.ids import id_positions
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import filter_answer_table
from quail.execution.rows import Rows
from quail.execution.runtimes.filter import filter_scores, tested_table
from quail.progress import answer_sink

# the key of a NaN stop key value
NAN_KEY = ("NaN",)


class FilterOperator(SingleNodeOperator):
    """One AI filter's stage, its answers, and its result table."""

    @classmethod
    def build(cls, region, nodes, chain):
        """Build one filter over the chain's documents."""
        query = region.query
        return cls(nodes[0], region.document_ids(), query.async_answers,
                   len(query.loaded_model.pipeline.canvas_ids))

    @classmethod
    def streams(cls, node) -> bool:
        return True

    @classmethod
    def alone(cls, region, operators, table, settings):
        """Filters alone keep each stop key group's documents apart.

        One filter writes no KV unless the plan asks; several share
        the document's KV between their questions.
        """
        from quail.backends.quail.executor.region import pipeline_rows

        node = operators[0].node
        keys = stop_keys(region.context.state, node)
        return replace(
            settings, label=f"filter ({len(operators)} predicates)",
            attention=node.attention or "unified",
            paged=node.arena_writes or len(operators) > 1,
            stop_keys=keys,
            stop_groups=None if keys is None else keys.of(
                region.document_ids(), copies_of(pipeline_rows(node, region.inputs))),
            num_key_survivors=node.num_key_survivors)

    def __init__(self, node, ids, async_answers, canvas_tokens=0):
        self.node = node
        self.ids = ids
        self.canvas_tokens = canvas_tokens
        self.stages = filter_stages(
            [list(node.question_token_ids)],
            async_answers)
        self.answers = {}
        self.finished = []      # (document, passed) settled since the last take

    @staticmethod
    def prepare_group(operators):
        """Retain the common question preamble across consecutive filters."""
        stages = filter_stages(
            [operator.node.question_token_ids for operator in operators],
            operators[0].stages[0].readout)
        for operator, stage in zip(operators, stages):
            operator.stages = [stage]

    def settle(self, document, survived, row):
        self.finished.append((document, survived))

    def take_finished(self):
        """Report the settled documents to the answer sink; return the passed.

        The sink receives ``{"kind": "filter", "node", "alias", "stages",
        "documents"}``, where ``documents`` lists ``[document id, stage,
        passed]`` per document. The documents that passed are the rows
        of the filter's rows port.
        """
        finished, self.finished = self.finished, []
        sink = answer_sink()
        if sink is not None and finished:
            sink({"kind": "filter", "node": self.node.node_id,
                  "alias": self.node.alias, "stages": 1,
                  "documents": [[int(self.ids[document]), 0, passed]
                                for document, passed in finished]})
        passed = [document for document, survived in finished if survived]
        return passed_rows(self.node, self.node.alias,
                           [self.ids[document] for document in passed])

    def read_answers(self, every) -> int | None:
        """Record filter answers and leave token accounting to the region."""
        for stage in every:
            for document, row in stage.items():
                self.answers.setdefault(document, []).append(int(row[0]))
        return len(self.answers) * (len(self.stages[0].suffixes[0])
                                    + self.frame_tokens
                                    + self.canvas_tokens)

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        return filter_result(
            self.node, self.answers, tokens, self.ids, gpu_s=gpu_s,
            chunks=chunks, borrowed_tokens=stats.get("borrowed_tokens", 0),
            pack_s=stats.get("pack_s", 0.0))


class ConditionGate(SingleNodeOperator):
    """A chain's condition that drops documents whose tested values fail.

    A value is a settled column-operator result, such as a label, an
    extracted answer or span, or a map answer, or a source column of
    the table; the region then skips the document's later stages and
    frees its KV.
    """

    stages = ()

    @classmethod
    def streams(cls, node) -> bool:
        return True

    @classmethod
    def build(cls, region, nodes, chain):
        """Bind a condition to the operators computing its columns and the table."""
        node = nodes[0]
        source_id = node.inputs[0].source.node_id
        producers = {}
        for name in node.tested_columns:
            operator = next((operator for operator in reversed(chain)
                             if name in operator.tested_columns), None)
            if operator is not None:
                producers[name] = operator
        source = next(operator for operator in reversed(chain)
                      if any(member.node_id == source_id
                             for member in operator.nodes))
        return cls(node, producers, source,
                   region.context.sources.get(columns_key(node.aliases[0])),
                   region.document_ids)

    def __init__(self, node, producers, source, columns, document_ids=None):
        self.node = node
        # Callable() -> the region's document ids, by position
        self.document_ids = document_ids
        self.finished = []  # documents that passed since the last take
        self.producers = producers  # column -> the operator computing it
        self.source = source  # the operator whose rows the Filter reads
        self.columns = columns  # the table's source columns, by document
        # id(operator) -> document id -> its index among the operator's
        # documents, extended as documents arrive
        self.positions = {id(operator): {} for operator in producers.values()}
        self.accepted = {}
        self.decided = {}  # key -> kept, from decide, until gate asks

    @property
    def answer_dependencies(self):
        """The decoder answer streams the tested columns wait for."""
        return tuple(dict.fromkeys(
            operator.first + operator.column_sources[name][0]
            for name, operator in self.producers.items()
            if operator.answer_limits))

    def read(self, name: str, documents):
        """Return a tested column's values for documents, by document id."""
        operator = self.producers.get(name)
        if operator is not None:
            positions = self.positions[id(operator)]
            for index in range(len(positions), len(operator.ids)):
                positions[operator.ids[index]] = index
            return operator.tested_values(name, [positions[document]
                                                 for document in documents])
        if self.columns is None or name not in self.columns.column_names:
            raise KeyError(f"{self.node.node_id!r} reads {name!r}, which no "
                           f"operator computes and the scan does not hold")
        if len(documents) == 1:
            # one document at a time: a slice copies nothing
            return self.columns.column(name).slice(documents[0], 1)
        return pc.take(self.columns.column(name), pa.array(documents))

    def decide(self, keys):
        """Test the documents of keys together; return the keys kept.

        The region calls this with the documents entering the next
        stage in one chunk, so the condition runs once per chunk; gate
        then answers each document from the saved decision.
        """
        if not keys:
            return keys
        documents = [key[1] for key in keys]
        values = {column: self.read(column, documents)
                  for column in self.node.tested_columns}
        kept = accepts(self.node.predicate, pa.table(values))
        kept = kept.to_numpy(zero_copy_only=False).tolist()
        self.decided.update(zip(keys, kept))
        return [key for key, keep in zip(keys, kept) if keep]

    def gate(self, key):
        kept = self.decided.pop(key, None)
        if kept is not None:
            return None if kept else Stage.DROP
        # a document decide did not see, such as one asked outside a
        # chunk report, is tested alone; a repeated value, such as a
        # label, is evaluated once
        values = {column: self.read(column, [key[1]])
                  for column in self.node.tested_columns}
        token = tuple(_hashable(found[0].as_py()) for found in values.values())
        if token not in self.accepted:
            self.accepted[token] = bool(
                accepts(self.node.predicate, pa.table(values))[0].as_py())
        return None if self.accepted[token] else Stage.DROP

    def settle(self, document, survived, row):
        if survived:
            self.finished.append(document)

    def take_finished(self):
        """Return the documents that passed the condition as its last stage."""
        finished, self.finished = self.finished, []
        if not finished or self.document_ids is None:
            return []
        ids = self.document_ids()
        return passed_rows(self.node, self.node.aliases[0],
                           [ids[document] for document in finished])

    def read_answers(self, every):
        return None

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        source = self.node.inputs[0].source
        rows = self.source.results()[source.node_id].outputs[source.port]
        if all(name in rows.column_names for name in self.node.tested_columns):
            return filter_scores(self.node, rows)
        return filter_scores(self.node, rows, tested_table(
            self.node, rows, lambda name, ids: self.read(name, ids.to_pylist())))


def _hashable(value):
    """Return a struct as sorted pairs and a list as a tuple, so a gate can cache."""
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    return value


def passed_rows(node, alias, document_ids) -> list:
    """The finished rows of documents that passed a filter, by their ids."""
    if not len(document_ids) or finished_rows() is None:
        return []
    ids = arrow_ids(np.sort(np.asarray(document_ids, dtype=np.int64)))
    return [finished_table(node.node_id, f"rows:{alias}", Rows({alias: ids}).table())]


def filter_result(node, answers, tokens, document_ids,
                  gpu_s: float = 0.0, chunks: int = 0,
                  borrowed_tokens: int = 0, pack_s: float = 0.0) -> NodeResult:
    """Build a filter result using global document IDs and execution metrics.

    Args:
        node: Physical filter node.
        answers: Mapping from local document index to Boolean stage answers.
        tokens: Fresh tokens processed by the model.
        document_ids: Global document IDs in local index order.
        gpu_s: Forward-pass GPU seconds.
        chunks: Number of timed forward passes.
        borrowed_tokens: Prefix tokens read from another document's KV.
        pack_s: Host seconds spent building forward chunks.

    Returns:
        A NodeResult containing surviving IDs, filter answers, and metrics.
        The filter answers are a filter_answer_table, one row per answer.
    """
    local = np.fromiter(answers, dtype=np.int64, count=len(answers))
    passed = np.fromiter((len(row) == 1 and all(row) for row in answers.values()),
                         dtype=bool, count=len(answers))
    positions = np.asarray(document_ids, dtype=np.int64)
    survivors = arrow_ids(np.sort(positions[local[passed]]))
    lengths = np.fromiter((len(row) for row in answers.values()),
                          dtype=np.int64, count=len(answers))
    answered = np.repeat(positions[local], lengths)
    values = (np.concatenate([np.asarray(row, dtype=bool)
                              for row in answers.values()])
              if len(answers) else np.zeros(0, dtype=bool))
    return NodeResult(
        outputs={
            f"rows:{node.alias}": Rows({node.alias: survivors}),
            f"filter_answers:{node.alias}": filter_answer_table(
                node.alias, answered,
                np.full(len(answered), node.written_position), values),
        },
        metrics=NodeMetrics(
            input_rows=len(document_ids),
            output_rows=len(survivors),
            evaluated_documents=len(answers),
            fresh_tokens=tokens,
            gpu_s=gpu_s,
            chunks=chunks,
            extension={
                **({"borrowed_prefix_tokens": borrowed_tokens}
                   if borrowed_tokens else {}),
                **({"pack_s": round(pack_s, 3)} if pack_s else {}),
            },
        ),
    )


class StopKeyGroups:
    """Number each document's stop key group, batch by batch.

    Documents with equal values in an AI filter's ``stop_key`` columns
    share a group number; the scheduler reads one document of a group at
    a time and skips the rest once the node's num_key_survivors survive.
    A number, once given, stays the same for the documents that arrive
    later.

    Args:
        node: The AiFilter, with a stop key.
        columns: The alias's value table, one row per document.
    """

    def __init__(self, node, columns):
        if columns is None:
            raise KeyError(
                f"the stop key of {node.alias!r} needs its value columns")
        self.node = node
        self.columns = columns
        self.numbers = {}

    def of(self, document_ids, copies=None) -> list:
        """Return each document's group number.

        A deduplicated document stands for its copies, which may sit in
        several groups: its entry is then a tuple with each copy's group.

        Args:
            document_ids: The documents' positions in the value table.
            copies: Per document, the positions of the documents it
                stands for, as an Arrow list array; None when each
                stands for itself.
        """
        if copies is None:
            return self._numbers(arrow_ids(document_ids))
        if isinstance(copies, pa.ChunkedArray):
            copies = copies.combine_chunks()
        groups = self._numbers(arrow_ids(copies.flatten()))
        offsets = copies.offsets.to_numpy()
        return [tuple(groups[start:end]) for start, end in zip(offsets, offsets[1:])]

    def _numbers(self, positions) -> list:
        """The group number of each position's stop key values."""
        values = [equality_keys(self.columns.column(name).take(positions),
                                nan_equal=True).to_pylist()
                  for name in self.node.stop_key]
        # every NaN is one key, though no two NaN floats are equal
        keys = (tuple(NAN_KEY if value != value else value for value in row)
                for row in zip(*values))
        return [self.numbers.setdefault(key, len(self.numbers)) for key in keys]


def copies_of(rows):
    """The documents each row stands for, deduplicated Rows' COPIES column, or None."""
    return (rows.column(COPIES) if isinstance(rows, Rows)
            and COPIES in rows.column_names else None)


def stop_keys(state, node) -> StopKeyGroups | None:
    """The stop key groups of an AI filter, or None without a stop key.

    Args:
        state: The query's runtime state, with the value columns.
        node: The AiFilter.
    """
    if not node.stop_key:
        return None
    return StopKeyGroups(node, state.get("columns", {}).get(node.alias))


def filter_stop_groups(state, node, rows) -> list | None:
    """Each document's stop key group for a filter over the given rows.

    Args:
        state: The query's runtime state, with the value columns.
        node: The AiFilter.
        rows: The documents it reads, as Rows or ids. Deduplicated Rows
            carry a COPIES column for a stop key.
    """
    keys = stop_keys(state, node)
    return None if keys is None else keys.of(id_positions(rows), copies_of(rows))
