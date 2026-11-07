"""Execute LLM filters and conditions on classification results."""


import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.backends.quail.executor.parts import (
    SingleResultPart,
    execute_parts,
    execution_prefix_tree,
)
from quail.backends.quail.executor.stages import Stage, filter_stages
from quail.backends.quail.executor.state import QueryExecutionState
from quail.common.expressions import accepts
from quail.common.pairs import columns_key
from quail.common.selection import group_ids
from quail.common.tokens import DocumentKeys, DocumentPrefixes
from quail.execution.ids import document_ids as arrow_ids
from quail.execution.ids import id_positions
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import filter_answer_table
from quail.execution.rows import Rows
from quail.execution.runtimes.filter import filter_scores, tested_table
from quail.progress import answer_sink


def execute_filters(state: QueryExecutionState, nodes, inputs) -> dict:
    """Run filter predicates together while each document retains its KV."""
    state.gpu_timing = bool(inputs.get("gpu_timing", False))
    node = nodes[0]
    documents, ids = inputs["documents"], inputs["document_ids"]
    parts = [FilterPart(member, ids, state.async_answers,
                        len(state.loaded_model.pipeline.canvas_ids))
             for member in nodes]
    if len(parts) == 1 and "document_done" in inputs:
        parts[0].document_done = inputs["document_done"]
    # the rows a limit keeps follow admission order, which borrowing
    # sorts by prefix; a query without ORDER BY promises no rows
    tree = execution_prefix_tree(node, documents, state.loaded_model.arena)
    return execute_parts(
        state, parts, documents, DocumentKeys(node.alias, ids),
        keep_kv=nodes[-1].keep_kv, prefix_tree=tree,
        attention=node.attention or "unified",
        limit=inputs.get("limit"), stop_groups=inputs.get("stop_groups"),
        num_key_survivors=inputs.get("num_key_survivors", 1),
        paged=node.arena_writes or len(nodes) > 1,
        label=f"filter ({len(nodes)} predicates)")


class FilterPart(SingleResultPart):
    """Filter stages and result tables within a pipeline."""

    execute_group = staticmethod(execute_filters)

    @staticmethod
    def model_inputs(node, inputs, context, sequence=None, nodes=None):
        if nodes is not None:
            inputs = inputs[node.node_id]
        return filter_inputs(context.state, node, next(iter(inputs.values())))

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        """Prepare one filter's requests and answer notifications."""
        return cls(nodes[0], ids, state.async_answers,
                   len(state.loaded_model.pipeline.canvas_ids))

    def __init__(self, node, ids, async_answers, canvas_tokens=0):
        self.node = node
        self.ids = ids
        self.canvas_tokens = canvas_tokens
        self.stages = filter_stages(
            [list(node.question_token_ids)],
            async_answers)
        self.document_done = filter_document_sink(node, ids)
        self.answers = {}

    @staticmethod
    def prepare_group(parts):
        """Retain the common question preamble across consecutive filters."""
        stages = filter_stages(
            [part.node.question_token_ids for part in parts],
            parts[0].stages[0].readout)
        for part, stage in zip(parts, stages):
            part.stages = [stage]

    def finish(self, every) -> int | None:
        """Record filter answers and leave token accounting to the pipeline."""
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


class ConditionGate(SingleResultPart):
    """Pipeline condition that drops documents whose tested values fail.

    A value is a settled column-operator result, such as a label, an
    extracted answer or span, or a map answer, or a source column of
    the table; the pipeline then skips the document's later stages and
    frees its KV.
    """

    stages = ()

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        """Bind a condition to the parts computing its columns and the source table."""
        node = nodes[0]
        source_id = node.inputs[0].source.node_id
        producers = {}
        for name in node.tested_columns:
            part = next((part for part in reversed(parts)
                         if name in part.tested_columns), None)
            if part is not None:
                producers[name] = part
        source = next(part for part in reversed(parts)
                      if any(member.node_id == source_id for member in part.nodes))
        return cls(node, producers, source,
                   context.sources.get(columns_key(node.aliases[0])))

    def __init__(self, node, producers, source, columns):
        self.node = node
        self.producers = producers  # column -> the part computing it
        self.source = source  # the part whose rows the Filter reads
        self.columns = columns  # the table's source columns, by document
        self.positions = {
            id(part): {document: index for index, document in enumerate(part.ids)}
            for part in producers.values()}
        self.accepted = {}
        self.decided = {}  # key -> kept, from decide, until gate asks

    @property
    def answer_dependencies(self):
        """The decoder answer streams the tested columns wait for."""
        return tuple(dict.fromkeys(
            part.first + part.column_sources[name][0]
            for name, part in self.producers.items() if part.answer_limits))

    def read(self, name: str, documents):
        """Return a tested column's values for documents, by document id."""
        part = self.producers.get(name)
        if part is not None:
            positions = self.positions[id(part)]
            return part.tested_values(name, [positions[document]
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

        The pipeline calls this with the documents entering the next
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

    def finish(self, every):
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


def filter_document_sink(node, document_ids):
    """The sequence's document_done callback, or None when nobody listens.

    Reports each chunk's finished documents to the answer sink as
    ``{"kind": "filter", "node", "alias", "stages", "documents"}`` where
    ``documents`` lists ``[row index, last stage asked, passed]`` per
    document. A document that passed its last stage survived the filter;
    every other listed document failed at the stage given.
    """
    sink = answer_sink()
    if sink is None:
        return None
    stages = 1

    def document_done(finished):
        sink({"kind": "filter", "node": node.node_id, "alias": node.alias,
              "stages": stages,
              "documents": [[int(document_ids[position]), stage, passed]
                            for position, stage, passed in finished]})

    return document_done


def stop_groups(node, columns, document_ids) -> list | None:
    """Return each document's stop key group, or None without a stop key.

    Documents with equal values in the node's ``stop_key`` columns
    share a group number; the scheduler reads one document of a group
    at a time and skips the rest once the node's num_key_survivors
    survive.

    Args:
        node: The AiFilter.
        columns: The alias's value table, one row per document.
        document_ids: The documents' positions in that table.
    """
    if not node.stop_key:
        return None
    if columns is None:
        raise KeyError(
            f"the stop key of {node.alias!r} needs its value columns")
    positions = arrow_ids(document_ids)
    return group_ids([columns.column(name).take(positions) for name in node.stop_key])


def filter_inputs(state, node, document_ids) -> dict:
    """Scheduler inputs for one filter sequence over the given documents."""
    document_ids = id_positions(document_ids)
    return {
        "documents": DocumentPrefixes(
            node.document_head, state["documents"][node.alias], document_ids
        ),
        "document_ids": document_ids,
        "limit": state["filter_limit"],
        "stop_groups": stop_groups(
            node, state.get("columns", {}).get(node.alias), document_ids),
        "num_key_survivors": node.num_key_survivors,
        "document_done": filter_document_sink(node, document_ids),
    }
