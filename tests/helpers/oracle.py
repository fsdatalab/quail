"""One fake model for the end-to-end tests: it answers from planted truth.

The oracle stands in for the GPU. It answers every model operator
from the truth it was given and builds each result with the
production builders (filter_result, score_rows, extraction_rows, and
the join answer record the executor writes), so a test never spells
a port name or a result shape.
"""

from __future__ import annotations

import time
from functools import partial

import numpy as np
import pyarrow as pa

from quail.backends.base import GpuContext
from quail.backends.quail.executor.operators.filter import filter_result
from quail.backends.quail.executor.operators.sae_features import (
    FEATURE_LIST,
)
from quail.backends.quail.reranker import (
    RerankerBatch,
    RerankerModelExecution,
    score_rows,
)
from quail.execution.answer_stops import Stop, answer_stop, copied_completions
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import true_answer_rows
from quail.execution.rows import Rows, extraction_rows
from quail.execution.spans import locate_span
from quail.execution.streaming import region_stream
from quail.physical import (
    AiClassify,
    AiExtract,
    AiFilter,
    AiJoin,
    AiMap,
    AiSaeFeatures,
    AiScore,
    ClassifySpec,
)
from quail.physical.plan import graph_from_envelope
from quail.specs import H100_SXM, QWEN3_4B_FP8

# the most seconds an operator above a region may take to tighten the
# region's hint after receiving a row; a slower one fails the test
HINT_WAIT_SECONDS = 30.0


def question_text(token_ids) -> str:
    """Return the text of a question's token ids in the tests' tokenizer."""
    from helpers.sessions import encoder

    found = encoder().decode([int(token) for token in token_ids])
    return found.decode() if isinstance(found, bytes) else found


class Oracle:
    """A model execution whose answers come from planted truth.

    Args:
        filters: alias -> {question key: a bit per document}, where the
            key is matched against the start of the question's tokens;
            or a callable (alias, document, written_position) -> bool.
        joins: {(anchor, *partners): rule(anchor_id, *partner_ids)}; or
            a callable (node, assignment) -> bool over alias -> id.
        labels: A callable (spec, ids) -> the label of one row or None,
            a sequence indexed by the first id, or None.
        scores: A callable (spec, ids) -> the score of one row.
        answers: A callable (spec, document) -> the extracted text or
            None; or a sequence indexed by document for every question.
        maps: A callable (spec, document) -> the written answer; None
            writes the map's name and the document's index.
        features: A callable (spec, document) -> a feature list of
            {"feature", "strength"} mappings; None plants one feature
            per document, its id modulo 7 at strength id + 1.
        probabilities: Whether a classification reports label
            probabilities when its spec asks for them.
        retain: How many of a keep_kv filter's survivors the oracle
            retains in the arena, as the executor would.
        join_metrics: Metrics every join stage reports beside its rows.
        streams: Whether a score call reports each row as it is scored.
        tightens_after: Source rows of the first alias after whose
            filter answer or score the oracle waits until the region's
            hint tightens, so a test decides when the operator above the
            region tightens its test. With it, a filter reports each
            document as it finishes.
        pipelines: Whether the executor groups operators into pipelines,
            as the GPU executor does; the oracle then runs a pipeline's
            nodes in order.
    """

    # the oracle runs one node at a time; a pipeline falls back to that
    execute_pipeline = None

    def __init__(self, filters=None, joins=None, labels=None, scores=None,
                 answers=None, features=None, maps=None, *,
                 probabilities=False, retain=0, join_metrics=None,
                 streams=False, pipelines=False, tightens_after=()):
        self.filters = filters or {}
        self.joins = joins or {}
        self.labels = labels
        self.scores = scores or (lambda spec, ids: (sum(ids) + 1) / 10)
        self.answers = answers
        self.features = features
        self.maps = maps or (lambda spec, document: f"{spec.name} {document}")
        self.probabilities = probabilities
        self.retain = retain
        self.join_metrics = join_metrics
        self.streams_answers = streams
        self.tightens_after = frozenset(tightens_after)
        self.arena = None
        self.texts = {}
        self.documents = {}
        # by alias, each document's row in the registered table, which the
        # planted truth uses, for an input that holds some of a table's rows
        self.source_rows = {}
        self.calls = []       # (spec name, rows) of every score or label call
        self.asked = []       # (alias, document) of every filter answer
        # characters of map and extraction answers written, one a token,
        # and the answers stopped before their end
        self.answer_tokens = 0
        self.stopped_early = 0
        if pipelines:
            self.execute_pipeline = self._execute_pipeline

    def _source(self, alias, document) -> int:
        """The registered table's row of an input's document position."""
        rows = self.source_rows.get(alias)
        return int(document) if rows is None else rows[int(document)]

    def _sources(self, aliases, ids) -> tuple:
        return tuple(self._source(alias, document)
                     for alias, document in zip(aliases, ids))

    # ---- the executor's model execution interface ----------------

    def _execute_pipeline(self, nodes, inputs, context) -> dict:
        """Run a pipeline's nodes in order, each reading the one before it."""
        results = {}
        for node in nodes:
            node_inputs = dict(inputs[node.node_id])
            for port in node.inputs:
                if port.source.node_id in results:
                    node_inputs[port.name] = results[
                        port.source.node_id].outputs[port.source.port]
            results[node.node_id] = context.runtimes[node.runtime_key].execute(
                node, node_inputs, context)
        return results

    def execute(self, node, inputs) -> NodeResult:
        if isinstance(node, AiFilter):
            return self._filter(node, inputs)
        if isinstance(node, AiJoin):
            return self.execute_predicates((node,), inputs)[node.node_id]
        if isinstance(node, AiExtract):
            return self._extract(node, inputs)
        if isinstance(node, AiSaeFeatures):
            return self._sae_features(node, inputs)
        if isinstance(node, AiMap):
            return self._map(node, inputs)
        if isinstance(node, AiClassify) and node.spec.partner is not None:
            return self._classify_joined(node, inputs)
        if isinstance(node, (AiScore, AiClassify)):
            return self._reranker().execute(node, inputs.get("score_inputs", inputs))
        raise TypeError(f"the oracle has no answers for {node.type_name}")

    def execute_predicates(self, nodes, inputs) -> dict:
        """Answer a group of join stages, each over the survivors of the last."""
        anchor_ids = list(inputs["anchor_ids"])
        lists_for = inputs.get("anchor_partners")
        survivors = set(range(len(anchor_ids)))
        results = {}
        for index, node in enumerate(nodes):
            partners = inputs["partner_indices"][node.written_position]
            members = {} if lists_for and node.pairs_from else None
            rows = {}
            for local in sorted(survivors):
                mine = (range(len(partners)) if members is None
                        else list(lists_for(anchor_ids[local], index)))
                if members is not None:
                    members[local] = mine
                rows[local] = [
                    self._join_answer(node, anchor_ids[local], partners[member])
                    for member in mine]
            if inputs.get("anchor_done") is not None:
                for local, row in rows.items():
                    inputs["anchor_done"](local, row)
            passing = {local for local, row in rows.items() if any(row)}
            survivors = (survivors - passing if node.semantics == "anti"
                         else passing)
            kept = [anchor_ids[local] for local in sorted(survivors)]
            answer = {
                "rows": rows, "anchor_index": anchor_ids,
                "partner_index": partners, "anchor_partners": members,
                "anchor": node.anchor, "partners": list(node.partners),
                "semantics": node.semantics, "selectivity": node.selectivity,
                "written_position": node.written_position,
            }
            metrics = self.join_metrics or NodeMetrics()
            results[node.node_id] = NodeResult(
                {f"rows:{node.anchor}": Rows({node.anchor: kept}),
                 f"join_answers:{node.written_position}": answer},
                NodeMetrics(
                    input_rows=len(anchor_ids), output_rows=len(kept),
                    evaluated_document_pairs=sum(len(row) for row in rows.values()),
                    gpu_s=metrics.gpu_s, peak_gpu_bytes=metrics.peak_gpu_bytes,
                    extension={**metrics.extension, "answers": [rows]}))
        return results

    # ---- the reranker interface, for scores and labels -------------

    def score(self, spec, rows, documents, on_answers=None) -> RerankerBatch:
        rows = np.asarray(rows).reshape(-1, len(spec.aliases))
        self.calls.append((spec.name, rows.tolist()))
        ids = [self._sources(spec.aliases, row) for row in rows]
        if isinstance(spec, ClassifySpec):
            values = np.asarray([self._label(spec, row) for row in ids], dtype=object)
            probabilities = None
            if self.probabilities and spec.probabilities:
                probabilities = np.asarray([
                    [np.nan] * len(spec.labels) if value is None
                    else [0.8 if label == value else 0.1 for label in spec.labels]
                    for value in values])
            return RerankerBatch(values, fresh_tokens=len(rows), cached_tokens=0,
                                 suffix_tokens=3 * len(rows),
                                 probabilities=probabilities)
        scores = [self.scores(spec, row) for row in ids]
        if on_answers is not None:
            for position, value in enumerate(scores):
                self._answered(ids[position][0], partial(
                    on_answers, np.array([position]), [value]))
        return RerankerBatch(scores, fresh_tokens=len(rows), cached_tokens=0)

    # ---- answering ------------------------------------------------

    def _filter(self, node, inputs) -> NodeResult:
        document_ids = list(inputs["document_ids"])
        groups = inputs.get("stop_groups")
        needed = inputs.get("num_key_survivors", 1)
        survivors = {}
        answers = {}
        stream = region_stream()
        for local, document in enumerate(document_ids):
            # a document that stands for copies has one group per copy
            found = (() if groups is None else groups[local]
                     if isinstance(groups[local], tuple) else (groups[local],))
            # the scheduler skips a document once each of its stop key
            # groups has enough survivors
            if found and all(survivors.get(group, 0) >= needed for group in found):
                continue
            # and a document the region's hint drops, before admitting it
            if stream is not None and not stream.keeps(node.alias, int(document)):
                continue
            self.asked.append((node.alias, int(document)))
            answers[local] = [self._filter_answer(node, document)]
            if self.tightens_after and inputs.get("document_done") is not None:
                self._answered(self._source(node.alias, document), partial(
                    inputs["document_done"], [(local, 0, bool(answers[local][0]))]))
            if answers[local][0]:
                for group in found:
                    survivors[group] = survivors.get(group, 0) + 1
        if inputs.get("document_done") is not None and not self.tightens_after:
            # the real loop reports a chunk's finished documents by position
            inputs["document_done"]([
                (local, 0, bool(row[0])) for local, row in answers.items()])
        if node.keep_kv and self.arena is not None:
            kept = [document for local, document in enumerate(document_ids)
                    if answers[local][0]]
            for document in kept[:self.retain]:
                self.arena.retain((node.alias, document), 16)
        return filter_result(node, answers, 10 * len(answers), document_ids)

    def _answered(self, row: int, send) -> None:
        """Send one answer on; after a row in tightens_after, wait for the hint.

        Args:
            row: The answered row of the registered table.
            send: Sends the answer on toward the operator above the region.

        Raises:
            AssertionError: The hint did not tighten within
                HINT_WAIT_SECONDS.
        """
        stream = region_stream()
        hint = None if stream is None else stream.hint
        if hint is None or row not in self.tightens_after:
            send()
            return
        before = hint.generation()
        send()
        deadline = time.monotonic() + HINT_WAIT_SECONDS
        while hint.generation() == before:
            if time.monotonic() > deadline:
                raise AssertionError(f"the hint did not tighten after row {row}")
            time.sleep(0.001)

    def _filter_answer(self, node, document) -> bool:
        document = self._source(node.alias, document)
        if callable(self.filters):
            return bool(self.filters(node.alias, document, node.written_position))
        truth = self.filters[node.alias]
        question = question_text(node.question_token_ids)
        for key, bits in truth.items():
            if key in question:
                return bool(bits[document])
        raise KeyError(f"no filter truth for {node.alias!r} matches {question[:40]!r}")

    def _join_answer(self, node, anchor_id, partner_ids) -> bool:
        anchor_id, *partner_ids = self._sources(
            (node.anchor, *node.partners), (anchor_id, *partner_ids))
        if callable(self.joins):
            assignment = dict(zip((node.anchor, *node.partners),
                                  (anchor_id, *partner_ids)))
            return bool(self.joins(node, assignment))
        rule = self.joins[(node.anchor, *node.partners)]
        return bool(rule(anchor_id, *partner_ids))

    def _label(self, spec, ids):
        if callable(self.labels):
            return self.labels(spec, ids)
        return None if self.labels is None else self.labels[ids[0]]

    def _classify_joined(self, node, inputs) -> NodeResult:
        """Label the pairs a join kept; a pair without a label has no row."""
        (answers,) = inputs["score_inputs"].values()
        spec = node.spec
        pairs = []
        if isinstance(answers, pa.Table):
            # the join's answers over every pair, as a table
            kept = true_answer_rows(answers)
            pairs = list(zip(*(kept.column(alias).to_pylist()
                               for alias in spec.aliases)))
        else:
            for local, row in answers["rows"].items():
                anchor = answers["anchor_index"][local]
                members = (answers["anchor_partners"] or {}).get(local)
                for position, bit in enumerate(row):
                    if not bit:
                        continue
                    member = position if members is None else members[position]
                    (partner,) = answers["partner_index"][member]
                    pairs.append((int(anchor), int(partner)))
        labels = {pair: self._label(spec, pair) for pair in pairs}
        labeled = [pair for pair in pairs if labels[pair] is not None]
        rows = np.asarray(labeled, dtype=np.int32).reshape(-1, 2)
        return NodeResult(
            {node.rows_port: score_rows(rows, spec.aliases, spec.name,
                                        [labels[pair] for pair in labeled],
                                        pa.string())},
            NodeMetrics(input_rows=len(pairs), output_rows=len(labeled)))

    def _extract(self, node, inputs) -> NodeResult:
        spec = node.spec
        ids = [int(document) for document in inputs["document_ids"]]
        texts = self.texts.get(spec.alias)
        found = [self._answer(spec, self._source(spec.alias, document))
                 for document in ids]
        found = [self._written(node, text, None if texts is None
                               else str(texts[document]))
                 for document, text in zip(ids, found)]
        spans = [None if text is None or texts is None
                 else locate_span(str(texts[document]), text)
                 for document, text in zip(ids, found)]
        return NodeResult(
            {node.rows_port: extraction_rows(
                spec.alias, spec.name, spec.span_name, ids, found, spans)},
            NodeMetrics(input_rows=len(ids), output_rows=len(ids),
                        evaluated_documents=len(ids)))

    def _sae_features(self, node, inputs) -> NodeResult:
        spec = node.spec
        ids = [int(document) for document in inputs["document_ids"]]
        values = [self._feature_list(spec, self._source(spec.aliases[0], document))
                  for document in ids]
        rows = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
        return NodeResult(
            {node.rows_port: score_rows(rows, spec.aliases, spec.name,
                                        values, FEATURE_LIST)},
            NodeMetrics(input_rows=len(ids), output_rows=len(ids),
                        evaluated_documents=len(ids)))

    def _map(self, node, inputs) -> NodeResult:
        spec = node.spec
        ids = [int(document) for document in inputs["document_ids"]]
        rows = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
        return NodeResult(
            {node.rows_port: score_rows(
                rows, spec.aliases, spec.name,
                [self._written(node, self.maps(spec, document), None)
                 for document in ids], pa.string())},
            NodeMetrics(input_rows=len(ids), output_rows=len(ids),
                        evaluated_documents=len(ids)))

    def _written(self, node, text, source):
        """Write an answer one character a token, as far as the executor would.

        The executor stops once the node's tests are decided on the
        answer so far; source is an extraction's document text.
        """
        stop = answer_stop(node)
        if text is None or stop is None:
            self.answer_tokens += 0 if text is None else len(text)
            return text
        for length in range(1, len(text)):
            prefix = text[:length]
            if source is None:
                completions = [dict(prefix=prefix)]
            else:
                # the copy is at any place the document holds the prefix
                completions = copied_completions(source, [
                    (start, start + length)
                    for start in range(len(source))
                    if source.startswith(prefix, start)])
            if stop.decide_any(completions) is not Stop.CONTINUE:
                self.answer_tokens += length
                self.stopped_early += 1
                return prefix if source is None else prefix.strip()
        self.answer_tokens += len(text)
        return text

    def _feature_list(self, spec, document) -> list:
        if callable(self.features):
            return self.features(spec, document)
        return [{"feature": document % 7, "strength": float(document + 1)}]

    def _answer(self, spec, document):
        if callable(self.answers):
            return self.answers(spec, document)
        return None if self.answers is None else self.answers[document]

    def _reranker(self) -> RerankerModelExecution:
        return RerankerModelExecution(GpuContext(
            gpu_index=0, gpus=1, model=QWEN3_4B_FP8, device=H100_SXM,
            query_settings={"reranker": self, "documents": self.documents}))


def oracle_executor(oracle: Oracle, **report):
    """Return a physical executor that runs each request's graph on the oracle.

    Args:
        oracle: The oracle answering the model operators.
        **report: Report fields the response carries, such as wall_s.
    """
    def execute(request):
        from helpers.fakes import graph_state
        from quail.backends.quail.graph import execute_single_graph
        from quail.execution.types import PhysicalResponse
        from quail.physical import Scan

        graph = graph_from_envelope(request.plan)
        scans = [node for node in graph.nodes if isinstance(node, Scan)]
        oracle.documents = {node.alias: request.inputs[node.input_id].documents
                            for node in scans}
        oracle.texts = {node.alias: request.inputs[node.input_id].texts
                        for node in scans}
        oracle.source_rows = {alias: rows.to_pylist()
                              for alias, rows in request.source_rows.items()}
        state = graph_state(oracle, oracle.documents)
        state["texts"] = oracle.texts
        oracle.arena = state["gpu"].arena
        if getattr(request, "relations", None):
            state["columns"] = request.column_tables()
        result = execute_single_graph(state, {
            "filter_limit": None,
            "deduplicate": request.plan["settings"].get("deduplicate", {})}, graph)
        outputs = result.pop("_outputs")
        return PhysicalResponse(outputs, {
            **result, "wall_s": 1.0, "boot_s": 0.5, "fresh_tokens": 1234, **report})

    execute.oracle = oracle
    return execute


def finish(query, oracle: Oracle, **report):
    """Run a query to its result on the oracle."""
    from quail.api.execute import execute_query

    return execute_query(query, physical_executor=oracle_executor(oracle, **report))
