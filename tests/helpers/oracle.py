"""One fake model for the end-to-end tests: it answers from planted truth.

The oracle stands in for the GPU. Its ModelOperators run in the
production RegionExecution, one node at a time: the region prepares
each node's documents, its stop key groups, its join inputs, and its
candidate rows, and reports the rows each operator finishes to the
region's FinishedRows, one answered row at a time. The oracle answers
each pushed document from the truth it was given and builds each
result and each finished row with the production builders
(filter_result, scored_result, extraction_rows, passed_rows,
finished_scores, settled_pairs, and the join answer record the
executor writes), so a test never spells a port name or a result
shape.
"""

from __future__ import annotations

import copy
import json
import threading
import time
import traceback
from functools import partial

import _quail_native
import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.model_operator import ModelOperator
from quail.backends.quail.executor.operators.filter import (
    FilterOperator,
    filter_result,
    filter_stop_groups,
)
from quail.backends.quail.executor.operators.join import (
    prepare_join_inputs,
    settled_pairs,
)
from quail.backends.quail.executor.operators.sae_features import (
    FEATURE_LIST,
)
from quail.backends.quail.executor.operators.score import (
    candidate_rows,
    finished_scores,
    hinted_rows,
    scored_result,
)
from quail.backends.quail.executor.region import (
    RegionDocuments,
    pipeline_rows,
    region_of,
)
from quail.backends.quail.reranker import score_rows, scored_batch
from quail.execution.answer_stops import Stop, answer_stop, copied_completions
from quail.execution.finished import (
    finished_rows,
    finished_table,
    keeps,
    region_hint,
)
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import true_answer_rows
from quail.execution.rows import Rows, extraction_rows
from quail.execution.spans import locate_span
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
from quail.progress import answer_sink, set_answer_sink

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
        truth: A callable (kind, question, texts) -> the answer of any
            operator, from the question's text and the text of each
            document it reads; kind is "filter", "join", "label",
            "score", "map", or "extract". It replaces the arguments
            above, and reads each input by position.
        probabilities: Whether a classification reports label
            probabilities when its spec asks for them.
        retain: How many of a keep_kv filter's survivors the oracle
            retains in the arena, as the executor would.
        join_metrics: Metrics every join stage reports beside its rows.
        tightens_after: Source rows of the first alias after whose
            filter answer, join anchor, or score the oracle waits until
            the region's hint tightens, so a test decides when the
            operator above the region tightens its test. With it, a
            filter reports each document as it finishes.
        pipelines: Whether the runner groups operators into chains, as
            it does for the GPU executor; the oracle then runs a chain's
            nodes one at a time, in order.
    """

    # the oracle has no GPU loop
    query = None

    def __init__(self, filters=None, joins=None, labels=None, scores=None,
                 answers=None, features=None, maps=None, *, truth=None,
                 probabilities=False, retain=0, join_metrics=None,
                 pipelines=False, tightens_after=()):
        self.filters = filters or {}
        self.joins = joins or {}
        self.labels = labels
        self.scores = scores or (lambda spec, ids: (sum(ids) + 1) / 10)
        self.answers = answers
        self.features = features
        self.maps = maps or (lambda spec, document: f"{spec.name} {document}")
        self.truth = truth
        if truth is not None:
            self.scores = lambda spec, ids: truth(
                "score", spec.query_template, self._texts(spec.aliases, ids))
            self.maps = lambda spec, document: truth(
                "map", spec.query_template,
                self._texts(spec.aliases[:1], [document]))
            self.answers = lambda spec, document: truth(
                "extract", question_text(spec.prompt_token_parts[1]),
                self._texts(spec.aliases[:1], [document]))
        self.probabilities = probabilities
        self.retain = retain
        self.join_metrics = join_metrics
        self.groups_operators = pipelines
        self.tightens_after = frozenset(tightens_after)
        self.arena = None
        self.last_node = None
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
        self.operator_types = {
            AiFilter.type_name: OracleFilter, AiJoin.type_name: OracleJoin,
            AiScore.type_name: OracleScore, AiClassify.type_name: OracleScore,
            AiExtract.type_name: OracleExtract, AiMap.type_name: OracleMap,
            AiSaeFeatures.type_name: OracleFeatures}

    def _source(self, alias, document) -> int:
        """The registered table's row of an input's document position."""
        rows = None if self.truth is not None else self.source_rows.get(alias)
        return int(document) if rows is None else rows[int(document)]

    def _texts(self, aliases, ids) -> tuple:
        """The text of each input's document, an empty string for NULL."""
        found = []
        for alias, document in zip(aliases, ids):
            text = self.texts[alias][int(document)]
            text = text.as_py() if hasattr(text, "as_py") else text
            found.append("" if text is None else str(text))
        return tuple(found)

    def _sources(self, aliases, ids) -> tuple:
        return tuple(self._source(alias, document)
                     for alias, document in zip(aliases, ids))

    # ---- the executor's model execution interface ----------------

    def execute_pipeline(self, nodes, inputs, context) -> dict:
        """Run each node in order, as a chain of its own in the context's region."""
        if len(nodes) == 1:
            return region_of(self, context).run(nodes, inputs)
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

    def limit(self, context, node) -> int | None:
        """The rows after which a node stops, or None for every row.

        The GPU loop stops a chain once enough documents pass its last
        operator. The oracle runs a chain one node at a time, so only
        the last node stops; the nodes before it answer every document.
        """
        return (context.state.get("filter_limit")
                if node.node_id == self.last_node else None)

    # ---- answering ------------------------------------------------

    def _answered(self, row: int, send) -> None:
        """Send one answer on; after a row in tightens_after, wait for the hint.

        Args:
            row: The answered row of the registered table.
            send: Sends the answer on toward the operator above the region.

        Raises:
            AssertionError: The hint did not tighten within
                HINT_WAIT_SECONDS.
        """
        hint = region_hint()
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
        if self.truth is not None:
            return bool(self.truth(
                "filter", question_text(node.question_token_ids),
                self._texts([node.alias], [document])))
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
        if self.truth is not None:
            return bool(self.truth(
                "join", question_text(node.frame_token_ids),
                self._texts((node.anchor, *node.partners),
                            (anchor_id, *partner_ids))))
        if callable(self.joins):
            assignment = dict(zip((node.anchor, *node.partners),
                                  (anchor_id, *partner_ids)))
            return bool(self.joins(node, assignment))
        rule = self.joins[(node.anchor, *node.partners)]
        return bool(rule(anchor_id, *partner_ids))

    def _label(self, spec, ids):
        if self.truth is not None:
            return self.truth("label", spec.query_template,
                              self._texts(spec.aliases, ids))
        if callable(self.labels):
            return self.labels(spec, ids)
        return None if self.labels is None else self.labels[ids[0]]

    def _classify_joined(self, node, inputs) -> NodeResult:
        """Label the pairs a join kept; a pair without a label has no row."""
        (answers,) = inputs.values()
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

    def _extract(self, node, document_ids) -> NodeResult:
        spec = node.spec
        ids = [int(document) for document in document_ids]
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

    def _sae_features(self, node, document_ids) -> NodeResult:
        spec = node.spec
        ids = [int(document) for document in document_ids]
        values = [self._feature_list(spec, self._source(spec.aliases[0], document))
                  for document in ids]
        rows = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
        return NodeResult(
            {node.rows_port: score_rows(rows, spec.aliases, spec.name,
                                        values, FEATURE_LIST)},
            NodeMetrics(input_rows=len(ids), output_rows=len(ids),
                        evaluated_documents=len(ids)))

    def _map(self, node, document_ids) -> NodeResult:
        spec = node.spec
        ids = [int(document) for document in document_ids]
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
        stop = answer_stop(node, region_hint())
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


class OracleOperator(ModelOperator):
    """A node the oracle answers from planted truth as documents are pushed.

    Args:
        region: The RegionExecution running the node.
        node: The physical node.
        num_documents: How many documents, or rows, the node reads.
    """

    def __init__(self, region, node, num_documents):
        self.region = region
        self.oracle = region.model
        self.node = node
        self.nodes = (node,)
        self.num_documents = num_documents
        self.result_value = None
        self.finished = []      # finished rows since the last take

    def layout(self, region):
        return RegionDocuments(range(self.num_documents), None)

    def read_answers(self, every):
        return None

    def close(self, tokens, spans, stats, offset) -> dict:
        return self.results()

    def results(self) -> dict:
        return {self.node.node_id: self.result_value}


class OracleFilter(FilterOperator):
    """An AI filter answered from truth, streamed as the production one is.

    It skips a document whose stop key groups have enough survivors, or
    that the region's hint drops, before asking it, as admission does.
    """

    @classmethod
    def build(cls, region, nodes, chain):
        node = nodes[0]
        operator = cls.__new__(cls)
        operator.region, operator.oracle, operator.node = region, region.model, node
        operator.ids = list(region.document_ids())
        operator.groups = filter_stop_groups(
            region.context.state, node, pipeline_rows(node, region.inputs))
        # a key stops once a document passes the chain's last filter, as
        # the production loop counts survivors of its last stage
        for earlier in chain:
            if isinstance(earlier, OracleFilter):
                earlier.groups = None
        operator.answers, operator.finished = {}, []
        return operator

    def layout(self, region):
        return RegionDocuments(range(len(self.ids)), None)

    def push(self, documents):
        oracle, node = self.oracle, self.node
        survivors = {}
        # each document is a chunk of its own, unless an answer sink
        # takes the answers, which the real loop sends a chunk at a time
        sink = answer_sink()
        one_by_one = ((finished_rows() is not None and sink is None)
                      or (oracle.tightens_after and (
                          sink is not None or finished_rows() is not None)))
        limit = oracle.limit(self.region.context, node)
        num_passed = sum(1 for answer in self.answers.values() if answer[0])
        for local in documents:
            if limit is not None and num_passed >= limit:
                break
            document = self.ids[local]
            # a document that stands for copies has one group per copy
            found = (() if self.groups is None else self.groups[local]
                     if isinstance(self.groups[local], tuple)
                     else (self.groups[local],))
            if found and all(survivors.get(group, 0) >= node.num_key_survivors
                             for group in found):
                continue
            if not keeps(node.alias, int(document)):
                continue
            oracle.asked.append((node.alias, int(document)))
            passed = oracle._filter_answer(node, document) != node.negated
            self.answers[local] = [passed]
            self.finished.append((local, passed))
            if one_by_one:
                oracle._answered(oracle._source(node.alias, document),
                                 partial(self.region.send_finished, self))
            if passed:
                num_passed += 1
                for group in found:
                    survivors[group] = survivors.get(group, 0) + 1
        if not one_by_one:
            self.region.send_finished(self)
        if node.keep_kv and oracle.arena is not None:
            kept = [document for local, document in enumerate(self.ids)
                    if self.answers.get(local, [False])[0]]
            for document in kept[:oracle.retain]:
                oracle.arena.retain((node.alias, document), 16)

    def read_answers(self, every):
        return None

    def close(self, tokens, spans, stats, offset) -> dict:
        self.result_value = filter_result(
            self.node, self.answers, 10 * len(self.answers), self.ids)
        return self.results()


class OracleJoin(OracleOperator):
    """A group of join stages answered from truth over the region's join inputs."""

    combine = True

    @classmethod
    def build(cls, region, nodes, chain):
        inputs = prepare_join_inputs(nodes, region.inputs, region.context)
        operator = cls(region, nodes[0], len(inputs["anchor_ids"]))
        operator.nodes, operator.inputs = tuple(nodes), inputs
        return operator

    def push(self, documents):
        self._results = self._predicates(self.nodes, self.inputs)

    def take_finished(self):
        finished, self.finished = self.finished, []
        return finished

    def results(self) -> dict:
        return self._results

    def _predicates(self, nodes, inputs) -> dict:
        """Answer a group of join stages, each over the survivors of the last."""
        oracle = self.oracle
        anchor_ids = list(inputs["anchor_ids"])
        lists_for = inputs.get("anchor_partners")
        survivors = set(range(len(anchor_ids)))
        skipped = set()
        results = {}
        for index, node in enumerate(nodes):
            partners = inputs["partner_indices"][node.written_position]
            members = {} if lists_for and node.pairs_from else None
            rows = {}
            for local in sorted(survivors):
                if index == 0 and not keeps(node.anchor, int(anchor_ids[local])):
                    # admission skips an anchor the region's hint drops
                    skipped.add(local)
                    continue
                mine = (range(len(partners)) if members is None
                        else list(lists_for(anchor_ids[local], index)))
                if members is not None:
                    members[local] = mine
                rows[local] = [
                    oracle._join_answer(node, anchor_ids[local], partners[member])
                    for member in mine]
                # an anchor's pairs are reported once answered
                self.finished = settled_pairs(node, int(anchor_ids[local]), [
                    partners[member] for member, answer in zip(mine, rows[local])
                    if answer])
                oracle._answered(oracle._source(node.anchor, anchor_ids[local]),
                                 partial(self.region.send_finished, self))
            for local, row in rows.items():
                inputs["anchor_done"](local, row)
            passing = {local for local, row in rows.items() if any(row)}
            survivors = (survivors - passing if node.semantics == "anti"
                         else passing) - skipped
            kept = [anchor_ids[local] for local in sorted(survivors)]
            answer = {
                "rows": rows, "anchor_index": anchor_ids,
                "partner_index": partners, "anchor_partners": members,
                "anchor": node.anchor, "partners": list(node.partners),
                "semantics": node.semantics, "selectivity": node.selectivity,
                "written_position": node.written_position,
            }
            metrics = oracle.join_metrics or NodeMetrics()
            results[node.node_id] = NodeResult(
                {f"rows:{node.anchor}": Rows({node.anchor: kept}),
                 f"join_answers:{node.written_position}": answer},
                NodeMetrics(
                    input_rows=len(anchor_ids), output_rows=len(kept),
                    evaluated_document_pairs=sum(len(row) for row in rows.values()),
                    gpu_s=metrics.gpu_s, peak_gpu_bytes=metrics.peak_gpu_bytes,
                    extension={**metrics.extension, "answers": [rows]}))
        return results


class OracleScore(OracleOperator):
    """A score or classification of its candidate rows, answered from truth.

    The rows, their priors, and the rows the region's hint skips come
    from the production score operator's helpers; a classification of
    the pairs a join kept labels them from the join's answers.
    """

    @classmethod
    def build(cls, region, nodes, chain):
        node = nodes[0]
        inputs = region.node_inputs(node)
        if isinstance(node, AiClassify) and node.spec.partner is not None:
            operator = cls(region, node, 1)
            operator.joined = inputs
            return operator
        if node.spec is None:
            raise ValueError("AI.SCORE needs a score specification")
        rows, priors = candidate_rows(node, inputs)
        operator = cls(region, node, len(rows))
        operator.joined = None
        operator.rows = np.asarray(rows.array(), dtype=np.int32).reshape(
            -1, len(node.spec.aliases))
        operator.priors, operator.skip = priors, hinted_rows(node, rows)
        return operator

    def push(self, documents):
        if self.joined is not None:
            self.result_value = self.oracle._classify_joined(self.node, self.joined)
            return
        oracle, node, spec, rows = self.oracle, self.node, self.node.spec, self.rows
        if len(spec.arguments) != len(spec.aliases):
            raise ValueError(
                "AI.SCORE needs one prompt argument per document relation")
        ids = [oracle._sources(spec.aliases, row) for row in rows]
        labeled = isinstance(spec, ClassifySpec)
        sink = answer_sink()
        # the scheduler asks skip as it admits each row, in order; each
        # answered row is reported as the real loop reports a chunk's
        asked, values, probabilities = [], [], []
        limit = oracle.limit(self.region.context, node)
        for position in range(len(rows)):
            if limit is not None and len(asked) >= limit:
                break
            if self.skip is not None and self.skip(position):
                continue
            asked.append(position)
            value = (oracle._label(spec, ids[position]) if labeled
                     else oracle.scores(spec, ids[position]))
            values.append(value)
            probability = None
            if labeled and oracle.probabilities and spec.probabilities:
                probability = [np.nan] * len(spec.labels) if value is None else [
                    0.8 if label == value else 0.1 for label in spec.labels]
                probabilities.append(probability)
            oracle._answered(ids[position][0], partial(
                self._send, rows[[position]], value, probability, sink))
        if len(rows):
            oracle.calls.append((spec.name, rows[asked].tolist()))
        self.result_value = scored_result(
            node, rows[asked], values, self.priors, fresh_tokens=len(asked),
            suffix_tokens=(3 * len(asked) if labeled else 0),
            probabilities=(np.asarray(probabilities).reshape(-1, len(spec.labels))
                           if probabilities else None))

    def _send(self, rows, value, probability, sink):
        if sink is not None and value is not None:
            sink(scored_batch(self.node, rows, [value]))
        self.finished = finished_scores(
            self.node, rows, [value],
            None if probability is None else np.asarray([probability]))
        self.region.send_finished(self)

    def take_finished(self):
        finished, self.finished = self.finished, []
        return finished


class OracleDocuments(OracleOperator):
    """An extraction, a map, or a feature call over the chain's documents."""

    @classmethod
    def build(cls, region, nodes, chain):
        operator = cls(region, nodes[0], len(region.document_ids()))
        operator.ids = region.document_ids()
        return operator


    def push(self, documents):
        port = self.node.rows_port
        if not len(self.ids):
            self.result_value = self.answer(self.ids)
            return
        # each document's answer is reported as it ends, as the real
        # loop reports a chunk's
        parts = []
        limit = self.oracle.limit(self.region.context, self.node)
        for document in self.ids[:limit]:
            parts.append(self.answer([document]).outputs[port])
            self.finished = [finished_table(self.node.node_id, port,
                                            parts[-1].table())]
            self.region.send_finished(self)
        count = len(parts)
        self.result_value = NodeResult({port: Rows.concat(parts)}, NodeMetrics(
            input_rows=count, output_rows=count, evaluated_documents=count))

    def take_finished(self):
        finished, self.finished = self.finished, []
        return finished if finished_rows() is not None else []


class OracleExtract(OracleDocuments):
    def answer(self, ids):
        return self.oracle._extract(self.node, ids)


class OracleMap(OracleDocuments):
    def answer(self, ids):
        return self.oracle._map(self.node, ids)


class OracleFeatures(OracleDocuments):
    def answer(self, ids):
        return self.oracle._sae_features(self.node, ids)


def _last_model_node(model, graph) -> str | None:
    """The id of the one model node no other model node reads, or None."""
    nodes = [node for node in graph.nodes if node.type_name in model.operator_types]
    read = {port.source.node_id for node in nodes for port in node.inputs}
    last = [node.node_id for node in nodes if node.node_id not in read]
    return last[0] if len(last) == 1 else None


def _serve(model, path, description=None):
    """Serve each region as a GPU worker does, answering from the oracle.

    Args:
        model: The oracle this worker answers from.
        path: The socket Quail listens at for this worker.
        description: What the worker answers a describe with, as
            ``engine_description`` returns it.
    """
    from helpers.fakes import graph_state
    from quail.backends.quail.graph import payload_tables, serve_region

    end = _quail_native.WorkerEnd(path)
    while (message := end.next()) is not None:
        try:
            if message[0] == "describe":
                end.described(json.dumps(description))
                continue
            if message[0] == "boot":
                end.booted(json.dumps({"boot_s": 0.5, "kind": "warm"}))
                continue
            region = json.loads(message[1])
            payload = {**region["query"], **payload_tables(message[2])}
            # the oracle answers about the region's documents
            model.documents = payload["documents"]
            model.texts = {alias: payload["texts"].get(alias)
                           for alias in payload["documents"]}
            model.source_rows = {alias: rows.to_pylist() for alias, rows
                                 in payload["source_rows"].items()}
            state = graph_state(model, payload["documents"])
            state["texts"] = payload["texts"]
            state["columns"] = payload["columns"]
            model.arena = state["gpu"].arena
            # as the GPU worker does, send each answer to a listener
            if region["listening"][1]:
                set_answer_sink(
                    lambda event: end.ask(json.dumps({"answer": [event]})))
            graph = graph_from_envelope(payload["physical_plan"])
            model.last_node = _last_model_node(model, graph)
            try:
                serve_region(state, payload, graph, end,
                             region["wanted"], region["hinted"])
            finally:
                set_answer_sink(None)
        except Exception:
            end.fail(traceback.format_exc())


class OracleWorkers:
    """Workers that answer from an oracle, in place of the GPUs' workers.

    A query run with ``physical_executor=OracleWorkers(...)`` runs as on
    GPUs: Rust sends each worker the region and its tables, and each
    worker is a thread that answers from the oracle. A plan on several
    GPUs gives each a copy of the oracle; the copies share the lists of
    what was asked.

    Args:
        oracle: The oracle answering the model operators.
        **report: Report fields the run carries, such as wall_s.
    """

    def __init__(self, oracle: Oracle, **report):
        self.oracle = oracle
        self.report = {"wall_s": 1.0, "boot_s": 0.5, "fresh_tokens": 1234,
                       **report}
        self._workers = None

    def start_workers(self, gpus: int) -> dict:
        """Start a worker per GPU; returns them with a boot record."""
        oracle = self.oracle
        models = [oracle] if gpus == 1 else [
            copy.copy(oracle) for _ in range(gpus)]
        self._workers = workers = _quail_native.Workers(gpus)
        for model, path in zip(models, workers.paths, strict=True):
            threading.Thread(target=_serve, args=(model, str(path)),
                             daemon=True).start()
        workers.connect()
        return {"workers": workers, "boot": {"boot_s": 0.5, "kind": "warm"}}

    def close(self) -> None:
        """Stop the workers."""
        workers, self._workers = self._workers, None
        if workers is not None:
            workers.close()

    def __call__(self, request):
        """Run a request's graph on the oracle, as a physical executor.

        A test that wraps the executor in a callable of its own runs a
        request this way.
        """
        from helpers.fakes import run_on_workers
        from quail.execution.types import PhysicalResponse
        from quail.physical import Scan
        from quail.specs import DEVICES

        scans = [node for node in graph_from_envelope(request.plan).nodes
                 if isinstance(node, Scan)]
        booted = self.start_workers(request.gpus)
        try:
            payload = {
                "physical_plan": request.plan, **request.plan["settings"],
                "documents": {node.alias: request.inputs[node.input_id].documents
                              for node in scans},
                "columns": request.column_tables(),
                "texts": {node.alias: request.inputs[node.input_id].texts
                          for node in scans
                          if request.inputs[node.input_id].texts is not None},
                "source_rows": {alias: pa.table({"row": rows})
                                for alias, rows in request.source_rows.items()}}
            result = run_on_workers(
                booted["workers"], payload, graph_from_envelope(request.plan),
                DEVICES[request.plan["device"]])
        finally:
            self.close()
        return PhysicalResponse(result.pop("_outputs"), {**result, **self.report})


def oracle_executor(oracle: Oracle, **report) -> OracleWorkers:
    """Return workers that answer each request from the oracle.

    Args:
        oracle: The oracle answering the model operators.
        **report: Report fields the run carries, such as wall_s.
    """
    return OracleWorkers(oracle, **report)


def finish(query, oracle: Oracle, **report):
    """Run a query to its result on the oracle."""
    from quail.api.execute import execute_query

    return execute_query(query, physical_executor=oracle_executor(oracle, **report))
