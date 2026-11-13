"""The planted truth of the end-to-end tests, and workers that answer from it.

An ``Oracle`` holds what the model would say: each filter's and join's
answer, each label, score, written answer, copied text, and feature
list. ``OracleWorkers`` stand for the GPUs' workers: each runs its
regions with the production executor on a fake model whose forward
pass answers from the oracle (helpers/planted_model.py). Nothing here
decides which rows an operator reads or returns.
"""

from __future__ import annotations

import copy
import json
import threading
import time
import traceback

import _quail_native
import numpy as np
import pyarrow as pa

from helpers.planted_model import PlantedModel, load
from quail.execution.finished import region_hint
from quail.physical import AiExtract, AiJoin, AiMap
from quail.physical.plan import graph_from_envelope
from quail.progress import set_answer_sink

# the most seconds an operator above a region may take to tighten the
# region's hint after receiving a row; a slower one fails the test
HINT_WAIT_SECONDS = 30.0


def question_text(token_ids) -> str:
    """Return the text of a question's token ids in the tests' tokenizer."""
    from helpers.sessions import encoder

    found = encoder().decode([int(token) for token in token_ids])
    return found.decode() if isinstance(found, bytes) else found


class Oracle:
    """The planted truth a fake model answers from, and what it was asked.

    Args:
        filters: alias -> {question key: a bit per document}, where the
            key is matched against the start of the question's tokens;
            or a callable (alias, document, written_position) -> bool.
        joins: {(anchor, *partners): rule(anchor_id, *partner_ids)}; or
            a callable (node, assignment) -> bool over alias -> id.
        labels: A callable (spec, ids) -> the label of one row, or a
            sequence indexed by the first id.
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
        join_metrics: The GPU seconds, peak GPU bytes, and extension
            counters each worker reports for a join, in place of what a
            CPU run measures.
        chunk_tokens: The forward pass size, in place of the plan's.
        tightens_after: Source rows of the first alias; after the
            region reports one of them, the worker waits until the
            region's hint tightens, so a test decides when the operator
            above the region tightens its test.
    """

    def __init__(self, filters=None, joins=None, labels=None, scores=None,
                 answers=None, features=None, maps=None, *, truth=None,
                 join_metrics=None, chunk_tokens=None, tightens_after=()):
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
        self.join_metrics = join_metrics
        self.chunk_tokens = chunk_tokens
        self.tightens_after = frozenset(tightens_after)
        self.texts = {}
        self.documents = {}
        # by alias, each document's row in the registered table, which the
        # planted truth uses, for an input that holds some of a table's rows
        self.source_rows = {}
        # (spec name, rows) of every score or classification a region ran
        self.calls = []
        self.asked = []       # (alias, document) of every filter answer
        # tokens of map and extraction answers written, and the answers
        # stopped before their end
        self.answer_tokens = 0
        self.stopped_early = 0

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

    def _filter_answer(self, node, document) -> bool:
        """The planted answer of a filter's question about a document."""
        self.asked.append((node.alias, int(document)))
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
        """The planted answer of a join about an anchor and its partners."""
        aliases = (node.anchor, *node.partners)
        ids = self._sources(aliases, (anchor_id, *partner_ids))
        if self.truth is not None:
            return bool(self.truth("join", question_text(node.frame_token_ids),
                                   self._texts(aliases, ids)))
        if callable(self.joins):
            return bool(self.joins(node, dict(zip(aliases, ids))))
        return bool(self.joins[aliases](*ids))

    def label(self, spec, ids):
        """The planted label of a row, or None."""
        ids = self._sources(spec.aliases, ids)
        if self.truth is not None:
            return self.truth("label", spec.query_template,
                              self._texts(spec.aliases, ids))
        if callable(self.labels):
            return self.labels(spec, ids)
        return None if self.labels is None else self.labels[ids[0]]

    def score(self, spec, ids) -> float:
        """The planted score of a row."""
        return float(self.scores(spec, self._sources(spec.aliases, ids)))

    def written(self, spec, document) -> str:
        """The planted answer a map writes about a document."""
        return self.maps(spec, self._source(spec.aliases[0], document))

    def extracted(self, spec, document):
        """The planted text an extraction copies from a document, or None."""
        document = self._source(spec.alias, document)
        if callable(self.answers):
            return self.answers(spec, document)
        return None if self.answers is None else self.answers[document]

    def _written(self, node, text, found) -> None:
        """Take what the executor wrote of a map's or extraction's planted answer.

        A region calls this once it has run, per answer in the order
        the model was asked; a test that reads the written answers
        replaces it.

        Args:
            node: The AiMap or AiExtract.
            text: The planted answer.
            found: The answer the region returned.
        """

    def feature_list(self, spec, document) -> list:
        """The planted features of a document."""
        document = self._source(spec.aliases[0], document)
        if callable(self.features):
            return self.features(spec, document)
        return [{"feature": document % 7, "strength": float(document + 1)}]


class OracleEnd:
    """A worker's connection that records what the region wrote and reported.

    Args:
        end: The ``_quail_native.WorkerEnd``.
        oracle: The Oracle.
    """

    def __init__(self, end, oracle):
        self.end = end
        self.oracle = oracle
        self.waited = set()     # the rows of tightens_after already waited for
        self.model = None       # the running region's PlantedModel

    def __getattr__(self, name):
        return getattr(self.end, name)

    def begin(self, model) -> None:
        """Start a region that runs on the model."""
        self.model, self.waited = model, set()

    def finished(self, table) -> None:
        """Send finished rows; after a row in tightens_after, wait for the hint.

        Raises:
            AssertionError: The hint did not tighten within
                HINT_WAIT_SECONDS.
        """
        oracle, hint = self.oracle, region_hint()
        metadata = table.schema.metadata
        rows = set()
        # a model node's rows port completes its rows; a join reports
        # its answers before it
        if (hint is not None and metadata[b"quail.port"].startswith(b"rows")
                and metadata[b"quail.node"].decode() in self.model.asking):
            alias = next(name for name in table.column_names
                         if name in oracle.documents)
            rows = {oracle._source(alias, document)
                    for document in table.column(alias).to_pylist()}
        rows = (rows & oracle.tightens_after) - self.waited
        self.waited |= rows
        before = hint.generation() if rows else None
        self.end.finished(table)
        deadline = time.monotonic() + HINT_WAIT_SECONDS
        while rows and hint.generation() == before:
            if time.monotonic() > deadline:
                raise AssertionError(f"the hint did not tighten after {sorted(rows)}")
            time.sleep(0.001)

    def ran(self, report, tables) -> None:
        """Send what the region ran, with the oracle's join metrics.

        The oracle counts the answer tokens the region's maps and
        extractions wrote, and takes each written answer.
        """
        oracle, report = self.oracle, json.loads(report)
        planted = oracle.join_metrics
        for node_id, metrics in report["metrics"].items():
            extension = metrics.get("extension") or {}
            oracle.answer_tokens += extension.get("answer_tokens", 0)
            oracle.stopped_early += extension.get("answers_stopped_early", 0)
            if planted is not None and isinstance(
                    self.model.asking.get(node_id), AiJoin):
                metrics.update(
                    gpu_s=planted.gpu_s, peak_gpu_bytes=planted.peak_gpu_bytes,
                    extension={**extension, **planted.extension})
        written = {}
        for table in tables:
            node = self.model.asking.get(table.schema.metadata[b"quail.node"].decode())
            if isinstance(node, (AiMap, AiExtract)):
                spec = node.spec
                written[node.node_id] = dict(zip(
                    table.column(spec.aliases[0]).to_pylist(),
                    table.column(spec.name).to_pylist()))
        for node, document, text in self.model.writes:
            oracle._written(node, text, written.get(node.node_id, {}).get(document))
        self.end.ran(json.dumps(report), tables)


def _serve(oracle, path, description=None, gpu_index=0, gpus=1):
    """Serve each region as a GPU worker does, on a model that answers from truth.

    Args:
        oracle: The oracle this worker's model answers from.
        path: The socket Quail listens at for this worker.
        description: What the worker answers a describe with, as
            ``engine_description`` returns it.
        gpu_index: Which of the run's GPUs this worker stands for.
        gpus: How many GPUs the run has.
    """
    from helpers.fakes import fake_readout
    from quail.backends.quail.graph import payload_tables, serve_region
    from quail.builtins import built_in_registry

    registry = built_in_registry()
    end = OracleEnd(_quail_native.WorkerEnd(path), oracle)
    gpu = None
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
            plan = payload["physical_plan"]
            # the oracle answers about the region's documents
            oracle.documents = payload["documents"]
            oracle.texts = {alias: payload["texts"].get(alias)
                            for alias in payload["documents"]}
            oracle.source_rows = {alias: rows.to_pylist() for alias, rows
                                  in payload["source_rows"].items()}
            graph = graph_from_envelope(plan)
            device = registry.device(plan["device"])
            if gpu is None:
                gpu = load(registry.model(plan["model"]), device, gpu_index, gpus)
            model = PlantedModel(oracle, graph, gpu, payload)
            end.begin(model)
            gpu.pipeline.forward_chunk = model.forward_chunk
            chunk_tokens = oracle.chunk_tokens or payload["chunk_tokens"]
            gpu.execution.bind_query(
                torch=gpu.torch, answer_rows=object(), chunk_tokens=chunk_tokens,
                async_answers=fake_readout(
                    submit=lambda rows: (rows[:, model.yes] > rows[:, model.no]
                                         ).tolist()),
                async_scores=fake_readout(
                    dtype=np.float32,
                    submit=lambda rows: rows[:, model.yes].numpy()))
            state = {
                "gpu": gpu, "documents": payload["documents"],
                "texts": payload["texts"], "columns": payload["columns"],
                "runtimes": registry.runtimes, "device": device,
                "chunk_tokens": chunk_tokens,
                "gpu_timing": payload.get("gpu_timing", False)}
            # as the GPU worker does, send each answer to a listener
            if region["listening"][1]:
                set_answer_sink(
                    lambda event: end.ask(json.dumps({"answer": [event]})))
            try:
                serve_region(state, payload, graph, end,
                             region["wanted"], region["hinted"])
            finally:
                set_answer_sink(None)
        except Exception:
            end.fail(traceback.format_exc())
    if gpu is not None:
        gpu.idle.append(gpu)


class OracleWorkers:
    """Workers that answer from an oracle, in place of the GPUs' workers.

    A query run with ``physical_executor=OracleWorkers(...)`` runs as on
    GPUs: Rust sends each worker the region and its tables, and each
    worker is a thread that runs the region on a model answering from
    the oracle. A plan on several GPUs gives each a copy of the oracle;
    the copies share the lists of what was asked.

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
        for index, (model, path) in enumerate(
                zip(models, workers.paths, strict=True)):
            threading.Thread(target=_serve, args=(model, str(path), None, index, gpus),
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
