"""Quail model backend."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pyarrow as pa

from quail.backends.base import GpuContext
from quail.backends.quail.executor import loop
from quail.backends.quail.executor.classify import ClassifyStages, PairPartners
from quail.backends.quail.executor.models import supported_archs
from quail.backends.quail.executor.score import QuailScorer
from quail.backends.quail.executor.stages import Stage, filter_stages, run_stages
from quail.backends.quail.graph import (
    filter_document_sink,
    filter_result,
    stage_partner_lists,
)
from quail.backends.quail.worker import execute_quail_request, prepare_quail_request
from quail.execution.reranker import (
    RerankerModelExecution,
    _score_table,
    attach_prior_columns,
    classify_outputs,
    filter_scores,
    scored_batch,
)
from quail.execution.runner import (
    NodeMetrics,
    NodeResult,
    foreign_call,
    foreign_outputs,
)
from quail.execution.tokens import DocumentKeys, DocumentPrefixes, prefix_tree
from quail.logical import Alias, LabelIn, is_score, shared_preamble
from quail.logical.prompts import true_false_token_ids
from quail.physical import (
    AiClassify,
    AiFilter,
    AiJoin,
    AiScore,
    Barrier,
    Foreign,
    LabelFilter,
    PhysicalNode,
    ValueType,
)
from quail.planner import plan_quail
from quail.planner.classify import has_label, plan_classify
from quail.planner.physical_optimizer import (
    ModelRegion,
    PhysicalCandidate,
    PlanningContext,
    SupportResult,
)
from quail.planner.plan import Refusal
from quail.planner.reranker import plan_reranker
from quail.progress import answer_sink, logger


class QuailModelExecution:
    """Quail model state shared by model nodes on one GPU executor."""

    def __init__(self, context: GpuContext):
        self.context = context
        self._state: dict[str, Any] = {}
        self._reranker: RerankerModelExecution | None = None

    @property
    def state(self) -> Mapping[str, Any]:
        """Return Quail's private loaded model state."""
        return self._state

    def bind_loaded_model(self, *, model, arena, pipeline) -> None:
        """Attach the loaded model objects owned by this executor."""
        self._state.update(model=model, arena=arena, pipeline=pipeline)
        spec = getattr(self.context, "model", None)
        if spec is not None:
            self._state.update(model_name=spec.name, model_spec=spec)

    def close(self) -> None:
        """Drop every reference to the loaded model so its memory can go."""
        self._state.clear()
        self._reranker = None

    def bind_query(self, *, torch, async_answers, answer_rows,
                   chunk_tokens: int) -> None:
        """Attach state that is valid for the current query.

        answer_rows are the retained output rows every readout scores
        against; async_answers is the TRUE/FALSE readout over them.
        """
        self._state.update(
            torch=torch,
            async_answers=async_answers,
            answer_rows=answer_rows,
            chunk_tokens=chunk_tokens,
        )

    def execute(
        self,
        node: PhysicalNode,
        inputs: Mapping[str, Any],
    ) -> Any:
        if isinstance(node, AiScore):
            execution = self._reranker_execution(inputs["documents"])
            # the scorer reads the query's timing choice from the state
            self._state["gpu_timing"] = bool(inputs.get("gpu_timing", False))
            if "score_rows" in inputs:
                return execution.execute_rows(node, inputs["score_rows"])
            return execution.execute(node, inputs["score_inputs"])
        if not isinstance(node, (AiFilter, AiJoin)):
            raise TypeError(
                f"Quail cannot execute physical node {node.type_name!r}")
        return self._execute_quail_node(node, inputs)

    def _reranker_execution(self, documents) -> RerankerModelExecution:
        """Return the reranker bound to this query's document tokens."""
        execution = self._reranker
        if execution is None or execution.documents is not documents:
            execution = RerankerModelExecution(GpuContext(
                gpu_index=self.context.gpu_index,
                gpu_count=self.context.gpu_count,
                model=self.context.model, device=self.context.device,
                query_settings={
                    "documents": documents,
                    "reranker": QuailScorer(self._state),
                },
            ))
            self._reranker = execution
        return execution

    def execute_pipeline(self, pipeline, inputs, context) -> dict:
        """Run one table's chain of per-document operators as one run.

        The members' stages are concatenated and driven by one stage
        scheduler over the documents the first member was given, so a
        document goes through every operator with its KV resident. A
        label filter or per-batch apply between two stages gates the
        next stage; one after the last stage runs over the documents
        that came out of it.

        Args:
            pipeline: The Pipeline (quail.execution.pipelines).
            inputs: Member node id -> its inputs from outside the
                pipeline, by port name.
            context: The graph's execution context.

        Returns:
            Node id -> NodeResult for every member.
        """
        state = self._state
        state["gpu_timing"] = bool(context.state.get("gpu_timing", False))
        members = list(pipeline.members)
        alias = pipeline.alias
        first = members[0]
        # the first member's documents: its ids input, or its one input
        anchor_port = next((port.name for port in first.inputs
                            if port.source.port == f"ids:{alias}"), None)
        source = (inputs[first.node_id][anchor_port] if anchor_port is not None
                  else next(iter(inputs[first.node_id].values())))
        ids = list(source)
        gpu_inputs = {"gpu_timing": context.state.get("gpu_timing", False)}
        # the members with stages, then the gates after the last of them
        last_staged = max(index for index, member in enumerate(members)
                          if isinstance(member, (AiFilter, AiClassify, AiJoin)))
        staged, trailing = members[:last_staged + 1], members[last_staged + 1:]
        joins = [index for index, member in enumerate(staged)
                 if isinstance(member, AiJoin)]
        parts = []
        for member in staged[:joins[0] if joins else -1]:
            parts.append(self._part(member, ids, parts, inputs, context))
        prefixes = DocumentPrefixes(context.state["pre"],
                                    context.state["docs"][alias], ids)
        sink = staged[-1]
        results = {}
        if joins:
            # the join settles each anchor's KV; a classification of
            # the pairs it keeps runs as stages after the join's
            join = staged[joins[0]]
            chain = {"documents": prefixes, "document_ids": ids,
                     "parts": parts, "after": [],
                     "pairs": {part.node.written_pos: part.rows
                               for part in parts
                               if isinstance(part, _ApplyGate)
                               and part.node.ids == "pairs"}}
            join_inputs = context.model_inputs(
                join, inputs[join.node_id], context, chain)
            for member in staged[joins[0] + 1:]:
                chain["after"].append(self._pair_part(
                    member, join, ids, join_inputs, context))
            results[join.node_id] = self._execute_quail_node(join, join_inputs)
            context.model_result(join, results[join.node_id], context)
            for part in chain["after"]:
                results[part.node.node_id] = part.result_value
        else:
            own = self._part(sink, ids, parts, inputs, context)
            stages = _gated_stages(parts + [own])
            stats = {}
            every, spans, tokens = run_stages(
                state["torch"], state["arena"], state["pipeline"], stages,
                prefixes, state["chunk_tokens"],
                anchor_keys=DocumentKeys(alias, ids),
                attention_mode=getattr(first, "attention", None) or None,
                prefix_tree=_prefix_tree(first, prefixes, state["arena"]),
                stats=stats, staging=_staging(state),
                on_chunk=lambda transitions: _report_chain_transitions(
                    parts + [own], transitions),
                label=f"pipeline {alias} ({len(stages)} stages)",
                conditioning=_chain_conditioning(parts + [own]))
            _complete_chain(parts + [own], every, spans, tokens, stats,
                            state["torch"], gpu_inputs)
        for part in parts:
            results[part.node.node_id] = part.result_value
        if not joins:
            results[sink.node_id] = own.result_value
        # gates after the last stage read the documents that came out
        for member in trailing:
            member_inputs = dict(inputs[member.node_id])
            for port in member.inputs:
                if port.source.node_id in results:
                    member_inputs[port.name] = results[
                        port.source.node_id].outputs[port.source.port]
            results[member.node_id] = context.runtimes[
                member.runtime_key].execute(member, member_inputs, context)
        return results

    def _part(self, node, ids, parts, inputs, context):
        """The pipeline part for one member, after the parts before it."""
        if isinstance(node, AiFilter):
            return _FilterPart(node, ids, self._state["async_answers"])
        if isinstance(node, AiClassify):
            position = {document: index for index, document in enumerate(ids)}
            plan = ClassifyStages(self._state, node.spec, len(ids),
                                  lambda key: position[key[1]])
            plan.seeds = ids
            by_node = {part.node.node_id: part for part in parts}
            priors = [(by_node[port.source.node_id], port.source.port)
                      for port in node.inputs
                      if port.value_type is ValueType.SCORES
                      and port.source.node_id in by_node]
            return _ClassifyPart(node, ids, plan, priors)
        if isinstance(node, LabelFilter):
            labeled = next(part for part in reversed(parts)
                           if isinstance(part, _ClassifyPart)
                           and part.node.spec.name == node.score_name)
            return _LabelGate(node, labeled)
        if isinstance(node, Foreign):
            values = {}
            for port in node.inputs:
                if not port.source.port.startswith("ids:"):
                    continue
                port_alias = port.source.port.split(":", 1)[1]
                values[port_alias] = (
                    ids if port.source.node_id in {
                        part.node.node_id for part in parts}
                    else inputs[node.node_id][port.name])
            call, metrics = foreign_call(node, values, context)
            return _ApplyGate(node, call, metrics, values)
        raise TypeError(f"{node.type_name!r} is not a pipeline operator")

    def _pair_part(self, node, join, ids, join_inputs, context):
        """The part classifying the pairs the join keeps, on its anchors."""
        if not isinstance(node, AiClassify) or node.spec.partner is None:
            raise TypeError(
                f"{node.type_name!r} cannot follow a join in its pipeline")
        spec = node.spec
        stage = next(stage for stage in join.stages
                     if set(spec.aliases) == {stage.anchor, *stage.partners})
        # the join's partner tuples, one partner alias each
        partner_ids = [int(entry[0]) if isinstance(entry, (tuple, list))
                       else int(entry)
                       for entry in join_inputs["partner_indices"][
                           stage.written_pos]]
        documents = context.state["docs"][spec.partner]
        kept = _KeptPairs(join.stages.index(stage))
        partners = PairPartners(
            ids=partner_ids,
            documents=[documents[partner] for partner in partner_ids],
            kept=kept.of)
        position = {document: index for index, document in enumerate(ids)}
        plan = ClassifyStages(self._state, spec, len(ids),
                              lambda key: position[key[1]], partners=partners)
        return _ClassifyPart(node, ids, plan, kept=kept)

    def _execute_quail_node(
        self,
        node: AiFilter | AiJoin,
        inputs: Mapping[str, Any],
    ) -> Any:

        missing = {
            "torch", "async_answers", "chunk_tokens",
            "arena", "pipeline",
        } - set(self._state)
        if missing:
            raise RuntimeError(
                f"Quail model execution is missing state {sorted(missing)}")
        torch = self._state["torch"]
        arena = self._state["arena"]
        pipeline = self._state["pipeline"]
        async_answers = self._state["async_answers"]
        chunk_tokens = self._state["chunk_tokens"]

        if isinstance(node, AiFilter):
            document_ids = inputs["document_ids"]
            retain_survivors = inputs.get("retain_survivors", ())
            if retain_survivors is False:
                retain_survivors = ()
            # sorted admission would change which rows a limit keeps
            tree = (None if inputs.get("limit") is not None
                    else _prefix_tree(node, inputs["documents"], arena))
            stats = {}
            answers, spans, tokens = loop.run_filter(
                torch,
                arena,
                pipeline,
                async_answers,
                inputs["documents"],
                [list(question) for question in node.question_token_ids],
                chunk_tokens,
                limit=inputs.get("limit"),
                arena_writes=node.arena_writes,
                arena_keys=DocumentKeys(node.alias, document_ids),
                retain_survivors=retain_survivors,
                document_done=inputs.get("document_done"),
                prefix_tree=tree,
                attention_mode=node.attention or None,
                stats=stats, staging=_staging(self._state),
            )
            return filter_result(
                node, answers, tokens, document_ids,
                gpu_s=_gpu_seconds(torch, spans, inputs),
                chunks=_chunks(spans, inputs),
                borrowed_tokens=stats.get("borrowed_tokens", 0),
                pack_s=stats.get("pack_s", 0.0))

        stage_frames = inputs["stage_frames"]
        chain = inputs.get("chain")
        lists_for = inputs.get("anchor_partners")
        join_stages = [
            Stage(suffixes=suffixes, readout=async_answers, frame=frame,
                  requests=(None if lists_for is None
                            else (lambda key, j=j: lists_for(key[1])[j])),
                  label=f"join stage {j}")
            for j, (suffixes, frame) in enumerate(
                zip(inputs["stage_suffixes"], stage_frames))
        ]
        join_stats = {}
        leading = 0
        parts = []
        after = []
        if chain is not None:
            # the chain's stages lead the join's: a survivor goes on to
            # the join with its KV resident, past the chain's gates;
            # a classification of the pairs kept follows the join's
            filter_ids = chain["document_ids"]
            parts = chain["parts"]
            after = chain.get("after", [])
            join_part = _StagesPart(join_stages)
            stages = _gated_stages(parts + [join_part] + after)
            leading = len(stages) - len(join_stages) - sum(
                len(part.stages) for part in after)
            root = parts[0].node if parts else node
            prefixes = chain["documents"]
            anchor_keys = DocumentKeys(node.anchor, filter_ids)
            tree = _prefix_tree(root, prefixes, arena)
            attention = (node.attention or getattr(root, "attention", None)
                         or None)
        else:
            stages = join_stages
            prefixes = inputs["prefixes"]
            anchor_keys = inputs["anchor_keys"]
            tree = _prefix_tree(node, inputs["prefixes"], arena)
            attention = node.attention or None
        anchor_done = inputs["anchor_done"]
        settled = set()
        join_rows = {}
        if after:
            # the join's last stage records each anchor's kept partners
            # for the parts after it, and its row settles the anchor
            last = len(join_stages) - 1

            def kept(a, row, stage=join_stages[last]):
                indices = (None if lists_for is None
                           else lists_for(anchor_keys[a][1])[last])
                join_rows[a] = row
                for part in after:
                    part.kept.record(a, row, indices)
                return bool(any(row))

            join_stages[last].decide = kept

        def on_settled(anchor, survived, row):
            settled.add(anchor)
            anchor_done(anchor, join_rows.get(anchor, row) if after else row)

        def on_chunk(transitions):
            _report_chain_transitions(parts, transitions)

        every, spans, tokens = run_stages(
            torch, arena, pipeline, stages, prefixes, chunk_tokens,
            anchor_keys=anchor_keys, on_settled=on_settled,
            attention_mode=attention, prefix_tree=tree, stats=join_stats,
            on_chunk=on_chunk, label=f"join ({len(join_stages)} stages)",
            staging=_staging(self._state),
            conditioning=_chain_conditioning(parts))
        answers = every[leading:leading + len(join_stages)]
        after_tokens = 0
        if after:
            low = leading + len(join_stages)
            for part in after:
                high = low + len(part.stages)
                own = part.finish(every[low:high]) or 0
                part_spans = [span for span in spans if low <= span[0] < high]
                part.result_value = part.result(
                    own, _gpu_seconds(torch, part_spans, inputs),
                    _chunks(part_spans, inputs), {})
                after_tokens += own
                low = high
            spans = [span for span in spans
                     if span[0] < leading + len(join_stages)]
            tokens -= after_tokens
        if chain is not None:
            # the join packs frames and partner suffixes; the rest is the chain's
            join_tokens = _streamed_tokens(join_stages, answers, lists_for,
                                           anchor_keys)
            # the join's anchors are the documents that reached it, in
            # the chain's order: those with a first-stage row and those
            # that settled there with no partner to ask; their rows are
            # re-keyed to that order
            reached = sorted(set(answers[0] if answers else ()) | settled)
            local_of = {position: local for local, position in enumerate(reached)}
            answers = [{local_of[position]: row for position, row in stage.items()}
                       for stage in answers]
            anchor_ids = [filter_ids[position] for position in reached]
            if parts:
                kv_round = {"hits": len(reached), "misses": 0}
                _complete_chain(parts, every[:leading], spans,
                                tokens - join_tokens, join_stats, torch, inputs)
                for part in parts:
                    if part.result_value is None:
                        raise RuntimeError(
                            "a pipeline part finished without a result")
                join_stats = {}
                spans = [span for span in spans if span[0] >= leading]
                tokens = join_tokens
            else:
                # the join led the pipeline and packed the anchors itself
                kv_round = {"hits": 0, "misses": len(reached)}
        else:
            anchor_ids = list(inputs["anchor_ids"])
            kv_round = inputs.get("kv_round") or {}
        group = inputs["group"]
        last = answers[-1] if answers else {}
        matched = {
            anchor_ids[int(local)]
            for local, row in last.items() if any(row)
        }
        if group[-1]["semantics"] == "anti":
            survivors = [document for document in anchor_ids
                         if document not in matched]
        else:
            survivors = [document for document in anchor_ids
                         if document in matched]
        outputs = {f"ids:{node.anchor}": survivors}
        partner_lists = stage_partner_lists(group, lists_for, anchor_ids)
        for stage, stage_answers, members in zip(
                node.stages, answers, partner_lists):
            outputs[f"join_answers:{stage.written_pos}"] = {
                "rows": stage_answers,
                "anchor_index": anchor_ids,
                "partner_index": inputs["partner_indices"][
                    stage.written_pos
                ],
                "anchor_partners": members,
                "anchor": node.anchor,
                "partners": list(stage.partners),
                "semantics": stage.semantics,
                "selectivity": stage.selectivity,
                "written_pos": stage.written_pos,
            }
        return NodeResult(
            outputs=outputs,
            metrics=NodeMetrics(
                input_rows=len(anchor_ids),
                output_rows=len(survivors),
                evaluated_document_pairs=sum(
                    sum(len(row) for row in stage.values())
                    for stage in answers
                ),
                kv_hits=kv_round.get("hits", 0),
                kv_misses=kv_round.get("misses", 0),
                fresh_tokens=tokens,
                gpu_s=_gpu_seconds(torch, spans, inputs),
                chunks=_chunks(spans, inputs),
                extension={
                    "answers": answers,
                    **({"borrowed_prefix_tokens": join_stats["borrowed_tokens"]}
                       if join_stats.get("borrowed_tokens") else {}),
                    **({"pack_s": round(join_stats["pack_s"], 3)}
                       if join_stats.get("pack_s") else {}),
                },
            ),
        )


def _staging(state):
    """The node's reusable input transfer buffers."""
    if "input_staging" not in state:
        state["input_staging"] = loop.InputStaging(state["torch"])
    state["input_staging"].fixed_tokens.clear()
    return state["input_staging"]


class _StagesPart:
    """Stages the sink of a pipeline adds after the parts before it."""

    node = None
    gate = None
    document_done = None
    result_value = None

    def __init__(self, stages):
        self.stages = stages

    def finish(self, every):
        return None

    def result(self, tokens, gpu_s, chunks, stats):
        return None


class _FilterPart:
    """A filter chain's stages inside a pipeline."""

    gate = None
    result_value = None

    def __init__(self, node, ids, async_answers):
        self.node = node
        self.ids = ids
        self.stages = filter_stages(
            [list(question) for question in node.question_token_ids],
            async_answers)
        self.document_done = filter_document_sink(node, ids)
        self.answers = {}

    def finish(self, every) -> int | None:
        """Record the chain's answers; its tokens are the run's remainder."""
        for stage in every:
            for document, row in stage.items():
                self.answers.setdefault(document, []).append(int(row[0]))
        return None

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        return filter_result(
            self.node, self.answers, tokens, self.ids, gpu_s=gpu_s,
            chunks=chunks, borrowed_tokens=stats.get("borrowed_tokens", 0),
            pack_s=stats.get("pack_s", 0.0))


class _ClassifyPart:
    """A classification's stages inside a pipeline."""

    gate = None
    document_done = None
    result_value = None

    def __init__(self, node, ids, plan: ClassifyStages, priors=(), kept=None):
        self.node = node
        self.ids = ids
        self.plan = plan
        self.priors = priors      # (part, port) of each label table read
        self.kept = kept          # a pair classification's _KeptPairs
        self.stages = plan.stages
        self.reached = 0
        self.label_tokens = 0

    def label(self, document) -> str | None:
        """The document's label so far, by its id."""
        return self.plan.labels[0][self.ids.index(document)] \
            if not hasattr(self, "_position") else \
            self.plan.labels[0][self._position[document]]

    def finish(self, every) -> int:
        """Label the documents; returns the frame and suffix tokens packed."""
        self.reached = len(every[0]) if every else 0
        self.label_tokens, streamed = self.plan.finish(every)
        return streamed

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        spec = self.node.spec
        plan = self.plan
        if plan.partners is not None:
            pairs = sorted(plan.pair_labels)
            rows = np.asarray(
                [[self.ids[anchor], plan.partners.ids[partner]]
                 for anchor, partner in pairs], dtype=np.int32).reshape(-1, 2)
            table = _score_table(rows, spec.aliases, spec.name,
                                 [plan.pair_labels[pair] for pair in pairs],
                                 pa.string())
            sink = answer_sink()
            if sink is not None and len(rows):
                sink(scored_batch(self.node, rows, table))
            return NodeResult(classify_outputs(self.node, table), NodeMetrics(
                input_rows=len(pairs), output_rows=len(pairs),
                evaluated_document_pairs=len(pairs), fresh_tokens=tokens,
                gpu_s=gpu_s, chunks=chunks,
                extension={"output": spec.name, "aliases": list(spec.aliases),
                           "input_rows": len(pairs),
                           "label_tokens": self.label_tokens}))
        # a document whose answer named no label has no row
        labeled = [index for index, label in enumerate(plan.labels[0])
                   if label is not None]
        rows = np.asarray([[self.ids[index]] for index in labeled],
                          dtype=np.int32).reshape(-1, 1)
        table = _score_table(rows, spec.aliases, spec.name,
                             [plan.labels[0][index] for index in labeled],
                             pa.string())
        for name, values in plan.later().items():
            table = table.append_column(
                name, pa.array([values[index] for index in labeled],
                               pa.string()))
        # the labels the parts before it gave these documents, as a
        # classify node on its own carries them from its scores input
        table = attach_prior_columns(table, {
            spec.aliases[0]: part.result_value.outputs[port]
            for part, port in self.priors})
        sink = answer_sink()
        if sink is not None and len(rows):
            sink(scored_batch(self.node, rows, table))
        return NodeResult(classify_outputs(self.node, table), NodeMetrics(
            input_rows=self.reached, output_rows=len(labeled),
            evaluated_documents=self.reached, fresh_tokens=tokens,
            gpu_s=gpu_s, chunks=chunks,
            extension={
                "output": spec.name, "aliases": list(spec.aliases),
                "input_rows": self.reached,
                "label_tokens": self.label_tokens,
                "borrowed_prefix_tokens": stats.get("borrowed_tokens", 0),
                "pack_s": round(stats.get("pack_s", 0.0), 3),
            }))


class _KeptPairs:
    """The partners a join's last stage kept for each anchor."""

    def __init__(self, stage: int):
        self.stage = stage
        self.partners = {}

    def record(self, anchor, row, indices) -> None:
        """Record the anchor's kept partners from its row of answers."""
        self.partners[anchor] = [
            position if indices is None else indices[position]
            for position, bit in enumerate(row) if bit]

    def of(self, anchor) -> list:
        return self.partners.get(anchor, [])


class _LabelGate:
    """A label filter between two stages: it gates on the label read."""

    stages = ()
    document_done = None
    result_value = None

    def __init__(self, node, labeled: _ClassifyPart):
        self.node = node
        self.labeled = labeled
        self.accepted = set(node.accepted)
        self.position = {document: index
                         for index, document in enumerate(labeled.ids)}

    def gate(self, key):
        label = self.labeled.plan.labels[0][self.position[key[1]]]
        return None if label in self.accepted else Stage.DROP

    def finish(self, every):
        return None

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        return filter_scores(self.node,
                             self.labeled.result_value.outputs["scores"])


class _ApplyGate:
    """A per-batch apply between two stages, called on each document."""

    stages = ()
    document_done = None
    result_value = None

    def __init__(self, node, call, metrics, values):
        self.node = node
        self.call = call
        self.metrics = metrics
        self.values = values
        self.produced = []
        self.rows = {}          # anchor -> partners, for pairs

    def gate(self, key):
        document = key[1]
        node = self.node
        if node.ids == "pairs":
            partner = next(alias for alias in node.aliases
                           if alias != key[0])
            partners = list(self.values[partner])
            pairs = self.call({key[0]: [document], partner: partners})
            self.produced.extend(pairs)
            mine = [b if a == document else a for a, b in pairs]
            if not mine:
                return Stage.DROP
            self.rows[document] = mine
            return None
        kept = self.call({key[0]: [document]})
        self.produced.extend(kept)
        return None if kept else Stage.DROP

    def finish(self, every):
        return None

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        return NodeResult(foreign_outputs(self.node, self.produced),
                          self.metrics())


def _chain_conditioning(parts):
    """The ConditioningRows a chain's canvas classification denoises with."""
    return next((part.plan.conditioning for part in parts
                 if isinstance(part, _ClassifyPart)
                 and part.plan.conditioning is not None), None)


def _gated_stages(parts) -> list:
    """The parts' stages in order; each gate guards the next stage."""
    stages = []
    pending = []
    for part in parts:
        if not part.stages:
            pending.append(part.gate)
            continue
        first = part.stages[0]
        if pending:
            gates = list(pending)
            asked = first.requests

            def guarded(key, gates=gates, asked=asked):
                for gate in gates:
                    if gate(key) is Stage.DROP:
                        return Stage.DROP
                return None if asked is None else asked(key)

            first.requests = guarded
            pending = []
        stages.extend(part.stages)
    if pending:
        raise ValueError("a pipeline ends with a gate and no stage after it")
    return stages


def _part_slices(parts) -> list:
    offset = 0
    slices = []
    for part in parts:
        slices.append((offset, offset + len(part.stages)))
        offset += len(part.stages)
    return slices


def _report_chain_transitions(parts, transitions) -> None:
    """Hand each filter part the documents that settled at its stages."""
    for part, (low, high) in zip(parts, _part_slices(parts)):
        if part.document_done is None:
            continue
        finished = [(anchor, stage - low, passed)
                    for anchor, stage, passed in transitions
                    if low <= stage < high
                    and (not passed or stage == high - 1)]
        if finished:
            part.document_done(finished)


def _complete_chain(parts, every, spans, tokens, stats, torch, inputs):
    """Finish every part of a driven chain and store its result.

    tokens are the run's fresh tokens less the consumer's own. The
    first part with stages, which packed the documents, takes them
    less the frames and suffixes of the parts after it; prefix
    borrowing and packing time go to it as well.
    """
    slices = _part_slices(parts)
    own = [part.finish(every[low:high])
           for part, (low, high) in zip(parts, slices)]
    staged = [index for index, part in enumerate(parts) if part.stages]
    remainder = tokens - sum(own[index] or 0 for index in staged[1:])
    for index, (part, (low, high)) in enumerate(zip(parts, slices)):
        part_spans = [span for span in spans if low <= span[0] < high]
        part.result_value = part.result(
            remainder if staged and index == staged[0] else (own[index] or 0),
            _gpu_seconds(torch, part_spans, inputs),
            _chunks(part_spans, inputs), stats if staged and index == staged[0]
            else {})


def _streamed_tokens(join_stages, answers, lists_for, anchor_keys) -> int:
    """Tokens the join's stages packed: frames and partner suffixes."""
    total = 0
    previous = None
    for j, (stage, rows) in enumerate(zip(join_stages, answers)):
        frame = list(stage.frame)
        writes = bool(frame) and frame != previous
        previous = frame
        lengths = [len(suffix) for suffix in stage.suffixes]
        for anchor in rows:
            indices = (None if lists_for is None
                       else lists_for(anchor_keys[anchor][1])[j])
            total += (len(frame) if writes else 0) + (
                sum(lengths) if indices is None
                else sum(lengths[i] for i in indices))
    return total


def _prefix_tree(node, documents, arena):
    """The node's prefix tree, or None when the plan did not ask for one."""
    shares = (node.spec.share_prefixes if isinstance(node, AiClassify)
              else node.share_prefixes)
    if not shares:
        return None
    started = time.perf_counter()
    tree = prefix_tree(documents, arena.page_tokens)
    logger.info(
        "prefix sharing on %s: %s documents borrow %s tokens "
        "(tree built in %.2f s)",
        node.spec.aliases[0] if isinstance(node, AiClassify)
        else getattr(node, "alias", None) or node.anchor,
        len(documents), tree.shared_tokens, time.perf_counter() - started)
    return tree


def _gpu_seconds(torch, spans, inputs) -> float:
    """Seconds the loop's forward chunks ran on the GPU; 0.0 unless asked."""
    if not inputs.get("gpu_timing"):
        return 0.0
    # every chunk's answers were read, so its end event has completed
    torch.cuda.synchronize()
    return sum(start.elapsed_time(end) for _, start, end in spans) / 1000.0


def _chunks(spans, inputs) -> int:
    """Forward chunks the loop launched; 0 unless timing was asked for."""
    return len(spans) if inputs.get("gpu_timing") else 0


def expected_join_nodes(plan) -> tuple[PhysicalNode, ...]:
    """Return the join nodes selected from Quail's planning estimates."""
    return tuple(
        node for node in plan.nodes if isinstance(node, (AiJoin, Barrier))
    )


def expected_join_stages(plan) -> tuple:
    """Return Quail's expected join stages in execution order."""
    return tuple(
        stage
        for node in expected_join_nodes(plan)
        if isinstance(node, AiJoin)
        for stage in node.stages
    )


class QuailBackend:
    """Plan and start Quail model execution."""

    name = "quail"
    runtime_package = "vllm==0.26.0"

    def supports(self, model, device, gpu_count: int) -> SupportResult:
        if device.name not in {"h100-sxm", "rtx-pro-6000-blackwell-server"}:
            return SupportResult.reject(
                f"Quail does not support device {device.name!r}")
        if model.arch not in supported_archs():
            return SupportResult.reject(
                f"Quail does not support model {model.name!r}: "
                f"no forward pass for architecture {model.arch!r}")
        if gpu_count not in {1, 2, 4, 8}:
            return SupportResult.reject(
                "Quail requires 1, 2, 4, or 8 GPUs")
        return SupportResult.accept()

    def plan(
        self,
        region: ModelRegion,
        context: PlanningContext,
    ) -> tuple[PhysicalCandidate, ...]:

        operators = region.logical_plan.operators()
        if has_label(region.logical_plan):
            if context.model.role == "reranker":
                return (PhysicalCandidate(None, Refusal(
                    reasons=("a reranker model cannot run AI.CLASSIFY",),
                    constraint="reranker_only_scores", needed=1, available=0,
                    unit="AI.CLASSIFY expressions"), float("inf")),)
            asked = any(not isinstance(predicate.expression, LabelIn)
                        for predicates in operators.filters.values()
                        for predicate in predicates)
            # a classification with nothing else runs on its own planner,
            # which chains classifications; beside AI.IF or joins it is
            # a step of the general plan
            if not asked and not operators.joins \
                    and len(operators.scans) == 1:
                return plan_classify(region, context, backend_name=self.name)
        has_score = any(
            is_score(predicate.expression)
            for predicates in operators.filters.values()
            for predicate in predicates
        ) or any(is_score(join.predicate) for join in operators.joins) or any(
            isinstance(expression, Alias) and expression.expression.kind == "score"
            for expression in region.logical_plan.root.columns
        )
        if context.model.role == "reranker":
            if not has_score:
                refusal = Refusal(
                    reasons=("a reranker model can only be used with AI.SCORE",),
                    constraint="reranker_only_scores",
                    needed=1,
                    available=0,
                    unit="AI.SCORE expressions",
                )
                return (PhysicalCandidate(None, refusal, float("inf")),)
        if has_score:
            return plan_reranker(region, context, backend_name=self.name)

        plan = plan_quail(
            region.logical_plan,
            model=context.model,
            device=context.device,
            doc_tokens=context.document_tokens,
            gpus=context.gpu_count,
            order=context.order,
            pair_fractions=context.pair_fractions,
            context=context,
        )
        if not hasattr(plan, "graph"):
            return (
                PhysicalCandidate(
                    graph=None,
                    plan=plan,
                    estimated_seconds=float("inf"),
                ),
            )
        plan = self._bind_runtime_data(plan, region, context)
        return (
            PhysicalCandidate(
                graph=plan.graph,
                plan=plan,
                estimated_seconds=plan.estimated_seconds,
            ),
        )

    def start(self, context: GpuContext) -> QuailModelExecution:
        return QuailModelExecution(context)

    def _bind_runtime_data(self, plan, region, context):
        """Put tokenized prompts and answer tokens in the physical plan."""
        tokenizer = context.tokenizer
        operators = region.logical_plan.operators()
        filters, joins = operators.filters, operators.joins
        encoded_nodes = []
        for node in plan.nodes:
            if isinstance(node, AiFilter):
                predicates = filters[node.alias]
                questions = tuple(
                    tuple(predicates[stage.written_pos].prompt.tail_token_ids)
                    for stage in node.stages
                )
                if any(not question for question in questions):
                    raise ValueError("filter prompts have no token ids")
                node = replace(node, question_token_ids=questions)
            elif isinstance(node, AiJoin):
                stages = []
                for stage in node.stages:
                    prompt = joins[stage.written_pos].prompt
                    runtime_ids = {
                        alias: (tuple(label), tuple(frame))
                        for alias, label, frame in prompt.label_token_ids
                    }
                    stages.append(replace(
                        stage,
                        frame_token_ids=runtime_ids[stage.anchor][1],
                        label_token_ids=tuple(
                            (alias, runtime_ids[alias][0])
                            for alias in stage.partners
                        ),
                        tail_token_ids=tuple(prompt.tail_token_ids),
                    ))
                node = replace(node, stages=tuple(stages))
            encoded_nodes.append(node)

        true_ids, false_ids = (
            true_false_token_ids(tokenizer) if tokenizer is not None
            else ([], []))
        prompts = operators.prompts
        pre_ids = (
            list(tokenizer(shared_preamble(context.model.turn_prefix)))
            if tokenizer is not None else
            list(prompts[0].preamble_token_ids) if prompts else []
        )
        return replace(
            plan,
            nodes=tuple(encoded_nodes),
            root=plan.root,
            settings={
                **plan.settings,
                "true_ids": true_ids,
                "false_ids": false_ids,
                "pre_ids": pre_ids,
                "filter_limit": (
                    None if any(
                        isinstance(node, AiJoin)
                        for node in encoded_nodes
                    ) else region.logical_plan.root.limit
                ),
            },
        )

    def prepare_request(self, context) -> None:
        """Boot the GPU for a request before its documents are ready."""
        prepare_quail_request(context)

    def execute_request(self, context) -> Any:
        """Run one Quail request inside a compute process."""
        return execute_quail_request(context)
