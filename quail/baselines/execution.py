"""Execution of one request node over a loaded engine."""

from __future__ import annotations

import itertools
import time
from functools import partial
from typing import Any, Mapping

from quail.backends.base import GpuContext
from quail.baselines.nodes import RequestExecution
from quail.baselines.operators import (
    allowed_members_by_anchor,
    choose_documents,
    classification_table,
    classify_documents,
    extract_answers,
    extract_documents,
    filter_answers,
    join_answers,
    operator_at_a_time_filter,
    pipelined_filter,
    request_step,
    text_value,
    token_list,
)
from quail.baselines.request_scheduling import (
    canvas_answer,
    run_join_grouped,
    text_answer,
    true_bit,
)
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.physical import PhysicalNode


class RequestModelExecution:
    """Execute one request engine node with one loaded engine."""

    def __init__(self, context: GpuContext):
        settings = context.query_settings
        self.client = settings["client"]
        self.sampling_params = settings["sampling_params"]
        self.documents = settings["documents"]
        self.document_texts = settings.get("document_texts", {})
        self.pairs = {}         # written position -> pair table, from ports
        true_ids = set(settings["true_ids"])
        if context.model.canvas_tokens == 1:
            self.read_answer = partial(
                canvas_answer, true_ids=true_ids,
                false_ids=set(settings["false_ids"]))
        elif context.model.canvas_tokens:
            self.read_answer = text_answer
        else:
            self.read_answer = partial(true_bit, true_ids=true_ids)
        self.decides = context.model.role == "decision"
        if self.decides:
            from quail.baselines.vllm_decision import decision_bit, pooling_params

            self.sampling_params = pooling_params(settings["decision_offsets"])
            self.read_answer = decision_bit
        self.capacity = settings["capacity"]
        self.filter_submission = settings["filter_submission"]
        self.join_submission = settings["join_submission"]
        self.submit_text = bool(getattr(self.client, "accepts_text", False))

    def _classify(self, spec, bodies) -> dict:
        if self.decides:
            return choose_documents(self.client, spec, bodies)
        return classify_documents(self.client, spec, bodies)

    def _join_submission(self, prefixes) -> str:
        """Suffix-major only when every anchor prefix fits in KV at once.

        Suffix-major sends every anchor for one partner before the next
        partner, so anchors that do not all fit are recomputed once per
        partner; anchor-major keeps each anchor's prefix hot instead.
        """
        if self.join_submission != "suffix-major":
            return self.join_submission
        held = sum(len(prefix) for prefix in prefixes)
        if held > self.capacity["kv_cache_size_tokens"]:
            return "anchor-major"
        return "suffix-major"

    def execute(
        self,
        node: PhysicalNode,
        inputs: Mapping[str, Any],
    ) -> NodeResult:
        if not isinstance(node, RequestExecution):
            raise TypeError(
                f"request backend cannot execute {node.type_name!r}"
            )
        for port in node.inputs:
            if port.source.port.startswith("pairs:"):
                position = int(port.source.port.split(":", 1)[1])
                self.pairs[position] = inputs[port.name]
        started = time.perf_counter()
        # the scans supply the documents their regular predicates kept
        survivors = {
            alias: list(range(len(self.documents[alias])))
            for alias in node.aliases
        }
        for port in node.inputs:
            if port.source.port.startswith("ids:"):
                alias = port.source.port.split(":", 1)[1]
                if alias in survivors:
                    survivors[alias] = [int(document)
                                        for document in inputs[port.name]]
        outputs = {}
        steps = []

        self._execute_filters(node, survivors, outputs, steps)
        self._execute_classifications(node, survivors, outputs, steps)
        self._execute_joins(node, survivors, outputs, steps)
        self._execute_joined_classifications(node, survivors, outputs, steps)
        self._execute_extractions(node, survivors, outputs, steps)

        for alias in node.aliases:
            outputs[f"ids:{alias}"] = survivors[alias]
        wall_s = time.perf_counter() - started
        return NodeResult(
            outputs,
            NodeMetrics(
                wall_s=wall_s,
                input_rows=sum(len(self.documents[alias]) for alias in node.aliases),
                output_rows=sum(len(value) for value in survivors.values()),
                evaluated_documents=sum(
                    step["requests"] if step["kind"] == "filter" else step["n_in"]
                    for step in steps if not step.get("partner")
                    and step["kind"] != "join"),
                evaluated_document_pairs=sum(step["requests"] for step in steps
                                             if step.get("partner")
                                             or step["kind"] == "join"),
                fresh_tokens=sum(step["fresh_tokens"] for step in steps),
                cached_tokens=sum(step["cached_tokens"] for step in steps),
                extension={
                    "steps": steps,
                    "prompt_tokens": sum(step["prompt_tokens"] for step in steps),
                    "requests": sum(step["requests"] for step in steps),
                    "capacity": dict(self.capacity),
                },
            ),
        )

    def _execute_filters(self, node, survivors, outputs, steps):
        """Submit filters and collect their answers."""
        for filter_index, spec in enumerate(node.filters):
            document_ids = list(survivors[spec.alias])
            bodies = [
                token_list(node.preamble_token_ids)
                + token_list(self.documents[spec.alias][document])
                for document in document_ids
            ]
            body_texts = None
            question_texts = None
            if self.submit_text:
                texts = self.document_texts.get(spec.alias)
                if (
                    texts is None
                    or len(spec.question_texts) != len(spec.question_token_ids)
                ):
                    raise ValueError("vLLM text prompts are missing filter text")
                body_texts = [
                    node.preamble_text + text_value(texts, document)
                    for document in document_ids
                ]
                question_texts = list(spec.question_texts)
            if self.filter_submission == "operator-at-a-time":
                result = operator_at_a_time_filter(
                    self.client,
                    self.sampling_params,
                    bodies,
                    spec.question_token_ids,
                    self.read_answer,
                    body_texts=body_texts,
                    question_texts=question_texts,
                )
            elif self.filter_submission == "pipelined":
                result = pipelined_filter(
                    self.client,
                    self.sampling_params,
                    bodies,
                    spec.question_token_ids,
                    self.read_answer,
                    f"filter-{filter_index}-{spec.alias}",
                    body_texts=body_texts,
                    question_texts=question_texts,
                )
            else:
                raise ValueError(
                    f"unknown filter submission {self.filter_submission!r}"
                )
            global_answers = {
                (document_ids[local], stage): answer
                for (local, stage), answer in result["answers"].items()
            }
            outputs[f"filter_answers:{spec.alias}"] = filter_answers(
                spec.alias,
                spec.written_positions,
                global_answers,
            )
            survivors[spec.alias] = [
                document_ids[local] for local in result["survivors"]
            ]
            steps.append(request_step(result,
                kind="filter",
                alias=spec.alias,
                submission=self.filter_submission,
                n_in=len(document_ids),
                n_out=len(survivors[spec.alias]),
                doc_cap=result["doc_cap"],
            ))

    def _execute_classifications(self, node, survivors, outputs, steps):
        """Submit classifications and collect their answers."""
        for spec in node.classifies:
            if spec.partner is not None:
                continue
            document_ids = list(survivors[spec.alias])
            result = self._classify(
                spec, [
                    token_list(node.preamble_token_ids)
                    + token_list(self.documents[spec.alias][document])
                    for document in document_ids])
            # a document whose answer names no label has no label row
            # and leaves the query
            table = classification_table(
                spec, {spec.alias: document_ids}, result["labels"])
            outputs[f"label_answers:{spec.output}"] = table
            labeled = list(zip(table.column(spec.alias).to_pylist(),
                               table.column(spec.output).to_pylist()))
            survivors[spec.alias] = table.column(spec.alias).to_pylist()
            test_answers = {}
            for index, (position, accepted) in enumerate(spec.tests):
                kept = set(accepted)
                alive = set(survivors[spec.alias])
                for document, label in labeled:
                    if document in alive:
                        test_answers[(document, index)] = label in kept
                survivors[spec.alias] = [
                    document for document, label in labeled
                    if label in kept and document in alive]
            if spec.tests:
                outputs[f"label_in_answers:{spec.output}"] = (
                    filter_answers(
                        spec.alias, [position for position, _ in spec.tests],
                        test_answers))
            steps.append(request_step(result,
                kind="classify",
                alias=spec.alias,
                output=spec.output,
                n_in=len(document_ids),
                n_out=len(survivors[spec.alias]),
                unmatched=result["unmatched"],
            ))

    def _execute_joins(self, node, survivors, outputs, steps):
        """Submit joins and collect their answers."""
        for spec in node.joins:
            alias_documents = {
                alias: list(survivors[alias]) for alias in spec.aliases
            }
            if spec.anchor is not None:
                anchor = spec.anchor
            else:
                anchor = max(
                    spec.aliases,
                    key=lambda alias: (
                        sum(
                            len(self.documents[alias][document])
                            for document in alias_documents[alias]
                        ) / max(1, len(alias_documents[alias]))
                    ),
                )
            partners = tuple(
                alias for alias in spec.aliases if alias != anchor
            )
            labels = dict(spec.label_token_ids)
            frames = dict(spec.frame_token_ids)
            anchor_ids = alias_documents[anchor]
            preamble = token_list(node.preamble_token_ids)
            prefixes = [
                preamble
                + token_list(self.documents[anchor][document])
                + token_list(frames[anchor])
                for document in anchor_ids
            ]
            members = list(itertools.product(
                *(alias_documents[alias] for alias in partners)
            ))
            suffixes = []
            prefix_texts = None
            suffix_texts = None
            if self.submit_text:
                labels_text = dict(spec.label_texts)
                frames_text = dict(spec.frame_texts)
                if (
                    any(alias not in self.document_texts for alias in spec.aliases)
                    or set(labels_text) != set(spec.aliases)
                    or set(frames_text) != set(spec.aliases)
                ):
                    raise ValueError("vLLM text prompts are missing join text")
                prefix_texts = [
                    node.preamble_text
                    + text_value(self.document_texts[anchor], document)
                    + frames_text[anchor]
                    for document in anchor_ids
                ]
                suffix_texts = []
            for member in members:
                suffix = []
                suffix_text = ""
                for alias, document in zip(partners, member):
                    suffix.extend(token_list(labels[alias]))
                    suffix.extend(token_list(
                        self.documents[alias][document]
                    ))
                    if suffix_texts is not None:
                        suffix_text += labels_text[alias]
                        suffix_text += text_value(
                            self.document_texts[alias], document
                        )
                suffix.extend(token_list(spec.tail_token_ids))
                suffixes.append(suffix)
                if suffix_texts is not None:
                    suffix_texts.append(suffix_text + spec.tail_text)
            # a join over pairs asks each anchor about its own members
            allowed = None
            request_pairs = None
            if spec.written_pos in self.pairs:
                allowed = allowed_members_by_anchor(
                    anchor, partners, anchor_ids, members,
                    self.pairs[spec.written_pos])
                request_pairs = [
                    (anchor_index, member_index)
                    for anchor_index, mine in enumerate(allowed)
                    for member_index in mine
                ]

            submission = self._join_submission(prefixes)
            if prefixes and suffixes and request_pairs != []:
                result = run_join_grouped(
                    self.client,
                    self.sampling_params,
                    prefixes,
                    suffixes,
                    self.read_answer,
                    submission=submission,
                    pairs=request_pairs,
                    prefix_texts=prefix_texts,
                    suffix_texts=suffix_texts,
                )
                answers = [bool(answer) for answer in result["answers"]]
            else:
                result = {
                    "wall": 0.0,
                    "fresh_tokens": 0,
                    "cached_tokens": 0,
                    "prompt_tokens": 0,
                }
                answers = []

            rows = []
            for anchor_index, anchor_id in enumerate(anchor_ids):
                mine = (range(len(members)) if allowed is None
                        else allowed[anchor_index])
                for member_index in mine:
                    by_alias = {anchor: anchor_id}
                    by_alias.update(zip(partners, members[member_index]))
                    rows.append(tuple(by_alias[alias] for alias in spec.aliases))
            outputs[f"join_answers:{spec.written_pos}"] = join_answers(
                spec,
                rows,
                answers,
                anchor,
                partners,
            )
            true_rows = [row for row, answer in zip(rows, answers) if answer]
            if spec.semantics == "full":
                for alias_index, alias in enumerate(spec.aliases):
                    survivors[alias] = sorted({
                        row[alias_index] for row in true_rows
                    })
            else:
                anchor_index = spec.aliases.index(anchor)
                matched = {row[anchor_index] for row in true_rows}
                if spec.semantics == "anti":
                    survivors[anchor] = [
                        document for document in survivors[anchor]
                        if document not in matched
                    ]
                else:
                    survivors[anchor] = sorted(matched)
            steps.append(request_step(result,
                kind="join",
                written_pos=spec.written_pos,
                anchor=anchor,
                partners=list(partners),
                submission=submission,
                pairs=len(rows),
                true_pairs=len(true_rows),
            ))

    def _execute_joined_classifications(self, node, survivors, outputs, steps):
        """Submit joined classifications and collect their answers."""
        for spec in node.classifies:
            if spec.partner is None:
                continue
            # the rows the join kept, each decoded as anchor, note,
            # partner block, and the question
            answers = outputs[f"join_answers:{spec.join_written_pos}"]
            kept = answers.filter(answers.column("answer"))
            pairs = list(zip(kept.column(spec.alias).to_pylist(),
                             kept.column(spec.partner).to_pylist()))
            note, partner_label = (
                token_list(ids) for ids in spec.join_layout_token_ids)
            result = self._classify(
                spec, [
                    token_list(node.preamble_token_ids)
                    + token_list(self.documents[spec.alias][anchor])
                    + note + partner_label
                    + token_list(self.documents[spec.partner][partner])
                    for anchor, partner in pairs])
            table = classification_table(
                spec, {spec.alias: kept.column(spec.alias),
                       spec.partner: kept.column(spec.partner)}, result["labels"])
            outputs[f"label_answers:{spec.output}"] = table
            steps.append(request_step(result,
                kind="classify",
                alias=spec.alias,
                partner=spec.partner,
                output=spec.output,
                n_in=len(pairs),
                n_out=table.num_rows,
                unmatched=result["unmatched"],
            ))

    def _execute_extractions(self, node, survivors, outputs, steps):
        """Submit extractions and collect their answers."""
        for spec in node.extracts:
            document_ids = list(survivors[spec.alias])
            texts = self.document_texts.get(spec.alias)
            if texts is None:
                raise ValueError(
                    f"AI.EXTRACT needs the text of {spec.alias!r}, which "
                    f"the scan did not load")
            document_texts = [text_value(texts, document)
                              for document in document_ids]
            preamble = token_list(spec.preamble_token_ids
                                   or node.preamble_token_ids)
            preamble_text = spec.preamble_text or node.preamble_text
            result = extract_documents(
                self.client, spec, [
                    preamble + token_list(self.documents[spec.alias][document])
                    for document in document_ids],
                document_texts,
                body_texts=([preamble_text + text for text in document_texts]
                            if self.submit_text else None))
            outputs[f"extract_answers:{spec.output}"] = extract_answers(
                spec, document_ids, result["answers"], result["spans"])
            steps.append(request_step(result,
                kind="extract",
                alias=spec.alias,
                output=spec.output,
                n_in=len(document_ids),
                n_out=len(document_ids) - result["unanswered"],
                unanswered=result["unanswered"],
                unaligned=result["unaligned"],
            ))
