"""Shared planning and execution for request based model backends."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Mapping

import pyarrow as pa

from quail.backends.base import GpuContext
from quail.backends.request_scheduling import (
    canvas_answer,
    run_join_grouped,
    text_answer,
    true_bit,
)
from quail.cost import budgets
from quail.execution.labels import match_label
from quail.execution.pairs import (
    allowed_members,
    members_by_partner,
    pair_partner,
    partner_map,
)
from quail.execution.result import answer_table
from quail.execution.runner import (
    ExecutionContext,
    GenericRunner,
    ModelNodeRuntime,
    NodeMetrics,
    NodeResult,
    compute_subgraph,
    scalar_node_metrics,
)
from quail.execution.types import PhysicalResponse, export_physical_outputs
from quail.logical import Aggregate as LogicalAggregate
from quail.logical import (
    Alias,
    Apply,
    answer_row_offsets,
    effective_selectivity,
    filter_question_text,
    join_label,
    join_outer_input,
    label_text,
    render_join_frame,
    shared_preamble,
)
from quail.physical import (
    Limit,
    PhysicalNode,
    PortRef,
    Project,
    Recombine,
    RequestClassifySpec,
    RequestExecution,
    RequestFilterSpec,
    RequestJoinSpec,
    Scan,
)
from quail.physical.base import input_ports
from quail.planner import (
    column_tests,
    default_order_rule,
    hash_join_nodes,
    order_filters_indexed,
    preamble_tokens,
)
from quail.planner import (
    join_specs as logical_join_specs,
)
from quail.planner.join_order import search_joins
from quail.planner.physical_optimizer import PhysicalCandidate, SupportResult
from quail.planner.plan import CorpusStats, PhysicalPlan, Refusal
from quail.planner.statistics import prepare_filter_costs, summarize_alias
from quail.planner.validation import has_label


def _answer_ids(tokenizer) -> tuple[list[int], list[int]]:
    true_ids = set()
    false_ids = set()
    for word in ("TRUE", " TRUE", "True", " True"):
        tokens = tokenizer(word)
        if tokens:
            true_ids.add(int(tokens[0]))
    for word in ("FALSE", " FALSE", "False", " False"):
        tokens = tokenizer(word)
        if tokens:
            false_ids.add(int(tokens[0]))
    return sorted(true_ids), sorted(false_ids)


def plan_request_backend(
    region,
    context,
    *,
    backend_name: str,
    filter_submission: str,
    join_submission: str,
    scores_labels: bool = False,
) -> tuple[PhysicalCandidate, ...]:
    """Build one physical request plan for a request engine.

    Args:
        region: The logical model region to plan.
        context: The planning context.
        backend_name: The backend the plan names.
        filter_submission: How filter stages become requests.
        join_submission: How join tuples become requests.
        scores_labels: Whether the engine decodes an answer as text,
            which AI.CLASSIFY needs.
    """
    if any(isinstance(node, Apply) for node in region.logical_plan.walk()):
        return (PhysicalCandidate(
            graph=None,
            plan=Refusal(
                reasons=("apply() functions run on the Quail backend only",),
                constraint="apply_needs_quail_backend",
                needed=1, available=0, unit="backends"),
            estimated_seconds=float("inf"),
        ),)
    root = region.logical_plan.result
    if root.order or root.distinct or root.offset \
            or isinstance(root.input, LogicalAggregate):
        return (PhysicalCandidate(
            graph=None,
            plan=Refusal(
                reasons=("ORDER BY, DISTINCT, OFFSET, and GROUP BY run on "
                         "the Quail backend only",),
                constraint="sort_needs_quail_backend",
                needed=1, available=0, unit="backends"),
            estimated_seconds=float("inf"),
        ),)
    classifies = has_label(region.logical_plan)
    if any(isinstance(column, Alias)
           and getattr(column.expression, "probabilities", False)
           for column in region.logical_plan.projection.columns):
        return (PhysicalCandidate(
            graph=None,
            plan=Refusal(
                reasons=(f"AI.CLASSIFY on {backend_name} decodes the label "
                         f"as text and has no probability for the others",),
                constraint="classify_probabilities_need_quail_backend",
                needed=1, available=0, unit="backends"),
            estimated_seconds=float("inf"),
        ),)
    # a diffusion model's vLLM canvas is sized for a filter's one
    # answer token, not a label
    scores_labels = scores_labels and not context.model.canvas_tokens
    # a decision model scores one option block per label
    decides = context.model.role == "decision"
    if classifies and not ((scores_labels or decides)
                           and context.tokenizer is not None):
        reason = (f"AI.CLASSIFY on {backend_name} decodes each answer as "
                  f"text, which its engine does not return for "
                  f"{context.model.name!r}"
                  if context.tokenizer is not None
                  else "AI.CLASSIFY planning needs the model's tokenizer")
        return (PhysicalCandidate(
            graph=None,
            plan=Refusal(
                reasons=(reason,),
                constraint="classify_needs_quail_backend",
                needed=1, available=0, unit="backends"),
            estimated_seconds=float("inf"),
        ),)
    operators = region.logical_plan.operators()
    scans, filters, joins = operators.scans, operators.filters, operators.joins
    ask_filters = {alias: list(ps) for alias, ps in filters.items() if ps}
    # the column tests keep an estimated fraction of each table, as
    # the Quail planner's statistics do
    stats = {
        alias: CorpusStats(
            n_docs=round(len(lengths) * context.scan_fractions.get(alias, 1.0)),
            total_tokens=round(sum(int(length) for length in lengths)
                               * context.scan_fractions.get(alias, 1.0)),
            max_doc_tokens=max((int(length) for length in lengths), default=0),
        )
        for alias, lengths in context.document_tokens.items()
    }
    nodes = []
    input_refs = []
    aliases = tuple(scan.alias for scan in scans)
    for scan in scans:
        if scan.alias not in stats:
            raise ValueError(f"no document tokens for alias {scan.alias!r}")
        summary = stats[scan.alias]
        n_docs = len(context.document_tokens[scan.alias])
        node = Scan(
            node_id=f"scan:{scan.alias}",
            alias=scan.alias,
            input_id=scan.alias,
            n_docs=n_docs,
            total_tokens=sum(context.document_tokens[scan.alias]),
            shard_ranges=((0, n_docs),),
            shard_token_loads=(summary.total_tokens,),
            predicates=column_tests(scan),
            expected_docs=float(summary.n_docs) if scan.predicates else None,
        )
        nodes.append(node)
        input_refs.append(PortRef(node.node_id, f"ids:{scan.alias}"))

    for node in hash_join_nodes(joins, context.pair_fractions, input_refs):
        nodes.append(node)
        input_refs.append(PortRef(node.node_id, f"pairs:{node.written_pos}"))

    rule = context.order or default_order_rule(ask_filters, joins)[0]
    chunk_tokens = budgets.chunk_budget(context.model, context.device)
    preamble_count = preamble_tokens(ask_filters, joins)
    filter_orders = {
        alias: order_filters_indexed(prepare_filter_costs(
            predicates,
            prefix_tokens=(
                preamble_count + stats[alias].mean_doc_tokens
            ),
            model=context.model,
            device=context.device,
            chunk_tokens=chunk_tokens,
        ), rule)
        for alias, predicates in ask_filters.items()
    }
    live = {
        alias: float(summary.n_docs) for alias, summary in stats.items()
    }
    for alias, predicates in operators.all_filters().items():
        for predicate in predicates:
            live[alias] *= effective_selectivity(predicate.selectivity)
    search_specs = logical_join_specs(joins, context.pair_fractions)
    join_search = search_joins(
        search_specs,
        live,
        {
            alias: summarize_alias(lengths, context.model.sliding_window)
            for alias, lengths in context.document_tokens.items()
        },
        {},
        preamble_count,
        chunk_tokens,
        context.model,
        context.device,
        fixed_order=rule == "as_written",
    )
    join_sequence = (
        join_search["seq"] if join_search is not None else [
            (position, spec["anchor"] or spec["aliases"][0])
            for position, spec in enumerate(search_specs)
        ]
    )

    filter_specs = []
    prompts = []
    first_anchor = join_sequence[0][1] if join_sequence else None
    for alias in sorted(aliases, key=lambda alias: alias == first_anchor):
        predicates = filters.get(alias, ())
        if alias not in ask_filters:
            continue
        prompts.extend(predicate.prompt for predicate in ask_filters[alias])
        written_positions = tuple(filter_orders[alias])
        questions = tuple(
            tuple(predicates[position].prompt.tail_token_ids)
            for position in written_positions
        )
        question_texts = tuple(
            filter_question_text(predicates[position].prompt)
            for position in written_positions
        )
        if any(not question for question in questions):
            raise ValueError("filter prompts have no token ids")
        filter_specs.append(RequestFilterSpec(
            alias=alias,
            written_positions=written_positions,
            question_token_ids=questions,
            question_texts=question_texts,
        ))

    # each classification labels the alias's documents after its AI.IF
    # chain; the filters on its labels keep the accepted documents for the joins
    label_plan = operators.labels
    classify_specs = []
    for call, alias in label_plan.calls:
        prompts.append(call.prompt)
        pair = {}
        if len(call.aliases()) == 2:
            # a classification of the rows a join keeps: decoded after
            # that join, one request per row
            partner = call.aliases()[1]
            parts = {part_alias: (label, frame)
                     for part_alias, label, frame in call.prompt.label_token_ids}
            written_pos = next(
                (position for position, join in enumerate(joins)
                 if {argument.alias for argument in join.prompt.args}
                 == set(call.aliases())), None)
            if written_pos is None:
                raise ValueError(
                    f"the classification of {alias!r} x {partner!r} pairs "
                    f"needs a join of the two")
            pair = dict(partner=partner,
                        join_layout_token_ids=(tuple(parts[alias][1]),
                                        tuple(parts[partner][0])),
                        join_written_pos=written_pos)
        classify_specs.append(RequestClassifySpec(
            alias=alias,
            output=label_plan.names[call],
            tail_token_ids=tuple(call.prompt.tail_token_ids),
            labels=tuple(call.labels),
            label_token_ids=tuple(
                tuple(context.tokenizer(label_text(label)))
                for label in call.labels),
            tests=tuple(
                (test.position, test.values)
                for test in operators.label_filters.get(alias, ())
                if test.call == call),
            option_offsets=(_option_offsets(call.prompt, context.tokenizer)
                            if decides else ()),
            **pair,
        ))

    join_specs = []
    for written_pos, selected_anchor in join_sequence:
        join = joins[written_pos]
        prompt = join.prompt
        prompts.append(prompt)
        aliases_in_prompt = tuple(argument.alias for argument in prompt.args)
        labels = tuple(
            (str(alias), tuple(label))
            for alias, label, _frame in prompt.label_token_ids
        )
        frames = tuple(
            (str(alias), tuple(frame))
            for alias, _label, frame in prompt.label_token_ids
        )
        text_parts = tuple(
            (
                argument.alias,
                join_label(index),
                render_join_frame(prompt.template, index),
            )
            for index, argument in enumerate(prompt.args)
        )
        if len(labels) != len(aliases_in_prompt) or not prompt.tail_token_ids:
            raise ValueError("join prompts have no token ids")
        outer_aliases = tuple(dict.fromkeys(
            field.alias for field in join_outer_input(join).output_schema()
        ))
        join_specs.append(RequestJoinSpec(
            written_pos=written_pos,
            aliases=aliases_in_prompt,
            outer_aliases=outer_aliases,
            anchor=selected_anchor,
            semantics=join.semantics,
            selectivity=join.selectivity,
            label_token_ids=labels,
            frame_token_ids=frames,
            tail_token_ids=tuple(prompt.tail_token_ids),
            label_texts=tuple(
                (alias, label) for alias, label, _frame in text_parts
            ),
            frame_texts=tuple(
                (alias, frame) for alias, _label, frame in text_parts
            ),
            tail_text=prompt.tail,
        ))

    preambles = {
        tuple(prompt.preamble_token_ids)
        for prompt in prompts
        if prompt.preamble_token_ids
    }
    if len(preambles) > 1:
        raise ValueError("request prompts have different preambles")
    preamble = next(iter(preambles), ())
    preamble_texts = {prompt.preamble for prompt in prompts}
    if len(preamble_texts) > 1:
        raise ValueError("request prompts have different preamble text")
    if not preamble and prompts and context.tokenizer is not None:
        preamble = tuple(context.tokenizer(
            shared_preamble(context.model.turn_prefix)))

    request_node = RequestExecution(
        node_id="request-model",
        inputs=input_ports(tuple(input_refs)),
        backend_name=backend_name,
        aliases=aliases,
        preamble_token_ids=tuple(preamble),
        preamble_text=(
            next(iter(preamble_texts), "")
            or shared_preamble(context.model.turn_prefix)
        ),
        filters=tuple(filter_specs),
        joins=tuple(join_specs),
        classifies=tuple(classify_specs),
    )
    nodes.append(request_node)

    full_joins = [spec for spec in join_specs if spec.semantics == "full"]
    if len(full_joins) == 1 and len(join_specs) == 1:
        # one full join and no gates: its true pairs are the result rows
        sink_input = PortRef(
            request_node.node_id, f"join_answers:{full_joins[0].written_pos}")
    elif full_joins:
        result_aliases = tuple(dict.fromkeys(
            alias for spec in full_joins for alias in spec.aliases
        ))
        edges = tuple(
            PortRef(request_node.node_id, f"join_answers:{spec.written_pos}")
            for spec in full_joins
        ) + tuple(
            PortRef(request_node.node_id, f"ids:{alias}")
            for alias in result_aliases
        )
        nodes.append(Recombine(
            node_id="recombine",
            inputs=input_ports(edges),
            alias_order=result_aliases,
        ))
        sink_input = PortRef("recombine", "tuples")
    else:
        sink_input = PortRef(request_node.node_id, f"ids:{aliases[0]}")

    label_ports = tuple(
        PortRef(request_node.node_id, f"label_answers:{name}")
        for name in label_plan.projected.values())
    nodes.append(Project(
        node_id="project",
        inputs=input_ports((sink_input, *label_ports)),
        columns=tuple(
            column.name if isinstance(column, Alias)
            else f"{column.alias}.{column.column}"
            for column in region.logical_plan.projection.columns
        ),
    ))
    if region.logical_plan.result.limit is not None:
        nodes.append(Limit(
            node_id="limit",
            inputs=input_ports((PortRef("project", "rows"),)),
            count=region.logical_plan.result.limit,
        ))

    true_ids, false_ids = _answer_ids(context.tokenizer)
    plan = PhysicalPlan(
        model=context.model.name,
        device=context.device.name,
        workers=1,
        backend=backend_name,
        estimated_seconds=0.0,
        nodes=tuple(nodes),
        settings={
            "filter_submission": filter_submission,
            "join_submission": join_submission,
            "true_ids": true_ids,
            "false_ids": false_ids,
            "order_rule": rule,
            **({"decision_offsets": list(answer_row_offsets(
                context.model.prompt_layout, context.tokenizer,
                context.model.turn_suffix))} if decides else {}),
        },
    )
    return (PhysicalCandidate(
        graph=plan.graph,
        plan=plan,
        estimated_seconds=plan.estimated_seconds,
    ),)


def _option_offsets(prompt, tokenizer) -> tuple[int, ...]:
    """Distances before a choice prompt's last row of each option's end, then 0.

    The tail's segments are the question, one block per option, and
    the close, each tokenized apart.
    """
    lengths = [len(tokenizer(segment)) for segment in prompt.tail_segments]
    return tuple(sum(lengths[index + 1:])
                 for index in range(1, len(lengths) - 1)) + (0,)


def _filter_answer_table(alias, written_positions, answers) -> pa.Table:
    documents = []
    predicates = []
    values = []
    for (document, stage_index), answer in sorted(answers.items()):
        documents.append(int(document))
        predicates.append(int(written_positions[stage_index]))
        values.append(bool(answer))
    schema = pa.schema(
        [
            pa.field(alias, pa.int32(), nullable=False),
            pa.field("predicate", pa.int32(), nullable=False),
            pa.field("answer", pa.bool_(), nullable=False),
        ],
        metadata={
            b"quail.kind": b"filter_answers",
            b"quail.alias": alias.encode("utf-8"),
        },
    )
    return pa.Table.from_arrays(
        [
            pa.array(documents, type=pa.int32()),
            pa.array(predicates, type=pa.int32()),
            pa.array(values, type=pa.bool_()),
        ],
        schema=schema,
    )


def _allowed_members(anchor, partners, anchor_ids, members, pairs) -> list:
    """Per anchor, the member indices its equality conditions allow."""
    position = partners.index(pair_partner(anchor, partners))
    by_partner = members_by_partner(members, position)
    rows = partner_map(pairs, anchor, partners[position])
    return [allowed_members(rows, by_partner, anchor_id)
            for anchor_id in anchor_ids]


def _join_answer_table(spec, rows, answers, anchor, partners) -> pa.Table:
    columns = {
        alias: [int(row[index]) for row in rows]
        for index, alias in enumerate(spec.aliases)
    }
    return answer_table(
        columns,
        answers,
        "join_answers",
        metadata={
            "anchor": anchor,
            "partners": ",".join(partners),
            "semantics": spec.semantics,
            "written_pos": spec.written_pos,
            **(
                {"selectivity": spec.selectivity}
                if spec.selectivity is not None else {}
            ),
        },
    )


def _token_list(values) -> list[int]:
    return [int(token) for token in values]


def _text_value(values, index: int) -> str:
    value = values[index]
    return value.as_py() if hasattr(value, "as_py") else str(value)


def _operator_at_a_time_filter(client, sampling_params, bodies, questions,
                               read_answer, *, body_texts=None,
                               question_texts=None):
    active = list(range(len(bodies)))
    answers = {}
    requests = prompt_tokens = cached_tokens = 0
    started = time.perf_counter()
    stages = []
    for stage_index, question in enumerate(questions):
        evaluated = list(active)
        if body_texts is not None and question_texts is not None:
            prompts = [
                body_texts[index] + question_texts[stage_index]
                for index in evaluated
            ]
        else:
            prompts = [
                {"prompt_token_ids": bodies[index] + list(question)}
                for index in evaluated
            ]
        outputs = (
            client.generate(prompts, sampling_params, use_tqdm=False)
            if prompts else []
        )
        next_active = []
        for document, output in zip(evaluated, outputs):
            answer = bool(read_answer(output))
            answers[(document, stage_index)] = answer
            requests += 1
            prompt_tokens += len(output.prompt_token_ids)
            cached_tokens += int(getattr(output, "num_cached_tokens", 0) or 0)
            if answer:
                next_active.append(document)
        active = next_active
        stages.append({
            "stage": stage_index,
            "n_in": len(evaluated),
            "n_out": len(active),
        })
    return {
        "wall_s": time.perf_counter() - started,
        "survivors": active,
        "answers": answers,
        "requests": requests,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "fresh_tokens": prompt_tokens - cached_tokens,
        "stages": stages,
        "doc_cap": None,
    }


def _pipelined_filter(client, sampling_params, bodies, questions, read_answer,
                      tag, *, body_texts=None, question_texts=None):
    if not bodies:
        return {
            "wall_s": 0.0,
            "survivors": [],
            "answers": {},
            "requests": 0,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "fresh_tokens": 0,
            "stages": [],
            "doc_cap": 0,
        }
    result = client.run_filter_chain(
        sampling_params,
        bodies,
        [list(question) for question in questions],
        read_answer,
        tag=tag,
        body_texts=body_texts,
        question_texts=question_texts,
    )
    stages = []
    for stage in range(1, len(questions) + 1):
        evaluated = [
            document for document in range(len(bodies))
            if (document, stage) in result["answers"]
        ]
        stages.append({
            "stage": stage - 1,
            "n_in": len(evaluated),
            "n_out": sum(
                bool(result["answers"][(document, stage)])
                for document in evaluated
            ),
        })
    return {
        "wall_s": result["wall"],
        "survivors": result["survivors"],
        "answers": {
            (document, stage - 1): bool(answer)
            for (document, stage), answer in result["answers"].items()
        },
        "requests": result["requests"],
        "prompt_tokens": result["prompt_tokens"],
        "cached_tokens": result["cached_tokens"],
        "fresh_tokens": result["prompt_tokens"] - result["cached_tokens"],
        "stages": stages,
        "doc_cap": result["doc_cap"],
    }


def _classify_documents(client, spec, bodies) -> dict:
    """Decode one label per prompt body with greedy generation.

    Generation stops after one more token than the longest label has.

    Args:
        client: The engine client.
        spec: The RequestClassifySpec with the labels and the prompt tail.
        bodies: Token ids for each prompt before the tail, preamble included.

    Returns:
        A dict with one label per body (None when the answer names no label),
        the unmatched count, the request and token counts, and the generation
        seconds. Fresh tokens count the uncached prompt tokens and each
        generated token fed back to decode the next one.
    """
    tail = _token_list(spec.tail_token_ids)
    longest = max(len(ids) for ids in spec.label_token_ids)
    params = client.decode_params(longest + 1)
    prompts = [{"prompt_token_ids": body + tail} for body in bodies]
    started = time.perf_counter()
    outputs = client.generate(prompts, params, use_tqdm=False) if prompts else []
    wall_s = time.perf_counter() - started
    labels = []
    unmatched = 0
    prompt_tokens = cached_tokens = generated_tokens = fed_back = 0
    for output in outputs:
        prompt_tokens += len(output.prompt_token_ids)
        cached_tokens += int(getattr(output, "num_cached_tokens", 0) or 0)
        generated = len(output.outputs[0].token_ids)
        generated_tokens += generated
        # each generated token but the last is fed back through the model
        fed_back += max(generated - 1, 0)
        text = output.outputs[0].text or ""
        label = match_label(text, spec.labels)
        unmatched += label is None
        labels.append(label)
    return {
        "labels": labels,
        "wall_s": wall_s,
        "requests": len(prompts),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "fresh_tokens": prompt_tokens - cached_tokens + fed_back,
        "generated_tokens": generated_tokens,
        "unmatched": unmatched,
    }


def _choose_documents(client, spec, bodies) -> dict:
    """Score one option block per label per prompt body; take the best label.

    Args:
        client: The decision engine client.
        spec: The RequestClassifySpec with the labels, the prompt tail,
            and the option offsets.
        bodies: Token ids for each prompt before the tail, preamble included.

    Returns:
        The same fields as _classify_documents; no answer is unmatched
        and none generates tokens.
    """
    from quail.backends.vllm_decision import choice_index, pooling_params

    tail = _token_list(spec.tail_token_ids)
    prompts = [{"prompt_token_ids": body + tail} for body in bodies]
    started = time.perf_counter()
    outputs = (client.generate(prompts, pooling_params(spec.option_offsets))
               if prompts else [])
    wall_s = time.perf_counter() - started
    prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    cached_tokens = sum(int(output.num_cached_tokens or 0) for output in outputs)
    return {
        "labels": [spec.labels[choice_index(output)] for output in outputs],
        "wall_s": wall_s,
        "requests": len(prompts),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "fresh_tokens": prompt_tokens - cached_tokens,
        "generated_tokens": 0,
        "unmatched": 0,
    }


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
            from quail.backends.vllm_decision import decision_bit, pooling_params

            self.sampling_params = pooling_params(settings["decision_offsets"])
            self.read_answer = decision_bit
        self.capacity = settings["capacity"]
        self.filter_submission = settings["filter_submission"]
        self.join_submission = settings["join_submission"]
        self.submit_text = bool(getattr(self.client, "accepts_text", False))

    def _classify(self, spec, bodies) -> dict:
        if self.decides:
            return _choose_documents(self.client, spec, bodies)
        return _classify_documents(self.client, spec, bodies)

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
        # the scans supply the documents their column tests kept
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
        fresh_tokens = cached_tokens = requests = 0
        evaluated_documents = evaluated_pairs = 0

        for filter_index, spec in enumerate(node.filters):
            document_ids = list(survivors[spec.alias])
            bodies = [
                _token_list(node.preamble_token_ids)
                + _token_list(self.documents[spec.alias][document])
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
                    node.preamble_text + _text_value(texts, document)
                    for document in document_ids
                ]
                question_texts = list(spec.question_texts)
            if self.filter_submission == "operator-at-a-time":
                result = _operator_at_a_time_filter(
                    self.client,
                    self.sampling_params,
                    bodies,
                    spec.question_token_ids,
                    self.read_answer,
                    body_texts=body_texts,
                    question_texts=question_texts,
                )
            elif self.filter_submission == "pipelined":
                result = _pipelined_filter(
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
            outputs[f"filter_answers:{spec.alias}"] = _filter_answer_table(
                spec.alias,
                spec.written_positions,
                global_answers,
            )
            survivors[spec.alias] = [
                document_ids[local] for local in result["survivors"]
            ]
            steps.append({
                "kind": "filter",
                "alias": spec.alias,
                "submission": self.filter_submission,
                "n_in": len(document_ids),
                "n_out": len(survivors[spec.alias]),
                "wall_s": result["wall_s"],
                "requests": result["requests"],
                "fresh_tokens": result["fresh_tokens"],
                "cached_tokens": result["cached_tokens"],
                "doc_cap": result["doc_cap"],
            })
            requests += result["requests"]
            fresh_tokens += result["fresh_tokens"]
            cached_tokens += result["cached_tokens"]
            evaluated_documents += result["requests"]

        for spec in node.classifies:
            if spec.partner is not None:
                continue
            document_ids = list(survivors[spec.alias])
            result = self._classify(
                spec, [
                    _token_list(node.preamble_token_ids)
                    + _token_list(self.documents[spec.alias][document])
                    for document in document_ids])
            # a document whose answer names no label has no label row
            # and leaves the query
            labeled = [(document, label) for document, label
                       in zip(document_ids, result["labels"])
                       if label is not None]
            outputs[f"label_answers:{spec.output}"] = pa.table({
                spec.alias: pa.array([d for d, _ in labeled], pa.int32()),
                spec.output: pa.array([label for _, label in labeled],
                                      pa.string()),
            })
            survivors[spec.alias] = [document for document, _ in labeled]
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
                    _filter_answer_table(
                        spec.alias, [position for position, _ in spec.tests],
                        test_answers))
            steps.append({
                "kind": "classify",
                "alias": spec.alias,
                "output": spec.output,
                "n_in": len(document_ids),
                "n_out": len(survivors[spec.alias]),
                "wall_s": result["wall_s"],
                "requests": result["requests"],
                "fresh_tokens": result["fresh_tokens"],
                "cached_tokens": result["cached_tokens"],
                "generated_tokens": result["generated_tokens"],
                "unmatched": result["unmatched"],
            })
            requests += result["requests"]
            fresh_tokens += result["fresh_tokens"]
            cached_tokens += result["cached_tokens"]
            evaluated_documents += len(document_ids)

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
            preamble = _token_list(node.preamble_token_ids)
            prefixes = [
                preamble
                + _token_list(self.documents[anchor][document])
                + _token_list(frames[anchor])
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
                    + _text_value(self.document_texts[anchor], document)
                    + frames_text[anchor]
                    for document in anchor_ids
                ]
                suffix_texts = []
            for member in members:
                suffix = []
                suffix_text = ""
                for alias, document in zip(partners, member):
                    suffix.extend(_token_list(labels[alias]))
                    suffix.extend(_token_list(
                        self.documents[alias][document]
                    ))
                    if suffix_texts is not None:
                        suffix_text += labels_text[alias]
                        suffix_text += _text_value(
                            self.document_texts[alias], document
                        )
                suffix.extend(_token_list(spec.tail_token_ids))
                suffixes.append(suffix)
                if suffix_texts is not None:
                    suffix_texts.append(suffix_text + spec.tail_text)
            # a join over pairs asks each anchor about its own members
            allowed = None
            request_pairs = None
            if spec.written_pos in self.pairs:
                allowed = _allowed_members(
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
            outputs[f"join_answers:{spec.written_pos}"] = _join_answer_table(
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
            steps.append({
                "kind": "join",
                "written_pos": spec.written_pos,
                "anchor": anchor,
                "partners": list(partners),
                "submission": submission,
                "pairs": len(rows),
                "true_pairs": len(true_rows),
                "wall_s": result["wall"],
                "fresh_tokens": result["fresh_tokens"],
                "cached_tokens": result["cached_tokens"],
            })
            requests += len(rows)
            evaluated_pairs += len(rows)
            fresh_tokens += result["fresh_tokens"]
            cached_tokens += result["cached_tokens"]

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
                _token_list(ids) for ids in spec.join_layout_token_ids)
            result = self._classify(
                spec, [
                    _token_list(node.preamble_token_ids)
                    + _token_list(self.documents[spec.alias][anchor])
                    + note + partner_label
                    + _token_list(self.documents[spec.partner][partner])
                    for anchor, partner in pairs])
            labeled = [(pair, label) for pair, label
                       in zip(pairs, result["labels"]) if label is not None]
            outputs[f"label_answers:{spec.output}"] = pa.table({
                spec.alias: pa.array([a for (a, _), _ in labeled], pa.int32()),
                spec.partner: pa.array([p for (_, p), _ in labeled], pa.int32()),
                spec.output: pa.array([label for _, label in labeled],
                                      pa.string()),
            })
            steps.append({
                "kind": "classify",
                "alias": spec.alias,
                "partner": spec.partner,
                "output": spec.output,
                "n_in": len(pairs),
                "n_out": len(labeled),
                "wall_s": result["wall_s"],
                "requests": result["requests"],
                "fresh_tokens": result["fresh_tokens"],
                "cached_tokens": result["cached_tokens"],
                "unmatched": result["unmatched"],
            })
            requests += result["requests"]
            fresh_tokens += result["fresh_tokens"]
            cached_tokens += result["cached_tokens"]
            evaluated_pairs += result["requests"]

        for alias in node.aliases:
            outputs[f"ids:{alias}"] = survivors[alias]
        wall_s = time.perf_counter() - started
        return NodeResult(
            outputs,
            NodeMetrics(
                wall_s=wall_s,
                input_rows=sum(len(self.documents[alias]) for alias in node.aliases),
                output_rows=sum(len(value) for value in survivors.values()),
                evaluated_documents=evaluated_documents,
                evaluated_document_pairs=evaluated_pairs,
                fresh_tokens=fresh_tokens,
                cached_tokens=cached_tokens,
                extension={
                    "steps": steps,
                    "requests": requests,
                    "capacity": dict(self.capacity),
                },
            ),
        )


def execute_request_graph(context, backend, engine_state, boot):
    """Run a request backend over its compute subgraph."""
    try:
        import torch
    except ImportError:
        torch = None


    envelope = context.request.plan
    documents = {
        node.alias: context.request.inputs[node.input_id].documents
        for node in context.graph.nodes
        if isinstance(node, Scan)
    }
    document_texts = {
        node.alias: context.request.inputs[node.input_id].texts
        for node in context.graph.nodes
        if isinstance(node, Scan)
        and context.request.inputs[node.input_id].texts is not None
    }
    settings = dict(envelope["settings"])
    model_execution = backend.start(GpuContext(
        gpu_index=0,
        gpu_count=1,
        model=context.registry.model(envelope["model"]),
        device=context.registry.device(envelope["device"]),
        query_settings={
            **settings,
            **engine_state,
            "documents": documents,
            "document_texts": document_texts,
        },
    ))
    if engine_state["client"].reset_prefix_cache() is False:
        raise RuntimeError("request engine did not reset its prefix cache")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    compute_graph = compute_subgraph(context.graph)
    run = GenericRunner().run(
        compute_graph,
        ExecutionContext(
            runtimes=context.registry.runtimes,
            model_execution=model_execution,
            sources={
                **{alias: range(len(value))
                   for alias, value in documents.items()},
                **context.request.relations,
            },
            functions=context.registry.functions,
        ),
    )
    metrics = run.metrics
    peak_bytes = (
        int(torch.cuda.max_memory_allocated())
        if torch is not None and torch.cuda.is_available() else 0
    )
    request_result = run.nodes["request-model"]
    report = {
        "backend": backend.name,
        "wall_s": round(metrics.wall_s, 2),
        "boot_s": boot["boot_s"],
        "boot_kind": boot["kind"],
        "boot": boot,
        "fresh_tokens": metrics.fresh_tokens,
        "cached_tokens": metrics.cached_tokens,
        "peak_gib": round(peak_bytes / 2**30, 2),
        "node_metrics": scalar_node_metrics(run.nodes),
        "backend_metrics": dict(request_result.metrics.extension),
    }
    return PhysicalResponse(
        export_physical_outputs(compute_graph, run),
        report,
    )


def request_runtimes() -> dict:
    """Return runtimes for the request backends' physical node."""
    return {RequestExecution.runtime_key: ModelNodeRuntime()}


SUPPORTED_DEVICES = frozenset({"h100-sxm", "rtx-pro-6000-blackwell-server"})


def _warm_boot() -> dict:
    return {
        "kind": "warm",
        "llm_init_s": 0.0,
        "weight_load_s": None,
        "kv_profile_s": None,
        "boot_s": 0.0,
    }


@dataclass(frozen=True)
class RequestBackend:
    """A model backend that submits one request per prompt to an engine.

    The engine adapter boots vLLM or SGLang and returns a client with
    generate, reset_prefix_cache, and run_filter_chain. The submission
    strategies decide how filter stages and join tuples become
    requests.
    """

    name: str
    engine: Any
    filter_submission: str
    join_submission: str = "anchor-major"

    @property
    def runtime_package(self) -> str:
        return self.engine.runtime_package

    def supports(self, model, device, gpu_count: int) -> SupportResult:
        label = self.engine.label
        roles = getattr(self.engine, "roles", frozenset({"generative"}))
        if model.role not in roles:
            return SupportResult.reject(
                f"{label} does not support model {model.name!r}: "
                f"it serves {', '.join(sorted(roles))} models"
            )
        if device.name not in SUPPORTED_DEVICES:
            return SupportResult.reject(
                f"{label} does not support device {device.name!r}"
            )
        if gpu_count != 1:
            return SupportResult.reject(
                f"the {label} request backends use one model copy on one GPU"
            )
        return SupportResult.accept()

    def plan(self, region, context):
        return plan_request_backend(
            region,
            context,
            backend_name=self.name,
            filter_submission=self.filter_submission,
            join_submission=self.join_submission,
            scores_labels=getattr(self.engine, "scores_labels", False),
        )

    def start(self, context):
        return RequestModelExecution(context)

    def execute_request(self, context):
        envelope = context.request.plan
        model = context.registry.model(envelope["model"])
        state_key = ("request-engine", self.engine.kind, model.name)
        engine_state = context.runtime_state.get(state_key)
        if engine_state is None:
            allowed_ids = sorted(set(
                envelope["settings"]["true_ids"]
            ) | set(envelope["settings"]["false_ids"]))
            engine_state, boot = self.engine.boot(model, allowed_ids)
            context.runtime_state[state_key] = engine_state
        else:
            boot = _warm_boot()
        return execute_request_graph(context, self, engine_state, boot)
