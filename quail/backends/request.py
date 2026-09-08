"""Shared planning and execution for request based model backends."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any, Mapping

import pyarrow as pa

from quail.backends.base import GpuContext
from quail.backends.request_scheduling import (
    join_cache_accounting,
    longest_common_prefix,
    run_join_grouped,
    split_cached_tokens,
    true_bit,
)
from quail.execution import PhysicalResponse, export_physical_outputs
from quail.logical import SHARED_PRE
from quail.physical import (
    DocumentInput,
    Limit,
    PhysicalNode,
    PortRef,
    Project,
    Recombine,
    RequestExecution,
    RequestFilterSpec,
    RequestJoinSpec,
)
from quail.physical.base import input_ports
from quail.planner import (
    budgets,
    collect_operators,
    default_order_rule,
    order_filters_indexed,
    preamble_tokens,
)
from quail.planner import (
    join_specs as logical_join_specs,
)
from quail.planner.joins import search_joins, summarize_alias
from quail.planner.plan import CorpusStats, PhysicalPlan
from quail.planning import PhysicalCandidate, SupportResult
from quail.runtime.result import answer_table
from quail.runtime.runner import (
    ExecutionContext,
    GenericRunner,
    ModelNodeRuntime,
    NodeMetrics,
    NodeResult,
    compute_subgraph,
    scalar_node_metrics,
)


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
) -> tuple[PhysicalCandidate, ...]:
    """Build one physical request plan for a request engine."""
    scans, filters, joins = collect_operators(region.logical_plan)
    stats = {
        alias: CorpusStats(
            n_docs=len(lengths),
            total_tokens=sum(int(length) for length in lengths),
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
        node = DocumentInput(
            node_id=f"input:{scan.alias}",
            alias=scan.alias,
            input_id=scan.alias,
            n_docs=summary.n_docs,
            total_tokens=summary.total_tokens,
            shard_ranges=((0, summary.n_docs),),
            shard_token_loads=(summary.total_tokens,),
        )
        nodes.append(node)
        input_refs.append(PortRef(node.node_id, f"ids:{scan.alias}"))

    rule = context.order or default_order_rule(filters, joins)[0]
    chunk_tokens = budgets.chunk_budget(context.model, context.device)
    shared_preamble = preamble_tokens(filters, joins)
    filter_orders = {
        alias: order_filters_indexed(
            predicates,
            rule,
            prefix_tokens=(
                shared_preamble + stats[alias].mean_doc_tokens
            ),
            model=context.model,
            device=context.device,
            chunk_tokens=chunk_tokens,
        )
        for alias, predicates in filters.items()
    }
    live = {
        alias: float(summary.n_docs) for alias, summary in stats.items()
    }
    for alias, predicates in filters.items():
        for predicate in predicates:
            live[alias] *= (
                predicate.selectivity
                if predicate.selectivity is not None else 1.0
            )
    search_specs = logical_join_specs(joins)
    join_search = search_joins(
        search_specs,
        live,
        {
            alias: summarize_alias(lengths)
            for alias, lengths in context.document_tokens.items()
        },
        {},
        shared_preamble,
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
        if not predicates:
            continue
        prompts.extend(predicate.prompt for predicate in predicates)
        written_positions = tuple(filter_orders[alias])
        questions = tuple(
            tuple(predicates[position].prompt.tail_token_ids)
            for position in written_positions
        )
        if any(not question for question in questions):
            raise ValueError("filter prompts have no token ids")
        filter_specs.append(RequestFilterSpec(
            alias=alias,
            written_positions=written_positions,
            question_token_ids=questions,
        ))

    join_specs = []
    for written_pos, selected_anchor in join_sequence:
        join = joins[written_pos]
        prompt = join.predicate
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
        if len(labels) != len(aliases_in_prompt) or not prompt.tail_token_ids:
            raise ValueError("join prompts have no token ids")
        outer_aliases = tuple(dict.fromkeys(
            field.alias for field in join.inputs[0].output_schema()
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
        ))

    preambles = {
        tuple(prompt.preamble_token_ids)
        for prompt in prompts
        if prompt.preamble_token_ids
    }
    if len(preambles) > 1:
        raise ValueError("request prompts have different preambles")
    preamble = next(iter(preambles), ())
    if not preamble and prompts and context.tokenizer is not None:
        preamble = tuple(context.tokenizer(SHARED_PRE))

    request_node = RequestExecution(
        node_id="request-model",
        inputs=input_ports(tuple(input_refs)),
        backend_name=backend_name,
        aliases=aliases,
        preamble_token_ids=tuple(preamble),
        filters=tuple(filter_specs),
        joins=tuple(join_specs),
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

    nodes.append(Project(
        node_id="sink",
        inputs=input_ports((sink_input,)),
        columns=tuple(
            f"{column.alias}.{column.column}"
            for column in region.logical_plan.root.columns
        ),
    ))
    if region.logical_plan.root.limit is not None:
        nodes.append(Limit(
            node_id="limit",
            inputs=input_ports((PortRef("sink", "rows"),)),
            count=region.logical_plan.root.limit,
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
        },
    )
    return (PhysicalCandidate(
        graph=plan.graph,
        plan=plan,
        estimated_seconds=plan.estimated_seconds,
    ),)


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


def _operator_at_a_time_filter(client, sampling_params, bodies, questions,
                               true_ids, block_size):
    active = list(range(len(bodies)))
    answers = {}
    prior = {}
    requests = prompt_tokens = cached_tokens = 0
    regret_tokens = cross_row_cached = 0
    cached_own = cached_other = 0
    started = time.perf_counter()
    stages = []
    for stage_index, question in enumerate(questions):
        evaluated = list(active)
        prompts = [
            {"prompt_token_ids": bodies[index] + list(question)}
            for index in evaluated
        ]
        outputs = (
            client.generate(prompts, sampling_params, use_tqdm=False)
            if prompts else []
        )
        next_active = []
        for document, prompt, output in zip(evaluated, prompts, outputs):
            answer = bool(true_bit(output, true_ids))
            answers[(document, stage_index)] = answer
            requests += 1
            prompt_tokens += len(output.prompt_token_ids)
            cached = int(getattr(output, "num_cached_tokens", 0) or 0)
            cached_tokens += cached
            # the document's own earlier requests computed a prefix this
            # one could hit; cached tokens beyond it but inside the body
            # came from another document sharing a prefix
            could_hit = max(
                (
                    longest_common_prefix(prompt["prompt_token_ids"], earlier)
                    for earlier in prior.get(document, ())
                ),
                default=0,
            )
            could_hit = (could_hit // block_size) * block_size
            regret_tokens += max(0, could_hit - cached)
            own, shared, other = split_cached_tokens(
                cached, could_hit, 0, len(bodies[document]))
            cached_own += own
            cross_row_cached += shared
            cached_other += other
            if answer:
                next_active.append(document)
                prior.setdefault(document, []).append(prompt["prompt_token_ids"])
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
        "prior": prior,
        "requests": requests,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "fresh_tokens": prompt_tokens - cached_tokens,
        "regret_tokens": regret_tokens,
        "cross_row_cached_tokens": cross_row_cached,
        "cached_own_tokens": cached_own,
        "cached_other_tokens": cached_other,
        "stages": stages,
        "doc_cap": None,
    }


def _pipelined_filter(client, sampling_params, bodies, questions, true_ids,
                      tag):
    if not bodies:
        return {
            "wall_s": 0.0,
            "survivors": [],
            "answers": {},
            "prior": {},
            "requests": 0,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "fresh_tokens": 0,
            "regret_tokens": 0,
            "cross_row_cached_tokens": 0,
            "cached_own_tokens": 0,
            "cached_other_tokens": 0,
            "stages": [],
            "doc_cap": 0,
        }
    result = client.run_filter_chain(
        sampling_params,
        bodies,
        [list(question) for question in questions],
        true_ids,
        tag=tag,
    )
    prior = {}
    for (document, stage), answer in result["answers"].items():
        if answer:
            prior.setdefault(document, []).append(
                bodies[document] + list(questions[stage - 1])
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
        "prior": prior,
        "requests": result["requests"],
        "prompt_tokens": result["prompt_tokens"],
        "cached_tokens": result["cached_tokens"],
        "fresh_tokens": result["prompt_tokens"] - result["cached_tokens"],
        "regret_tokens": int(result.get("regret_tokens") or 0),
        "cross_row_cached_tokens": int(
            result.get("cross_row_cached_tokens") or 0
        ),
        "cached_own_tokens": int(result.get("cached_own_tokens") or 0),
        "cached_other_tokens": int(result.get("cached_other_tokens") or 0),
        "stages": stages,
        "doc_cap": result["doc_cap"],
    }


class RequestModelExecution:
    """Execute one request engine node with one loaded engine."""

    def __init__(self, context: GpuContext):
        settings = context.query_settings
        self.client = settings["client"]
        self.sampling_params = settings["sampling_params"]
        self.documents = settings["documents"]
        self.true_ids = set(settings["true_ids"])
        self.capacity = settings["capacity"]
        self.filter_submission = settings["filter_submission"]
        self.join_submission = settings["join_submission"]

    def execute(
        self,
        node: PhysicalNode,
        inputs: Mapping[str, Any],
    ) -> NodeResult:
        if not isinstance(node, RequestExecution):
            raise TypeError(
                f"request backend cannot execute {node.type_name!r}"
            )
        del inputs
        started = time.perf_counter()
        survivors = {
            alias: list(range(len(self.documents[alias])))
            for alias in node.aliases
        }
        prior = {}
        outputs = {}
        steps = []
        fresh_tokens = cached_tokens = requests = 0
        evaluated_documents = evaluated_pairs = regret_tokens = 0
        cross_row_cached = 0
        block_size = int(self.capacity["block_size"])

        for filter_index, spec in enumerate(node.filters):
            document_ids = list(survivors[spec.alias])
            bodies = [
                _token_list(node.preamble_token_ids)
                + _token_list(self.documents[spec.alias][document])
                for document in document_ids
            ]
            if self.filter_submission == "operator-at-a-time":
                result = _operator_at_a_time_filter(
                    self.client,
                    self.sampling_params,
                    bodies,
                    spec.question_token_ids,
                    self.true_ids,
                    block_size,
                )
            elif self.filter_submission == "pipelined":
                result = _pipelined_filter(
                    self.client,
                    self.sampling_params,
                    bodies,
                    spec.question_token_ids,
                    self.true_ids,
                    f"filter-{filter_index}-{spec.alias}",
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
            for local, prompts in result["prior"].items():
                prior.setdefault(
                    (spec.alias, document_ids[local]), []
                ).extend(prompts)
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
                "regret_tokens": result["regret_tokens"],
                "cross_row_cached_tokens": result["cross_row_cached_tokens"],
                "cached_own_tokens": result["cached_own_tokens"],
                "cached_other_tokens": result["cached_other_tokens"],
                "doc_cap": result["doc_cap"],
            })
            requests += result["requests"]
            fresh_tokens += result["fresh_tokens"]
            cached_tokens += result["cached_tokens"]
            regret_tokens += result["regret_tokens"]
            cross_row_cached += result["cross_row_cached_tokens"]
            evaluated_documents += result["requests"]

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
            document_spans = [
                (len(preamble),
                 len(preamble) + len(self.documents[anchor][document]))
                for document in anchor_ids
            ]
            members = list(itertools.product(
                *(alias_documents[alias] for alias in partners)
            ))
            suffixes = []
            for member in members:
                suffix = []
                for alias, document in zip(partners, member):
                    suffix.extend(_token_list(labels[alias]))
                    suffix.extend(_token_list(
                        self.documents[alias][document]
                    ))
                suffix.extend(_token_list(spec.tail_token_ids))
                suffixes.append(suffix)

            if prefixes and suffixes:
                result = run_join_grouped(
                    self.client,
                    self.sampling_params,
                    prefixes,
                    suffixes,
                    self.true_ids,
                    submission=self.join_submission,
                )
                answers = [bool(answer) for answer in result["answers"]]
                seen_lengths = [
                    max(
                        (
                            longest_common_prefix(prefix, earlier)
                            for earlier in prior.get((anchor, document), ())
                        ),
                        default=0,
                    )
                    for document, prefix in zip(anchor_ids, prefixes)
                ]
                accounting = join_cache_accounting(
                    prefixes,
                    len(suffixes),
                    result["cached_per_request"],
                    seen_lengths,
                    block_size,
                    document_spans,
                )
                regret = accounting["regret_tokens"]
                cross_row = accounting["cross_row_cached_tokens"]
                cached_own = accounting["cached_own_tokens"]
                cached_other = accounting["cached_other_tokens"]
            else:
                result = {
                    "wall": 0.0,
                    "fresh_tokens": 0,
                    "cached_tokens": 0,
                    "prompt_tokens": 0,
                }
                answers = []
                regret = 0
                cross_row = 0
                cached_own = 0
                cached_other = 0

            rows = []
            for anchor_id in anchor_ids:
                for member in members:
                    by_alias = {anchor: anchor_id}
                    by_alias.update(zip(partners, member))
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
            for document, prefix in zip(anchor_ids, prefixes):
                if document in survivors[anchor]:
                    prior.setdefault((anchor, document), []).append(prefix)
            steps.append({
                "kind": "join",
                "written_pos": spec.written_pos,
                "anchor": anchor,
                "partners": list(partners),
                "submission": self.join_submission,
                "pairs": len(rows),
                "true_pairs": len(true_rows),
                "wall_s": result["wall"],
                "fresh_tokens": result["fresh_tokens"],
                "cached_tokens": result["cached_tokens"],
                "regret_tokens": regret,
                "cross_row_cached_tokens": cross_row,
                "cached_own_tokens": cached_own,
                "cached_other_tokens": cached_other,
            })
            requests += len(rows)
            evaluated_pairs += len(rows)
            fresh_tokens += result["fresh_tokens"]
            cached_tokens += result["cached_tokens"]
            regret_tokens += regret
            cross_row_cached += cross_row

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
                regret_tokens=regret_tokens,
                extension={
                    "steps": steps,
                    "requests": requests,
                    "cross_row_cached_tokens": cross_row_cached,
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
        if isinstance(node, DocumentInput)
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
                alias: range(len(value))
                for alias, value in documents.items()
            },
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
        "regret_tokens": metrics.regret_tokens,
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


SUPPORTED_MODELS = frozenset({"qwen3-4b-fp8", "qwen3-32b-fp8"})
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
        if model.name not in SUPPORTED_MODELS:
            return SupportResult.reject(
                f"{label} does not support model {model.name!r}"
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
            engine_state, boot = self.engine.boot(model.hf_name, allowed_ids)
            context.runtime_state[state_key] = engine_state
        else:
            boot = _warm_boot()
        return execute_request_graph(context, self, engine_state, boot)
