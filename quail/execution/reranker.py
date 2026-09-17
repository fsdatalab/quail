"""Execute AI.SCORE expressions with vLLM reranker models."""

import gc
import itertools
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import pyarrow as pa
from pyarrow import compute as pc

from quail.backends.base import GpuContext
from quail.execution.result import answer_table
from quail.execution.runner import (
    ExecutionContext,
    GenericRunner,
    NodeMetrics,
    NodeResult,
    compute_subgraph,
    scalar_node_metrics,
)
from quail.execution.types import PhysicalResponse, export_physical_outputs
from quail.physical import AiScore, ScoreFilter, ValueType


def normalized_yes_score(logit_difference: float) -> float:
    """Return the stable two-class softmax probability for YES."""
    if logit_difference >= 0:
        return 1.0 / (1.0 + math.exp(-logit_difference))
    value = math.exp(logit_difference)
    return value / (1.0 + value)


def compare_score(score: float, comparison: str, threshold: float) -> bool:
    """Evaluate one supported AI.SCORE comparison."""
    if comparison == "<":
        return score < threshold
    if comparison == "<=":
        return score <= threshold
    if comparison == ">":
        return score > threshold
    if comparison == ">=":
        return score >= threshold
    raise ValueError(f"unsupported AI.SCORE comparison {comparison!r}")


@dataclass(frozen=True)
class RerankerBatch:
    """Scores and token counts returned by one reranker call."""

    scores: tuple[float, ...]
    fresh_tokens: int
    cached_tokens: int


class RerankerModel(Protocol):
    """Model interface used by AI.SCORE execution."""

    def score(self, prompts) -> RerankerBatch:
        """Score complete tokenized reranker prompts."""


class Qwen3VllmReranker:
    """Qwen3 reranker served by the vLLM pooling runner."""

    def __init__(self, llm):
        self.llm = llm

    def score(self, prompts) -> RerankerBatch:
        """Return normalized YES scores for tokenized prompts."""
        from vllm import PoolingParams

        outputs = self.llm.encode(
            [{"prompt_token_ids": [int(token) for token in prompt]}
             for prompt in prompts],
            pooling_task="classify",
            use_tqdm=False,
            pooling_params=PoolingParams(use_activation=False),
        )
        prompt_tokens = sum(
            len(getattr(output, "prompt_token_ids", ()))
            for output in outputs
        )
        cached_tokens = sum(
            int(getattr(output, "num_cached_tokens", 0) or 0)
            for output in outputs
        )
        return RerankerBatch(
            scores=tuple(
                normalized_yes_score(float(output.outputs.data.item()))
                for output in outputs
            ),
            fresh_tokens=prompt_tokens - cached_tokens,
            cached_tokens=cached_tokens,
        )

    def close(self) -> None:
        """Release the vLLM engine when another model must load."""
        engine = getattr(self.llm, "llm_engine", None)
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()


def prepare_reranker_request(context) -> None:
    """Load a reranker before the request's document columns arrive."""
    state, boot = _loaded_reranker(context)
    state["prepared_boot"] = boot


def _score_table(rows, aliases, name, scores) -> pa.Table:
    arrays = {
        alias: pa.array(
            (row[index] for row in rows),
            type=pa.int32(),
        )
        for index, alias in enumerate(aliases)
    }
    arrays[name] = pa.array(scores, type=pa.float64())
    return pa.table(arrays).replace_schema_metadata({
        b"quail.kind": b"score_rows",
        b"quail.aliases": ",".join(aliases).encode("utf-8"),
    })


def _ids(value, alias: str) -> list[int]:
    if isinstance(value, pa.Table):
        return [int(item) for item in value.column(alias).to_pylist()]
    return [int(item) for item in value]


class RerankerModelExecution:
    """Compute score columns without applying their comparisons."""

    def __init__(self, context: GpuContext):
        self.reranker = context.query_settings["reranker"]
        self.documents = context.query_settings["documents"]

    @staticmethod
    def _sources(node, inputs) -> list[tuple[object, object]]:
        return [
            (port, inputs[port.name])
            for port in node.inputs
        ]

    def _candidate_rows(self, node, inputs):
        spec = node.spec
        sources = self._sources(node, inputs)
        by_alias = {}
        pairs = None
        prior = None
        for port, value in sources:
            if port.value_type is ValueType.PAIRS:
                pairs = value
                continue
            if port.value_type is ValueType.SCORES:
                prior = value if len(spec.aliases) == 1 else prior
                for alias in spec.aliases:
                    if alias in value.column_names:
                        by_alias[alias] = _ids(value, alias)
                continue
            if port.source.port.startswith("ids:"):
                alias = port.source.port.split(":", 1)[1]
                by_alias[alias] = _ids(value, alias)

        if len(spec.aliases) == 1:
            alias = spec.aliases[0]
            rows = [(document,) for document in by_alias[alias]]
            return rows, prior

        left, right = spec.aliases
        allowed_left = set(by_alias[left])
        allowed_right = set(by_alias[right])
        if pairs is None:
            rows = list(itertools.product(by_alias[left], by_alias[right]))
        else:
            rows = [
                (int(a), int(b))
                for a, b in zip(
                    pairs.column(left).to_pylist(),
                    pairs.column(right).to_pylist(),
                )
                if a in allowed_left and b in allowed_right
            ]
        return rows, None

    def execute(
        self,
        node: AiScore,
        inputs: Mapping[str, object],
    ) -> NodeResult:
        if not isinstance(node, AiScore):
            raise TypeError(type(node).__name__)
        if node.spec is None:
            raise ValueError("AI.SCORE needs a score specification")
        started = time.perf_counter()
        spec = node.spec
        rows, prior = self._candidate_rows(node, inputs)
        if len(spec.arguments) != len(spec.aliases):
            raise ValueError(
                "AI.SCORE needs one prompt argument per document relation"
            )

        parts = spec.prompt_token_parts
        if len(parts) != len(spec.aliases) + 1:
            raise ValueError("AI.SCORE needs tokenized prompt parts")
        prompts = []
        for row in rows:
            tokens = list(parts[0])
            for index, alias in enumerate(spec.aliases):
                tokens.extend(self.documents[alias][row[index]])
                tokens.extend(parts[index + 1])
            prompts.append(tokens)
        batch = (
            self.reranker.score(prompts)
            if rows else RerankerBatch((), 0, 0)
        )
        if prior is None:
            table = _score_table(rows, spec.aliases, spec.name, batch.scores)
        else:
            table = prior.append_column(
                spec.name,
                pa.array(batch.scores, type=pa.float64()),
            )
        count = len(rows)
        return NodeResult(
            {"scores": table},
            NodeMetrics(
                wall_s=time.perf_counter() - started,
                input_rows=count,
                output_rows=count,
                evaluated_documents=(
                    count if len(spec.aliases) == 1 else 0
                ),
                evaluated_document_pairs=(
                    count if len(spec.aliases) == 2 else 0
                ),
                fresh_tokens=batch.fresh_tokens,
                cached_tokens=batch.cached_tokens,
                extension={
                    "output": spec.name,
                    "aliases": list(spec.aliases),
                    "input_rows": count,
                },
            ),
        )


def _filter_answer_table(node: ScoreFilter, table, answers) -> pa.Table:
    alias = node.aliases[0]
    return pa.table({
        alias: pc.cast(table.column(alias), pa.int32()),
        "predicate": pa.array(
            [node.written_pos] * table.num_rows, type=pa.int32()
        ),
        "answer": pa.array(answers, type=pa.bool_()),
    }).replace_schema_metadata({
        b"quail.kind": b"filter_answers",
        b"quail.alias": alias.encode("utf-8"),
    })


class ScoreFilterRuntime:
    """Apply a numeric score comparison and retain the score column."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, ScoreFilter):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("ScoreFilter needs one score input")
        table = next(iter(inputs.values()))
        scores = table.column(node.score_name).to_pylist()
        answers = [
            compare_score(float(score), node.comparison, node.threshold)
            for score in scores
        ]
        mask = pa.array(answers, type=pa.bool_())
        filtered = table.filter(mask)
        if len(node.aliases) == 1:
            answer_name = f"filter_answers:{node.aliases[0]}"
            answer_relation = _filter_answer_table(node, table, answers)
        else:
            left, right = node.aliases
            answer_name = f"join_answers:{node.written_pos}"
            answer_relation = answer_table(
                {
                    left: table.column(left).to_pylist(),
                    right: table.column(right).to_pylist(),
                },
                answers,
                "join_answers",
                metadata={
                    "written_pos": node.written_pos,
                    "anchor": left,
                    "partners": right,
                    "semantics": "full",
                    "comparison": node.comparison,
                    "threshold": node.threshold,
                },
            )
        return NodeResult(
            {
                "scores": filtered,
                answer_name: answer_relation,
            },
            NodeMetrics(
                input_rows=table.num_rows,
                output_rows=filtered.num_rows,
            ),
        )


def execute_reranker_request(context, *, backend_name: str):
    """Run the reranker compute graph and export its Arrow outputs."""
    state, boot = _loaded_reranker(context)
    boot = state.pop("prepared_boot", boot)
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    envelope = context.request.plan
    documents = {
        node.alias: context.request.inputs[node.input_id].documents
        for node in context.graph.nodes
        if node.type_name == "quail.scan"
    }
    model_execution = RerankerModelExecution(GpuContext(
        gpu_index=0,
        gpu_count=context.gpu_count,
        model=context.registry.model(envelope["model"]),
        device=context.registry.device(envelope["device"]),
        query_settings={
            "reranker": state["reranker"],
            "documents": documents,
        },
    ))
    compute_graph = compute_subgraph(context.graph)
    run = GenericRunner().run(
        compute_graph,
        ExecutionContext(
            runtimes=context.registry.runtimes,
            model_execution=model_execution,
            sources={
                **{
                    alias: range(len(value))
                    for alias, value in documents.items()
                },
                **context.request.relations,
            },
        ),
    )
    metrics = run.metrics
    wall_s = metrics.wall_s
    evaluated = (
        metrics.evaluated_document_pairs or metrics.evaluated_documents
    )
    device = context.registry.device(envelope["device"])
    backend_metrics = [
        dict(result.metrics.extension)
        for node_id, result in run.nodes.items()
        if compute_graph.node(node_id).type_name == AiScore.type_name
    ]
    report = {
        "backend": backend_name,
        "wall_s": wall_s,
        "boot_s": boot["boot_s"],
        "boot_kind": boot["kind"],
        "boot": boot,
        "fresh_tokens": metrics.fresh_tokens,
        "cached_tokens": metrics.cached_tokens,
        "peak_gib": (
            round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if torch is not None and torch.cuda.is_available() else 0.0
        ),
        "node_metrics": scalar_node_metrics(run.nodes),
        "backend_metrics": {"scores": backend_metrics},
        "evaluated_documents": metrics.evaluated_documents,
        "evaluated_document_pairs": metrics.evaluated_document_pairs,
        "usd_per_query": (
            wall_s / 3600 * context.request.gpu_count
            * (device.usd_per_hour or 0.0)
        ),
    }
    if metrics.evaluated_document_pairs:
        report["document_pairs_per_second"] = evaluated / max(wall_s, 1e-9)
    else:
        report["documents_per_second"] = evaluated / max(wall_s, 1e-9)
    return PhysicalResponse(
        export_physical_outputs(compute_graph, run),
        report,
    )


def _loaded_reranker(context):
    model = context.registry.model(context.request.plan["model"])
    state_key = ("qwen3-reranker", model.name, context.gpu_count)
    state = context.runtime_state.get(state_key)
    if state is not None:
        return state, {
            "kind": "warm",
            "llm_init_s": 0.0,
            "weight_load_s": None,
            "kv_profile_s": None,
            "boot_s": 0.0,
        }
    _release_other_rerankers(context.runtime_state, keep=state_key)
    state, boot = _boot(model, context.gpu_count)
    context.runtime_state[state_key] = state
    return state, boot


def _release_other_rerankers(runtime_state, *, keep) -> None:
    """Release a different reranker before loading the requested model."""
    keys = [
        key for key in runtime_state
        if isinstance(key, tuple)
        and key[:1] == ("qwen3-reranker",)
        and key != keep
    ]
    for key in keys:
        state = runtime_state.pop(key)
        reranker = state.pop("reranker", None)
        if reranker is not None:
            reranker.close()
        del reranker
        del state
    if not keys:
        return
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _boot(model, gpu_count: int):
    from vllm import LLM

    started = time.perf_counter()
    llm = LLM(
        model=model.hf_name,
        revision=model.revision,
        runner="pooling",
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        disable_log_stats=True,
        data_parallel_size=gpu_count,
    )
    boot_s = time.perf_counter() - started
    return (
        {"reranker": Qwen3VllmReranker(llm)},
        {
            "kind": "cold",
            "llm_init_s": round(boot_s, 2),
            "weight_load_s": None,
            "kv_profile_s": None,
            "boot_s": round(boot_s, 2),
        },
    )
