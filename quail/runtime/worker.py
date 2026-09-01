"""Modal worker: runs filter chains and join stages on the GPU,
returns raw answer rows.
"""

import gc
import itertools
import json
import os
import time

import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "numpy", "pyarrow")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          # JIT artifacts persist on the kernel-cache volume so each
          # DeepGEMM/Triton configuration compiles once ever, not
          # once per container
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
    .add_local_python_source("quail")
)

# House rule: never create new Modal app names - caches and warm state
# ride on the app. The engine worker lives here, permanently.
app = modal.App("quail-engine")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)


# Process-global state: the container IS the session-side cache. A
# warm container keeps the loaded model and the arena across
# execute() calls, which is what makes a session's later queries boot
# in milliseconds.
_BOOTED = {}      # model name -> dict(model, arena, pipeline)


def validate_plan_payload(payload: dict) -> None:
    """Validate the typed plan before any model call."""
    from quail.extensions import built_in_registry
    from quail.physical import check_plan_envelope, decode_graph

    envelope = payload.get("physical_plan")
    if envelope is None:
        raise ValueError("worker payload has no typed physical plan")
    check_plan_envelope(envelope)
    if envelope["backend"] != "quail":
        raise ValueError(
            f"worker does not have backend {envelope['backend']!r}")
    if envelope["model"] != payload["model"]:
        raise ValueError("physical plan and payload name different models")
    if envelope["workers"] != payload["workers"]:
        raise ValueError("physical plan and payload name different GPU counts")
    registry = built_in_registry()
    missing = set(envelope["node_types"]) - set(registry.codecs)
    if missing:
        raise ValueError(
            f"worker does not have physical node codecs {sorted(missing)}")
    graph = decode_graph(envelope["graph"], registry.codecs)
    graph.validate(runtime_keys=set(registry.runtimes))
    graph.validate_backend(envelope["backend"])


def _release_vllm_parallel_state() -> None:
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()


def release_booted_models() -> dict:
    """Release Quail GPU state before another engine uses this process."""
    import torch

    released = len(_BOOTED)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _BOOTED.clear()
    _release_vllm_parallel_state()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return {
        "models_released": released,
        "vllm_parallel_state_released": True,
        "cuda_allocated_bytes": (
            int(torch.cuda.memory_allocated())
            if torch.cuda.is_available() else 0),
        "cuda_reserved_bytes": (
            int(torch.cuda.memory_reserved())
            if torch.cuda.is_available() else 0),
    }


def _execute_payload(payload: dict) -> dict:
    import torch
    import torch.nn.functional as F

    from quail.executor.arena import KVArena
    from quail.executor.attention import FILTER_ATTENTION, Pipeline
    from quail.executor.loop import (
        AsyncAnswers,
        warm_kernels,
    )
    from quail.planner import budgets
    from quail.specs import DEVICES, MODELS

    validate_plan_payload(payload)
    if payload["kv_dtype"] != "bf16":
        raise ValueError(
            f"KV is always bf16; got {payload['kv_dtype']!r}")

    spec = MODELS[payload["model"]]
    device = DEVICES["h100-sxm"]

    t_boot = time.perf_counter()
    boot = dict(kind="warm", load_model_s=0.0, arena_s=0.0,
                pipeline_s=0.0, warm_kernels_s=0.0)
    booted = _BOOTED.get(spec.name)
    if booted is None:
        from quail.executor.model import load_model
        t0 = time.perf_counter()
        model = load_model(spec.hf_name)
        boot["load_model_s"] = time.perf_counter() - t0
        # budgets.* is tiny CPU; fold into arena_s so the four phases
        # cover the cold-load span without a leftover residual
        t0 = time.perf_counter()
        chunk_tokens = budgets.chunk_budget(spec, device)
        arena_tok = budgets.arena_tokens(spec, device, chunk_tokens)
        arena = KVArena(n_layers=spec.layers,
                        n_pages=arena_tok // budgets.PAGE_TOKENS,
                        page_tokens=budgets.PAGE_TOKENS,
                        n_kv=spec.n_kv, d_head=spec.d_head,
                        dtype=torch.bfloat16)
        boot["arena_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pipeline = Pipeline(model, arena,
                            attention_mode=FILTER_ATTENTION)
        boot["pipeline_s"] = time.perf_counter() - t0
        from quail.backends import GpuContext, QuailBackend
        execution = QuailBackend().start(GpuContext(
            gpu_index=0,
            gpu_count=payload["workers"],
            model=spec,
            device=device,
            query_settings={
                "chunk_tokens": payload["chunk_tokens"],
                "kv_dtype": payload["kv_dtype"],
            },
        ))
        execution.bind_loaded_model(
            model=model, arena=arena, pipeline=pipeline
        )
        booted = dict(execution=execution, warmed=False)
        _BOOTED[spec.name] = booted
        boot["kind"] = "cold"
    execution = booted["execution"]
    model = execution.state["model"]
    arena = execution.state["arena"]
    pipeline = execution.state["pipeline"]
    # the worker has no tokenizer: the TRUE/FALSE token ids ride in the
    # payload
    answerer = _PayloadAnswerer(torch, F, model, payload["true_ids"],
                                payload["false_ids"])
    async_ans = AsyncAnswers(torch, answerer)
    chunk_tokens = payload["chunk_tokens"]
    execution.bind_query(
        torch=torch,
        async_answers=async_ans,
        chunk_tokens=chunk_tokens,
    )

    if not booted["warmed"]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            # boot-side warmup on synthetic tokens: the compile pass
            # once ever per stack+model+budget (marker on the kernel
            # cache volume), the millisecond-loads touch pass on
            # every container after that
            warm = warm_kernels(torch, arena, pipeline, async_ans,
                                chunk_tokens,
                                model_name=spec.hf_name)
        torch.cuda.synchronize()
        kernel_cache.commit()   # keep the compiles even if the run dies
        boot["warm_kernels_s"] = time.perf_counter() - t0
        boot["warm_tier"] = warm["tier"]
        booted["warmed"] = True
        boot["kind"] = "cold"

    boot_s = time.perf_counter() - t_boot   # load + compile
    for k in ("load_model_s", "arena_s", "pipeline_s",
              "warm_kernels_s"):
        boot[k] = round(boot[k], 2)
    boot["boot_s"] = round(boot_s, 2)
    state = dict(model_execution=execution,
                 model=model, arena=arena, pipeline=pipeline,
                 spec=spec, torch=torch, F=F)
    report = _execute_single(state, payload)
    report["boot_s"] = boot["boot_s"]
    report["boot_kind"] = boot["kind"]
    report["boot"] = boot
    os.makedirs("/results/runs", exist_ok=True)
    result_path = f"/results/runs/run_{time.time_ns()}.json"
    report["result_volume_path"] = result_path
    with open(result_path, "w") as f:
        json.dump(dict(
            wall_s=report["wall_s"], boot_s=report["boot_s"],
            boot_kind=report["boot_kind"], boot=boot,
            fresh_tokens=report["fresh_tokens"],
            regret_tokens=report.get("regret_tokens"),
            join_optimizer=report.get("join_optimizer"),
            kv_manager=report.get("kv_manager")), f)
    results_vol.commit()
    kernel_cache.commit()    # persist any JIT artifacts this run built
    return report


@app.function(image=image, gpu="H100!", timeout=7200, memory=98304,
              scaledown_window=300, max_containers=1,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def execute(payload: dict) -> dict:
    return _execute_payload(payload)


def _execute_single(state, payload: dict) -> dict:
    """Execute the typed Quail graph on one GPU."""
    from quail.extensions import built_in_registry
    from quail.physical import decode_graph
    from quail.runtime.quail_graph import execute_single_graph
    from quail.runtime.tokens import decode_payload_documents
    from quail.specs import DEVICES

    registry = built_in_registry()
    graph = decode_graph(
        payload["physical_plan"]["graph"], registry.codecs
    )
    graph.validate(runtime_keys=set(registry.runtimes))
    runtime_state = {
        **state,
        "docs": decode_payload_documents(payload["docs"]),
        "runtimes": registry.runtimes,
        "model_spec": state["spec"],
        "device": DEVICES["h100-sxm"],
        "chunk_tokens": payload["chunk_tokens"],
    }
    return execute_single_graph(runtime_state, payload, graph)


# ------------------------------------------------- multi-GPU dispatch
#
# One executor per GPU, as its own child process (its own CUDA
# context and arena). The parent is the in-container coordinator: it
# splits the payload with quail.runtime.coordinator, runs the filter
# round, merges survivors, runs the join round, and merges the
# answers - no network hop anywhere between rounds. Children persist
# across execute calls, so their models stay loaded for the whole
# session.

_CHILDREN = []      # [(process, connection)] in GPU order


def _child_main(gpu_idx, conn):
    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    state = {"gpu_index": gpu_idx}
    while True:
        try:
            kind, data = conn.recv()
        except EOFError:
            break
        if kind == "shutdown":
            break
        try:
            if kind == "filters":
                conn.send(("ok", _child_filters(state, data)))
            elif kind == "joins":
                conn.send(("ok", _child_joins(state, data)))
        except Exception:
            import traceback
            conn.send(("err", traceback.format_exc()))


def _child_boot(state, sub):
    import torch
    import torch.nn.functional as F

    from quail.executor.arena import KVArena
    from quail.executor.attention import FILTER_ATTENTION, Pipeline
    from quail.executor.loop import AsyncAnswers, warm_kernels
    from quail.executor.model import load_model
    from quail.planner import budgets
    from quail.specs import DEVICES, MODELS

    spec = MODELS[sub["model"]]
    device = DEVICES["h100-sxm"]
    boot = dict(kind="warm", load_model_s=0.0, arena_s=0.0,
                pipeline_s=0.0, warm_kernels_s=0.0)
    t_boot = time.perf_counter()
    if "model_execution" not in state:
        t0 = time.perf_counter()
        model = load_model(spec.hf_name)
        boot["load_model_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        chunk_tokens = budgets.chunk_budget(spec, device)
        arena_tok = budgets.arena_tokens(spec, device, chunk_tokens)
        arena = KVArena(n_layers=spec.layers,
                        n_pages=arena_tok // budgets.PAGE_TOKENS,
                        page_tokens=budgets.PAGE_TOKENS,
                        n_kv=spec.n_kv, d_head=spec.d_head,
                        dtype=torch.bfloat16)
        boot["arena_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pipeline = Pipeline(model, arena,
                            attention_mode=FILTER_ATTENTION)
        boot["pipeline_s"] = time.perf_counter() - t0
        from quail.backends import GpuContext, QuailBackend
        execution = QuailBackend().start(GpuContext(
            gpu_index=state["gpu_index"],
            gpu_count=sub["workers"],
            model=spec,
            device=device,
            query_settings={
                "chunk_tokens": sub["chunk_tokens"],
                "kv_dtype": sub["kv_dtype"],
            },
        ))
        execution.bind_loaded_model(
            model=model, arena=arena, pipeline=pipeline
        )
        state.update(torch=torch, F=F, model_execution=execution,
                     model=model, arena=arena, pipeline=pipeline,
                     spec=spec, warmed=False)
        boot["kind"] = "cold"
    answerer = _PayloadAnswerer(torch, F, state["model"],
                                sub["true_ids"], sub["false_ids"])
    state["async_ans"] = AsyncAnswers(torch, answerer)
    state["chunk_tokens"] = sub["chunk_tokens"]
    state["model_execution"].bind_query(
        torch=torch,
        async_answers=state["async_ans"],
        chunk_tokens=state["chunk_tokens"],
    )
    if "runtime_context" not in state:
        from quail.extensions import built_in_registry
        from quail.runtime.runner import ExecutionContext
        state["runtime_context"] = ExecutionContext(
            runtimes=built_in_registry().runtimes,
            model_execution=state["model_execution"],
        )
    if not state["warmed"]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            warm = warm_kernels(torch, state["arena"],
                                state["pipeline"],
                                state["async_ans"],
                                state["chunk_tokens"],
                                model_name=state["spec"].hf_name)
        torch.cuda.synchronize()
        kernel_cache.commit()
        boot["warm_kernels_s"] = time.perf_counter() - t0
        boot["warm_tier"] = warm["tier"]
        state["warmed"] = True
        boot["kind"] = "cold"
    for k in ("load_model_s", "arena_s", "pipeline_s",
              "warm_kernels_s"):
        boot[k] = round(boot[k], 2)
    boot["boot_s"] = round(time.perf_counter() - t_boot, 2)
    state["boot"] = boot


def _child_filters(state, sub):
    import time as _time

    from quail.executor.attention import FILTER_ATTENTION
    from quail.extensions import built_in_registry
    from quail.physical import PackedFilter, decode_graph
    from quail.runtime.tokens import chain_tokens

    _child_boot(state, sub)
    boot = state["boot"]
    torch = state["torch"]
    arena = state["arena"]
    if sub.get("start_query", True):
        _reset_child_query(state)
    out = dict(filters={}, survivors={}, retained={}, fresh_tokens=0,
               boot_s=boot["boot_s"], boot_kind=boot["kind"],
               boot=boot)
    pre = sub.get("pre_ids") or []
    limit = sub.get("limit")
    t0 = _time.perf_counter()
    with torch.inference_mode():
        state["pipeline"].attention_mode = FILTER_ATTENTION
        node_id = sub.get("node_id")
        if node_id is not None:
            graph = decode_graph(
                sub["physical_plan"]["graph"],
                built_in_registry().codecs,
            )
            node = graph.node(node_id)
            if not isinstance(node, PackedFilter):
                raise TypeError(
                    f"child filter received {node.type_name!r}")
            alias = node.alias
            index = sub["doc_index"][alias]
            keep = (range(len(sub["docs"][alias]))
                    if node.keep_kv else ())
            runtime_context = state["runtime_context"]
            result = runtime_context.runtimes[node.runtime_key].execute(
                node,
                {
                    "documents": [
                        chain_tokens(pre, document)
                        for document in sub["docs"][alias]
                    ],
                    "document_ids": index,
                    "limit": limit,
                    "retain_survivors": keep,
                },
                runtime_context,
            )
            answers = result.outputs[f"filter_answers:{alias}"]
            if node.keep_kv:
                out["retained"][alias] = sorted(
                    key[1] for key in arena.accounting.retained
                    if key[0] == alias)
            out["fresh_tokens"] += result.metrics.fresh_tokens
            # answers is keyed by document position; every answered
            # document's prefix was computed once in this query
            state["seen"].update((alias, document)
                                 for document in answers)
            out["filters"][alias] = {
                int(document): row for document, row in answers.items()
            }
            out["survivors"][alias] = list(
                result.outputs[f"ids:{alias}"]
            )
    torch.cuda.synchronize()
    out["wall_s"] = round(_time.perf_counter() - t0, 2)
    out["peak_gib"] = round(
        torch.cuda.max_memory_allocated() / 2**30, 2)
    return out


def _child_joins(state, sub):
    import time as _time

    from quail.executor.attention import JOIN_ATTENTION
    from quail.extensions import built_in_registry
    from quail.physical import AdaptiveJoinPlan, AnchoredJoin, decode_graph
    from quail.runtime.coordinator import stage_for_anchor
    from quail.runtime.quail_graph import _join_round_kv, _tuple_suffix
    from quail.runtime.tokens import chain_tokens

    _child_boot(state, sub)
    torch = state["torch"]
    arena = state["arena"]
    if sub.get("start_query", False):
        _reset_child_query(state)
    pre = sub.get("pre_ids") or []
    registry = built_in_registry()
    encoded_node = sub["physical_node"]
    node = registry.codecs[encoded_node["type"]].decode(encoded_node)
    if not isinstance(node, AnchoredJoin):
        raise TypeError(f"child join received {node.type_name!r}")
    graph = decode_graph(
        sub["physical_plan"]["graph"], registry.codecs
    )
    adaptive = next(
        physical_node for physical_node in graph.nodes
        if isinstance(physical_node, AdaptiveJoinPlan)
    )
    group = [
        stage_for_anchor(adaptive.join_specs[index], node.anchor)
        for index in node.stage_idxs
    ]
    anchor_alias = node.anchor
    anchors_glob = list(sub["anchor_index"])
    anchor_docs = sub["anchor_docs"]
    # kept KV whose consumer groups are behind us, and kept anchors a
    # barrier thinned off this shard, have no reader here
    drop = set(sub.get("drop_kept") or ())
    alive_now = set(anchors_glob)
    for key in list(arena.accounting.retained):
        stale = key[0] == anchor_alias and key[1] not in alive_now
        if key[0] in drop or stale:
            arena.free_key(key)
    retain_anchor = bool(sub.get("retain_anchor"))
    seen = state.setdefault("seen", set())
    out_joins, tokens_total = [], 0
    t0 = _time.perf_counter()
    with torch.inference_mode():
        state["pipeline"].attention_mode = JOIN_ATTENTION
        stage_suffixes, tuple_globs = [], []
        for j in group:
            # partner docs ride keyed by local position; tuples
            # combine one local position per partner alias
            locals_ = [range(len(sub["partners"][p]["index"]))
                       for p in j["partners"]]
            combos = list(itertools.product(*locals_))
            tuple_globs.append(
                [[sub["partners"][p]["index"][c]
                  for p, c in zip(j["partners"], combo)]
                 for combo in combos])
            part_docs = {p: sub["partners"][p]["docs"]
                         for p in j["partners"]}
            stage_suffixes.append(
                [_tuple_suffix(j, part_docs, combo)
                 for combo in combos])
        prefixes = [chain_tokens(pre, d) for d in anchor_docs]
        anchor_keys = [(anchor_alias, g) for g in anchors_glob]
        round_kv = _join_round_kv(
            anchor_keys, [len(p) for p in prefixes],
            arena.accounting.owned, seen)

        def anchor_done(a, row):
            matched = any(row)
            alive = (not matched if group[-1]["semantics"] == "anti"
                     else matched)
            if alive and retain_anchor:
                arena.retain(anchor_keys[a], len(prefixes[a]))
            else:
                arena.free_key(anchor_keys[a])

        runtime_context = state["runtime_context"]
        result = runtime_context.runtimes[node.runtime_key].execute(
            node,
            {
                "prefixes": prefixes,
                "stage_suffixes": stage_suffixes,
                "stage_frames": [
                    join.get("frame") or [] for join in group
                ],
                "anchor_keys": anchor_keys,
                "anchor_done": anchor_done,
                "anchor_ids": anchors_glob,
                "partner_indices": {
                    stage.written_pos: tuples
                    for stage, tuples in zip(node.stages, tuple_globs)
                },
                "group": group,
            },
            runtime_context,
        )
        ans = result.metrics.extension["answers"]
        seen.update(anchor_keys)
        # settle anchors run_join never called back (no live suffixes)
        last = ans[-1] if ans else {}
        for a, key in enumerate(anchor_keys):
            if key not in arena.accounting.owned:
                continue
            matched = any(last.get(a, []))
            alive = (not matched if group[-1]["semantics"] == "anti"
                     else matched)
            if alive and retain_anchor:
                arena.retain(key, len(prefixes[a]))
            else:
                arena.free_key(key)
        tokens_total += result.metrics.fresh_tokens
        for si, j in enumerate(group):
            out_joins.append(dict(
                rows={int(a): row for a, row in ans[si].items()},
                anchor_index=anchors_glob,
                partner_index=tuple_globs[si]))
    if sub.get("final_group"):
        for key in list(arena.accounting.owned):
            arena.free_key(key)
    torch.cuda.synchronize()
    retained = {}
    for alias, document in arena.accounting.retained:
        retained.setdefault(alias, []).append(document)
    return dict(joins=out_joins, fresh_tokens=tokens_total,
                retained={alias: sorted(documents)
                          for alias, documents in retained.items()},
                kv_round=round_kv,
                kv_totals=dict(
                    evicted_keys=arena.evicted_keys,
                    evicted_pages=arena.evicted_pages,
                    evicted_prefix_tokens=arena.evicted_prefix_tokens),
                wall_s=round(_time.perf_counter() - t0, 2))


def _reset_child_query(state):
    """Clear KV and counters before a child starts a new query."""
    arena = state["arena"]
    for key in list(arena.accounting.owned):
        arena.free_key(key)
    arena.reset_stats()
    state["seen"] = set()


def _ensure_children(k):
    import multiprocessing as mp
    if len(_CHILDREN) >= k:
        return
    ctx = mp.get_context("spawn")
    for gpu in range(len(_CHILDREN), k):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(target=_child_main, args=(gpu, child_conn),
                           daemon=True)
        proc.start()
        _CHILDREN.append((proc, parent_conn))


def _round(kind, subs):
    """Send one round to children and collect results."""
    for (_, conn), sub in zip(_CHILDREN, subs):
        conn.send((kind, sub))
    outs = []
    for (_, conn), _sub in zip(_CHILDREN, subs):
        status, data = conn.recv()
        if status != "ok":
            raise RuntimeError(f"GPU child failed:\n{data}")
        outs.append(data)
    return outs


def _execute_multi(payload: dict) -> dict:
    """Execute the typed Quail graph across GPU child processes."""
    validate_plan_payload(payload)
    from quail.extensions import built_in_registry
    from quail.physical import decode_graph
    from quail.runtime.quail_distributed import execute_distributed_graph
    from quail.runtime.tokens import decode_payload_documents
    from quail.specs import DEVICES, MODELS

    payload = dict(payload)
    payload["docs"] = decode_payload_documents(payload["docs"])
    gpu_count = payload["workers"]
    _ensure_children(gpu_count)
    registry = built_in_registry()
    graph = decode_graph(
        payload["physical_plan"]["graph"], registry.codecs
    )
    graph.validate(runtime_keys=set(registry.runtimes))
    report = execute_distributed_graph(
        payload,
        graph,
        gpu_count,
        _round,
        MODELS[payload["model"]],
        DEVICES["h100-sxm"],
        registry.runtimes,
    )
    results_vol.commit()
    kernel_cache.commit()
    return report


@app.function(image=image, gpu="H100!:2", timeout=7200, memory=131072,
              scaledown_window=300, max_containers=1,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def execute_2(payload: dict) -> dict:
    return _execute_multi(payload)


@app.function(image=image, gpu="H100!:4", timeout=7200, memory=196608,
              scaledown_window=300, max_containers=1,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def execute_4(payload: dict) -> dict:
    return _execute_multi(payload)


@app.function(image=image, gpu="H100!:8", timeout=7200, memory=262144,
              scaledown_window=300, max_containers=1,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def execute_8(payload: dict) -> dict:
    return _execute_multi(payload)


class _PayloadAnswerer:
    """Answerer using TRUE/FALSE token ids from the payload."""

    def __init__(self, torch, F, model, true_ids, false_ids):
        self.F = F
        self.allowed = sorted(set(true_ids) | set(false_ids))
        # the head weight lives on the CPU when untied (moved there at
        # load); slice where it lives, keep only the slice on the GPU
        weight = model.lm_head.weight
        sel = torch.tensor(self.allowed, device=weight.device)
        self.weights = weight.index_select(0, sel).to(
            device="cuda", dtype=torch.bfloat16)
        self.true_cols = torch.tensor(
            [i for i, t in enumerate(self.allowed)
             if t in set(true_ids)], device="cuda")
        self.false_cols = torch.tensor(
            [i for i, t in enumerate(self.allowed)
             if t in set(false_ids)], device="cuda")

    def __call__(self, normed):
        scores = self.F.linear(normed, self.weights)
        t = scores.index_select(1, self.true_cols).amax(dim=1)
        f = scores.index_select(1, self.false_cols).amax(dim=1)
        return (t > f).int().cpu().tolist()
