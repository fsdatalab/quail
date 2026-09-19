"""Run any QuailB query through the engine under instrumentation.

Records the GPU timeline, KV accounting, and torch.profiler windows.

The cell runs each named query through the real planner and the real
worker execution core and records, per forward pass: launch time,
packed tokens, and GPU time from the CUDA events the loops already
create. Per phase it records walls and CPU seconds per loop step;
across the query, every blocked-admission eviction call and the KV
regret. The worker's `kv_manager` block counts all evicted documents,
including retained-pool replacements.
All instrumentation wraps the engine from this script (module
attributes and instance attributes); no engine file changes.

KV regret: the fresh tokens spent recomputing a document prefix
whose KV this query already computed once under the same
(alias, document) key. With an unlimited KV arena every one of
those tokens would have been a KV hit. First computations and
tuple-suffix tokens are not regret; no cache of any size avoids
them.

torch.profiler windows (kernel activity only, CUDA) are derived
from the run, not from the query: when a phase (one filter or join
loop) starts, four candidate windows register - "evicting" (arms on
the phase's first retained-KV eviction), "early" (chunk 3), "mid"
(chunk 18), "late" (chunk 60). One capture runs at a time; whichever
candidate arms first wins, and one that never arms never fires and
blocks nothing. A second, unprofiled pass of each query supplies the
walls a report cites, so profiler overhead never touches a headline
number.

Run (`--queries` is a comma-separated list of QuailB ids):

    uv run modal run experiments/profile_quail.py::run_smoke --queries IMDB-3,BIO-2
    uv run modal run experiments/profile_quail.py::run --queries IMDB-3,BIO-2 \
        --out-prefix myrun

Outputs on the quail-results volume (pick an --out-prefix that does
not overwrite files a report already cites):

    /results/ablations/<prefix>_<queryslug>.json
    /results/ablations/<prefix>_traces/<slug>_<op><n>_<window>.chrome.json.gz

The recorded 2026-08-30 runs used prefixes "discrepancy" and
"ringfix" through this cell's predecessor
(experiments/discrepancy_timeline.py, which hardcoded the two queries
and their windows). Their reports were removed on 2026-09-06; the data
stays under /results/ablations/ on the quail-results volume.
"""

import json
import os
import time

import modal

from quail.bench.images import gpu_image

image = gpu_image()

# House rule: never create new Modal app names - new GPU cells attach
# to an existing app.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

GPU_KW = dict(image=image, gpu="H100!", memory=98304,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})

DATA_DIR = "/results/quailb_data"

# Candidate profiler windows per phase: (name, arming rule, chunks
# covered). Listed in priority order - when several are armed at the
# same forward pass, the first listed captures.
WINDOW_TEMPLATE = (
    ("evicting", "evicting", 24),
    ("early", "chunk3", 12),
    ("mid", "chunk18", 12),
    ("late", "chunk60", 12),
)


def _slug(qid):
    return qid.lower().replace("-", "")


# ------------------------------------------------------------- helpers

def _write(result, name):
    print(json.dumps(result, indent=2, default=str)[:4000], flush=True)
    os.makedirs("/results/ablations", exist_ok=True)
    with open(f"/results/ablations/{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    kernel_cache.commit()


def _boot_state(model, **pipeline_kwargs):
    """Boot the worker state dict, as the worker's own boot does.

    Mirrors quail.execution.execute._execute_physical's boot with the
    shipping pipeline; warm_kernels runs the same tiered warmup.
    pipeline_kwargs go to the model's pipeline, for experiments that
    swap a kernel.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.loop import warm_kernels
    from quail.backends.quail.executor.model import load_model
    from quail.backends.quail.executor.models import build_pipeline
    from quail.backends.quail.executor.readout import AnswerRows, AsyncAnswers
    from quail.cost import budgets
    from quail.specs import DEVICES, MODELS

    spec = MODELS[model]
    device = DEVICES["h100-sxm"]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    chunk_tokens = budgets.chunk_budget(spec, device)
    model_mod = load_model(spec.hf_name, revision=spec.revision,
                           max_batched_tokens=chunk_tokens,
                           moe_backend=spec.moe_backend)
    full_pages, sliding_pages = budgets.arena_pages(spec, device, chunk_tokens)
    arena = KVArena(n_layers=spec.layers,
                    n_pages=full_pages,
                    page_tokens=budgets.PAGE_TOKENS,
                    n_kv=spec.n_kv, d_head=spec.d_head,
                    dtype=torch.bfloat16, layer_kv=spec.kv_shapes,
                    sliding_layers=spec.sliding_layer_set,
                    sliding_window=spec.sliding_window,
                    n_sliding_pages=sliding_pages)
    pipeline = build_pipeline(spec, model_mod, arena, **pipeline_kwargs)
    from quail.backends import GpuContext, QuailBackend
    execution = QuailBackend().start(GpuContext(
        gpu_index=0,
        gpu_count=1,
        model=spec,
        device=device,
        query_settings={
            "chunk_tokens": chunk_tokens,
        },
    ))
    execution.bind_loaded_model(
        model=model_mod, arena=arena, pipeline=pipeline
    )
    answerer = AnswerRows.from_tokenizer(torch, F, model_mod, tokenizer)
    async_ans = AsyncAnswers(torch, answerer)
    with torch.inference_mode():
        warm = warm_kernels(torch, arena, pipeline, async_ans,
                            chunk_tokens, model_name=spec.hf_name)
    torch.cuda.synchronize()
    kernel_cache.commit()
    state = dict(model_execution=execution,
                 model=model_mod, arena=arena, pipeline=pipeline,
                 spec=spec, torch=torch, F=F)
    return state, chunk_tokens, warm


def _quailb_session(model, sf, gpus=1):
    """Build the QUAIL-B tables and a registered session."""
    import quail
    from quail.bench.quailb import queries, register_tables
    from quail.planner.plan import EngineConfig
    from quail_b.data import build_sets

    d = build_sets(DATA_DIR, sf)
    results_vol.commit()
    sess = quail.Session(EngineConfig(
        gpus=gpus,
        model=model,
        backend="quail",
        device="h100-sxm",
    ))
    register_tables(sess, d)
    return sess, queries(sess)


def _run_query(state, build, captured):
    """One query through the real planner and worker core."""
    from quail.backends.quail.executor.readout import AnswerRows, AsyncAnswers
    from quail.backends.quail.worker import (
        execute_single,
        quail_runtime_payload,
    )
    from quail.execution.execute import (
        _validate_physical_request,
        execute_query,
    )
    from quail.execution.types import PhysicalResponse

    def execute(request):
        registry = query.session.registry
        graph, _ = _validate_physical_request(request, registry)
        payload = quail_runtime_payload(request, graph)
        rows = AnswerRows(
            state["torch"], state["F"], state["model"],
            payload["true_ids"], payload["false_ids"],
        )
        arena_pages = payload.get("arena_pages")
        if arena_pages is not None:
            state["arena"].resize(*arena_pages, free_resident=True)
        state["model_execution"].bind_query(
            torch=state["torch"],
            async_answers=AsyncAnswers(state["torch"], rows),
            answer_rows=rows,
            chunk_tokens=payload["chunk_tokens"],
        )
        report = execute_single(state, payload, registry, graph)
        outputs = report.pop("_outputs")
        report.pop("filters", None)
        report.pop("joins", None)
        captured.clear()
        captured.update(report)
        return PhysicalResponse(outputs, report)

    query = build()
    return execute_query(query, physical_executor=execute)


def _cupti_preinit(torch):
    """Run one throwaway profiled kernel before any measured query.

    CUPTI starts lazily on the first profiled kernel, so this keeps that
    start out of a measured query.
    """
    x = torch.ones(1024, device="cuda")
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]):
        (x * 2.0).sum().item()
    torch.cuda.synchronize()


# ----------------------------------------------------- profiler windows

class ProfilerWindows:
    """Short kernel-activity captures driven from forward-pass starts.

    open_phase registers this phase's candidates from WINDOW_TEMPLATE
    and drops the previous phase's unfired ones. Each candidate arms
    when its condition first holds; one capture runs at a time,
    covers `chunks` forward passes, then writes a chrome trace.
    Kernel activity only (CUDA), so the CPU side of the loop runs
    untraced.
    """

    def __init__(self, torch, arena, slug, trace_dir):
        self.torch = torch
        self.arena = arena
        self.slug = slug
        self.trace_dir = trace_dir
        self.pending = []
        self.phase_tag = ""
        self.session = None
        self.left = 0
        self.meta = []

    def open_phase(self, op, ordinal, chunk_seq):
        if self.session is not None:
            self._close(chunk_seq)
        self.phase_tag = f"{op}{ordinal}"
        self.pending = [
            dict(name=name, arm=arm, chunks=chunks,
                 evict_base=self.arena.evicted_keys)
            for name, arm, chunks in WINDOW_TEMPLATE]

    def _armed(self, spec, idx):
        if spec["arm"] == "evicting":
            return self.arena.evicted_keys > spec["evict_base"]
        return idx >= int(spec["arm"].removeprefix("chunk"))

    def on_chunk(self, idx, chunk_seq):
        """Called before each forward pass launches.

        Args:
            idx: Chunk index within the phase.
            chunk_seq: Chunk index across the query.
        """
        if self.session is not None:
            self.left -= 1
            if self.left <= 0:
                self._close(chunk_seq)
            return
        for i, spec in enumerate(self.pending):
            if self._armed(spec, idx):
                self.pending.pop(i)
                break
        else:
            return
        prof = self.torch.profiler.profile(
            activities=[self.torch.profiler.ProfilerActivity.CUDA])
        prof.__enter__()
        self.session = (spec, prof, self.phase_tag)
        self.left = spec["chunks"]
        self.meta.append(dict(
            name=f"{self.phase_tag}_{spec['name']}",
            first_chunk=chunk_seq, chunks=spec["chunks"]))

    def _close(self, chunk_seq):
        spec, prof, tag = self.session
        self.session = None
        # enqueued kernels must finish before collection stops
        self.torch.cuda.synchronize()
        prof.__exit__(None, None, None)
        os.makedirs(self.trace_dir, exist_ok=True)
        path = f"{self.trace_dir}/{self.slug}_{tag}_{spec['name']}" \
               ".chrome.json.gz"
        prof.export_chrome_trace(path)
        self.meta[-1]["last_chunk"] = chunk_seq
        self.meta[-1]["trace"] = path
        print(f"[profile] window {tag}_{spec['name']} -> {path}",
              flush=True)

    def finish(self, chunk_seq):
        if self.session is not None:
            self._close(chunk_seq)


# --------------------------------------------------------- the recorder

class LoopRecorder:
    """Records the execution loops from outside the engine.

    Wraps quail.backends.quail.executor.loop.run_filter / run_join (module
    attributes, bound at the worker's call time), the pipeline's
    forward_chunk, and the arena's blocked-admission eviction method.
    Restores everything in unpatch().
    """

    def __init__(self, state, profiler=None):
        import quail.backends.quail.executor.loop as loop_mod

        self.loop_mod = loop_mod
        self.torch = state["torch"]
        self.arena = state["arena"]
        self.pipeline = state["pipeline"]
        self.profiler = profiler
        self.t0 = time.perf_counter()
        self.phases = []
        self.evictions = []
        self.seen = {}          # key -> prefix tokens computed before
        self.chunk_seq = 0
        self._phase = None      # (record, launches, phase chunk idx)
        self._orig = dict(run_filter=loop_mod.run_filter,
                          run_join=loop_mod.run_join,
                          forward=self.pipeline.forward_chunk,
                          evict=self.arena.evict_retained)
        loop_mod.run_filter = self._run_filter
        loop_mod.run_join = self._run_join
        self.pipeline.forward_chunk = self._forward_chunk
        self.arena.evict_retained = self._evict_retained

    def unpatch(self):
        self.loop_mod.run_filter = self._orig["run_filter"]
        self.loop_mod.run_join = self._orig["run_join"]
        self.pipeline.forward_chunk = self._orig["forward"]
        self.arena.evict_retained = self._orig["evict"]

    def _now(self):
        return time.perf_counter() - self.t0

    # ---- wrapped engine entry points --------------------------------

    def _forward_chunk(self, chunk):
        if self._phase is None:     # a forward outside the two loops
            return self._orig["forward"](chunk)
        record, launches, idx = self._phase
        if self.profiler is not None:
            self.profiler.on_chunk(idx, self.chunk_seq)
        launches.append((self._now(), chunk.tokens))
        self._phase = (record, launches, idx + 1)
        self.chunk_seq += 1
        return self._orig["forward"](chunk)

    def _evict_retained(self, pages_needed):
        keys = self._orig["evict"](pages_needed)
        self.evictions.append(dict(
            t=round(self._now(), 4),
            phase=len(self.phases),
            pages_needed=int(pages_needed),
            keys_evicted=len(keys)))
        return keys

    def _open_phase(self, op, label, kv=None):
        record = dict(op=op, label=label, t_start=round(self._now(), 4),
                      kv=kv or {})
        if self.profiler is not None:
            self.profiler.open_phase(op, len(self.phases),
                                     self.chunk_seq)
        self._phase = (record, [], 0)
        return record, time.perf_counter()

    def _close_phase(self, record, t_call, spans, tokens):
        # the loops drain their answers before returning, so the
        # events are complete once this synchronize returns
        self.torch.cuda.synchronize()
        record["wall_s"] = round(time.perf_counter() - t_call, 4)
        record["tokens"] = int(tokens)
        _, launches, _ = self._phase
        gpu_ms = [e0.elapsed_time(e1) for _, e0, e1 in spans]
        record["n_chunks"] = len(launches)
        record["gpu_s"] = round(sum(gpu_ms) / 1e3, 4)
        record["chunks"] = [
            dict(t=round(t, 4), tokens=int(n), gpu_ms=round(ms, 3))
            for (t, n), ms in zip(launches, gpu_ms)]
        self._phase = None
        self.phases.append(record)
        print(f"[profile] {record['op']} {record['label']}: "
              f"wall {record['wall_s']}s gpu {record['gpu_s']}s "
              f"chunks {record['n_chunks']} tokens {record['tokens']}",
              flush=True)

    def _run_filter(self, torch, arena, pipeline, async_ans, doc_ids,
                    question_ids, budget, **kw):
        keys = kw.get("arena_keys") or list(range(len(doc_ids)))
        record, t_call = self._open_phase(
            "filter", f"{len(doc_ids)} docs x {len(question_ids)} stages")
        kw.setdefault("timing", {})
        out = self._orig["run_filter"](
            torch, arena, pipeline, async_ans, doc_ids, question_ids,
            budget, **kw)
        answers, spans, tokens = out
        record["timing"] = {k: round(v, 4) if isinstance(v, float)
                            else v for k, v in kw["timing"].items()}
        # answers is keyed by document position; every answered
        # document's prefix was computed once in this query
        for d in answers:
            self.seen.setdefault(keys[d], len(doc_ids[d]))
        self._close_phase(record, t_call, spans, tokens)
        return out

    def _run_join(self, torch, arena, pipeline, async_ans,
                  anchor_prefixes, stage_suffixes, budget, **kw):
        keys = kw.get("anchor_keys") or list(range(len(anchor_prefixes)))
        kv = dict(anchors=len(keys), hits=0, hit_tokens=0,
                  regret_tokens=0, first_tokens=0)
        for key, prefix in zip(keys, anchor_prefixes):
            if self.arena.is_resident(key):
                kv["hits"] += 1
                kv["hit_tokens"] += len(prefix)
            elif key in self.seen:
                kv["regret_tokens"] += len(prefix)
            else:
                kv["first_tokens"] += len(prefix)
        tuples = sum(len(s) for s in stage_suffixes)
        record, t_call = self._open_phase(
            "join", f"{len(keys)} anchors, {tuples} tuples", kv)
        out = self._orig["run_join"](
            torch, arena, pipeline, async_ans, anchor_prefixes,
            stage_suffixes, budget, **kw)
        _, spans, tokens = out
        for k, p in zip(keys, anchor_prefixes):
            self.seen.setdefault(k, len(p))
        self._close_phase(record, t_call, spans, tokens)
        return out

    # ---- results ----------------------------------------------------

    def result(self):
        return dict(phases=self.phases, evictions=self.evictions,
                    regret_tokens=sum(p["kv"].get("regret_tokens", 0)
                                      for p in self.phases),
                    kv_hit_tokens=sum(p["kv"].get("hit_tokens", 0)
                                      for p in self.phases))


# ------------------------------------------------------------ the cell

def _measure(state, qdefs, qid, profiled, trace_dir):
    torch = state["torch"]
    windows = None
    if profiled:
        windows = ProfilerWindows(torch, state["arena"], _slug(qid),
                                  trace_dir)
    recorder = LoopRecorder(state, profiler=windows)
    captured = {}
    try:
        _run_query(state, qdefs[qid][1], captured)
    finally:
        recorder.unpatch()
    if windows is not None:
        windows.finish(recorder.chunk_seq)
    out = recorder.result()
    out["engine_wall_s"] = captured.get("wall_s")
    out["fresh_tokens"] = captured.get("fresh_tokens")
    out["kv_manager"] = captured.get("kv_manager")
    out["executed_join_plan"] = captured.get("executed_join_plan")
    if windows is not None:
        out["windows"] = windows.meta
    return out


@app.function(timeout=3600, **GPU_KW)
def measure(model: str = "qwen3-4b-fp8", sf: float = 0.1,
            queries: tuple = (),
            out_prefix: str = "profile") -> str:
    """Run each named query twice, once unprofiled and once profiled.

    The unprofiled pass gives the cited walls and the chunk timeline.
    The profiled pass gives the derived trace windows.

    out_prefix names the output files and trace directory; pick one
    that does not overwrite files a report already cites.
    """
    if not queries:
        raise ValueError("pass at least one QuailB query id")
    state, chunk_tokens, warm = _boot_state(model)
    _cupti_preinit(state["torch"])
    sess, qdefs = _quailb_session(model, sf)
    missing = [q for q in queries if q not in qdefs]
    if missing:
        raise KeyError(f"unknown QuailB queries {missing}; "
                       f"known: {sorted(qdefs)}")
    trace_dir = f"/results/ablations/{out_prefix}_traces"
    summary = dict(cell="profile_quail", model=model, sf=sf,
                   chunk_tokens=chunk_tokens, warm=warm, queries={})
    for qid in queries:
        result = dict(query=qid, sf=sf, model=model,
                      chunk_tokens=chunk_tokens)
        result["unprofiled"] = _measure(state, qdefs, qid,
                                        profiled=False,
                                        trace_dir=trace_dir)
        result["profiled"] = _measure(state, qdefs, qid,
                                      profiled=True,
                                      trace_dir=trace_dir)
        name = f"{out_prefix}_{_slug(qid)}"
        if sf != 0.1:
            name += f"_sf{sf}"
        _write(result, name)
        u = result["unprofiled"]
        summary["queries"][qid] = dict(
            engine_wall_s=u["engine_wall_s"],
            fresh_tokens=u["fresh_tokens"],
            regret_tokens=u["regret_tokens"],
            phases=[dict(op=p["op"], wall_s=p["wall_s"],
                         gpu_s=p["gpu_s"], n_chunks=p["n_chunks"])
                    for p in u["phases"]])
    return json.dumps(summary, indent=2)


# ---------------------------------------------------------- entrypoints

def _parse_queries(queries):
    out = tuple(q.strip() for q in queries.split(",") if q.strip())
    if not out:
        raise ValueError("--queries must name at least one QuailB id")
    return out


@app.local_entrypoint()
def run(queries: str, model: str = "qwen3-4b-fp8", sf: float = 0.1,
        out_prefix: str = "profile"):
    handle = measure.spawn(model, sf, _parse_queries(queries), out_prefix)
    print(f"profile_quail fc: {handle.object_id}", flush=True)
    print(handle.get())


@app.local_entrypoint()
def run_smoke(queries: str, model: str = "qwen3-4b-fp8",
              out_prefix: str = "profile"):
    """Run the full harness on the sf 0.01 tables before the measured run.

    This takes minutes, compared with tens of minutes for the measured run.
    """
    handle = measure.spawn(model, 0.01, _parse_queries(queries),
                           out_prefix)
    print(f"profile_quail smoke fc: {handle.object_id}", flush=True)
    print(handle.get())
