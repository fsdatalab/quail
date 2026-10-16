"""Measure AI.EXTRACT passes, rounds, and drafts for one draft length on Modal.

Runs one QUAIL-B extraction query in a fresh process with the plan's
draft_tokens replaced. The timed runs record every forward pass: its
tokens, answer rows, documents prefilled, resident KV tokens, and GPU
milliseconds; --sweep alternates draft lengths between timed runs on
one GPU. Optional profiled runs sum GPU time by kernel name
(--kernel-runs) or host time by function (--host-runs). The logged
run, which is not timed, records every round: the live streams, up to
16 of their continuations, and what the round accepted, so other draft
lengths and draft choices can be replayed on the CPU. The record is
saved to /results/ablations/extract_draft/.

    uv run modal run --detach experiments/cells/extract_draft.py::measure
        --query CUAD-3 --sf 1.0 --draft-tokens 16

--sql with --table runs an ad hoc query over one benchmark table instead
of a QUAIL-B query; the table's parquet must already be on the volume.
"""

import cProfile
import json
import pstats
import sys
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from bench.quailb_parallel import (
    DATA_DIR,
    VOLUMES,
    app,
    ensure_data,
    results_vol,
)
from quail.deploy.images import gpu_image

OUT_DIR = "/results/ablations/extract_draft"
LOGGED_STREAMS = 16
LOGGED_TOKENS = 64
image = gpu_image()


def _set_draft(draft_tokens) -> dict:
    """Plan every extraction with the draft setting the returned dict holds.

    The setting is a draft length, or "adaptive" for the policy with
    the plan's max_answer_tokens - 1 as its cap.
    """
    import quail.planner.operators.extract as planned
    from quail.physical import ExtractSpec

    bind = planned.bind_inference
    setting = {"draft_tokens": draft_tokens}

    def bind_with_draft(cls, *args, **kwargs):
        spec = bind(cls, *args, **kwargs)
        if cls is ExtractSpec:
            draft = setting["draft_tokens"]
            if draft == "adaptive":
                spec = replace(spec, draft_policy="adaptive",
                               draft_tokens=spec.max_answer_tokens - 1)
            else:
                spec = replace(spec, draft_policy="fixed", draft_tokens=int(draft))
        return spec

    planned.bind_inference = bind_with_draft
    return setting


class Passes:
    """GPU time, tokens, rows, prefills, and resident KV of each forward pass.

    Also the host seconds spent in each executor step: packing a chunk,
    launching the forward pass, submitting the readout, deciding
    answers, and choosing the next chunk.
    """

    STEPS = ("pack", "forward", "submit", "decide", "next_chunk")

    def __init__(self, torch):
        import quail.backends.quail.executor.chunk as chunk
        import quail.backends.quail.executor.readout_groups as readout_groups
        import quail.backends.quail.executor.stage_executor as stage_executor
        from quail.backends.quail.executor import admission

        self.torch = torch
        self.active = False
        self.entries = []
        self.host = dict.fromkeys(self.STEPS, 0.0)
        self.gpu_steps = []
        self._wrap(chunk, "pack_chunk", "pack")
        self._wrap(readout_groups.ReadoutGroups, "submit", "submit", gpu=True)
        self._wrap(stage_executor.StageExecutor, "_report_next", "decide")
        self._wrap(admission.JoinAdmission, "next_chunk", "next_chunk")
        forward = stage_executor.forward_chunk
        passes = self

        def timed(pipeline, arena, chunk):
            if not passes.active:
                return forward(pipeline, arena, chunk)
            facts = (int(chunk.tokens), int(len(chunk.final_indices)),
                     len(chunk.fresh_keys),
                     int(arena.accounting.resident_tokens))
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            began = time.perf_counter()
            try:
                return forward(pipeline, arena, chunk)
            finally:
                end.record()
                passes.host["forward"] += time.perf_counter() - began
                passes.entries.append((began, start, end, facts))

        stage_executor.forward_chunk = timed

    def _wrap(self, owner, name, step, gpu=False):
        original = getattr(owner, name)
        passes = self
        torch = self.torch

        def timed(*args, **kwargs):
            events = None
            if gpu and passes.active:
                events = (torch.cuda.Event(enable_timing=True),
                          torch.cuda.Event(enable_timing=True))
                events[0].record()
            began = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                if passes.active:
                    passes.host[step] += time.perf_counter() - began
                    if events is not None:
                        events[1].record()
                        passes.gpu_steps.append((step, *events))

        setattr(owner, name, timed)

    def start(self):
        self.entries = []
        self.host = dict.fromkeys(self.STEPS, 0.0)
        self.gpu_steps = []
        self.torch.cuda.synchronize()
        self.started = time.perf_counter()
        self.active = True

    def stop(self) -> dict:
        self.torch.cuda.synchronize()
        window = time.perf_counter() - self.started
        self.active = False
        rows = [[round((began - self.started) * 1e3, 2),
                 round(start.elapsed_time(end), 3), *facts]
                for began, start, end, facts in self.entries]
        readout_s = sum(start.elapsed_time(end) for step, start, end in self.gpu_steps
                        if step == "submit") / 1e3
        return {"window_s": round(window, 4),
                "gpu_s": round(sum(row[1] for row in rows) / 1e3, 4),
                # GPU seconds of the readouts, outside the forward passes
                "readout_gpu_s": round(readout_s, 4),
                "host_s": {step: round(value, 3) for step, value in self.host.items()},
                # host start ms, GPU ms, tokens, answer rows, documents
                # prefilled, resident KV tokens before the pass
                "passes": rows}


class Rounds:
    """Every round's live streams and result, and every settled answer."""

    def __init__(self):
        from quail.backends.quail.executor.extract import decode, stages
        from quail.backends.quail.executor.extract.candidates import ensure_tokens

        self.active = False
        self.drafts, self.verified, self.settled = [], [], []
        rounds = self
        step_suffix = stages.ExtractionStages._step_suffix
        verify = decode.Decoder.verify
        settle = decode.Decoder.settle

        def logged_suffix(stage, q, key):
            fed = step_suffix(stage, q, key)
            if rounds.active and fed is not stages.Stage.SKIP:
                import numpy as np

                at = np.searchsorted(stage.document_ids, key[1], side="right") - 1
                doc = stage.docs[stage.document_order[at]]
                answer = doc.answers[q]
                position = len(answer.written)
                live = sorted(answer.live, key=lambda s: s.char)
                kept = live[:LOGGED_STREAMS] if len(live) > 1 else []
                ensure_tokens(kept, position + LOGGED_TOKENS, stage.vocabulary)
                rounds.drafts.append([
                    doc.index, q, position, len(live),
                    [[s.char, [int(t) for t in
                               s.stream[position:position + LOGGED_TOKENS]]]
                     for s in kept]])
            return fed

        def logged_verify(decoder, doc, q, rows, fed, accepted=None):
            verify(decoder, doc, q, rows, fed, accepted)
            if rounds.active:
                answer = doc.answers[q]
                rounds.verified.append([doc.index, q, len(fed), answer.discard,
                                        answer.done, len(answer.written)])

        def logged_settle(decoder, doc, q):
            settle(decoder, doc, q)
            if rounds.active:
                answer = doc.answers[q]
                rounds.settled.append([
                    doc.index, q, [int(t) for t in answer.written],
                    len(answer.live), answer.none_wins, answer.answer is None,
                    len(doc.prefix)])

        stages.ExtractionStages._step_suffix = logged_suffix
        decode.Decoder.verify = logged_verify
        decode.Decoder.settle = logged_settle

    def start(self):
        self.drafts, self.verified, self.settled = [], [], []
        self.active = True

    def stop(self) -> dict:
        self.active = False
        # draft rows: document, question, answer tokens so far, live
        # streams, then [character, next tokens] of up to 16 of them;
        # verify rows: document, question, fed, discarded, done,
        # answer tokens after the round; settle rows: document,
        # question, answer tokens, live streams, none won, null answer,
        # prompt head and document tokens
        return {"drafts": self.drafts, "verified": self.verified,
                "settled": self.settled}


def _host_totals(profiler, top: int = 60) -> dict:
    """Return the host functions with the most own time and the most total time."""
    stats = pstats.Stats(profiler)

    def ranked(key):
        rows = sorted(stats.stats.items(), key=lambda item: -item[1][key])[:top]
        # file:line(function), calls, own seconds, total seconds
        return [[f"{path.split('/root/')[-1]}:{line}({name})", calls,
                 round(own, 3), round(total, 3)]
                for (path, line, name), (_, calls, own, total, _) in rows]

    return {"by_own_s": ranked(2), "by_total_s": ranked(3)}


def _kernel_totals(profiler) -> list:
    """Return [kernel name, GPU ms, launches], most GPU time first."""
    from torch.autograd import DeviceType

    totals = {}
    for event in profiler.events():
        if event.device_type == DeviceType.CUDA:
            ms, count = totals.get(event.name, (0.0, 0))
            totals[event.name] = (ms + event.time_range.elapsed_us() / 1e3,
                                  count + 1)
    return sorted(([name, round(ms, 3), count]
                   for name, (ms, count) in totals.items()),
                  key=lambda row: -row[1])


def _sql_runner(session, sql: str, table: str, sf: float, limit: int = 0,
                nfc: bool = False):
    """Return a run function for an ad hoc SQL query over one benchmark table.

    limit keeps the table's first rows, for a sample of a large table.
    nfc normalizes the text column to NFC: AI.EXTRACT refuses a text
    whose tokens do not reproduce its bytes, as happens with characters
    the tokenizer normalizes, such as U+037E in some SEC filings.
    """
    import unicodedata

    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail

    rows = pq.read_table(Path(DATA_DIR) / f"sf{sf}" / f"{table}.parquet")
    if limit:
        rows = rows.slice(0, limit)
    if nfc:
        column = rows.schema.get_field_index("text")
        texts = [unicodedata.normalize("NFC", text)
                 for text in rows.column("text").to_pylist()]
        rows = rows.set_column(column, "text", pa.array(texts, type=pa.string()))
    session.register(table, quail.DocumentProvider.from_table(rows, id_col="id"))

    def run():
        started = time.perf_counter()
        result = session.sql(sql).run()
        result.collect()
        wall_s = time.perf_counter() - started
        extracts = result.answer_tables.get("extracts", {})
        return SimpleNamespace(
            measurements={**result.report, "wall_s": wall_s},
            extract_answers={name: table for name, table in extracts.items()})

    return run


def child(arguments: str) -> None:
    """Run the query: timed, then profiled, then one logged run; print RESULT."""
    import torch

    import quail
    from bench.quailb import run_query
    from quail_b.benchmark import load_benchmark

    args = json.loads(arguments)
    setting = _set_draft(args["draft_tokens"])
    passes = Passes(torch)
    rounds = Rounds()
    sf = args["sf"]
    config = quail.EngineConfig(gpus=1, model=args["model"], backend="quail",
                                device="h100-sxm", gpu_timing=True)
    record = {**args, "runs": []}
    with quail.Session(config) as session:
        if args.get("sql"):
            run_once = _sql_runner(session, args["sql"], args["table"], sf,
                                   args.get("limit", 0), args.get("nfc", False))
        else:
            suite = load_benchmark([args["query_id"]], scale_factor=sf,
                                   data_dir=Path(DATA_DIR) / f"sf{sf}",
                                   collection_id=args["collection_id"] or None)
            spec = suite.queries[0]
            tables = {relation.table: suite.tables[relation.table]
                      for relation in spec.info.relations}
            run_once = partial(run_query, session, spec, tables)
        drafts = args.get("sweep") or [args["draft_tokens"]]
        modes = ([("timed", d) for _ in range(args["timed_runs"]) for d in drafts]
                 + [("kernels", args["draft_tokens"])] * args["kernel_runs"]
                 + [("host", args["draft_tokens"])] * args.get("host_runs", 0)
                 + [("logged", args["draft_tokens"])])
        for mode, draft in modes:
            setting["draft_tokens"] = draft
            profiler = None
            if mode == "timed":
                before = torch.cuda.memory_stats()
                passes.start()
            elif mode == "host":
                profiler = cProfile.Profile()
                profiler.enable()
            elif mode == "kernels":
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA])
                profiler.__enter__()
            else:
                rounds.start()
            output = run_once()
            run = {"mode": mode, "draft_tokens": draft}
            if mode == "timed":
                run["passes"] = passes.stop()
                after = torch.cuda.memory_stats()
                # a cudaMalloc retry flushes the cache and waits for every stream
                run["allocator"] = {
                    name: after.get(name, 0) - before.get(name, 0)
                    for name in ("num_alloc_retries", "num_sync_all_streams",
                                 "num_device_alloc", "num_device_free")}
            elif mode == "host":
                profiler.disable()
                run["host"] = _host_totals(profiler)
            elif mode == "kernels":
                torch.cuda.synchronize()
                profiler.__exit__(None, None, None)
                run["kernels"] = _kernel_totals(profiler)
            else:
                run["rounds"] = rounds.stop()
            measures = output.measurements
            run["measurements"] = {
                name: measures.get(name) for name in (
                    "wall_s", "model_wall_s", "gpu_s", "chunks", "planning_s",
                    "fresh_tokens", "input_tokens", "input_tokens_per_second",
                    "query_s", "usd_per_query")}
            run["node_metrics"] = measures.get("node_metrics")
            if mode == "logged":
                answers = next(iter(output.extract_answers.values()), None) if (
                    getattr(output, "extract_answers", None)) else None
                run["answers"] = None if answers is None else answers.to_pylist()
            record["runs"].append(run)
            print(f"{mode} run: wall_s {measures.get('wall_s')}", flush=True)
    path = Path(OUT_DIR) / (f"{args['tag']}-{args['query_id']}-sf{sf}-"
                            f"{args['model']}-d{args['draft_tokens']}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, default=str))
    print(f"RESULT {path}", flush=True)


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def measure_draft(arguments: str) -> str:
    """Run child() in a fresh process and return the saved record's path."""
    import os
    import subprocess

    command = [sys.executable, "-c",
               "import sys, extract_draft; extract_draft.child(sys.argv[1])",
               arguments]
    env = {**os.environ, "PYTHONPATH": "/root:" + os.environ.get("PYTHONPATH", "")}
    done = subprocess.run(command, env=env, cwd="/tmp", capture_output=True,
                          text=True)
    print(done.stdout[-30000:], flush=True)
    if done.returncode:
        print(done.stderr[-12000:], flush=True)
        raise RuntimeError("extract_draft child failed")
    results_vol.commit()
    return next(line[7:] for line in done.stdout.splitlines()
                if line.startswith("RESULT "))


@app.local_entrypoint()
def measure(query: str = "CUAD-3", sf: float = 1.0, model: str = "qwen3-4b-fp8",
            draft_tokens: str = "16", timed_runs: int = 2, kernel_runs: int = 0,
            host_runs: int = 0, sweep: str = "", sql: str = "", table: str = "",
            limit: int = 0, nfc: bool = False, tag: str = "draft"):
    collection = ""
    if sql:
        # an ad hoc query over a table ensure_data already wrote
        query = table
    else:
        data = ensure_data.spawn(sf, [query], "")
        print(f"function call id: {data.object_id} (data)", flush=True)
        collection = data.get()
    draft = draft_tokens if draft_tokens == "adaptive" else int(draft_tokens)
    arguments = json.dumps(dict(query_id=query, sf=sf, collection_id=collection,
                                sql=sql, table=table, limit=limit, nfc=nfc,
                                model=model, draft_tokens=draft,
                                timed_runs=timed_runs, kernel_runs=kernel_runs,
                                host_runs=host_runs,
                                sweep=[d if d == "adaptive" else int(d)
                                       for d in sweep.split(",") if d],
                                tag=tag))
    call = measure_draft.spawn(arguments)
    print(f"function call id: {call.object_id} (d={draft_tokens})", flush=True)
    print(f"saved to {call.get()}", flush=True)
