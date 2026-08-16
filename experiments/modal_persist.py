"""Persisted KV: is restoring a corpus cheaper than recomputing it?

The second query over the same documents can either re-read them
through prefill or load their saved KV back from host memory. Which
one wins is a bandwidth question with a measurable threshold: restore
beats recompute when

    channel bandwidth  >  kappa * prefill rate

where kappa is the KV bytes per token and the prefill rate is how
fast the GPU regenerates that KV from text. At 4B fp8 that threshold
is 73,728 bytes/token * 97,000 tokens/s = 7.2 GB/s. A bigger model
lowers the threshold (it prefills slower), which is why restore wins
more easily as models grow.

Two containers, because a second engine boot in one container dies in
DeepGEMM warmup (the core forks from a CUDA-initialized parent):

  stage "baseline"  cold query, then reset the prefix cache and run
                    the same query again - the recompute cost
  stage "store"     cold query with the CPU offload connector
                    writing KV through, then two restores

The store stage uses vLLM's OffloadingConnector with its plain CPU
spec, which allocates its pool with pin_memory=True (cudaHostAlloc).
The probe (modal_pinprobe.py) measures that memory at 55.4 GB/s for
a raw copy, against a 64 GB/s PCIe Gen5 spec; the end-to-end restore
through the connector measures about 10.2 GB/s, so roughly 5x is
lost inside the connector's per-block transfer loop. That gap is
open - it is the "might be super handicapped" TODO.

Not the tiering spec: it backs its pool with a file-backed /dev/shm
region this sandbox cannot pin (cudaHostRegister returns 304 on
file-backed mappings) and its unpinned fallback died natively
mid-query. Plain CPU spec needs no workarounds.

Run both stages, then compare:
    modal run experiments/modal_persist.py --stage baseline
    modal run experiments/modal_persist.py --stage store

Transfer tracing: the store stage wraps the connector's offload
handler so every transfer job logs one JSON line at submit and one at
finish, the finish line carrying the CUDA-event-timed copy duration
the handler already measures. Dividing bytes by those event-timed
seconds gives copy-only bandwidth; dividing the same bytes by the
query wall gives the effective number reported before. The gap
between the two is time the channel sat idle between jobs -
orchestration, not copying. The wrap is installed as a sitecustomize
module on PYTHONPATH because vLLM runs the GPU worker in a spawned
engine-core process: a patch applied only in this process would miss
it, while a fresh interpreter imports sitecustomize before anything
else.
"""

import json
import os

import modal

from workload import MODEL, N_FILTERS, hf_cache, image, results_vol

app = modal.App("quail-persist")
persist_image = image.add_local_python_source("workload")


# ---- transfer tracing -------------------------------------------------

_TRACE_HOOK = '''\
"""Trace vLLM CPU-offload transfers to JSONL (quail persist stage).

Active only when QUAIL_XFER_TRACE is set. Wraps
SingleDirectionOffloadingHandler so each transfer job logs a submit
record and a finish record with the CUDA-event-timed duration, to
QUAIL_XFER_TRACE.<pid>. Installed via an import hook so the patch
applies in whichever process imports the handler."""
import importlib.abc
import importlib.util
import json
import os
import sys
import time

_TARGET = "vllm.v1.kv_offload.cpu.gpu_worker"
_PATH = os.environ.get("QUAIL_XFER_TRACE")


def _log(rec):
    with open(f"{_PATH}.{os.getpid()}", "a") as f:
        f.write(json.dumps(rec) + "\\n")


def _patch(mod):
    cls = mod.SingleDirectionOffloadingHandler
    orig_submit = cls.transfer_async
    orig_finished = cls.get_finished

    def transfer_async(self, job_id, src_spec, dst_spec):
        _log(dict(ev="submit", t=time.time(),
                  dir="d2h" if self.gpu_to_cpu else "h2d", job=job_id,
                  src_blocks=len(getattr(src_spec, "block_ids", ()))))
        return orig_submit(self, job_id, src_spec, dst_spec)

    def get_finished(self):
        results = orig_finished(self)
        if results:
            now = time.time()
            for r in results:
                _log(dict(ev="finish", t=now,
                          dir="d2h" if self.gpu_to_cpu else "h2d",
                          job=r.job_id, bytes=r.transfer_size,
                          secs=r.transfer_time))
        return results

    cls.transfer_async = transfer_async
    cls.get_finished = get_finished


class _Hook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        self._orig_loader = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return self._orig_loader.create_module(spec)

    def exec_module(self, module):
        self._orig_loader.exec_module(module)
        _patch(module)


if _PATH:
    sys.meta_path.insert(0, _Hook())
'''


def _install_xfer_trace(trace_path):
    """Arm the trace for this process and every process spawned after.

    Must run before anything imports vllm: the hook has to sit on
    sys.meta_path ahead of the first gpu_worker import, and PYTHONPATH
    has to carry it into the spawned engine-core interpreter."""
    hook_dir = "/tmp/quail_hook"
    os.makedirs(hook_dir, exist_ok=True)
    with open(os.path.join(hook_dir, "sitecustomize.py"), "w") as f:
        f.write(_TRACE_HOOK)
    os.environ["QUAIL_XFER_TRACE"] = trace_path
    os.environ["PYTHONPATH"] = (hook_dir + os.pathsep
                                + os.environ.get("PYTHONPATH", ""))
    # PYTHONPATH covers a spawned engine core (fresh interpreters
    # import sitecustomize); exec covers this process and fork
    # children, which never rerun interpreter startup. A cached
    # system sitecustomize would make `import sitecustomize` a no-op,
    # so the source is executed directly.
    exec(compile(_TRACE_HOOK, "quail_xfer_hook", "exec"),
         {"__name__": "quail_xfer_hook"})


def _collect_xfer_events(trace_path):
    import glob
    events = []
    for p in glob.glob(trace_path + ".*"):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    events.sort(key=lambda e: e["t"])
    return events


def _xfer_summary(events, windows):
    """Aggregate the traced jobs.

    copy_GBps divides bytes by the event-timed busy seconds - the DMA
    engine's own rate, valid because transfers are strictly serialized
    on chained events. effective_GBps divides the same bytes by the
    query wall. Their ratio is the fraction of the window the channel
    actually spent copying. submit_to_finish latency includes waiting
    behind earlier jobs and the polling cadence of get_finished, so it
    is an upper bound on per-job overhead, not a pure measurement."""
    finishes = [e for e in events if e["ev"] == "finish" and e.get("secs")]
    submits = {(e["dir"], e["job"]): e["t"]
               for e in events if e["ev"] == "submit"}
    directions = {}
    for direction in ("h2d", "d2h"):
        rows = [e for e in finishes if e["dir"] == direction]
        if not rows:
            continue
        total = sum(e["bytes"] for e in rows)
        busy = sum(e["secs"] for e in rows)
        rates = sorted(e["bytes"] / e["secs"] / 1e9 for e in rows)
        lat = sorted(e["t"] - submits[(direction, e["job"])]
                     for e in rows if (direction, e["job"]) in submits)
        directions[direction] = dict(
            jobs=len(rows), gb=round(total / 1e9, 3),
            busy_s=round(busy, 3),
            copy_GBps=round(total / busy / 1e9, 2),
            job_GBps_p50=round(rates[len(rates) // 2], 2),
            job_GBps_p90=round(rates[min(len(rates) - 1,
                                         int(len(rates) * 0.9))], 2),
            submit_to_finish_p50_s=(round(lat[len(lat) // 2], 4)
                                    if lat else None))
    per_window = []
    for w in windows:
        rows = [e for e in finishes
                if e["dir"] == "h2d" and w["t0"] <= e["t"] <= w["t1"]]
        span = w["t1"] - w["t0"]
        gb = sum(e["bytes"] for e in rows) / 1e9
        busy = sum(e["secs"] for e in rows)
        per_window.append(dict(
            name=w["name"], wall_s=round(span, 3), load_jobs=len(rows),
            load_gb=round(gb, 3), load_busy_s=round(busy, 3),
            channel_busy_frac=round(busy / span, 4) if span else None,
            effective_GBps=round(gb / span, 2) if span else None,
            copy_GBps=round(gb / busy, 2) if busy else None))
    return dict(directions=directions, windows=per_window)


@app.function(image=persist_image, gpu="H100!", timeout=3600,
              memory=131072,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def persist_run(n_docs: int = 1000, stage: str = "baseline",
                cpu_gb: int = 96, connector: str = "stock",
                store_gb: int = 0) -> dict:
    import time as _time

    trace_path = "/tmp/quail_xfer"
    if stage == "store":
        # must precede the first vllm import in this process
        _install_xfer_trace(trace_path)

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine as Engine

    from quail.configs import MODELS
    from quail.runtime.engine_client import run_filter_chain
    from workload import CFG_NAME, build_corpus, yes_no_ids

    tok = AutoTokenizer.from_pretrained(MODEL)
    yes_ids, no_ids = yes_no_ids(tok)
    body_ids, q_ids, _flags = build_corpus(tok, n_docs)
    corpus = sum(len(b) for b in body_ids)
    kappa = MODELS[CFG_NAME].kappa
    kv_bytes = corpus * kappa
    sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                        allowed_token_ids=sorted(yes_ids | no_ids))
    # This cell runs on the STOCK scheduler: the offload connector
    # manages the KV lifecycle, and reconciling it with plan-owned
    # memory is open work. The point here is the channel, not the
    # executor.
    pool_budget = 700_000

    # a store capacity keeps the longest documents (recompute cost per
    # byte rises with length); the client stamps max_offload_tokens=0
    # under the threshold so short documents never occupy the cap.
    # store_gb=0 means uncapped: everything stores.
    store_min = 0
    if stage == "store":
        from quail.plan.planner import store_length_threshold
        cap_bytes = store_gb * (1 << 30) if store_gb else None
        store_min = store_length_threshold(
            [len(b) for b in body_ids], cap_bytes, kappa)

    report = dict(n_docs=n_docs, stage=stage, model=MODEL,
                  connector=connector, corpus_tokens=corpus, kappa=kappa,
                  kv_bytes_estimate=kv_bytes, store_gb=store_gb,
                  store_min_doc_tokens=store_min,
                  stored_docs=(sum(1 for b in body_ids
                                   if len(b) >= store_min)
                               if store_min else 0))
    print(f"[persist] stage {stage}: corpus {corpus:,} tokens, KV "
          f"about {kv_bytes / 1e9:.1f} GB", flush=True)

    def engine_args(store):
        kw = dict(model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
                  gpu_memory_utilization=0.92,
                  enable_prefix_caching=True, disable_log_stats=True)
        if store:
            # "quail" swaps in the coalescing worker (one transfer per
            # step, not per request); "stock" is the measured baseline
            tc = dict(kv_connector="OffloadingConnector",
                      kv_role="kv_both",
                      kv_connector_extra_config=dict(
                          cpu_bytes_to_use=cpu_gb * (1 << 30)))
            if connector == "quail":
                tc["kv_connector"] = "QuailOffloadingConnector"
                tc["kv_connector_module_path"] = "quail.engineext.offload"
            elif connector != "stock":
                raise ValueError(f"connector must be stock or quail, "
                                 f"got {connector!r}")
            kw["kv_transfer_config"] = KVTransferConfig(**tc)
        return EngineArgs(**kw)

    def one_query(engine, tag):
        return run_filter_chain(engine, sp, body_ids, q_ids,
                                pool_budget, tag=tag,
                                store_min_tokens=store_min)

    if stage == "baseline":
        engine = Engine.from_engine_args(engine_args(store=False))
        base = one_query(engine, "pb")
        engine.reset_prefix_cache()
        base2 = one_query(engine, "pb2")
        report["baseline_cold_s"] = round(base["wall"], 2)
        report["baseline_recompute_s"] = round(base2["wall"], 2)
        report["survivors"] = base["survivors"]
        print(f"[persist] baseline cold {base['wall']:.2f}s, "
              f"recompute after reset {base2['wall']:.2f}s", flush=True)
    else:
        windows = []

        def timed_query(engine, tag):
            t0 = _time.time()
            out = one_query(engine, tag)
            windows.append(dict(name=tag, t0=t0, t1=_time.time()))
            return out

        engine = Engine.from_engine_args(engine_args(store=True))
        q1 = timed_query(engine, "ps1")
        _time.sleep(8)                 # let offload writes drain
        engine.reset_prefix_cache()
        print("[persist] prediction: copy-only load bandwidth far above "
              "the ~10 GB/s wall-effective number means the loss is "
              "between jobs (orchestration); copy-only itself ~10 means "
              "the 32 KB descriptor granularity is the limit", flush=True)
        q2 = timed_query(engine, "ps2")
        engine.reset_prefix_cache()
        q3 = timed_query(engine, "ps3")
        report["store_cold_offload_s"] = round(q1["wall"], 2)
        report["store_restore_s"] = round(q2["wall"], 2)
        report["store_restore2_s"] = round(q3["wall"], 2)
        best = min(q2["wall"], q3["wall"])
        report["restore_effective_GBps"] = round(kv_bytes / best / 1e9, 2)
        report["outcomes_identical_within_store"] = (
            q1["survivors"] == q2["survivors"] == q3["survivors"])
        report["survivors"] = q1["survivors"]
        print(f"[persist] with store: cold+offload {q1['wall']:.2f}s, "
              f"restore {q2['wall']:.2f}s then {q3['wall']:.2f}s "
              f"({report['restore_effective_GBps']} GB/s effective)",
              flush=True)
        events = _collect_xfer_events(trace_path)
        report["xfer"] = _xfer_summary(events, windows)
        report["xfer_events"] = events
        suffix = "_quail" if connector == "quail" else ""
        with open(f"/results/persist_xfer_trace{suffix}.jsonl", "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        loads = report["xfer"]["directions"].get("h2d")
        if loads:
            print(f"[persist] loads: {loads['jobs']} jobs, {loads['gb']} "
                  f"GB, copy-only {loads['copy_GBps']} GB/s (per-job p50 "
                  f"{loads['job_GBps_p50']} GB/s)", flush=True)
    try:
        engine.engine_core.shutdown()
    except Exception:
        pass
    slim = {k: v for k, v in report.items() if k != "xfer_events"}
    tag = "_quail" if connector == "quail" else ""
    with open(f"/results/persist_{stage}{tag}.json", "w") as f:
        json.dump(slim, f, indent=2)
    results_vol.commit()
    return report


@app.local_entrypoint()
def main(n_docs: int = 1000, stage: str = "baseline", cpu_gb: int = 96,
         connector: str = "stock", store_gb: int = 0, out: str = ""):
    data = persist_run.remote(n_docs, stage, cpu_gb, connector, store_gb)
    events = data.pop("xfer_events", None)
    tag = "_quail" if connector == "quail" else ""
    path = out or f"results/engine/persist_{stage}{tag}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved {path}")
    if events:
        import gzip
        tp = f"results/engine/persist_xfer_trace{tag}.jsonl.gz"
        with gzip.open(tp, "wt") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        print(f"saved {tp} ({len(events)} events)")
