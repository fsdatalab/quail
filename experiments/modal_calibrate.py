"""Calibration sweeps for the step cost models.

Each cell is one engine step of a chosen composition (see
quail/plan/calib.py). The engine runs in this process, so every
request of a cell is queued before the first step and the stock vLLM
scheduler must run the cell as exactly one step; the step trace
verifies that, and the client verifies every request's cached-token
count. Per cell: 2 warmup rounds, then 5 measured rounds; the row
keeps every round and the median. A cell that fails its checks or
spreads past 10 percent is retried once, then kept with its flags.

The c6 family measures transfer rates with no engine: GPU-host copies
both directions, pinned and unpinned, container disk, and the results
volume.

Run with (tee every run):
  modal run experiments/modal_calibrate.py --families alpha
  modal run experiments/modal_calibrate.py --families c1
  modal run experiments/modal_calibrate.py --families c2,c4,c5
  modal run experiments/modal_calibrate.py --families c6
  modal run experiments/modal_calibrate.py --families all
"""

import json
import os

import modal

from workload import hf_cache, image, results_vol

app = modal.App("quail-calibrate")
calib_image = image.add_local_python_source("workload")

SPREAD_FLAG = 0.10


def _read_trace(path, mark):
    """Trace records appended since `mark` lines; returns them and the
    new mark. Records flush per step (QUAIL_STEPTRACE_FLUSH), so by
    the time a synchronous generate returns, its steps are on disk."""
    if not os.path.exists(path):
        return [], mark
    with open(path) as f:
        lines = f.readlines()
    return [json.loads(x) for x in lines[mark:]], len(lines)


def _median(xs):
    xs = sorted(xs)
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


@app.function(image=calib_image, gpu="H100!", timeout=7200, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def calibrate(families: str = "all", reps: int = 5,
              warmups: int = 2) -> str:
    # Env before any vllm import: the engine must run in this process
    # (shared clock, requests queued before the first step, trace file
    # readable per cell), and strict single-tenant mode would refuse
    # the untagged calibration requests.
    trace_path = "/tmp/calib_trace.jsonl"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["QUAIL_SINGLE_TENANT"] = "0"
    os.environ["QUAIL_STEPTRACE"] = trace_path
    os.environ["QUAIL_STEPSHAPES"] = "1"
    os.environ["QUAIL_STEPTRACE_FLUSH"] = "1"

    import gc
    import time

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from quail.plan import calib
    from workload import (IMAGE_STAMP, MODEL, build_flat_pool, kv_pool_tokens,
                          nonce_alphabet, sched_cfg, yes_no_ids)

    tag = families.replace(",", "-")
    outpath = f"/results/calibrate_{tag}.json"
    rows = []

    def emit(row):
        rows.append(row)
        print("[calib] " + json.dumps(row), flush=True)
        with open(outpath, "w") as f:
            json.dump(rows, f, indent=2)
        results_vol.commit()

    cells = calib.cells_for(families)
    run_c6 = families == "all" or "c6" in families.split(",")

    tok = AutoTokenizer.from_pretrained(MODEL)
    yes, no = yes_no_ids(tok)
    sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                        allowed_token_ids=sorted(yes | no))
    pool_ids = build_flat_pool(tok)
    alphabet = nonce_alphabet(tok)
    state = dict(counter=0, cursor=0)

    def seq(n_tokens):
        """A fresh unique sequence of exactly n_tokens tokens."""
        nonce = calib.make_nonce_ids(state["counter"], alphabet)
        state["counter"] += 1
        ids, state["cursor"] = calib.build_sequence(
            pool_ids, state["cursor"], n_tokens, nonce)
        return ids

    def round_prompts(cell, docs):
        """One measured round's prompts, in the order of
        cell["requests"]: fresh content for fresh requests, warmed
        document plus a fresh suffix for cached ones. Suffixes are
        fresh every round so only the document ever hits the cache."""
        prompts, doc_i = [], 0
        for r in cell["requests"]:
            if r["cached"]:
                prompts.append(docs[doc_i] + seq(r["new"]))
                doc_i += 1
            else:
                prompts.append(seq(r["new"]))
        return prompts

    def run_cell_once(llm, cell, mark):
        llm.reset_prefix_cache()
        docs = [seq(h) for h in cell["warm"]]
        if docs:
            llm.generate([{"prompt_token_ids": d} for d in docs], sp,
                         use_tqdm=False)
            _, mark = _read_trace(trace_path, mark)  # skip warm steps
        rounds = []
        for r in range(warmups + reps):
            prompts = round_prompts(cell, docs)
            t0 = time.monotonic()
            outs = llm.generate([{"prompt_token_ids": p} for p in prompts],
                                sp, use_tqdm=False)
            wall_ms = (time.monotonic() - t0) * 1e3
            recs, mark = _read_trace(trace_path, mark)
            ok_step, why_step = calib.check_step(cell, recs)
            cached = [int(o.num_cached_tokens or 0) for o in outs]
            ok_cache, why_cache = calib.check_cached(cell, cached)
            work = [x for x in recs if x.get("tokens", 0) > 0]
            rec = work[0] if len(work) == 1 else {}
            rounds.append(dict(
                measured=r >= warmups,
                ok=ok_step and ok_cache,
                why="; ".join(w for w in (why_step, why_cache) if w),
                wall_ms=round(wall_ms, 3),
                exec_ms=rec.get("exec_ms"),
                sched_ms=rec.get("sched_ms"),
                update_ms=rec.get("update_ms"),
                kv_used_tokens=rec.get("kv_used_tokens")))
        return rounds, mark

    def summarize(cell, rounds, attempt):
        good = [r for r in rounds if r["measured"] and r["ok"]
                and r["exec_ms"] is not None]
        execs = [r["exec_ms"] for r in good]
        med = _median(execs) if execs else None
        spread = ((max(execs) - min(execs)) / med
                  if execs and med else None)
        return dict(
            family=cell["family"], cell=cell["name"],
            n=cell["n"], b=cell["b"], p=cell["p"], a=cell["a"],
            requests=cell["requests"], warm_tokens=sum(cell["warm"]),
            rounds=rounds, attempt=attempt,
            valid=len(good) == sum(1 for r in rounds if r["measured"]),
            exec_ms_median=med,
            sched_ms_median=_median(
                [r["sched_ms"] for r in good if r["sched_ms"] is not None]
            ) if good else None,
            update_ms_median=_median(
                [r["update_ms"] for r in good if r["update_ms"] is not None]
            ) if good else None,
            wall_ms_median=_median([r["wall_ms"] for r in good])
            if good else None,
            spread=round(spread, 4) if spread is not None else None,
            stable=spread is not None and spread <= SPREAD_FLAG)

    if cells:
        boot = calib.BOOT
        print(f"[calib] boot {boot}", flush=True)
        llm = LLM(
            model=MODEL,
            kv_cache_dtype="fp8",
            max_model_len=boot["max_model_len"],
            max_num_seqs=boot["max_num_seqs"],
            max_num_batched_tokens=boot["max_num_batched_tokens"],
            gpu_memory_utilization=boot["gpu_memory_utilization"],
            enable_prefix_caching=True,
            disable_log_stats=True,
            scheduler_cls="quail.engineext.scheduler.QuailScheduler",
            compilation_config={
                "max_cudagraph_capture_size": boot["cudagraph_capture"],
                "cudagraph_capture_sizes": [boot["cudagraph_capture"]],
            })
        pool_tokens = kv_pool_tokens(llm) or 946_800
        kept, dropped = calib.feasible(cells, pool_tokens)
        emit(dict(meta="boot", families=families, boot=boot,
                  sched=sched_cfg(llm), pool_tokens=pool_tokens,
                  dropped=dropped, reps=reps, warmups=warmups,
                  stamp=IMAGE_STAMP, vllm=vllm.__version__))

        ordered = [calib.drift_cell("start")] + kept + [
            calib.drift_cell("end")]
        mark = 0
        for cell in ordered:
            rounds, mark = run_cell_once(llm, cell, mark)
            row = summarize(cell, rounds, attempt=1)
            if not (row["valid"] and row["stable"]):
                print(f"[calib] retry {cell['name']}: valid "
                      f"{row['valid']}, spread {row['spread']}",
                      flush=True)
                rounds, mark = run_cell_once(llm, cell, mark)
                row = summarize(cell, rounds, attempt=2)
            emit(row)

        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)

    if run_c6:
        for row in transport_rows(torch):
            emit(row)

    print(f"[calib] done: {len(rows)} rows -> {outpath}", flush=True)
    return json.dumps(rows)


def transport_rows(torch):
    """Transfer-rate cells, no engine. GPU-host both directions with
    and without pinned host memory (4 GiB), container disk write and
    read (16 GiB, page cache dropped between), and the results volume
    (4 GiB). Rates in bytes per second; the fits turn them into
    per-token fetch costs."""
    import time

    rows = []
    gib = 1024 ** 3

    n = 4 * gib
    dev = torch.empty(n, dtype=torch.uint8, device="cuda")
    for pinned in (False, True):
        host = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
        for direction, src, dst in (("d2h", dev, host), ("h2d", host, dev)):
            times = []
            for _ in range(5):
                torch.cuda.synchronize()
                t0 = time.monotonic()
                dst.copy_(src)
                torch.cuda.synchronize()
                times.append(time.monotonic() - t0)
            med = _median(times)
            rows.append(dict(
                family="c6", cell=f"c6_{direction}_"
                f"{'pinned' if pinned else 'unpinned'}",
                bytes=n, seconds=[round(t, 4) for t in times],
                bytes_per_s=round(n / med, 0)))
        del host
    del dev
    torch.cuda.empty_cache()

    def disk_cell(path, size, label):
        chunk = bytes(64 * 1024 * 1024)
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        t0 = time.monotonic()
        written = 0
        while written < size:
            os.write(fd, chunk)
            written += len(chunk)
        os.fsync(fd)
        write_s = time.monotonic() - t0
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        t0 = time.monotonic()
        read = 0
        while True:
            buf = os.read(fd, 64 * 1024 * 1024)
            if not buf:
                break
            read += len(buf)
        read_s = time.monotonic() - t0
        os.close(fd)
        os.unlink(path)
        return [dict(family="c6", cell=f"c6_{label}_write", bytes=written,
                     seconds=[round(write_s, 3)],
                     bytes_per_s=round(written / write_s, 0)),
                dict(family="c6", cell=f"c6_{label}_read", bytes=read,
                     seconds=[round(read_s, 3)],
                     bytes_per_s=round(read / read_s, 0))]

    rows.extend(disk_cell("/tmp/calib_disk_probe.bin", 16 * gib, "disk"))
    rows.extend(disk_cell("/results/calib_volume_probe.bin", 4 * gib,
                          "volume"))
    return rows


@app.local_entrypoint()
def main(families: str = "all", reps: int = 5, warmups: int = 2,
         out: str = ""):
    data = calibrate.remote(families=families, reps=reps, warmups=warmups)
    out = out or ("results/engine/calibrate_"
                  f"{families.replace(',', '-')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(data)
    print(f"saved {out}")
