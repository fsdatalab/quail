"""Profile ten seconds of steady-state LightOnOCR parsing on Quail's executor.

Parses ParseBench pages with PageParser, as AI.PARSE does, and once
START_AFTER pages have finished (every lane is busy by then) records a
torch.profiler trace for WINDOW_S seconds, then stops. The trace's GPU
kernels are summed per CUDA stream and per kernel family: the vision
encoder runs on its own stream when encode_ahead is on, and prefill
passes and decode graphs run on the default stream. GPU clocks, power,
and utilization are sampled every 100 ms through the window.

Each configuration runs in its own process: encode_ahead 16 (the
vision encoder on a side stream, beside prefill and decode, as
AI.PARSE runs it) and encode_ahead 0 (the encoder inside each prefill
pass).

Run from the repository root and tee every line:

    uv run modal run experiments/cells/parse_window_profile.py \
      --prediction "State the expected result before starting." \
      2>&1 | tee results/parse-window-profile.log

Writes /results/ablations/parse_profile/window_<encode_ahead>_<run
id>.json (the tables) and .trace.json (the Chrome trace).
"""

import json
import re
import subprocess
import sys
import time
import uuid

import modal

from quail.bench.images import gpu_image

app = modal.App("quail-milestone1")
image = gpu_image()
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
volumes = {
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/kernels": kernel_cache,
    "/results": results_vol,
}

MODEL = "lightonocr-3-0.8b"
PAGES_PATH = "/results/ablations/lightonocr_parsebench/pages.parquet"
OUT_DIR = "/results/ablations/parse_profile"
LANES = 256
MAX_OUTPUT_TOKENS = 4096
CONTEXT_TOKENS = 8448
BUDGET = 16384
WORKERS = 24
WINDOW = 96
START_AFTER = 300
WINDOW_S = 10.0
CONFIGURATIONS = (16, 0)
FAMILIES = (
    ("attention", r"flash|fmha|attn"),
    ("linear attention", r"chunk_|fused_recurrent|causal_conv1d|gated_delta|"
                         r"l2norm|post_conv|recompute_w_u|solve_tril"),
    ("matmul", r"gemm|cutlass|nvjet|xmma|cublas|deep_gemm|matmul"),
    ("norm and activation", r"norm|silu|gelu|act_and_mul|rotary|rope"),
    ("copy", r"copy|memcpy|memset|cat|index|gather|scatter|fill"),
)


def _pages():
    """Yield ((pdf, page), PNG bytes) in the prepared order."""
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(PAGES_PATH).iter_batches(
            batch_size=64, columns=["pdf", "page", "png"]):
        for row in batch.to_pylist():
            yield (row["pdf"], row["page"]), row["png"]


def _family(name: str) -> str:
    for family, pattern in FAMILIES:
        if re.search(pattern, name, re.IGNORECASE):
            return family
    return "other"


def _busy(intervals) -> float:
    """Microseconds covered by the union of (start, end) intervals."""
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def _kernels(trace_path: str, window_us: float) -> dict:
    """Sum the trace's GPU kernels per stream and family."""
    events = json.load(open(trace_path))["traceEvents"]
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy",
                                                     "gpu_memset")]
    streams = {}
    for event in kernels:
        stream = streams.setdefault(event["args"].get("stream"), dict(
            intervals=[], families={}, names={}))
        stream["intervals"].append((event["ts"], event["ts"] + event["dur"]))
        family = _family(event["name"])
        stream["families"][family] = stream["families"].get(family, 0) + event["dur"]
        stream["names"][event["name"]] = stream["names"].get(event["name"], 0) \
            + event["dur"]
    out = {}
    for stream, value in streams.items():
        out[str(stream)] = dict(
            busy_share=_busy(value["intervals"]) / window_us,
            kernel_s=sum(value["families"].values()) / 1e6,
            families_s={k: v / 1e6 for k, v in sorted(
                value["families"].items(), key=lambda item: -item[1])},
            top_kernels_s=[(name[:100], dur / 1e6) for name, dur in sorted(
                value["names"].items(), key=lambda item: -item[1])[:15]])
    every = [interval for value in streams.values()
             for interval in value["intervals"]]
    return dict(gpu_busy_share=_busy(every) / window_us, streams=out)


def _profile_pass(encode_ahead: int, stem: str) -> dict:
    """Parse until steady, profile WINDOW_S seconds, and stop."""
    import torch
    from transformers import AutoProcessor

    from quail.backends.quail.executor.model import resolve_model_path
    from quail.backends.quail.executor.parse import PageParser, PagePreparer
    from quail.specs import MODELS

    spec = MODELS[MODEL]
    path = resolve_model_path(spec.hf_name, spec.revision)
    prompt = AutoProcessor.from_pretrained(path).apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}]}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    preparer = PagePreparer(path, prompt, workers=WORKERS)
    parser = PageParser(spec, lanes=LANES, max_output_tokens=MAX_OUTPUT_TOKENS,
                        context_tokens=CONTEXT_TOKENS, budget=BUDGET,
                        encode_ahead=encode_ahead)
    for future in preparer.started:
        future.result()
    run = parser.parse(preparer.pages(_pages(), window=WINDOW))
    finished = 0
    for _ in run:
        finished += 1
        if finished == START_AFTER:
            break
    before = dict(parser.stats)
    sampler = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,clocks.mem,"
         "power.draw,temperature.gpu,utilization.gpu",
         "--format=csv,noheader,nounits", "-lms", "100"],
        stdout=subprocess.PIPE, text=True)
    profiler = torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
    profiler.start()
    started = time.perf_counter()
    in_window = 0
    for _ in run:
        in_window += 1
        if time.perf_counter() - started >= WINDOW_S:
            break
    torch.cuda.synchronize()
    window_s = time.perf_counter() - started
    profiler.stop()
    sampler.terminate()
    run.close()
    preparer.close()
    samples = [[float(x) for x in line.split(",")]
               for line in sampler.communicate()[0].splitlines() if line.strip()]
    columns = list(zip(*samples)) if samples else [[]] * 6
    after = parser.stats
    delta = {name: after[name] - before.get(name, 0) for name in after
             if isinstance(after[name], (int, float))}
    trace = f"{stem}.trace.json"
    profiler.export_chrome_trace(trace)
    cpu = sorted(profiler.key_averages(), key=lambda e: -e.self_cpu_time_total)
    return dict(
        encode_ahead=encode_ahead, window_s=window_s, pages_finished=in_window,
        stats_in_window=delta,
        gpu_samples=dict(
            sm_mhz_mean=sum(columns[0]) / max(len(columns[0]), 1),
            sm_mhz_max=max(columns[1], default=0),
            mem_mhz_mean=sum(columns[2]) / max(len(columns[2]), 1),
            power_w_mean=sum(columns[3]) / max(len(columns[3]), 1),
            temperature_c_max=max(columns[4], default=0),
            utilization_mean=sum(columns[5]) / max(len(columns[5]), 1)),
        **_kernels(trace, window_s * 1e6),
        top_cpu_ops_s=[(e.key[:80], e.self_cpu_time_total / 1e6, e.count)
                       for e in cpu[:15]],
        trace=trace)


@app.function(image=image, gpu="H100!", cpu=32, memory=131072, timeout=3600,
              volumes=volumes)
def profile(run_id: str) -> dict:
    """Profile each configuration in its own process."""
    from pathlib import Path

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    results = []
    for encode_ahead in CONFIGURATIONS:
        stem = f"{OUT_DIR}/window_{encode_ahead}_{run_id}"
        out = f"/tmp/window_{encode_ahead}.json"
        subprocess.run([sys.executable, __file__, str(encode_ahead), stem, out],
                       check=True)
        result = json.loads(Path(out).read_text())
        Path(f"{stem}.json").write_text(json.dumps(result, indent=1))
        results_vol.commit()
        print(json.dumps(result, indent=1), flush=True)
        results.append(result)
    return dict(run_id=run_id, results=results)


@app.local_entrypoint()
def main(prediction: str = ""):
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    print(f"prediction: {prediction}", flush=True)
    print(f"run id: {run_id}", flush=True)
    call = profile.spawn(run_id)
    print(f"function call id: {call.object_id}", flush=True)
    call.get()


if __name__ == "__main__" and len(sys.argv) == 4:
    from pathlib import Path

    encode_ahead, stem, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    Path(out).write_text(json.dumps(_profile_pass(encode_ahead, stem)))
