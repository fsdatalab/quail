"""Parse every ParseBench page with LightOnOCR-3-0.8B on stock vLLM.

Measures the stock vLLM baseline for AI.PARSE (issue #148): pages per
second, output tokens per page, how busy the GPU is, and, in profile
mode, which kernels a steady decode step spends its time in. Each
page's stored text (the PDF's text layer, read with pypdfium2) is
saved beside its OCR output so a CPU script can count how much of the
output a draft from the stored text would cover.

Pages are rendered as the model card says: 400 DPI, then shrunk to
2,048 px on the longest edge. The prompt is the empty transcription
prompt with thinking off; decoding is greedy with at most 4,096 output
tokens. Pages stream into vLLM's engine from a bounded window so the
images never all sit in memory at once.

Run from the repository root and tee every line:

    uv run modal run --detach experiments/cells/lightonocr_parsebench_vllm.py \
      --mode prepare 2>&1 | tee results/lightonocr-parsebench-prepare.log
    uv run modal run --detach experiments/cells/lightonocr_parsebench_vllm.py \
      --mode run --max-num-seqs 256 \
      --prediction "State the expected result before starting." \
      2>&1 | tee results/lightonocr-parsebench-run-256.log

Modes: prepare renders the pages and reads their stored text; smoke
parses the first 20 pages; run parses every page; profile parses the
first PROFILE_PAGES pages in-process and profiles PROFILE_STEPS decode
steps once the batch is full.

Writes /results/ablations/lightonocr_parsebench/: pages.parquet from
prepare, and vllm_<mode>_<max_num_seqs>_<run id>.parquet (one record
per page) and .json (the summary) from the other modes.
"""

import json
import time
import uuid

import modal

from quail.bench.images import cpu_image, gpu_image

PYPDFIUM2 = "pypdfium2==5.13.0"
app = modal.App("quail-milestone1")
image = gpu_image(packages=(PYPDFIUM2,))
render_image = cpu_image(packages=(PYPDFIUM2,))
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
volumes = {
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/kernels": kernel_cache,
    "/results": results_vol,
}

DATASET = "llamaindex/ParseBench"
DATASET_REVISION = "2805a1d940f95a203e0ae4b88be9934f7765b3fc"
MODEL = "lightonai/LightOnOCR-3-0.8B"
MODEL_REVISION = "36d8636925327c671bd8e479c517a004f7566bd9"
OUT_DIR = "/results/ablations/lightonocr_parsebench"
PAGES_PATH = f"{OUT_DIR}/pages.parquet"
SPLITS = ("chart", "layout", "table", "text_content", "text_formatting")
RENDER_DPI = 400
LONGEST_EDGE = 2048
MAX_OUTPUT_TOKENS = 4096
# a square page at 2,048 px is 4,096 image tokens; the context holds
# that, the chat turn, and a full answer
MAX_MODEL_LEN = 8448
MAX_BATCHED_TOKENS = 16384
SMOKE_PAGES = 20
PROFILE_PAGES = 600
PROFILE_WARM_STEPS = 150
PROFILE_STEPS = 40
TOP_KERNELS = 60


def _render(path: str) -> list[dict] | str:
    """Render every page of one PDF, or load one page image, with its stored text.

    Returns:
        One dict per page, or PDFium's error message when the file does
        not open.
    """
    import io

    import pypdfium2 as pdfium
    from PIL import Image

    if path.lower().endswith((".png", ".jpg", ".jpeg")):
        # ParseBench's layout split also holds page images; they have no
        # stored text and keep their size unless longer than the edge
        started = time.perf_counter()
        picture = Image.open(path).convert("RGB")
        picture.thumbnail((LONGEST_EDGE, LONGEST_EDGE))
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        return [dict(page=1, width=picture.width, height=picture.height,
                     png=buffer.getvalue(),
                     render_ms=(time.perf_counter() - started) * 1e3,
                     stored_text="", extract_ms=0.0)]
    try:
        pdf = pdfium.PdfDocument(path)
    except pdfium.PdfiumError as error:
        return str(error)
    pages = []
    for index in range(len(pdf)):
        page = pdf[index]
        started = time.perf_counter()
        textpage = page.get_textpage()
        text = textpage.get_text_bounded()
        extract_ms = (time.perf_counter() - started) * 1e3
        textpage.close()
        started = time.perf_counter()
        bitmap = page.render(scale=RENDER_DPI / 72)
        picture = bitmap.to_pil().convert("RGB")
        picture.thumbnail((LONGEST_EDGE, LONGEST_EDGE))
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        render_ms = (time.perf_counter() - started) * 1e3
        pages.append(dict(page=index + 1, width=picture.width,
                          height=picture.height, png=buffer.getvalue(),
                          render_ms=render_ms, stored_text=text,
                          extract_ms=extract_ms))
        page.close()
    pdf.close()
    return pages


def _download():
    """Download the dataset, retrying when the Hub rate-limits the requests."""
    import os
    from pathlib import Path

    # anonymous downloads through the Hub's xet service hit 429 at the
    # default 8 workers on 2,084 files (run of 2026-10-12)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download

    for attempt in range(6):
        try:
            return Path(snapshot_download(
                DATASET, repo_type="dataset", revision=DATASET_REVISION,
                allow_patterns=["docs/**", "*.jsonl"], max_workers=4))
        except Exception as error:
            print(f"download attempt {attempt + 1} failed: {error}", flush=True)
            hf_cache.commit()
            time.sleep(30 * (attempt + 1))
    raise RuntimeError("the dataset download kept failing")


@app.function(image=render_image, cpu=32, memory=65536, timeout=3600,
              volumes=volumes)
def prepare() -> dict:
    """Render ParseBench's pages and save them with their stored text."""
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    root = _download()
    categories = {}
    for split in SPLITS:
        for line in (root / f"{split}.jsonl").open():
            row = json.loads(line)
            categories.setdefault(row["pdf"], set()).add(row["category"])
    pdfs = sorted(categories)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=32) as pool:
        rendered = list(pool.map(_render, [str(root / pdf) for pdf in pdfs],
                                 chunksize=4))
    seconds = time.perf_counter() - started
    rows, unreadable = [], {}
    for pdf, pages in zip(pdfs, rendered):
        if isinstance(pages, str):
            unreadable[pdf] = pages
            continue
        for page in pages:
            rows.append(dict(pdf=pdf, folder=pdf.split("/")[1],
                             categories=sorted(categories[pdf]), **page))
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), PAGES_PATH, row_group_size=64)
    results_vol.commit()
    empty = sum(1 for row in rows if not row["stored_text"].strip())
    return dict(pdfs=len(pdfs), pages=len(rows), wall_s=seconds,
                render_ms_mean=sum(r["render_ms"] for r in rows) / len(rows),
                extract_ms_mean=sum(r["extract_ms"] for r in rows) / len(rows),
                pages_without_stored_text=empty, unreadable=unreadable,
                path=PAGES_PATH)


def _page_stream(limit: int | None, window: int):
    """Yield (row, PIL image) in page order, decoding ahead in a thread."""
    import io
    import queue
    import threading

    import pyarrow.parquet as pq
    from PIL import Image

    columns = ["pdf", "folder", "page", "width", "height", "png"]
    ready = queue.Queue(maxsize=window)

    def produce():
        count = 0
        for batch in pq.ParquetFile(PAGES_PATH).iter_batches(
                batch_size=64, columns=columns):
            for row in batch.to_pylist():
                if limit is not None and count >= limit:
                    ready.put(None)
                    return
                picture = Image.open(io.BytesIO(row.pop("png")))
                picture.load()
                ready.put((row, picture))
                count += 1
        ready.put(None)

    threading.Thread(target=produce, daemon=True).start()
    while (item := ready.get()) is not None:
        yield item


class _GpuSampler:
    """Sample GPU utilization with nvidia-smi every 100 ms."""

    def __enter__(self):
        import subprocess

        self.process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE, text=True)
        return self

    def __exit__(self, *exc):
        self.process.terminate()
        lines = self.process.communicate()[0].splitlines()
        samples = [tuple(float(x) for x in line.split(","))
                   for line in lines if line.strip()]
        self.utilization = [s[0] for s in samples]
        self.memory_mib = [s[1] for s in samples]


def _kernel_table(profiler) -> list[dict]:
    """Return the profiled CUDA kernels by total GPU time."""
    rows = []
    for event in profiler.key_averages():
        device_us = getattr(event, "self_device_time_total", 0)
        if device_us <= 0:
            continue
        rows.append(dict(name=event.key, calls=event.count,
                         gpu_us=device_us))
    rows.sort(key=lambda row: -row["gpu_us"])
    return rows


@app.function(image=image, gpu="H100!", memory=98304, timeout=2 * 3600,
              volumes=volumes)
def parse(mode: str, max_num_seqs: int, run_id: str) -> dict:
    """Parse pages with LightOnOCR on stock vLLM and save every output."""
    import os
    from pathlib import Path

    if mode == "profile":
        # the engine core runs in this process, so torch.profiler sees
        # its kernels
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    limit = {"smoke": SMOKE_PAGES, "profile": PROFILE_PAGES}.get(mode)
    processor = AutoProcessor.from_pretrained(MODEL, revision=MODEL_REVISION)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}]}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    started = time.perf_counter()
    llm = LLM(model=MODEL, revision=MODEL_REVISION,
              max_model_len=MAX_MODEL_LEN, max_num_seqs=max_num_seqs,
              max_num_batched_tokens=MAX_BATCHED_TOKENS,
              enable_prefix_caching=False, mm_processor_cache_gb=0,
              limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.9,
              seed=0)
    load_s = time.perf_counter() - started
    engine = llm.llm_engine
    params = SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS,
                            skip_special_tokens=True)
    window = 2 * max_num_seqs
    stream = _page_stream(limit, window)
    keys, added, records, steps = {}, {}, [], []
    kernels, profile = None, None
    in_flight, exhausted, next_id = 0, False, 0
    with _GpuSampler() as sampler:
        started = time.perf_counter()
        while True:
            while not exhausted and in_flight < window:
                item = next(stream, None)
                if item is None:
                    exhausted = True
                    break
                row, picture = item
                request_id = str(next_id)
                next_id += 1
                keys[request_id] = row
                added[request_id] = time.perf_counter() - started
                engine.add_request(request_id, {
                    "prompt": prompt,
                    "multi_modal_data": {"image": picture}}, params)
                in_flight += 1
            if not engine.has_unfinished_requests():
                break
            profiling = (mode == "profile" and kernels is None
                         and len(steps) == PROFILE_WARM_STEPS)
            if profiling:
                torch.cuda.synchronize()
                window_started = time.perf_counter()
                with torch.profiler.profile(activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    for _ in range(PROFILE_STEPS):
                        outputs = engine.step()
                        steps.append(dict(
                            s=time.perf_counter() - started,
                            running=engine.get_num_unfinished_requests()))
                        records += _finished(outputs, keys, added, started)
                    torch.cuda.synchronize()
                window_s = time.perf_counter() - window_started
                kernels = _kernel_table(profiler)
                profile = dict(steps=PROFILE_STEPS, wall_s=window_s,
                               gpu_s=sum(k["gpu_us"] for k in kernels) / 1e6,
                               running=steps[-1]["running"])
                in_flight = engine.get_num_unfinished_requests()
                continue
            step_started = time.perf_counter()
            outputs = engine.step()
            steps.append(dict(s=time.perf_counter() - started,
                              step_s=time.perf_counter() - step_started,
                              running=engine.get_num_unfinished_requests()))
            finished = _finished(outputs, keys, added, started)
            records += finished
            in_flight -= len(finished)
        wall_s = time.perf_counter() - started
    output_tokens = [r["output_tokens"] for r in records]
    summary = dict(
        run_id=run_id, mode=mode, model=MODEL, model_revision=MODEL_REVISION,
        dataset_revision=DATASET_REVISION, max_num_seqs=max_num_seqs,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        max_model_len=MAX_MODEL_LEN, max_output_tokens=MAX_OUTPUT_TOKENS,
        prefix_caching=False, sampling="greedy", load_s=load_s,
        pages=len(records), wall_s=wall_s, pages_per_s=len(records) / wall_s,
        output_tokens=sum(output_tokens),
        output_tokens_per_s=sum(output_tokens) / wall_s,
        image_tokens=sum(r["prompt_tokens"] for r in records),
        pages_at_max_tokens=sum(1 for r in records
                                if r["output_tokens"] >= MAX_OUTPUT_TOKENS),
        engine_steps=len(steps),
        gpu_utilization_mean=(sum(sampler.utilization)
                              / max(len(sampler.utilization), 1)),
        gpu_memory_mib_max=max(sampler.memory_mib, default=0),
        profile=profile, kernels=(kernels or [])[:TOP_KERNELS],
        steps=steps)
    stem = f"{OUT_DIR}/vllm_{mode}_{max_num_seqs}_{run_id}"
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), f"{stem}.parquet")
    Path(f"{stem}.json").write_text(json.dumps(summary, indent=1))
    results_vol.commit()
    summary.pop("steps")
    summary["kernels"] = summary["kernels"][:15]
    summary["paths"] = [f"{stem}.parquet", f"{stem}.json"]
    return summary


def _finished(outputs, keys, added, started) -> list[dict]:
    """Turn the finished requests among `outputs` into page records."""
    now = time.perf_counter() - started
    records = []
    for output in outputs:
        if not output.finished:
            continue
        row = keys.pop(output.request_id)
        completion = output.outputs[0]
        records.append(dict(
            **row, request_id=output.request_id,
            prompt_tokens=len(output.prompt_token_ids),
            output_tokens=len(completion.token_ids),
            token_ids=list(completion.token_ids), text=completion.text,
            finish_reason=completion.finish_reason,
            added_s=added.pop(output.request_id), finished_s=now))
    return records


@app.local_entrypoint()
def main(mode: str = "smoke", max_num_seqs: int = 256, prediction: str = ""):
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    print(f"prediction: {prediction}", flush=True)
    print(f"run id: {run_id}", flush=True)
    if mode == "prepare":
        call = prepare.spawn()
    else:
        call = parse.spawn(mode, max_num_seqs, run_id)
    print(f"function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=1, default=str), flush=True)
