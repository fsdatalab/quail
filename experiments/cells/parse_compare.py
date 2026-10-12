"""Parse every ParseBench PDF page with Quail and with stock vLLM, in one container.

Both engines run LightOnOCR-3-0.8B with the same settings and do the
same work inside the timed loop, for the same 2,037 PDF pages: render
each page from its PDF (400 DPI, shrunk to 2,048 px on the longest
edge), run the checkpoint's image processor, and transcribe the page
with greedy decoding and at most 4,096 output tokens.

- Quail runs ``SELECT p.path, p.page, AI.PARSE(p.document) AS text
  FROM pages p`` through a Session with Qwen3 4B fp8 as the reading
  model and ``parser="lightonocr-3-0.8b"``. Its worker processes render
  and run the image processor; saved parse results are off.
- Stock vLLM runs one LLMEngine (max_num_seqs 256, 16,384 batched
  tokens, a context of 8,448, prefix caching off), fed from a window
  of 512 requests. Worker processes render, as for Quail; vLLM runs the
  image processor in add_request, as it does by default.

Each pass runs in its own process, so the GPU starts empty, and warms
up on 8 pages before the timed query. The passes run vLLM, Quail,
vLLM, Quail on one H100, so both engines see the same GPU and host
twice. Query time excludes model startup and the warm-up.

Run from the repository root and tee every line:

    uv run modal run --detach experiments/cells/parse_compare.py --mode run \
      --prediction "State the expected result before starting." \
      2>&1 | tee results/parse-compare-run.log

Modes: smoke reads the first SMOKE_FILES files; run reads every PDF.

Writes /results/ablations/parse_compare/<engine>_<pass>_<mode>_<run
id>.parquet (one record per page) and summary_<mode>_<run id>.json.
"""

import json
import subprocess
import sys
import time
import uuid

import modal

from quail.bench.images import gpu_image
from quail.specs import H100_USD_PER_HOUR

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

DATASET = "llamaindex/ParseBench"
DATASET_REVISION = "2805a1d940f95a203e0ae4b88be9934f7765b3fc"
OUT_DIR = "/results/ablations/parse_compare"
READER = "qwen3-4b-fp8"
PARSER = "lightonocr-3-0.8b"
SQL = "SELECT p.path, p.page, AI.PARSE(p.document) AS text FROM pages p"
WARM_SQL = SQL + " LIMIT 8"
WARM_PAGES = 8
SMOKE_FILES = 20
ORDER = ("vllm", "quail", "vllm", "quail")
# vLLM's settings in the ParseBench runs, and Quail's OCR settings
MAX_NUM_SEQS = 256
MAX_BATCHED_TOKENS = 16384
MAX_MODEL_LEN = 8448
MAX_OUTPUT_TOKENS = 4096
LONGEST_EDGE = 2048
WINDOW = 2 * MAX_NUM_SEQS
WORKERS = 24


def _files(limit):
    """ParseBench's PDF files from the cached snapshot, sorted."""
    import os
    from pathlib import Path

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(
        DATASET, repo_type="dataset", revision=DATASET_REVISION,
        allow_patterns=["docs/**", "*.jsonl"], max_workers=4))
    pdfs = set()
    for split in ("chart", "layout", "table", "text_content", "text_formatting"):
        for line in (root / f"{split}.jsonl").open():
            pdfs.add(str(root / json.loads(line)["pdf"]))
    files = sorted(path for path in pdfs if path.lower().endswith(".pdf"))
    return files[:limit] if limit else files


class _GpuSampler:
    """Sample GPU utilization with nvidia-smi every 100 ms."""

    def __enter__(self):
        self.process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE, text=True)
        return self

    def __exit__(self, *exc):
        self.process.terminate()
        samples = [float(x) for x in self.process.communicate()[0].split()]
        self.mean = sum(samples) / max(len(samples), 1)


def _render(item):
    """Render one page in a worker process."""
    from quail.pdf import render_page

    return render_page(*item)


def _vllm_pass(files, stem) -> dict:
    """Parse every page with stock vLLM; time rendering, processing, and decoding."""
    import multiprocessing
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    from quail.pdf import read_pdfs
    from quail.specs import MODELS

    spec = MODELS[PARSER]
    pages = read_pdfs(files).pages()
    items = [(pages[row][0].path, pages[row][1][0], LONGEST_EDGE)
             for row in range(len(pages))]
    # the workers start while the model loads, as Quail's do
    pool = ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context("spawn"))
    processor = AutoProcessor.from_pretrained(spec.hf_name, revision=spec.revision)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}]}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    started = time.perf_counter()
    llm = LLM(model=spec.hf_name, revision=spec.revision,
              max_model_len=MAX_MODEL_LEN, max_num_seqs=MAX_NUM_SEQS,
              max_num_batched_tokens=MAX_BATCHED_TOKENS,
              enable_prefix_caching=False, mm_processor_cache_gb=0,
              limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.9,
              seed=0)
    load_s = time.perf_counter() - started
    engine = llm.llm_engine
    params = SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS,
                            skip_special_tokens=True)

    def run(chosen):
        ahead, keys, records = deque(), {}, []
        next_item, in_flight, add_s = 0, 0, 0.0
        started = time.perf_counter()
        while True:
            while next_item < len(chosen) and len(ahead) < WINDOW:
                ahead.append((next_item, pool.submit(_render, chosen[next_item])))
                next_item += 1
            while ahead and in_flight < WINDOW and (
                    ahead[0][1].done() or not engine.has_unfinished_requests()):
                index, future = ahead.popleft()
                picture = future.result()
                added = time.perf_counter()
                keys[str(index)] = chosen[index]
                engine.add_request(str(index), {
                    "prompt": prompt, "multi_modal_data": {"image": picture}}, params)
                add_s += time.perf_counter() - added
                in_flight += 1
                if next_item < len(chosen):
                    ahead.append((next_item, pool.submit(_render, chosen[next_item])))
                    next_item += 1
            if not engine.has_unfinished_requests():
                if not ahead:
                    break
                continue
            for output in engine.step():
                if not output.finished:
                    continue
                path, page, _ = keys.pop(output.request_id)
                completion = output.outputs[0]
                records.append(dict(
                    path=path, page=page,
                    prompt_tokens=len(output.prompt_token_ids),
                    output_tokens=len(completion.token_ids),
                    text=completion.text))
                in_flight -= 1
        return records, time.perf_counter() - started, add_s

    run(items[:WARM_PAGES])
    with _GpuSampler() as sampler:
        records, wall_s, add_s = run(items)
    pool.shutdown()
    pq.write_table(pa.Table.from_pylist(records), f"{stem}.parquet")
    return dict(engine="vllm", load_s=load_s, query_s=wall_s, add_request_s=add_s,
                pages=len(records),
                input_tokens=sum(r["prompt_tokens"] for r in records),
                output_tokens=sum(r["output_tokens"] for r in records),
                gpu_utilization_mean=sampler.mean)


def _quail_pass(files, stem) -> dict:
    """Run the AI.PARSE query through a Session; time planning and execution."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail

    started = time.perf_counter()
    session = quail.Session(quail.EngineConfig(
        model=READER, device="h100-sxm", parser=PARSER, save_parses=False))
    session.register("pages", quail.read_pdfs(files))
    # the warm-up loads both models, the parser, and its decode graphs
    session.sql(WARM_SQL).run().collect()
    load_s = time.perf_counter() - started
    with _GpuSampler() as sampler:
        started = time.perf_counter()
        result = session.sql(SQL).run()
        rows = result.collect().to_pylist()
        wall_s = time.perf_counter() - started
    parse = result.report["node_metrics"]["parse:p"]
    session.close()
    pq.write_table(pa.Table.from_pylist(
        [dict(path=r["p.path"], page=r["p.page"], text=r["text"]) for r in rows]),
        f"{stem}.parquet")
    return dict(engine="quail", load_s=load_s, query_s=wall_s,
                planning_s=result.report.get("planning_s"),
                pages=len(rows), input_tokens=parse.get("parse_input_tokens"),
                output_tokens=parse.get("parse_output_tokens"),
                truncated_pages=parse.get("parse_truncated_pages"),
                gpu_utilization_mean=sampler.mean)


def _agreement(a, b) -> int:
    """Pages whose text is identical in two passes' records."""
    import pyarrow.parquet as pq

    texts = {(r["path"], r["page"]): r["text"]
             for r in pq.read_table(f"{a}.parquet").to_pylist()}
    return sum(texts.get((r["path"], r["page"])) == r["text"]
               for r in pq.read_table(f"{b}.parquet").to_pylist())


@app.function(image=image, gpu="H100!", cpu=32, memory=131072,
              timeout=3 * 3600, volumes=volumes)
def compare(mode: str, run_id: str) -> dict:
    """Run the passes in ORDER, each in its own process, and compare them."""
    from pathlib import Path

    files = _files(SMOKE_FILES if mode == "smoke" else None)
    Path("/tmp/files.json").write_text(json.dumps(files))
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    passes, stems = [], []
    for index, engine in enumerate(ORDER):
        stem = f"{OUT_DIR}/{engine}_{index}_{mode}_{run_id}"
        out = f"/tmp/pass_{index}.json"
        print(f"pass {index}: {engine}", flush=True)
        subprocess.run([sys.executable, __file__, engine, "/tmp/files.json", stem, out],
                       check=True)
        result = json.loads(Path(out).read_text())
        seconds = result["query_s"]
        result.update(pass_index=index,
                      usd_per_query=seconds / 3600 * H100_USD_PER_HOUR,
                      input_tokens_per_second=result["input_tokens"] / seconds,
                      pages_per_second=result["pages"] / seconds)
        print(json.dumps(result), flush=True)
        passes.append(result)
        stems.append(stem)
        results_vol.commit()
    vllm = [i for i, engine in enumerate(ORDER) if engine == "vllm"]
    ours = [i for i, engine in enumerate(ORDER) if engine == "quail"]
    summary = dict(
        run_id=run_id, mode=mode, files=len(files), order=list(ORDER),
        passes=passes,
        identical_quail_vs_vllm=_agreement(stems[vllm[0]], stems[ours[0]]),
        identical_vllm_vs_vllm=_agreement(stems[vllm[0]], stems[vllm[1]]),
        identical_quail_vs_quail=_agreement(stems[ours[0]], stems[ours[1]]))
    path = f"{OUT_DIR}/summary_{mode}_{run_id}.json"
    Path(path).write_text(json.dumps(summary, indent=1))
    results_vol.commit()
    summary["path"] = path
    return summary


@app.local_entrypoint()
def main(mode: str = "smoke", prediction: str = ""):
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    print(f"prediction: {prediction}", flush=True)
    print(f"run id: {run_id}", flush=True)
    call = compare.spawn(mode, run_id)
    print(f"function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=1, default=str), flush=True)


if __name__ == "__main__" and len(sys.argv) == 5:
    # one pass, in its own process: engine, the JSON file listing the
    # PDFs, the record stem, and the output file
    from pathlib import Path

    engine, files, stem, out = sys.argv[1], sys.argv[2], *sys.argv[3:]
    files = json.loads(Path(files).read_text())
    run = _vllm_pass if engine == "vllm" else _quail_pass
    Path(out).write_text(json.dumps(run(files, stem)))
