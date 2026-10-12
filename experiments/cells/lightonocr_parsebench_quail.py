"""Parse ParseBench pages with LightOnOCR-3-0.8B on Quail's executor.

Runs quail.backends.quail.executor.parse.PageParser over the pages
experiments/cells/lightonocr_parsebench_vllm.py prepared, with the same
prompt, greedy decoding, output limit, and context as the vLLM runs,
and compares every page's output tokens with the saved vLLM run. Page
preparation (the image processor) runs in worker processes ahead of
the GPU; the workers start while the model loads, and the startup time
reported (load_s) covers both.

Run from the repository root and tee every line:

    uv run modal run --detach experiments/cells/lightonocr_parsebench_quail.py \
      --mode smoke --prediction "State the expected result before starting." \
      2>&1 | tee results/lightonocr-parsebench-quail-smoke.log

--encode-ahead 0 runs the vision tower inside each prefill instead of
in a thread on its own stream.

Modes: smoke parses the first 20 pages; run parses every page.

Writes /results/ablations/lightonocr_parsebench/quail_<mode>_<lanes>_<run
id>.parquet (one record per page) and .json (the summary).
"""

import json
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
OUT_DIR = "/results/ablations/lightonocr_parsebench"
PAGES_PATH = f"{OUT_DIR}/pages.parquet"
# the vLLM run the outputs are compared with: max_num_seqs 256
VLLM_RUN = f"{OUT_DIR}/vllm_run_256_20261012T003224Z-64e10e.parquet"
MAX_OUTPUT_TOKENS = 4096
CONTEXT_TOKENS = 8448
# rows per step, decode rows included: vLLM's max_num_batched_tokens
# in the baseline runs
BUDGET = 16384
ENCODE_AHEAD = 16
SMOKE_PAGES = 20
WORKERS = 24
WINDOW = 96


def _pages(limit):
    """Yield ((pdf, page), PNG bytes) in the prepared order."""
    import pyarrow.parquet as pq

    count = 0
    for batch in pq.ParquetFile(PAGES_PATH).iter_batches(
            batch_size=64, columns=["pdf", "page", "png"]):
        for row in batch.to_pylist():
            if limit is not None and count >= limit:
                return
            yield (row["pdf"], row["page"]), row["png"]
            count += 1


def _agreement(records) -> dict:
    """Compare each page's tokens with the saved vLLM run's."""
    import pyarrow.parquet as pq

    vllm = {(r["pdf"], r["page"]): r["token_ids"] for r in pq.read_table(
        VLLM_RUN, columns=["pdf", "page", "token_ids"]).to_pylist()}
    exact, shares = 0, []
    for record in records:
        theirs = vllm[(record["pdf"], record["page"])]
        ours = record["token_ids"]
        common = 0
        for a, b in zip(ours, theirs):
            if a != b:
                break
            common += 1
        exact += ours == theirs
        shares.append(common / max(len(theirs), 1))
        record["common_prefix_tokens"] = common
        record["vllm_output_tokens"] = len(theirs)
    return dict(pages_identical=exact,
                common_prefix_share_mean=sum(shares) / len(shares))


@app.function(image=image, gpu="H100!", cpu=32, memory=131072,
              timeout=2 * 3600, volumes=volumes)
def parse(mode: str, lanes: int, run_id: str, encode_ahead: int) -> dict:
    """Parse pages on Quail's executor and save every output."""
    import subprocess
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoProcessor

    from quail.backends.quail.executor.model import resolve_model_path
    from quail.backends.quail.executor.parse import PageParser, PagePreparer
    from quail.specs import MODELS

    spec = MODELS[MODEL]
    started = time.perf_counter()
    path = resolve_model_path(spec.hf_name, spec.revision)
    prompt = AutoProcessor.from_pretrained(path).apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}]}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    # the workers start while the model loads
    preparer = PagePreparer(path, prompt, workers=WORKERS)
    parser = PageParser(spec, lanes=lanes, max_output_tokens=MAX_OUTPUT_TOKENS,
                        context_tokens=CONTEXT_TOKENS, budget=BUDGET,
                        encode_ahead=encode_ahead)
    if prompt != parser.prompt():
        raise ValueError("the workers' prompt differs from the parser's")
    for future in preparer.started:
        future.result()
    load_s = time.perf_counter() - started
    limit = SMOKE_PAGES if mode == "smoke" else None
    sampler = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
         "-lms", "100"], stdout=subprocess.PIPE, text=True)
    records = []
    started = time.perf_counter()
    pages = preparer.pages(_pages(limit), window=WINDOW)
    for (pdf, page), tokens, prompt_tokens in parser.parse(pages):
        records.append(dict(
            pdf=pdf, page=page, prompt_tokens=prompt_tokens,
            output_tokens=len(tokens), token_ids=tokens,
            finished_s=time.perf_counter() - started))
    wall_s = time.perf_counter() - started
    preparer.close()
    sampler.terminate()
    utilization = [float(x) for x in sampler.communicate()[0].split()]
    tokenizer = parser.processor.tokenizer
    for record in records:
        record["text"] = tokenizer.decode(record["token_ids"],
                                          skip_special_tokens=True)
    output_tokens = sum(r["output_tokens"] for r in records)
    summary = dict(
        run_id=run_id, mode=mode, model=MODEL, backend="quail", lanes=lanes,
        max_output_tokens=MAX_OUTPUT_TOKENS, context_tokens=CONTEXT_TOKENS,
        budget=BUDGET,
        encode_ahead=encode_ahead, workers=WORKERS, sampling="greedy",
        load_s=load_s, pages=len(records), wall_s=wall_s,
        pages_per_s=len(records) / wall_s, output_tokens=output_tokens,
        output_tokens_per_s=output_tokens / wall_s,
        image_tokens=sum(r["prompt_tokens"] for r in records),
        pages_at_max_tokens=sum(1 for r in records
                                if r["output_tokens"] >= MAX_OUTPUT_TOKENS),
        gpu_utilization_mean=sum(utilization) / max(len(utilization), 1),
        compared_with=VLLM_RUN, **parser.stats, **_agreement(records))
    stem = f"{OUT_DIR}/quail_{mode}_{lanes}_{run_id}"
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), f"{stem}.parquet")
    Path(f"{stem}.json").write_text(json.dumps(summary, indent=1))
    results_vol.commit()
    summary["paths"] = [f"{stem}.parquet", f"{stem}.json"]
    summary["first_pages"] = [
        dict(pdf=r["pdf"], tokens=r["output_tokens"],
             vllm_tokens=r["vllm_output_tokens"],
             common=r["common_prefix_tokens"]) for r in records[:SMOKE_PAGES]]
    return summary


@app.local_entrypoint()
def main(mode: str = "smoke", lanes: int = 256, encode_ahead: int = ENCODE_AHEAD,
         prediction: str = ""):
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    print(f"prediction: {prediction}", flush=True)
    print(f"run id: {run_id}", flush=True)
    call = parse.spawn(mode, lanes, run_id, encode_ahead)
    print(f"function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=1, default=str), flush=True)
