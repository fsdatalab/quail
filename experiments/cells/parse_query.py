"""Run AI.PARSE queries through a Session on ParseBench's PDFs.

Registers ParseBench's PDF files with quail.read_pdfs and runs, on one
H100 with Qwen3 4B fp8 as the reading model:

- ``parse``: SELECT AI.PARSE(p.document) AS text for every page, with
  the LightOnOCR-3-0.8B parser. Each page's text is compared with the
  saved stock vLLM run of the same model.
- ``classify``: AI.CLASSIFY(AI.PARSE(p.document), [...]) for every
  page, once with each parser. The classification runs in rounds of
  parsed pages. Its label is compared with the ParseBench folder the
  file is in (chart, table, text_content, ...) as a plain agreement
  count: a label agrees when the folder name starts with it.

Each query runs twice in one session; the second run is reported, so
model and parser startup stay out of the query time. The first run's
report is saved too.

Run from the repository root and tee every line:

    uv run modal run --detach experiments/cells/parse_query.py --mode smoke \
      --prediction "State the expected result before starting." \
      2>&1 | tee results/parse-query-smoke.log

Modes: smoke reads the first SMOKE_FILES PDF files; run reads every one.

Writes /results/ablations/parse_query/<query>_<parser>_<mode>_<run
id>.parquet (one record per page) and .json (both runs' reports).
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

DATASET = "llamaindex/ParseBench"
DATASET_REVISION = "2805a1d940f95a203e0ae4b88be9934f7765b3fc"
OUT_DIR = "/results/ablations/parse_query"
# the stock vLLM run of LightOnOCR the parsed text is compared with
VLLM_RUN = ("/results/ablations/lightonocr_parsebench/"
            "vllm_run_256_20261012T003224Z-64e10e.parquet")
READER = "qwen3-4b-fp8"
SMOKE_FILES = 20
LABELS = ("chart", "table", "text")
PARSE_SQL = "SELECT p.path, p.page, AI.PARSE(p.document) AS text FROM pages p"
CLASSIFY_SQL = (
    "SELECT p.path, p.page, AI.CLASSIFY(AI.PARSE(p.document), "
    "['chart', 'table', 'text']) AS kind FROM pages p")
QUERIES = (("parse", "lightonocr-3-0.8b", PARSE_SQL),
           ("classify", "lightonocr-3-0.8b", CLASSIFY_SQL),
           ("classify", "pdfium", CLASSIFY_SQL))


def _files(limit):
    """ParseBench's PDF files from the cached snapshot, and their folders."""
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
    return files[:limit] if limit else files, root


def _vllm_texts(root):
    """The stock vLLM run's text per (path, page)."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from quail.backends.quail.executor.model import resolve_model_path
    from quail.specs import MODELS

    spec = MODELS["lightonocr-3-0.8b"]
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_path(spec.hf_name, spec.revision))
    return {(str(root / r["pdf"]), r["page"]):
            tokenizer.decode(r["token_ids"], skip_special_tokens=True)
            for r in pq.read_table(VLLM_RUN, columns=["pdf", "page", "token_ids"])
            .to_pylist()}


@app.function(image=image, gpu="H100!", cpu=32, memory=131072,
              timeout=3 * 3600, volumes=volumes)
def run_query(query: str, parser: str, sql: str, mode: str, run_id: str) -> dict:
    """Run one query twice in one Session and save its rows and reports."""
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail

    files, root = _files(SMOKE_FILES if mode == "smoke" else None)
    session = quail.Session(quail.EngineConfig(
        model=READER, device="h100-sxm", parser=parser))
    started = time.perf_counter()
    session.register("pages", quail.read_pdfs(files))
    register_s = time.perf_counter() - started
    print(session.sql(sql).explain(), flush=True)
    reports = []
    for attempt in range(2):
        started = time.perf_counter()
        result = session.sql(sql).run()
        rows = result.collect().to_pylist()
        wall_s = time.perf_counter() - started
        reports.append(dict(attempt=attempt, wall_s=wall_s, rows=len(rows),
                            report=result.report))
        print(f"run {attempt}: {len(rows)} rows in {wall_s:.1f} s", flush=True)
    session.close()
    summary = dict(run_id=run_id, query=query, parser=parser, sql=sql, mode=mode,
                   reader=READER, files=len(files), register_s=register_s,
                   query_s=reports[1]["wall_s"], rows=len(rows), reports=reports)
    if query == "parse":
        vllm = _vllm_texts(root)
        same = [row["text"] == vllm.get((row["p.path"], row["p.page"]))
                for row in rows]
        summary.update(pages_identical_to_vllm=sum(same),
                       pages_compared=sum((row["p.path"], row["p.page"]) in vllm
                                          for row in rows))
    else:
        folders = [Path(row["p.path"]).parent.name for row in rows]
        agree = [row["kind"] is not None and folder.startswith(row["kind"])
                 for row, folder in zip(rows, folders)]
        summary.update(label_agrees_with_folder=sum(agree),
                       labels={label: sum(row["kind"] == label for row in rows)
                               for label in LABELS},
                       folders={folder: folders.count(folder)
                                for folder in sorted(set(folders))})
    stem = f"{OUT_DIR}/{query}_{parser}_{mode}_{run_id}"
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), f"{stem}.parquet")
    Path(f"{stem}.json").write_text(json.dumps(summary, indent=1, default=str))
    results_vol.commit()
    summary["paths"] = [f"{stem}.parquet", f"{stem}.json"]
    for report in summary["reports"]:
        report["report"] = {key: report["report"].get(key) for key in (
            "wall_s", "boot_s", "planning_s", "fresh_tokens", "usd_per_query")} | {
            "parse": report["report"].get("node_metrics", {}).get("parse:p")}
    return summary


@app.local_entrypoint()
def main(mode: str = "smoke", only: str = "", prediction: str = ""):
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    print(f"prediction: {prediction}", flush=True)
    print(f"run id: {run_id}", flush=True)
    calls = []
    for query, parser, sql in QUERIES:
        if only and only != f"{query}_{parser}":
            continue
        call = run_query.spawn(query, parser, sql, mode, run_id)
        print(f"function call id {query}_{parser}: {call.object_id}", flush=True)
        calls.append(call)
    for call in calls:
        print(json.dumps(call.get(), indent=1, default=str), flush=True)
