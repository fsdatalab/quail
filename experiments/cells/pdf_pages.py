r"""Ask CUAD clause questions about PDF page images: Quail and stock vLLM.

Each contract's PDF becomes one row per page with AI.PDF_PAGES, and a
query asks a vision model about each page image:

    SELECT p.id FROM pages p
    WHERE AI.IF(PROMPT('Does this page state which law governs ...? {0}', p.image))

The page labels say which pages hold each CUAD clause (the dataset's
build_cuad_pdf.py on the quail-results volume). Each backend runs in its
own container on the same rendered pages; the result is saved to the
quail-results volume under /pdf_pages/<run id>/.

    uv run modal run --detach -m experiments.cells.pdf_pages \
      --subset small --model qwen3.5-4b-bf16 2>&1 | tee "$run_log"

The volume holds the inputs: /cuad_pdf/<subset>/contracts.parquet and
page_labels.parquet.
"""

from __future__ import annotations

import json
import os
import time

import modal

from quail.deploy.images import gpu_image

app = modal.App("quail-milestone1")
image = gpu_image()
RESULTS = "/results"
volumes = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
    RESULTS: modal.Volume.from_name("quail-results"),
}
H100_USD_PER_HOUR = 3.9492
# one question per clause category, each a page-level yes or no
QUESTIONS = {
    "Governing Law": "Does this page state which jurisdiction's law governs "
                     "the agreement?",
    "Termination For Convenience": "Does this page let a party terminate the "
                                   "agreement without cause or for convenience?",
    "Cap On Liability": "Does this page cap or limit a party's liability or "
                        "damages?",
    "Expiration Date": "Does this page state when the agreement's term expires "
                       "or how long it lasts?",
}
PROMPT_TEMPLATE = "{question} Answer yes or no.\n\n{{0}}"
PAGES_SQL = (
    "SELECT c.contract_id || '/' || CAST(u['page'] AS VARCHAR) AS id, "
    "c.contract_id, u['page'] AS page, u['image'] AS image "
    "FROM (SELECT c.contract_id, UNNEST(AI.PDF_PAGES(c.pdf, {edge})) AS u "
    "FROM contracts c) c")


def _prompt(category: str) -> str:
    return PROMPT_TEMPLATE.format(question=QUESTIONS[category]).replace("'", "''")


def queries() -> dict[str, str]:
    """Each query by name: one filter per clause, and their OR."""
    found = {f"filter: {category}": (
        f"SELECT p.id FROM pages p WHERE AI.IF(PROMPT('{_prompt(category)}', "
        "p.image))") for category in QUESTIONS}
    terms = " OR ".join(f"AI.IF(PROMPT('{_prompt(category)}', p.image))"
                        for category in QUESTIONS)
    found["filter: any clause"] = f"SELECT p.id FROM pages p WHERE {terms}"
    return found


def _rendered_pages(session, subset: str, edge: int):
    """The subset's pages as PNG images, and the seconds rendering took."""
    import pyarrow.parquet as pq

    import quail

    contracts = pq.read_table(f"{RESULTS}/cuad_pdf/{subset}/contracts.parquet")
    session.register("contracts", quail.DocumentProvider.from_table(
        contracts.select(["contract_id", "pdf"]), id_col="contract_id"))
    started = time.perf_counter()
    pages = session.sql(PAGES_SQL.format(edge=edge)).collect()
    return pages, time.perf_counter() - started


@app.function(image=image, gpu="H100!", memory=98304, timeout=4 * 3600,
              volumes=volumes)
def run_backend(backend: str, model: str, subset: str, edge: int,
                run_id: str) -> dict:
    """Run every query on one backend and save its rows and measurements."""
    import pyarrow.parquet as pq

    import quail
    from bench.quailb import _submission_to_answer_s

    if backend != "quail":
        os.environ["QUAIL_VLLM_IMAGES"] = "1"
    config = quail.EngineConfig(model=model, device="h100-sxm", backend=backend)
    record = {"backend": backend, "model": model, "subset": subset,
              "longest_edge": edge, "queries": {}}
    labels = pq.read_table(
        f"{RESULTS}/cuad_pdf/{subset}/page_labels.parquet").to_pylist()
    with quail.Session(config) as session:
        pages, render_s = _rendered_pages(session, subset, edge)
        record.update(pages=pages.num_rows, render_s=round(render_s, 2))
        session.register("pages", quail.DocumentProvider.from_table(
            pages.select(["id", "contract_id", "page", "image"]), id_col="id"))
        for name, sql in queries().items():
            submitted = time.perf_counter()
            query = session.sql(sql)
            frontend_s = time.perf_counter() - submitted
            result = query.run()
            query_s = _submission_to_answer_s(
                backend, result.report, frontend_s=frontend_s,
                answer_prepare_s=0.0)
            ids = sorted(result.collect().column("id").to_pylist())
            metrics = result.report.get("backend_metrics", {})
            record["queries"][name] = {
                "query_s": round(query_s, 3), "ids": ids,
                "prompt_tokens": metrics.get("prompt_tokens"),
                "boot_s": result.report.get("boot_s"),
                "accuracy": scores(ids, labels, _categories(name)),
            }
            print(f"{backend} {name}: {query_s:.1f} s, {len(ids)} rows",
                  flush=True)
    directory = f"{RESULTS}/pdf_pages/{run_id}"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{backend}.json", "w") as out:
        json.dump(record, out)
    volumes[RESULTS].commit()
    return record


def scores(ids, labels, categories) -> dict:
    """Precision, recall, and F1 of returned page ids against labels."""
    truth = {row["contract_id"] + "/" + str(row["page"])
             for row in labels if row["present"] and row["category"] in categories}
    found = set(ids)
    hit = len(found & truth)
    precision = hit / len(found) if found else 0.0
    recall = hit / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hit else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "returned": len(found), "labeled": len(truth)}


@app.local_entrypoint()
def main(subset: str = "small", model: str = "qwen3.5-4b-bf16",
         backends: str = "quail,stock_vllm", edge: int = 1024):
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{subset}-{model}-{edge}"
    calls = {backend: run_backend.spawn(backend, model, subset, edge, run_id)
             for backend in backends.split(",")}
    for backend, call in calls.items():
        print(f"{backend} function call id: {call.object_id}", flush=True)
    records = {backend: call.get() for backend, call in calls.items()}
    print(f"saved to quail-results:/pdf_pages/{run_id}/", flush=True)
    summarize(records)


def _categories(name: str) -> list[str]:
    """The clause categories a query asks about."""
    if name.endswith("any clause"):
        return list(QUESTIONS)
    return [name.removeprefix("filter: ")]


def summarize(records: dict) -> None:
    """Print each query's time, throughput, cost, and accuracy per backend."""
    vllm = records.get("stock_vllm")
    for name in queries():
        questions = len(_categories(name))
        print(f"\n{name}")
        for backend, record in records.items():
            found = record["queries"][name]
            tokens = None
            if vllm is not None:
                single = [vllm["queries"][f"filter: {c}"]["prompt_tokens"]
                          for c in QUESTIONS]
                per_page = sum(single) / len(single) / vllm["pages"]
                tokens = per_page * record["pages"] * questions
            seconds = found["query_s"]
            usd = seconds / 3600 * H100_USD_PER_HOUR
            rate = f"{tokens / seconds:,.0f}" if tokens else "?"
            print(f"  {backend:12} {seconds:8.1f} s  {rate:>10} input tok/s  "
                  f"${usd:.4f}/query  {found['accuracy']}")
