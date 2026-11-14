r"""Ask CUAD clause questions about PDF pages, as page images and as text.

Each contract's PDF becomes one row per page. A query asks Qwen3.5 about
each page, read one of two ways:

- image: the page image AI.PDF_PAGES renders, read by the vision encoder;
- text: the page's stored text, AI.PARSE_DOCUMENT split at its page breaks.

    SELECT p.id AS page_id FROM pages p
    WHERE AI.IF(PROMPT('Does this page state which law governs ...? {0}', p.image))

The page labels say which pages hold each CUAD clause (the dataset's
build_cuad_pdf.py on the quail-results volume). Each mode runs in its own
container on the same pages; the result is saved to the quail-results
volume under /pdf_pages/<run id>/.

    uv run modal run --detach -m experiments.cells.pdf_pages \
      --subset small --model qwen3.5-4b-bf16 2>&1 | tee "$run_log"

The volume holds the inputs: /cuad_pdf/<subset>/contracts.parquet and
page_labels.parquet.
"""

from __future__ import annotations

import json
import os
import re
import time

import modal

from quail.backends.quail.executor.images import png_size
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
TEXTS_SQL = "SELECT c.contract_id, AI.PARSE_DOCUMENT(c.pdf) AS text FROM contracts c"
PAGES_SQL = (
    "SELECT c.contract_id || '/' || CAST(u['page'] AS VARCHAR) AS id, "
    "c.contract_id, u['page'] AS page, u['image'] AS image "
    "FROM (SELECT c.contract_id, UNNEST(AI.PDF_PAGES(c.pdf, {edge})) AS u "
    "FROM contracts c) c")


def _prompt(category: str) -> str:
    return PROMPT_TEMPLATE.format(question=QUESTIONS[category]).replace("'", "''")


# the classification's label for each clause, and for a page of none
CLAUSE_LABELS = {
    "Governing Law": "governing law",
    "Termination For Convenience": "termination for convenience",
    "Cap On Liability": "cap on liability",
    "Expiration Date": "expiration date",
}
OTHER = "other"


def queries(column: str = "image") -> dict[str, str]:
    """Each query by name: one filter per clause, all four, and a classification.

    Args:
        column: The pages' document column, image or text.
    """
    def asks(category):
        return f"AI.IF(PROMPT('{_prompt(category)}', p.{column}))"

    found = {f"filter: {category}": (
        f"SELECT p.id AS page_id FROM pages p WHERE {asks(category)}")
        for category in QUESTIONS}
    tags = ", ".join(f"{asks(category)} AS tag_{index}"
                     for index, category in enumerate(QUESTIONS))
    found["tag: every clause"] = f"SELECT p.id AS page_id, {tags} FROM pages p"
    labels = ", ".join(f"'{label}'" for label in [*CLAUSE_LABELS.values(), OTHER])
    found["classify: clause"] = (
        "SELECT p.id AS page_id, AI.CLASSIFY(PROMPT('Which clause does this page "
        f"state? {{0}}', p.{column}), [{labels}]) AS clause FROM pages p")
    return found


# the synthetic subset: pages of drawn text, each a clause or not, which
# checks the image path without PDFs
SYNTHETIC_PAGES = {
    "Governing Law": "This Agreement shall be governed by and construed in "
                     "accordance with the laws of the State of Delaware.",
    "Termination For Convenience": "Either party may terminate this Agreement "
                                   "at any time, for any reason, upon thirty "
                                   "days written notice.",
    "Cap On Liability": "In no event shall either party's aggregate liability "
                        "exceed the fees paid in the twelve months before the "
                        "claim.",
    "Expiration Date": "This Agreement shall remain in effect for a term of "
                       "three years from the Effective Date.",
    None: "The Supplier shall deliver the Products to the Buyer's warehouse "
          "in the quantities set out in each purchase order.",
}


def synthetic_pages(edge: int):
    """Pages of drawn text, and their labels: each clause once per size."""
    import io
    import textwrap

    import pyarrow as pa
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=28)
    rows, labels = [], []
    for size_index, (width, height) in enumerate(
            [(edge * 3 // 4 // 32 * 32, edge), (edge, edge * 3 // 4 // 32 * 32)]):
        for index, (category, text) in enumerate(SYNTHETIC_PAGES.items()):
            page = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(page)
            draw.multiline_text((48, 64), textwrap.fill(text, 40), fill="black",
                                font=font, spacing=12)
            out = io.BytesIO()
            page.save(out, format="PNG")
            contract = f"s{size_index}"
            rows.append({"id": f"{contract}/{index + 1}", "contract_id": contract,
                         "page": index + 1, "image": out.getvalue()})
            for known in QUESTIONS:
                labels.append({"contract_id": contract, "page": index + 1,
                               "category": known, "present": known == category})
    return pa.Table.from_pylist(rows), labels


def _rendered_pages(session, subset: str, edge: int):
    """The subset's pages as PNG images, and the seconds rendering took."""
    import pyarrow.parquet as pq

    import quail

    if subset == "synthetic":
        started = time.perf_counter()
        return synthetic_pages(edge)[0], time.perf_counter() - started

    contracts = pq.read_table(f"{RESULTS}/cuad_pdf/{subset}/contracts.parquet")
    session.register("contracts", quail.DocumentProvider.from_table(
        contracts.select(["contract_id", "pdf"]), id_col="contract_id"))
    started = time.perf_counter()
    pages = session.sql(PAGES_SQL.format(edge=edge)).collect()
    # a column keeps its table's qualifier, such as c.contract_id
    pages = pages.rename_columns([name.split(".")[-1] for name in pages.column_names])
    return pages, time.perf_counter() - started


def _page_texts(session, subset: str):
    """Each page's stored text, by AI.PARSE_DOCUMENT, and the seconds it took."""
    import pyarrow as pa

    started = time.perf_counter()
    texts = session.sql(TEXTS_SQL).collect()
    rows = []
    for contract, text in zip(texts.column(0).to_pylist(),
                              texts.column(1).to_pylist(), strict=True):
        for page, found in enumerate((text or "").split("\f"), start=1):
            rows.append({"id": f"{contract}/{page}", "contract_id": contract,
                         "page": page, "text": found})
    return pa.Table.from_pylist(rows), time.perf_counter() - started


def _register_contracts(session, subset: str) -> None:
    import pyarrow.parquet as pq

    import quail

    contracts = pq.read_table(f"{RESULTS}/cuad_pdf/{subset}/contracts.parquet")
    session.register("contracts", quail.DocumentProvider.from_table(
        contracts.select(["contract_id", "pdf"]), id_col="contract_id"))


def _prompt_tokens(session, sql_prompt: str) -> int:
    """The tokens of a prompt around its document: its turn and template."""
    prefix, suffix = session.model.turn
    template = sql_prompt.replace("''", "'").replace("{0}", "")
    return len(session.tokenizer(prefix + template + suffix))


@app.function(image=image, gpu="H100!", memory=98304, timeout=4 * 3600,
              volumes=volumes)
def run_mode(mode: str, model: str, subset: str, edge: int, run_id: str) -> dict:
    """Run every query over the pages read one way; save rows and measurements.

    Args:
        mode: image or text.
        model: The model spec's name.
        subset: small, medium, or synthetic.
        edge: The page images' longest side, in pixels.
        run_id: The run's folder on the volume.
    """
    import pyarrow.parquet as pq

    import quail
    from bench.quailb import _submission_to_answer_s

    config = quail.EngineConfig(model=model, device="h100-sxm")
    record = {"mode": mode, "model": model, "subset": subset,
              "longest_edge": edge, "queries": {}}
    labels = (synthetic_pages(edge)[1] if subset == "synthetic" else pq.read_table(
        f"{RESULTS}/cuad_pdf/{subset}/page_labels.parquet").to_pylist())
    with quail.Session(config) as session:
        if mode == "image":
            pages, prepare_s = _rendered_pages(session, subset, edge)
            document_tokens = [2 + (width // 32) * (height // 32) for width, height in
                               map(png_size, pages.column("image").to_pylist())]
        else:
            _register_contracts(session, subset)
            pages, prepare_s = _page_texts(session, subset)
            document_tokens = [len(session.tokenizer(text))
                               for text in pages.column("text").to_pylist()]
        record.update(pages=pages.num_rows, prepare_s=round(prepare_s, 2),
                      document_tokens=sum(document_tokens))
        session.register("pages", quail.DocumentProvider.from_table(
            pages.select(["id", "contract_id", "page", mode]), id_col="id"))
        for name, sql in queries(mode).items():
            prompts = re.findall(r"PROMPT\('((?:[^']|'')*)'", sql)
            input_tokens = sum(
                sum(document_tokens) + len(pages) * _prompt_tokens(session, prompt)
                for prompt in prompts)
            submitted = time.perf_counter()
            query = session.sql(sql)
            frontend_s = time.perf_counter() - submitted
            result = query.run()
            query_s = _submission_to_answer_s(
                "quail", result.report, frontend_s=frontend_s, answer_prepare_s=0.0)
            table = result.collect()
            ids = table.column("page_id").to_pylist()
            found = {"query_s": round(query_s, 3), "input_tokens": input_tokens,
                     "input_tokens_per_second": round(input_tokens / query_s),
                     "usd_per_query": round(query_s / 3600 * H100_USD_PER_HOUR, 5)}
            if "clause" in table.column_names:
                clauses = dict(zip(ids, table.column("clause").to_pylist()))
                found["accuracy"] = classify_scores(clauses, labels)
            elif "tag_0" in table.column_names:
                found["accuracy"] = {
                    category: scores(
                        [page for page, tag in zip(ids, table.column(f"tag_{index}")
                                                   .to_pylist()) if tag],
                        labels, [category])
                    for index, category in enumerate(QUESTIONS)}
            else:
                found.update(ids=sorted(ids),
                             accuracy=scores(ids, labels, _categories(name)))
            record["queries"][name] = found
            print(f"{mode} {name}: {query_s:.1f} s, {len(ids)} rows", flush=True)
    directory = f"{RESULTS}/pdf_pages/{run_id}"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{mode}.json", "w") as out:
        json.dump(record, out)
    volumes[RESULTS].commit()
    return record


def _clauses(labels) -> dict[str, set[str]]:
    """Each labeled page's clause categories, by page id."""
    found: dict[str, set[str]] = {}
    for row in labels:
        if row["category"] not in QUESTIONS:
            continue
        clauses = found.setdefault(f"{row['contract_id']}/{row['page']}", set())
        if row["present"]:
            clauses.add(row["category"])
    return found


def classify_scores(answers: dict, labels) -> dict:
    """The share of pages whose label names one of their clauses, or other."""
    truth = _clauses(labels)
    right = sum(
        (answer == OTHER and not truth[page]) or any(
            CLAUSE_LABELS[category] == answer for category in truth[page])
        for page, answer in answers.items())
    return {"accuracy": round(right / max(1, len(answers)), 3),
            "pages": len(answers)}


def scores(ids, labels, categories) -> dict:
    """Precision, recall, and F1 of returned page ids against labels.

    A page is labeled when it holds every one of the categories.
    """
    truth = {page for page, clauses in _clauses(labels).items()
             if set(categories) <= clauses}
    found = set(ids)
    hit = len(found & truth)
    precision = hit / len(found) if found else 0.0
    recall = hit / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hit else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "returned": len(found), "labeled": len(truth)}


@app.local_entrypoint()
def main(subset: str = "small", model: str = "qwen3.5-4b-bf16",
         modes: str = "image,text", edge: int = 1024):
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{subset}-{model}-{edge}"
    calls = {mode: run_mode.spawn(mode, model, subset, edge, run_id)
             for mode in modes.split(",")}
    for mode, call in calls.items():
        print(f"{mode} function call id: {call.object_id}", flush=True)
    records = {mode: call.get() for mode, call in calls.items()}
    print(f"saved to quail-results:/pdf_pages/{run_id}/", flush=True)
    summarize(records)


def summarize(records: dict) -> None:
    """Print each query's time, throughput, cost, and accuracy per mode."""
    for mode, record in records.items():
        print(f"\n{mode}: {record['pages']} pages, {record['document_tokens']:,} "
              f"document tokens, prepared in {record['prepare_s']} s")
        for name, found in record["queries"].items():
            print(f"  {name:36} {found['query_s']:7.1f} s "
                  f"{found['input_tokens_per_second']:>9,} input tok/s "
                  f"${found['usd_per_query']:.4f}/query  {found['accuracy']}")


def _categories(name: str) -> list[str]:
    """The clause categories a query asks about."""
    if not name.startswith("filter: "):
        return list(QUESTIONS)
    return [name.removeprefix("filter: ")]
