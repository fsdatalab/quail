r"""Answer OfficeQA Pro questions from Treasury Bulletin pages in Quail.

Each question is one row whose document is the paper's base prompt, the
pages the question cites, and the question, read greedily by Qwen3.8 27B
with thinking off:

    SELECT q.question_id,
           AI.MAP(q.parts, 'Give your final answer in the FINAL_ANSWER tags.', 8192)
    FROM questions q

The pages are read three ways: page images (AI.PDF_PAGES at 2,048
pixels), the text the PDF stores (AI.PARSE_DOCUMENT), and the LightOnOCR
markdown the issue #148 experiment saved. reward.py scores each at 0%
tolerance (score.py). Inputs and results are on the quail-results volume:
/officeqa/ (pages.parquet, questions.parquet, reward.py, score.py,
reader_prompt.json) and /officeqa/runs/<run id>/parsers.json.

    uv run modal run --detach -m experiments.cells.officeqa 2>&1 | tee "$run_log"
"""

from __future__ import annotations

import json
import os
import sys
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
LONGEST_EDGE = 2048
PAGES_SQL = ("SELECT p.page_id, array_element(AI.PDF_PAGES(p.pdf, {edge}), 1)['image'] "
             "AS image FROM officeqa_pages p")
# AI.MAP writes its instruction after the document; this one repeats the
# base prompt's own format rule
INSTRUCTION = "Give your final answer in the FINAL_ANSWER tags."

def question_parts(questions, pages: dict, prompt: dict) -> list[list[bytes]]:
    """Each question's document: the base prompt, its pages, and the question.

    Args:
        questions: The questions table.
        pages: Each page's part by page id: its PNG, or its text as UTF-8.
        prompt: The reader prompt's text.
    """
    def page(found: bytes) -> bytes:
        return found if found.startswith(b"\x89PNG") else found + b"\n\n"

    return [[(prompt["system"] + "\n\n").encode(),
             *(page(pages[found]) for found in page_ids),
             prompt["user_after_page_images"].format(question=question).encode()]
            for question, page_ids in zip(questions.column("question").to_pylist(),
                                          questions.column("page_ids").to_pylist(),
                                          strict=True)]


def _prompt_tokens(session, parts: list[bytes]) -> int:
    """A prompt's full length: its turn, its text, and its image tokens."""
    prefix, suffix = session.model.turn
    found = len(session.tokenizer(prefix + suffix))
    for part in parts:
        if part.startswith(b"\x89PNG"):
            width, height = png_size(part)
            found += 2 + (width // 32) * (height // 32)
        else:
            found += len(session.tokenizer(part.decode()))
    return found


MODEL = "qwen3.8-27b-fp8"
ANSWER_TOKENS = 8_192
# parsed text the issue #148 experiment saved, by page id
SAVED_PARSES = {"lightonocr": "officeqa_nemotron/parsed_lightonocr.json"}
TEXTS_SQL = "SELECT p.page_id, AI.PARSE_DOCUMENT(p.pdf) AS text FROM officeqa_pages p"


@app.function(image=image, gpu="H100!", memory=98304, timeout=8 * 3600,
              volumes=volumes)
def run_parsers(run_id: str) -> dict:
    """Answer every question from each way of reading its pages; score each."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail
    from bench.quailb import _submission_to_answer_s

    root = f"{RESULTS}/officeqa"
    sys.path.insert(0, root)
    from score import accuracy

    pages = pq.read_table(f"{root}/pages.parquet", columns=["page_id", "pdf"])
    questions = pq.read_table(f"{root}/questions.parquet")
    truth = dict(zip(questions.column("question_id").to_pylist(),
                     questions.column("answer").to_pylist(), strict=True))
    with open(f"{root}/reader_prompt.json") as found:
        prompt = json.load(found)
    sql = (f"SELECT q.question_id, AI.MAP(q.parts, '{INSTRUCTION}', "
           f"{ANSWER_TOKENS}) AS response FROM questions q")
    record = {"model": MODEL, "answer_tokens": ANSWER_TOKENS,
              "longest_edge": LONGEST_EDGE, "inputs": {}}
    config = quail.EngineConfig(model=MODEL, device="h100-sxm")
    with quail.Session(config) as session:
        session.register("officeqa_pages", quail.DocumentProvider.from_table(
            pages, id_col="page_id"))
        started = time.perf_counter()
        rendered = session.sql(PAGES_SQL.format(edge=LONGEST_EDGE)).collect()
        render_s = time.perf_counter() - started
        started = time.perf_counter()
        stored = session.sql(TEXTS_SQL).collect()
        parse_s = time.perf_counter() - started
        readings = {
            "page images": (dict(zip(rendered.column(0).to_pylist(),
                                     rendered.column(1).to_pylist(), strict=True)),
                            render_s),
            "pdf text (AI.PARSE_DOCUMENT)": (
                {page: (text or "").encode() for page, text in zip(
                    stored.column(0).to_pylist(), stored.column(1).to_pylist(),
                    strict=True)}, parse_s)}
        for name, path in SAVED_PARSES.items():
            with open(f"{RESULTS}/{path}") as found:
                parsed = json.load(found)
            readings[name] = ({page: value["markdown"].encode()
                               for page, value in parsed.items()}, None)
        for index, (name, (by_page, prepare_s)) in enumerate(readings.items()):
            table = pa.table({
                "question_id": questions.column("question_id"),
                "parts": pa.array(question_parts(questions, by_page, prompt),
                                  pa.list_(pa.binary()))})
            alias = f"questions_{index}"
            session.register(alias, quail.DocumentProvider.from_table(
                table, id_col="question_id"))
            submitted = time.perf_counter()
            query = session.sql(sql.replace("FROM questions q", f"FROM {alias} q"))
            frontend_s = time.perf_counter() - submitted
            result = query.run()
            query_s = _submission_to_answer_s(
                "quail", result.report, frontend_s=frontend_s, answer_prepare_s=0.0)
            answers = result.collect()
            responses = dict(zip(answers.column(0).to_pylist(),
                                 answers.column(1).to_pylist(), strict=True))
            mean, _ = accuracy(truth, responses)
            input_tokens = sum(_prompt_tokens(session, parts)
                               for parts in table.column("parts").to_pylist())
            record["inputs"][name] = {
                "accuracy": round(100 * mean, 1), "query_s": round(query_s, 1),
                "prepare_s": None if prepare_s is None else round(prepare_s, 1),
                "input_tokens": input_tokens,
                "input_tokens_per_second": round(input_tokens / query_s),
                "output_tokens": sum(len(session.tokenizer(text or ""))
                                     for text in responses.values()),
                "usd_per_query": round(query_s / 3600 * H100_USD_PER_HOUR, 4),
                "responses": responses}
            print(f"{name}: {100 * mean:.1f}% in {query_s:.0f} s, "
                  f"{input_tokens:,} input tokens", flush=True)
    directory = f"{root}/runs/{run_id}"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/parsers.json", "w") as out:
        json.dump(record, out)
    volumes[RESULTS].commit()
    return {name: {key: value for key, value in found.items() if key != "responses"}
            for name, found in record["inputs"].items()}


@app.local_entrypoint()
def main():
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{MODEL}-parsers"
    call = run_parsers.spawn(run_id)
    print(f"function call id: {call.object_id}", flush=True)
    found = call.get()
    print(f"saved to quail-results:/officeqa/runs/{run_id}/", flush=True)
    for name, measured in found.items():
        print(f"  {name:30} {json.dumps(measured)}", flush=True)
