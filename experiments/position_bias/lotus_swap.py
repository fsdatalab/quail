"""Swap test with the LOTUS pairwise prompt over every pair of SALES-5 rows.

SALES-5 keeps the 225 sales calls in the Negotiation stage, so there
are 225 * 224 / 2 = 25,200 unordered pairs. Each pair is asked in both
orders with the LOTUS pairwise instruction, and the answer is read
from the first token: AI.CLASSIFY over "Document 1" and "Document 2"
with probabilities, so P(Document 1) is the probability that the
first record wins. Jev answers the same text as a choice question.

    uv run modal run --detach -m experiments.position_bias.lotus_swap \
        2>&1 | tee results/position_bias/lotus_swap.log

    read -s TYPESAFE_API_KEY
    export TYPESAFE_API_KEY
    uv run python -m experiments.position_bias.lotus_swap --jev

Records: /results/position_bias/lotus_swap/{model}.json on the
quail-results volume.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import modal

try:
    from quail.deploy.images import gpu_image
    image = gpu_image().add_local_python_source("experiments")
except ImportError:    # a container without the local quail package
    image = None

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)
results = modal.Volume.from_name("quail-results", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache,
           "/root/.cache/kernels": kernel_cache,
           "/results": results}

MODELS = ("decision-2.0-kai-0.6b-bf16", "qwen3-0.6b-bf16", "d1-2b-bf16",
          "lfm2-2.6b-bf16", "qwen3-4b-fp8", "qwen3.8-27b-fp8",
          "qwen3.6-35b-a3b-fp8")
JEV_MODEL = "jev-1.13.0"
RECORD_DIR = "/results/position_bias/lotus_swap"
LABELS = ("Document 1", "Document 2")
# rows per AI.CLASSIFY query; bounds the host memory of one batch
BATCH_ROWS = 4096

LOTUS_INSTRUCTION_TEXT = (
    "Your job is to select and return the most relevant document to the "
    "user's question.\nCarefully read the user's question and the two "
    "documents provided below.\nEnd your response with the label of the "
    "document such as \"Document NUMBER\".\nNUMBER must be either 1 or 2, "
    "depending on which document is most relevant.\nYou must pick a number "
    "and cannot say things like \"None\" or \"Neither\".")
QUESTION_TEXT = (
    "Which sales call most clearly shows that the customer commits to "
    "buying, such as accepting the terms, agreeing to sign, or asking for "
    "the contract or paperwork? Interest, questions, or agreeing to "
    "another meeting do not count.")
PAIR_TEMPLATE = (LOTUS_INSTRUCTION_TEXT + "\n\nQuestion: " + QUESTION_TEXT
                 + "\n\n{0}")


def pair_document(first: str, second: str) -> str:
    """The two calls as LOTUS lays them out: Document 1, then Document 2."""
    return f"Document 1:\n{first}\n\nDocument 2:\n{second}"


def load_negotiation_calls():
    """The SALES-5 rows: (ids, id -> transcript, id -> commits label)."""
    import pyarrow.compute as pc

    from experiments.position_bias.tasks import TASKS, load_task

    data = load_task(TASKS["sales"])
    from quail_b.data import load_table

    calls = load_table("sales_calls", scale_factor=0.1)
    kept = calls.filter(pc.equal(calls["deal_stage"], "Negotiation"))
    ids = [str(value) for value in kept["id"].to_pylist()]
    return ids, {key: data.texts[key] for key in ids}, \
        {key: data.labels[key] for key in ids}


def run_swap(ids, texts, labels, choose) -> dict:
    """Ask every unordered pair in both orders and summarize the answers.

    Args:
        ids: Record ids; every pair (a, b) with a before b is asked.
        texts: Id -> text.
        labels: Id -> whether the record is relevant.
        choose: Takes documents and returns P(Document 1) for each.
    """
    from experiments.position_bias.experiment import swap_summary

    pairs = list(itertools.combinations(ids, 2))
    documents = ([pair_document(texts[a], texts[b]) for a, b in pairs]
                 + [pair_document(texts[b], texts[a]) for a, b in pairs])
    start = time.perf_counter()
    probabilities = choose(documents)
    seconds = time.perf_counter() - start
    forward, backward = probabilities[:len(pairs)], probabilities[len(pairs):]
    rows = [{"a": a, "b": b, "label_a": labels[a], "label_b": labels[b],
             "p_forward": p, "p_backward": q}
            for (a, b), p, q in zip(pairs, forward, backward)]
    return {"num_records": len(ids), "num_relevant": sum(labels.values()),
            "num_pairs": len(pairs), "seconds": seconds,
            "template": PAIR_TEMPLATE, "rows": rows,
            "summary": swap_summary(rows)}


class QuailChooser:
    """P(Document 1) from AI.CLASSIFY's probabilities in one Quail session."""

    def __init__(self, session):
        self.session = session
        self.num_batches = 0

    def __call__(self, documents) -> list[float]:
        """Return P(Document 1) for each document."""
        import pyarrow as pa

        import quail

        quoted = PAIR_TEMPLATE.replace("'", "''")
        options = ", ".join(f"'{label}'" for label in LABELS)
        probabilities = []
        for start in range(0, len(documents), BATCH_ROWS):
            chunk = list(documents[start:start + BATCH_ROWS])
            name = f"pairs_{self.num_batches}"
            self.num_batches += 1
            table = pa.table({"id": pa.array(range(len(chunk)), pa.int64()),
                              "document": pa.array(chunk, pa.string())})
            self.session.register(
                name, quail.DocumentProvider.from_table(table, id_col="id"))
            result = self.session.sql(
                f"SELECT b.id, AI.CLASSIFY(PROMPT('{quoted}', b.document), "
                f"ARRAY[{options}], {{'probabilities': true}}) AS winner "
                f"FROM {name} b").collect()
            maps = result["winner_probabilities"].to_pylist()
            by_id = {index: dict(entry) for index, entry
                     in zip(result.column(0).to_pylist(), maps)}
            probabilities.extend(by_id[index][LABELS[0]]
                                 for index in range(len(chunk)))
        return probabilities


@app.function(image=image, gpu="H100!", memory=98304, timeout=12 * 3600,
              volumes=VOLUMES)
def swap_model(model: str, num_records: int = 0) -> dict:
    """Run the LOTUS swap test for one model and save it to the volume.

    Args:
        model: The Quail model.
        num_records: Use only the first this many calls; 0 uses all
            225 and saves the record.
    """
    from contextlib import ExitStack

    import quail

    ids, texts, labels = load_negotiation_calls()
    if num_records:
        relevant = [key for key in ids if labels[key]]
        ids = list(dict.fromkeys(ids[:num_records] + relevant[:2]))
    with ExitStack() as stack:
        session = stack.enter_context(quail.Session(
            quail.EngineConfig(model=model, device="h100-sxm")))
        record = run_swap(ids, texts, labels, QuailChooser(session))
    record["model"] = model
    summary = record["summary"]
    print(f"[{model}] same slot: tied {summary['tied']['same_slot_rate']:.4f}, "
          f"mixed {summary['mixed']['same_slot_rate']:.4f}, "
          f"bias {summary['all']['bias']:+.3f}", flush=True)
    if not num_records:
        path = Path(RECORD_DIR) / f"{model}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))
        results.commit()
    return summary


@app.local_entrypoint()
def main(models: str = ",".join(MODELS), num_records: int = 0):
    """Start one call per model and print each function call id."""
    calls = [(model, swap_model.spawn(model, num_records))
             for model in models.split(",")]
    for model, call in calls:
        print(f"{model}: {call.object_id}", flush=True)
    for model, call in calls:
        try:
            summary = call.get()
        except Exception as error:     # report every model, then go on
            print(f"{model}: failed: {error!r}", flush=True)
            continue
        print(f"{model}: tied {summary['tied']['same_slot_rate']:.4f} "
              f"mixed {summary['mixed']['same_slot_rate']:.4f}", flush=True)


def jev_main():
    """Run the swap test against Jev and upload the record to the volume."""
    import asyncio
    import os

    import httpx

    api_key = os.environ["TYPESAFE_API_KEY"]
    question = {"type": "choice",
                "instructions": LOTUS_INSTRUCTION_TEXT + "\n\nQuestion: "
                + QUESTION_TEXT,
                "criteria": {"1": LABELS[0], "2": LABELS[1]}}

    async def ask(client, semaphore, document):
        payload = {"model": JEV_MODEL, "state": document,
                   "questions": {"winner": question}}
        delay = 0.25
        async with semaphore:
            for _ in range(12):
                try:
                    response = await client.post(
                        "https://api.typesafe.ai/v1/systemone",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json=payload)
                except httpx.HTTPError:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 8)
                    continue
                if response.status_code == 429 or response.status_code >= 500:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 8)
                    continue
                response.raise_for_status()
                answer = response.json()["answers"]["winner"]
                return float(answer["probabilities"]["1"])
        raise RuntimeError("Jev request retries exhausted")

    async def choose_all(documents):
        semaphore = asyncio.Semaphore(64)
        async with httpx.AsyncClient(
                timeout=120, limits=httpx.Limits(max_connections=64)) as client:
            return await asyncio.gather(*(ask(client, semaphore, document)
                                          for document in documents))

    ids, texts, labels = load_negotiation_calls()
    record = run_swap(ids, texts, labels,
                      lambda documents: asyncio.run(choose_all(documents)))
    record["model"] = JEV_MODEL
    local = Path("results/position_bias/lotus_swap") / f"{JEV_MODEL}.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(record))
    with results.batch_upload(force=True) as upload:
        upload.put_file(str(local), f"/position_bias/lotus_swap/{JEV_MODEL}.json")
    print(json.dumps(record["summary"], indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jev", action="store_true",
                        help="run against the Jev API instead of Modal")
    if parser.parse_args().jev:
        jev_main()
