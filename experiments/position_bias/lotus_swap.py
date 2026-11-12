"""Swap test with the LOTUS pairwise prompt, asking each pair in both orders.

Sales asks every pair of the 225 SALES-5 calls (the Negotiation
stage), 225 * 224 / 2 = 25,200 unordered pairs. Support and Legal ask
5,000 pairs sampled from the task's records. Each pair is asked in
both orders with the LOTUS pairwise instruction, and the answer is
read from the first token: AI.CLASSIFY over "Document 1" and
"Document 2" with probabilities, so P(Document 1) is the probability
that the first record wins. Jev answers the same text as a choice
question.

    uv run modal run --detach -m experiments.position_bias.lotus_swap \
        2>&1 | tee results/position_bias/lotus_swap.log
    uv run modal run --detach -m experiments.position_bias.lotus_swap \
        --tasks support,legal 2>&1 | tee results/position_bias/lotus_swap_2.log

    read -s TYPESAFE_API_KEY
    export TYPESAFE_API_KEY
    uv run python -m experiments.position_bias.lotus_swap --jev --tasks sales

Records: /results/position_bias/lotus_swap/{model}.json for Sales and
/results/position_bias/lotus_swap/{task}/{model}.json for the others,
on the quail-results volume.
"""

import argparse
import itertools
import json
import random
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
QUESTION_TEXTS = {
    "sales": (
        "Which sales call most clearly shows that the customer commits to "
        "buying, such as accepting the terms, agreeing to sign, or asking "
        "for the contract or paperwork? Interest, questions, or agreeing to "
        "another meeting do not count."),
    "support": (
        "Which customer message most clearly expresses frustration or "
        "dissatisfaction with the agent, such as repeating a request, "
        "objecting to a refusal, or complaining about the service?"),
    "legal": (
        "Which excerpt most clearly argues that the cited case's reasoning "
        "does not apply here?"),
    "legal_passages": (
        "Which legal passage most strongly supports a court's consideration "
        "of an argument raised late or not preserved?"),
    "sales_outcome": (
        "In which sales call is the customer more likely to sign a contract "
        "and buy, judging by their commitment, urgency, and remaining "
        "objections?"),
}
TASK_NAMES = ("sales", "support", "legal", "legal_passages")
NUM_SAMPLED_PAIRS = 5000


# LOTUS publishes no pointwise prompt; this is its pairwise instruction
# reworded for one document, answered TRUE or FALSE
LOTUS_POINTWISE_INSTRUCTION_TEXT = (
    "Your job is to decide whether the document answers the user's "
    "question.\nCarefully read the user's question and the document "
    "provided below.\nAnswer TRUE if the document answers the question, "
    "FALSE otherwise.")
POINTWISE_QUESTION_TEXTS = {
    "sales": (
        "Does the customer in this sales call commit to buying, such as "
        "accepting the terms, agreeing to sign, or asking for the contract "
        "or paperwork? Interest, questions, or agreeing to another meeting "
        "do not count."),
    "support": (
        "Does this customer message express frustration or dissatisfaction "
        "with the agent, such as repeating a request, objecting to a "
        "refusal, or complaining about the service?"),
    "legal": (
        "Does this excerpt argue that the cited case's reasoning does not "
        "apply here?"),
    "sales_outcome": (
        "Is the customer in this sales call likely to sign a contract and "
        "buy, judging by their commitment, urgency, and remaining "
        "objections?"),
}


def pointwise_template(task: str) -> str:
    """The AI.SCORE prompt for one record, in the style of the LOTUS prompt."""
    return (LOTUS_POINTWISE_INSTRUCTION_TEXT + "\n\nQuestion: "
            + POINTWISE_QUESTION_TEXTS[task] + "\n\nDocument:\n{0}")


def pair_template(task: str) -> str:
    """The AI.CLASSIFY prompt for one task: instruction, question, documents."""
    return (LOTUS_INSTRUCTION_TEXT + "\n\nQuestion: " + QUESTION_TEXTS[task]
            + "\n\n{0}")


def record_path(task: str, model: str) -> str:
    """The record's path on the volume, relative to its root."""
    if task == "sales":
        return f"position_bias/lotus_swap/{model}.json"
    return f"position_bias/lotus_swap/{task}/{model}.json"


def pair_document(first: str, second: str) -> str:
    """The two calls as LOTUS lays them out: Document 1, then Document 2."""
    return f"Document 1:\n{first}\n\nDocument 2:\n{second}"


def load_pairs(task: str):
    """One task's unordered pairs, texts, and labels.

    Returns:
        (pairs, id -> text, id -> label). Sales gives every pair of the
        SALES-5 calls; the other tasks give NUM_SAMPLED_PAIRS distinct
        pairs drawn uniformly with seed 0.
    """
    import pyarrow.compute as pc

    from experiments.position_bias.tasks import TASKS, load_task
    from quail_b.data import load_table

    if task == "legal_passages":
        # no QUAIL-B labels for this question; every record is unlabeled
        passages = load_table("citation_passages", scale_factor=0.1)
        ids = [str(value) for value in passages["id"].to_pylist()]
        texts = dict(zip(ids, passages["passage_text"].to_pylist()))
        return sample_pairs(ids), texts, dict.fromkeys(ids, False)
    data = load_task(TASKS[task])
    if task == "sales":
        calls = load_table("sales_calls", scale_factor=0.1)
        kept = calls.filter(pc.equal(calls["deal_stage"], "Negotiation"))
        ids = [str(value) for value in kept["id"].to_pylist()]
        pairs = list(itertools.combinations(ids, 2))
    else:
        ids = data.ids
        pairs = sample_pairs(ids)
    return pairs, {key: data.texts[key] for key in ids}, \
        {key: data.labels[key] for key in ids}


def sample_pairs(ids) -> list[tuple[str, str]]:
    """NUM_SAMPLED_PAIRS distinct unordered pairs, drawn uniformly with seed 0."""
    rng = random.Random(0)
    chosen = set()
    while len(chosen) < NUM_SAMPLED_PAIRS:
        first, second = rng.sample(list(ids), 2)
        chosen.add(tuple(sorted((first, second))))
    return sorted(chosen)


def run_swap(pairs, texts, labels, choose, template) -> dict:
    """Ask each unordered pair in both orders and summarize the answers.

    Args:
        pairs: Unordered (a, b) pairs; (a, b) is asked, then (b, a).
        texts: Id -> text.
        labels: Id -> whether the record is relevant.
        choose: Takes documents and returns P(Document 1) for each.
        template: The prompt, recorded with the answers.
    """
    from experiments.position_bias.experiment import swap_summary

    documents = ([pair_document(texts[a], texts[b]) for a, b in pairs]
                 + [pair_document(texts[b], texts[a]) for a, b in pairs])
    start = time.perf_counter()
    probabilities = choose(documents)
    seconds = time.perf_counter() - start
    forward, backward = probabilities[:len(pairs)], probabilities[len(pairs):]
    rows = [{"a": a, "b": b, "label_a": labels[a], "label_b": labels[b],
             "p_forward": p, "p_backward": q}
            for (a, b), p, q in zip(pairs, forward, backward)]
    return {"num_records": len(texts), "num_relevant": sum(labels.values()),
            "num_pairs": len(pairs), "seconds": seconds,
            "template": template, "rows": rows,
            "summary": swap_summary(rows)}


class QuailChooser:
    """P(Document 1) from AI.CLASSIFY's probabilities in one Quail session."""

    def __init__(self, session, template: str):
        self.session = session
        self.template = template
        self.num_batches = 0

    def __call__(self, documents) -> list[float]:
        """Return P(Document 1) for each document."""
        import pyarrow as pa

        import quail

        quoted = self.template.replace("'", "''")
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
def swap_model(model: str, task: str = "sales", num_pairs: int = 0) -> dict:
    """Run the LOTUS swap test for one model and task in its own container."""
    return swap_in_process(model, task, num_pairs)


@app.function(image=image, gpu="H100!", memory=98304, timeout=12 * 3600,
              volumes=VOLUMES)
def swap_models(models: list[str], task: str) -> dict:
    """Run the LOTUS swap test for several models, one after another.

    Quail releases one model's engine before it loads the next, so the
    models share this container's GPU in turn.

    Returns:
        Model -> its summary, or the error text when it failed.
    """
    summaries = {}
    for model in models:
        try:
            summaries[model] = swap_in_process(model, task, 0)
        except Exception as error:     # run the remaining models
            summaries[model] = repr(error)
            print(f"[{model} {task}] failed: {error!r}", flush=True)
    return summaries


def swap_in_process(model: str, task: str = "sales", num_pairs: int = 0) -> dict:
    """Run the LOTUS swap test for one model and task, and save it.

    Args:
        model: The Quail model.
        task: "sales", "support", or "legal".
        num_pairs: Ask only the first this many pairs and save nothing;
            0 asks them all and saves the record to the volume.
    """
    from contextlib import ExitStack

    import quail

    pairs, texts, labels = load_pairs(task)
    if num_pairs:
        pairs = pairs[:num_pairs]
    template = pair_template(task)
    with ExitStack() as stack:
        session = stack.enter_context(quail.Session(
            quail.EngineConfig(model=model, device="h100-sxm")))
        record = run_swap(pairs, texts, labels,
                          QuailChooser(session, template), template)
    record.update(model=model, task=task)
    if not num_pairs:
        path = Path("/results") / record_path(task, model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))
        results.commit()
    summary = record["summary"]
    print(f"[{model} {task}] same slot: all "
          f"{summary['all']['same_slot_rate']:.4f}, "
          f"bias {summary['all']['bias']:+.3f}", flush=True)
    return summary


@app.local_entrypoint()
def main(models: str = ",".join(MODELS), tasks: str = "sales",
         num_pairs: int = 0, one_container: bool = False):
    """Start one call per model and task, and print each function call id.

    With `one_container`, each task runs every model in one call.
    """
    if one_container:
        calls = [(task, swap_models.spawn(models.split(","), task))
                 for task in tasks.split(",")]
        for task, call in calls:
            print(f"{task}: {call.object_id}", flush=True)
        for task, call in calls:
            for model, summary in call.get().items():
                rate = (summary if isinstance(summary, str)
                        else f"same slot {summary['all']['same_slot_rate']:.4f}")
                print(f"{model} {task}: {rate}", flush=True)
        return
    calls = [(f"{model} {task}", swap_model.spawn(model, task, num_pairs))
             for task in tasks.split(",") for model in models.split(",")]
    for model, call in calls:
        print(f"{model}: {call.object_id}", flush=True)
    for model, call in calls:
        try:
            summary = call.get()
        except Exception as error:     # report every run, then go on
            print(f"{model}: failed: {error!r}", flush=True)
            continue
        print(f"{model}: same slot {summary['all']['same_slot_rate']:.4f}",
              flush=True)


class JevChooser:
    """P(Document 1) from Jev's choice answers, many requests at a time."""

    def __init__(self, task: str, concurrency: int = 64):
        import os

        self.api_key = os.environ["TYPESAFE_API_KEY"]
        self.concurrency = concurrency
        self.question = {"type": "choice",
                         "instructions": LOTUS_INSTRUCTION_TEXT + "\n\nQuestion: "
                         + QUESTION_TEXTS[task],
                         "criteria": {"1": LABELS[0], "2": LABELS[1]}}

    def __call__(self, documents) -> list[float]:
        """Return P(Document 1) for each pair document."""
        import asyncio

        return asyncio.run(self._choose_all(documents))

    async def _ask(self, client, semaphore, document) -> float:
        import asyncio

        import httpx

        payload = {"model": JEV_MODEL, "state": document,
                   "questions": {"winner": self.question}}
        delay = 0.25
        async with semaphore:
            for _ in range(12):
                try:
                    response = await client.post(
                        "https://api.typesafe.ai/v1/systemone",
                        headers={"Authorization": f"Bearer {self.api_key}"},
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

    async def _choose_all(self, documents) -> list[float]:
        import asyncio

        import httpx

        semaphore = asyncio.Semaphore(self.concurrency)
        limits = httpx.Limits(max_connections=self.concurrency)
        async with httpx.AsyncClient(timeout=120, limits=limits) as client:
            return await asyncio.gather(*(self._ask(client, semaphore, document)
                                          for document in documents))


def jev_main(task: str):
    """Run the swap test on one task against Jev and upload the record."""
    pairs, texts, labels = load_pairs(task)
    record = run_swap(pairs, texts, labels, JevChooser(task), pair_template(task))
    record.update(model=JEV_MODEL, task=task)
    local = Path("results") / record_path(task, JEV_MODEL)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(record))
    with results.batch_upload(force=True) as upload:
        upload.put_file(str(local), "/" + record_path(task, JEV_MODEL))
    print(json.dumps(record["summary"], indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jev", action="store_true",
                        help="run against the Jev API instead of Modal")
    parser.add_argument("--tasks", default="sales")
    arguments = parser.parse_args()
    if arguments.jev:
        for name in arguments.tasks.split(","):
            jev_main(name)
