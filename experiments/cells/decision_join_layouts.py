"""Decision-2.0-Kai-0.6B join accuracy with the fixed text moved before the partner.

A join pair is a review (DOCUMENT {0}, kept in KV) and an aspect
(DOCUMENT {1}, computed per pair). Three layouts, scored by the
model's own code (fp32 under bf16 autocast) on the two QUAIL-B IMDB
join predicates at sf 0.1, against the Qwen3 32B reference labels:

- trained: review, aspect, question, options, closing line (Quail's
  decision2-noul layout today; about 70 fixed tokens per pair).
- options_first: review, question, options, aspect, closing line
  (about 17 fixed tokens per pair; the option rows do not see the
  aspect).
- question_first: review, question, aspect, options, closing line
  (about 47 fixed tokens per pair).

Prediction: options_first loses accuracy against trained, by an amount
I cannot estimate; question_first loses less than options_first.

    uv run modal run experiments/cells/decision_join_layouts.py \
        2>&1 | tee /tmp/decision_join_layouts.log

The summary is written to
/results/decision_join_layouts/<run>_summary.json on the
quail-results volume.
"""

import json
import time

import modal

HF_NAME = "vllm-sr/Decision-2.0-Kai-0.6B"
REVISION = "881bee413681d80ebeac86afcda8b4138dae516e"
COLLECTION = "gt_72abc9af3feaea668e493ece67e980a0"
N_REVIEWS = 400
SEED = 20261003
NOTE = "\n\n(The document above is DOCUMENT {0}.)"
QUESTION = "\n\nTask type: noul\nQuestion:\n"
OPTIONS = ('\n<option>\n{"description":"No","key":"false"}\n</option>',
           '\n<option>\n{"description":"Yes","key":"true"}\n</option>')
CLOSING = ("\n\nSelect the single option best supported by the context "
           "and instructions.\nDecision:")

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("quail-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers==5.17.0", "safetensors",
                 "huggingface_hub", "numpy")
)


def layouts(review, aspect, question):
    """Per layout, the prompt as segments: text, no option, yes option, text."""
    partner = "\n\nDOCUMENT {1}:\n" + aspect
    context = "Context:\n" + review + NOTE
    return {
        "trained": [context + partner + QUESTION + question + "\nOptions:",
                    *OPTIONS, CLOSING],
        "options_first": [context + QUESTION + question + "\nOptions:",
                          *OPTIONS, partner + CLOSING],
        "question_first": [context + QUESTION + question + partner
                           + "\nOptions:", *OPTIONS, CLOSING],
    }


@app.function(image=image, gpu="H100!", memory=65536, timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results})
def score(pairs: list[dict], run: str) -> dict:
    """Answer every pair in every layout; compare with the labels."""
    import os
    import sys

    import torch
    from huggingface_hub import snapshot_download

    path = snapshot_download(HF_NAME, revision=REVISION)
    sys.path.insert(0, path)
    from decision2._vendor.dev2model.decision_model import (
        DecisionModel,
        collate,
    )

    model, tokenizer = DecisionModel.from_checkpoint(path)
    model = model.cuda().eval()

    def item(i, segments):
        ids, ends = [], []
        for segment in segments:
            ids += tokenizer.encode(segment, add_special_tokens=False)
            ends.append(len(ids) - 1)
        return {"id": str(i), "ids": ids, "candidate_positions": ends[1:3],
                "query_position": len(ids) - 1, "label": 0,
                "keys": ["false", "true"], "task_type": "noul",
                "family": "imdb", "teacher_probs": None}

    @torch.no_grad()
    def answer(items):
        bits = []
        for start in range(0, len(items), 32):
            batch = collate(items[start:start + 32], tokenizer.pad_token_id)
            batch = {k: v.cuda() if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch)
            bits += (logits[:, 1] > logits[:, 0]).int().tolist()
        return bits

    names = list(layouts("", "", ""))
    answers = {name: answer([
        item(i, layouts(p["review"], p["aspect"], p["question"])[name])
        for i, p in enumerate(pairs)]) for name in names}
    fixed = {name: sum(len(tokenizer.encode(s, add_special_tokens=False))
                       for s in layouts("", "", "")[name][1:])
             for name in names}
    summary = {"run": run, "pairs": len(pairs), "per_predicate": {}}
    for predicate in sorted({p["predicate"] for p in pairs}):
        index = [i for i, p in enumerate(pairs) if p["predicate"] == predicate]
        labels = [pairs[i]["label"] for i in index]
        row = {"n": len(index), "label_true": sum(labels)}
        for name in names:
            got = [answers[name][i] for i in index]
            tp = sum(g and lab for g, lab in zip(got, labels))
            row[name] = {
                "accuracy": sum(g == lab for g, lab in zip(got, labels)) / len(index),
                "predicted_true": sum(got),
                "precision": tp / max(1, sum(got)),
                "recall": tp / max(1, sum(labels)),
                "agreement_with_trained": sum(
                    g == t for g, t in zip(got, (answers["trained"][i]
                                                 for i in index))) / len(index),
            }
        summary["per_predicate"][predicate] = row
    summary["tokens_after_first_option_segment"] = fixed
    out = f"/results/decision_join_layouts/{run}_summary.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    results.commit()
    return {"path": out, **summary}


def _pairs():
    """Sampled review x aspect pairs of both IMDB join predicates, labeled."""
    import random
    import tempfile

    import pyarrow.parquet as pq

    import quail_b

    directory = tempfile.mkdtemp()
    for name in ("reviews", "aspects"):
        pq.write_table(quail_b.load_table(name, scale_factor=0.1),
                       f"{directory}/{name}.parquet")
    truth = quail_b.load_benchmark(
        ["IMDB-2", "IMDB-8"], scale_factor=0.1, data_dir=directory,
        collection_id=COLLECTION).ground_truth
    reviews = pq.read_table(f"{directory}/reviews.parquet").to_pylist()
    aspects = pq.read_table(f"{directory}/aspects.parquet").to_pylist()
    sample = random.Random(SEED).sample(reviews, N_REVIEWS)
    pairs = []
    for key, labels in truth.predicates.items():
        question = labels.predicate["template"]
        for review in sample:
            for aspect in aspects:
                pairs.append({
                    "predicate": key, "question": question,
                    "review": review["body"], "aspect": aspect["aspect"],
                    "label": int(truth.answer(key, review["id"], aspect["id"]))})
    return pairs


@app.local_entrypoint()
def main():
    pairs = _pairs()
    print(f"{len(pairs)} pairs", flush=True)
    call = score.spawn(pairs, time.strftime("%Y%m%d-%H%M%S"))
    print(f"score function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)
