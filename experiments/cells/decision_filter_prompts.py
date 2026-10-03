"""Decision-2.0-Kai-0.6B on the QUAIL-B IMDB filters: model or prompt?

Every QUAIL-B filter template ends with "answer TRUE if ..., FALSE
otherwise", while the decision2-noul layout labels its options "No" and
"Yes". The model's own code (fp32 under bf16 autocast) answers the
IMDB F1, F4, and F5 filters over all 5,000 sf 0.1 reviews three ways,
scored against the Qwen3 32B reference labels:

- as_run: Quail's decision2-noul prompt, options "No" and "Yes".
- true_false_options: the same prompt, options described "FALSE" and
  "TRUE".
- question_only: the template's first sentence alone, options "No" and
  "Yes" (a hand-edited prompt, an upper bound for prompt fixes).

Quail's own F1 answers from a QUAIL-B IMDB-1 run ($RUN, a directory
under /results/benchmarks/quailb) are compared with as_run, which is
the same prompt.

Prediction: true_false_options closes part of the 14-point F1 gap to
Qwen3 4B (75.6% against 89.7%); Quail agrees with as_run on at least
97% of reviews.

    uv run modal run experiments/cells/decision_filter_prompts.py \
        --quail-answers $RUN/quail/imdb/IMDB-1/filters-0.parquet \
        2>&1 | tee /tmp/decision_filter_prompts.log

The summary is written to
/results/decision_filter_prompts/<run>_summary.json on the
quail-results volume.
"""

import json
import time

import modal

HF_NAME = "vllm-sr/Decision-2.0-Kai-0.6B"
REVISION = "881bee413681d80ebeac86afcda8b4138dae516e"
COLLECTION = "gt_72abc9af3feaea668e493ece67e980a0"
FILTERS = ("F1", "F4", "F5")

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("quail-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers==5.17.0", "safetensors",
                 "huggingface_hub", "numpy", "pyarrow")
)


def _option(key, description):
    option = json.dumps({"key": key, "description": description},
                        ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"))
    return f"\n<option>\n{option}\n</option>"


@app.function(image=image, gpu="H100!", memory=65536, timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results})
def score(prompts: dict, labels: dict, quail_answers: str, run: str) -> dict:
    """Answer every prompt variant; compare with the labels and Quail."""
    import os
    import sys

    import pyarrow.parquet as pq
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
        return {"id": str(i), "ids": ids, "candidate_positions": ends[-3:-1],
                "query_position": len(ids) - 1, "label": 0,
                "keys": ["false", "true"], "task_type": "noul",
                "family": "imdb", "teacher_probs": None}

    @torch.no_grad()
    def answer(rows):
        bits = []
        for start in range(0, len(rows), 32):
            batch = collate([item(start + k, s) for k, s in
                             enumerate(rows[start:start + 32])],
                            tokenizer.pad_token_id)
            batch = {k: v.cuda() if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch)
            bits += (logits[:, 1] > logits[:, 0]).int().tolist()
        return bits

    summary = {"run": run, "filters": {}}
    answers = {}
    for name, variants in prompts.items():
        truth = labels[name]
        row = {"n": len(truth), "label_true": sum(truth.values())}
        ids = list(truth)
        for variant, by_id in variants.items():
            got = answer([by_id[i] for i in ids])
            answers[(name, variant)] = dict(zip(ids, got))
            want = [truth[i] for i in ids]
            tp = sum(g and w for g, w in zip(got, want))
            row[variant] = {
                "accuracy": sum(g == w for g, w in zip(got, want)) / len(ids),
                "predicted_true": sum(got),
                "precision": tp / max(1, sum(got)),
                "recall": tp / max(1, sum(want)),
            }
        summary["filters"][name] = row
    if quail_answers:
        quail = {r["r"]: int(r["answer"])
                 for r in pq.read_table(quail_answers).to_pylist()}
        reference = answers[("F1", "as_run")]
        common = [i for i in quail if i in reference]
        summary["quail_F1_vs_as_run"] = {
            "n": len(common),
            "agreement": sum(quail[i] == reference[i] for i in common)
            / max(1, len(common)),
            "source": quail_answers,
        }
    out = f"/results/decision_filter_prompts/{run}_summary.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    results.commit()
    return {"path": out, **summary}


def _prompts():
    """Per filter and variant, each review's prompt as segments; the labels."""
    import tempfile

    import pyarrow.parquet as pq

    import quail_b
    from quail.logical import ColumnRef, bind_prompt

    directory = tempfile.mkdtemp()
    pq.write_table(quail_b.load_table("reviews", scale_factor=0.1),
                   f"{directory}/reviews.parquet")
    pq.write_table(quail_b.load_table("aspects", scale_factor=0.1),
                   f"{directory}/aspects.parquet")
    truth = quail_b.load_benchmark(
        ["IMDB-7"], scale_factor=0.1, data_dir=directory,
        collection_id=COLLECTION).ground_truth
    reviews = pq.read_table(f"{directory}/reviews.parquet").to_pylist()
    ref = ColumnRef("r", "reviews", "body")
    prompts, labels = {}, {}
    for name in FILTERS:
        template = getattr(quail_b.prompts, name)
        key = truth.key_for_template(template)
        labels[name] = {r["id"]: int(truth.answer(key, r["id"]))
                        for r in reviews}
        bound = bind_prompt(template, (ref,), layout="decision2-noul")
        first = template.split("\n\n")[0]
        plain = bind_prompt(first + "\n\n{0}", (ref,), layout="decision2-noul")
        variants = {}
        for variant, prompt in (("as_run", bound),
                                ("true_false_options", bound),
                                ("question_only", plain)):
            tail = list(prompt.tail_segments)
            if variant == "true_false_options":
                tail[1], tail[2] = _option("false", "FALSE"), _option("true", "TRUE")
            variants[variant] = {
                r["id"]: [prompt.preamble + r["body"] + tail[0], *tail[1:]]
                for r in reviews}
        prompts[name] = variants
    return prompts, labels


@app.local_entrypoint()
def main(quail_answers: str = ""):
    prompts, labels = _prompts()
    print({k: v["as_run"][next(iter(v["as_run"]))][0][-400:]
           for k, v in prompts.items()}, flush=True)
    call = score.spawn(prompts, labels, quail_answers,
                       time.strftime("%Y%m%d-%H%M%S"))
    print(f"score function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)
