"""Compare Quail's AI.CLASSIFY label scores with stock vLLM on the same tokens.

A QUAIL-B run (/results/benchmarks/quailb/20260928T153215Z-5beb7b9e)
and stock vLLM on Qwen3 4B picked different labels for 35 of 500
FEV-11 claims and 131 of 5,000 IMDB-11 reviews
(/results/ablations/classify-quail-same-model.json). Most of those
prompts tokenize differently: vLLM tokenized the whole prompt, Quail
the document and question apart. This cell scores the same token ids
four ways, for every disagreeing document and a sample that agreed:

- quail: Quail's production path, the question written as the
  document's frame and one suffix per label-trie node;
- quail_no_frame: Quail with the whole question in the anchor prefix;
- quail_fp32: the production path with float32 logits;
- vllm: stock vLLM's full-sequence prompt log probabilities.

    uv run modal run --detach experiments/cells/classify_scores.py \
      2>&1 | tee classify-scores.log

The summary goes to /results/ablations/classify-scores.json.
"""

import json
from pathlib import Path

import modal

from quail.bench import labeling

app = labeling.app
RESULT_PATH = Path("/results/ablations/classify-scores.json")
SAME_MODEL_PATH = Path("/results/ablations/classify-quail-same-model.json")
AGREEING_SAMPLE = 150
PREDICTION_TEXT = (
    "The frame path and the no-frame path give scores within 0.05 nats. "
    "Float32 logits move scores by at most 0.1 nats. On identical token "
    "ids, Quail and vLLM label scores differ by a median of about 0.1 "
    "nats with a 99th percentile under 1 nat, and every label "
    "disagreement is a pair of labels within that difference."
)


def _volumes() -> dict:
    return {"/root/.cache/huggingface": labeling.hf_cache,
            "/root/.cache/kernels": labeling.kernel_cache,
            "/results": labeling.results_vol}


def _documents() -> dict:
    """Return query id -> (classification, [(id, text), ...]) to score."""
    import quail_b

    checks = json.loads(SAME_MODEL_PATH.read_text())["checks"]
    chosen = {}
    for query_id in ("FEV-11", "IMDB-11"):
        query = quail_b.get_query(query_id)
        (operator,) = query._info.classifies
        relation = next(r for r in query._info.relations
                        if r.alias == operator.relation)
        table = quail_b.load_table(relation.table, scale_factor=0.1)
        texts = dict(zip(table.column("id").to_pylist(),
                         table.column(relation.text_column).to_pylist()))
        disagree = [item["id"] for item in
                    checks[f"{query_id}:{operator.id}"]["disagreements"]]
        agree = [doc for doc in texts if doc not in set(disagree)]
        ids = disagree + agree[:AGREEING_SAMPLE]
        chosen[query_id] = (operator, relation, [(doc, texts[doc]) for doc in ids],
                            len(disagree))
    return chosen


def _inputs(tokenizer, operator, relation, documents):
    from quail.logical import ColumnRef, bind_classify_prompt, label_text

    def encode(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    prompt = bind_classify_prompt(
        operator.prompt, (ColumnRef(operator.relation, relation.table,
                                    relation.text_column),),
        operator.labels, operator.descriptions, encode)
    labels = [tuple(encode(label_text(label))) for label in operator.labels]
    contexts = [list(prompt.preamble_token_ids) + encode(text)
                for _, text in documents]
    return list(prompt.tail_token_ids), labels, contexts


@app.function(image=labeling.image, gpu="H100!", memory=98304, timeout=3600,
              volumes=_volumes())
def quail_scores() -> dict:
    """Score the chosen documents on Quail's executor three ways."""
    import torch
    from transformers import AutoTokenizer

    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.loop import run_join
    from quail.backends.quail.executor.model import (
        full_output_head,
        load_model,
    )
    from quail.backends.quail.executor.models import build_pipeline
    from quail.backends.quail.executor.readout import AsyncLabelLogprobs
    from quail.cost import budgets
    from quail.execution.labels import label_scores, label_trie, trie_targets
    from quail.specs import DEVICES, MODELS

    labeling._mount()
    spec = MODELS["qwen3-4b-fp8"]
    device = DEVICES["h100-sxm"]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name,
                                              revision=spec.revision)
    model = load_model(spec.hf_name, revision=spec.revision)
    head = full_output_head(model)
    # the float32 head copy takes 1.56 GB, so it exists before the
    # arena, which is sized to half the usual pages
    head32 = head.float()
    chunk = budgets.chunk_budget(spec, device)
    arena = KVArena(n_layers=spec.layers,
                    n_pages=budgets.arena_tokens(spec, device, chunk)
                    // budgets.PAGE_TOKENS // 2,
                    page_tokens=budgets.PAGE_TOKENS, n_kv=spec.n_kv,
                    d_head=spec.d_head, dtype=torch.bfloat16)
    pipeline = build_pipeline(spec, model, arena)

    def score(readout, prefixes, suffixes, frame, nodes, targets, labels):
        answers, _, _ = run_join(
            torch, arena, pipeline, readout, prefixes, [suffixes], chunk,
            stage_frames=[frame] if frame else None,
            anchor_keys=[("probe", index) for index in range(len(prefixes))])
        return [label_scores(labels, nodes, targets, answers[0][a]).tolist()
                for a in range(len(prefixes))]

    out = {}
    with torch.inference_mode():
        for query_id, (operator, relation, documents, _) in (
                _documents().items()):
            tail, labels, contexts = _inputs(tokenizer, operator, relation,
                                             documents)
            trie = label_trie(labels)
            nodes = sorted(trie, key=lambda prefix: (len(prefix), prefix))
            targets = trie_targets(trie)
            suffixes = [[tail[-1], *node] for node in nodes]
            bf16 = AsyncLabelLogprobs(torch, torch.nn.functional, head, targets)
            fp32 = AsyncLabelLogprobs(torch, torch.nn.functional, head32,
                                      targets)
            out[query_id] = {
                "ids": [doc for doc, _ in documents],
                "quail": score(bf16, contexts, suffixes, tail[:-1], nodes,
                               targets, labels),
                "quail_no_frame": score(
                    bf16, [c + tail[:-1] for c in contexts], suffixes, None,
                    nodes, targets, labels),
                "quail_fp32": score(fp32, contexts, suffixes, tail[:-1],
                                    nodes, targets, labels),
            }
    return out


@app.function(image=labeling.image, gpu="H100!", memory=98304, timeout=3600,
              volumes=_volumes())
def vllm_scores() -> dict:
    """Score the chosen documents with stock vLLM's prompt log probabilities."""
    from vllm import LLM, SamplingParams

    from quail.specs import MODELS

    labeling._mount()
    spec = MODELS["qwen3-4b-fp8"]
    llm = LLM(model=spec.hf_name, revision=spec.revision,
              tokenizer_revision=spec.revision, max_model_len=32768,
              gpu_memory_utilization=0.9, disable_log_stats=True, seed=0)
    tokenizer = llm.get_tokenizer()
    out = {}
    for query_id, (operator, relation, documents, _) in _documents().items():
        tail, labels, contexts = _inputs(tokenizer, operator, relation,
                                         documents)
        prompts, owners = [], []
        for index, context in enumerate(contexts):
            for label in labels:
                prompts.append({"prompt_token_ids": context + tail + list(label)})
                owners.append(index)
        outputs = llm.generate(prompts, SamplingParams(
            max_tokens=1, prompt_logprobs=0, detokenize=False), use_tqdm=False)
        scores = [[] for _ in contexts]
        for prompt, owner, output, label in zip(
                prompts, owners, outputs, labels * len(contexts)):
            start = len(prompt["prompt_token_ids"]) - len(label)
            scores[owner].append(sum(
                output.prompt_logprobs[start + i][token].logprob
                for i, token in enumerate(label)))
        out[query_id] = {"ids": [doc for doc, _ in documents], "vllm": scores}
    return out


@app.function(image=labeling.publish_image, memory=16384, timeout=1800,
              volumes=_volumes())
def compare(quail_call: str, vllm_call: str) -> dict:
    """Summarize the score differences and save them to the volume."""
    import numpy as np

    quail = modal.FunctionCall.from_id(quail_call).get()
    vllm = modal.FunctionCall.from_id(vllm_call).get()
    summary = {"prediction": PREDICTION_TEXT, "function_calls": {
        "quail": quail_call, "vllm": vllm_call}, "queries": {}}
    for query_id, scores in quail.items():
        (operator, _, _, disagree) = _documents()[query_id]
        runs = {name: np.asarray(value) for name, value in scores.items()
                if name != "ids"}
        runs["vllm"] = np.asarray(vllm[query_id]["vllm"])
        winners = {name: value.argmax(axis=1) for name, value in runs.items()}

        def gap(a, b):
            difference = np.abs(runs[a] - runs[b])
            return {"median": float(np.median(difference)),
                    "p99": float(np.quantile(difference, 0.99)),
                    "max": float(difference.max()),
                    "same_label": int((winners[a] == winners[b]).sum())}

        summary["queries"][query_id] = {
            "documents": len(scores["ids"]),
            "disagreed_in_run": disagree,
            "labels": list(operator.labels),
            "quail_vs_no_frame": gap("quail", "quail_no_frame"),
            "quail_vs_fp32": gap("quail", "quail_fp32"),
            "quail_vs_vllm": gap("quail", "vllm"),
            "quail_fp32_vs_vllm": gap("quail_fp32", "vllm"),
        }
    labeling._atomic_json(RESULT_PATH, summary)
    labeling.results_vol.commit()
    return summary


@app.local_entrypoint()
def main():
    print(PREDICTION_TEXT, flush=True)
    quail_call = quail_scores.spawn()
    vllm_call = vllm_scores.spawn()
    print(f"[classify-scores] quail function call id: {quail_call.object_id}",
          flush=True)
    print(f"[classify-scores] vllm function call id: {vllm_call.object_id}",
          flush=True)
    quail_call.get()
    vllm_call.get()
    call = compare.spawn(quail_call.object_id, vllm_call.object_id)
    print(f"[classify-scores] compare function call id: {call.object_id}",
          flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)
    print(f"[classify-scores] saved {RESULT_PATH}", flush=True)
