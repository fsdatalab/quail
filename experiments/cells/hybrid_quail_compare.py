"""Compare Quail's Qwen3.5 forward pass with stock vLLM on the same tokens.

Two steps, each in its own container so the engines do not share a GPU.
reference boots stock vLLM and records, for each (document, question)
token sequence, the TRUE minus FALSE logprob of its next token. quail
loads Quail's pipeline, runs each sequence alone (one document, one
question, one filter stage), then every sequence in one chunk, and
records the same difference. The saved records are compared on the
quail-results volume.

Chains run each document alone through both stages, and the second
stage's difference is compared with stock vLLM on document plus second
question.

Prediction: the per-sequence difference agrees with stock vLLM's within
0.25 nats at the median and 1.0 at the maximum, the second stage of a
chain does the same, and the answer bits
(TRUE > FALSE) agree on at least 95 percent of sequences, alone and
batched.

    uv run modal run experiments/cells/hybrid_quail_compare.py \
        --model qwen3.5-4b-bf16 2>&1 | tee /tmp/hybrid_quail_compare.log

Records: /results/ablations/<model>-quail-compare-{reference,quail}.json.
"""

import json
import os
import statistics

import modal

from experiments.hybrid import run_filter

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
SEQUENCES = 24
BUDGET = 8192
DOCUMENT_WORDS = (40, 200, 700)
QUESTION = "Is the text above about a person?"
# the second stage of a chain shares the first stage's leading tokens
SECOND_QUESTION = "Is the text above about a place?"
TOPICS = ("a river", "an engineer", "a market", "a storm", "a violinist",
          "a bakery", "a mountain", "a teacher")


def _text(index):
    words = DOCUMENT_WORDS[index % len(DOCUMENT_WORDS)]
    topic = TOPICS[index % len(TOPICS)]
    sentence = (f"Entry {index} describes {topic} in detail, including the "
                f"place, the season, and what happened next. ")
    return (sentence * (words // 14 + 1))[: words * 6]


def _sequences(spec, tokenizer, question=QUESTION):
    """Return (document ids, question ids) pairs with the chat turn applied."""
    pairs = []
    for index in range(SEQUENCES):
        document = tokenizer(spec.turn_prefix + _text(index),
                             add_special_tokens=False)["input_ids"]
        ids = tokenizer("\n" + question + spec.turn_suffix,
                        add_special_tokens=False)["input_ids"]
        pairs.append((document, ids))
    return pairs


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def reference(model: str) -> dict:
    """Stock vLLM's TRUE minus FALSE logprob for each sequence."""
    import os

    from transformers import AutoTokenizer

    from quail.baselines.vllm import VLLMEngine
    from quail.planner.prompt_lowering import true_false_ids
    from quail.specs import MODELS

    spec = MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_name, revision=spec.revision or None)
    true_ids, false_ids = true_false_ids(tokenizer)
    allowed = sorted(true_ids | false_ids)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    class Engine(VLLMEngine):
        """The tuned engine with logprobs taken after the answer mask."""

        def llm_kwargs(self, spec):
            return {**super().llm_kwargs(spec),
                    "logprobs_mode": "processed_logprobs"}

    state, _ = Engine().boot(spec, allowed)
    pairs = _sequences(spec, tokenizer)
    second = _sequences(spec, tokenizer, SECOND_QUESTION)
    prompts = [{"prompt_token_ids": document + question}
               for document, question in pairs + second]
    from vllm import SamplingParams

    # the engine's own parameters read the answer as text; the masked
    # logprobs of the answer tokens differ by the logits' difference
    params = SamplingParams(max_tokens=1, temperature=0.0,
                            allowed_token_ids=allowed, logprobs=len(allowed))
    outputs = state["client"].generate(prompts, params)
    differences = []
    for output in outputs:
        logprobs = output.outputs[0].logprobs[0]
        best_true = max(logprobs[i].logprob for i in true_ids if i in logprobs)
        best_false = max(logprobs[i].logprob for i in false_ids
                         if i in logprobs)
        differences.append(best_true - best_false)
    record = {"model": model, "tokens": [len(d) + len(q) for d, q in pairs],
              "difference": differences[:SEQUENCES],
              "difference_second": differences[SEQUENCES:],
              "true_ids": sorted(true_ids),
              "false_ids": sorted(false_ids)}
    os.makedirs("/results/ablations", exist_ok=True)
    with open(f"/results/ablations/{model}-quail-compare-reference.json",
              "w") as out:
        json.dump(record, out)
    results.commit()
    return record


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def quail(model: str) -> dict:
    """Quail's TRUE minus FALSE logit for each sequence, alone and batched."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.model_loading import load_model
    from quail.backends.quail.executor.models import build_pipeline
    from quail.backends.quail.executor.readout import AnswerRows, AsyncAnswers
    from quail.backends.quail.worker import checkpoint_arena_args
    from quail.cost import budgets
    from quail.specs import H100_SXM, MODELS

    spec = MODELS[model]
    results.reload()
    with open(f"/results/ablations/{model}-quail-compare-reference.json") as f:
        ref = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_name, revision=spec.revision or None)
    pairs = _sequences(spec, tokenizer)
    allowed = sorted(set(ref["true_ids"]) | set(ref["false_ids"]))
    loaded = load_model(spec.hf_name, spec.revision or None,
                        answer_token_ids=allowed, max_batched_tokens=BUDGET,
                        text_only=spec.vision_tower)
    full_pages, sliding = budgets.arena_pages(spec, H100_SXM, BUDGET)
    arena = KVArena(
        n_layers=spec.layers, n_pages=full_pages,
        page_tokens=budgets.PAGE_TOKENS, n_kv=spec.n_kv, d_head=spec.d_head,
        dtype=torch.bfloat16, layer_kv=spec.kv_shapes,
        sliding_layers=spec.sliding_layer_set,
        sliding_window=spec.sliding_window, n_sliding_pages=sliding,
        **checkpoint_arena_args(spec, H100_SXM, BUDGET, full_pages))
    pipeline = build_pipeline(spec, loaded, arena)
    rows = AnswerRows(torch, F, loaded, set(ref["true_ids"]),
                      set(ref["false_ids"]))

    class Recording(AsyncAnswers):
        """Reads each answer's TRUE minus FALSE logit and its bit."""

        def __init__(self):
            super().__init__(torch, rows)
            self.differences = []

        def submit(self, normed):
            true, false = rows.logits(normed)
            self.differences.extend((true - false).float().tolist())
            return super().submit(normed)

    record = {"model": model, "slots": arena.n_state_slots,
              "pages": full_pages, "alone": [], "batched": [],
              "alone_bits": [], "batched_bits": []}
    with torch.inference_mode():
        for document, question in pairs:
            answers = Recording()
            bits = run_filter(
                torch, arena, pipeline, answers, [document], [question],
                BUDGET)
            record["alone"].append(answers.differences[-1])
            record["alone_bits"].append(int(bits[0][0]))
            for key in arena.resident_keys():
                arena.free_key(key)
        answers = Recording()
        order = []
        bits = run_filter(
            torch, arena, pipeline, answers, [d for d, _ in pairs],
            [pairs[0][1]], BUDGET,
            document_done=lambda done: order.extend(p for p, _, _ in done))
        record["batched_bits"] = [int(bits[i][0]) for i in range(len(pairs))]
        record["batched_differences"] = answers.differences
        record["batched_order"] = order
        for key in arena.resident_keys():
            arena.free_key(key)
        # chains: one document, two stages; a document that fails the
        # first stage never reaches the second
        second = _sequences(spec, tokenizer, SECOND_QUESTION)
        record["chain"] = []
        record["fresh_second"] = []
        for (document, first), (_, other) in zip(pairs, second):
            answers = Recording()
            run_filter(torch, arena, pipeline, answers, [document],
                        [first, other], BUDGET)
            record["chain"].append(answers.differences)
            for key in arena.resident_keys():
                arena.free_key(key)
            fresh = Recording()
            run_filter(torch, arena, pipeline, fresh, [document], [other], BUDGET)
            record["fresh_second"].append(fresh.differences[-1])
            for key in arena.resident_keys():
                arena.free_key(key)
        # every document through both stages in one run
        answers = Recording()
        order = []
        bits = run_filter(
            torch, arena, pipeline, answers, [d for d, _ in pairs],
            [pairs[0][1], second[0][1]], BUDGET,
            document_done=lambda done: order.extend(p for p, _, _ in done))
        record["chain_batched_answers"] = [
            [int(a) for a in bits[i]] for i in range(len(pairs))]
    torch.cuda.synchronize()
    reached = [(row[1], ref["difference_second"][i])
               for i, row in enumerate(record["chain"]) if len(row) == 2]
    chain_errors = [abs(a - b) for a, b in reached]
    first_errors = [abs(row[0] - ref["difference"][i])
                    for i, row in enumerate(record["chain"])]

    def report(name, values, bits, ordered=True):
        # a batched chunk reads its answers in packing order, so its
        # differences are compared as sorted lists
        expected = ref["difference"]
        if not ordered:
            values, expected = sorted(values), sorted(expected)
        errors = [abs(v - r) for v, r in zip(values, expected)]
        agree = [int(b == (r > 0)) for b, r in
                 zip(bits, ref["difference"])]
        return {f"{name}_median_abs_error": statistics.median(errors),
                f"{name}_max_abs_error": max(errors),
                f"{name}_bit_agreement": sum(agree) / len(agree)}

    by_document = {}
    for position, value in zip(record["batched_order"],
                               record["batched_differences"]):
        by_document[position] = value
    mapped = [abs(by_document[i] - ref["difference"][i])
              for i in range(len(pairs)) if i in by_document]
    expected_chain = []
    for first, second_value in zip(ref["difference"],
                                   ref["difference_second"]):
        expected_chain.append([int(first > 0), int(second_value > 0)]
                              if first > 0 else [0])
    chain_match = sum(int(got == want) for got, want in zip(
        record["chain_batched_answers"], expected_chain))
    reuse_errors = [abs(row[1] - fresh) for row, fresh in
                    zip(record["chain"], record["fresh_second"]) if len(row) == 2]
    record["summary"] = {
        "reuse_vs_fresh_second_max_abs_error": max(reuse_errors, default=0.0),
        "batched_mapped_documents": len(mapped),
        "batched_mapped_median_abs_error": (
            statistics.median(mapped) if mapped else None),
        "batched_mapped_max_abs_error": max(mapped) if mapped else None,
        "chain_batched_answer_agreement": chain_match / len(pairs),
        "chain_documents_reaching_second_stage": len(reached),
        "chain_first_stage_max_abs_error": max(first_errors),
        "chain_second_stage_median_abs_error": (
            statistics.median(chain_errors) if chain_errors else None),
        "chain_second_stage_max_abs_error": (
            max(chain_errors) if chain_errors else None),
        **report("alone", record["alone"], record["alone_bits"]),
        **report("batched", record["batched_differences"][:len(pairs)],
                 record["batched_bits"], ordered=False)}
    os.makedirs("/results/ablations", exist_ok=True)
    with open(f"/results/ablations/{model}-quail-compare-quail.json",
              "w") as out:
        json.dump(record, out)
    results.commit()
    return record


@app.local_entrypoint()
def main(model: str = "qwen3.5-4b-bf16", reuse_reference: bool = False):
    if not reuse_reference:
        first = reference.spawn(model)
        print(f"function call id: {first.object_id} (reference)", flush=True)
        first.get()
    second = quail.spawn(model)
    print(f"function call id: {second.object_id} (quail)", flush=True)
    print(json.dumps(second.get()["summary"], indent=1), flush=True)
