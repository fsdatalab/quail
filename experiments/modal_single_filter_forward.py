"""Compare stock vLLM with a packed forward pass for one filter.

Both methods run the first planted filter over the same tokenized
documents in one Modal container on one H100. The vLLM control uses the
best corrected production setting. The packed method concatenates whole
prompts, resets position ids at each prompt boundary, runs one model
forward pass per token batch, and scores only the allowed YES and NO
tokens. It does not allocate persistent KV.

Prediction: the packed method should process input tokens 5 to 10 percent
faster because it removes paged KV writes, request handling, and the full
vocabulary projection. The FP8 matrix multiplications, FP8 activation
quantization, and attention remain in both methods.

Result: the prediction was false. vLLM averaged 97,637 input tokens per
second, while packed Transformers averaged 33,295. The packed path used
6.934 GiB but reached only 34.1 percent of vLLM throughput.

Run:
    modal run experiments/modal_single_filter_forward.py::probe
    modal run experiments/modal_single_filter_forward.py::compare
"""

import modal

from workload import IMAGE_BASE, MODEL, hf_cache, results_vol


FLASH_ATTN_4_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "fa4-v4.0.0.beta26/flash_attn_4-4.0.0b26-py3-none-any.whl"
)
BEST_BATCH_TOKENS = 25_305
VLLM_GRAPH_TOKENS = 8_192

forward_image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "yappi", "accelerate")
    .pip_install("kernels==0.16.0")
    .pip_install(FLASH_ATTN_4_WHEEL, extra_options="--no-deps")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("workload")
)

app = modal.App("quail-single-filter-forward")


def _whole_prompt_batches(prompts, batch_tokens):
    """Pack complete prompts without splitting any prompt."""
    batches = []
    current = []
    current_tokens = 0
    for prompt in prompts:
        prompt_tokens = len(prompt)
        if prompt_tokens > batch_tokens:
            raise ValueError(
                f"prompt has {prompt_tokens} tokens but the batch limit is "
                f"{batch_tokens}"
            )
        if current and current_tokens + prompt_tokens > batch_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(prompt)
        current_tokens += prompt_tokens
    if current:
        batches.append(current)
    return batches


@app.function(image=forward_image, gpu="H100!", timeout=1800,
              volumes={"/root/.cache/huggingface": hf_cache})
def probe() -> str:
    """Check the direct FP8 and packed FlashAttention dependencies."""
    import importlib.metadata
    import json

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = [
        tokenizer("The flag is YES. Answer=", add_special_tokens=False)
        ["input_ids"],
        tokenizer("The flag is NO. Answer=", add_special_tokens=False)
        ["input_ids"],
    ]
    lengths = [len(ids) for ids in prompt_ids]
    cumulative = [0, lengths[0], sum(lengths)]
    input_ids = torch.tensor(
        [[token for ids in prompt_ids for token in ids]], device="cuda"
    )
    position_ids = torch.tensor(
        [[position for length in lengths for position in range(length)]],
        device="cuda",
    )
    cumulative_lengths = torch.tensor(
        cumulative, dtype=torch.int32, device="cuda"
    )
    with torch.inference_mode():
        hidden = model.model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            cu_seq_lens_q=cumulative_lengths,
            cu_seq_lens_k=cumulative_lengths,
            max_length_q=max(lengths),
            max_length_k=max(lengths),
        ).last_hidden_state
        separate_final = []
        for ids in prompt_ids:
            one_ids = torch.tensor([ids], device="cuda")
            one_positions = torch.arange(len(ids), device="cuda")[None, :]
            one_hidden = model.model(
                input_ids=one_ids,
                position_ids=one_positions,
                use_cache=False,
            ).last_hidden_state
            separate_final.append(one_hidden[0, -1])
        packed_final = hidden[0, torch.tensor(
            [lengths[0] - 1, sum(lengths) - 1], device="cuda"
        )]
        separate_final = torch.stack(separate_final)
        final_max_abs_difference = (
            packed_final - separate_final
        ).abs().max().item()
        final_cosine_similarity = torch.nn.functional.cosine_similarity(
            packed_final.float(), separate_final.float(), dim=1
        ).tolist()
    result = {
        "model": MODEL,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "flash_attn_4": importlib.metadata.version("flash-attn-4"),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "packed_cu_seqlens_call_succeeded": True,
        "model_class": type(model).__name__,
        "attention": model.config._attn_implementation,
        "quantization": model.config.quantization_config.to_dict(),
        "packed_hidden_shape": list(hidden.shape),
        "packed_vs_separate_final_max_abs_difference": (
            final_max_abs_difference
        ),
        "packed_vs_separate_final_cosine_similarity": (
            final_cosine_similarity
        ),
    }
    print(result, flush=True)
    return json.dumps(result)


@app.function(image=forward_image, gpu="H100!", timeout=7200, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def compare(n_docs: int = 10_000, reps: int = 3,
            batch_tokens: int = BEST_BATCH_TOKENS) -> str:
    import gc
    import json
    import time

    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    from workload import build_corpus, yes_no_ids

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    body_ids, question_ids, flags = build_corpus(tokenizer, n_docs)
    prompts = [body_ids[i] + question_ids[0] for i in range(n_docs)]
    expected = [int(flags[i][0]) for i in range(n_docs)]
    total_prompt_tokens = sum(map(len, prompts))
    longest_prompt = max(map(len, prompts))
    batches = _whole_prompt_batches(prompts, batch_tokens)
    yes_ids, no_ids = yes_no_ids(tokenizer)
    allowed_ids = sorted(yes_ids | no_ids)

    report = {
        "model": MODEL,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "n_docs": n_docs,
        "prompt_tokens": total_prompt_tokens,
        "longest_prompt": longest_prompt,
        "batch_tokens": batch_tokens,
        "packed_batches": len(batches),
        "allowed_tokens": {
            str(token_id): tokenizer.decode([token_id])
            for token_id in allowed_ids
        },
        "yes_no_id_overlap": sorted(yes_ids & no_ids),
        "prediction": "packed forward should be 5 to 10 percent faster",
        "runs": [],
    }
    print(
        f"[single filter] {n_docs:,} prompts, {total_prompt_tokens:,} tokens, "
        f"longest {longest_prompt:,}, {len(batches)} packed batches",
        flush=True,
    )

    graph_tokens = min(batch_tokens, VLLM_GRAPH_TOKENS)
    llm = LLM(
        model=MODEL,
        kv_cache_dtype="fp8",
        max_model_len=4608,
        max_num_seqs=4096,
        max_num_batched_tokens=batch_tokens,
        gpu_memory_utilization=0.88,
        enable_prefix_caching=False,
        disable_log_stats=True,
        compilation_config={
            "max_cudagraph_capture_size": graph_tokens,
            "cudagraph_capture_sizes": [graph_tokens],
        },
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        min_tokens=1,
        allowed_token_ids=allowed_ids,
    )
    vllm_prompts = [{"prompt_token_ids": prompt} for prompt in prompts]
    llm.generate(vllm_prompts[:64], sampling, use_tqdm=False)
    for rep in range(reps):
        started = time.perf_counter()
        outputs = llm.generate(vllm_prompts, sampling, use_tqdm=False)
        wall = time.perf_counter() - started
        predicted = []
        for output in outputs:
            token_id = int(output.outputs[0].token_ids[0])
            predicted.append(1 if token_id in yes_ids else 0)
        wrong = sum(a != b for a, b in zip(predicted, expected))
        row = {
            "method": "vllm",
            "rep": rep,
            "wall": round(wall, 4),
            "tokens_per_second": round(total_prompt_tokens / wall, 1),
            "wrong": wrong,
            "max_num_seqs": 4096,
            "gpu_memory_utilization": 0.88,
            "cudagraph_capture_sizes": [graph_tokens],
        }
        report["runs"].append(row)
        print(f"[single filter] {row}", flush=True)
    control_predictions = list(predicted)

    del outputs, llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_4",
    )
    model.eval()
    model.config.use_cache = False
    selected_ids = torch.tensor(allowed_ids, device="cuda")
    yes_columns = torch.tensor(
        [i for i, token_id in enumerate(allowed_ids) if token_id in yes_ids],
        device="cuda",
    )
    no_columns = torch.tensor(
        [i for i, token_id in enumerate(allowed_ids) if token_id in no_ids],
        device="cuda",
    )
    selected_weights = model.lm_head.weight.index_select(0, selected_ids)

    def packed_batch(batch):
        lengths = [len(prompt) for prompt in batch]
        flat_ids = [token for prompt in batch for token in prompt]
        flat_positions = [position for length in lengths
                          for position in range(length)]
        final_indices = []
        running = 0
        for length in lengths:
            running += length
            final_indices.append(running - 1)
        cumulative_lengths = [0]
        for length in lengths:
            cumulative_lengths.append(cumulative_lengths[-1] + length)
        return (
            torch.tensor(flat_ids, dtype=torch.long, device="cuda")[None, :],
            torch.tensor(flat_positions, dtype=torch.long,
                         device="cuda")[None, :],
            torch.tensor(final_indices, dtype=torch.long, device="cuda"),
            torch.tensor(cumulative_lengths, dtype=torch.int32,
                         device="cuda"),
            max(lengths),
        )

    @torch.inference_mode()
    def run_packed(selected_batches):
        all_predictions = []
        for batch in selected_batches:
            (input_ids, position_ids, final_indices, cumulative_lengths,
             max_length) = packed_batch(batch)
            hidden = model.model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
                cu_seq_lens_q=cumulative_lengths,
                cu_seq_lens_k=cumulative_lengths,
                max_length_q=max_length,
                max_length_k=max_length,
            ).last_hidden_state[0].index_select(0, final_indices)
            scores = F.linear(hidden, selected_weights)
            yes_scores = scores.index_select(1, yes_columns).amax(dim=1)
            no_scores = scores.index_select(1, no_columns).amax(dim=1)
            all_predictions.extend((yes_scores > no_scores).int().cpu().tolist())
        return all_predictions

    run_packed(batches[:1])
    torch.cuda.synchronize()
    for rep in range(reps):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        predicted = run_packed(batches)
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        wrong = sum(a != b for a, b in zip(predicted, expected))
        disagreements = sum(
            a != b for a, b in zip(predicted, control_predictions)
        )
        row = {
            "method": "packed_forward",
            "rep": rep,
            "wall": round(wall, 4),
            "tokens_per_second": round(total_prompt_tokens / wall, 1),
            "wrong": wrong,
            "disagrees_with_vllm": disagreements,
            "batches": len(batches),
            "persistent_kv": False,
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated() / 2**30, 3
            ),
        }
        report["runs"].append(row)
        print(f"[single filter] {row}", flush=True)

    rates = {}
    for method in ("vllm", "packed_forward"):
        values = [row["tokens_per_second"] for row in report["runs"]
                  if row["method"] == method]
        rates[method] = round(sum(values) / len(values), 1)
    report["mean_tokens_per_second"] = rates
    report["packed_speedup"] = round(
        rates["packed_forward"] / rates["vllm"], 4
    )

    outpath = "/results/single_filter_forward.json"
    with open(outpath, "w") as output:
        json.dump(report, output, indent=2)
    results_vol.commit()
    report_json = json.dumps(report, indent=2)
    print(report_json, flush=True)
    return report_json


@app.local_entrypoint()
def main(n_docs: int = 10_000, reps: int = 3,
         batch_tokens: int = BEST_BATCH_TOKENS,
         out: str = "results/engine/single_filter_forward.json"):
    import json
    import os

    report = json.loads(compare.remote(n_docs, reps, batch_tokens))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as output:
        json.dump(report, output, indent=2)
    print(f"saved {out}")
