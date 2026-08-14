"""Compare stock vLLM with a packed forward pass for one filter.

Both methods run the first planted filter over the same tokenized
documents in one Modal container on one H100. The vLLM control uses the
best corrected production setting. The packed method concatenates whole
prompts, resets position ids at each prompt boundary, runs one forward
pass per token batch with no engine and no saved KV, and scores only
the allowed YES and NO tokens at each document's last position.

History: the first version of this file ran the packed side on eager
HuggingFace Transformers and measured 33,295 tokens per second against
the control's 97,637 - 0.34x, with the loss attributed to kernels, not
the idea (banked in results/engine/single_filter_forward.json and git
history). This version replaces that packed side with vLLM's own
kernels, called directly: the checkpoint is loaded through vLLM's model
loader (so the weights are merged and processed exactly as the engine
runs them), the matrix multiplies go through the same DeepGEMM path the
engine uses, attention is vLLM's bundled variable-length FlashAttention
with no KV written, and the norm+quant and silu+quant steps run in two
variants - the separate kernels the engine executes today, and the
fused kernels vLLM ships but cannot reach for this checkpoint (the
graph-dump experiment showed its fusion patterns have no quant node to
match, because the quant lives inside the compiled linear op). Calling
the kernels directly is the only route to that A/B.

Prediction, stated before the run, against the same-container control:
  - packed on vLLM kernels, separate quant: within 5 percent of the
    control either side (about 93,000 to 102,000 tokens per second).
    Removing the engine is worth roughly nothing (GPU 99.5 percent
    busy, 2.94 ms fixed cost per ~263 ms step), skipping KV writes
    is worth about 0.2 percent, and eager per-op launches without
    CUDA graphs cost a few percent back.
  - packed with the fused kernels: 4 to 8 percent faster than the
    separate-quant variant. The fused sites are the two norm+quant
    pairs per layer (minus the first layer, which runs unfused) and
    the silu+quant pair; the attention-output quant has no fusion
    partner and stays separate in both variants.
  - wrong answers stay within a few tens of the control's 2,990 of
    10,000. Disagreements with the control in the low thousands are
    expected, not a failure: the fix experiment measured 1,768
    flipped answers from swapping one silu kernel.
  - the profiled repetition of the fused variant shows
    rms_norm_per_block_quant and silu_and_mul_per_block_quant
    kernels; the separate-quant variant shows neither.

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

# Kernel classes, first match wins; attention before gemm because the
# sm90 FlashAttention mainloop is a cutlass::device_kernel.
KERNEL_CLASS_RULES = (
    ("attention", ("attn", "attention", "flash", "fmha")),
    ("gemm", ("gemm", "cutlass", "nvjet")),
    ("quantize", ("quant", "scale", "cast")),
    ("norm", ("norm", "rms")),
    ("elementwise", ("silu", "gelu", "add", "mul", "residual")),
)


def _classify_kernel(name):
    k = name.lower()
    for cls, keys in KERNEL_CLASS_RULES:
        if any(s in k for s in keys):
            return cls
    return "other"


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


def _load_vllm_model():
    """The checkpoint as vLLM's processed module: merged qkv and
    gate_up, FP8 weights and block scales laid out for DeepGEMM. No
    engine and no KV pool - just the weights and layer modules."""
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model
    from vllm.utils.network_utils import get_open_port

    # enforce_eager keeps compilation out of it; custom ops then
    # default on, so module calls run vLLM's CUDA kernels eagerly.
    config = EngineArgs(model=MODEL, dtype="bfloat16",
                        enforce_eager=True).create_engine_config()
    with set_current_vllm_config(config):
        # the model classes read the parallel groups even on one GPU,
        # and the group setup itself reads the current config
        init_distributed_environment(
            world_size=1, rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            local_rank=0, backend="nccl")
        ensure_model_parallel_initialized(1, 1)
        model = get_model(vllm_config=config)
    torch.cuda.synchronize()
    return model


class PackedPipeline:
    """One filter as plain forward passes over packed chunks.

    Calls vLLM's kernels directly: DeepGEMM for the matrix multiplies,
    the module's own norm, QK-norm, and rotary layers, bundled varlen
    FlashAttention with no KV, and either the engine's separate
    per-token-group quant or the shipped fused norm+quant and
    silu+quant kernels."""

    GROUP = 128

    def __init__(self, model):
        import torch
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

        self.torch = torch
        self.model = model
        self.layers = model.model.layers
        self.embed = model.model.embed_tokens
        self.final_norm = model.model.norm
        self.rotary = self.layers[0].self_attn.rotary_emb
        attn = self.layers[0].self_attn
        self.num_q_heads = attn.num_heads
        self.num_kv_heads = attn.num_kv_heads
        self.head_dim = attn.head_dim
        self.use_ue8m0 = bool(is_deep_gemm_e8m0_used())
        self.fp8 = torch.float8_e4m3fn

    @staticmethod
    def weight_scale(linear):
        for name in ("weight_scale", "weight_scale_inv"):
            scale = getattr(linear, name, None)
            if scale is not None:
                return scale
        raise AttributeError(f"no weight scale on {type(linear).__name__}")

    def gemm(self, q_input, input_scale, linear):
        # mirrors run_deepgemm in vLLM's flashinfer scaled_mm kernel
        from vllm.utils.deep_gemm import fp8_gemm_nt
        out = self.torch.empty(
            (q_input.shape[0], linear.weight.shape[0]),
            dtype=self.torch.bfloat16, device=q_input.device,
        )
        fp8_gemm_nt((q_input, input_scale),
                    (linear.weight, self.weight_scale(linear)),
                    out, is_deep_gemm_e8m0_used=self.use_ue8m0)
        return out

    def quant(self, x):
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )
        return per_token_group_quant_fp8(
            x, group_size=self.GROUP, column_major_scales=True,
            use_ue8m0=self.use_ue8m0,
        )

    def _col_major_scales(self, n_tokens, width):
        return self.torch.empty(
            (width // self.GROUP, n_tokens),
            dtype=self.torch.float32, device="cuda",
        ).permute(-1, -2)

    def fused_norm_quant(self, hidden, norm, residual):
        """rms_norm_per_block_quant: the fused kernel vLLM ships but
        its patterns cannot reach for this checkpoint. Mutates
        residual in place, like the engine's fused-add norm."""
        result = self.torch.empty_like(hidden, dtype=self.fp8)
        scales = self._col_major_scales(*hidden.shape)
        self.torch.ops._C.rms_norm_per_block_quant(
            result=result, input=hidden, weight=norm.weight, scale=scales,
            epsilon=norm.variance_epsilon, scale_ub=None, residual=residual,
            group_size=self.GROUP, is_scale_transposed=True,
        )
        return result, scales

    def fused_silu_quant(self, gate_up):
        n_tokens, doubled = gate_up.shape
        result = self.torch.empty(
            (n_tokens, doubled // 2), dtype=self.fp8, device="cuda")
        scales = self._col_major_scales(n_tokens, doubled // 2)
        self.torch.ops._C.silu_and_mul_per_block_quant(
            out=result, input=gate_up, scales=scales,
            group_size=self.GROUP, scale_ub=None, is_scale_transposed=True,
        )
        return result, scales

    def silu_and_mul(self, gate_up):
        n_tokens, doubled = gate_up.shape
        out = self.torch.empty(
            (n_tokens, doubled // 2),
            dtype=gate_up.dtype, device=gate_up.device)
        self.torch.ops._C.silu_and_mul(out, gate_up)
        return out

    def attention(self, q, k, v, cu_seqlens, max_seqlen):
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        n_tokens = q.shape[0]
        out = flash_attn_varlen_func(
            q.view(n_tokens, self.num_q_heads, self.head_dim),
            k.view(n_tokens, self.num_kv_heads, self.head_dim),
            v.view(n_tokens, self.num_kv_heads, self.head_dim),
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=True,
        )
        return out.reshape(n_tokens, self.num_q_heads * self.head_dim)

    @staticmethod
    def pack(batch):
        import torch
        lengths = [len(prompt) for prompt in batch]
        flat_ids = [token for prompt in batch for token in prompt]
        flat_positions = [position for length in lengths
                          for position in range(length)]
        final_indices = []
        running = 0
        for length in lengths:
            running += length
            final_indices.append(running - 1)
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        return (
            torch.tensor(flat_ids, dtype=torch.long, device="cuda"),
            torch.tensor(flat_positions, dtype=torch.long, device="cuda"),
            torch.tensor(final_indices, dtype=torch.long, device="cuda"),
            torch.tensor(cumulative, dtype=torch.int32, device="cuda"),
            max(lengths),
        )

    def forward_chunk(self, packed, fused):
        (input_ids, positions, final_indices, cu_seqlens,
         max_seqlen) = packed
        n_tokens = input_ids.shape[0]
        hidden = self.embed(input_ids)
        residual = None
        for layer in self.layers:
            attn = layer.self_attn
            if residual is None:
                # first layer: no residual to fuse, run unfused
                residual = hidden
                normed = layer.input_layernorm(hidden)
                q_in, q_scale = self.quant(normed)
            elif fused:
                q_in, q_scale = self.fused_norm_quant(
                    hidden, layer.input_layernorm, residual)
            else:
                normed, residual = layer.input_layernorm(hidden, residual)
                q_in, q_scale = self.quant(normed)
            qkv = self.gemm(q_in, q_scale, attn.qkv_proj)

            q_width = self.num_q_heads * self.head_dim
            kv_width = self.num_kv_heads * self.head_dim
            q, k, v = qkv.split([q_width, kv_width, kv_width], dim=-1)
            q = attn.q_norm(
                q.reshape(n_tokens, self.num_q_heads, self.head_dim)
            ).reshape(n_tokens, q_width)
            k = attn.k_norm(
                k.reshape(n_tokens, self.num_kv_heads, self.head_dim)
            ).reshape(n_tokens, kv_width)
            q, k = self.rotary(positions, q, k)
            attn_out = self.attention(q, k, v, cu_seqlens, max_seqlen)

            # the attention output quant has no fusion partner in
            # either variant
            o_in, o_scale = self.quant(attn_out)
            hidden = self.gemm(o_in, o_scale, attn.o_proj)

            if fused:
                g_in, g_scale = self.fused_norm_quant(
                    hidden, layer.post_attention_layernorm, residual)
            else:
                normed, residual = layer.post_attention_layernorm(
                    hidden, residual)
                g_in, g_scale = self.quant(normed)
            gate_up = self.gemm(g_in, g_scale, layer.mlp.gate_up_proj)
            if fused:
                d_in, d_scale = self.fused_silu_quant(gate_up)
            else:
                d_in, d_scale = self.quant(self.silu_and_mul(gate_up))
            hidden = self.gemm(d_in, d_scale, layer.mlp.down_proj)

        last_hidden = hidden.index_select(0, final_indices)
        last_residual = residual.index_select(0, final_indices)
        normed, _ = self.final_norm(last_hidden, last_residual)
        return normed


@app.function(image=forward_image, gpu="H100!", timeout=3600, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache})
def probe(n_docs: int = 8) -> str:
    """Check the kernels and the loader before the timed run: op
    schemas, module attribute names and shapes, and one packed chunk
    through both variants with finite outputs and matching answers."""
    import json

    import torch
    from transformers import AutoTokenizer

    from workload import build_corpus

    result = {"ops": {}}
    for op_name in ("rms_norm_per_block_quant",
                    "silu_and_mul_per_block_quant",
                    "silu_and_mul", "fused_add_rms_norm"):
        op = getattr(torch.ops._C, op_name, None)
        result["ops"][op_name] = (
            str(op.default._schema) if op is not None else "MISSING"
        )

    model = _load_vllm_model()
    pipeline = PackedPipeline(model)
    layer = model.model.layers[0]
    result["module"] = {
        "model_class": type(model).__name__,
        "num_layers": len(model.model.layers),
        "q_heads": pipeline.num_q_heads,
        "kv_heads": pipeline.num_kv_heads,
        "head_dim": pipeline.head_dim,
        "qkv_weight": [list(layer.self_attn.qkv_proj.weight.shape),
                       str(layer.self_attn.qkv_proj.weight.dtype)],
        "qkv_scale": [list(
            PackedPipeline.weight_scale(layer.self_attn.qkv_proj).shape),
            str(PackedPipeline.weight_scale(
                layer.self_attn.qkv_proj).dtype)],
        "gate_up_weight": list(layer.mlp.gate_up_proj.weight.shape),
        "down_weight": list(layer.mlp.down_proj.weight.shape),
        "lm_head": hasattr(model, "lm_head"),
        "use_ue8m0": pipeline.use_ue8m0,
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    body_ids, question_ids, _flags = build_corpus(tokenizer, n_docs)
    prompts = [body_ids[i] + question_ids[0] for i in range(n_docs)]
    packed = PackedPipeline.pack(prompts)
    with torch.inference_mode():
        unfused = pipeline.forward_chunk(packed, fused=False)
        fused = pipeline.forward_chunk(packed, fused=True)
    diff = (unfused.float() - fused.float()).abs().max().item()
    result["chunk"] = {
        "tokens": int(packed[0].shape[0]),
        "docs": n_docs,
        "unfused_finite": bool(torch.isfinite(unfused).all().item()),
        "fused_finite": bool(torch.isfinite(fused).all().item()),
        "fused_vs_unfused_max_abs_diff": round(diff, 4),
    }
    print(json.dumps(result, indent=2), flush=True)
    return json.dumps(result)


@app.function(image=forward_image, gpu="H100!", timeout=7200, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def compare(n_docs: int = 10_000, reps: int = 3,
            batch_tokens: int = BEST_BATCH_TOKENS,
            profile_docs: int = 1024) -> str:
    import gc
    import json
    import time

    import torch
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from workload import build_corpus, yes_no_ids

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    body_ids, question_ids, flags = build_corpus(tokenizer, n_docs)
    prompts = [body_ids[i] + question_ids[0] for i in range(n_docs)]
    expected = [int(flags[i][0]) for i in range(n_docs)]
    total_prompt_tokens = sum(map(len, prompts))
    batches = _whole_prompt_batches(prompts, batch_tokens)
    yes_ids, no_ids = yes_no_ids(tokenizer)
    allowed_ids = sorted(yes_ids | no_ids)

    report = {
        "model": MODEL,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "n_docs": n_docs,
        "prompt_tokens": total_prompt_tokens,
        "batch_tokens": batch_tokens,
        "packed_batches": len(batches),
        "prediction": (
            "packed on vLLM kernels within 5 percent of control; fused "
            "variant 4 to 8 percent over the unfused packed variant; "
            "wrong within tens of 2,990"
        ),
        "runs": [],
        "profiles": {},
    }
    print(
        f"[single filter] {n_docs:,} prompts, {total_prompt_tokens:,} "
        f"tokens, {len(batches)} packed batches",
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
        }
        report["runs"].append(row)
        print(f"[single filter] {row}", flush=True)
    control_predictions = list(predicted)

    del outputs, llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)

    model = _load_vllm_model()
    pipeline = PackedPipeline(model)
    selected_ids = torch.tensor(allowed_ids, device="cuda")
    yes_columns = torch.tensor(
        [i for i, token_id in enumerate(allowed_ids) if token_id in yes_ids],
        device="cuda",
    )
    no_columns = torch.tensor(
        [i for i, token_id in enumerate(allowed_ids) if token_id in no_ids],
        device="cuda",
    )
    selected_weights = model.lm_head.weight.index_select(
        0, selected_ids).to(torch.bfloat16)
    packed_batches = [PackedPipeline.pack(batch) for batch in batches]

    @torch.inference_mode()
    def run_packed(selected, fused):
        predictions = []
        for packed in selected:
            normed = pipeline.forward_chunk(packed, fused=fused)
            scores = F.linear(normed, selected_weights)
            yes_scores = scores.index_select(1, yes_columns).amax(dim=1)
            no_scores = scores.index_select(1, no_columns).amax(dim=1)
            predictions.extend((yes_scores > no_scores).int().cpu().tolist())
        return predictions

    for fused in (False, True):
        name = "packed_fused" if fused else "packed_separate_quant"
        # two warmup chunks: DeepGEMM compiles its kernels on first use
        run_packed(packed_batches[:2], fused)
        torch.cuda.synchronize()
        for rep in range(reps):
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            predicted = run_packed(packed_batches, fused)
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
            wrong = sum(a != b for a, b in zip(predicted, expected))
            row = {
                "method": name,
                "rep": rep,
                "wall": round(wall, 4),
                "tokens_per_second": round(total_prompt_tokens / wall, 1),
                "wrong": wrong,
                "disagrees_with_vllm": sum(
                    a != b for a, b in zip(predicted, control_predictions)
                ),
                "persistent_kv": False,
                "peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated() / 2**30, 3
                ),
            }
            report["runs"].append(row)
            print(f"[single filter] {row}", flush=True)

        profile_batches = _whole_prompt_batches(
            prompts[:profile_docs], batch_tokens)
        profiled = [PackedPipeline.pack(batch) for batch in profile_batches]
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            run_packed(profiled, fused)
        classes = dict(gemm=0, quantize=0, norm=0, elementwise=0,
                       attention=0, other=0)
        by_name = {}
        total = 0
        for event in prof.events():
            duration = getattr(event, "device_time_total", 0) or 0
            if event.device_type is None or duration <= 0:
                continue
            classes[_classify_kernel(event.key)] += duration
            by_name[event.key] = by_name.get(event.key, 0) + duration
            total += duration
        summary = {"total_kernel_us": round(total, 1)}
        for cls, us in classes.items():
            summary[f"{cls}_frac"] = round(us / total, 4) if total else 0
        summary["fused_kernels"] = {
            key[:100]: round(us, 1) for key, us in by_name.items()
            if "quant" in key.lower()
            and any(s in key.lower() for s in ("rms", "norm", "silu"))
        }
        summary["top_kernels"] = [
            [_classify_kernel(key), round(us, 1), key[:120]]
            for key, us in sorted(by_name.items(), key=lambda kv: -kv[1])[:12]
        ]
        report["profiles"][name] = summary
        print(f"[single filter] {name} profile "
              + json.dumps({k: v for k, v in summary.items()
                            if k != "top_kernels"}), flush=True)

    rates = {}
    for method in ("vllm", "packed_separate_quant", "packed_fused"):
        values = [row["tokens_per_second"] for row in report["runs"]
                  if row["method"] == method]
        rates[method] = round(sum(values) / len(values), 1)
    report["mean_tokens_per_second"] = rates
    report["relative_to_vllm"] = {
        method: round(rate / rates["vllm"], 4)
        for method, rate in rates.items()
    }

    outpath = "/results/single_filter_forward_vllm_kernels.json"
    with open(outpath, "w") as output:
        json.dump(report, output, indent=2)
    results_vol.commit()
    report_json = json.dumps(report, indent=2)
    print(report_json, flush=True)
    return report_json


@app.local_entrypoint()
def main(n_docs: int = 10_000, reps: int = 3,
         batch_tokens: int = BEST_BATCH_TOKENS,
         out: str = "results/engine/single_filter_forward_vllm_kernels.json"):
    import json
    import os

    report = json.loads(compare.remote(n_docs, reps, batch_tokens))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as output:
        json.dump(report, output, indent=2)
    print(f"saved {out}")
