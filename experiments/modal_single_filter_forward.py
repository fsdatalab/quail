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

Result: the pipeline works, the fused kernels finally fired, and they
are slower than the kernels they replace. Means: vLLM 97,321 tokens
per second, packed with separate quant 90,718 (0.932x, just past the
predicted 5 percent band), packed fused 86,566 (0.889x - the fused
variant lost 4.6 percent instead of gaining 4 to 8). The profile
says why: silu_and_mul_per_block_quant costs 845 ms per profiled
window against about 711 ms for the separate silu and quant pair it
replaces (19 percent worse), while rms_norm_per_block_quant is about
even with its pair (428 against about 410 ms). So the config-fusion
prize this branch chased is not just unreachable through the
compiler - at this batch size the shipped fused kernels lose
outright, and any fusion win now requires writing better kernels
than vLLM ships. Two real findings on the way: the packed pipeline
is substantially more accurate than the engine (2,229 and 2,131
wrong of 10,000 against the engine's 2,990), consistent with
attention reading exact bf16 K and V instead of the engine's fp8 KV
cache, and the remaining 6.8 percent speed gap to the engine is
about one third extra kernel time (the bundled varlen FlashAttention
dispatched an older kernel than the engine's FlashAttention-3, worth
about 2.4 percent) and two thirds eager per-op launch gaps that CUDA
graphs would close. Peak memory 7.57 GiB with no KV pool.

Round 2: the two mechanical fixes from that result. The attention
call now requests FlashAttention-3 explicitly (the wrapper defaults
to version 2), and the token-level forward is captured once as a
CUDA graph at fixed shapes - every chunk padded to the full 25,305
tokens with one extra padding sequence, replayed through the
recorded kernel sequence, with the per-document tail (final norm and
logits) outside the graph. Padding costs about 1.5 percent of chunk
tokens.

Round 2 prediction, stated before the run:
  - separate quant, FlashAttention-3, still eager: 92,500 to 94,000
    tokens per second (the attention kernel alone, worth about 2.4
    percent over the banked 90,718).
  - separate quant, FlashAttention-3, CUDA graphs: within 2 percent
    of the engine either side (95,400 to 99,300), target at or above
    the engine's 97,321.
  - fused, FlashAttention-3, CUDA graphs: 4 to 5 percent below the
    separate-quant graphs cell; the fused kernels' deficit is device
    time and graphs do not change it.
  - wrong answers stay near 2,229; FlashAttention-3 rounds
    differently, so shifts of low hundreds and disagreement counts
    near 1,800 against the engine are expected.
  Falsifier: if the graphs cell stays 4 or more percent below the
  engine, the gap attribution was wrong, and the profiled repetition
  names what actually remains.

Round 2 result: FlashAttention-3 delivered, CUDA graphs did not, and
the falsifier fired - round 1's gap attribution was wrong. Means:
vLLM 98,059 tokens per second, eager FlashAttention-3 94,601
(0.965), graphed 93,199 (0.950), graphed fused 88,049 (0.898).
FlashAttention-3 cut the attention kernel from 411 to 228 ms per
profiled window, worth 3.5 percent end to end against the predicted
2.4. The graphs cell ran every chunk at the full padded 25,305
tokens and lost its 1.5 percent padding tax while recovering
nothing, because the eager loop was already GPU-bound: its kernel
time equals its wall time within measurement error - and rechecked,
round 1's did too. The "two thirds launch gaps" claim came from
comparing two different profilers and was wrong; the whole gap to
the engine was always kernel time. The padding design also puts a
near-empty chunk's tail into one long causal sequence whose
attention cost grows with the square of the pad length, visible in
the graphed profile's attention share. The fused kernels still lose
under graphs (896 ms for the fused silu+quant against about 711 for
the pair it replaces). Graphed replays reproduce the eager answers
exactly: 2,228 wrong of 10,000 in both, against the engine's 2,990,
and 2,152 for the fused variant. Reported peak memory fell to 4.55
GiB under graphs, partly an accounting effect of the capture pool
predating the counter reset. Best packed configuration: eager
FlashAttention-3 at 96.5 percent of the engine with the accuracy
win; the remaining 3.5 percent is named kernel work - the engine
compiles the QK-norm and rope region into one kernel where this
pipeline runs separate norm, rope, and reshape kernels - plus
per-chunk host copies, not launch overhead.

Round 3: our own fused kernels, plus the control the accuracy
finding needs. Two Triton kernels replace vLLM's fused ops in a new
"custom" packed variant: silu+group-quant (one program per token and
4-group block, reading gate and up once and writing fp8 and
power-of-two scales directly) and residual-add+norm+group-quant (one
program per token row). The scales get the same power-of-two
rounding as the separate quant path, so DeepGEMM's internal cast is
lossless for both and the numerics comparison is clean. The engine
side gains a second cell with the KV cache in bf16 instead of fp8,
identical otherwise: if the packed pipeline's accuracy edge comes
from the engine quantizing K and V, this cell recovers it.

Round 3 prediction, stated before the run. From the round 2
profiles, the separate silu+quant pair costs about 715 ms per
profiled window at an effective 1.7 TB/s, and vLLM's fused kernel
845 to 896 ms at 0.8 TB/s; the fused traffic floor is 666 GB per
window.
  - the probe microbenchmark: custom silu+quant beats the separate
    pair by 30 to 45 percent per call; custom norm+quant beats its
    pair by 25 to 35 percent. If either loses instead, the full run
    does not launch until the kernel is fixed.
  - end to end: packed custom lands 5 to 10 percent over the
    separate packed cell - 99,000 to 104,000 tokens per second
    against the engine's roughly 98,000, making the packed pipeline
    the fastest configuration measured in this project.
  - the bf16-KV engine cell answers within a few tens of the packed
    pipeline's 2,228 wrong (confirming the fp8 KV cache costs the
    engine about 7.6 points of absolute accuracy on this workload),
    at unchanged speed within 1 percent. If it stays near 2,990, the
    accuracy edge is not the KV cache and the suspect becomes
    attention-kernel numerics.
  - custom-variant answers stay within about 100 disagreements of
    the separate packed variant, wrong near 2,228.

Round 3 result: the central claims held and two numbers beat their
bands. Means on this container: engine fp8-KV 95,902, engine bf16-KV
101,518, packed separate 92,991, packed custom 103,913 tokens per
second. The custom cell is the fastest and the most accurate
configuration this project has measured: 8.4 percent over the
committed engine control, 2.4 percent over the fairest engine
configuration, with 2,191 wrong of 10,000 against the committed
control's 2,990. The custom kernels beat the separate packed cell by
11.7 percent end to end, above the predicted 5 to 10: the profiled
window's kernel time fell from 4.113 to 3.679 s, the separate silu
kernel is gone, and silu_mul_quant runs 301 ms where the pair it
replaces cost about 722. The probe microbenchmark had already
exceeded its gates (custom silu+quant 464 us per call against 1,078
for the pair and 1,370 for vLLM's fused kernel; custom norm+quant
160 against 329). The bf16-KV engine confirmed the accuracy
mechanism exactly - 2,225 wrong against the packed pipeline's 2,228,
so the fp8 KV cache costs the engine 765 answers on this workload -
and broke the speed prediction in the good direction: 5.9 percent
faster than the fp8-KV engine instead of the predicted within-1,
because the fp8 KV path also pays conversion work on every cache
write and attention read. For a single filter, where no KV stays
resident, fp8 KV was the wrong setting in the committed control all
along; bf16 KV is the correctly configured engine baseline for this
comparison, and the custom packed cell beats it too.

Round 4: the query/key norm and rotate region, the pipeline's worst
kernel by achieved bandwidth. The round 3 profile shows the per-head
norm running one tiny work group per 128-number head vector - about
15 million launches per profiled window at 23 GB/s - for 338 ms,
plus the separate rotate kernel and two layout copies around it. The
region moves only about 8 GB per window. One Triton kernel per token
now does all of it: loads the query and key parts of the qkv result,
normalizes each head against its norm weight, applies the rotation
from the position table, and writes query and key contiguously in
the layout attention wants. The value part was always a free view
and stays one.

Round 4 prediction, stated before the run:
  - probe microbenchmark: the fused kernel beats the four-step
    sequence it replaces (two per-head norms, the rotate, two
    copies) by at least 5x per call; given the 23 GB/s baseline,
    10 to 30x is plausible.
  - end to end: 7 to 11 percent over the round-3 custom cell,
    111,000 to 115,000 tokens per second, further above the bf16-KV
    engine's roughly 101,500.
  - answers stay within tens of the round-3 custom cell's 2,191
    wrong; the norm and rotation move from vLLM's kernels to fp32
    arithmetic in ours, so exact agreement is not expected.
  Falsifier: if the fused kernel wins its microbenchmark but the
  end-to-end gain lands under 5 percent, the region's cost was
  partly hidden elsewhere (the profiled classes misattribute it) and
  the profiled repetition says where.

Round 4 result: above the predicted band, and the probe explained
why before the timed run. The microbenchmark showed the four-step
path really costs 1,257 us per call - about double what the norm
class displayed, the rest hiding in the rotate and copy steps under
other classes - and the fused kernel runs it in 184.5 us, 6.8x.
Means: engine bf16-KV 100,904, packed custom 102,536, packed custom
with the qk kernel 119,629 tokens per second - 16.7 percent over
the round-3 cell against the predicted 7 to 11, and 18.6 percent
over the best engine configuration. In the profile the norm class
fell from 9.1 to 3.1 percent (the remainder is the fused kernel
itself plus the layer-0 and final norms) and the other class fell
from 5.2 percent to almost nothing; window kernel time went from
3.72 to 3.23 s. Wrong answers 2,213 of 10,000, within the predicted
tens of round 3's 2,191 and still better than every engine cell.
The pipeline is now 68 percent matrix multiplies and 8 percent
attention; what remains fusable is about 420 ms of quant work per
window and the epilogue idea inside the gate_up multiply.

Round 5: the epilogue. A Triton block-scaled fp8 multiply for the
gate_up projection whose closing phase does the silu and the group
quant while the finished tile is still in registers, so the 985 MB
16-bit intermediate is never written or read. Each program owns one
tile of the silu output and loads both its gate and up parent
columns itself, which also reads the input tile once instead of
twice. The risk is split in two on purpose: the probe benches (a)
DeepGEMM plus our silu+quant kernel (the current champion), (b) our
Triton multiply with a plain epilogue plus the same silu+quant
kernel, and (c) our Triton multiply with the fused epilogue. c
against b prices the epilogue at equal multiply quality; c against
a decides whether the pipeline switches. A DeepGEMM hardware-counter
measurement (modal_profiling.py::ncudeepgemm) sizes how much of the
gap is even available.

Round 5 prediction, stated before the run, with honest uncertainty:
  - c beats b by 15 to 25 percent per call (the epilogue removes the
    985 MB write and read and the second kernel's launch; the silu
    kernel it absorbs cost 464 us against the multiply's roughly
    2,000).
  - whether c beats a is a coin flip that turns on our multiply
    alone: DeepGEMM runs the gate_up shape at about 2,042 us and
    Triton block-scaled multiplies of this shape typically land
    between parity and 1.7x slower. If our multiply is within about
    15 percent of DeepGEMM, c wins outright and the timed run
    launches; if not, the epilogue verdict still stands from c
    against b, the pipeline keeps DeepGEMM plus the separate kernel,
    and the banked conclusion is the measured price of the missing
    epilogue interface on DeepGEMM's side.
  - end to end if c wins: 3 to 6 percent over round 4 (the separate
    silu kernel's 318 ms per window vanishes and the multiply
    absorbs part of it back), 123,000 to 127,000 tokens per second.
  - numerics: dequantized agreement with the a path at one fp8
    rounding step; answers within tens of round 4's 2,213.

Round 5 result: the negative branch, with the numbers to close it.
Our Triton multiply alone ran the gate_up shape in 4,233 us against
DeepGEMM's 1,877 - 2.26x off, far outside the within-15-percent
condition - so the fused version (4,562) lost to the champion pair
(2,479) and the pipeline keeps DeepGEMM with the separate silu
kernel. The epilogue also underperformed its band at equal multiply
quality: fused against plain-plus-kernel saved 4.8 percent, not 15
to 25, because the silu and quant work in the closing phase slowed
our multiply by 329 us - a multiply this far from optimal is not
limited by its output writes, so deleting them buys little.
Numerics were correct within one fp8 step at the top of the range.
No timed run launched, per the committed rule. The hardware
counters (results/engine/ncu_deepgemm_details.txt) give the real
kernels' utilization for the first time: 84 to 91 percent of the
compute ceiling in isolation (qkv 84.3, o 86.4, down 90.9; the
gate_up render was undersampled), against about 70 percent
effective in-pipeline by flop arithmetic - the gap is tail waves,
not slow kernels, and the remaining multiply headroom is thoroughly
defended. Round 4 stands as this branch's result: 119,629 tokens
per second, 18.6 percent over the best engine configuration, more
accurate than every engine cell, with no engine and no KV.

Round 6: the missing rung of the ablation ladder. The KV-cache
write cannot be removed alone: in the engine's attention the read
goes through the very pages the write just filled (the v1
flash_attn backend runs reshape_and_cache_flash and then reads via
block_table), so "engine kernels minus only the write" is not
constructible. The constructible clean rung keeps KV entirely and
removes only the engine: vLLM's model compiled exactly as the
engine compiles it, a real paged fp8 cache, block tables and
attention metadata built by hand for each packed chunk. Five cells
in one container then form the whole ladder: engine fp8, engine
bf16, engine kernels without the engine, packed stock parts with no
KV, packed with our three kernels.

Round 6 prediction, stated before the run:
  - packed_enginekernels lands within 2 percent of the fp8 engine
    either side. Pure engine removal has measured near zero three
    times, and this rung shares every kernel with the engine,
    including the cache write and paged read.
  - its wrong count lands within 10 of the engine's 2,990, with
    disagreements against the engine under 50: same kernels, same
    fp8 cache rounding, only the chunk boundaries differ.
  - the other rungs repeat their banked values within container
    noise.
  Falsifier: if packed_enginekernels sits 3 or more percent from
  the fp8 engine, engine software was never actually free, and
  every earlier "the engine is not the cost" conclusion gets
  requalified by what the profiled repetition shows.

Round 6 result: the falsifier fired, and the diagnosis changed what
the rung is. The cache binding did not take - the attention layer
expects a list of cache tensors indexed by virtual engine and the
runner assigned a raw tensor - so the backend took its no-cache
path: no reshape_and_cache kernel anywhere in the cell's profile,
attention direct over the current bf16 values. As run, the rung is
therefore "the engine's compiled kernels, no engine, no KV write" -
the ablation originally asked for, which the backend's own no-cache
fallback makes constructible after all, correcting this round's
opening claim. The speed ladder, one container: engine fp8 96,946;
engine bf16 102,820; engine kernels without engine or cache
100,308; packed stock parts 93,579; packed with our kernels
121,045, the branch's new best. Engine software again measures
near free, and the 5.9 percent between the engine cells prices the
fp8 cache path. This rung's accuracy column (1,064 wrong of 10,000,
better than every other cell ever measured) is NOT validated:
better than everything is the signature of documents leaking
information to each other through the hand-built metadata, and
until a one-document-per-chunk check rules that out the number
should not be cited. The other four rungs reproduced their banked
values within container noise.

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


def _load_vllm_model(compiled=False, kv_cache_dtype="auto"):
    """The checkpoint as vLLM's processed module: merged qkv and
    gate_up, FP8 weights and block scales laid out for DeepGEMM. No
    engine and no KV pool - just the weights and layer modules.

    With compiled=True the model is built exactly as the engine
    builds it: torch.compile on, the O2 pass defaults, so its norm,
    silu, and rotate kernels are the engine's own compiled ones.
    Graph capture stays off (measured worth nothing at these chunk
    sizes)."""
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model
    from vllm.utils.network_utils import get_open_port

    # enforce_eager keeps compilation out of the eager loader; custom
    # ops then default on, so module calls run vLLM's CUDA kernels.
    args = dict(model=MODEL, dtype="bfloat16",
                kv_cache_dtype=kv_cache_dtype)
    if compiled:
        # mirror the engine cell so the compiled token ranges cover
        # our chunks; graphs off (measured worth nothing here)
        args.update(max_model_len=4608, max_num_seqs=4096,
                    max_num_batched_tokens=BEST_BATCH_TOKENS,
                    compilation_config={"cudagraph_mode": 0})
    else:
        args["enforce_eager"] = True
    config = EngineArgs(**args).create_engine_config()
    with set_current_vllm_config(config):
        # the model classes read the parallel groups even on one GPU,
        # and the group setup itself reads the current config
        import torch.distributed as dist
        if not dist.is_initialized():
            init_distributed_environment(
                world_size=1, rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
                local_rank=0, backend="nccl")
            ensure_model_parallel_initialized(1, 1)
        model = get_model(vllm_config=config)
    torch.cuda.synchronize()
    return model, config


class EngineKernelRunner:
    """The engine's exact kernels with the engine removed. vLLM's
    compiled model, a real paged KV cache, block tables, and
    attention metadata built by hand per chunk - the same write and
    paged read the engine runs, driven from the packed loop. The
    KV-cache write cannot be removed alone: the attention read goes
    through the pages the write just filled, so this rung keeps KV
    entirely and isolates pure engine removal."""

    BLOCK = 16

    def __init__(self, tokens_cap, max_docs, kv_cache_dtype="fp8"):
        import torch
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionBackend,
        )

        self.torch = torch
        self.model, self.config = _load_vllm_model(
            compiled=True, kv_cache_dtype=kv_cache_dtype)
        ctx = self.config.compilation_config.static_forward_context
        self.layer_names = [name for name, layer in ctx.items()
                            if hasattr(layer, "kv_cache")]
        num_blocks = tokens_cap // self.BLOCK + max_docs + 8
        shape = FlashAttentionBackend.get_kv_cache_shape(
            num_blocks, self.BLOCK, 8, 128,
            cache_dtype_str=kv_cache_dtype)
        cache_dtype = (torch.uint8 if kv_cache_dtype.startswith("fp8")
                       else torch.bfloat16)
        for name in self.layer_names:
            ctx[name].kv_cache = torch.zeros(
                shape, dtype=cache_dtype, device="cuda")

    def metadata(self, packed):
        import torch
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionMetadata,
        )

        input_ids, positions, _final, cu_seqlens, max_seqlen = packed
        n_tokens = input_ids.shape[0]
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
        n_seqs = lens.shape[0]
        blocks_per_seq = (lens.to(torch.int64) + self.BLOCK - 1) // self.BLOCK
        max_bps = int(blocks_per_seq.max())
        starts = torch.zeros(n_seqs, dtype=torch.int64, device="cuda")
        starts[1:] = torch.cumsum(blocks_per_seq, 0)[:-1]
        block_table = (starts[:, None]
                       + torch.arange(max_bps, device="cuda")[None, :])
        block_table = block_table.to(torch.int32)
        seq_of_token = torch.repeat_interleave(
            torch.arange(n_seqs, device="cuda"), lens.to(torch.int64))
        token_block = block_table[
            seq_of_token, positions // self.BLOCK].to(torch.int64)
        slot_mapping = token_block * self.BLOCK + positions % self.BLOCK
        md = FlashAttentionMetadata(
            num_actual_tokens=n_tokens,
            max_query_len=max_seqlen,
            query_start_loc=cu_seqlens,
            max_seq_len=max_seqlen,
            seq_lens=lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
        )
        return {name: md for name in self.layer_names}

    def run(self, packed):
        from vllm.forward_context import set_forward_context

        input_ids, positions, final_indices, _cu, _max = packed
        md = self.metadata(packed)
        with self.torch.inference_mode():
            with set_forward_context(md, self.config,
                                     num_tokens=input_ids.shape[0]):
                hidden = self.model(input_ids=input_ids,
                                    positions=positions)
        return hidden.index_select(0, final_indices)


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
        self.fa_version = 2
        self.fuse_qk = False

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

    def _triton_kernels(self):
        """Our own fused kernels, built lazily so the module imports
        on machines without triton. Scales get the same power-of-two
        rounding the separate quant path uses, so DeepGEMM's internal
        cast is lossless for both."""
        if hasattr(self, "_kernels"):
            return self._kernels
        import triton
        import triton.language as tl

        @triton.jit
        def silu_mul_quant(gu_ptr, q_ptr, s_ptr, stride_gu, stride_q,
                           s_stride_g, s_stride_t,
                           HALF: tl.constexpr, GROUP: tl.constexpr,
                           GPB: tl.constexpr, UE8M0: tl.constexpr):
            t = tl.program_id(0)
            block = tl.program_id(1)
            offs = block * GROUP * GPB + tl.arange(0, GROUP * GPB)
            gate = tl.load(gu_ptr + t * stride_gu + offs).to(tl.float32)
            up = tl.load(gu_ptr + t * stride_gu + HALF + offs).to(tl.float32)
            y = gate * tl.sigmoid(gate) * up
            y2 = tl.reshape(y, (GPB, GROUP))
            amax = tl.max(tl.abs(y2), axis=1)
            scale = tl.maximum(amax, 1e-10) / 448.0
            if UE8M0:
                scale = tl.math.exp2(tl.ceil(tl.math.log2(scale)))
            q = y2 / scale[:, None]
            q = tl.minimum(tl.maximum(q, -448.0), 448.0)
            tl.store(q_ptr + t * stride_q + offs,
                     tl.reshape(q, (GROUP * GPB,)).to(q_ptr.dtype.element_ty))
            g_idx = block * GPB + tl.arange(0, GPB)
            tl.store(s_ptr + g_idx * s_stride_g + t * s_stride_t, scale)

        @triton.jit
        def add_rms_norm_quant(x_ptr, res_ptr, w_ptr, q_ptr, s_ptr,
                               stride_x, stride_res, stride_q,
                               s_stride_g, s_stride_t, eps,
                               H: tl.constexpr, BLOCK: tl.constexpr,
                               GROUP: tl.constexpr, NG: tl.constexpr,
                               UE8M0: tl.constexpr):
            t = tl.program_id(0)
            offs = tl.arange(0, BLOCK)
            mask = offs < H
            x = tl.load(x_ptr + t * stride_x + offs, mask=mask,
                        other=0.0).to(tl.float32)
            r = tl.load(res_ptr + t * stride_res + offs, mask=mask,
                        other=0.0).to(tl.float32)
            x = x + r
            tl.store(res_ptr + t * stride_res + offs,
                     x.to(res_ptr.dtype.element_ty), mask=mask)
            ms = tl.sum(x * x, axis=0) / H
            rstd = 1.0 / tl.sqrt(ms + eps)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            y = x * rstd * w
            y2 = tl.reshape(y, (BLOCK // GROUP, GROUP))
            amax = tl.max(tl.abs(y2), axis=1)
            scale = tl.maximum(amax, 1e-10) / 448.0
            if UE8M0:
                scale = tl.math.exp2(tl.ceil(tl.math.log2(scale)))
            q = y2 / scale[:, None]
            q = tl.minimum(tl.maximum(q, -448.0), 448.0)
            tl.store(q_ptr + t * stride_q + offs,
                     tl.reshape(q, (BLOCK,)).to(q_ptr.dtype.element_ty),
                     mask=mask)
            g_idx = tl.arange(0, BLOCK // GROUP)
            tl.store(s_ptr + g_idx * s_stride_g + t * s_stride_t, scale,
                     mask=g_idx < NG)

        @triton.jit
        def qk_norm_rope(qkv_ptr, q_out_ptr, k_out_ptr, cs_ptr, pos_ptr,
                         qw_ptr, kw_ptr, stride_qkv, stride_qo, stride_ko,
                         eps, QH: tl.constexpr, KH: tl.constexpr,
                         HD: tl.constexpr, HALF: tl.constexpr):
            t = tl.program_id(0)
            pos = tl.load(pos_ptr + t)
            half_offs = tl.arange(0, HALF)
            cos = tl.load(cs_ptr + pos * HD + half_offs).to(tl.float32)
            sin = tl.load(cs_ptr + pos * HD + HALF + half_offs).to(tl.float32)
            qw_a = tl.load(qw_ptr + half_offs).to(tl.float32)
            qw_b = tl.load(qw_ptr + HALF + half_offs).to(tl.float32)
            kw_a = tl.load(kw_ptr + half_offs).to(tl.float32)
            kw_b = tl.load(kw_ptr + HALF + half_offs).to(tl.float32)

            # query heads: each head split into its two rotation halves
            q_heads = tl.arange(0, QH)
            qa_offs = q_heads[:, None] * HD + half_offs[None, :]
            qb_offs = qa_offs + HALF
            qa = tl.load(qkv_ptr + t * stride_qkv + qa_offs).to(tl.float32)
            qb = tl.load(qkv_ptr + t * stride_qkv + qb_offs).to(tl.float32)
            ms = (tl.sum(qa * qa, axis=1) + tl.sum(qb * qb, axis=1)) / HD
            rstd = 1.0 / tl.sqrt(ms + eps)
            qa = qa * rstd[:, None] * qw_a[None, :]
            qb = qb * rstd[:, None] * qw_b[None, :]
            out_a = qa * cos[None, :] - qb * sin[None, :]
            out_b = qb * cos[None, :] + qa * sin[None, :]
            tl.store(q_out_ptr + t * stride_qo + qa_offs,
                     out_a.to(q_out_ptr.dtype.element_ty))
            tl.store(q_out_ptr + t * stride_qo + qb_offs,
                     out_b.to(q_out_ptr.dtype.element_ty))

            # key heads, offset past the query width in the qkv row
            k_heads = tl.arange(0, KH)
            ka_offs = k_heads[:, None] * HD + half_offs[None, :]
            kb_offs = ka_offs + HALF
            base = qkv_ptr + t * stride_qkv + QH * HD
            ka = tl.load(base + ka_offs).to(tl.float32)
            kb = tl.load(base + kb_offs).to(tl.float32)
            ms = (tl.sum(ka * ka, axis=1) + tl.sum(kb * kb, axis=1)) / HD
            rstd = 1.0 / tl.sqrt(ms + eps)
            ka = ka * rstd[:, None] * kw_a[None, :]
            kb = kb * rstd[:, None] * kw_b[None, :]
            out_a = ka * cos[None, :] - kb * sin[None, :]
            out_b = kb * cos[None, :] + ka * sin[None, :]
            tl.store(k_out_ptr + t * stride_ko + ka_offs,
                     out_a.to(k_out_ptr.dtype.element_ty))
            tl.store(k_out_ptr + t * stride_ko + kb_offs,
                     out_b.to(k_out_ptr.dtype.element_ty))

        self._kernels = {"silu": silu_mul_quant,
                         "norm": add_rms_norm_quant,
                         "qk": qk_norm_rope}
        return self._kernels

    def forward_chunk(self, packed, fused):
        (input_ids, positions, final_indices, cu_seqlens,
         max_seqlen) = packed
        hidden, residual = self.forward_tokens(
            input_ids, positions, cu_seqlens, max_seqlen, fused)
        return self.select_and_norm(hidden, residual, final_indices)

    def select_and_norm(self, hidden, residual, final_indices):
        last_hidden = hidden.index_select(0, final_indices)
        last_residual = residual.index_select(0, final_indices)
        normed, _ = self.fused_add_rms_norm(
            last_hidden, last_residual, self.final_norm)
        return normed

    def forward_tokens(self, input_ids, positions, cu_seqlens,
                       max_seqlen, fused):
        """The token-level forward: everything whose shapes depend
        only on the token count, so it can be captured as a CUDA
        graph at a fixed size."""
        n_tokens = input_ids.shape[0]
        hidden = self.embed(input_ids)
        residual = None
        for layer in self.layers:
            attn = layer.self_attn
            if residual is None:
                # first layer: no residual to fuse, run unfused
                residual = hidden
                q_in, q_scale = self.quant(
                    self.rms_norm(hidden, layer.input_layernorm))
            elif fused == "custom":
                q_in, q_scale = self.custom_norm_quant(
                    hidden, layer.input_layernorm, residual)
            else:
                normed, residual = self.fused_add_rms_norm(
                    hidden, residual, layer.input_layernorm)
                q_in, q_scale = self.quant(normed)
            qkv = self.gemm(q_in, q_scale, attn.qkv_proj)

            q_width = self.num_q_heads * self.head_dim
            kv_width = self.num_kv_heads * self.head_dim
            if self.fuse_qk:
                q, k = self.custom_qk_norm_rope(qkv, positions, attn)
                v = qkv.split([q_width, kv_width, kv_width], dim=-1)[2]
            else:
                q, k, v = qkv.split([q_width, kv_width, kv_width], dim=-1)
                q = self.rms_norm(
                    q.reshape(-1, self.head_dim).contiguous(), attn.q_norm
                ).reshape(n_tokens, q_width)
                k = self.rms_norm(
                    k.reshape(-1, self.head_dim).contiguous(), attn.k_norm
                ).reshape(n_tokens, kv_width)
                q, k = self.rotary(positions, q, k)
            attn_out = self.attention(q, k, v, cu_seqlens, max_seqlen)

            # the attention output quant has no fusion partner in
            # either variant
            o_in, o_scale = self.quant(attn_out)
            hidden = self.gemm(o_in, o_scale, attn.o_proj)

            if fused == "custom":
                g_in, g_scale = self.custom_norm_quant(
                    hidden, layer.post_attention_layernorm, residual)
            else:
                normed, residual = self.fused_add_rms_norm(
                    hidden, residual, layer.post_attention_layernorm)
                g_in, g_scale = self.quant(normed)
            gate_up = self.gemm(g_in, g_scale, layer.mlp.gate_up_proj)
            if fused == "custom":
                d_in, d_scale = self.custom_silu_quant(gate_up)
            else:
                d_in, d_scale = self.quant(self.silu_and_mul(gate_up))
            hidden = self.gemm(d_in, d_scale, layer.mlp.down_proj)

        return hidden, residual


@app.function(image=forward_image, gpu="H100!", timeout=3600, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache})
def probe(n_docs: int = 8) -> str:
    """Check the kernels and the loader before the timed run: op
    schemas, module attribute names and shapes, and one packed chunk
    through both variants with finite outputs and matching answers."""
    import json

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from workload import build_corpus, yes_no_ids

    # the model load also loads vLLM's _C extension; op checks before
    # it report every op missing
    model, _model_config = _load_vllm_model()
    result = {"ops": {}}
    for op_name in ("rms_norm_per_block_quant",
                    "silu_and_mul_per_block_quant",
                    "silu_and_mul", "fused_add_rms_norm", "rms_norm"):
        op = getattr(torch.ops._C, op_name, None)
        result["ops"][op_name] = (
            str(op.default._schema) if op is not None else "MISSING"
        )
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
    body_ids, question_ids, flags = build_corpus(tokenizer, n_docs)
    prompts = [body_ids[i] + question_ids[0] for i in range(n_docs)]
    expected = [int(flags[i][0]) for i in range(n_docs)]
    yes_ids, no_ids = yes_no_ids(tokenizer)
    allowed_ids = sorted(yes_ids | no_ids)
    selected_weights = model.lm_head.weight.index_select(
        0, torch.tensor(allowed_ids, device="cuda")).to(torch.bfloat16)
    yes_columns = torch.tensor(
        [i for i, t in enumerate(allowed_ids) if t in yes_ids],
        device="cuda")
    no_columns = torch.tensor(
        [i for i, t in enumerate(allowed_ids) if t in no_ids],
        device="cuda")

    def answers(normed):
        scores = F.linear(normed, selected_weights)
        yes_scores = scores.index_select(1, yes_columns).amax(dim=1)
        no_scores = scores.index_select(1, no_columns).amax(dim=1)
        return (yes_scores > no_scores).int().cpu().tolist()

    pipeline.fa_version = 3
    packed = PackedPipeline.pack(prompts)
    with torch.inference_mode():
        unfused = pipeline.forward_chunk(packed, fused=False)
        custom = pipeline.forward_chunk(packed, fused="custom")
    unfused_answers = answers(unfused)
    custom_answers = answers(custom)
    result["chunk"] = {
        "tokens": int(packed[0].shape[0]),
        "docs": n_docs,
        "fa_version": pipeline.fa_version,
        "unfused_finite": bool(torch.isfinite(unfused).all().item()),
        "unfused_wrong": sum(
            a != b for a, b in zip(unfused_answers, expected)),
        "custom_finite": bool(torch.isfinite(custom).all().item()),
        "custom_wrong": sum(
            a != b for a, b in zip(custom_answers, expected)),
        "custom_vs_unfused_disagreements": sum(
            a != b for a, b in zip(custom_answers, unfused_answers)),
    }

    # dequantized agreement of the custom kernels against the
    # separate path they replace, on fresh tensors
    with torch.inference_mode():
        check = torch.randn(256, 19456, dtype=torch.bfloat16,
                            device="cuda")
        q_ref, s_ref = pipeline.quant(pipeline.silu_and_mul(check))
        q_new, s_new = pipeline.custom_silu_quant(check)
        deq_ref = q_ref.float().view(256, -1, 128) * s_ref.float()[:, :, None]
        deq_new = q_new.float().view(256, -1, 128) * s_new.float()[:, :, None]
        silu_deq_diff = (deq_ref - deq_new).abs().max().item()
        norm_module = model.model.layers[1].post_attention_layernorm
        h_ref = torch.randn(256, 2560, dtype=torch.bfloat16, device="cuda")
        r_ref = torch.randn(256, 2560, dtype=torch.bfloat16, device="cuda")
        h_new, r_new = h_ref.clone(), r_ref.clone()
        normed, _ = pipeline.fused_add_rms_norm(h_ref, r_ref, norm_module)
        q_ref, s_ref = pipeline.quant(normed)
        q_new, s_new = pipeline.custom_norm_quant(h_new, norm_module, r_new)
        deq_ref = q_ref.float().view(256, -1, 128) * s_ref.float()[:, :, None]
        deq_new = q_new.float().view(256, -1, 128) * s_new.float()[:, :, None]
        norm_deq_diff = (deq_ref - deq_new).abs().max().item()
        residual_diff = (r_ref.float() - r_new.float()).abs().max().item()
    result["custom_kernel_check"] = {
        "silu_dequant_max_diff": round(silu_deq_diff, 5),
        "norm_dequant_max_diff": round(norm_deq_diff, 5),
        "norm_residual_max_diff": round(residual_diff, 5),
    }

    # the fused qk kernel against the four-step path it replaces
    attn1 = model.model.layers[1].self_attn
    cache = pipeline.rotary.cos_sin_cache
    result["qk_check"] = {"cos_sin_cache": [list(cache.shape),
                                            str(cache.dtype)]}
    with torch.inference_mode():
        qkv_t = torch.randn(256, 6144, dtype=torch.bfloat16,
                            device="cuda")
        pos_t = torch.arange(256, device="cuda")

        def qk_reference(qkv_in, pos):
            q, k, _v = qkv_in.split([4096, 1024, 1024], dim=-1)
            q = pipeline.rms_norm(
                q.reshape(-1, 128).contiguous(), attn1.q_norm
            ).reshape(-1, 4096)
            k = pipeline.rms_norm(
                k.reshape(-1, 128).contiguous(), attn1.k_norm
            ).reshape(-1, 1024)
            return pipeline.rotary(pos, q, k)

        q_ref, k_ref = qk_reference(qkv_t.clone(), pos_t)
        q_new, k_new = pipeline.custom_qk_norm_rope(qkv_t, pos_t, attn1)
        result["qk_check"]["q_max_diff"] = round(
            (q_ref.float() - q_new.float()).abs().max().item(), 5)
        result["qk_check"]["k_max_diff"] = round(
            (k_ref.float() - k_new.float()).abs().max().item(), 5)

        pipeline.fuse_qk = True
        custom_qk = pipeline.forward_chunk(packed, fused="custom")
        pipeline.fuse_qk = False
    custom_qk_answers = answers(custom_qk)
    result["qk_check"]["chunk_wrong"] = sum(
        a != b for a, b in zip(custom_qk_answers, expected))
    result["qk_check"]["chunk_vs_custom_disagreements"] = sum(
        a != b for a, b in zip(custom_qk_answers, custom_answers))

    # kernel-level timing at the real chunk size: the go or no-go
    # signal for the full run
    def bench(fn, iters=30, warmups=5):
        with torch.inference_mode():
            for _ in range(warmups):
                fn()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
        return round(start.elapsed_time(end) * 1000 / iters, 1)

    big_gate_up = torch.randn(BEST_BATCH_TOKENS, 19456,
                              dtype=torch.bfloat16, device="cuda")
    big_hidden = torch.randn(BEST_BATCH_TOKENS, 2560,
                             dtype=torch.bfloat16, device="cuda")
    big_residual = torch.randn(BEST_BATCH_TOKENS, 2560,
                               dtype=torch.bfloat16, device="cuda")
    result["microbench_us_per_call"] = {
        "silu_separate_pair": bench(
            lambda: pipeline.quant(pipeline.silu_and_mul(big_gate_up))),
        "silu_vllm_fused": bench(
            lambda: pipeline.fused_silu_quant(big_gate_up)),
        "silu_custom": bench(
            lambda: pipeline.custom_silu_quant(big_gate_up)),
        "norm_separate_pair": bench(
            lambda: pipeline.quant(pipeline.fused_add_rms_norm(
                big_hidden, big_residual, norm_module)[0])),
        "norm_vllm_fused": bench(
            lambda: pipeline.fused_norm_quant(
                big_hidden, norm_module, big_residual)),
        "norm_custom": bench(
            lambda: pipeline.custom_norm_quant(
                big_hidden, norm_module, big_residual)),
    }
    big_qkv = torch.randn(BEST_BATCH_TOKENS, 6144, dtype=torch.bfloat16,
                          device="cuda")
    big_pos = torch.arange(BEST_BATCH_TOKENS, device="cuda") % 3072
    result["microbench_us_per_call"]["qk_four_step"] = bench(
        lambda: qk_reference(big_qkv, big_pos))
    result["microbench_us_per_call"]["qk_custom"] = bench(
        lambda: pipeline.custom_qk_norm_rope(big_qkv, big_pos, attn1))
    del big_qkv

    # round 6: the engine's exact kernels driven without the engine
    import time as _time
    runner = EngineKernelRunner(BEST_BATCH_TOKENS, 64,
                                kv_cache_dtype="fp8")
    with torch.inference_mode():
        t0 = _time.perf_counter()
        first = runner.run(packed)
        torch.cuda.synchronize()
        first_wall = _time.perf_counter() - t0
        t0 = _time.perf_counter()
        second = runner.run(packed)
        torch.cuda.synchronize()
        second_wall = _time.perf_counter() - t0
    ek_answers = answers(second)
    result["enginekernels_check"] = {
        "first_call_s": round(first_wall, 2),
        "second_call_s": round(second_wall, 3),
        "finite": bool(torch.isfinite(second).all().item()),
        "wrong": sum(a != b for a, b in zip(ek_answers, expected)),
        "vs_unfused_disagreements": sum(
            a != b for a, b in zip(ek_answers, unfused_answers)),
    }
    del runner
    gc = __import__("gc")
    gc.collect()
    torch.cuda.empty_cache()

    print(json.dumps(result, indent=2), flush=True)
    return json.dumps(result)


@app.function(image=forward_image, gpu="H100!", timeout=10800,
              memory=65536,
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
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        min_tokens=1,
        allowed_token_ids=allowed_ids,
    )
    vllm_prompts = [{"prompt_token_ids": prompt} for prompt in prompts]
    control_predictions = None
    # the second engine cell answers whether the packed pipeline's
    # accuracy edge comes from the engine's fp8 KV cache: same boot,
    # KV held in bf16 instead
    # the presentation ladder: the engine, then engine kernels without
    # the engine, then our kernels. Add ("vllm_bf16kv", "auto") here to
    # re-run the best engine setting; it and the stock-parts rung are
    # banked in the round 6 result.
    for engine_name, kv_dtype in (("vllm", "fp8"),):
        llm = LLM(
            model=MODEL,
            kv_cache_dtype=kv_dtype,
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
                "method": engine_name,
                "rep": rep,
                "wall": round(wall, 4),
                "tokens_per_second": round(total_prompt_tokens / wall, 1),
                "wrong": wrong,
                "kv_cache_dtype": kv_dtype,
            }
            if control_predictions is not None:
                row["disagrees_with_vllm"] = sum(
                    a != b for a, b in zip(predicted, control_predictions)
                )
            report["runs"].append(row)
            print(f"[single filter] {row}", flush=True)
        if control_predictions is None:
            control_predictions = list(predicted)

        del outputs, llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)

    model, _model_config = _load_vllm_model()
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
    def run_packed(selected, fused, runner=None):
        predictions = []
        for packed in selected:
            if runner is None:
                normed = pipeline.forward_chunk(packed, fused=fused)
            else:
                normed = runner.run(packed)
            scores = F.linear(normed, selected_weights)
            yes_scores = scores.index_select(1, yes_columns).amax(dim=1)
            no_scores = scores.index_select(1, no_columns).amax(dim=1)
            predictions.extend((yes_scores > no_scores).int().cpu().tolist())
        return predictions

    pipeline.fa_version = 3
    max_docs = max(len(batch) for batch in batches) + 1
    cells = (
        ("packed_enginekernels", None, "engine", False),
        ("packed_custom_qk", "custom", None, True),
    )
    for name, fused, runner_kind, fuse_qk in cells:
        pipeline.fuse_qk = fuse_qk
        if runner_kind == "engine":
            runner = EngineKernelRunner(batch_tokens, max_docs,
                                        kv_cache_dtype="fp8")
        else:
            runner = None
        # two warmup chunks: DeepGEMM compiles its kernels on first use
        run_packed(packed_batches[:2], fused, runner)
        torch.cuda.synchronize()
        for rep in range(reps):
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            predicted = run_packed(packed_batches, fused, runner)
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
            wrong = sum(a != b for a, b in zip(predicted, expected))
            row = {
                "method": name,
                "rep": rep,
                "wall": round(wall, 4),
                "tokens_per_second": round(total_prompt_tokens / wall, 1),
                "wrong": wrong,
                "disagrees_with_engine": sum(
                    a != b for a, b in zip(predicted, control_predictions)
                ),
                "kernels": ("engine-compiled, KV kept"
                            if runner_kind == "engine" else
                            f"fa{pipeline.fa_version}, no KV"),
                "fused_qk_norm_rope": fuse_qk,
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
            run_packed(profiled, fused, runner)
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
        del runner
        gc.collect()
        torch.cuda.empty_cache()

    rates = {}
    for method in ("vllm", "packed_enginekernels", "packed_custom_qk"):
        values = [row["tokens_per_second"] for row in report["runs"]
                  if row["method"] == method]
        rates[method] = round(sum(values) / len(values), 1)
    report["mean_tokens_per_second"] = rates
    report["relative_to_engine"] = {
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
