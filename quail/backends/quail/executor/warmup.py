"""Compile kernel shapes once and load cached kernels in each GPU process."""

import time

from quail.backends.quail.executor.model_loading import full_output_head
from quail.backends.quail.executor.stages import Stage, filter_stages, run_stages
from quail.progress import Progress, logger, quiet

# ------------------------------------------------------------- warmup
#
# compile_kernels JIT-compiles (nvcc/Triton) each kernel configuration
# once per (software stack, GPU, model, budget) into the libraries'
# kernel directories; a marker file records that the pass ran.
# touch_kernels loads the cached binaries once per GPU process.
#
# Kernels key on token counts and model constants, never on token
# values, so both passes run on synthetic ids.

# Bump when either pass covers a different set of shapes. A bumped
# version invalidates every marker, so the next boot re-runs the
# compile pass.
WARMUP_VERSION = 10

# Triton compiles one copy of a kernel for each class of its integer
# arguments: equal to 1, a multiple of 16, or neither. Quail's row
# dependent strides come from a chunk's row count n and, under tree
# attention, the count m of its rows that read cached KV (m <= n).
# These (n, m) pairs cover every class pair.
ROW_CLASSES = ((1, 1), (32, 1), (32, 16), (32, 17),
               (33, 1), (33, 16), (33, 17))
# tokens of each cached document the reading rows attend to
ROW_CLASS_CACHED = 64


def _warm_inputs(budget, question_tokens=16):
    """Generate synthetic document and question tokens for kernel warmup.

    Cycled to any length the passes need. Fixed small ids; only the
    counts matter to the kernels.
    """
    document = [10 + (i % 500) for i in range(512)]
    question = [10 + (i % 500) for i in range(question_tokens)]
    q_max = len(question)
    warm_documents, used = [], 0
    while used + len(document) + q_max <= budget:
        warm_documents.append(document)
        used += len(document) + q_max
    return warm_documents or [document[:max(8, budget - q_max)]], question, document


def _warm_convolutions(torch, pipeline, budget):
    """Compile and load convolution kernels for every private state capacity."""
    from quail.backends.quail.executor.gdn import GdnPhase, causal_conv_kernel

    # Each sequence uses at least one token of the chunk budget.
    largest = 1 << (budget - 1).bit_length()
    for weight, bias, activate in pipeline.convolutions():
        channels, width = weight.shape
        history = weight.new_empty((largest, width - 1, channels))
        history[:1].zero_()
        x = weight.new_zeros((2, channels))
        phase = GdnPhase(
            rows=None, lengths=(2,), cu_seqlens=torch.tensor(
                [0, 2], dtype=torch.int32, device=weight.device))
        for power in range(largest.bit_length()):
            causal_conv_kernel(x, weight, phase, history[:1 << power],
                               bias, activate)


def _forward_warm(torch, arena, pipeline, async_ans, budget, *,
                  join_chunk):
    """Run real forward passes over every attention path.

    Runs each attention path the model has with arena writes, then the
    unpaged causal fast path. join_chunk adds a join chunk and a
    classification chunk.
    """
    _warm_convolutions(torch, pipeline, budget)
    # a readout of several trailing rows needs a question that long
    trailing = 1 if async_ans is None else async_ans.trailing_rows
    warm_documents, question, document = _warm_inputs(budget, max(16, trailing))
    q_max = len(question)
    modes = ("unified", "tree") if pipeline.tree_attention else ("unified",)
    for mode in modes:
        logger.debug("kernels: warming %s attention, full chunk", mode)
        run_stages(
            torch, arena, pipeline,
            filter_stages([question], async_ans), warm_documents, budget,
            attention_mode=mode)
        # small chunks, one document each: the trailing-chunk shapes
        # of gated multi-stage runs (see ModelPipeline.warm_tokens)
        for t in pipeline.warm_tokens:
            if t >= budget:
                continue
            logger.debug("kernels: warming %s attention, %s tokens", mode, t)
            body = (document * (t // len(document) + 1))[:max(8, t - q_max)]
            run_stages(
                torch, arena, pipeline,
                filter_stages([question], async_ans), [body], budget,
                attention_mode=mode)
        if not pipeline.canvas_ids and getattr(
                pipeline, "warm_row_classes", True):
            logger.debug("kernels: warming %s attention row classes", mode)
            _warm_row_classes(torch, arena, pipeline, mode)
    join_chunk = join_chunk and getattr(pipeline, "warm_joins", True)
    if join_chunk:
        logger.debug("kernels: warming join forward pass")
        run_stages(
            torch, arena, pipeline,
            [Stage(suffixes=[question] * 8, readout=async_ans)], warm_documents, budget)
    if join_chunk and trailing == 1:
        # a classification: the question as the frame after each
        # document, then many short suffixes of mixed lengths; a
        # decision model classifies with its own readout
        logger.debug("kernels: warming classification forward pass")
        labels = [question[:1 + i % 6] for i in range(26)]
        run_stages(
            torch, arena, pipeline,
            [Stage(suffixes=labels, readout=async_ans, frame=question)],
            warm_documents, budget)
    logger.debug("kernels: warming filter without KV writes")
    run_stages(
        torch, arena, pipeline,
        filter_stages([question], async_ans), warm_documents, budget,
        paged=False, default_attention="unified")


def _warm_row_classes(torch, arena, pipeline, attention_mode):
    """Run one chunk for each row-count class pair in ROW_CLASSES.

    Each chunk holds m one-token rows that read a cached document and
    keep their token's KV, as greedy decode rounds do, and one fresh
    filler document that brings the chunk to n rows.

    Args:
        torch: The torch module.
        arena: The KV arena the passes write.
        pipeline: The model pipeline.
        attention_mode: The attention path, "tree" or "unified".
    """
    from quail.backends.quail.executor.chunk import pack_chunk
    from quail.backends.quail.executor.stage_executor import forward_chunk

    cached = ROW_CLASS_CACHED
    readers = max(m for _, m in ROW_CLASSES)
    keys = [("warm-rows", index) for index in range(readers)]
    prefix = list(range(10, 10 + cached))
    for key in keys:
        arena.activate(key, cached, capacity_tokens=cached + 1,
                       base_tokens=cached)
    try:
        # a forward pass ends in a norm over its answer rows, which
        # fails with a CUDA error when there are none: each document
        # takes a one-token question whose answer is not read
        forward_chunk(pipeline, arena, pack_chunk(
            torch, arena, [dict(key=key, prefix=prefix, f=cached, suffixes=[[11]])
                           for key in keys], attention_mode=attention_mode))
        for n, m in ROW_CLASSES:
            groups = [dict(key=key, prefix=None, f=cached, suffixes=[[11]],
                           write_suffix_tokens=1, single=True)
                      for key in keys[:m]]
            filler = ("warm-rows", "filler", n, m)
            if n > m:
                arena.activate(filler, n - m, base_tokens=n - m)
                groups.append(dict(key=filler, prefix=prefix[:n - m],
                                   f=n - m, suffixes=[]))
            try:
                forward_chunk(pipeline, arena, pack_chunk(
                    torch, arena, groups, attention_mode=attention_mode))
            finally:
                if arena.is_resident(filler):
                    arena.free_key(filler)
    finally:
        for key in keys:
            if arena.is_resident(key):
                arena.free_key(key)


def warm_label_readout(torch, head):
    """Run the label readout in each form, at several row and target counts.

    Loads the readout's kernels into the process: the head projection,
    log-sum-exp, the target selection, and the row scatter of a
    many-row answer.

    Args:
        torch: The torch module.
        head: The model's whole output head.
    """
    from quail.backends.quail.executor.readout import AsyncLabelLogprobs

    block = AsyncLabelLogprobs.BLOCK_ROWS
    hidden = torch.zeros((block + 300, head.shape[1]), dtype=torch.bfloat16,
                         device=head.device)
    # a kernel's first launch waits for the GPU's queued work, which in
    # a query is a whole forward pass; PyTorch selects up to 16 indices
    # with one kernel and more with another
    for targets in (16, 128):
        for normalize, rows in ((True, 1), (True, 4), (False, 1)):
            readout = AsyncLabelLogprobs(torch, torch.nn.functional, head,
                                         list(range(targets)), rows=rows,
                                         normalize=normalize)
            for count in (1, 37, 200, block, block + 300):
                readout.result(readout.submit(
                    hidden[:count],
                    rows_per_answer=[1] * count if rows > 1 else None))


def _triton_warmup_m_values(budget: int) -> list[int]:
    """Token counts that compile the Ada block-FP8 GEMM tiles."""
    return sorted({min(n, budget) for n in (1, 16, 64, 256, 1024, budget)})


def compile_kernels(torch, arena, pipeline, async_ans, budget):
    """Build every GEMM kernel configuration, then warm every path.

    On DeepGEMM devices, uses vLLM's warmup heuristic to list every
    token count up to the budget at which the chosen GEMM
    configuration changes. On L40S, walks a few token counts through
    the same Triton block-FP8 kernel Engine.gemm uses. A pipeline
    without gemm_warmup skips the GEMM sweep. Runs once per (software
    stack, GPU, model, budget).
    """
    if not pipeline.gemm_warmup:
        _forward_warm(torch, arena, pipeline, async_ans, budget,
                      join_chunk=True)
        return
    linears = pipeline.linears()
    if pipeline.engine.use_deep_gemm:
        from vllm.model_executor.warmup.deep_gemm_warmup import (
            _generate_optimal_warmup_m_values,
        )
        work = [(m, lin) for lin in linears
                for m in _generate_optimal_warmup_m_values(
                    budget, lin.weight.shape[0], torch.device("cuda"))]
    else:
        counts = _triton_warmup_m_values(budget)
        work = [(m, lin) for lin in linears for m in counts]
    progress = Progress("kernels: GEMM warmup", total=len(work),
                        unit="configurations", emit=logger.info)
    logger.info("kernels: starting %s GEMM warmup configurations", len(work))
    with torch.inference_mode():
        # one buffer per linear, row-sliced per call; the sweep only
        # needs each kernel launched once
        cur, buf = None, None
        for done, (m, lin) in enumerate(work, 1):
            if lin is not cur:
                buf = torch.randn(budget, lin.weight.shape[1],
                                  device="cuda",
                                  dtype=torch.bfloat16)
                cur = lin
            q, s = pipeline.engine.quant(buf[:m])
            pipeline.engine.gemm(q, s, lin)
            progress.update(done)
        buf = None
        torch.cuda.synchronize()
    progress.finish("kernels: GEMM warmup done")
    logger.info("kernels: warming filter and join forward passes")
    _forward_warm(torch, arena, pipeline, async_ans, budget,
                  join_chunk=True)


def touch_kernels(torch, arena, pipeline, async_ans, budget):
    """Run each hot kernel once per GPU process.

    Cached binaries then load at boot instead of mid-run.
    """
    _forward_warm(torch, arena, pipeline, async_ans, budget,
                  join_chunk=True)


def _marker_path(model_name, budget):
    import os
    root = os.path.expanduser(os.environ.get(
        "QUAIL_CACHE_DIR", "~/.cache/quail/kernels"))
    safe = model_name.replace("/", "--")
    return os.path.join(root, f"quail-warm-{safe}-{int(budget)}.json")


def _marker_identity(torch, model_name, budget):
    try:
        import vllm
        vllm_version = vllm.__version__
    except ImportError:
        vllm_version = None
    return dict(warmup_version=WARMUP_VERSION, model=model_name,
                budget=int(budget), vllm=vllm_version,
                torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name())


def warm_kernels(torch, arena, pipeline, async_ans, budget, *,
                 model_name, force_compile=False, model=None):
    """Compile once under a file lock, then warm each GPU process.

    Args:
        torch: The torch module.
        arena: The KV arena the warm passes write.
        pipeline: The model pipeline.
        async_ans: The TRUE/FALSE readout of the warm passes.
        budget: Chunk token budget.
        model_name: The model's name, for the compile marker.
        force_compile: Run the compile pass even when the marker matches.
        model: The loaded model; when given, its label readout is
            warmed too.

    Returns:
        A dict with tier ("compile" or "touch"), warm_s (warmup
        seconds), and wait_s (seconds spent waiting for the lock).
    """
    import fcntl
    import json
    import os

    with quiet():
        path = _marker_path(model_name, budget)
        identity = _marker_identity(torch, model_name, budget)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t0 = time.perf_counter()
        tier = "touch"
        # The marker check, compilation, and publication must be one
        # operation across processes sharing the kernel directory.
        with open(path + ".lock", "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.info("kernels: waiting for another worker to compile")
                fcntl.flock(lock, fcntl.LOCK_EX)
            wait_s = time.perf_counter() - t0
            on_disk = None
            try:
                with open(path) as f:
                    on_disk = json.load(f)
            except (OSError, ValueError):
                pass
            if force_compile or on_disk != identity:
                logger.info("kernels: starting compile pass")
                compile_kernels(torch, arena, pipeline, async_ans, budget)
                torch.cuda.synchronize()
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(identity, f, indent=1)
                os.replace(tmp, path)
                tier = "compile"
        if tier == "touch":
            logger.info("kernels: warming cached filter and join kernels")
            touch_kernels(torch, arena, pipeline, async_ans, budget)
        if model is not None:
            warm_label_readout(torch, full_output_head(model))
        torch.cuda.synchronize()
        warm_s = round(time.perf_counter() - t0 - wait_s, 2)
        wait_s = round(wait_s, 2)
        logger.info("kernels: %s pass done in %s s; waited %s s for compilation",
                    tier, warm_s, wait_s)
        return dict(tier=tier, warm_s=warm_s, wait_s=wait_s)


