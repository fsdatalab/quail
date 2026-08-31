"""Pipelined SGLang baseline for QUAIL-B queries.

Runs on H100s via Modal. Uses the same submission strategy as
``pipelined_vllm``: each document's next filter stage goes in once
the document passes its current stage, with token-budget admission
(in waves of blocking generate calls -- see
``StockSGLangClient.run_pipelined_filter_chain``), and every join
runs the full cross product. The prompts, query definitions, and
per-query bookkeeping are imported from ``baselines.stock_vllm.run``,
so the baselines measure exactly the same work.

Join pairs are submitted suffix-major in anchor tiles
(``baselines.stock.suffix_major_tiled_order``) instead of vLLM's
anchor-major order: SGLang's radix cache stores KV only for finished
requests, so each engine gets the pair order its prefix cache can
exploit, over the identical cross product.

Engine settings mirror the vLLM baseline configuration:

    vLLM                              SGLang
    gpu_memory_utilization=0.91       mem_fraction_static=0.78
                                      (see MEM_FRACTION_STATIC)
    max_num_seqs=4096                 max_running_requests=4096
    max_num_batched_tokens=25305      chunked_prefill_size=25296,
                                      max_prefill_tokens=25296
                                      (25305 rounded down to a
                                      16-token page multiple)
    16-token cache blocks             page_size=16
    enable_prefix_caching=True        radix cache on (default)
    allowed_token_ids=TRUE/FALSE      logit_bias=+1000 on the same ids
    (KV pool sized after profiling)   disable_prefill_cuda_graph=True
    (tokenizer in-process)            skip_tokenizer_init=True

Default run (BIO-2 and IMDB-3 at sf=0.1 on qwen3-4b-fp8):
    uv run modal run -m baselines.stock_sglang.run::main

All 30 queries: pass ``--query ""``.

Boot-and-generate probe without benchmark data:
    uv run modal run -m baselines.stock_sglang.run::probe
"""

import json
import time
import uuid
from pathlib import Path

import modal

from quail.bench.quailb import QUERY_ORDER, SELECTIVITY_ESTIMATE_COLLECTION

app = modal.App("quail-milestone1")

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("sglang==0.5.18", "huggingface_hub[hf_transfer]",
                 "pandas", "pyarrow", "datasets")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1",
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton",
          "FLASHINFER_CACHE_DIR": "/root/.cache/kernels/flashinfer"})
    .add_local_python_source("quail", "baselines")
)

hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

# BIO-2 submits 563,500 requests at once; their token id lists live in
# both the driver and the scheduler process, so the host needs more RAM
# than the GPU function default.
GPU_KW = dict(image=image, gpu="H100!", memory=98304,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})

DATA_DIR = "/results/quailb_data"

BASELINE = "pipelined_sglang"
DEFAULT_QUERY_IDS = "BIO-2,IMDB-3"
# vLLM's gpu_memory_utilization=0.91 covers weights, KV, and the
# activation working set, because vLLM profiles a full-size forward
# (including logits for max_num_seqs requests) before sizing its KV
# pool. SGLang's mem_fraction_static covers only weights plus KV;
# everything else must fit in the remainder. Measured on BIO-2, that
# remainder must hold about 6.5 GB of non-PyTorch allocations (kernel
# workspaces) plus the answer-step peak: the radix cache packs
# batches up to max_running_requests=4096, and the final-position
# logits and the logit-bias tensor are each
# 4096 x vocab x 4 bytes = 2.3 GB. 0.91 and 0.85 both ran out of GPU
# memory mid-join; 0.78 leaves about 17 GB for all of it, at the cost
# of a KV pool about 13% smaller than vLLM's 479,248 tokens.
MEM_FRACTION_STATIC = 0.78
MAX_NUM_SEQS = 4096
MAX_NUM_BATCHED_TOKENS = 25_305
# Both joins are host-bound, not GPU-bound: the measured BIO-2 join
# moved 5,545 fresh tokens/s while the per-request bookkeeping ran at
# 448 pairs/s. 16-token pages (vLLM's block size) cut the radix-tree
# and KV-index work per request 16x; the cost is that prefix hits
# round down to a page multiple, about 8 extra fresh tokens per
# request, which the idle GPU absorbs. sglang requires
# chunked_prefill_size to be divisible by page_size, so vLLM's 25,305
# token budget rounds down to 25,296 (9 tokens, 0.04%).
PAGE_SIZE = 16
CHUNKED_PREFILL_TOKENS = (MAX_NUM_BATCHED_TOKENS // PAGE_SIZE) * PAGE_SIZE

# vLLM restricts decoding with allowed_token_ids. SGLang has no direct
# equivalent, so the same token ids get a large additive bias instead.
# The bias is applied to float32 logits before the greedy argmax; 1000
# exceeds any raw logit gap while adding the same constant to every
# allowed id, so the chosen token matches vLLM's restricted argmax.
TRUE_FALSE_LOGIT_BIAS = 1000.0


class _Completion:
    __slots__ = ("token_ids",)

    def __init__(self, token_ids):
        self.token_ids = token_ids


class _RequestOutput:
    __slots__ = ("outputs", "prompt_token_ids", "num_cached_tokens")

    def __init__(self, prompt_token_ids, token_ids, num_cached_tokens):
        self.outputs = [_Completion(token_ids)]
        self.prompt_token_ids = prompt_token_ids
        self.num_cached_tokens = num_cached_tokens


class StockSGLangClient:
    """The subset of vLLM's LLM surface used by the pipelined client.

    Wraps a running sglang Engine so ``run_query`` from
    ``baselines.stock_vllm.run`` works unchanged. Filters chain per
    document through ``run_pipelined_filter_chain``; joins always go
    suffix-major in anchor tiles sized to ``join_tile_budget_tokens``
    (``baselines.stock.suffix_major_tiled_order``): the radix cache
    stores a prompt's KV only when its request finishes, so vLLM's
    anchor-major order recomputes an anchor for every sibling that is
    already in flight.
    """

    join_submission = "suffix-major-tiled"

    # Handing sglang all 563,500 BIO-2 pairs in one generate() call
    # keeps the driver process pegged for the whole join: one asyncio
    # task per request, and the Modal health heartbeat thread starves
    # until Modal marks the container unhealthy. Each slice is still
    # four times deeper than max_running_requests, so the engine's
    # queue never runs dry inside a slice. The pause between slices
    # only needs to hand the GIL to the heartbeat thread; the engine's
    # queue is empty during it, so every extra tenth of a second is
    # idle GPU time at each of BIO-2's 34 slice boundaries.
    submit_slice = 16_384
    slice_pause_s = 0.1

    def __init__(self, engine, capacity):
        self.engine = engine
        self.block_size = capacity["block_size"]
        # Half the KV pool bounds a join tile: in-flight suffixes and
        # the previous tile's leftovers share the pool with the tile's
        # anchors. Filter admission gets the whole pool, matching the
        # budget stock vLLM's pipelined client receives.
        self.join_tile_budget_tokens = (
            capacity["kv_cache_size_tokens"] // 2)
        self.filter_budget_tokens = capacity["kv_cache_size_tokens"]

    def run_pipelined_filter_chain(self, sampling_params, body_ids,
                                   question_ids, true_ids, tag="q"):
        """Chain filter stages per document with token-budget admission.

        Mirrors stock vLLM's pipelined client: a document holds one of
        doc_cap admission slots for its whole chain, and a TRUE answer
        sends the document's next stage while other documents are
        still on earlier stages. SGLang's scheduler runs in a separate
        process with no synchronous add_request/step surface, so the
        chain advances in waves of blocking generate() calls: every
        alive document has exactly one request per wave, and wave
        boundaries stand in for vLLM's engine step loop. Sequential
        stages keep the radix cache effective: a document's stage j+1
        always finds its body cached, because stage j finished first.

        Returns:
            Dict shaped like baselines.stock.run_filter_chain's.
        """
        del tag
        n_stages = len(question_ids)
        longest_tail = max(len(q) for q in question_ids)
        sizes = [len(body) + longest_tail + 1 for body in body_ids]
        mean_request = sum(sizes) // max(1, len(sizes))
        cap = min(max(1, self.filter_budget_tokens // mean_request),
                  MAX_NUM_SEQS)
        counters = dict(requests=0, prompt_tokens=0, cached_tokens=0)
        answers = {}
        survivors = []

        t0 = time.time()
        alive = []                       # (document index, stage)
        next_doc = 0
        while alive or next_doc < len(body_ids):
            while next_doc < len(body_ids) and len(alive) < cap:
                alive.append((next_doc, 0))
                next_doc += 1
            prompts = [
                {"prompt_token_ids": body_ids[index]
                 + question_ids[stage]}
                for index, stage in alive
            ]
            outputs = self.generate(prompts, sampling_params)
            advanced = []
            for (index, stage), out in zip(alive, outputs):
                counters["requests"] += 1
                counters["prompt_tokens"] += len(out.prompt_token_ids)
                counters["cached_tokens"] += out.num_cached_tokens
                token_ids = out.outputs[0].token_ids
                got = 1 if (token_ids and
                            int(token_ids[0]) in true_ids) else 0
                answers[(index, stage + 1)] = got
                if got and stage + 1 < n_stages:
                    advanced.append((index, stage + 1))
                elif got:
                    survivors.append(index)
            alive = advanced
        wall = time.time() - t0
        return dict(wall=wall, survivors=sorted(survivors),
                    answers=answers, doc_cap=cap,
                    budget_tokens=self.filter_budget_tokens,
                    block_size=self.block_size,
                    max_num_seqs=MAX_NUM_SEQS,
                    **counters)

    def generate(self, prompts, sampling_params, use_tqdm=False):
        outputs = []
        for start in range(0, len(prompts), self.submit_slice):
            if start:
                time.sleep(self.slice_pause_s)
            outputs.extend(self._generate_slice(
                prompts[start:start + self.submit_slice],
                sampling_params))
        return outputs

    def _generate_slice(self, prompts, sampling_params):
        input_ids = [list(p["prompt_token_ids"]) for p in prompts]
        raw = self.engine.generate(
            input_ids=input_ids, sampling_params=dict(sampling_params))
        if isinstance(raw, dict):
            raw = [raw]
        if len(raw) != len(input_ids):
            raise RuntimeError(
                f"sglang returned {len(raw)} outputs for "
                f"{len(input_ids)} requests")
        outputs = []
        for ids, result in zip(input_ids, raw):
            meta = result["meta_info"]
            outputs.append(_RequestOutput(
                ids,
                list(result.get("output_ids") or []),
                int(meta.get("cached_tokens") or 0)))
        return outputs

    def reset_prefix_cache(self):
        ret = self.engine.flush_cache()
        return bool(getattr(ret, "success", True))


def time_engine_boot(**engine_kwargs):
    """Cold-construct sglang's Engine(...) and time it.

    Returns:
        Tuple of (engine, boot_dict) shaped like stock_boot's dict.
    """
    import sglang as sgl

    t0 = time.perf_counter()
    engine = sgl.Engine(**engine_kwargs)
    total = time.perf_counter() - t0
    boot = dict(kind="cold", llm_init_s=round(total, 2),
                weight_load_s=None, kv_profile_s=None,
                boot_s=round(total, 2))
    return engine, boot


def _sglang_filter_capacity(engine):
    """Read the post-startup KV capacity from sglang."""
    info = engine.get_server_info()
    capacity = info.get("max_total_num_tokens")
    page_size = info.get("page_size")
    if capacity is None or page_size is None:
        raise RuntimeError("sglang did not report its KV capacity")
    max_running = info.get("max_running_requests")
    return dict(
        kv_cache_size_tokens=int(capacity),
        num_gpu_blocks=None,
        block_size=int(page_size),
        max_num_seqs=(None if max_running is None else int(max_running)),
        kv_cache_dtype=str(info.get("kv_cache_dtype")),
    )


def _engine_settings(engine):
    """Report the sglang settings that matter for the comparison."""
    info = engine.get_server_info()
    fields = ("version", "attention_backend", "sampling_backend",
              "schedule_policy", "page_size", "chunked_prefill_size",
              "max_prefill_tokens", "max_running_requests",
              "mem_fraction_static", "max_total_num_tokens",
              "kv_cache_dtype", "disable_radix_cache",
              "cuda_graph_max_bs_decode", "cuda_graph_max_bs_prefill")
    return {field: info.get(field) for field in fields}


def _boot_client(model, mem_fraction_static):
    """Boot one sglang Engine plus tokenizer and answer token sets."""
    from baselines.stock_vllm.run import MODELS
    from quail.executor.loop import true_false_ids
    from transformers import AutoTokenizer

    hf_name = MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    true, false = true_false_ids(tokenizer)
    allowed = sorted(true | false)

    # Prefill CUDA graphs retain ~130 MB of capture memory per shape
    # (91 shapes up to the 25,296-token budget), which does not fit
    # next to a KV pool sized at 0.91 of the GPU; vLLM sizes its KV
    # pool after profiling, so its 0.91 already accounts for
    # activations. Disabling the prefill graph keeps the memory split
    # equivalent. This workload packs many cached-prefix requests per
    # prefill batch, so per-batch launch overhead is amortized anyway.
    # skip_tokenizer_init: the client sends token ids and reads token
    # ids back, so the detokenizer would be pure per-request overhead;
    # with the flag set the scheduler sends results straight to the
    # driver and the detokenizer process sits idle.
    engine, boot = time_engine_boot(
        model_path=hf_name,
        mem_fraction_static=mem_fraction_static,
        max_running_requests=MAX_NUM_SEQS,
        chunked_prefill_size=CHUNKED_PREFILL_TOKENS,
        max_prefill_tokens=CHUNKED_PREFILL_TOKENS,
        page_size=PAGE_SIZE,
        skip_tokenizer_init=True,
        disable_radix_cache=False,
        disable_prefill_cuda_graph=True,
        log_level="warning")
    capacity = _sglang_filter_capacity(engine)
    llm = StockSGLangClient(engine, capacity)
    sp = {"temperature": 0.0, "max_new_tokens": 1,
          "logit_bias": {str(t): TRUE_FALSE_LOGIT_BIAS
                         for t in allowed}}
    return llm, boot, sp, tokenizer, true, allowed, hf_name, capacity


@app.function(timeout=2400, **GPU_KW)
def probe(model: str = "qwen3-4b-fp8",
          mem_fraction_static: float = MEM_FRACTION_STATIC) -> str:
    """Boot the engine and run a few filter and join shaped requests."""
    from baselines.stock_vllm.run import _run_filter_chain, _run_join
    from quail.bench.quailb import F1, F4, DISCUSS_ASPECT

    (llm, boot, sp, tokenizer, true, allowed, hf_name,
     capacity) = _boot_client(model, mem_fraction_static)
    settings = _engine_settings(llm.engine)
    print(f"[probe] boot: {boot}", flush=True)
    print(f"[probe] KV capacity: {capacity} "
          f"join_tile_budget_tokens={llm.join_tile_budget_tokens}",
          flush=True)
    print(f"[probe] settings: {settings}", flush=True)

    llm.generate([{"prompt_token_ids": allowed}], sp, use_tqdm=False)

    texts = [
        "A luminous, big-hearted film; the ensemble cast is "
        "extraordinary and the final act soars.",
        "Dreadful pacing and wooden acting; I walked out before "
        "the end.",
        "The soundtrack carries several scenes, though the plot "
        "meanders.",
    ]
    filter_result = _run_filter_chain(
        llm, sp, true, [F1, F4], texts, tokenizer, "pipelined",
        capacity, tag="probe")
    print(f"[probe] filter chain answers: {filter_result['answers']} "
          f"survivors={filter_result['survivors']} "
          f"doc_cap={filter_result['doc_cap']} "
          f"prompt_tokens={filter_result['prompt_tokens']} "
          f"cached_tokens={filter_result['cached_tokens']}", flush=True)

    (join_result, n_pairs, _sl, _sr, n_true, _tp, _pairs, join_answers,
     anchor, _mdt, _prefixes) = _run_join(
        llm, sp, true, DISCUSS_ASPECT, texts, ["the acting", "the plot"],
        tokenizer)
    print(f"[probe] join: {n_true}/{n_pairs} TRUE anchor={anchor} "
          f"answers={join_answers} "
          f"fresh={join_result['fresh_tokens']}", flush=True)

    flushed = llm.reset_prefix_cache()
    print(f"[probe] flush_cache success: {flushed}", flush=True)
    return json.dumps(dict(
        boot=boot, capacity=capacity, settings=settings,
        filter_answers={f"{index}-{stage}": answer
                        for (index, stage), answer
                        in filter_result["answers"].items()},
        filter_survivors=filter_result["survivors"],
        join_answers=join_answers, join_true=n_true,
        join_pairs=n_pairs, flush=flushed))


def _run_query_batch(model, sf, query_ids_csv, reps,
                     ground_truth_collection, prediction,
                     mem_fraction_static, lf=1):
    """Boot one sglang Engine and run the stage-major baseline."""
    from baselines.stock_vllm.run import define_all_queries, run_query
    from quail.bench.quailb import (
        DATA_SEED,
        SOURCE_REVISIONS,
        build_sets,
        queries as quail_queries,
        register_sets,
    )

    data_path = build_sets(DATA_DIR, sf, lf)
    (llm, boot, sp, tokenizer, true, allowed, hf_name,
     filter_capacity) = _boot_client(model, mem_fraction_static)
    engine_settings = _engine_settings(llm.engine)
    print(f"[{BASELINE}] boot: {boot}", flush=True)
    print(f"[{BASELINE}] KV capacity: {filter_capacity} "
          f"join_tile_budget_tokens={llm.join_tile_budget_tokens}",
          flush=True)
    print(f"[{BASELINE}] settings: {engine_settings}", flush=True)

    llm.generate([{"prompt_token_ids": allowed}], sp, use_tqdm=False)

    queries = define_all_queries()
    ids = [q.strip() for q in query_ids_csv.split(",") if q.strip()]
    for qid in ids:
        if qid not in queries:
            raise ValueError(
                f"unknown query {qid!r}; available: {sorted(queries)}")

    evaluator = None
    truth = None
    evaluation_queries = {}
    if ground_truth_collection:
        import quail
        from quail.bench.evaluate import (
            BenchmarkEvaluator,
            LocalVolumeFiles,
            corpus_identity,
            load_ground_truth,
            read_corpus,
        )
        from quail.planner.plan import EngineConfig

        corpus_rows = read_corpus(data_path)
        corpus = corpus_identity(
            corpus_rows, sf, DATA_SEED, SOURCE_REVISIONS)
        truth = load_ground_truth(
            LocalVolumeFiles("/results"),
            scale_factor=sf,
            corpus_id=corpus["corpus_id"],
            collection_id=ground_truth_collection,
        )
        evaluator = BenchmarkEvaluator(truth, corpus_rows)
        session = quail.Session(
            EngineConfig(gpus=1, model=model),
            tokenizer=lambda text: tokenizer.encode(
                text, add_special_tokens=False),
        )
        register_sets(session, data_path)
        evaluation_queries = quail_queries(session)

    all_results = []
    for rep in range(reps):
        rep_results = []
        for qid in ids:
            if not llm.reset_prefix_cache():
                raise RuntimeError(
                    "sglang refused to flush its radix cache before "
                    f"{BASELINE} {qid}")
            print(f"\n[{BASELINE}] rep={rep} {qid}", flush=True)
            try:
                entry = run_query(
                    llm, sp, true, tokenizer,
                    qid, queries[qid], DATA_DIR, sf,
                    evaluator=evaluator,
                    quail_query=(evaluation_queries[qid][1]()
                                 if evaluator else None),
                    filter_submission="pipelined",
                    filter_capacity=filter_capacity)
            except Exception as e:                      # noqa: BLE001
                entry = dict(query=qid,
                             error=f"{type(e).__name__}: {e}")
                print(f"  ERROR: {e}", flush=True)
            rep_results.append(entry)
        all_results.append(rep_results)

    return dict(
        baseline=BASELINE, model=model, hf_name=hf_name, sf=sf, lf=lf,
        boot=boot, reps=reps,
        prediction=prediction,
        ground_truth_workload=None,
        ground_truth_collection=(None if truth is None else
                                 truth.collection_id),
        ground_truth=(None if truth is None else {
            "collection_id": truth.collection_id,
            "corpus_id": truth.corpus_id,
            "reference_model": truth.reference_model,
        }),
        query_ids=ids,
        filter_submission="pipelined",
        filter_capacity=filter_capacity,
        join_submission=llm.join_submission,
        join_tile_budget_tokens=llm.join_tile_budget_tokens,
        submission=("pipelined per-document filter chain, full cross "
                    "product per join, join pairs submitted "
                    "suffix-major in anchor tiles"),
        checkpoint="pre-quantized FP8",
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=CHUNKED_PREFILL_TOKENS,
        page_size=PAGE_SIZE,
        skip_tokenizer_init=True,
        mem_fraction_static=mem_fraction_static,
        enable_prefix_caching=True,
        true_false_logit_bias=TRUE_FALSE_LOGIT_BIAS,
        engine="sglang",
        engine_settings=engine_settings,
        results=all_results)


@app.function(timeout=7200, **GPU_KW)
def run_query_batch(model: str = "qwen3-4b-fp8", sf: float = 0.1,
                    query_ids_csv: str = DEFAULT_QUERY_IDS,
                    reps: int = 1,
                    ground_truth_collection: str = "",
                    prediction: str = "",
                    mem_fraction_static: float =
                    MEM_FRACTION_STATIC) -> str:
    """Boot one sglang Engine and run one batch of queries."""

    report = _run_query_batch(
        model, sf, query_ids_csv, reps, ground_truth_collection,
        prediction, mem_fraction_static)
    return json.dumps(report)


@app.function(timeout=300, image=image,
              volumes={"/results": results_vol})
def save_report(report_json: str, label: str) -> str:
    """Save a report to the quail-results volume."""
    out_dir = Path("/results") / BASELINE / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    with open(out_path, "w") as f:
        f.write(report_json)
    results_vol.commit()
    return str(out_path)


@app.local_entrypoint()
def main(model: str = "qwen3-4b-fp8", sf: float = 0.1,
         query: str = DEFAULT_QUERY_IDS, reps: int = 1,
         ground_truth_collection: str = SELECTIVITY_ESTIMATE_COLLECTION,
         prediction: str = "",
         mem_fraction_static: float = MEM_FRACTION_STATIC):
    if query:
        ids = [q.strip() for q in query.split(",") if q.strip()]
    else:
        ids = list(QUERY_ORDER)
    label = f"{time.strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    if prediction:
        print(f"PREDICTION: {prediction}", flush=True)

    fc = run_query_batch.spawn(
        model=model, sf=sf, query_ids_csv=",".join(ids), reps=reps,
        ground_truth_collection=ground_truth_collection,
        prediction=prediction,
        mem_fraction_static=mem_fraction_static)
    print(f"function call id: {fc.object_id}")
    report = json.loads(fc.get())

    out_path = save_report.remote(json.dumps(report, indent=2), label)
    print(f"\n[{BASELINE}] saved {out_path}")
    print(f"  {len(ids)} queries, {reps} reps")
    print(json.dumps(report))
