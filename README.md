# Quail — a QUery-Aware Inference Layer

Quail runs AI SQL filter queries over document collections by
scheduling the work *inside* vLLM instead of through its client API.
It is a custom scheduler and memory manager: the query plan, not the
engine's heuristics, decides what enters GPU memory and what stays.

The problem it exists for: an inference engine keeps or drops KV cache
by recency, because it does not know what the query will ask next. A
query does know. When five filters run over the same document, the
engine re-reads that document up to five times; the plan can see that
coming and hold the KV instead.

Scope right now is deliberately narrow: **gated filter queries, Qwen3
4B fp8, one H100**. Open-ended maps, classification, multi-model and
multi-GPU work are out until the filter case is fully understood.

## The workload

Five gated yes/no filters over 10,000 IMDB movie reviews. Each
document ends in a planted metadata line —

    [FLAGS] FLAG_1=YES FLAG_2=YES FLAG_3=NO FLAG_4=YES FLAG_5=NO

— and filter *j* asks for the value of `FLAG_j`. A document that fails
a filter skips the rest. Selectivities are 0.9, 0.9, 0.9, 0.8, 0.8, so
about 47% of documents survive all five.

The answer is in the text, so the query measures execution rather than
reasoning. Documents average 320 tokens (median 246, max 2,947),
3,203,917 tokens in total — **3.4 times the KV pool**, so the corpus
cannot be held resident and something has to decide what to keep.

Answers are constrained to the yes/no token ids with `max_tokens=1`,
so every stage answers from its prefill pass and no decode step ever
runs. That is what "zero decode" means here.

## What we measured

Qwen3-4B fp8 on one H100 SXM 80 GB, via Modal.

**Speed of light.** The dense projections cross the roofline ridge
(591 FLOP/byte) at about 400 tokens per step, so any step budget past
~400 saturates them. Attention only overtakes the projections at a
context of ~12,300 tokens, far past our 320-token documents. The
analytical ceiling is 274,861 prefill tokens/s.

**The step model.** Sweeping `max_num_batched_tokens` from 512 to
25,305, unprofiled, through the synchronous API:

    T_step(B) = 10.7 us x B  +  3.1 ms

a ceiling of about 94,000 tokens/s and a knee near B = 290, which
agrees with the analytical ridge. Measured throughput at B = 25,305 is
97,005 tokens/s against the 274,861 ceiling — an overhead multiplier
of **0.49**.

**Where the 10.7 us goes** (torch profiler inside the engine core):
GEMMs 6.5 us (61%), fp8 quantize/scale 2.0 (19%), normalization 1.35
(13%), elementwise 0.74 (7%), attention 0.45 (4%). Nsight Compute says
the GEMM kernels themselves run at 91-93% of the compute ceiling with
DRAM at 28-30%, so the gap to peak is the step *mix*, not slow
kernels. The GPU is 99.5% busy and the host scheduling hides entirely
behind it.

**KV rewind against stock pipelining**, 10k documents, 5 filters, same
container, 3 reps:

| | wall | corpus reads | requests |
|---|---|---|---|
| stock pipelining (fair concurrency) | 42.9 s | 1.23x | 23,315 |
| KV rewind | 39.8 s | 1.14x | 10,000 |

**1.08x.** The honest number. Setting stock's concurrency to the
engine's 4,096 sequence cap instead of the pool-derived 2,048
overflows the KV pool by 1.6x, forces the prefix cache to evict
between stages, and costs it 2.40x reads and 80.1 s — which would look
like a 2x win for rewind and is really a misconfiguration. The
surviving 0.09x of rereads is block alignment: a prefix cache matches
whole 16-token blocks, so the block straddling the document boundary
is recomputed on every stage, while a rewind cuts by token position
and keeps it.

**Persisted KV.** After the first query has computed the corpus, a
second query can restore its KV from host memory instead of
recomputing: 2.33 s against 4.54 s, **2.0x**. Restore wins when the
channel beats `kappa x prefill_rate` = 7.2 GB/s at 4B. The measured
channel: 64 GB/s PCIe Gen5 on paper, 55.4 GB/s for a raw copy from
alloc-pinned host memory, and 10.2 GB/s end to end through vLLM's
offload connector. That last gap is unexplained and open.

## The three mechanisms

**Pipelining** — send a document to the next filter the moment the
previous one answers, instead of running the corpus through filter 1
before starting filter 2. Obvious, and any client can do it.

**Token-based admission** — a document enters the engine only when its
worst-case tokens fit a budget sized from the KV pool. A request count
cannot do this job: it cannot see that one document is 44 tokens and
another 2,947. This is what makes the stock comparison fair, and what
keeps the pool from overflowing.

**KV rewind** — one living request per document for the whole chain.
The scheduler judges each answer at prefill, erases the question's KV
back to the document boundary *by token position*, and appends the
next question onto the still-resident document KV. Because the request
never finishes, its KV cannot be evicted between stages. No client can
do this: the public API has no way to erase part of a request or to
keep its KV pinned.

## Layout

    quail/
      configs.py             model and device specs
      roofline.py            analytical component roofline (no runs needed)
      plan/cost.py           every calibrated constant, in one table
      plan/planner.py        describe the query, get a plan or a refusal
      runtime/engine_client.py  the two executors (stock baseline, rewind)
      engineext/chainlogic.py   pure decision logic, no vLLM, unit-tested
      engineext/scheduler.py    the vLLM scheduler subclass

    experiments/
      workload.py            the corpus, the questions, the Modal image
      modal_filters.py       rewind against stock pipelining
      modal_profiling.py     batch sweep, torch profiler, ncu
      modal_persist.py       restore against recompute
      modal_pinprobe.py      which pinned-memory paths work, and how fast

    plots/make_figures.py    rebuilds every figure in results/plots

`chainlogic.py` imports no vLLM, torch, or numpy: every decision the
scheduler makes is computed there on plain values and unit-tested with
no engine installed, so a vLLM version bump can break the adapter but
not the rules.

## Running it

```bash
pip install -e ".[dev]"      # CPU: planner, roofline, tests
python -m pytest tests/ -q
python -m quail.roofline     # the analytical table, no GPU needed

modal run experiments/modal_filters.py                  # the headline result
modal run experiments/modal_profiling.py::batchsweep    # the step model
modal run experiments/modal_profiling.py::torchprof     # the kernel mix
modal run experiments/modal_profiling.py::ncubench      # GEMM speed of light
modal run experiments/modal_persist.py --stage baseline
modal run experiments/modal_persist.py --stage store

python plots/make_figures.py                            # rebuild figures
```

Engine work needs Modal (`vllm==0.26.0`, one H100); there is no local
GPU path. The scheduler subclass reaches into vLLM internals pinned at
that exact version.

## Open questions

- The offload connector delivers 10.2 GB/s from memory that copies at
  55.4 GB/s. Five-sixths of the channel is lost somewhere inside it.
- `B = 1024` is reproducibly slower than `B = 512` in the sweep. Not
  explained; excluded from the fit and reported.
- The per-token cost is not really constant: it rises from 4.65 us at
  B = 512 to 6.51 at B = 25,305 for the GEMMs alone, most likely L2
  behaviour. The linear model absorbs this into an average.
- The 4B fp8 checkpoint misreads about 27% of flag values it can see.
  Identical across executors, so it does not affect any comparison
  here, but it makes this checkpoint unfit for accuracy claims.
- Decode is entirely unaddressed. Everything above is prefill.
