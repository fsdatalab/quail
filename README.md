# Quail Exploration

Making AI SQL filter queries fast by changing how the inference
engine manages KV memory.

This repo is the **exploration**: one workload, measured six ways, and
the code that produced the numbers. The actual system will get its own
repo once the design settles.

Scope today: gated filters, Qwen3-4B fp8, one H100.

---

## The setup

- 10,000 IMDB movie reviews, 3,203,917 tokens total
- Mean 320 tokens per document, median 246, min 44, max 2,947
- 5 gated yes/no filters; a document that fails one skips the rest
- Pass rates 0.9, 0.9, 0.9, 0.8, 0.8 → 47% survive all five
- Answers constrained to the YES/NO tokens, `max_tokens=1` → zero decode
- KV pool holds 946,800 tokens, so the corpus is **3.4x the pool**

Each document ends in a planted line, and filter *j* asks what it says:

    [FLAGS] FLAG_1=YES FLAG_2=YES FLAG_3=NO FLAG_4=YES FLAG_5=NO

The answer is in the text, so this measures execution, not reasoning.

---

## What we tried

| | what it does | who can do it |
|---|---|---|
| **Pipelining** | send a doc to filter 2 as soon as filter 1 answers | any client |
| **Token-based admission** | admit a doc only when its tokens fit a budget sized from the KV pool | any client, if it knows the pool size |
| **KV rewind** | one living request per doc: erase the question's KV back to the document boundary, append the next question | engine internals only |

Why admission is not a request count: documents run 44 to 2,947
tokens, so a request cap cannot tell you how much memory it is about
to consume.

Why rewind needs the engine: the public API has no way to erase part
of a request's KV or to keep a request alive across answers.

---

## Results

**Filters** — 10k docs, 5 filters, same container, 3 reps each:

| | wall | prefill tokens | requests |
|---|---|---|---|
| stock vLLM | 42.9 s | 1.23x corpus | 23,315 |
| KV rewind | 39.8 s | 1.20x corpus | 10,000 |

- **1.08x faster**, from two things: 2.6% fewer tokens prefilled, and
  2.3x fewer requests to schedule, sample, and detokenize
- *Prefill tokens* = tokens the engine actually computed, over the
  3,203,917-token corpus. Both sides are above 1.00x because the
  questions get prefilled too, not just the documents
- Both sides get the same memory budget. vLLM's default 4,096
  concurrent requests needs 1.58x the pool, thrashes, and prefills
  2.40x the corpus in 80.1 s — not a fair baseline
- Stock's extra 0.03x is block alignment: the engine matches cache in
  16-token blocks, so the block spanning the document/question
  boundary never matches and is recomputed every filter. A rewind cuts
  at the exact token
- **Measurement note.** The client-side counter
  (`prompt_tokens - cached_tokens`) is right for stock but wrong for
  chain mode: a rewind rewrites `prompt_token_ids`, so the final
  snapshot shows only `[document + last question]` and the
  intermediate prefills vanish. It reported an identical 1.143x for
  three operators that differ by 30% in wall time. Chain-mode numbers
  here come from the scheduler's step trace, which sees every prefill

**Cross-query reuse** — 1k docs (315k tokens), second query over the
same documents:

| | time |
|---|---|
| recompute | 4.54 s |
| restore from CPU memory | 2.33 s |

- **2.0x.** Restore wins when the channel beats `kappa x prefill_rate`
  = 7.2 GB/s at 4B
- Larger models make it easier: they read text more slowly, the
  transfer does not

---

## Is the hardware the limit?

- GPU is **99.5% busy** during filters — no idle time to reclaim
- Achieved **96,749 tok/s** against a **274,861 tok/s** ceiling → 0.35
- Not slow kernels: ncu puts the GEMMs at **91-93% of peak** (DRAM 28-30%)

Where each token's 10.4 µs goes (torch profiler, B=25,305):

| component | µs/token | share |
|---|---|---|
| matrix multiplies | 6.51 | 61% |
| fp8 quantize/scale | 2.03 | 19% |
| normalization | 1.35 | 13% |
| elementwise/activation | 0.74 | 7% |
| attention | 0.45 | 4% |

**39% of every token is memory-bound work between the multiplies.**

Step cost, fitted over B = 512 … 25,305 (unprofiled, synchronous API):

    T_step(B) = 10.40 µs x B + 2.94 ms

- Ceiling 96.2k tok/s, knee at B ≈ 283
- Analytical roofline puts the dense-projection ridge at B ≈ 416 —
  same story from the other side
- Attention only overtakes the projections at S ≈ 12,300 tokens; our
  documents are 320

---

## Run it

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q     # 45 tests, no GPU
python -m quail.roofline       # analytical limits, no GPU
```

Engine runs need Modal (one H100, `vllm==0.26.0` pinned):

```bash
modal run experiments/modal_filters.py                # the headline result
modal run experiments/modal_profiling.py::batchsweep  # the step model
modal run experiments/modal_profiling.py::torchprof   # the kernel mix
modal run experiments/modal_profiling.py::ncubench    # GEMM speed of light
modal run experiments/modal_persist.py --stage baseline
modal run experiments/modal_persist.py --stage store
modal run experiments/modal_pinprobe.py               # CPU-GPU bandwidth

python plots/make_figures.py                          # redraw figures
```

---

## Layout

```
quail/
  roofline.py                analytical limits from specs alone
  configs.py                 model and device specs
  plan/planner.py            query in, plan or refusal out
  plan/cost.py               every measured constant, one table
  runtime/engine_client.py   the two executors
  engineext/scheduler.py     the vLLM scheduler subclass
  engineext/chainlogic.py    its decision logic — no vLLM, unit-tested

experiments/
  workload.py                the corpus every experiment shares
  modal_filters.py           rewind vs stock
  modal_profiling.py         sweep, profiler, ncu
  modal_persist.py           restore vs recompute
  modal_pinprobe.py          which pinned-memory paths work

plots/make_figures.py        rebuilds every figure in results/plots
```

---

## Open problems

- vLLM's offload connector delivers 10.2 GB/s from memory that copies
  at 55.4 GB/s. Five sixths of the link is lost inside it
- B=1024 is reproducibly slower than B=512. Unexplained, excluded from
  the fit, not hidden
- Per-token cost is not constant: GEMMs cost 4.65 µs/token at B=512 and
  6.51 at B=25,305. Probably L2. The linear model averages over it
- This checkpoint misreads ~27% of flag values that are written in the
  text. Identical across executors, so comparisons hold — but do not
  use this setup for accuracy claims
- Decode is untouched. Everything here is prefill

---

## Out of scope (for now)

Open-ended maps, classification, speculation and forking, 32B, multi-GPU.
All removed on purpose; git history has them.
