# Shared join prompts and stock vLLM

Date: 2026-08-26.

## Result

Quail was faster than stock vLLM on all six join comparisons. The largest
difference was BIO-2 at 4B. Quail took 31.12 seconds. Stock vLLM took 250.00
seconds, which was 8.03 times slower.

Figure: plots/stock_vllm_vs_quail.png

The plot uses a log scale because the times span more than two orders of
magnitude.

| Model | Query | Pairs | SoL estimate (s) | Quail (s) | Quail / SoL | Stock vLLM (s) | Stock / Quail |
|---|---|---:|---:|---:|---:|---:|---:|
| 4B | BIO-2 | 122,800 | 15.59 | 31.12 | 2.00 times | 250.00 | 8.03 times |
| 4B | FEV-2 | 5,700 | 0.57 | 1.29 | 2.27 times | 3.34 | 2.59 times |
| 4B | IMDB-2 | 60,000 | 9.28 | 20.55 | 2.21 times | 29.13 | 1.42 times |
| 32B | BIO-2 | 122,800 | 104.15 | 181.05 | 1.74 times | 283.96 | 1.57 times |
| 32B | FEV-2 | 5,700 | 4.71 | 8.47 | 1.80 times | 10.49 | 1.24 times |
| 32B | IMDB-2 | 60,000 | 77.71 | 139.88 | 1.80 times | 155.10 | 1.11 times |

Quail uses the cold pass and excludes model load time. Stock vLLM uses the
timer inside `baselines.stock.run_join_grouped`. That timer covers only the
`llm.generate()` call. CPU prompt construction and the Modal call are outside
both timers.

The SoL estimate is computed from the exact prompt tokens, attention pairs,
model specifications, and H100 specifications. It is not a measured system.

## Prediction

The prediction was recorded before the warmed stock runs.

| Model | Query | Predicted stock time (s) | Measured stock time (s) | Result |
|---|---|---:|---:|---|
| 4B | BIO-2 | 155 to 180 | 250.00 | Above the range |
| 4B | FEV-2 | 2.3 to 2.8 | 3.34 | Above the range |
| 4B | IMDB-2 | 23 to 28 | 29.13 | Above the range |
| 32B | BIO-2 | 225 to 250 | 283.96 | Above the range |
| 32B | FEV-2 | 10.5 to 12.5 | 10.49 | 0.01 seconds below the range |
| 32B | IMDB-2 | 145 to 165 | 155.10 | Inside the range |

The prediction assumed the 64 pair warmup would remove the long delay before
BIO-2 began processing requests. It did not. That delay comes from passing
122,800 requests into vLLM. It is inside `llm.generate()` and belongs in the
stock baseline time.

An earlier run without the 64 pair warmup measured 206.88, 2.53, and 25.54
seconds at 4B. It measured 283.34, 11.51, and 156.52 seconds at 32B. The plot
uses the warmed run because that matches the existing stock experiment setup.
The warmup itself took less than one second for every query. The difference
between the two 4B runs shows that the stock timing has meaningful run to run
variation.

## Stock vLLM implementation

The stock result comes from `baselines.stock.run_join_grouped`.

For each join, it does the following:

1. It creates one vLLM request for every document pair.
2. It orders those requests by anchor document.
3. It passes the full list to one `llm.generate()` call.
4. It lets the vLLM scheduler enforce the active sequence and token limits.

The stock filter client is different. `baselines.stock.run_filter_chain`
computes a document admission limit from the token budget and carries only
TRUE documents to the next filter stage. That filter admission logic is not
used by `run_join_grouped`.

The separate vLLM opbench code uses the same join submission pattern. It adds
detailed metric collection while the request is running. It is kept as a
diagnostic tool and is not shown as another performance baseline.

## Shared join prompts

Quail and stock vLLM use the exact same complete prompt token IDs for every
document pair. The prompt order is:

```text
DOCUMENT:
<anchor document>

(The document above is DOCUMENT {0}.)

Evaluate TRUE or FALSE for the following question: <join question>

DOCUMENT {1}:
<partner document>
ANSWER:
```

This order processes the complete static question once per anchor in Quail.
Each pair adds only the partner label, partner document, and answer cue. Tests
compare the exact complete token IDs for both possible anchor choices. A
separate test covers a join with three inputs.

For `merge_quant`, the suffix contains the partner label, partner document,
and `ANSWER:`. The persistent document KV contains `DOCUMENT:\n` and the
anchor document. The active kept KV also contains the anchor note and complete
question. `merge_quant` does not save suffix KV.

## Planner accounting

The planner counts the prompt in the same order used at runtime. For a join
with two inputs, it computes:

```text
anchors * (shared preamble + anchor document + anchor note + question)
+ pairs * (partner label + partner document + answer cue)
```

The measured Quail fresh token counts matched the planner exactly.

| Query | Planned fresh tokens | Measured fresh tokens | Change from the old prompt |
|---|---:|---:|---:|
| BIO-2 | 2,635,599 | 2,635,599 | 76 percent fewer than 10,979,399 |
| IMDB-2 | 2,419,233 | 2,419,233 | 59 percent fewer than 5,944,233 |
| FEV-2 | 145,359 | 145,359 | The evidence input is the anchor |

Stock vLLM reuses complete 16 token KV blocks. Its measured fresh prompt token
counts were 2,904,666 for BIO-2, 152,994 for FEV-2, and 2,439,636 for IMDB-2
at 4B. The 32B counts were 2,883,690, 152,994, and 2,439,636.

## Settings

- The scale factor was 0.1.
- The model weights were FP8 for every run and estimate.
- Stock vLLM loaded `Qwen/Qwen3-4B-FP8` or `Qwen/Qwen3-32B-FP8`.
- KV was BF16 for both measured systems.
- Each model used one exact `H100!` request and one model copy.
- vLLM was version 0.26.0 with CUDA 13.0.1.
- Prefix caching was enabled with 16 token blocks.
- vLLM could use 92 percent of GPU memory, matching Quail's memory pool.
- `max_num_seqs` was 4,096.
- `max_num_batched_tokens` was 25,305.
- The model context limit was 40,960 tokens.
- Each timed stock join followed a 64 pair warmup with the same prompt shape.
- Stock vLLM submitted 122,800 BIO-2 requests, 5,700 FEV-2 requests, and
  60,000 IMDB-2 requests.

The three timed queries do not have filter stages. Ground truth labels do not
change which rows reach these joins. This report does not measure accuracy.

## What the measurements mean

Stock vLLM was closest to Quail on IMDB-2. It was 1.42 times slower at 4B and
1.11 times slower at 32B. The largest gap was BIO-2 at 4B, where stock vLLM
was 8.03 times slower.

The earlier comparison included a second vLLM series from the instrumented
vLLM opbench path. That path and the stock join client use the same join
submission pattern. Showing them as two different join baselines was
misleading, so the second series has been removed.

## Source data

The raw result files are on the `quail-results` Modal volume.

- Quail 4B, function call `fc-01M0YBGJBC4KSB3JZF5089YRDQ`:
  `/results/benchmarks/quailb/runs/qb_20260826T061917Z_2a6a3ed0/20260826T061917Z-quailb-sf0.1-lf1-qwen3-4b-fp8.json`
- Quail 32B, function call `fc-01M0YBRJZT1JB4X8FX4W6X0SR1`:
  `/results/benchmarks/quailb/runs/qb_20260826T062254Z_9843d222/20260826T062254Z-quailb-sf0.1-lf1-qwen3-32b-fp8.json`
- Stock vLLM 4B, function call `fc-01M0ZVTJ8GHGGMFW9W6Q0PWR1D`:
  `/results/stock_quailb/2026-08-26_202641_qwen3-4b-fp8/summary.json`
- Stock vLLM 32B, function call `fc-01M0ZVTJ5CJ4BG41SGWZV5H7WP`:
  `/results/stock_quailb/2026-08-26_202641_qwen3-32b-fp8/summary.json`
- Preliminary stock vLLM 4B repeat, function call
  `fc-01M0ZT45A3Q7MCHWWA2ERAF2Q6`:
  `/results/stock_quailb/2026-08-26_195812_qwen3-4b-fp8/summary.json`
- Preliminary stock vLLM 32B repeat, function call
  `fc-01M0ZTV9RBSF430PRTJ4EVJFNW`:
  `/results/stock_quailb/2026-08-26_201016_qwen3-32b-fp8/summary.json`
- SoL estimate:
  `/results/sol/sol_quailb_sf0.1.json`

The plot script takes a work directory as its first argument. Its docstring
contains the five `modal volume get` commands needed to rebuild the figure.
No raw experiment data is committed.
