# Shared join prompts and vLLM baselines

Date: 2026-08-26. The runs used sf=0.1, Qwen3 4B fp8 or Qwen3 32B
fp8, and one exact `H100!` request per model copy on Modal.

## Result

Quail was faster than both vLLM baselines on all six join comparisons.
The largest difference was BIO-2 at 4B. Quail took 31.12 seconds,
compared with 187.17 seconds for naive vLLM and 261.38 seconds for
stock vLLM.

Figure: plots/vllm_opbench_vs_quail.png

The plot uses a log scale because the measured times span more than two
orders of magnitude.

| Model | Query | Pairs | Quail (s) | Naive vLLM (s) | Compared with Quail | Stock vLLM (s) | Compared with Quail |
|---|---|---:|---:|---:|---:|---:|---:|
| 4B | BIO-2 | 122,800 | 31.12 | 187.17 | 6.01 times slower | 261.38 | 8.40 times slower |
| 4B | FEV-2 | 5,700 | 1.29 | 2.16 | 1.67 times slower | 3.10 | 2.40 times slower |
| 4B | IMDB-2 | 60,000 | 20.55 | 24.36 | 1.19 times slower | 31.32 | 1.52 times slower |
| 32B | BIO-2 | 122,800 | 181.05 | 302.96 | 1.67 times slower | 265.80 | 1.47 times slower |
| 32B | FEV-2 | 5,700 | 8.47 | 12.27 | 1.45 times slower | 11.20 | 1.32 times slower |
| 32B | IMDB-2 | 60,000 | 139.88 | 187.93 | 1.34 times slower | 155.58 | 1.11 times slower |

For Quail, the table uses the cold pass and excludes model load time. For
both vLLM baselines, the table uses the time inside `llm.generate()`. CPU
prompt building and the Modal call are outside that timer.

## The join prompt fix

The old Quail join prompt put the complete static question after every
partner document. Quail therefore processed the same question again for
every pair. The new order is:

```text
DOCUMENT:
<anchor document>

(The document above is DOCUMENT {0}.)

Evaluate TRUE or FALSE for the following question: <join question>

DOCUMENT {1}:
<partner document>
ANSWER:
```

The new order has the following effects:

- Quail processes the anchor note and complete question once per anchor.
- Each pair adds only the partner label, partner document, and answer cue.
- Quail, stock vLLM, and naive vLLM now use the exact same token IDs for
  every complete prompt.
- Tests compare the exact token IDs when either input is the anchor. A
  separate test covers a join with three inputs.

For `merge_quant`, the suffix is the partner label, the partner document,
and `ANSWER:`. The persistent document KV contains `DOCUMENT:\n` and the
anchor document. The active kept KV during the join also contains the
anchor note and complete question. `merge_quant` does not save suffix KV.

## Planner accounting

The planner now counts the prompt in the same order that the runtime uses.
For a two input join, it computes:

```text
anchors * (shared preamble + anchor document + anchor note + question)
+ pairs * (partner label + partner document + answer cue)
```

The Quail runs matched the predicted fresh token counts exactly:

| Query | Predicted fresh tokens | Measured fresh tokens | Change from the old prompt |
|---|---:|---:|---:|
| BIO-2 | 2,635,599 | 2,635,599 | 76 percent fewer than 10,979,399 |
| IMDB-2 | 2,419,233 | 2,419,233 | 59 percent fewer than 5,944,233 |
| FEV-2 | 145,359 | 145,359 | The evidence input is the anchor |

The vLLM prefix cache processed slightly more fresh tokens because it reuses
complete 16 token blocks. It processed 2,675,146 fresh tokens for BIO-2,
2,438,148 for IMDB-2, and 152,994 for FEV-2.

## Baseline settings

Both baselines use the same driver and the same complete prompt token IDs.
They differ only in how vLLM loads the model weights:

| Configuration | Model weights | Quantization setting |
|---|---|---|
| Naive vLLM | Base Qwen3 checkpoint | vLLM converts weights to fp8 at load time |
| Stock vLLM | Qwen3 FP8 checkpoint | vLLM loads the checkpoint as provided |

Both baselines used these settings:

- The Modal request was `gpu="H100!"`.
- vLLM was version 0.26.0 with CUDA 13.0.1.
- Each model used one GPU and one model copy.
- Prefix caching was enabled with 16 token blocks.
- `max_num_seqs` was 4,096.
- `max_num_batched_tokens` was 25,305.
- The model context limit remained 40,960 tokens. The value 4,096 controls
  the number of admitted requests, not the prompt length.
- Each join submitted its complete cross product in one `llm.generate()`
  call. BIO-2 submitted 122,800 requests, FEV-2 submitted 5,700 requests,
  and IMDB-2 submitted 60,000 requests.

The two saved Quail files have `H100` in an old pricing label. The actual
Modal worker decorator requested `H100!`. Commit `002ce3c` changes the saved
label to `H100!` and makes the baseline reject any other GPU request.

## Predictions

The predictions were recorded before the final runs:

- Naive 4B predicted 180 to 205 seconds for BIO-2, 2 to 3 seconds for
  FEV-2, and 22 to 25 seconds for IMDB-2. All three measurements were
  inside those ranges.
- Naive 32B predicted 260 to 300 seconds for BIO-2, 10 to 13 seconds for
  FEV-2, and 180 to 220 seconds for IMDB-2. BIO-2 took 302.96 seconds,
  which was 2.96 seconds above the range. The other two were inside.
- Stock 4B was expected to be close to naive 4B. The prediction was wrong.
  Stock 4B took 74.21 seconds longer on BIO-2 and 6.96 seconds longer on
  IMDB-2.
- Stock 32B predicted 240 to 300 seconds for BIO-2, 10 to 13 seconds for
  FEV-2, and 160 to 210 seconds for IMDB-2. BIO-2 and FEV-2 were inside
  those ranges. IMDB-2 took 155.58 seconds, which was 4.42 seconds below
  the range.
- The filter prediction was 9 to 12 seconds at 4B and 65 to 75 seconds at
  32B. Naive 4B took 10.04 seconds, stock 4B took 10.27 seconds, and naive
  32B took 71.31 seconds. Stock 32B took 61.19 seconds, which was below the
  predicted range.

## What the measurements mean

The earlier comparison used different join prompt orders. It also made
Quail process the static question once per pair. The corrected comparison
uses the same token IDs for every system, and Quail is faster on every join
tested here.

The 4B stock checkpoint was slower than the base checkpoint converted to
fp8 at load time. The prompt, batch order, and vLLM settings were the same,
so prompt formatting does not explain that difference. This experiment did
not isolate the checkpoint difference further.

The performance reruns did not measure accuracy. The join prompt changed,
so accuracy from the old prompt is not valid for this report. New ground
truth and new accuracy runs are required before reporting accuracy.

## Source data

The raw result files are on the `quail-results` Modal volume:

- Quail 4B, function call `fc-01M0YBGJBC4KSB3JZF5089YRDQ`:
  `/results/benchmarks/quailb/runs/qb_20260826T061917Z_2a6a3ed0/20260826T061917Z-quailb-sf0.1-lf1-qwen3-4b-fp8.json`
- Quail 32B, function call `fc-01M0YBRJZT1JB4X8FX4W6X0SR1`:
  `/results/benchmarks/quailb/runs/qb_20260826T062254Z_9843d222/20260826T062254Z-quailb-sf0.1-lf1-qwen3-32b-fp8.json`
- Naive vLLM 4B, function call `fc-01M0YESS2J9D5Z9VMY43PDNPPV`:
  `/results/vllm_opbench/2026-08-26_071748/summary.json`
- Naive vLLM 32B, function call `fc-01M0YESR04Y51JJWTRCZTN822S`:
  `/results/vllm_opbench/2026-08-26_071907/summary.json`
- Stock vLLM 4B, function call `fc-01M0YESR36G9ZSSQQZ2973HCTB`:
  `/results/vllm_opbench/2026-08-26_071911/summary.json`
- Stock vLLM 32B, function call `fc-01M0YESXBW6PEWVK0MZM5KSJGW`:
  `/results/vllm_opbench/2026-08-26_071944/summary.json`

The plot script takes a work directory as its first argument and contains
the six `modal volume get` commands needed to rebuild the figure. No raw
experiment data is committed.
