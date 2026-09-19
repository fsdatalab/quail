# QUAIL-B on DiffusionGemma 26B-A4B

- Both methods run `diffusion-gemma-26b-a4b-fp8` at sf=0.1 on one H100, on the same
  prompts: the Gemma 4 chat turn with the empty thinking channel
  prefilled, and one canvas row after the answer cue on Quail's side.
  Quail and pipelined stock vLLM share a physical GPU within each
  query family.
- The measured run is `/results/benchmarks/quailb/20260919T054043Z-8c1fa18b/`
  on `quail-results`; reference labels are collection
  `gt_be81cb241d74555dc2da79b5b0662554` (Qwen3 32B FP8 answering).
- Quail reads the TRUE and FALSE logits at its one canvas row and
  compares them. Pipelined stock vLLM denoises the checkpoint's
  256-row canvas per request, since its diffusion sampler takes no
  temperature, min_tokens, or allowed_token_ids; its answer is the
  first TRUE or FALSE word in the generated text, and a request that
  ends its turn without an answer word counts as FALSE. That readout
  is what stock vLLM gives, and it costs the baseline recall.
- Pipelined stock vLLM batches 65,536 tokens and 8 sequences, with
  prefix caching. vLLM caps this model at 8 sequences per step because
  its diffusion sampler holds a [sequences, canvas rows, vocabulary]
  float32 tensor. Quail's chunk budget is 65,536 tokens.
- Quail is faster on 22 of the 22 queries the
  baseline finished.
  The median speedup is 28.62x and the maximum is 259.37x on FEV-2.
  Speedup is pipelined stock vLLM time divided by Quail time.
  Query time excludes startup and result collection.
  GPU cost is query seconds / 3,600 times $3.9492.
- SoL means speed of light: ideal GPU time from arithmetic and memory
  traffic at the hardware's peak rates, with ideal batching, unlimited
  retained KV, every distinct prompt prefix computed once, one canvas
  row per evaluation, and exact reference-label survivors. It is an
  estimate, not a measured backend, and has no accuracy.
- Fresh input tokens count every input position a forward pass
  processes, repeated computation included. KV regret is recomputed
  tokens as a share of fresh tokens. Token throughput is total
  requested input tokens divided by query seconds.
- Pipelined stock vLLM did not finish IMDB-9, IMDB-10, BIO-2, BIO-3, FEV-7, FEV-8, FEV-9, FEV-10 inside the ten hour limit of its query
  family's container, so those baseline values are not
  measured. They show as a cross at the axis in the figures and
  as "Not measured" in the tables, never as zero.
- A prefill-only stock vLLM is possible for this model but is not
  what this run measured: with a one-row canvas, one denoising step,
  and the sequence cap lifted, stock vLLM returns the TRUE and FALSE
  logprobs at the canvas row for 62 of 64 IMDB reviews at 57,708
  prompt tokens per second on that 64-review batch
  (`/results/ablations/`
  `diffusion_gemma_stock_answers_canvas1_lp20_steps1.json`).

[Open the main vector PDF](plots/quailb_diffusion_gemma_main.pdf)

Figure: plots/quailb_diffusion_gemma_main.pdf

## IMDB

[Open the IMDB vector PDF](plots/quailb_diffusion_gemma_imdb.pdf)

Figure: plots/quailb_diffusion_gemma_imdb.pdf

| Query | Input documents by alias and set |
|---|---|
| IMDB-1 | r (reviews) = 5,000 |
| IMDB-2 | r (reviews) = 5,000, a (aspects) = 12 |
| IMDB-3 | r (reviews) = 5,000, a (aspects) = 12 |
| IMDB-4 | r (reviews) = 5,000, a (aspects) = 12 |
| IMDB-5 | r (reviews) = 5,000, a (aspects) = 12 |
| IMDB-6 | r (reviews) = 5,000 |
| IMDB-7 | r (reviews) = 5,000 |
| IMDB-8 | r (reviews) = 5,000, a (aspects) = 12, a2 (aspects) = 12 |
| IMDB-9 | r1 (reviews) = 5,000, a1 (aspects) = 12, r2 (reviews) = 5,000, a2 (aspects) = 12 |
| IMDB-10 | r1 (reviews) = 5,000, a1 (aspects) = 12, r2 (reviews) = 5,000, a2 (aspects) = 12 |

| Query | Method | Seconds | Tokens/second | $/query | $/million input tokens | KV regret (%) |
|---|---|---:|---:|---:|---:|---:|
| IMDB-1 | Quail | 19.66 | 93,254.63 | 0.02157 | 0.011763 | 2.75 |
| IMDB-1 | Pipelined vLLM | 516.30 | 3,551.01 | 0.56638 | 0.308926 | 2.47 |
| IMDB-1 | SoL estimate | 5.845 | 313,654.67 | 0.00641 | 0.003497 | 0 (assumed) |
| IMDB-2 | Quail | 32.38 | 674,201.11 | 0.03552 | 0.001627 | 3.69 |
| IMDB-2 | Pipelined vLLM | 5898.60 | 3,700.99 | 6.47076 | 0.296408 | 15.18 |
| IMDB-2 | SoL estimate | 9.636 | 2,265,573.76 | 0.01057 | 0.000484 | 0 (assumed) |
| IMDB-3 | Quail | 34.84 | 561,045.12 | 0.03822 | 0.001955 | 3.53 |
| IMDB-3 | Pipelined vLLM | 2697.99 | 4,138.50 | 2.95970 | 0.265072 | 27.93 |
| IMDB-3 | SoL estimate | 9.703 | 2,039,949.33 | 0.01064 | 0.000538 | 0 (assumed) |
| IMDB-4 | Quail | 23.59 | 357,650.61 | 0.02588 | 0.003067 | 3.06 |
| IMDB-4 | Pipelined vLLM | 833.72 | 4,859.93 | 0.91459 | 0.225724 | 10.95 |
| IMDB-4 | SoL estimate | 7.003 | 1,140,773.12 | 0.00768 | 0.000962 | 0 (assumed) |
| IMDB-5 | Quail | 22.77 | 315,471.67 | 0.02498 | 0.003477 | 3.00 |
| IMDB-5 | Pipelined vLLM | 648.52 | 4,481.87 | 0.71143 | 0.244764 | 5.83 |
| IMDB-5 | SoL estimate | 6.806 | 988,725.05 | 0.00747 | 0.001110 | 0 (assumed) |
| IMDB-6 | Quail | 19.40 | 122,631.13 | 0.02128 | 0.008946 | 2.78 |
| IMDB-6 | Pipelined vLLM | 559.21 | 3,740.43 | 0.61345 | 0.293281 | 2.89 |
| IMDB-6 | SoL estimate | 5.997 | 392,729.37 | 0.00658 | 0.002793 | 0 (assumed) |
| IMDB-7 | Quail | 19.80 | 140,361.92 | 0.02172 | 0.007816 | 2.81 |
| IMDB-7 | Pipelined vLLM | 568.60 | 3,830.14 | 0.62375 | 0.286413 | 2.96 |
| IMDB-7 | SoL estimate | 6.147 | 444,098.28 | 0.00674 | 0.002470 | 0 (assumed) |
| IMDB-8 | Quail | 42.06 | 820,808.99 | 0.04614 | 0.001336 | 5.93 |
| IMDB-8 | Pipelined vLLM | 9667.69 | 3,606.17 | 10.60546 | 0.304201 | 32.85 |
| IMDB-8 | SoL estimate | 13.439 | 2,958,877.66 | 0.01474 | 0.000371 | 0 (assumed) |
| IMDB-9 | Quail | 74.56 | 755,818.91 | 0.08179 | 0.001451 | 39.01 |
| IMDB-9 | Pipelined vLLM | Not measured | | | | |
| IMDB-9 | SoL estimate | 19.157 | 3,215,322.47 | 0.02102 | 0.000341 | 0 (assumed) |
| IMDB-10 | Quail | 76.36 | 708,093.74 | 0.08377 | 0.001549 | 38.46 |
| IMDB-10 | Pipelined vLLM | Not measured | | | | |
| IMDB-10 | SoL estimate | 18.364 | 3,243,181.20 | 0.02015 | 0.000338 | 0 (assumed) |

Correctness against saved reference labels:

| Query | Method | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|
| IMDB-1 | Quail | 94.22 | 97.64 | 95.08 |
| IMDB-1 | Pipelined vLLM | 53.84 | 94.538 | 44.955 |
| IMDB-2 | Quail | 88.89 | 90.161 | 74.296 |
| IMDB-2 | Pipelined vLLM | 89.30 | 83.595 | 83.875 |
| IMDB-3 | Quail | 89.39 | 87.683 | 73.004 |
| IMDB-3 | Pipelined vLLM | 82.19 | 79.908 | 39.166 |
| IMDB-4 | Quail | 90.20 | 78.029 | 71.265 |
| IMDB-4 | Pipelined vLLM | 82.44 | 69.279 | 18.259 |
| IMDB-5 | Quail | 91.29 | 77.703 | 72.447 |
| IMDB-5 | Pipelined vLLM | 80.41 | 68.05 | 8.6316 |
| IMDB-6 | Quail | 94.28 | 85.246 | 85.659 |
| IMDB-6 | Pipelined vLLM | 79.49 | 74.803 | 18.411 |
| IMDB-7 | Quail | 94.71 | 84.265 | 85.268 |
| IMDB-7 | Pipelined vLLM | 78.67 | 67.742 | 6.25 |
| IMDB-8 | Quail | 84.49 | 51.964 | 43.471 |
| IMDB-8 | Pipelined vLLM | 83.63 | 46.821 | 48.154 |
| IMDB-9 | Quail | 86.21 | 48.277 | 35.363 |
| IMDB-9 | Pipelined vLLM | Not measured | | |
| IMDB-10 | Quail | 86.24 | 47.132 | 34.919 |
| IMDB-10 | Pipelined vLLM | Not measured | | |

## BIO

[Open the BIO vector PDF](plots/quailb_diffusion_gemma_bio.pdf)

Figure: plots/quailb_diffusion_gemma_bio.pdf

| Query | Input documents by alias and set |
|---|---|
| BIO-1 | r (reports) = 500 |
| BIO-2 | r (reports) = 500, m (terms) = 1,127 |
| BIO-3 | r (reports) = 500, m (terms) = 1,127 |

| Query | Method | Seconds | Tokens/second | $/query | $/million input tokens | KV regret (%) |
|---|---|---:|---:|---:|---:|---:|
| BIO-1 | Quail | 24.88 | 80,297.23 | 0.02729 | 0.013662 | 0.29 |
| BIO-1 | Pipelined vLLM | 77.11 | 25,908.38 | 0.08459 | 0.042342 | 0.27 |
| BIO-1 | SoL estimate | 9.027 | 221,320.94 | 0.00990 | 0.004957 | 0 (assumed) |
| BIO-2 | Quail | 217.06 | 10,385,388.67 | 0.23811 | 0.000106 | 3.83 |
| BIO-2 | Pipelined vLLM | Not measured | | | | |
| BIO-2 | SoL estimate | 74.700 | 30,177,256.48 | 0.08195 | 0.000036 | 0 (assumed) |
| BIO-3 | Quail | 184.28 | 10,256,614.08 | 0.20216 | 0.000107 | 3.74 |
| BIO-3 | Pipelined vLLM | Not measured | | | | |
| BIO-3 | SoL estimate | 49.866 | 26,833,887.64 | 0.05470 | 0.000041 | 0 (assumed) |

Correctness against saved reference labels:

| Query | Method | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|
| BIO-1 | Quail | 79.20 | 75.779 | 99.06 |
| BIO-1 | Pipelined vLLM | 72.40 | 73.879 | 87.774 |
| BIO-2 | Quail | 96.30 | 54.026 | 51.461 |
| BIO-2 | Pipelined vLLM | Not measured | | |
| BIO-3 | Quail | 96.01 | 48.15 | 51.886 |
| BIO-3 | Pipelined vLLM | Not measured | | |

## FEV

[Open the FEV vector PDF](plots/quailb_diffusion_gemma_fev.pdf)

Figure: plots/quailb_diffusion_gemma_fev.pdf

| Query | Input documents by alias and set |
|---|---|
| FEV-1 | c (claims) = 500 |
| FEV-2 | c (claims) = 500, e (evidence) = 287 |
| FEV-3 | c (claims) = 500, e (evidence) = 287 |
| FEV-4 | c (claims) = 500, e (evidence) = 287 |
| FEV-5 | c (claims) = 500, e (evidence) = 287 |
| FEV-6 | c (claims) = 500, e (evidence) = 287 |
| FEV-7 | c (claims) = 500, e (evidence) = 287, e2 (evidence) = 287 |
| FEV-8 | c1 (claims) = 500, e1 (evidence) = 287, c2 (claims) = 500, e2 (evidence) = 287 |
| FEV-9 | c1 (claims) = 500, e1 (evidence) = 287, c2 (claims) = 500, e2 (evidence) = 287 |
| FEV-10 | c (claims) = 500, e (evidence) = 287 |

| Query | Method | Seconds | Tokens/second | $/query | $/million input tokens | KV regret (%) |
|---|---|---:|---:|---:|---:|---:|
| FEV-1 | Quail | 0.46 | 88,178.26 | 0.00050 | 0.012441 | 10.42 |
| FEV-1 | Pipelined vLLM | 54.29 | 747.14 | 0.05956 | 1.468274 | 9.32 |
| FEV-1 | SoL estimate | 0.126 | 321,909.57 | 0.00014 | 0.003408 | 0 (assumed) |
| FEV-2 | Quail | 50.99 | 1,419,499.78 | 0.05594 | 0.000773 | 3.27 |
| FEV-2 | Pipelined vLLM | 13225.45 | 5,472.80 | 14.50832 | 0.200446 | 22.89 |
| FEV-2 | SoL estimate | 14.986 | 4,829,729.77 | 0.01644 | 0.000227 | 0 (assumed) |
| FEV-3 | Quail | 30.80 | 1,373,748.54 | 0.03379 | 0.000799 | 3.36 |
| FEV-3 | Pipelined vLLM | 2244.65 | 5,370.95 | 2.46238 | 0.204247 | 20.62 |
| FEV-3 | SoL estimate | 9.095 | 4,683,297.34 | 0.00998 | 0.000234 | 0 (assumed) |
| FEV-4 | Quail | 6.96 | 785,081.47 | 0.00764 | 0.001397 | 3.17 |
| FEV-4 | Pipelined vLLM | 100.28 | 3,317.35 | 0.11001 | 0.330685 | 4.69 |
| FEV-4 | SoL estimate | 1.857 | 2,940,745.37 | 0.00204 | 0.000373 | 0 (assumed) |
| FEV-5 | Quail | 17.13 | 1,332,827.32 | 0.01879 | 0.000823 | 3.39 |
| FEV-5 | Pipelined vLLM | 460.04 | 5,852.24 | 0.50466 | 0.187450 | 11.68 |
| FEV-5 | SoL estimate | 5.291 | 4,490,501.84 | 0.00580 | 0.000244 | 0 (assumed) |
| FEV-6 | Quail | 4.36 | 709,186.70 | 0.00478 | 0.001547 | 3.32 |
| FEV-6 | Pipelined vLLM | 85.35 | 2,987.19 | 0.09363 | 0.367234 | 4.18 |
| FEV-6 | SoL estimate | 1.328 | 2,401,408.69 | 0.00146 | 0.000457 | 0 (assumed) |
| FEV-7 | Quail | 74.83 | 1,417,339.66 | 0.08209 | 0.000774 | 5.28 |
| FEV-7 | Pipelined vLLM | Not measured | | | | |
| FEV-7 | SoL estimate | 24.937 | 4,911,742.34 | 0.02736 | 0.000223 | 0 (assumed) |
| FEV-8 | Quail | 101.15 | 1,435,288.24 | 0.11096 | 0.000764 | 14.27 |
| FEV-8 | Pipelined vLLM | Not measured | | | | |
| FEV-8 | SoL estimate | 39.515 | 4,931,364.63 | 0.04335 | 0.000222 | 0 (assumed) |
| FEV-9 | Quail | 33.08 | 1,325,166.29 | 0.03629 | 0.000828 | 16.74 |
| FEV-9 | Pipelined vLLM | Not measured | | | | |
| FEV-9 | SoL estimate | 13.174 | 4,778,883.94 | 0.01445 | 0.000230 | 0 (assumed) |
| FEV-10 | Quail | 2.19 | 124,957.99 | 0.00240 | 0.008779 | 3.84 |
| FEV-10 | Pipelined vLLM | Not measured | | | | |
| FEV-10 | SoL estimate | 0.639 | 429,898.55 | 0.00070 | 0.002552 | 0 (assumed) |

Correctness against saved reference labels:

| Query | Method | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|
| FEV-1 | Quail | 97.20 | 97.945 | 97.279 |
| FEV-1 | Pipelined vLLM | 53.80 | 94.366 | 22.789 |
| FEV-2 | Quail | 94.88 | 88.509 | 3.751 |
| FEV-2 | Pipelined vLLM | 94.88 | 86.087 | 3.9089 |
| FEV-3 | Quail | 94.18 | 93.252 | 2.9383 |
| FEV-3 | Pipelined vLLM | 93.37 | 87.273 | 0.92789 |
| FEV-4 | Quail | 99.05 | 85.714 | 10.435 |
| FEV-4 | Pipelined vLLM | 94.87 | 100 | 0.86957 |
| FEV-5 | Quail | 93.87 | 94.776 | 4.3036 |
| FEV-5 | Pipelined vLLM | 85.80 | 81.25 | 0.44053 |
| FEV-6 | Quail | 98.94 | 100 | 15.714 |
| FEV-6 | Pipelined vLLM | 83.17 | 100 | 1.4286 |
| FEV-7 | Quail | 93.48 | 42.169 | 0.0098518 |
| FEV-7 | Pipelined vLLM | Not measured | | |
| FEV-8 | Quail | 90.17 | 36.697 | 0.00035541 |
| FEV-8 | Pipelined vLLM | Not measured | | |
| FEV-9 | Quail | 89.08 | 51.724 | 0.00071323 |
| FEV-9 | Pipelined vLLM | Not measured | | |
| FEV-10 | Quail | 97.06 | 100 | 92.623 |
| FEV-10 | Pipelined vLLM | Not measured | | |

## LEP

[Open the LEP vector PDF](plots/quailb_diffusion_gemma_lep.pdf)

Figure: plots/quailb_diffusion_gemma_lep.pdf

| Query | Input documents by alias and set |
|---|---|
| LEP-1 | d (citation_contexts) = 500 |
| LEP-2 | d (citation_contexts) = 500, s (citation_passages) = 433 |
| LEP-3 | d (citation_contexts) = 500, s (citation_passages) = 433 |
| LEP-4 | d (citation_contexts) = 500, s (citation_passages) = 433 |
| LEP-5 | d (citation_contexts) = 500, s (citation_passages) = 433 |

| Query | Method | Seconds | Tokens/second | $/query | $/million input tokens | KV regret (%) |
|---|---|---:|---:|---:|---:|---:|
| LEP-1 | Quail | 1.93 | 73,073.58 | 0.00212 | 0.015012 | 3.32 |
| LEP-1 | Pipelined vLLM | 55.06 | 2,561.42 | 0.06040 | 0.428277 | 2.91 |
| LEP-1 | SoL estimate | 0.446 | 316,314.12 | 0.00049 | 0.003468 | 0 (assumed) |
| LEP-2 | Quail | 188.33 | 381,019.25 | 0.20660 | 0.002879 | 1.31 |
| LEP-2 | Pipelined vLLM | 20566.09 | 3,489.11 | 22.56100 | 0.314407 | 9.51 |
| LEP-2 | SoL estimate | 55.216 | 1,299,582.69 | 0.06057 | 0.000844 | 0 (assumed) |
| LEP-3 | Quail | 9.49 | 475,554.58 | 0.01041 | 0.002307 | 1.64 |
| LEP-3 | Pipelined vLLM | 795.82 | 4,945.00 | 0.87301 | 0.221840 | 9.25 |
| LEP-3 | SoL estimate | 2.125 | 1,485,419.30 | 0.00233 | 0.000739 | 0 (assumed) |
| LEP-4 | Quail | 8.74 | 447,511.78 | 0.00959 | 0.002451 | 1.68 |
| LEP-4 | Pipelined vLLM | 218.90 | 4,270.40 | 0.24013 | 0.256885 | 2.61 |
| LEP-4 | SoL estimate | 1.119 | 1,183,673.07 | 0.00123 | 0.000927 | 0 (assumed) |
| LEP-5 | Quail | 7.28 | 397,383.38 | 0.00799 | 0.002761 | 2.23 |
| LEP-5 | Pipelined vLLM | 191.08 | 3,864.17 | 0.20961 | 0.283890 | 5.80 |
| LEP-5 | SoL estimate | 1.152 | 1,013,814.30 | 0.00126 | 0.001082 | 0 (assumed) |

Correctness against saved reference labels:

| Query | Method | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|
| LEP-1 | Quail | 98.00 | 61.905 | 86.667 |
| LEP-1 | Pipelined vLLM | 96.40 | 42.105 | 53.333 |
| LEP-2 | Quail | 99.34 | 14.947 | 39.4 |
| LEP-2 | Pipelined vLLM | 98.67 | 9.4218 | 55.4 |
| LEP-3 | Quail | 99.56 | 10.526 | 13.333 |
| LEP-3 | Pipelined vLLM | 99.14 | 4.3478 | 13.333 |
| LEP-4 | Quail | 99.45 | 5.5556 | 16.667 |
| LEP-4 | Pipelined vLLM | 98.36 | 0 | 0 |
| LEP-5 | Quail | 98.26 | 11.111 | 25 |
| LEP-5 | Pipelined vLLM | 84.25 | 0 | 0 |

## AGENT

[Open the AGENT vector PDF](plots/quailb_diffusion_gemma_agent.pdf)

Figure: plots/quailb_diffusion_gemma_agent.pdf

| Query | Input documents by alias and set |
|---|---|
| AGENT-1 | t (agent_traces) = 1,772 |
| AGENT-2 | t (agent_traces) = 1,772 |

| Query | Method | Seconds | Tokens/second | $/query | $/million input tokens | KV regret (%) |
|---|---|---:|---:|---:|---:|---:|
| AGENT-1 | Quail | 279.53 | 68,392.63 | 0.30664 | 0.016040 | 68.23 |
| AGENT-1 | Pipelined vLLM | 440.71 | 43,379.53 | 0.48346 | 0.025288 | 0.78 |
| AGENT-1 | SoL estimate | 46.622 | 410,055.89 | 0.05114 | 0.002675 | 0 (assumed) |
| AGENT-2 | Quail | 280.54 | 68,298.00 | 0.30775 | 0.016062 | 68.08 |
| AGENT-2 | Pipelined vLLM | 404.97 | 47,312.94 | 0.44425 | 0.023186 | 0.77 |
| AGENT-2 | SoL estimate | 46.983 | 407,811.34 | 0.05154 | 0.002690 | 0 (assumed) |

Correctness against saved reference labels:

| Query | Method | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|
| AGENT-1 | Quail | 47.18 | 38.964 | 95.23 |
| AGENT-1 | Pipelined vLLM | 48.53 | 38.537 | 84.046 |
| AGENT-2 | Quail | 93.34 | 81.903 | 99.62 |
| AGENT-2 | Pipelined vLLM | 91.14 | 87 | 82.543 |
