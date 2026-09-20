# QUAIL-B on DiffusionGemma 26B-A4B

- Both methods run `diffusion-gemma-26b-a4b-fp8` at sf=0.1 on one H100, on the same
  prompts: the Gemma 4 chat turn with the empty thinking channel
  prefilled, and one canvas row after the answer cue on Quail's side.
  Quail and pipelined stock vLLM share a physical GPU within each
  query family.
- The measured run is `/results/benchmarks/quailb/20260919T220437Z-1550de75/`
  on `quail-results`; reference labels are collection
  `gt_be81cb241d74555dc2da79b5b0662554` (Qwen3 32B FP8 answering).
- Quail reads the TRUE and FALSE logits at its one canvas row and
  compares them. Pipelined stock vLLM runs the same one-row canvas
  with one denoising step, so a request is one prefill pass plus
  that row; its diffusion sampler takes no temperature, min_tokens,
  or allowed_token_ids, so the answer is read by ranking TRUE
  against FALSE in the top 500 logprobs at the canvas row (the words
  rank within 500 on 511 of 512 probed reviews,
  `/results/ablations/diffusion_gemma_readout_probe_k5000.json`). A
  request with neither word in its 500 falls back to its generated
  text, then counts as FALSE.
- The canvas row's input token is a random draw: vLLM draws it per
  request, Quail fixes one draw. On 512 IMDB reviews under four seeds,
  53 reviews change their answer and accuracy against the labels
  ranges 0.904 to 0.941 (`/results/ablations/`
  `diffusion_gemma_canvas_seeds.json`), so a few percent of the two
  methods' answers differ for that reason.
- Pipelined stock vLLM batches 65,536 tokens and 127 sequences, with
  prefix caching. vLLM caps this model at 8 sequences per step when
  the setting is 128 or more, sized for its 256-row canvas, so the
  baseline stays just under that trigger. Quail's chunk budget is
  65,536 tokens with no sequence cap.
- Quail is faster on 28 of the 30 queries the
  baseline finished.
  The median speedup is 11.73x and the maximum is 36.80x on FEV-1.
  Speedup is pipelined stock vLLM time divided by Quail time.
  Query time excludes startup and result collection.
  GPU cost is query seconds / 3,600 times $3.9492.
- Quail is slower on AGENT-1, AGENT-2: on AGENT-1 Quail computes 19,119,565 fresh tokens against the baseline's 6,122,049; on AGENT-2 Quail computes 19,162,093 fresh tokens against the baseline's 6,164,577. The agent
  corpus's traces share prefixes (every later turn's trace
  starts with the earlier turn's), which vLLM's prefix cache
  reuses and Quail's filter path computes again for each
  document.
- On the 30 queries both methods finished, they gave
  the same TRUE or FALSE on 91.30 percent of the
  2,675,597 predicate evaluations both made (per query below).
- SoL means speed of light: ideal GPU time from arithmetic and memory
  traffic at the hardware's peak rates, with ideal batching, unlimited
  retained KV, every distinct prompt prefix computed once, one canvas
  row per evaluation, and exact reference-label survivors. It is an
  estimate, not a measured backend, and has no accuracy.
- Fresh input tokens count every input position a forward pass
  processes, repeated computation included. KV regret is recomputed
  tokens as a share of fresh tokens. Token throughput is total
  requested input tokens divided by query seconds.
- The baseline values of BIO-3, FEV-8, FEV-9, FEV-10 come from the rerun `20260920T013141Z-5124c19e` of those queries: Modal restarted the measured run's orchestrator before their containers finished.
- The baseline values of BIO-1, AGENT-1, AGENT-2 come from the rerun `20260920T032348Z-1e56c382` of those queries: the baseline engine now runs vLLM's engine core in-process; in its own process, the step loop's canvas-row logprobs came back unreliably on long prompts and the baseline's AGENT-1 answers drifted toward noise.

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
| IMDB-1 | Quail | 19.94 | 91,945.14 | 0.02187 | 0.011931 | 2.75 |
| IMDB-1 | Pipelined vLLM | 67.69 | 27,085.03 | 0.07426 | 0.040502 | 2.47 |
| IMDB-1 | SoL estimate | 5.845 | 313,654.67 | 0.00641 | 0.003497 | 0 (assumed) |
| IMDB-2 | Quail | 32.41 | 673,577.04 | 0.03555 | 0.001629 | 3.69 |
| IMDB-2 | Pipelined vLLM | 321.90 | 67,818.06 | 0.35312 | 0.016176 | 15.18 |
| IMDB-2 | SoL estimate | 9.636 | 2,265,573.76 | 0.01057 | 0.000484 | 0 (assumed) |
| IMDB-3 | Quail | 34.79 | 561,851.45 | 0.03816 | 0.001952 | 3.53 |
| IMDB-3 | Pipelined vLLM | 324.54 | 62,976.88 | 0.35602 | 0.017419 | 37.29 |
| IMDB-3 | SoL estimate | 9.703 | 2,039,949.33 | 0.01064 | 0.000538 | 0 (assumed) |
| IMDB-4 | Quail | 23.64 | 356,894.16 | 0.02593 | 0.003074 | 3.06 |
| IMDB-4 | Pipelined vLLM | 160.88 | 64,138.38 | 0.17649 | 0.017104 | 25.83 |
| IMDB-4 | SoL estimate | 7.003 | 1,140,773.12 | 0.00768 | 0.000962 | 0 (assumed) |
| IMDB-5 | Quail | 22.86 | 314,229.66 | 0.02508 | 0.003491 | 3.00 |
| IMDB-5 | Pipelined vLLM | 145.26 | 59,976.97 | 0.15935 | 0.018290 | 21.80 |
| IMDB-5 | SoL estimate | 6.806 | 988,725.05 | 0.00747 | 0.001110 | 0 (assumed) |
| IMDB-6 | Quail | 19.46 | 122,253.03 | 0.02135 | 0.008973 | 2.78 |
| IMDB-6 | Pipelined vLLM | 74.91 | 33,689.87 | 0.08218 | 0.032562 | 3.66 |
| IMDB-6 | SoL estimate | 5.997 | 392,729.37 | 0.00658 | 0.002793 | 0 (assumed) |
| IMDB-7 | Quail | 19.84 | 140,078.93 | 0.02176 | 0.007831 | 2.81 |
| IMDB-7 | Pipelined vLLM | 83.98 | 36,184.51 | 0.09213 | 0.030317 | 4.35 |
| IMDB-7 | SoL estimate | 6.147 | 444,098.28 | 0.00674 | 0.002470 | 0 (assumed) |
| IMDB-8 | Quail | 42.08 | 820,418.87 | 0.04616 | 0.001337 | 5.93 |
| IMDB-8 | Pipelined vLLM | 582.07 | 69,431.05 | 0.63853 | 0.015800 | 37.03 |
| IMDB-8 | SoL estimate | 13.439 | 2,958,877.66 | 0.01474 | 0.000371 | 0 (assumed) |
| IMDB-9 | Quail | 75.11 | 750,284.36 | 0.08240 | 0.001462 | 39.01 |
| IMDB-9 | Pipelined vLLM | 903.74 | 69,232.26 | 0.99140 | 0.015845 | 57.19 |
| IMDB-9 | SoL estimate | 19.157 | 3,215,322.47 | 0.02102 | 0.000341 | 0 (assumed) |
| IMDB-10 | Quail | 76.58 | 706,059.52 | 0.08401 | 0.001554 | 38.46 |
| IMDB-10 | Pipelined vLLM | 905.30 | 67,649.25 | 0.99311 | 0.016216 | 60.42 |
| IMDB-10 | SoL estimate | 18.364 | 3,243,181.20 | 0.02015 | 0.000338 | 0 (assumed) |

Correctness against saved reference labels:

| Query | Method | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|
| IMDB-1 | Quail | 94.22 | 97.64 | 95.08 |
| IMDB-1 | Pipelined vLLM | 92.76 | 94.092 | 97.053 |
| IMDB-2 | Quail | 88.89 | 90.161 | 74.296 |
| IMDB-2 | Pipelined vLLM | 84.41 | 72.927 | 83.551 |
| IMDB-3 | Quail | 89.39 | 87.689 | 72.998 |
| IMDB-3 | Pipelined vLLM | 84.99 | 68.571 | 82.978 |
| IMDB-4 | Quail | 90.20 | 78.029 | 71.265 |
| IMDB-4 | Pipelined vLLM | 86.00 | 55.872 | 82.571 |
| IMDB-5 | Quail | 91.29 | 77.703 | 72.447 |
| IMDB-5 | Pipelined vLLM | 87.43 | 55.468 | 81.816 |
| IMDB-6 | Quail | 94.28 | 85.246 | 85.659 |
| IMDB-6 | Pipelined vLLM | 90.68 | 69.365 | 92.151 |
| IMDB-7 | Quail | 94.71 | 84.265 | 85.268 |
| IMDB-7 | Pipelined vLLM | 91.79 | 67.857 | 90.476 |
| IMDB-8 | Quail | 84.49 | 51.964 | 43.471 |
| IMDB-8 | Pipelined vLLM | 75.43 | 29.229 | 52.116 |
| IMDB-9 | Quail | 86.21 | 48.277 | 35.363 |
| IMDB-9 | Pipelined vLLM | 78.59 | 23.926 | 45.786 |
| IMDB-10 | Quail | 86.24 | 47.132 | 34.919 |
| IMDB-10 | Pipelined vLLM | 78.70 | 22.706 | 45.719 |

Answers the two methods gave each other:

| Query | Evaluations both made | Same answer (%) |
|---|---:|---:|
| IMDB-1 | 5,000 | 94.94 |
| IMDB-2 | 60,000 | 87.81 |
| IMDB-3 | 51,620 | 88.08 |
| IMDB-4 | 18,388 | 89.34 |
| IMDB-5 | 14,985 | 90.67 |
| IMDB-6 | 6,184 | 92.98 |
| IMDB-7 | 6,935 | 94.07 |
| IMDB-8 | 93,012 | 83.92 |
| IMDB-9 | 152,976 | 85.32 |
| IMDB-10 | 144,680 | 85.39 |

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
| BIO-1 | Quail | 24.76 | 80,686.39 | 0.02716 | 0.013596 | 0.29 |
| BIO-1 | Pipelined vLLM | 36.05 | 55,417.34 | 0.03955 | 0.019795 | 0.27 |
| BIO-1 | SoL estimate | 9.027 | 221,320.94 | 0.00990 | 0.004957 | 0 (assumed) |
| BIO-2 | Quail | 216.22 | 10,425,735.20 | 0.23719 | 0.000105 | 3.83 |
| BIO-2 | Pipelined vLLM | 4640.74 | 485,752.80 | 5.09089 | 0.002258 | 24.98 |
| BIO-2 | SoL estimate | 74.700 | 30,177,256.48 | 0.08195 | 0.000036 | 0 (assumed) |
| BIO-3 | Quail | 182.95 | 10,331,177.06 | 0.20070 | 0.000106 | 3.74 |
| BIO-3 | Pipelined vLLM | 3959.62 | 528,396.05 | 4.34370 | 0.002076 | 31.59 |
| BIO-3 | SoL estimate | 49.866 | 26,833,887.64 | 0.05470 | 0.000041 | 0 (assumed) |

Correctness against saved reference labels:

| Query | Method | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|
| BIO-1 | Quail | 79.20 | 75.779 | 99.06 |
| BIO-1 | Pipelined vLLM | 62.20 | 64.009 | 93.103 |
| BIO-2 | Quail | 96.30 | 54.026 | 51.461 |
| BIO-2 | Pipelined vLLM | 90.91 | 24.828 | 62.603 |
| BIO-3 | Quail | 96.01 | 48.15 | 51.886 |
| BIO-3 | Pipelined vLLM | 90.94 | 20.261 | 59.462 |

Answers the two methods gave each other:

| Query | Evaluations both made | Same answer (%) |
|---|---:|---:|
| BIO-1 | 500 | 80.20 |
| BIO-2 | 563,500 | 93.18 |
| BIO-3 | 445,665 | 93.08 |

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
| FEV-1 | Pipelined vLLM | 16.93 | 2,395.87 | 0.01857 | 0.457872 | 9.32 |
| FEV-1 | SoL estimate | 0.126 | 321,909.57 | 0.00014 | 0.003408 | 0 (assumed) |
| FEV-2 | Quail | 51.18 | 1,414,230.05 | 0.05614 | 0.000776 | 3.27 |
| FEV-2 | Pipelined vLLM | 807.60 | 89,623.94 | 0.88594 | 0.012240 | 22.89 |
| FEV-2 | SoL estimate | 14.986 | 4,829,729.77 | 0.01644 | 0.000227 | 0 (assumed) |
| FEV-3 | Quail | 30.64 | 1,380,922.16 | 0.03361 | 0.000794 | 3.36 |
| FEV-3 | Pipelined vLLM | 503.30 | 89,549.23 | 0.55212 | 0.012250 | 22.74 |
| FEV-3 | SoL estimate | 9.095 | 4,683,297.34 | 0.00998 | 0.000234 | 0 (assumed) |
| FEV-4 | Quail | 6.31 | 865,953.57 | 0.00692 | 0.001267 | 3.17 |
| FEV-4 | Pipelined vLLM | 73.48 | 84,252.33 | 0.08061 | 0.013020 | 16.79 |
| FEV-4 | SoL estimate | 1.857 | 2,940,745.37 | 0.00204 | 0.000373 | 0 (assumed) |
| FEV-5 | Quail | 17.09 | 1,335,946.87 | 0.01875 | 0.000821 | 3.39 |
| FEV-5 | Pipelined vLLM | 280.60 | 89,548.23 | 0.30782 | 0.012250 | 23.64 |
| FEV-5 | SoL estimate | 5.291 | 4,490,501.84 | 0.00580 | 0.000244 | 0 (assumed) |
| FEV-6 | Quail | 4.34 | 712,454.84 | 0.00476 | 0.001540 | 3.32 |
| FEV-6 | Pipelined vLLM | 45.27 | 74,732.78 | 0.04966 | 0.014679 | 21.82 |
| FEV-6 | SoL estimate | 1.328 | 2,401,408.69 | 0.00146 | 0.000457 | 0 (assumed) |
| FEV-7 | Quail | 74.61 | 1,421,518.93 | 0.08185 | 0.000772 | 5.28 |
| FEV-7 | Pipelined vLLM | 1596.81 | 90,835.85 | 1.75170 | 0.012077 | 23.64 |
| FEV-7 | SoL estimate | 24.937 | 4,911,742.34 | 0.02736 | 0.000223 | 0 (assumed) |
| FEV-8 | Quail | 100.37 | 1,446,442.21 | 0.11011 | 0.000758 | 14.27 |
| FEV-8 | Pipelined vLLM | 2371.11 | 91,698.77 | 2.60111 | 0.011963 | 49.19 |
| FEV-8 | SoL estimate | 39.515 | 4,931,364.63 | 0.04335 | 0.000222 | 0 (assumed) |
| FEV-9 | Quail | 33.23 | 1,319,184.50 | 0.03645 | 0.000832 | 16.74 |
| FEV-9 | Pipelined vLLM | 826.44 | 92,571.95 | 0.90660 | 0.011850 | 47.21 |
| FEV-9 | SoL estimate | 13.174 | 4,778,883.94 | 0.01445 | 0.000230 | 0 (assumed) |
| FEV-10 | Quail | 2.20 | 124,390.00 | 0.00241 | 0.008819 | 3.84 |
| FEV-10 | Pipelined vLLM | 10.45 | 26,403.25 | 0.01146 | 0.041548 | 3.41 |
| FEV-10 | SoL estimate | 0.639 | 429,898.55 | 0.00070 | 0.002552 | 0 (assumed) |

Correctness against saved reference labels:

| Query | Method | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|
| FEV-1 | Quail | 97.20 | 97.945 | 97.279 |
| FEV-1 | Pipelined vLLM | 93.80 | 91.746 | 98.299 |
| FEV-2 | Quail | 94.88 | 88.509 | 3.751 |
| FEV-2 | Pipelined vLLM | 88.55 | 10.554 | 15.544 |
| FEV-3 | Quail | 94.18 | 93.252 | 2.9383 |
| FEV-3 | Pipelined vLLM | 87.87 | 9.889 | 13.435 |
| FEV-4 | Quail | 99.05 | 85.714 | 10.435 |
| FEV-4 | Pipelined vLLM | 91.80 | 2.2364 | 18.261 |
| FEV-5 | Quail | 93.87 | 94.776 | 4.3036 |
| FEV-5 | Pipelined vLLM | 87.85 | 11.201 | 14.571 |
| FEV-6 | Quail | 98.94 | 100 | 15.714 |
| FEV-6 | Pipelined vLLM | 92.29 | 2.6639 | 18.571 |
| FEV-7 | Quail | 93.48 | 42.169 | 0.0098518 |
| FEV-7 | Pipelined vLLM | 82.24 | 1.6704 | 1.8935 |
| FEV-8 | Quail | 90.17 | 36.697 | 0.00035541 |
| FEV-8 | Pipelined vLLM | 84.24 | 0.22035 | 0.32171 |
| FEV-9 | Quail | 89.08 | 51.724 | 0.00071323 |
| FEV-9 | Pipelined vLLM | 83.67 | 0.29713 | 0.28353 |
| FEV-10 | Quail | 96.85 | 99.115 | 91.803 |
| FEV-10 | Pipelined vLLM | 94.14 | 90.984 | 90.984 |

Answers the two methods gave each other:

| Query | Evaluations both made | Same answer (%) |
|---|---:|---:|
| FEV-1 | 500 | 95.00 |
| FEV-2 | 143,500 | 92.41 |
| FEV-3 | 84,017 | 92.28 |
| FEV-4 | 11,184 | 92.54 |
| FEV-5 | 44,587 | 92.43 |
| FEV-6 | 6,252 | 93.27 |
| FEV-7 | 210,084 | 91.21 |
| FEV-8 | 282,839 | 90.40 |
| FEV-9 | 84,707 | 90.42 |
| FEV-10 | 951 | 94.74 |

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
| LEP-1 | Quail | 1.70 | 82,960.00 | 0.00186 | 0.013223 | 3.32 |
| LEP-1 | Pipelined vLLM | 18.71 | 7,537.79 | 0.02052 | 0.145533 | 2.91 |
| LEP-1 | SoL estimate | 0.446 | 316,314.12 | 0.00049 | 0.003468 | 0 (assumed) |
| LEP-2 | Quail | 185.10 | 387,668.05 | 0.20305 | 0.002830 | 1.31 |
| LEP-2 | Pipelined vLLM | 1300.44 | 55,179.29 | 1.42658 | 0.019881 | 9.51 |
| LEP-2 | SoL estimate | 55.216 | 1,299,582.69 | 0.06057 | 0.000844 | 0 (assumed) |
| LEP-3 | Quail | 9.34 | 483,191.97 | 0.01025 | 0.002270 | 1.64 |
| LEP-3 | Pipelined vLLM | 248.60 | 67,384.43 | 0.27271 | 0.016280 | 10.23 |
| LEP-3 | SoL estimate | 2.125 | 1,485,419.30 | 0.00233 | 0.000739 | 0 (assumed) |
| LEP-4 | Quail | 9.48 | 412,579.43 | 0.01040 | 0.002659 | 1.68 |
| LEP-4 | Pipelined vLLM | 208.25 | 62,792.86 | 0.22845 | 0.017470 | 11.26 |
| LEP-4 | SoL estimate | 1.119 | 1,183,673.07 | 0.00123 | 0.000927 | 0 (assumed) |
| LEP-5 | Quail | 7.17 | 403,479.92 | 0.00787 | 0.002719 | 2.23 |
| LEP-5 | Pipelined vLLM | 176.18 | 59,366.13 | 0.19327 | 0.018479 | 9.85 |
| LEP-5 | SoL estimate | 1.152 | 1,013,814.30 | 0.00126 | 0.001082 | 0 (assumed) |

Correctness against saved reference labels:

| Query | Method | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|
| LEP-1 | Quail | 98.00 | 61.905 | 86.667 |
| LEP-1 | Pipelined vLLM | 85.80 | 15.854 | 86.667 |
| LEP-2 | Quail | 99.34 | 14.947 | 39.4 |
| LEP-2 | Pipelined vLLM | 95.07 | 2.5918 | 55.6 |
| LEP-3 | Quail | 99.56 | 10.526 | 13.333 |
| LEP-3 | Pipelined vLLM | 95.64 | 0.34364 | 40 |
| LEP-4 | Quail | 99.45 | 5.5556 | 16.667 |
| LEP-4 | Pipelined vLLM | 95.44 | 0.28129 | 66.667 |
| LEP-5 | Quail | 98.26 | 11.111 | 25 |
| LEP-5 | Pipelined vLLM | 95.39 | 0.17301 | 50 |

Answers the two methods gave each other:

| Query | Evaluations both made | Same answer (%) |
|---|---:|---:|
| LEP-1 | 500 | 87.00 |
| LEP-2 | 216,500 | 95.53 |
| LEP-3 | 9,160 | 96.28 |
| LEP-4 | 8,315 | 95.81 |
| LEP-5 | 5,512 | 95.16 |

## AGENT

[Open the AGENT vector PDF](plots/quailb_diffusion_gemma_agent.pdf)

Figure: plots/quailb_diffusion_gemma_agent.pdf

| Query | Input documents by alias and set |
|---|---|
| AGENT-1 | t (agent_traces) = 1,772 |
| AGENT-2 | t (agent_traces) = 1,772 |

| Query | Method | Seconds | Tokens/second | $/query | $/million input tokens | KV regret (%) |
|---|---|---:|---:|---:|---:|---:|
| AGENT-1 | Quail | 279.44 | 68,414.66 | 0.30655 | 0.016035 | 68.23 |
| AGENT-1 | Pipelined vLLM | 159.15 | 120,124.37 | 0.17459 | 0.009132 | 0.78 |
| AGENT-1 | SoL estimate | 46.622 | 410,055.89 | 0.05114 | 0.002675 | 0 (assumed) |
| AGENT-2 | Quail | 280.31 | 68,354.04 | 0.30750 | 0.016049 | 68.08 |
| AGENT-2 | Pipelined vLLM | 157.58 | 121,591.07 | 0.17287 | 0.009022 | 0.77 |
| AGENT-2 | SoL estimate | 46.983 | 407,811.34 | 0.05154 | 0.002690 | 0 (assumed) |

Correctness against saved reference labels:

| Query | Method | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|
| AGENT-1 | Quail | 47.18 | 38.964 | 95.23 |
| AGENT-1 | Pipelined vLLM | 38.49 | 35.603 | 98.026 |
| AGENT-2 | Quail | 93.34 | 81.903 | 99.62 |
| AGENT-2 | Pipelined vLLM | 88.21 | 72.585 | 96.964 |

Answers the two methods gave each other:

| Query | Evaluations both made | Same answer (%) |
|---|---:|---:|
| AGENT-1 | 1,772 | 84.65 |
| AGENT-2 | 1,772 | 92.61 |
