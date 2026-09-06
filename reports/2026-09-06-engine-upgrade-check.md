# Engine image upgrade check: vLLM 0.28.0, SGLang 0.5.19, CUDA 13.3.1

Date: 2026-09-06. One H100! per query family on Modal, Qwen3 4B fp8.

## Setup

- The worker images moved from vLLM 0.26.0 to 0.28.0, which moves torch
  from 2.11 to 2.13 (Quail's kernels run on vLLM's torch, DeepGEMM, and
  FlashAttention builds too), from SGLang 0.5.18 to 0.5.19, and from the
  `nvidia/cuda:13.0.1-devel-ubuntu24.04` base image to 13.3.1. No other
  code changed.
- Six of the fastest QUAIL-B queries, one or two per family, ran on the
  new images through the same-GPU runner: IMDB-1, IMDB-2, BIO-1, FEV-4,
  LEP-1, LEP-8. Every method ran all six.
- The comparison is the same six queries from the full run of
  2026-09-05 on the old images (`reports/2026-09-05-quailb-two-regrets.md`),
  on different physical H100s. The two runs were 22 hours apart.
- The new images' kernel caches (Triton, DeepGEMM, vLLM's compile cache)
  started empty. Query time excludes boot, but the first query after a
  cold boot pays any compilation that happens on first use.

## Prediction

Stated at launch: no query's time changes by more than 10 percent, each
method's six-query total stays within 5 percent of the old-image run
(Quail 64.26 s, stock vLLM 81.62 s, pipelined vLLM 81.18 s, pipelined
SGLang 112.53 s), answers and accuracy unchanged, and cold boots slower
on the new images.

## Result

Figure: plots/engine_upgrade_check.png

| Method | Query | Old images (s) | New images (s) | Change | Boot, new | Accuracy old | Accuracy new |
|---|---|---:|---:|---:|---|---:|---:|
| Quail | IMDB-1 | 14.25 | 14.96 | +5.0% | cold 273 s | 0.8968 | 0.8968 |
| Quail | IMDB-2 | 21.29 | 21.84 | +2.6% | warm 0 s | 0.7850 | 0.7850 |
| Quail | BIO-1 | 21.64 | 21.80 | +0.7% | cold 279 s | 0.9360 | 0.9360 |
| Quail | FEV-4 | 4.66 | 5.27 | +13.1% | cold 268 s | 0.8931 | 0.8931 |
| Quail | LEP-1 | 1.08 | 1.43 | +32.4% | cold 263 s | 0.3120 | 0.3120 |
| Quail | LEP-8 | 1.34 | 1.34 | +0.0% | warm 0 s | 0.4826 | 0.4826 |
| Quail | total | 64.26 | 66.64 | +3.7% | | | |
| Stock vLLM | IMDB-1 | 17.11 | 17.96 | +5.0% | cold 84 s | 0.8952 | 0.8952 |
| Stock vLLM | IMDB-2 | 26.68 | 30.40 | +13.9% | warm 0 s | 0.7830 | 0.7843 |
| Stock vLLM | BIO-1 | 25.94 | 26.34 | +1.5% | cold 87 s | 0.9400 | 0.9420 |
| Stock vLLM | FEV-4 | 8.26 | 9.25 | +12.0% | cold 86 s | 0.9050 | 0.9175 |
| Stock vLLM | LEP-1 | 1.54 | 1.49 | -3.2% | cold 82 s | 0.2740 | 0.2720 |
| Stock vLLM | LEP-8 | 2.09 | 2.17 | +3.8% | warm 0 s | 0.4556 | 0.4636 |
| Stock vLLM | total | 81.62 | 87.61 | +7.3% | | | |
| Pipelined vLLM | IMDB-1 | 17.03 | 17.83 | +4.7% | warm 0 s | 0.8952 | 0.8952 |
| Pipelined vLLM | IMDB-2 | 26.96 | 30.63 | +13.6% | warm 0 s | 0.7830 | 0.7843 |
| Pipelined vLLM | BIO-1 | 25.55 | 25.89 | +1.3% | warm 0 s | 0.9400 | 0.9420 |
| Pipelined vLLM | FEV-4 | 8.03 | 8.40 | +4.6% | warm 0 s | 0.9050 | 0.9175 |
| Pipelined vLLM | LEP-1 | 1.36 | 1.41 | +3.7% | warm 0 s | 0.2740 | 0.2720 |
| Pipelined vLLM | LEP-8 | 2.25 | 2.20 | -2.2% | warm 0 s | 0.4556 | 0.4636 |
| Pipelined vLLM | total | 81.18 | 86.36 | +6.4% | | | |
| Pipelined SGLang | IMDB-1 | 18.58 | 18.87 | +1.6% | cold 353 s | 0.9108 | 0.9108 |
| Pipelined SGLang | IMDB-2 | 54.37 | 62.11 | +14.2% | warm 0 s | 0.7923 | 0.7923 |
| Pipelined SGLang | BIO-1 | 24.87 | 24.60 | -1.1% | cold 352 s | 0.9420 | 0.9420 |
| Pipelined SGLang | FEV-4 | 11.02 | 14.59 | +32.4% | cold 354 s | 0.9270 | 0.9270 |
| Pipelined SGLang | LEP-1 | 1.80 | 1.38 | -23.3% | cold 317 s | 0.4400 | 0.4400 |
| Pipelined SGLang | LEP-8 | 1.89 | 2.08 | +10.1% | warm 0 s | 0.5635 | 0.5635 |
| Pipelined SGLang | total | 112.53 | 123.63 | +9.9% | | | |

- **No method got faster.** Six-query totals: Quail 66.64 s (+3.7%),
  stock vLLM 87.61 s (+7.3%), pipelined vLLM 86.36 s (+6.4%), pipelined
  SGLang 123.63 s (+9.9%). The prediction on totals held only for Quail.
- **Quail is unchanged in steady state.** Its two warm second queries are
  IMDB-2 at +2.6% and LEP-8 at exactly the old time. Its first query in
  each family carries a fixed cost of roughly 0.4 to 0.7 seconds
  (LEP-1 1.08 to 1.43 s, FEV-4 4.66 to 5.27 s), which is first-use
  kernel compilation on the fresh image; on the 15 and 22 second first
  queries the same cost is 5% and 1%.
- **vLLM 0.28.0 is slower on the join.** IMDB-2, a warm second query of
  60,000 pair requests, went from 26.7 to 30.4 s for stock vLLM and from
  27.0 to 30.6 s for pipelined vLLM (+14% both), on the same GPU where
  Quail's IMDB-2 moved +2.6%. The filters moved between -3% and +5%
  except stock vLLM's FEV-4 (+12%, a cold first query).
- **SGLang 0.5.19 is slower on the join too**, IMDB-2 +14%, and its short
  queries scatter (LEP-1 -23%, FEV-4 +32%, LEP-8 +10%): one-second
  queries in wave scheduling are noisy.
- **Answers.** Quail and SGLang returned identical answers on every query
  (same accuracy, same fresh tokens). vLLM 0.28.0 flipped a few
  borderline answers: FEV-4 accuracy 0.9050 to 0.9175 with 9,939 more
  fresh tokens from the changed filter survivors, and small shifts on
  IMDB-2, BIO-1, LEP-1, and LEP-8. Its fp8 kernels are not bit-identical
  to 0.26.0's.
- **Cold boots** on the fresh images: Quail 263 to 279 s (the old image
  with warm caches booted in 18 s), vLLM 82 to 87 s (was 73 to 75),
  SGLang 317 to 354 s (was 314 to 391). The Quail number is cache
  building, not a regression: LEP-8 ran warm right after.

## What it means

The upgrade brings no speed gain on these queries and costs the request
backends 6 to 10 percent, most of it on the join. The eight queries that
moved more than 10 percent are short queries or cold first queries, so a
single six-query run cannot separate a real regression from run-to-run
noise for anything but IMDB-2, where all three request backends moved
the same way on their own GPUs.

The published QUAIL-B comparison was measured on the old versions. Keeping
the new pins means the code no longer matches that report until the full
benchmark is rerun on the new images (about an hour and $16), and vLLM
0.28.0's slightly different answers would change the accuracy columns.
Reverting keeps the report and the code consistent.

## Source data

New-image run `20260906T011248Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families`:

- Manifest:
  `/results/benchmarks/quailb/family-runs/20260906T011248Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json`
- Per method suite files under
  `/results/benchmarks/quailb/runs/qb_20260906T011248Z_*/`, listed in the
  manifest's `result_volume_paths`.
- Family function calls (Quail and vLLM container, SGLang container):
  IMDB `fc-01M1T4D1ZM6F4XPHHRJT7R5S35`, `fc-01M1T4D21KSRHQJGNMYJFPEKCH`;
  BioDEX `fc-01M1T4D24508PT5J2C72PA8A5R`, `fc-01M1T4D2668K1Z5TC6DX47Z3NP`;
  FEVER `fc-01M1T4D2819B6Q1TB7A3M97M6S`, `fc-01M1T4D29YGR6Y6KHQV4FT36CE`;
  LePaRD `fc-01M1T4D2BVE3N72FKDTMQSQ5ME`, `fc-01M1T4D2E3FWDD3SC7C7GR8ME8`.

Old-image run `20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families`:
see `reports/2026-09-05-quailb-two-regrets.md`.

## Rebuild

Run `reports/make_engine_upgrade_plots.py` from the repository root with
the two work directories; its docstring has the `modal volume get`
commands.
