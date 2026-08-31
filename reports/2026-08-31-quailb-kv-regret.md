# QuailB runtime and KV regret with Qwen3 4B fp8

## Setup

- The benchmark contains all 32 current QuailB queries at scale factor 0.1.
  Privacy queries are excluded.
- Every method uses Qwen3 4B fp8 on one H100!.
- Quail uses pipelining, token-based admission, and KV rewind.
- Stock vLLM submits separate requests in stage-major waves.
- Pipelined vLLM submits the next filter when each document finishes its
  current filter.
- Both vLLM methods use the same full cross product join implementation.
- Both vLLM methods use `gpu_memory_utilization=0.91`, a maximum of 25,305
  batched tokens, and space for 479,248 KV tokens.
- The final vLLM measurements use vLLM's default list of 51 CUDA graph sizes.
  vLLM captured both piecewise and full graphs. No custom compilation
  configuration was passed.
- Quail clears KV before each query. vLLM resets its prefix cache before each
  query and method.
- Query time and cost exclude model startup. Cost uses $3.9492 per H100! hour.
- Accuracy uses ground truth collection
  `gt_77bb8b128743a79aedddaa24c808c3f8`.

The final measurements come from these family files on the `quail-results`
volume:

- IMDB:
  `/results/benchmarks/quailb/families/20260831T070520Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/imdb.json`
- BioDEX:
  `/results/benchmarks/quailb/families/20260831T144252Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/biodex.json`
- FEVER:
  `/results/benchmarks/quailb/families/20260831T070520Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/fever.json`
- LePaRD:
  `/results/benchmarks/quailb/families/20260831T070520Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/lepard.json`
- Agent:
  `/results/benchmarks/quailb/families/20260831T070520Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/agent.json`

The family function calls were:

- IMDB: `fc-01M1BA66QNWH5970XTRJ4GPYN1`
- BioDEX: `fc-01M1C4C41CNCK6ZSFA3AV0RSBB`
- FEVER: `fc-01M1BA66ZD4XR9XARPKBHS6TDZ`
- LePaRD: `fc-01M1BA6731ESB7M5EZ71QQZRCP`
- Agent: `fc-01M1BA676J9YRVMASQV1R4D9FS`

A preceding run used one CUDA graph capture size of 8192. vLLM ran out of
memory, so none of its measurements are used here. Another run with the default
51 sizes completed four families but ran out of memory during BioDEX. The
focused BioDEX rerun used the same measured configuration and completed all
queries. The runner now records a failed query as `oom` and restarts vLLM
before the next query, but no query failed in the final BioDEX rerun.

## KV regret definition

KV regret counts prefix tokens that a method computed again after it had
already computed the same logical document prefix earlier in the query. An
unlimited KV implementation would keep that prefix, so its KV regret is zero.

- Quail records fresh prefix tokens when it recomputes an evicted logical
  prefix.
- The vLLM methods use each request's cached token count. They count only full
  16-token cache blocks when matching an earlier logical prefix.
- First use of a prefix is required work and does not count as regret.
- A pure filter that evaluates every document once has zero regret, even when
  vLLM reports no cached tokens.
- The metric tracks reuse of the same logical alias and row. It does not count
  common token prefixes across different rows.

The last rule is important for the Agent queries. Consecutive agent trace rows
share long token prefixes, and vLLM can reuse those prefixes across requests.
The rows have different logical identities, so all three methods report zero KV
regret on AGENT-1 and AGENT-2. KV regret therefore measures one specific source
of extra work. It does not measure every possible prefix reuse opportunity.

## Prediction

- Quail should take 1,600 to 1,800 total query seconds.
- Stock vLLM should take 4,000 to 4,400 seconds.
- Pipelined vLLM should take 3,700 to 4,100 seconds.
- Combined query cost should be $10.20 to $11.30, excluding startup.
- KV regret should be concentrated in filters followed by joins and repeated
  join chains.
- IMDB-3 should have about 1.22 million Quail regret tokens and 1.33 million
  stock vLLM regret tokens.

## Result

All 32 queries completed for all three methods. Every result row includes KV
regret.

- Quail took 1,629.61 seconds. Stock vLLM took 3,453.08 seconds, and pipelined
  vLLM took 3,298.33 seconds.
- Quail was 2.12 times faster than stock vLLM and 2.02 times faster than
  pipelined vLLM by total query time.
- Quail had the lowest query time on 30 of 32 queries. Pipelined vLLM had the
  lowest time on AGENT-1 and AGENT-2. Stock vLLM had no query with the lowest
  time.
- Quail cost $1.7877 for all 32 queries. Stock vLLM cost $3.7880, and
  pipelined vLLM cost $3.6183. Combined query cost was $9.1940.
- Quail processed 3,112.7 document pairs per second across the 23 queries with
  joins. Stock vLLM processed 1,122.7, and pipelined vLLM processed 1,133.5.
- Quail processed 37.4 documents per second across the nine filter-only
  queries. Stock vLLM processed 50.3, and pipelined vLLM processed 72.7. The
  two Agent queries account for Quail's lower filter-only result.
- Quail had 3,904,900 KV regret tokens. Stock vLLM had 8,042,512, and
  pipelined vLLM had 8,134,496. Quail reduced regret by 51.4% compared with
  stock vLLM and by 52.0% compared with pipelined vLLM.
- KV regret was 2.48% of Quail's measured fresh input tokens. It was 5.40% for
  stock vLLM and 5.70% for pipelined vLLM.
- Weighted answer accuracy was 70.07% for Quail and 69.30% for both vLLM
  methods.

The Quail time prediction was correct. The stock vLLM and pipelined vLLM times
were lower than predicted. The combined cost was also lower than predicted.
The prediction used older BioDEX times. In this run, stock vLLM took 1,454.57
seconds on BioDEX, compared with 2,359.59 seconds in the prior report.
Pipelined vLLM took 1,421.59 seconds, compared with 2,218.68 seconds in the
prior report. This experiment does not isolate the cause of that difference.

The IMDB-3 regret prediction was correct. Quail measured 1,217,732 tokens, and
both vLLM methods measured 1,326,432 tokens.

KV regret explains part of the runtime difference, but it does not explain all
of it. BIO-2 has zero regret for every method, but Quail was 7.13 times faster
than stock vLLM and 6.91 times faster than pipelined vLLM. AGENT-1 and AGENT-2
also have zero measured regret, but pipelined vLLM was about 2.44 times faster
than Quail because the metric does not count shared prefixes across different
trace rows.

## Aggregate metrics

Figure: plots/quailb_sf01_4b_metrics.png

The left column shows total query time, cost, and answer accuracy. The middle
column shows throughput. The right column shows KV regret.

| Method | Queries | Total time (s) | Total cost | Filter throughput | Join throughput | KV regret | Regret / fresh tokens | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Quail | 32 | 1,629.61 | $1.7877 | 37.4 docs/s | 3,112.7 pairs/s | 3,904,900 | 2.48% | 70.07% |
| Stock vLLM | 32 | 3,453.08 | $3.7880 | 50.3 docs/s | 1,122.7 pairs/s | 8,042,512 | 5.40% | 69.30% |
| Pipelined vLLM | 32 | 3,298.33 | $3.6183 | 72.7 docs/s | 1,133.5 pairs/s | 8,134,496 | 5.70% | 69.30% |

## Results by family

| Family | Queries | Quail (s) | Stock vLLM (s) | Pipelined vLLM (s) | Stock time / Quail time |
|---|---:|---:|---:|---:|---:|
| IMDB | 10 | 275.14 | 487.07 | 394.29 | 1.77x |
| BioDEX | 3 | 241.43 | 1,454.57 | 1,421.59 | 6.02x |
| FEVER | 9 | 290.68 | 801.21 | 769.77 | 2.76x |
| LePaRD | 8 | 341.87 | 437.14 | 515.95 | 1.28x |
| Agent | 2 | 480.49 | 273.09 | 196.73 | 0.57x |

A value above 1 in the last column means Quail was faster than stock vLLM.

## Nonzero KV regret

| Query | Quail | Stock vLLM | Pipelined vLLM |
|---|---:|---:|---:|
| IMDB-3 | 1,217,732 | 1,326,432 | 1,326,432 |
| IMDB-4 | 361,440 | 500,400 | 500,400 |
| IMDB-5 | 188,587 | 186,256 | 278,240 |
| IMDB-8 | 0 | 1,062,720 | 1,062,720 |
| IMDB-9 | 0 | 1,062,720 | 1,062,720 |
| IMDB-10 | 1,217,732 | 2,389,152 | 2,389,152 |
| BIO-3 | 919,409 | 1,079,840 | 1,079,840 |
| FEV-5 | 0 | 44,000 | 44,000 |
| FEV-8 | 0 | 132,384 | 132,384 |
| FEV-9 | 0 | 132,384 | 132,384 |
| LEP-3 | 0 | 71,936 | 71,936 |
| LEP-4 | 0 | 19,360 | 19,360 |
| LEP-5 | 0 | 13,600 | 13,600 |
| LEP-6 | 0 | 1,888 | 1,888 |
| LEP-7 | 0 | 19,440 | 19,440 |

Quail had nonzero regret on five queries. Both vLLM methods had nonzero regret
on 15 queries. Quail avoided all measured logical-prefix recomputation on the
repeated join chains IMDB-8 and IMDB-9. It also avoided all measured regret on
the FEVER and LePaRD queries.

## Per-query runtime, throughput, cost, and regret

Figure: plots/quailb_sf01_4b_per_query.png

Thin vertical lines separate the five query families. The horizontal line
separates the performance panels from KV regret.

The unit is documents per second for filter-only queries and evaluated document
pairs per second for queries with at least one join.

| Query | Unit | Quail | Stock vLLM | Pipelined vLLM |
|---|---|---:|---:|---:|
| IMDB-1 | docs/s | 14.88 s; 336.0 docs/s; $0.0163; 0 regret | 19.88 s; 251.6 docs/s; $0.0218; 0 regret | 17.30 s; 289.0 docs/s; $0.0190; 0 regret |
| IMDB-2 | pairs/s | 21.65 s; 2,771.4 pairs/s; $0.0238; 0 regret | 29.25 s; 2,051.5 pairs/s; $0.0321; 0 regret | 28.53 s; 2,102.7 pairs/s; $0.0313; 0 regret |
| IMDB-3 | pairs/s | 33.03 s; 1,591.3 pairs/s; $0.0362; 1,217,732 regret | 45.80 s; 1,146.0 pairs/s; $0.0502; 1,326,432 regret | 42.28 s; 1,241.5 pairs/s; $0.0464; 1,326,432 regret |
| IMDB-4 | pairs/s | 20.39 s; 739.8 pairs/s; $0.0224; 361,440 regret | 46.64 s; 327.5 pairs/s; $0.0512; 500,400 regret | 27.59 s; 553.6 pairs/s; $0.0303; 500,400 regret |
| IMDB-5 | pairs/s | 18.13 s; 481.2 pairs/s; $0.0199; 188,587 regret | 48.66 s; 183.7 pairs/s; $0.0534; 186,256 regret | 25.94 s; 344.6 pairs/s; $0.0285; 278,240 regret |
| IMDB-6 | docs/s | 14.91 s; 335.3 docs/s; $0.0164; 0 regret | 37.64 s; 132.8 docs/s; $0.0413; 0 regret | 18.83 s; 265.6 docs/s; $0.0207; 0 regret |
| IMDB-7 | docs/s | 15.09 s; 331.3 docs/s; $0.0166; 0 regret | 44.62 s; 112.1 docs/s; $0.0489; 0 regret | 20.65 s; 242.2 docs/s; $0.0226; 0 regret |
| IMDB-8 | pairs/s | 27.43 s; 3,346.3 pairs/s; $0.0301; 0 regret | 46.95 s; 2,123.2 pairs/s; $0.0515; 1,062,720 regret | 46.84 s; 2,128.3 pairs/s; $0.0514; 1,062,720 regret |
| IMDB-9 | pairs/s | 48.68 s; 3,118.1 pairs/s; $0.0534; 0 regret | 74.68 s; 2,138.4 pairs/s; $0.0819; 1,062,720 regret | 75.98 s; 2,101.9 pairs/s; $0.0833; 1,062,720 regret |
| IMDB-10 | pairs/s | 60.95 s; 2,470.9 pairs/s; $0.0669; 1,217,732 regret | 92.95 s; 1,637.2 pairs/s; $0.1020; 2,389,152 regret | 90.35 s; 1,684.5 pairs/s; $0.0991; 2,389,152 regret |
| BIO-1 | docs/s | 21.59 s; 23.2 docs/s; $0.0237; 0 regret | 28.52 s; 17.5 docs/s; $0.0313; 0 regret | 25.37 s; 19.7 docs/s; $0.0278; 0 regret |
| BIO-2 | pairs/s | 129.64 s; 4,346.7 pairs/s; $0.1422; 0 regret | 924.93 s; 609.2 pairs/s; $1.0146; 0 regret | 895.78 s; 629.1 pairs/s; $0.9827; 0 regret |
| BIO-3 | pairs/s | 90.20 s; 3,423.5 pairs/s; $0.0989; 919,409 regret | 501.11 s; 620.7 pairs/s; $0.5497; 1,079,840 regret | 500.44 s; 621.6 pairs/s; $0.5490; 1,079,840 regret |
| FEV-1 | docs/s | 0.29 s; 1,724.1 docs/s; $0.0003; 0 regret | 0.53 s; 940.6 docs/s; $0.0006; 0 regret | 0.40 s; 1,238.0 docs/s; $0.0004; 0 regret |
| FEV-2 | pairs/s | 29.14 s; 4,924.5 pairs/s; $0.0320; 0 regret | 81.27 s; 1,765.8 pairs/s; $0.0892; 0 regret | 77.33 s; 1,855.7 pairs/s; $0.0848; 0 regret |
| FEV-3 | pairs/s | 21.58 s; 4,801.1 pairs/s; $0.0237; 0 regret | 56.83 s; 1,888.9 pairs/s; $0.0623; 0 regret | 57.78 s; 1,857.6 pairs/s; $0.0634; 0 regret |
| FEV-4 | pairs/s | 4.76 s; 2,833.8 pairs/s; $0.0052; 0 regret | 9.01 s; 1,624.6 pairs/s; $0.0099; 0 regret | 8.75 s; 1,673.2 pairs/s; $0.0096; 0 regret |
| FEV-5 | pairs/s | 13.66 s; 4,519.1 pairs/s; $0.0150; 0 regret | 38.02 s; 1,692.0 pairs/s; $0.0417; 44,000 regret | 36.73 s; 1,751.2 pairs/s; $0.0403; 44,000 regret |
| FEV-6 | pairs/s | 3.65 s; 2,201.9 pairs/s; $0.0040; 0 regret | 7.56 s; 1,159.6 pairs/s; $0.0083; 0 regret | 6.98 s; 1,257.1 pairs/s; $0.0077; 0 regret |
| FEV-7 | pairs/s | 54.70 s; 4,916.3 pairs/s; $0.0600; 0 regret | 156.03 s; 1,721.6 pairs/s; $0.1712; 0 regret | 141.58 s; 1,897.3 pairs/s; $0.1553; 0 regret |
| FEV-8 | pairs/s | 85.22 s; 4,974.2 pairs/s; $0.0935; 0 regret | 232.43 s; 1,827.5 pairs/s; $0.2550; 132,384 regret | 229.74 s; 1,848.9 pairs/s; $0.2520; 132,384 regret |
| FEV-9 | pairs/s | 77.68 s; 4,943.4 pairs/s; $0.0852; 0 regret | 219.53 s; 1,770.2 pairs/s; $0.2408; 132,384 regret | 210.47 s; 1,846.3 pairs/s; $0.2309; 132,384 regret |
| LEP-1 | docs/s | 1.11 s; 450.5 docs/s; $0.0012; 0 regret | 1.58 s; 316.7 docs/s; $0.0017; 0 regret | 1.32 s; 380.0 docs/s; $0.0014; 0 regret |
| LEP-2 | pairs/s | 131.43 s; 1,647.3 pairs/s; $0.1442; 0 regret | 156.92 s; 1,379.7 pairs/s; $0.1721; 0 regret | 218.79 s; 989.5 pairs/s; $0.2400; 0 regret |
| LEP-3 | pairs/s | 94.06 s; 1,638.8 pairs/s; $0.1032; 0 regret | 119.61 s; 1,364.8 pairs/s; $0.1312; 71,936 regret | 120.93 s; 1,349.9 pairs/s; $0.1327; 71,936 regret |
| LEP-4 | pairs/s | 40.17 s; 1,616.9 pairs/s; $0.0441; 0 regret | 47.92 s; 1,373.5 pairs/s; $0.0526; 19,360 regret | 66.43 s; 990.7 pairs/s; $0.0729; 19,360 regret |
| LEP-5 | pairs/s | 25.23 s; 1,578.9 pairs/s; $0.0277; 0 regret | 36.76 s; 1,295.8 pairs/s; $0.0403; 13,600 regret | 40.62 s; 1,172.6 pairs/s; $0.0446; 13,600 regret |
| LEP-6 | pairs/s | 9.15 s; 1,419.7 pairs/s; $0.0100; 0 regret | 20.03 s; 886.4 pairs/s; $0.0220; 1,888 regret | 17.88 s; 992.9 pairs/s; $0.0196; 1,888 regret |
| LEP-7 | pairs/s | 39.39 s; 1,599.4 pairs/s; $0.0432; 0 regret | 51.80 s; 1,235.4 pairs/s; $0.0568; 19,440 regret | 48.01 s; 1,332.8 pairs/s; $0.0527; 19,440 regret |
| LEP-8 | docs/s | 1.33 s; 375.9 docs/s; $0.0015; 0 regret | 2.53 s; 197.7 docs/s; $0.0028; 0 regret | 1.97 s; 253.3 docs/s; $0.0022; 0 regret |
| AGENT-1 | docs/s | 239.75 s; 7.4 docs/s; $0.2630; 0 regret | 135.96 s; 13.0 docs/s; $0.1491; 0 regret | 97.99 s; 18.1 docs/s; $0.1075; 0 regret |
| AGENT-2 | docs/s | 240.74 s; 7.4 docs/s; $0.2641; 0 regret | 137.13 s; 12.9 docs/s; $0.1504; 0 regret | 98.74 s; 17.9 docs/s; $0.1083; 0 regret |

## Rebuild

Run `reports/make_quailb_sf01_4b_plots.py` from the repository root.
Its docstring contains every `modal volume get` command needed to rebuild both
figures and print all derived tables.
