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
separates the performance panels from KV regret. The dark horizontal mark on
each query is the speed of light (SoL) estimate from
`reports/make_sol_quailb.py`: the ideal one H100 time for the modeled work
with ideal packing, unlimited KV, and every distinct token prefix computed
once across the corpus, priced from counted constants.

The unit is documents per second for filter-only queries and evaluated document
pairs per second for queries with at least one join.

### Time relative to SoL

| Method | Total time (s) | SoL total (s) | Time / SoL | Median time / SoL | Best query | Worst query |
|---|---:|---:|---:|---:|---|---|
| Quail | 1,629.61 | 464.74 | 3.51x | 2.68x | BIO-1 (1.95x) | LEP-5 (51.36x) |
| Stock vLLM | 3,453.08 | 464.74 | 7.43x | 5.88x | BIO-1 (2.58x) | LEP-5 (74.83x) |
| Pipelined vLLM | 3,298.33 | 464.74 | 7.10x | 4.34x | AGENT-2 (2.06x) | LEP-5 (82.69x) |

Quail runs the 32 queries in 3.51 times the SoL total, against 7.43 times for
stock vLLM and 7.10 times for pipelined vLLM. The median query is 2.68 times
SoL for Quail. The BioDEX and most IMDB and FEVER queries sit between 2 and 3
times SoL for Quail.

The LePaRD join queries LEP-3 through LEP-7 sit 19 to 51 times above their
SoL marks for every method. That gap is not engine overhead. The SoL uses the
exact ground truth to decide which documents survive each filter, and on
LePaRD the ground truth is far stricter than the model. The first filter of
LEP-3, "reasoning does not apply", is true for 14 of the 500 excerpts in the
ground truth, so the SoL joins 14 excerpts and evaluates 6,062 pairs. Qwen3 4B
answers TRUE for 356 of the 500, so the measured run joins 356 excerpts and
evaluates 154,148 pairs, 25 times the modeled work. LEP-5 is the extreme
case: the ground truth leaves 0 excerpts after its three filters, so the SoL
evaluates no join pairs at all, while the measured run evaluates 39,836. The
same effect, in a milder form, is why FEV-7 through FEV-9 evaluate 1.6 to 2.1
times the pairs the SoL models. The three methods do the same model driven
work, so the comparison between them is unaffected; the SoL column shows what
the query would cost if the model's answers matched the ground truth.

The agent queries are the one place the vLLM methods beat Quail, and the SoL
says why. The agent trace rows are prefixes of each other: the corpus samples
one row every five assistant turns of the same trajectory, and 68.9% of the
corpus tokens are a prefix some other row already contains. The SoL credits
that sharing, so AGENT-1 needs 47.46 seconds of ideal work, not the 130.96
seconds it would need with every trace computed from scratch. Quail computes
every trace from scratch and lands at 5.05 times SoL. vLLM's prefix cache
serves part of the shared tokens from KV and lands at 2.86 times SoL for
stock and 2.06 times for pipelined. The regret metric does not count sharing
across different rows, which is why both vLLM methods report zero regret
here. Cross-document prefix reuse is work Quail's KV rewind does not do yet.

### Per-query metrics

The SoL column gives the estimate's seconds, throughput at its own modeled
work, and cost. Each method's cell ends with its time as a multiple of SoL.

| Query | Unit | SoL estimate | Quail | Stock vLLM | Pipelined vLLM |
|---|---|---:|---:|---:|---:|
| IMDB-1 | docs/s | 6.65 s; 751.6 docs/s; $0.0073 | 14.88 s; 336.0 docs/s; $0.0163; 0 regret; 2.24x SoL | 19.88 s; 251.6 docs/s; $0.0218; 0 regret; 2.99x SoL | 17.30 s; 289.0 docs/s; $0.0190; 0 regret; 2.60x SoL |
| IMDB-2 | pairs/s | 9.21 s; 6,513.9 pairs/s; $0.0101 | 21.65 s; 2,771.4 pairs/s; $0.0238; 0 regret; 2.35x SoL | 29.25 s; 2,051.5 pairs/s; $0.0321; 0 regret; 3.18x SoL | 28.53 s; 2,102.7 pairs/s; $0.0313; 0 regret; 3.10x SoL |
| IMDB-3 | pairs/s | 9.50 s; 5,060.2 pairs/s; $0.0104 | 33.03 s; 1,591.3 pairs/s; $0.0362; 1,217,732 regret; 3.48x SoL | 45.80 s; 1,146.0 pairs/s; $0.0502; 1,326,432 regret; 4.82x SoL | 42.28 s; 1,241.5 pairs/s; $0.0464; 1,326,432 regret; 4.45x SoL |
| IMDB-4 | pairs/s | 7.52 s; 1,640.7 pairs/s; $0.0082 | 20.39 s; 739.8 pairs/s; $0.0224; 361,440 regret; 2.71x SoL | 46.64 s; 327.5 pairs/s; $0.0512; 500,400 regret; 6.20x SoL | 27.59 s; 553.6 pairs/s; $0.0303; 500,400 regret; 3.67x SoL |
| IMDB-5 | pairs/s | 7.41 s; 1,082.5 pairs/s; $0.0081 | 18.13 s; 481.2 pairs/s; $0.0199; 188,587 regret; 2.45x SoL | 48.66 s; 183.7 pairs/s; $0.0534; 186,256 regret; 6.57x SoL | 25.94 s; 344.6 pairs/s; $0.0285; 278,240 regret; 3.50x SoL |
| IMDB-6 | docs/s | 6.78 s; 737.5 docs/s; $0.0074 | 14.91 s; 335.3 docs/s; $0.0164; 0 regret; 2.20x SoL | 37.64 s; 132.8 docs/s; $0.0413; 0 regret; 5.55x SoL | 18.83 s; 265.6 docs/s; $0.0207; 0 regret; 2.78x SoL |
| IMDB-7 | docs/s | 6.92 s; 722.3 docs/s; $0.0076 | 15.09 s; 331.3 docs/s; $0.0166; 0 regret; 2.18x SoL | 44.62 s; 112.1 docs/s; $0.0489; 0 regret; 6.45x SoL | 20.65 s; 242.2 docs/s; $0.0226; 0 regret; 2.98x SoL |
| IMDB-8 | pairs/s | 11.90 s; 8,772.1 pairs/s; $0.0131 | 27.43 s; 3,346.3 pairs/s; $0.0301; 0 regret; 2.30x SoL | 46.95 s; 2,123.2 pairs/s; $0.0515; 1,062,720 regret; 3.94x SoL | 46.84 s; 2,128.3 pairs/s; $0.0514; 1,062,720 regret; 3.94x SoL |
| IMDB-9 | pairs/s | 21.11 s; 7,786.9 pairs/s; $0.0232 | 48.68 s; 3,118.1 pairs/s; $0.0534; 0 regret; 2.31x SoL | 74.68 s; 2,138.4 pairs/s; $0.0819; 1,062,720 regret; 3.54x SoL | 75.98 s; 2,101.9 pairs/s; $0.0833; 1,062,720 regret; 3.60x SoL |
| IMDB-10 | pairs/s | 21.40 s; 7,124.9 pairs/s; $0.0235 | 60.95 s; 2,470.9 pairs/s; $0.0669; 1,217,732 regret; 2.85x SoL | 92.95 s; 1,637.2 pairs/s; $0.1020; 2,389,152 regret; 4.34x SoL | 90.35 s; 1,684.5 pairs/s; $0.0991; 2,389,152 regret; 4.22x SoL |
| BIO-1 | docs/s | 11.05 s; 45.3 docs/s; $0.0121 | 21.59 s; 23.2 docs/s; $0.0237; 0 regret; 1.95x SoL | 28.52 s; 17.5 docs/s; $0.0313; 0 regret; 2.58x SoL | 25.37 s; 19.7 docs/s; $0.0278; 0 regret; 2.30x SoL |
| BIO-2 | pairs/s | 62.00 s; 9,088.5 pairs/s; $0.0680 | 129.64 s; 4,346.7 pairs/s; $0.1422; 0 regret; 2.09x SoL | 924.93 s; 609.2 pairs/s; $1.0146; 0 regret; 14.92x SoL | 895.78 s; 629.1 pairs/s; $0.9827; 0 regret; 14.45x SoL |
| BIO-3 | pairs/s | 43.09 s; 8,003.8 pairs/s; $0.0473 | 90.20 s; 3,423.5 pairs/s; $0.0989; 919,409 regret; 2.09x SoL | 501.11 s; 620.7 pairs/s; $0.5497; 1,079,840 regret; 11.63x SoL | 500.44 s; 621.6 pairs/s; $0.5490; 1,079,840 regret; 11.61x SoL |
| FEV-1 | docs/s | 0.12 s; 4,139.2 docs/s; $0.0001 | 0.29 s; 1,724.1 docs/s; $0.0003; 0 regret; 2.40x SoL | 0.53 s; 940.6 docs/s; $0.0006; 0 regret; 4.40x SoL | 0.40 s; 1,238.0 docs/s; $0.0004; 0 regret; 3.34x SoL |
| FEV-2 | pairs/s | 12.84 s; 11,171.7 pairs/s; $0.0141 | 29.14 s; 4,924.5 pairs/s; $0.0320; 0 regret; 2.27x SoL | 81.27 s; 1,765.8 pairs/s; $0.0892; 0 regret; 6.33x SoL | 77.33 s; 1,855.7 pairs/s; $0.0848; 0 regret; 6.02x SoL |
| FEV-3 | pairs/s | 7.94 s; 10,694.6 pairs/s; $0.0087 | 21.58 s; 4,801.1 pairs/s; $0.0237; 0 regret; 2.72x SoL | 56.83 s; 1,888.9 pairs/s; $0.0623; 0 regret; 7.15x SoL | 57.78 s; 1,857.6 pairs/s; $0.0634; 0 regret; 7.27x SoL |
| FEV-4 | pairs/s | 1.87 s; 5,991.9 pairs/s; $0.0020 | 4.76 s; 2,833.8 pairs/s; $0.0052; 0 regret; 2.55x SoL | 9.01 s; 1,624.6 pairs/s; $0.0099; 0 regret; 4.82x SoL | 8.75 s; 1,673.2 pairs/s; $0.0096; 0 regret; 4.68x SoL |
| FEV-5 | pairs/s | 4.74 s; 9,923.9 pairs/s; $0.0052 | 13.66 s; 4,519.1 pairs/s; $0.0150; 0 regret; 2.88x SoL | 38.02 s; 1,692.0 pairs/s; $0.0417; 44,000 regret; 8.02x SoL | 36.73 s; 1,751.2 pairs/s; $0.0403; 44,000 regret; 7.75x SoL |
| FEV-6 | pairs/s | 1.37 s; 4,510.2 pairs/s; $0.0015 | 3.65 s; 2,201.9 pairs/s; $0.0040; 0 regret; 2.65x SoL | 7.56 s; 1,159.6 pairs/s; $0.0083; 0 regret; 5.50x SoL | 6.98 s; 1,257.1 pairs/s; $0.0077; 0 regret; 5.08x SoL |
| FEV-7 | pairs/s | 15.43 s; 10,822.9 pairs/s; $0.0169 | 54.70 s; 4,916.3 pairs/s; $0.0600; 0 regret; 3.54x SoL | 156.03 s; 1,721.6 pairs/s; $0.1712; 0 regret; 10.11x SoL | 141.58 s; 1,897.3 pairs/s; $0.1553; 0 regret; 9.17x SoL |
| FEV-8 | pairs/s | 19.50 s; 10,950.4 pairs/s; $0.0214 | 85.22 s; 4,974.2 pairs/s; $0.0935; 0 regret; 4.37x SoL | 232.43 s; 1,827.5 pairs/s; $0.2550; 132,384 regret; 11.92x SoL | 229.74 s; 1,848.9 pairs/s; $0.2520; 132,384 regret; 11.78x SoL |
| FEV-9 | pairs/s | 16.64 s; 10,791.4 pairs/s; $0.0183 | 77.68 s; 4,943.4 pairs/s; $0.0852; 0 regret; 4.67x SoL | 219.53 s; 1,770.2 pairs/s; $0.2408; 132,384 regret; 13.19x SoL | 210.47 s; 1,846.3 pairs/s; $0.2309; 132,384 regret; 12.65x SoL |
| LEP-1 | docs/s | 0.49 s; 1,012.6 docs/s; $0.0005 | 1.11 s; 450.5 docs/s; $0.0012; 0 regret; 2.25x SoL | 1.58 s; 316.7 docs/s; $0.0017; 0 regret; 3.20x SoL | 1.32 s; 380.0 docs/s; $0.0014; 0 regret; 2.66x SoL |
| LEP-2 | pairs/s | 58.09 s; 3,727.2 pairs/s; $0.0637 | 131.43 s; 1,647.3 pairs/s; $0.1442; 0 regret; 2.26x SoL | 156.92 s; 1,379.7 pairs/s; $0.1721; 0 regret; 2.70x SoL | 218.79 s; 989.5 pairs/s; $0.2400; 0 regret; 3.77x SoL |
| LEP-3 | pairs/s | 2.15 s; 2,825.4 pairs/s; $0.0024 | 94.06 s; 1,638.8 pairs/s; $0.1032; 0 regret; 43.84x SoL | 119.61 s; 1,364.8 pairs/s; $0.1312; 71,936 regret; 55.75x SoL | 120.93 s; 1,349.9 pairs/s; $0.1327; 71,936 regret; 56.36x SoL |
| LEP-4 | pairs/s | 1.09 s; 1,992.5 pairs/s; $0.0012 | 40.17 s; 1,616.9 pairs/s; $0.0441; 0 regret; 36.97x SoL | 47.92 s; 1,373.5 pairs/s; $0.0526; 19,360 regret; 44.10x SoL | 66.43 s; 990.7 pairs/s; $0.0729; 19,360 regret; 61.14x SoL |
| LEP-5 | pairs/s | 0.49 s; 1,899.4 pairs/s; $0.0005 | 25.23 s; 1,578.9 pairs/s; $0.0277; 0 regret; 51.36x SoL | 36.76 s; 1,295.8 pairs/s; $0.0403; 13,600 regret; 74.83x SoL | 40.62 s; 1,172.6 pairs/s; $0.0446; 13,600 regret; 82.69x SoL |
| LEP-6 | pairs/s | 0.48 s; 1,930.2 pairs/s; $0.0005 | 9.15 s; 1,419.7 pairs/s; $0.0100; 0 regret; 18.93x SoL | 20.03 s; 886.4 pairs/s; $0.0220; 1,888 regret; 41.44x SoL | 17.88 s; 992.9 pairs/s; $0.0196; 1,888 regret; 36.99x SoL |
| LEP-7 | pairs/s | 1.13 s; 1,554.5 pairs/s; $0.0012 | 39.39 s; 1,599.4 pairs/s; $0.0432; 0 regret; 34.89x SoL | 51.80 s; 1,235.4 pairs/s; $0.0568; 19,440 regret; 45.88x SoL | 48.01 s; 1,332.8 pairs/s; $0.0527; 19,440 regret; 42.53x SoL |
| LEP-8 | docs/s | 0.48 s; 1,034.4 docs/s; $0.0005 | 1.33 s; 375.9 docs/s; $0.0015; 0 regret; 2.75x SoL | 2.53 s; 197.7 docs/s; $0.0028; 0 regret; 5.23x SoL | 1.97 s; 253.3 docs/s; $0.0022; 0 regret; 4.08x SoL |
| AGENT-1 | docs/s | 47.46 s; 37.3 docs/s; $0.0521 | 239.75 s; 7.4 docs/s; $0.2630; 0 regret; 5.05x SoL | 135.96 s; 13.0 docs/s; $0.1491; 0 regret; 2.86x SoL | 97.99 s; 18.1 docs/s; $0.1075; 0 regret; 2.06x SoL |
| AGENT-2 | docs/s | 47.87 s; 37.0 docs/s; $0.0525 | 240.74 s; 7.4 docs/s; $0.2641; 0 regret; 5.03x SoL | 137.13 s; 12.9 docs/s; $0.1504; 0 regret; 2.86x SoL | 98.74 s; 17.9 docs/s; $0.1083; 0 regret; 2.06x SoL |
wrote /home/user/quail-exploration/reports/plots/quailb_sf01_4b_metrics.png
wrote /home/user/quail-exploration/reports/plots/quailb_sf01_4b_per_query.png

## Rebuild

Run `reports/make_quailb_sf01_4b_plots.py` from the repository root.
Its docstring contains every `modal volume get` command needed to rebuild both
figures and print all derived tables, including the SoL file at
`/results/sol/sol_quailb_sf0.1.json`.
