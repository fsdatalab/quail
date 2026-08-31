# Repeated prefixes in cumulative agent traces

## Setup

The SWE-Next dataset represents one agent trajectory with several cumulative
documents. Quail creates a document after every fifth assistant turn and
includes the following tool result when one exists. Later documents from the
same trajectory repeat the full text of the earlier documents. Quail excludes
documents over 24,000 Qwen3 tokens and trajectories without a user issue.

The scale factor 0.1 dataset has the following properties:

* It has 1,772 documents from 376 trajectories.

* It has 17,252,669 document tokens.

* The median document has 8,880.5 tokens, and the 90th percentile has
  17,988.2 tokens.

* The documents cover assistant turns 5 through 50.

The experiment runs two semantic filters:

* AGENT-1 asks whether the agent recovered after pursuing an approach that did
  not work.

* AGENT-2 asks whether the agent implemented a plausible fix that directly
  addresses the reported issue.

Both queries use Qwen3 4B fp8 on one H100. Quail uses token-based admission and
KV rewind. Stock vLLM submits separate requests in one stage major wave.
Pipelined vLLM limits submitted documents to 48 at a time. Both vLLM
configurations use automatic prefix caching. Their KV capacity is 479,248
tokens.

The measured vLLM process used the default list of 51 CUDA graph capture
sizes. No custom compilation configuration was passed.

## AGENT-2 selection

We used the same fixed sample of 200 documents to compare two candidate
queries. The sample contains 162 trajectories, covers turns 5 through 50, and
has a mean length of 9,483 tokens. Qwen3 32B fp8 provided the reference labels,
and Qwen3 4B fp8 provided the benchmark model labels.

We set the acceptance rule before reading the labels:

* Qwen3 32B selectivity had to be between 25% and 60%.

* The difference between the two models had to be no more than 10 percentage
  points.

* The models had to agree on at least 75% of documents.

The first candidate asked whether the agent identified a specific and
plausible cause of the issue. Qwen3 32B labeled 56% true, and Qwen3 4B labeled
81% true. The models agreed on 73% of documents. The 25 percentage point gap
and low agreement failed the rule, so we rejected the query.

The second candidate asked whether the agent implemented a plausible fix.
Qwen3 32B labeled 30.5% true, and Qwen3 4B labeled 40.0% true. The models agreed
on 90.5% of documents. The 9.5 percentage point gap and 90.5% agreement passed
the fixed rule, so AGENT-2 uses the second query. Its 30.5% selectivity was
slightly below the narrower 35% to 60% range predicted for this candidate.

The two sample runs cost $1.544 including model startup. The rejected candidate
cost $0.840, and the accepted candidate cost $0.704.

## Full labels

The full labeling prediction was 11 to 14 minutes including startup, with a
cost of $0.72 to $0.92. The run took 808.38 seconds, or 13.47 minutes, and cost
$0.8868 including startup. The result was inside both predicted ranges.

Qwen3 32B produced the following labels:

* AGENT-1 has 571 true documents out of 1,772, which is 32.22% selectivity.

* AGENT-2 has 537 true documents out of 1,772, which is 30.30% selectivity.

The model used 527.28 seconds after 259.70 seconds of startup. It processed
3,576 requests and 35,211,254 prompt tokens. A repeated sample of 32 labels had
zero answer changes when submitted in reverse order.

The complete collection has 21 predicates. It reused 19 unchanged label sets
from the prior collection and added the two corrected agent label sets.

## Benchmark prediction

The prediction recorded before the benchmark was:

* Quail would take about 238 seconds per query.

* Stock vLLM would take 132 to 135 seconds per query.

* Pipelined vLLM would take about 98 seconds per query.

* Both vLLM configurations would serve about 68% of prompt tokens from KV.

* AGENT-2 accuracy would improve because Qwen3 4B and Qwen3 32B agreed on 90.5%
  of the fixed sample.

## Per-query result

The runtime and KV predictions held. Stock vLLM was slightly faster than its
predicted range. AGENT-2 accuracy improved from 84.09% for the rejected query
to about 93.6% for the accepted query.

| Query | System | Query time, seconds | Throughput, documents/second | $/query | Accuracy | KV reuse |
|---|---|---:|---:|---:|---:|---:|
| AGENT-1 | Quail | 235.09 | 7.54 | $0.2579 | 75.00% | 0.00% |
| AGENT-1 | Stock vLLM | 129.54 | 13.68 | $0.1421 | 74.15% | 68.22% |
| AGENT-1 | Pipelined vLLM | 96.33 | 18.39 | $0.1057 | 74.15% | 68.22% |
| AGENT-2 | Quail | 235.91 | 7.51 | $0.2588 | 93.68% | 0.00% |
| AGENT-2 | Stock vLLM | 126.95 | 13.96 | $0.1393 | 93.57% | 68.05% |
| AGENT-2 | Pipelined vLLM | 97.24 | 18.22 | $0.1067 | 93.57% | 68.05% |

Figure: plots/agent_prefix_reuse_per_query.png

The primary time and cost numbers exclude model startup. Quail took 30.15
seconds to start, which costs $0.0331. The shared vLLM process took 149.92
seconds to start, which costs $0.1645.

## Aggregate result

| System | Total query time, seconds | Overall throughput, documents/second | Total query cost | Overall accuracy | KV reuse |
|---|---:|---:|---:|---:|---:|
| Quail | 471.00 | 7.52 | $0.5167 | 84.34% | 0.00% |
| Stock vLLM | 256.49 | 13.82 | $0.2814 | 83.86% | 68.13% |
| Pipelined vLLM | 193.57 | 18.31 | $0.2123 | 83.86% | 68.13% |

Figure: plots/agent_prefix_reuse_aggregate.png

Across both queries, stock vLLM was 1.84 times faster than Quail. Pipelined
vLLM was 2.43 times faster than Quail and 1.33 times faster than stock vLLM.

## KV reuse

| Query | System | Fresh prompt tokens | Cached prompt tokens | Cached fraction |
|---|---|---:|---:|---:|
| AGENT-1 | Quail | 17,389,113 | 0 | 0.00% |
| AGENT-1 | Stock vLLM | 5,526,889 | 11,862,224 | 68.22% |
| AGENT-1 | Pipelined vLLM | 5,526,889 | 11,862,224 | 68.22% |
| AGENT-2 | Quail | 17,431,641 | 0 | 0.00% |
| AGENT-2 | Stock vLLM | 5,569,417 | 11,862,224 | 68.05% |
| AGENT-2 | Pipelined vLLM | 5,569,417 | 11,862,224 | 68.05% |

Stock and pipelined vLLM served exactly the same number of prompt tokens from
KV. Their runtime difference therefore did not come from different prefix
reuse. The experiment did not collect a GPU profile, so it does not identify
the exact source of the pipelined vLLM advantage.

Quail reported no cached prompt tokens because each query has one filter and
Quail does not share KV between different document rows. The cumulative traces
share long prefixes across rows, so vLLM avoids about 68.1% of prompt
computation while Quail repeats it.

## Meaning

The experiment provides a realistic query where Quail is slower than stock
vLLM. Quail needs to share prefixes across documents in addition to
token-based admission and KV rewind. The prefix reuse must work across
different row ids whose text starts with the same tokens. Support for exact
duplicate documents would not cover this dataset.

The accepted AGENT-2 query also provides a useful quality benchmark. All three
methods reached about 93.6% accuracy. Qwen3 4B recall was 98.70% for Quail and
98.88% for both vLLM configurations.

## Data

The measured files are on the `quail-results` volume:

* Quail benchmark:
  `/results/benchmarks/quailb/runs/qb_20260831T062218Z_1192cd76/20260831T062218Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families.json`

* Stock vLLM benchmark:
  `/results/stock_vllm/20260831T062218Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json`

* Pipelined vLLM benchmark:
  `/results/pipelined_vllm/20260831T062218Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json`

* Combined benchmark manifest:
  `/results/benchmarks/quailb/family-runs/20260831T062218Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json`

* AGENT-1 labels:
  `/results/ground_truth/quailb/schema_v1/label_sets/agent/recovered_after_unsuccessful_approach/ls_52eed584f81b5065097d7657ac0bb99c/labels.parquet`

* AGENT-2 labels:
  `/results/ground_truth/quailb/schema_v1/label_sets/agent/implemented_plausible_fix/ls_d1780819414fd20753660bce03f10a0a/labels.parquet`

* Accepted AGENT-2 sample:
  `/results/ablations/agent2_implemented_fix_sample.json`

* Rejected AGENT-2 sample:
  `/results/ablations/agent2_identified_cause_sample.json`

* Shared sample rows:
  `/results/ablations/agent2_identified_cause_sample_rows.parquet`

* Corrected agent dataset:
  `/results/ground_truth/quailb/schema_v1/corpora/c_1aa2c4f0d0b6c816fd37aa5748c33341/agent_traces.parquet`

* Complete ground truth collection:
  `/results/ground_truth/quailb/schema_v1/collections/gt_77bb8b128743a79aedddaa24c808c3f8`

The benchmark family function call was
`fc-01M1B7QEBFJMFQYX2JPE5KVHEA`. The full labeling function call was
`fc-01M1B6F90WVRAC7975KYDNCVSS`. The collection build function call was
`fc-01M1B7NYTQA41B8R51MP132SJV`.

## Rebuild

Rebuild both figures with `reports/make_agent_prefix_reuse_plots.py`. Its
docstring contains every `modal volume get` command needed to pull the source
files.
