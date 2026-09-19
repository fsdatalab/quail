# Why the BIO speedups shrank: prompt tokens, a rule change, and the host

Date: 2026-09-19.

The all-chat comparison in pull request 128 shows Quail 4.99x faster
than pipelined stock vLLM on BIO-2, compared with 8.37x in the
September 12 report. This note traces that change to three causes
and rules out the two that would have mattered. Nothing in Quail or
vLLM got slower or faster. The prompt got bigger, the benchmark's
recomputed KV rule changed, and vLLM ran on a different machine.

## What changed

- `reports/quailb-comparison.md` and its generator
  `reports/make_quailb_comparison_plots.py` on pull request 128 now
  state the recomputed KV rule the benchmark uses and say what
  changed vLLM's BIO-2 time, with the measurements below. The
  earlier text said the cause was unknown.
- Pull request 124's description names the quail-bench commit its
  pin carries and says that the September 12 recomputed KV figures
  used an older rule and are not comparable.
- `experiments/bio2_prompt_layout.py` is a new Modal cell on the
  `quail-milestone1` app, at commit `1f35316` of branch
  `claude/focused-carson-huqcvm`. It runs BIO-2 through the real
  runner on pipelined stock vLLM in one container, once per prompt
  layout, on a fresh engine each time with BIO-1 run first. It
  refuses a container whose GPU starts hot or throttled and samples
  the GPU's temperature and clock every 30 seconds.

## Quail got slower because every pair got 9 more tokens

The non-thinking chat format closes each prompt with
`<|im_end|>`, `<|im_start|>assistant`, and an empty think block: 9
tokens in Qwen3's tokenizer. They come after each pair's own text, so
no engine can share them across pairs. BIO-2 has the same 563,500
pairs in both runs. Fresh tokens rose by 5.08 million for Quail and
for vLLM alike, 9.0 per pair.

Quail's speed did not change: 81,200 fresh tokens per second on
September 12 and 82,600 on September 18. With 49% more tokens it took
46% longer, 127.75 seconds to 187.01. The SoL estimate rose from 62.0
to 93.2 seconds, so the task itself grew by half. Had vLLM's time not
moved, the multiplier would be 5.72x.

## The September 12 recomputed KV figures were an accounting artifact

Between the two reports the quail-bench pin moved from `5f7ce82` to
`fc27f35`. quail-bench pull request 17, in between, changed the
minimum a run's requests need with unlimited KV: it now counts each
pair's partner label, partner document, and answer cue once per pair.
The older rule counted the label once per anchor and shared partner
document prefixes across pairs.

Checked on the CPU against the pinned BioDEX rows and the Qwen3
tokenizer: BIO-2 has 500 reports and 1,127 terms, the label is 6
tokens, and the terms share 1,537 tokens of prefix. The older rule
understates the minimum by 8,293 tokens per anchor, 4,146,500 in
total. The September 12 report listed Quail's BIO-2 recomputed KV as
4,148,977. The difference, 2,477, is BIO-1's figure, the filter-stage
residual. So that number was entirely the rule, not recomputation.
The same arithmetic reproduces both BIO-3 pair counts to the pair.

With the artifact removed, vLLM's real KV reuse is the same in both
runs: 154,747 recomputed tokens on September 12 and 157,341 on
September 18 for BIO-2.

## vLLM got faster because it ran on a different machine

The two runs were on different physical GPUs, `GPU-4afe15c5` on
September 12 and `GPU-55ea158d` on September 18, from the saved
family records. vLLM's time on this join is per-request scheduler
work on the CPU, about 2 ms per pair, so it depends on the host.

Prediction, stated before the run: the raw and chat layouts take the
same time per pair within 5% in one container, with chat no faster
than raw. A 13% chat advantage would instead mean the batch
composition, fewer requests per 25,305-token step, explains the
September 18 time.

Measured, one container, `GPU-3103ab9e`, no throttled sample in 181
readings, 31 to 69 C:

| Round | Layout | Wall time | Per pair | Fresh tokens |
|---|---|---:|---:|---:|
| 1 | raw | 1212.9 s | 2.152 ms | 10,391,144 |
| 1 | chat | 1206.6 s | 2.141 ms | 15,604,712 |
| 2 | chat | 1182.1 s | 2.098 ms | 15,604,712 |
| 2 | raw | 1192.0 s | 2.115 ms | 10,391,144 |

Chat over raw per pair is 0.993. The layout does not change vLLM's
time. The chat runs' fresh tokens equal the September 18 run's
exactly. Across the three machines the raw layout has run on, the
join took 1069 seconds (September 12), 1163 seconds (a first attempt
on September 18, cold engine), and 1213 seconds, a 13% spread with
nothing but the host changing.

- Result: `/results/ablations/bio2-prompt-layout-20260918T155056Z/result.json`,
  function call `fc-01M2TKCMJQVM0HV104NVCHTKY5`.
- The cold-engine raw time: `/results/ablations/bio2-prompt-layout-20260918T142630Z/joins.json`,
  function call `fc-01M2TEJJY4Z2G2DGG25NE8F28B`. That attempt ran all
  joins on one engine; vLLM 0.26.0's engine core crashed in a model
  step at the start of the second join and Modal's log rate limit
  dropped the exception, so the cell now uses a fresh engine per join
  and saves each child's output to the volume.
- A further attempt, `fc-01M2TG6KG63GSZTN2X3BRZ43AD`, was cancelled
  after Modal's health monitor reported its GPU at 92 C with the
  clock cut 69% on average. The cell now refuses such a container.

## Accuracy against the dataset's own labels

The report's answer agreement is measured against Qwen3 32B answers,
and those changed format at the same time as the 4B's. BioDEX records
each report's reactions, which is exactly BIO-2's question, so both
formats can be scored against that list: 2,543 true pairs of 563,500.

| Model, prompt | Said TRUE | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 4B, raw (September 12 run) | 116,156 | 1.9% | 84.9% | 3.6 |
| 4B, chat (September 18 run) | 2,045 | 20.9% | 16.8% | 18.7 |
| 32B, raw (September 12 reference) | 19,144 | 8.2% | 61.8% | 14.5 |
| 32B, chat (September 18 reference) | 17,974 | 9.7% | 68.4% | 17.0 |

The chat format made the 4B a usable classifier on this query, and
it now scores about the same as the 32B reference, with the opposite
failure: it misses most true reactions, the reference reports far too
many. The 32B says TRUE seven times more often than the data does, so
agreement with it rewards saying FALSE; that is why the report's
agreement column rose on the joins while recall against the
reference collapsed. This is one query with a strict answer key: a
synonym of a recorded reaction counts as wrong.

Answers scored: `quail/biodex/BIO-2/joins-0.parquet` under
`/results/benchmarks/quailb/family-runs/20260912T225100Z-902686c5/`
and `/results/benchmarks/quailb/family-runs/20260918T060700Z-biodex-chat/`;
references `ls_558442b4193e9a48bfe1aea9bc87a66a` and
`ls_4c0b69af26ad590e1b64ac5ffebf2484` under
`/results/ground_truth/quailb/schema_v1/label_sets/biodex/report_experienced_reaction/`;
tables `/results/quailb_data/sf0.1/reports.parquet` and `terms.parquet`.

## Not done

- Whether the chat format ranks pairs better or only moves the TRUE
  and FALSE boundary is not measured. Quail computes both logits per
  pair; saving the margin for one BIO-2 run per format would give the
  area under the ranking curve and the best F1 over thresholds.
- Of the 24 fresh tokens a BIO-2 pair costs, 19 are fixed text: the
  6-token partner label, the 4-token answer cue, and the 9-token chat
  closing. The label can move into the shared anchor prefix with no
  change to what the model sees, once quail-bench's minimum rule
  counts it there. Dropping the cue or the empty think block needs an
  accuracy check first. At Quail's measured 82,600 fresh tokens per
  second, the first two cuts would take BIO-2 from 187 to about 119
  seconds.
