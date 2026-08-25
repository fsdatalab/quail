# Packed unified joins

## Result

Packed unified joins put all partners for an anchor into the same forward
pass. They do not need one forward pass for each partner.

The new benchmark used the same three BioDEX query sets as the original
attention path figure. Packed unified was slower than `merge_quant` on all
three query sets. The difference ranged from 2.0% to 12.1%, so joins should
continue to select `merge_quant`.

Figure: plots/attention_paths.png

The plot reads `results/attention_paths.json` for the original filter queries
and `results/packed_unified_join.json` for the original join queries.

## Setup

- The model was Qwen3 4B fp8 on one H100.

- The chunk budget was 110,376 fresh tokens.

- The KV arena held 313,206 rows, with 16 rows in each page.

- The first join query set had 10 BioDEX report anchors and 256 reaction term
  partners for each anchor.

- The second join query set had 1 BioDEX report anchor and 2,560 reaction term
  partners.

- The third join query set had 100 BioDEX report anchors and 256 reaction term
  partners for each anchor.

The benchmark used the same `biodex_sample` report and term slices that the
original figure used. The benchmark changed only the unified join packing.

## Prediction

Packed unified should remain slower than `merge_quant` on all three query
sets, but the difference should stay below 10%.

The measured direction matched the prediction. The 1 by 2,560 query set was
12.1% slower, so the predicted upper bound was wrong for that query set.

## Measured result

| Join query set | `merge_quant` | packed unified | Packed unified compared with `merge_quant` | Forward passes |
|---|---:|---:|---:|---:|
| 10 by 256 | 11.04 microseconds per token | 11.88 | 7.6% slower | 1 |
| 1 by 2,560 | 12.27 microseconds per token | 13.76 | 12.1% slower | 1 |
| 100 by 256 | 10.75 microseconds per token | 10.96 | 2.0% slower | 10 |

The 10 by 256 result is the mean of two measured runs. The other two query
sets each have one measured run, which matches the original figure's run
count.

Packed unified used between 7,682 and 7,901 temporary pages in the largest
chunk for each query set. Those pages were claimed while the chunk was packed
and returned after the forward pass was queued. The packer did not reserve a
fixed number of suffix pages for each anchor.

## Correctness

A separate check compared the packed 10 by 256 result with unified batches
that contained one partner per anchor. The check found 0 pair differences
across all 2,560 pairs. Both layouts returned 2,558 TRUE answers.

The packed layout therefore keeps every partner independent while putting
all partners into the same FA3 batch.

## What the numbers mean

The earlier `unified (waves)` bars measured the wrong packing restriction.
The old implementation used 257 forward passes for the 10 by 256 query set
and 2,561 forward passes for the 1 by 2,560 query set. Packed unified uses one
forward pass for both query sets.

Packed unified removed the forward pass problem, but it did not beat
`merge_quant`. Packed unified copies the anchor's final partial page for each
partner and gives FA3 one page table row for each pair. `merge_quant` avoids
those private page rows and was faster on every measured query set.

## Result files

- The committed summary is `results/packed_unified_join.json`.

- The raw 10 by 256 result is at
  `/results/ablations/join_attention_paths_packed_10x256.json`. Its Modal
  function call id is `fc-01M0VMT3Y5MBYQAP196TX1HB5Y`.

- The raw 1 by 2,560 result is at
  `/results/ablations/join_attention_paths_packed_1x2560.json`. Its Modal
  function call id is `fc-01M0VMT42JBCZEBRR5AGG6KF26`.

- The raw 100 by 256 result is at
  `/results/ablations/join_attention_paths_packed_100x256.json`. Its Modal
  function call id is `fc-01M0VMT46XFKRSXJKV8YJK8CKC`.

- The raw correctness result is at
  `/results/ablations/packed_unified_join_parity_10x256.json`. Its Modal
  function call id is `fc-01M0VMHN3H61RNTNK79VP1W73B`.
