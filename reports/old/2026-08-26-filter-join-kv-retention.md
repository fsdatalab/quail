# Filter to join KV retention

## Setup

This check ran one filter followed by two full joins on one H100! with
Qwen3 4B fp8. One join was forced to use the filtered reports as
its anchor. The second join used another collection as its anchor.

The input had 10 reports, 12 candidates, and 6 labels. The join planner
ran once after all filter answers were known. The runtime used stable
`(alias, document index)` keys for KV.

Data: `/results/runs/run_1787795777696587173.json` on the
`quail-results` Modal volume. The Modal function call was
`fc-01M10EVX1BW60GFP5CHH47011H`.

## Prediction

Every report that passed the filter would remain in KV until the join
that used reports as its anchor. The small input would not need an
eviction. The final rows would equal a separate CPU recombination of the
raw join answer rows.

## Result

Seven reports passed the filter. All 7 stayed in KV and were KV hits at
the report anchored join. No retained document was evicted.

Figure: plots/filter_join_kv_retention.png

The join DP searched 7 states and generated 9 records. It chose the
label to candidate join first and the report to candidate join second.
The engine returned 14 rows. The separate CPU recombination returned the
same 14 rows.

The query took 0.11 seconds after boot and processed 7,563 fresh tokens.
The cold boot took 39.2 seconds and is not included in the query time.

## Meaning

Filter survivors can now cross the filter to join boundary without
recomputing the document prefix. The join order remains fixed after the
post filter DP finishes. This check did not create memory pressure, so it
does not measure the eviction policy under a full arena.
