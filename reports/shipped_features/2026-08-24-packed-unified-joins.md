# Packed unified joins

Packed unified joins put several suffixes for one anchor in the same
forward pass. Each suffix has its own row of KV arena page IDs. The row
uses the shared full anchor pages and temporary pages for the anchor
remainder and suffix KV.

The earlier unified join benchmark processed one partner for each anchor
in every forward pass. The packed layout processed all partners that fit
in the chunk. A 10 by 256 join therefore used 1 forward pass instead of
257.

Packed unified was correct but slower than `merge_quant` on the three
measured BioDEX query sets. Packed unified was 7.6%, 12.1%, and 2.0%
slower. Joins therefore continue to use `merge_quant`.

The split attention path and its comparison hook were also removed. Quail
now has the `unified` and `merge_quant` attention modes.

See `reports/2026-08-24-packed-unified-joins.md` (removed 2026-08-29; git history) and
`results/packed_unified_join.json` for the setup and measured results.
