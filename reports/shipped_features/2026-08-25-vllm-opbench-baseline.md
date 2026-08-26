# vllm-opbench: a second stock-vLLM baseline

## What changed

Added `baselines/vllm_opbench/`, a second stock-vLLM baseline
alongside the existing `baselines/stock.py`. It runs quail's own
document sets and predicates through plain vLLM: offline batched
`llm.generate()` (not one request at a time), the base checkpoint
quantized to fp8 at vLLM load time (not quail's own pre-quantized
checkpoint), and Prometheus metrics plus a KV-cache "oracle regret"
calculation on every run.

## Why

A second reference point for "how fast is generic vLLM," varying a
different thing than `stock.py` does: `stock.py` keeps quail's own
checkpoint and varies only the serving code; vllm-opbench also varies
the checkpoint, the way most vLLM users would actually run it.

## Fixes made during review

A line-by-line review, then an adversarial pass, found and fixed 6
bugs. The two that mattered: results now save after every query
instead of only at the end (a crash used to silently drop
already-computed results), and every saved answer now records which
document, or document pair, it came from.

## Before/after (full report: `2026-08-25-vllm-opbench-baseline.md`)

All 4 queries now run successfully at both 4B and 32B, fp8, sf=0.1.
This baseline's own correctness (right documents, right predicates,
right answers) is verified.

A rerun against QUAIL-B ground truth explains an earlier instability
in quail's own join-query reference numbers: 4B answers ~99-100% TRUE
on both join predicates regardless of serving path (near-zero
precision, not a real judgment) - but is fine on filters, scoring
100% precision / 90.2% recall on filter-reports (F7), so the
instability is specific to these two join predicates, not to 4B in
general. 32B has not been rechecked against the current ground truth
yet; its 98.5% accuracy figure predates a template fix that moved the
label rate it's scored against, and is expected to look worse once
rechecked.
