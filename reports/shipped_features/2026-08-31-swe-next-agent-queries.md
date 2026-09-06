# SWE-Next agent trace queries

## What changed

QuailB now includes 17,718 cumulative trace snapshots from
SWE-Next-SFT-Trajectories. The dataset builder saves one document after every
fifth assistant turn. Each later document contains the earlier trace prefix.

QuailB has two new filter queries. AGENT-1 asks whether the agent recovered
after an unsuccessful approach. AGENT-2 asks whether the agent implemented a
plausible fix for the reported issue. QuailB now has 32 default queries across
five document sets.

## Why

The Agent queries measure reuse across different documents that share long
token prefixes. Quail keeps KV for one document identity, so it does not reuse
the shared text across cumulative snapshots. Stock vLLM and pipelined vLLM use
automatic prefix caching across requests.

## Measured result

At scale factor 0.1, both Agent queries process 1,772 documents. Pipelined
vLLM took 97.99 seconds on AGENT-1 and 98.74 seconds on AGENT-2. Quail took
239.75 seconds and 240.74 seconds. The vLLM methods served about 68% of their
prompt tokens from KV, while Quail served none of the shared prefixes across
document rows from KV.

The full-run data is on the `quail-results` volume at
`/results/benchmarks/quailb/family-runs/20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json`.
The first agent prefix reuse
report was removed on 2026-09-06; its data stays on the `quail-results`
volume.
