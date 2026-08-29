# Legacy stock vLLM experiment

This package is not part of the current QuailB benchmark. The current runner
is `baselines/stock_vllm`. Its stage-major filter loop is in that package. It
uses `baselines/stock.py` for pipelined filters and for the join code shared by
both current vLLM configurations.

`baselines/old_stock` remains in the repository only to reproduce the older
operator experiment. Run it manually with:

```bash
uv run modal run -m baselines.old_stock.run
```
