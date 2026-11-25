# Civil Comments demo

The demo filters toxic comments from [Civil Comments][dataset] and joins
them to 30 classification criteria. It samples 10,000 comments by default
and runs [DiffusionGemma 26B-A4B fp8][model] on one H100.

Run it from the repository root. On a local H100:

```bash
uv run python demos/civil_comments/quail_backend.py \
  --limit 10000 --gpus 1 \
  --output results/civil-comments/quail/local
```

On Modal:

```bash
uv run modal run --detach demos/civil_comments/quail_modal.py \
  --limit 10000 --gpus 1 \
  2>&1 | tee /tmp/civil-comments-quail.log
```

The script saves the plan in `plan.txt`, the matching pairs in
`retained.parquet`, and query time, GPU cost, and F1 scores in
`summary.json`. Query time and cost exclude model startup.

[dataset]: https://huggingface.co/datasets/pietrolesci/civilcomments-wilds
[model]: https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic
