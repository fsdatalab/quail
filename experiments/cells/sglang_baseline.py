"""Measure FEV-9 with the current SGLang adapter on one Modal H100.

    uv run modal run --detach experiments/cells/sglang_baseline.py \
      2>&1 | tee /tmp/quail-sglang-baseline.log

Results, answers, GPU identity, and process cleanup are saved on quail-results.
"""

import json
from datetime import datetime, timezone

from quail.bench.quailb_parallel import app, run_sglang_query_family


@app.local_entrypoint()
def check():
    run_label = (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                 + '-sglang-suffix-major')
    prediction_text = (
        'Suffix-major pair ordering without tiles or blocking request slices '
        'should restore fresh computation near the earlier 4.10 million '
        'tokens, compared with 20.39 million for anchor-major submission. '
        'Expect runtime closer to 138.26 seconds than 304.92 seconds and '
        'answer agreement near 69.08 percent. All FEV-9 anchors fit in the '
        'previous tile budget, so removing tiles should preserve their '
        'ordering.'
    )
    run_dir = f"/results/benchmarks/quailb/{run_label}"
    call = run_sglang_query_family.spawn(
        model='qwen3-4b-fp8', sf=0.1, query_ids_csv='FEV-9',
        run_dir=run_dir,
        ground_truth_collection='gt_77bb8b128743a79aedddaa24c808c3f8',
    )
    print(f'function call id: {call.object_id}', flush=True)
    print(f'prediction: {prediction_text}', flush=True)
    result = json.loads(call.get())
    print(f'run directory: {run_dir}', flush=True)
    print(f'process cleanup: {result["process_cleanup"]}', flush=True)
    print(f'comparison volume path: {run_dir}/{result["result_path"]}',
          flush=True)
