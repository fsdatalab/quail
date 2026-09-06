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
    run_label = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-sglang-baseline-redesign'
    prediction = (
        'Removing submission barriers should reduce FEV-9 runtime below 138.26 seconds '
        'unless the common anchor-major order increases prefix computation enough to dominate. '
        'Expect answer agreement near 69.08 percent; check the saved answers because batching can change them. '
        'The driver should finish without a Modal heartbeat timeout.'
    )
    call = run_sglang_query_family.spawn(
        model='qwen3-4b-fp8', sf=0.1, lf=1, query_ids_csv='FEV-9',
        run_label=run_label, prediction=prediction,
        ground_truth_collection='gt_77bb8b128743a79aedddaa24c808c3f8',
    )
    print(f'function call id: {call.object_id}', flush=True)
    print(f'prediction: {prediction}', flush=True)
    result = json.loads(call.get())
    for method, suite in result['suites'].items():
        print(f'{method} result volume path: {suite["aggregate_volume_path"]}', flush=True)
    print(f'process cleanup: {result["process_cleanup"]}', flush=True)
    print(f'comparison volume path: {result["result_volume_path"]}', flush=True)
