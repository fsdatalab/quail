r"""Measure FEV-10, a join over pairs, against FEV-5 and FEV-9 on one H100.

FEV-10 is FEV-5 with an ordinary equality in the join: SUPPORT is asked
only of a claim and its own Wikipedia page. The queries run through the
standard family runner in one container on one GPU, scored against the
reference labels, with FEV-5 as the cross join twin and FEV-9 as the
check that joins without conditions are unchanged. FEV-1 runs first,
as in the saved suite, so the first query after the cold boot is not
one being compared: in a first attempt FEV-5 ran first and its filter
took 8.8 seconds instead of 0.3.

    uv run modal run --detach experiments/cells/pair_join.py \
      2>&1 | tee /tmp/quail-pair-join.log

Results, answers, and rows are saved on the quail-results volume under
the run directory the entrypoint prints.
"""

import json
from datetime import datetime, timezone

from quail.bench.quailb_parallel import app, ensure_data, run_query_family

QUERY_IDS = ("FEV-1", "FEV-5", "FEV-10", "FEV-9")
GROUND_TRUTH = "gt_77bb8b128743a79aedddaa24c808c3f8"
PREDICTION_TEXT = (
    "FEV-5 evaluated about 61,700 pairs in 13.35 seconds with 1,515,283 "
    "fresh tokens in the saved suite run. FEV-10 asks the same question "
    "of at most one pair per surviving claim, about 200 to 300 pairs, "
    "so its join work nearly disappears: about 160,000 fresh tokens, "
    "and 1 to 2.5 seconds against FEV-5's 13.35. Its answers on the "
    "shared pairs match FEV-5's bit for bit, its rows are FEV-5's rows "
    "restricted to the same page, and its output precision rises far "
    "above FEV-5's 1.05% while recall stays near 96%. FEV-9 is "
    "unchanged: about 39 seconds, 4,306,910 fresh tokens, zero "
    "recomputed KV tokens."
)


@app.local_entrypoint()
def check():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = f"/results/benchmarks/quailb/{stamp}-pair-join"
    print(f"prediction: {PREDICTION_TEXT}", flush=True)
    data_call = ensure_data.spawn(0.1, list(QUERY_IDS), GROUND_TRUTH)
    print(f"function call id: {data_call.object_id} (data)", flush=True)
    collection = data_call.get()
    call = run_query_family.spawn(
        model="qwen3-4b-fp8", sf=0.1, query_ids_csv=",".join(QUERY_IDS),
        run_dir=run_dir, ground_truth_collection=collection,
        include_baselines=False)
    print(f"function call id: {call.object_id} (fever, Quail)", flush=True)
    result = json.loads(call.get())
    print(f"run directory: {run_dir}", flush=True)
    print(f"gpu uuids: {result['gpu_uuids']}", flush=True)
    for query in result["suites"]["quail"]["queries"]:
        print(f"{query['id']}: {json.dumps(query['metrics'])}", flush=True)
