"""Profile one QUAIL-B query's host time on Modal.

Runs the query through the bench in-process under cProfile and prints
the functions with the most cumulative and own time, so the host work
between forward passes can be read off. The stats file goes to the
results volume.

    uv run modal run --detach experiments/cells/profile_query.py::profile
        --query BIO-5 --label-scoring trie_paths
"""

from quail.bench.quailb_parallel import (
    DATA_DIR,
    VOLUMES,
    app,
    ensure_data,
    image,
    results_vol,
)

PROFILE_DIR = "/results/ablations/profiles"


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def profile_query(query_id: str, sf: float, collection_id: str,
                  label_scoring: str, top: int) -> str:
    """Run the query under cProfile; returns the top functions as text."""
    import cProfile
    import io
    import pstats
    import tempfile
    from pathlib import Path

    from quail import EngineConfig
    from quail.bench.quailb import run_suite

    config = EngineConfig(gpus=1, model="qwen3-4b-fp8", backend="quail",
                          device="h100-sxm", label_scoring=label_scoring,
                          gpu_timing=True)
    output = Path(tempfile.mkdtemp()) / "run"
    profiler = cProfile.Profile()
    profiler.enable()
    suite = run_suite([query_id], sf=sf, config=config,
                      data_dir=Path(DATA_DIR) / f"sf{sf}",
                      ground_truth_collection=collection_id or None,
                      output_dir=output)
    profiler.disable()
    Path(PROFILE_DIR).mkdir(parents=True, exist_ok=True)
    stats_path = Path(PROFILE_DIR) / f"{query_id}-{label_scoring}.prof"
    profiler.dump_stats(str(stats_path))
    results_vol.commit()
    text = io.StringIO()
    stats = pstats.Stats(profiler, stream=text)
    for order in ("cumulative", "tottime"):
        text.write(f"\n== top {top} by {order}\n")
        stats.sort_stats(order).print_stats(top)
    query = next(q for q in suite["queries"] if q["id"] == query_id)
    measures = query["measurements"]
    nodes = {k: (v.get("gpu_s"), v.get("chunks"))
             for k, v in measures["node_metrics"].items()
             if k.startswith("ai-")}
    header = (f"{query_id}: wall {measures['wall_s']:.2f} s, "
              f"model {measures['model_wall_s']:.2f} s, gpu per node {nodes}, "
              f"stats at {stats_path}\n")
    return header + text.getvalue()


@app.local_entrypoint()
def profile(query: str = "BIO-5", sf: float = 0.1,
         label_scoring: str = "trie_paths", top: int = 35):
    data = ensure_data.spawn(sf, [query], "")
    print(f"function call id: {data.object_id} (data)", flush=True)
    collection = data.get()
    call = profile_query.spawn(query, sf, collection, label_scoring, top)
    print(f"function call id: {call.object_id} (profile {query})", flush=True)
    print(call.get(), flush=True)
