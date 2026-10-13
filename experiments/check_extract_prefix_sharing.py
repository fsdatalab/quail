"""Compare extraction prefix sharing on one H100 with saved answer tables.

    uv run modal run --detach -m experiments.check_extract_prefix_sharing \
        2>&1 | tee /tmp/check_extract_prefix_sharing.log

Results live on quail-results. Retrieve the return value with
modal.FunctionCall.from_id using the printed function call ID.
"""

import hashlib
import json
import time
from pathlib import Path

import modal

from quail.bench.images import gpu_image

app = modal.App("quail-milestone1")
VOLUMES = {
    "/root/.cache/huggingface": modal.Volume.from_name("quail-hf-cache"),
    "/root/.cache/kernels": modal.Volume.from_name("quail-kernel-cache"),
    "/results": modal.Volume.from_name("quail-results"),
}


@app.function(image=gpu_image(), gpu="H100!", memory=98304, timeout=1800,
              volumes=VOLUMES)
def compare(run: str, diagnose: bool = False):
    """Check one warm query with prefix sharing disabled and enabled."""
    import pyarrow.parquet as pq

    import quail
    from quail.backends.quail.executor.extract.decode import Decoder
    from quail.backends.quail.executor.operators import extract
    from quail.bench import quailb
    from quail.planner.plan import EngineConfig
    from quail.specs import H100_USD_PER_HOUR
    from quail_b import get_query
    from quail_b.minimum import token_metrics

    out = Path(f"/results/ablations/extract-prefix-sharing-{run}")
    out.mkdir(parents=True)
    tables = {"contracts": pq.read_table(
        "/results/quailb_data/sf1.0/contracts.parquet")}
    spec = get_query("CUAD-1")
    original = extract.run_stages
    original_choose = Decoder.choose
    enabled = True
    decisions = {}
    prefixes = {}
    targets = {index: value for index, value in enumerate(
        tables["contracts"].column("id").to_pylist())
        if value in ("ct69", "ct112", "ct125")}

    def choose(decoder, doc, question, row):
        identity = targets.get(doc.key[1])
        if identity is None or name == "warmup":
            return original_choose(decoder, doc, question, row)
        written = list(doc.answers[question].written)
        result = original_choose(decoder, doc, question, row)
        decisions.setdefault(name, []).append({
            "document": identity, "written": written,
            "row": [float(value) for value in row],
            "chosen": None if result is None else int(result),
            "done": doc.answers[question].done})
        return result

    def run_stages(*args, **kwargs):
        tree = kwargs.get("prefix_tree")
        if diagnose and tree is not None:
            prefixes[name] = [{"document": identity,
                               "shared_tokens": int(tree.shared[index]),
                               "parent": tree.parent[index]}
                              for index, identity in targets.items()]
        if not enabled:
            kwargs["prefix_tree"] = None
        return original(*args, **kwargs)

    extract.run_stages = run_stages
    if diagnose:
        Decoder.choose = choose
    stores = {}
    records = []
    answers = []
    try:
        with quail.Session(EngineConfig(
                model="qwen3-4b-fp8", device="h100-sxm")) as session:
            for name, table in tables.items():
                session.register(name, quail.DocumentProvider.from_table(
                    table, id_col="id"))
            for name, enabled in (("warmup", True), ("unshared", False),
                                  ("shared", True)):
                output = quailb.run_query(session, spec, tables)
                metrics = token_metrics(spec, output, tables, stores)
                report = output.measurements
                borrowed = sum(score["borrowed_prefix_tokens"]
                               for score in report["backend_metrics"]["scores"])
                record = dict(
                    configuration=name, query_s=output.runtime_s,
                    boot_s=report["boot_s"], borrowed_prefix_tokens=borrowed,
                    **metrics,
                    kv_regret_percent=100 * metrics["regret_tokens"]
                    / metrics["fresh_tokens"],
                    input_tokens_per_second=metrics["input_tokens"]
                    / output.runtime_s,
                    cost_usd=output.runtime_s / 3600 * H100_USD_PER_HOUR)
                records.append(record)
                table = output.extract_answers["extract-1"]
                pq.write_table(table, out / f"{name}.parquet")
                if name != "warmup":
                    answers.append(table)
                (out / f"{name}-plan.json").write_text(
                    json.dumps(report["executed_plan"], indent=2))
                print(json.dumps(record), flush=True)
                VOLUMES["/results"].commit()
        before, after = records[1:]
        summary = dict(
            query="CUAD-1", scale_factor=1.0, model="qwen3-4b-fp8",
            gpus=1, device="h100-sxm", documents=len(tables["contracts"]),
            prediction="Same answers; roughly 3,000 fewer fresh prefix tokens.",
            timing="Benchmark query time including preparation; startup excluded.",
            baseline="Disable only the extraction scheduler's prefix tree.",
            records=records, answers_equal=answers[0].equals(answers[1]),
            tokens_saved=before["fresh_tokens"] - after["fresh_tokens"])
        if diagnose:
            (out / "decisions.json").write_text(json.dumps(decisions))
            summary["target_prefixes"] = prefixes
        sources = {}
        for relative in ("quail/planner/physical_rules.py", "quail/cost/extract.py",
                         "quail/backends/quail/executor/operators/extract.py",
                         "quail/backends/request.py", "quail/bench/quailb.py"):
            source = Path("/root") / relative
            sources[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
        summary["source_sha256"] = sources
        (out / "CUAD-1.json").write_text(json.dumps(summary, indent=2))
        VOLUMES["/results"].commit()
        if not summary["answers_equal"] and not diagnose:
            raise AssertionError("Prefix sharing changed extraction answers")
        if before["borrowed_prefix_tokens"] or not after["borrowed_prefix_tokens"]:
            raise AssertionError("Prefix sharing did not follow the configuration")
        if summary["tokens_saved"] != after["borrowed_prefix_tokens"]:
            raise AssertionError("Fresh-token saving differs from borrowed tokens")
        return summary
    finally:
        extract.run_stages = original
        Decoder.choose = original_choose


@app.local_entrypoint()
def main(diagnose: bool = False):
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    call = compare.spawn(run, diagnose)
    print(f"function call id {call.object_id}", flush=True)
    print(f"results /results/ablations/extract-prefix-sharing-{run}/", flush=True)
