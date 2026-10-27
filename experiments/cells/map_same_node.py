"""Run Quail and stock vLLM on the same H100 over the map configurations.

One Modal function call runs both engines, one after the other, each
in its own child process so each gets a fresh CUDA context and the
whole GPU: Quail first, then vLLM. Each child records the GPU's UUID,
and the result asserts the two match. The configurations are:

- one short and one long map over the IMDB reviews and over the CUAD
  contracts;
- three maps over one table in one query, which Quail runs as one map
  pipeline and vLLM gets as every (document, instruction) prompt in
  one generate call;
- a classification then a map over the reviews, which Quail runs in
  one pipeline and vLLM gets as each document's two prompts together,
  the classification as one greedy token among the labels' tokens.

Quail decodes one token per round with GPU timing; vLLM decodes
greedily without speculation, with its own CUDA graph capture sizes,
prefix caching on, and its prefix cache reset before each
configuration. Query time excludes model startup for both; startup
is reported separately.

    run_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-map-same-node.log"
    uv run modal run --detach \
      experiments/cells/map_same_node.py::launch_same_node \
      2>&1 | tee "$run_log"

The result JSON lands under /results/ablations/ on the quail-results
volume; the function prints its path.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from bench.quailb_parallel import DATA_DIR, VOLUMES, app, image, results_vol

RESULT_DIR = "/results/ablations"

SHORT_MAP = ("Summarize the negative arguments the writer makes, "
             "in one sentence.")
LONG_MAP = ("Write a detailed summary of the document: the parties or "
            "people involved, the main points they make, and anything "
            "unresolved.")
TOPIC_MAP = "Name the main topic of the document in a few words."

TONE = ("tone", "What is the overall tone of this review?\n\n{0}",
        ("positive", "negative", "mixed"))

# (name, table, column, ((result column, instruction, max answer
# tokens), ...), classification asked before the maps or None)
SAME_NODE_CONFIGURATIONS = (
    ("imdb_short_map", "reviews", "body", (("answer", SHORT_MAP, 64),), None),
    ("imdb_long_map", "reviews", "body", (("answer", LONG_MAP, 256),), None),
    ("cuad_short_map", "contracts", "text", (("answer", SHORT_MAP, 64),), None),
    ("cuad_long_map", "contracts", "text", (("answer", LONG_MAP, 256),), None),
) + tuple(
    (name, table, column, (("negatives", SHORT_MAP, 64),
                           ("summary", LONG_MAP, 256),
                           ("topic", TOPIC_MAP, 32)), None)
    for name, table, column in (("imdb_three_maps", "reviews", "body"),
                                ("cuad_three_maps", "contracts", "text"))) + (
    ("imdb_classify_map", "reviews", "body", (("summary", LONG_MAP, 256),), TONE),)

SINGLE_SUBMISSION = "stock vLLM, operator-at-a-time submission"
MULTI_SUBMISSION = "stock vLLM, pipelining submission"


def _ensure_tables(scale_factor: float, names) -> Path:
    """Write the corpora parquets to the volume, as ensure_data does."""
    import pyarrow.parquet as pq

    import quail_b as benchmark
    from quail_b.data import CORPUS_COLUMNS

    directory = Path(DATA_DIR) / f"sf{scale_factor}"
    directory.mkdir(parents=True, exist_ok=True)
    for name in sorted(set(names)):
        path = directory / f"{name}.parquet"
        if (path.exists() and set(CORPUS_COLUMNS[name])
                <= set(pq.read_schema(path).names)):
            continue
        pq.write_table(benchmark.load_table(name, scale_factor=scale_factor),
                       path)
    results_vol.commit()
    return directory


def answer_stats(answers, ids) -> dict:
    """Return one map's answers by document id, with their counts and lengths."""
    lengths = sorted(len(a or "") for a in answers)
    return {
        "num_documents": len(answers),
        "num_empty_answers": sum(not a for a in answers),
        "answer_chars": {"min": lengths[0], "median": lengths[len(lengths) // 2],
                         "max": lengths[-1]},
        "sample_answers": answers[:3],
        "answers": dict(zip(ids, answers)),
    }


def _shown(value):
    """Return a run without its per-document answers and labels, for the log."""
    if isinstance(value, dict):
        return {key: _shown(item) for key, item in value.items()
                if key not in ("answers", "labels")}
    return value


def classified_query(session, table, column, maps, classify=None):
    """Return the query: an optional classification, then the maps.

    Args:
        session: The Quail session.
        table: The table the operators read.
        column: The document column.
        maps: (result column, instruction, max answer tokens) per map.
        classify: (result column, prompt template, labels), or None.
    """
    import quail

    query = session.docs(table).alias("d")
    names = []
    if classify is not None:
        name, template, labels = classify
        query = query.ai_classify(quail.prompt(template, quail.col(f"d.{column}")),
                                  list(labels), name=name)
        names.append(name)
    for name, instruction, _ in maps:
        query = query.ai_map(f"d.{column}", instruction, name=name)
    return query.select("d.id", *names, *(name for name, _, _ in maps))


def _run_quail(session, table, column, maps, classify, usd_per_hour) -> dict:
    """Run one configuration on Quail and return its measurements.

    Args:
        session: The Quail session.
        table: The table the operators read.
        column: The document column.
        maps: (result column, instruction, max answer tokens) per map;
            several maps go in one query, as one map pipeline.
        classify: (result column, prompt template, labels) of a
            classification asked before the maps, or None.
        usd_per_hour: The GPU price.

    Returns:
        The query's timings and the first map's answer statistics and
        inference metrics; with several maps, "maps" holds each one's;
        with a classification, every document's label and answer.
    """
    from dataclasses import replace

    from quail.physical import AiMap

    query = classified_query(session, table, column, maps, classify)
    caps = {name: cap for name, _, cap in maps}
    plan = query.plan()
    nodes = tuple(replace(node, spec=replace(
        node.spec, max_answer_tokens=caps[node.spec.name]))
        if isinstance(node, AiMap) else node for node in plan.nodes)
    started = time.perf_counter()
    result = query.run(plan=replace(plan, nodes=nodes))
    wall = time.perf_counter() - started
    report = result.report
    inferences = report.get("backend_metrics", {}).get("inferences", [{}]) or [{}]
    table_out = result.collect()
    ids = table_out.column("d.id").to_pylist()
    per_map = {}
    for name, _, _ in maps:
        per_map[name] = answer_stats(table_out.column(name).to_pylist(), ids)
        per_map[name]["inference"] = next(
            (inference for inference in inferences
             if inference.get("output") == name), {})
    measured = {
        "wall_s": round(wall, 2),
        "engine_wall_s": report.get("wall_s"),
        "gpu_s": report.get("gpu_s"),
        "chunks": report.get("chunks"),
        "fresh_tokens": report.get("fresh_tokens"),
        "usd_per_query": (report.get("wall_s") or wall) / 3600 * usd_per_hour,
        "maps": per_map,
    }
    if classify is not None:
        measured["labels"] = dict(zip(ids, table_out.column(classify[0]).to_pylist()))
        measured["answers"] = dict(zip(ids, table_out.column(maps[0][0]).to_pylist()))
    return measured


def _prompts(directory, table, column, instruction, spec) -> list[str]:
    """Render the map prompt text Quail shows, one string per document."""
    import pyarrow.parquet as pq

    from quail.logical.expressions import ColumnRef
    from quail.logical.prompts import bind_map_prompt

    prompt = bind_map_prompt((ColumnRef("d", table, column),), instruction,
                             spec.turn)
    texts = pq.read_table(directory / f"{table}.parquet").column(column)
    return [prompt.preamble + prompt.tail.format(text.as_py())
            for text in texts]


def _run_vllm(client, prompts, cap, keep_answers=False) -> dict:
    """Submit every prompt at once and measure the generation.

    Args:
        client: The booted vLLM client.
        prompts: The prompt texts.
        cap: The most answer tokens, one for every prompt or a list
            with one per prompt.
        keep_answers: Whether the result holds every answer, stripped,
            in prompt order, under "answers".
    """
    params = ([client.decode_params(each) for each in cap]
              if isinstance(cap, (list, tuple)) else client.decode_params(cap))
    started = time.perf_counter()
    outputs = client.generate(prompts, params, use_tqdm=False)
    wall = time.perf_counter() - started
    input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    answers = [output.outputs[0].text for output in outputs]
    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    lengths = sorted(len(answer.strip()) for answer in answers)
    kept = {"answers": [answer.strip() for answer in answers]} if keep_answers else {}
    return {
        **kept,
        "wall_s": round(wall, 2),
        "input_tokens": input_tokens,
        "input_tokens_per_second": round(input_tokens / wall, 1),
        "output_tokens": output_tokens,
        "num_documents": len(prompts),
        "num_empty_answers": sum(not answer.strip() for answer in answers),
        "answer_chars": {"min": lengths[0], "median": lengths[len(lengths) // 2],
                         "max": lengths[-1]},
        "sample_answers": [answer.strip() for answer in answers[:3]],
    }


def _gpu() -> str:
    """Return this process's GPU as nvidia-smi names it: UUID, then model."""
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True).stdout.strip()


def _chosen(only: str) -> tuple:
    """Return the configurations the comma-separated names choose; all when empty."""
    names = {name.strip() for name in only.split(",") if name.strip()}
    unknown = names - {config[0] for config in SAME_NODE_CONFIGURATIONS}
    if unknown:
        raise ValueError(f"unknown configurations: {sorted(unknown)}")
    return tuple(config for config in SAME_NODE_CONFIGURATIONS
                 if not names or config[0] in names)


def _quail(model: str, directory: Path, configurations) -> dict:
    """Run the configurations on Quail; return the GPU, startup, and runs."""
    import quail
    from bench.quailb import register_tables
    from quail.specs import H100_USD_PER_HOUR

    result = {"engine": "Quail", "gpu": _gpu(), "runs": {}}
    with quail.Session(quail.EngineConfig(
            model=model, device="h100-sxm", backend="quail",
            gpu_timing=True)) as session:
        register_tables(session, directory)
        for name, table, column, maps, classify in configurations:
            print(f"quail: running {name}", flush=True)
            run = _run_quail(session, table, column, maps, classify,
                             H100_USD_PER_HOUR)
            # the session boots the engine in its first query; the
            # engine's own wall time excludes that startup
            run["startup_s"] = round(run["wall_s"] - (run["engine_wall_s"] or 0.0), 2)
            result["runs"][name] = run
            print(json.dumps({name: _shown(run)}, default=str), flush=True)
    first = next(iter(result["runs"].values()), {})
    result["startup_s"] = first.get("startup_s")
    return result


def _vllm(model: str, directory: Path, configurations) -> dict:
    """Run the configurations on stock vLLM; return the GPU, boot, and runs."""
    # the engine core runs in process: a spawned core would not see the
    # gigatoken tokenizer registration
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.pop("QUAIL_VLLM_SPECULATIVE", None)
    from quail.baselines.vllm import VLLMEngine
    from quail.specs import DEVICES, H100_USD_PER_HOUR, MODELS

    spec = MODELS[model]
    # the classification prompts are Quail's own, read from its plan
    # before vLLM takes the GPU
    planned = {name: _classify_prompt(model, directory, table, column, maps, classify)
               for name, table, column, maps, classify in configurations
               if classify is not None}
    result = {"engine": "stock vLLM", "gpu": _gpu(), "speculative": None,
              "capture_sizes": "vllm", "runs": {}}
    with ExitStack() as resources:
        # vLLM's own CUDA graph capture sizes
        engine = VLLMEngine(capture_sizes=None)
        settings, boot = engine.boot(spec, [0], DEVICES["h100-sxm"],
                                     resources=resources)
        result["boot"] = boot
        result["engine_settings"] = {
            key: value for key, value in settings["capacity"].items()
            if isinstance(value, (int, float, str, bool))}
        client = settings["client"]
        for name, table, column, maps, classify in configurations:
            print(f"vllm: running {name}", flush=True)
            # each configuration prefills its documents itself, as
            # Quail does: the KV an earlier one cached is dropped
            if client.reset_prefix_cache() is False:
                raise RuntimeError("vLLM did not reset its prefix cache")
            if classify is not None:
                run = _vllm_classify_map(client, directory, table, column,
                                         maps[0], planned[name], spec)
                run["usd_per_query"] = run["wall_s"] / 3600 * H100_USD_PER_HOUR
                result["runs"][name] = run
                print(json.dumps({name: _shown(run)}, default=str), flush=True)
                continue
            per_map = [_prompts(directory, table, column, instruction, spec)
                       for _, instruction, _ in maps]
            # document-major order: a document's prompts arrive together,
            # so prefix caching can share its KV across them
            prompts = [prompt for document in zip(*per_map) for prompt in document]
            caps = [cap for _ in per_map[0] for _, _, cap in maps]
            run = _run_vllm(client, prompts, caps, keep_answers=True)
            answers = run.pop("answers")
            ids = _ids(directory, table)
            run["engine"] = SINGLE_SUBMISSION if len(maps) == 1 else MULTI_SUBMISSION
            run["num_requests"] = len(prompts)
            run["num_documents"] = len(per_map[0])
            run["usd_per_query"] = run["wall_s"] / 3600 * H100_USD_PER_HOUR
            run["maps"] = {
                map_name: answer_stats(answers[index::len(maps)], ids)
                for index, (map_name, _, _) in enumerate(maps)}
            result["runs"][name] = run
            print(json.dumps({name: _shown(run)}, default=str), flush=True)
    return result


def _ids(directory, table) -> list:
    """Return the table's document ids, in the order its prompts are rendered."""
    import pyarrow.parquet as pq

    return pq.read_table(directory / f"{table}.parquet").column("id").to_pylist()


def _classify_prompt(model, directory, table, column, maps, classify) -> dict:
    """Return the classification's prompt tokens and label tokens from Quail's plan."""
    import quail
    from bench.quailb import register_tables
    from quail.physical import AiClassify

    with quail.Session(quail.EngineConfig(model=model, device="h100-sxm")) as session:
        register_tables(session, directory)
        plan = classified_query(session, table, column, maps, classify).plan()
    node = next(node for node in plan.nodes if isinstance(node, AiClassify))
    if any(len(ids) != 1 for ids in node.spec.label_token_ids):
        raise ValueError(f"labels need one token each, got {node.spec.scoring}")
    preamble, tail = node.spec.prompt_token_parts
    return {"preamble": list(preamble), "tail": list(tail),
            "label_tokens": [ids[0] for ids in node.spec.label_token_ids],
            "labels": list(node.spec.labels)}


def _vllm_classify_map(client, directory, table, column, map_config, planned,
                       spec) -> dict:
    """Submit each document's classification and map request at once; measure them.

    The classification is one greedy token among the labels' tokens,
    the label Quail's letters rule picks; the map is greedy text, as in
    the map configurations. Requests arrive document by document, so
    prefix caching can share a document's KV across its two prompts.
    """
    import pyarrow.parquet as pq
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    _, instruction, cap = map_config
    source = pq.read_table(directory / f"{table}.parquet")
    texts = source.column(column).to_pylist()
    ids = source.column("id").to_pylist()
    tokenizer = client.llm.get_tokenizer()
    classify_prompts = [TokensPrompt(prompt_token_ids=planned["preamble"]
                                     + tokenizer.encode(text, add_special_tokens=False)
                                     + planned["tail"])
                        for text in texts]
    map_prompts = _prompts(directory, table, column, instruction, spec)
    prompts = [prompt for pair in zip(classify_prompts, map_prompts) for prompt in pair]
    label_params = SamplingParams(max_tokens=1, temperature=0.0,
                                  allowed_token_ids=planned["label_tokens"])
    map_params = client.decode_params(cap)
    params = [param for _ in texts for param in (label_params, map_params)]
    started = time.perf_counter()
    outputs = client.generate(prompts, params, use_tqdm=False)
    wall = time.perf_counter() - started
    by_token = dict(zip(planned["label_tokens"], planned["labels"]))
    labels = [by_token.get(output.outputs[0].token_ids[0]) for output in outputs[0::2]]
    answers = [output.outputs[0].text.strip() for output in outputs[1::2]]
    input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    return {
        "engine": MULTI_SUBMISSION,
        "wall_s": round(wall, 2),
        "input_tokens": input_tokens,
        "input_tokens_per_second": round(input_tokens / wall, 1),
        "output_tokens": sum(len(output.outputs[0].token_ids) for output in outputs),
        "num_requests": len(prompts),
        "maps": {map_config[0]: answer_stats(answers, ids)},
        "labels": dict(zip(ids, labels)),
        "answers": dict(zip(ids, answers)),
    }


_ENGINES = {"quail": _quail, "vllm": _vllm}


def _child(engine: str, model: str, directory: str, only: str, out: str) -> None:
    """Run one engine in this process and write its result JSON to out."""
    result = _ENGINES[engine](model, Path(directory), _chosen(only))
    Path(out).write_text(json.dumps(result, default=str))


def _matching(first: dict, second: dict) -> dict:
    """Return the documents two engines both answered, and the share that match."""
    ids = [key for key in first if key in second]
    same = sum((first[key] or "").strip() == (second[key] or "").strip()
               for key in ids)
    return {"num_documents": len(ids), "matching": same / max(len(ids), 1)}


def _agreement(first: dict, second: dict) -> dict:
    """Return each map's exact answer agreement, and the labels', if any."""
    agreement = {name: _matching(first["maps"][name]["answers"],
                                 second["maps"][name]["answers"])
                 for name in first.get("maps", {}) if name in second.get("maps", {})}
    if "labels" in first and "labels" in second:
        agreement["labels"] = _matching(first["labels"], second["labels"])
    return agreement


def _run_child(engine: str, model: str, directory: Path, only: str) -> dict:
    """Run one engine in a fresh interpreter on this GPU; return its result."""
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / f"{engine}.json"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
        started = time.perf_counter()
        subprocess.run(
            # the image holds this file, not the experiments package
            [sys.executable, __file__, engine, model, str(directory), only,
             str(out)],
            check=True, env=environment)
        result = json.loads(out.read_text())
    result["process_s"] = round(time.perf_counter() - started, 2)
    return result


@app.function(image=image, gpu="H100!", memory=98304, timeout=14400,
              volumes=VOLUMES)
def map_same_node(model: str = "qwen3-4b-fp8", scale_factor: float = 0.1,
                  only: str = "", engines: str = "quail,vllm") -> str:
    """Run the engines one after the other on this GPU; return the result's path.

    Args:
        model: The model to run.
        scale_factor: The corpus scale factor.
        only: Comma-separated configuration names to run; empty runs
            all seven.
        engines: Comma-separated engines to run, from "quail" and
            "vllm"; they run in that order.

    Raises:
        RuntimeError: The engines ran on different GPUs.
    """
    chosen_engines = [engine for engine in ("quail", "vllm")
                      if engine in {e.strip() for e in engines.split(",")}]
    if not chosen_engines:
        raise ValueError(f"no known engine in {engines!r}")
    configurations = _chosen(only)
    directory = _ensure_tables(scale_factor,
                               [config[1] for config in configurations])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    measurements = {"model": model, "scale_factor": scale_factor,
                    "configurations": [config[0] for config in configurations],
                    "engines": {}}
    for engine in chosen_engines:
        print(f"starting {engine}", flush=True)
        measurements["engines"][engine] = _run_child(engine, model, directory, only)
    for name in measurements["configurations"]:
        runs = [result["runs"].get(name, {})
                for result in measurements["engines"].values()]
        if len(runs) == 2 and all("maps" in run for run in runs):
            measurements.setdefault("agreement", {})[name] = _agreement(*runs)
    gpus = {engine: result["gpu"]
            for engine, result in measurements["engines"].items()}
    uuids = {gpu.split(",")[0] for gpu in gpus.values()}
    measurements["gpus"] = gpus
    measurements["same_gpu"] = len(uuids) == 1
    out = Path(RESULT_DIR) / f"map_same_node_{model}_sf{scale_factor}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(measurements, indent=2, default=str))
    results_vol.commit()
    print(f"saved {out}")
    if len(uuids) != 1:
        raise RuntimeError(f"the engines ran on different GPUs: {gpus}")
    return str(out)


@app.local_entrypoint()
def launch_same_node(model: str = "qwen3-4b-fp8", scale_factor: float = 0.1,
                     only: str = "", engines: str = "quail,vllm"):
    call = map_same_node.spawn(model=model, scale_factor=scale_factor, only=only,
                               engines=engines)
    print(f"function call id: {call.object_id}")
    print(call.get())


if __name__ == "__main__":
    _child(*sys.argv[1:6])
