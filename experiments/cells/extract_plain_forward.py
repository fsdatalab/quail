"""Compare extraction traces with fresh, unpaged Qwen forwards on Modal.

Each case is document:question:answer_index. The input is the exact planned
head, stored document tokens, question tail/cue, and the trace's accepted
answer prefix. The pipeline runs directly: no paging, KV reuse, or decode
CUDA graphs. Both prefixes are checked when the traces already diverged.
The same final hidden row is projected in head batches of 1, 64, and 512
to isolate output-head GEMM shape effects. FP8 forward batching can still
change rounding; this is not a bitwise oracle.

    uv run modal run --detach experiments/cells/extract_plain_forward.py::check \
        --old-trace /tmp/trace-old-cuad10.jsonl \
        --new-trace /tmp/trace-new-cuad10.jsonl \
        --cases 0:0:0,0:0:1,0:1:9,3:2:0,27:1:0,6:2:0
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from experiments.cells.profile_query import (
    DATA_DIR,
    PROFILE_DIR,
    VOLUMES,
    app,
    ensure_data,
    results_vol,
)
from experiments.cells.profile_query import (
    image as profile_image,
)

image = profile_image.add_local_python_source("experiments")


class ProbeCompleteError(Exception):
    """End the query after inspecting its prepared model inputs."""


def trace_index(records):
    """Index one query's trace, rejecting duplicate or malformed step keys."""
    indexed = {}
    for record in records:
        key = tuple(int(record[name]) for name in ("document", "question", "index"))
        if key in indexed:
            raise ValueError(f"duplicate trace key {key}; use one query per trace file")
        indexed[key] = record
    return indexed


def answer_prefix(indexed, document, question, index):
    """Read all accepted tokens preceding a requested answer step."""
    prefix = []
    for at in range(index):
        if (document, question, at) not in indexed:
            raise ValueError(f"trace has no prefix step {document}:{question}:{at}")
        token = int(indexed[document, question, at]["chosen"])
        if token < 0:
            raise ValueError(f"{document}:{question}:{index} follows a stopped answer")
        prefix.append(token)
    if (document, question, index) not in indexed:
        raise ValueError(f"trace has no step {document}:{question}:{index}")
    return prefix


def plain_chunk(torch, tokens, device):
    """Build one full causal sequence without arena reads or writes."""
    from quail.backends.quail.executor.chunk import Chunk

    count = len(tokens)
    if not count:
        raise ValueError("a plain forward needs a nonempty prompt")
    return Chunk(
        input_ids=torch.tensor(tokens, dtype=torch.int64, device=device),
        positions=torch.arange(count, dtype=torch.int64, device=device),
        final_indices=torch.tensor([count - 1], dtype=torch.int64, device=device),
        meta=dict(
            layer=0,
            canvas=None,
            unified=None,
            shared=None,
            reads=None,
            kv_src=None,
            kv_dst=None,
            cu_a=torch.tensor([0, count], dtype=torch.int32, device=device),
            max_a=count,
        ),
        attention_mode="unified",
        tokens=count,
        layout=[],
    )


def compare_scores(logprobs, stop, reference):
    """Compare the reference trace's named token scores with the plain forward."""
    candidates = {}
    for name in ("chosen", "allowed", "runner_up", "top"):
        token = int(reference[name])
        if token >= 0:
            candidates[name] = {"token": token, "plain_lp": float(logprobs[token])}
    recorded = {"allowed": reference["allowed_lp"], "top": reference["top_lp"]}
    if reference["allowed_lp"] is not None and reference["margin"] is not None:
        recorded["runner_up"] = reference["allowed_lp"] - reference["margin"]
    for name, value in recorded.items():
        if name in candidates and value is not None:
            candidates[name].update(
                trace_lp=value, delta=candidates[name]["plain_lp"] - value
            )
    return {
        "tokens": candidates,
        "stop_plain_lp": float(stop),
        "stop_trace_lp": reference["stop_lp"],
        "stop_delta": (
            None if reference["stop_lp"] is None else float(stop) - reference["stop_lp"]
        ),
    }


def inspect_cases(graph, context, arguments):
    """Run the requested fresh forwards using the query's actual loaded inputs."""
    from quail.backends.quail.executor.extract import load_vocabulary
    from quail.backends.quail.executor.extract.stages import ExtractionInputs
    from quail.backends.quail.executor.model_loading import full_output_head
    from quail.physical import AiExtract, AiMap

    gpu = context.state["gpu"]
    torch, pipeline, arena = gpu.torch, gpu.pipeline, gpu.arena
    if arena.num_state_slots or arena.has_sliding or pipeline.canvas_ids:
        raise ValueError("this probe supports full-attention causal models only")
    nodes = list(graph.topological_nodes())
    if any(isinstance(node, AiMap) for node in nodes):
        raise ValueError("use an extraction-only answer pipeline, such as CUAD-10")
    specs = [node.spec for node in nodes if isinstance(node, AiExtract)]
    if not specs:
        raise ValueError("the query has no extraction fields")
    cases = arguments["cases"]
    document_ids = sorted({case[0] for case in cases})
    state = SimpleNamespace(loaded_model=gpu.execution.loaded_model)
    vocabulary = load_vocabulary(state)
    alias = specs[0].alias
    prepared = ExtractionInputs(
        state,
        specs,
        vocabulary,
        document_ids,
        context.state["documents"][alias],
        context.state["texts"][alias],
        pipeline_head=specs[0].prompt_token_parts[0],
    )
    documents = dict(zip(document_ids, prepared.documents))
    traces = {
        name: trace_index(records) for name, records in arguments["traces"].items()
    }
    head = full_output_head(gpu.model)
    torch.cuda.synchronize()
    if pipeline.decode_graphs is not None:
        pipeline.decode_graphs.invalidate()
    arena.resize(1, free_resident=True)
    torch.cuda.empty_cache()
    records = []
    with torch.inference_mode():
        for document, question, index in cases:
            spec = specs[question]
            prefixes = {}
            for name, trace in traces.items():
                tokens = tuple(answer_prefix(trace, document, question, index))
                prefixes.setdefault(tokens, []).append(name)
            for answer, names in prefixes.items():
                prefix = list(map(int, documents[document].prefix))
                tail = list(spec.prompt_token_parts[1])
                tokens = [*prefix, *tail, *answer]
                if len(tokens) > pipeline.max_chunk_tokens:
                    raise ValueError(
                        f"plain input has {len(tokens)} rows; exceeds model limit"
                    )
                chunk = plain_chunk(torch, tokens, head.device)
                normed = pipeline.forward_chunk(chunk)
                if tuple(normed.shape[:1]) != (1,):
                    raise AssertionError(
                        "plain forward did not return its one final row"
                    )
                for head_rows in arguments["head_batches"]:
                    repeated = normed.repeat(head_rows, 1)
                    logits = torch.nn.functional.linear(
                        repeated.to(head.dtype), head
                    ).float()
                    logprobs = (logits - torch.logsumexp(logits, dim=1, keepdim=True))[
                        0
                    ]
                    stop = torch.logsumexp(
                        logprobs[list(vocabulary.stop_ids)], dim=0
                    ).item()
                    scores, top = logprobs.topk(8)
                    host = logprobs.cpu().numpy()
                    record = {
                        "head_rows": head_rows,
                        "document": document,
                        "question": question,
                        "index": index,
                        "output": spec.name,
                        "prefix_from": names,
                        "prefixes_equal": len(prefixes) == 1,
                        "num_prompt_tokens": len(prefix) + len(tail),
                        "num_input_tokens": len(tokens),
                        "last_position": len(tokens) - 1,
                        "answer_prefix": list(answer),
                        "input_sha256": hashlib.sha256(
                            json.dumps(tokens, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "top": list(zip(top.cpu().tolist(), scores.cpu().tolist())),
                        "comparisons": {
                            name: compare_scores(
                                host, stop, traces[name][document, question, index]
                            )
                            for name in names
                        },
                    }
                    if arguments["include_tokens"]:
                        record["input_tokens"] = tokens
                    records.append(record)
                    print("RESULT " + json.dumps(record), flush=True)
    torch.cuda.synchronize()
    return records


def child(input_path, output_path):
    """Load one query normally, replacing its execution with plain forwards."""
    import quail
    import quail.backends.quail.graph as graph_module
    from bench.quailb import run_query
    from quail_b.benchmark import load_benchmark

    arguments = json.loads(Path(input_path).read_text())
    print(f"quail from {quail.__file__}", flush=True)
    suite = load_benchmark(
        [arguments["query"]],
        scale_factor=arguments["scale_factor"],
        data_dir=Path(DATA_DIR) / f"sf{arguments['scale_factor']}",
        collection_id=arguments["collection_id"] or None,
    )
    spec = suite.queries[0]
    tables = {
        relation.table: suite.tables[relation.table] for relation in spec.info.relations
    }
    original = graph_module.GenericRunner
    records = []

    class PlainRunner(original):
        def run(self, graph, context, *args, **kwargs):
            records.extend(inspect_cases(graph, context, arguments))
            raise ProbeCompleteError

    graph_module.GenericRunner = PlainRunner
    config = quail.EngineConfig(
        gpus=1, model=arguments["model"], backend="quail", device="h100-sxm"
    )
    try:
        with quail.Session(config) as session:
            try:
                run_query(session, spec, tables)
            except ProbeCompleteError:
                pass
    finally:
        graph_module.GenericRunner = original
    if not records:
        raise RuntimeError("the query did not reach the plain-forward probe")
    Path(output_path).write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600, volumes=VOLUMES)
def plain_forward(arguments: dict, tag: str) -> str:
    """Run the probe in a fresh process and save its JSONL on the results volume."""
    if not tag or Path(tag).name != tag:
        raise ValueError("tag must be one filename component")
    results_vol.reload()
    directory = Path(PROFILE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{tag}.jsonl"
    with TemporaryDirectory(prefix="extract-plain-") as temporary:
        source = Path(temporary) / "inputs.json"
        source.write_text(json.dumps(arguments))
        command = [
            sys.executable,
            "-c",
            "import sys; from experiments.cells.extract_plain_forward import child; "
            "child(sys.argv[1], sys.argv[2])",
            str(source),
            str(output),
        ]
        env = {**os.environ, "PYTHONPATH": "/root:" + os.environ.get("PYTHONPATH", "")}
        env.pop("QUAIL_EXTRACT_TRACE", None)
        result = subprocess.run(
            command, env=env, cwd="/tmp", capture_output=True, text=True
        )
        print(result.stdout[-60000:], flush=True)
        if result.returncode:
            print(result.stderr[-8000:], flush=True)
            raise RuntimeError("plain-forward probe failed")
    results_vol.commit()
    print(f"saved {output}", flush=True)
    return output.read_text()


@app.local_entrypoint()
def check(
    old_trace: str,
    new_trace: str,
    cases: str = "0:0:0,0:0:1,0:1:9,3:2:0,27:1:0,6:2:0",
    query: str = "CUAD-10",
    scale_factor: float = 0.1,
    model: str = "qwen3-4b-fp8",
    collection_id: str = "",
    tag: str = "extract-plain-cuad10",
    output: str = "/tmp/extract-plain-cuad10.jsonl",
    include_tokens: bool = False,
    head_batches: str = "1,64,512",
):
    selected = [tuple(map(int, case.split(":"))) for case in cases.split(",")]
    if any(len(case) != 3 or min(case) < 0 for case in selected):
        raise ValueError("cases must be document:question:index triples")
    batches = [int(value) for value in head_batches.split(",")]
    if not batches or any(not 1 <= value <= 512 for value in batches):
        raise ValueError("head batches must be between 1 and 512")
    traces = {}
    for name, path in (("old", old_trace), ("new", new_trace)):
        indexed = trace_index(
            [
                json.loads(line)
                for line in Path(path).read_text().splitlines()
                if line.strip()
            ]
        )
        needed = set()
        for document, question, index in selected:
            answer_prefix(indexed, document, question, index)
            needed.update((document, question, at) for at in range(index + 1))
        traces[name] = [indexed[key] for key in sorted(needed)]
    data = ensure_data.spawn(scale_factor, [query], collection_id)
    print(f"function call id: {data.object_id} (data)", flush=True)
    arguments = dict(
        query=query,
        scale_factor=scale_factor,
        collection_id=data.get(),
        model=model,
        cases=selected,
        traces=traces,
        include_tokens=include_tokens,
        head_batches=batches,
    )
    call = plain_forward.spawn(arguments, tag)
    print(f"function call id: {call.object_id} (plain forward)", flush=True)
    text = call.get()
    Path(output).write_text(text)
    print(text, end="")
    print(f"saved {output}", flush=True)
