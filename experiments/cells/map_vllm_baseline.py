"""Run stock vLLM on the map quadrant configurations, for comparison.

Same corpora, prompt text, and answer caps as
experiments/cells/map_quadrants.py, submitted operator-at-a-time: every
document's prompt goes to vLLM at once and its continuous batching,
prefix caching, and CUDA graphs handle the rest. Decoding is greedy
with the configuration's token cap, like Quail's map.

    run_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-map-vllm.log"
    uv run modal run --detach \
      experiments/cells/map_vllm_baseline.py::launch_baseline \
      2>&1 | tee "$run_log"

The result JSON lands under /results/ablations/ on the quail-results
volume; the function prints its path.
"""

import json
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from bench.quailb_parallel import VOLUMES, app, image, results_vol
from experiments.cells.map_quadrants import (
    CONFIGURATIONS,
    RESULT_DIR,
    _ensure_tables,
)


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


def _run_one(client, prompts, cap) -> dict:
    """Submit every prompt at once and measure the generation."""
    params = client.decode_params(cap)
    started = time.perf_counter()
    outputs = client.generate(prompts, params, use_tqdm=False)
    wall = time.perf_counter() - started
    input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    answers = [output.outputs[0].text for output in outputs]
    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    lengths = sorted(len(answer.strip()) for answer in answers)
    return {
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


@app.function(image=image, gpu="H100!", memory=98304, timeout=14400,
              volumes=VOLUMES)
def map_vllm_baseline(model: str = "qwen3-4b-fp8", scale_factor: float = 0.1,
                      only: str = "") -> str:
    """Run the map configurations on stock vLLM; return the result's path.

    Args:
        model: The model to run.
        scale_factor: The corpus scale factor.
        only: Comma-separated configuration names to run; empty runs
            all four.
    """
    import os

    from quail.baselines.vllm import VLLMEngine
    from quail.specs import H100_USD_PER_HOUR, MODELS

    # the engine core runs in process, as the other cells that boot
    # vLLM directly set it: a spawned core would not see the gigatoken
    # tokenizer registration
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    spec = MODELS[model]
    directory = _ensure_tables(scale_factor,
                               [table for _, table, _, _, _ in CONFIGURATIONS])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    chosen = {name.strip() for name in only.split(",") if name.strip()}
    measurements = {
        "engine": "stock vLLM, operator-at-a-time submission",
        "model": model, "scale_factor": scale_factor, "runs": {},
    }
    with ExitStack() as resources:
        engine = VLLMEngine()
        settings, boot = engine.boot(spec, [0], resources=resources)
        measurements["boot"] = boot
        measurements["engine_settings"] = {
            key: value for key, value in settings["capacity"].items()
            if isinstance(value, (int, float, str, bool))}
        client = settings["client"]
        for name, table, column, instruction, cap in CONFIGURATIONS:
            if chosen and name not in chosen:
                continue
            print(f"running {name}", flush=True)
            prompts = _prompts(directory, table, column, instruction, spec)
            run = _run_one(client, prompts, cap)
            run["usd_per_query"] = run["wall_s"] / 3600 * H100_USD_PER_HOUR
            measurements["runs"][name] = run
            print(json.dumps({name: run}, default=str), flush=True)
    out = Path(RESULT_DIR) / f"map_vllm_{model}_sf{scale_factor}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(measurements, indent=2, default=str))
    results_vol.commit()
    print(f"saved {out}")
    return str(out)


@app.local_entrypoint()
def launch_baseline(model: str = "qwen3-4b-fp8", sf: float = 0.1,
                    only: str = ""):
    call = map_vllm_baseline.spawn(model=model, scale_factor=sf, only=only)
    print(f"function call id: {call.object_id}")
    print(call.get())
