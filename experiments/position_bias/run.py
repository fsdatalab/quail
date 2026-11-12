"""Run the position-bias experiment for each model and task on Modal.

Each (model, task) runs in its own H100 container: pointwise AI.SCORE
top-k, the swap test, and quickselect under each decision rule. A
pair is one document holding RECORD 1 and RECORD 2, scored with
AI.SCORE, so a score is P(RECORD 1 wins).

    uv run modal run --detach -m experiments.position_bias.run \
        2>&1 | tee results/position_bias/run.log
    uv run modal run -m experiments.position_bias.run \
        --models qwen3-4b-fp8 --tasks legal --smoke

Records: /results/position_bias/{model}/{task}.json on the
quail-results volume. The entrypoint prints each function call id.
"""

import json
from pathlib import Path

import modal

try:
    from quail.deploy.images import gpu_image
    image = gpu_image().add_local_python_source("experiments")
except ImportError:    # a container without the local quail package
    image = None

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)
results = modal.Volume.from_name("quail-results", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache,
           "/root/.cache/kernels": kernel_cache,
           "/results": results}

MODELS = ("decision-2.0-kai-0.6b-bf16", "qwen3-0.6b-bf16", "d1-2b-bf16",
          "lfm2-2.6b-bf16", "qwen3-4b-fp8", "qwen3.8-27b-fp8",
          "qwen3.6-35b-a3b-fp8")
TASK_NAMES = ("sales", "support", "legal")
# rows per AI.SCORE query; bounds the host memory of one batch
BATCH_ROWS = 4096


class QuailScorer:
    """Score documents with AI.SCORE in one Quail session."""

    def __init__(self, session):
        self.session = session
        self.num_batches = 0

    def __call__(self, template: str, documents) -> list[float]:
        """Return AI.SCORE's P(TRUE) for each document under the template."""
        import pyarrow as pa

        import quail

        scores = []
        quoted = template.replace("'", "''")
        for start in range(0, len(documents), BATCH_ROWS):
            chunk = list(documents[start:start + BATCH_ROWS])
            name = f"batch_{self.num_batches}"
            self.num_batches += 1
            table = pa.table({"id": pa.array(range(len(chunk)), pa.int64()),
                              "document": pa.array(chunk, pa.string())})
            self.session.register(
                name, quail.DocumentProvider.from_table(table, id_col="id"))
            result = self.session.sql(
                f"SELECT b.id, AI.SCORE(PROMPT('{quoted}', b.document)) "
                f"AS score FROM {name} b").collect()
            # columns in SELECT order: the row id, then the score
            by_id = dict(zip(result.column(0).to_pylist(),
                             result.column(1).to_pylist()))
            scores.extend(float(by_id[index]) for index in range(len(chunk)))
        return scores


@app.function(image=image, gpu="H100!", memory=98304, timeout=8 * 3600,
              volumes=VOLUMES)
def run_model_task(model: str, task_name: str, smoke: bool = False) -> dict:
    """Run one model on one task and save the record to the volume."""
    import time
    from contextlib import ExitStack

    import quail
    from experiments.position_bias.experiment import run_task
    from experiments.position_bias.tasks import TASKS, TaskData, load_task

    data = load_task(TASKS[task_name])
    if smoke:
        ids = data.ids[:40] + [key for key in data.ids if data.labels[key]][:6]
        ids = list(dict.fromkeys(ids))
        data = TaskData(data.task, {key: data.texts[key] for key in ids},
                        {key: data.labels[key] for key in ids})
    start = time.perf_counter()
    with ExitStack() as stack:
        session = stack.enter_context(quail.Session(
            quail.EngineConfig(model=model, device="h100-sxm")))
        scorer = QuailScorer(session)
        record = run_task(data, scorer, seeds=(0,) if smoke else (0, 1, 2),
                          num_swap_pairs=40 if smoke else 2000,
                          log=lambda line: print(f"[{model}] {line}",
                                                 flush=True))
    record.update(model=model, smoke=smoke,
                  wall_seconds=time.perf_counter() - start)
    if not smoke:
        path = Path(f"/results/position_bias/{model}/{task_name}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))
        results.commit()
    return record


@app.local_entrypoint()
def main(models: str = ",".join(MODELS), tasks: str = ",".join(TASK_NAMES),
         smoke: bool = False):
    """Start every (model, task), print each function call id, then wait."""
    calls = [(model, task, run_model_task.spawn(model, task, smoke))
             for model in models.split(",") for task in tasks.split(",")]
    for model, task, call in calls:
        print(f"{model} {task}: {call.object_id}", flush=True)
    for model, task, call in calls:
        try:
            record = call.get()
        except Exception as error:     # report every run, then go on
            print(f"{model} {task}: failed: {error!r}", flush=True)
            continue
        best = {entry["rule"]: entry["precision"]
                for entry in record["quickselect"]}
        print(f"{model} {task}: pointwise {record['pointwise']['precision']:.3f}, "
              f"quickselect {best}, bias "
              f"{record['swap']['summary']['all']['bias']:+.3f}", flush=True)
