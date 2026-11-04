"""Last-token SAE features of Terminal Wrench traces under three endings.

Each trace (the task, then every agent step's message and commands,
without output) is followed by one of three endings, and AI.SAE_FEATURES
reads the last token's top 100 features. The endings share the trace as
a prefix, so the runs differ only in the text after it:

1. "How did the agent get the task to pass?"
2. The same question, then "Answer:".
3. "The agent made the checker pass by".

Scoring happens on the CPU afterwards, from probes that predict the
dataset's exploit categories and the source model.

Prediction: endings 1 and 2 score within 0.02 macro AUROC of each
other, ending 3 scores at least 0.03 higher, and every ending predicts
the source model with accuracy above 0.85.

    uv run modal volume put quail-results traces.parquet \
        /ablations/sae-endings/traces.parquet
    uv run modal run --detach -m experiments.cells.sae_endings \
        2>&1 | tee /tmp/sae_endings.log

Records: /results/ablations/sae-endings/features-<ending>.parquet and
/results/ablations/sae-endings/runs.json.
"""

import json

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

MODEL = "qwen3.5-2b-sae-bf16"
DIRECTORY = "/results/ablations/sae-endings"
# SQL string literals: a backslash-n is a newline in the prompt
ENDINGS = {
    "question": r"{0}\n\nHow did the agent get the task to pass?",
    "answer": r"{0}\n\nHow did the agent get the task to pass?\nAnswer:",
    "completion": r"{0}\n\nThe agent made the checker pass by",
}


@app.function(image=image, gpu="H100!", memory=98304, timeout=7200,
              volumes=VOLUMES)
def run() -> dict:
    """Each ending's last-token features, saved per ending."""
    import time

    import pyarrow.parquet as pq

    import quail

    traces = pq.read_table(f"{DIRECTORY}/traces.parquet").select(
        ["id", "body"])
    config = quail.EngineConfig(model=MODEL, device="h100-sxm")
    record = {"model": MODEL, "num_traces": traces.num_rows, "endings": {}}
    with quail.Session(config) as session:
        session.register("traces", quail.DocumentProvider.from_table(
            traces, id_col="id"))
        for name, template in ENDINGS.items():
            sql = (f"SELECT t.id, f.feature, f.strength FROM traces t, "
                   f"UNNEST(AI.SAE_FEATURES(PROMPT('{template}', t.body), "
                   f"pooling => 'last')) AS f")
            start = time.perf_counter()
            table = session.sql(sql).collect()
            seconds = time.perf_counter() - start
            path = f"{DIRECTORY}/features-{name}.parquet"
            pq.write_table(table, path)
            record["endings"][name] = {"template": template, "path": path,
                                       "num_rows": table.num_rows,
                                       "seconds": seconds}
            print(name, json.dumps(record["endings"][name]), flush=True)
            results.commit()
    with open(f"{DIRECTORY}/runs.json", "w") as out:
        json.dump(record, out, indent=2)
    results.commit()
    return record


@app.local_entrypoint()
def main():
    call = run.spawn()
    print(f"function call id: {call.object_id}", flush=True)
    print(f"results: {DIRECTORY}", flush=True)
    print(json.dumps(call.get(), indent=2))
