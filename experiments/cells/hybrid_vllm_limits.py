"""Find the sequence limit and loaded weight size of a hybrid model in stock vLLM.

vLLM holds one recurrent-state block per running sequence and refuses
a max_num_seqs it cannot back. A boot at 4,096 sequences can run out
of memory in the sampler's logits buffer before that check, so this
cell boots the registered model once per candidate setting, each in
its own container, and records whether it started and the capacity
vLLM reports. It also renders the checkpoint's chat template with
thinking off and compares it with the spec's turn text, and records
the weight memory vLLM reports after loading.

Prediction: 256 and 512 start; the largest setting that starts is below
the 800 state blocks the 128-sequence boot reported; the weights load
as 29.47e9 bytes within 1 percent.

    uv run modal run experiments/cells/hybrid_vllm_limits.py \
        --model qwen3.5-27b-fp8 --settings 256,512,1024,2048 \
        2>&1 | tee /tmp/hybrid_vllm_limits.log

Each record is written to
/results/ablations/<model>-vllm-limits-<setting>.json on the
quail-results volume.
"""

import json
import logging
import os

import modal

try:
    from quail.bench.images import gpu_image
    image = gpu_image()
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


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        message = record.getMessage()
        if "Model loading took" in message or "KV cache" in message:
            self.lines.append(message)


def _model_memory(client):
    """Return the bytes vLLM reports for the loaded weights, or None."""
    llm = client.llm
    try:
        values = llm.collective_rpc(
            lambda worker: worker.model_runner.model_memory_usage)
        return int(values[0])
    except Exception as error:    # the worker path differs between vLLM builds
        return f"unavailable: {type(error).__name__}: {error}"[:300]


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def probe(model: str, max_num_seqs: int) -> dict:
    """Boot the model at one sequence setting and return what vLLM reported."""
    from dataclasses import replace

    from transformers import AutoTokenizer

    from quail.backends.vllm import VLLMEngine
    from quail.specs import MODELS

    spec = MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_name, revision=spec.revision or None)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "X"}], add_generation_prompt=True,
        enable_thinking=False, tokenize=False)
    expected = spec.turn_prefix + "X" + spec.turn_suffix
    record = {"model": model, "rendered_turn": rendered,
              "spec_turn": expected, "turn_matches": rendered == expected,
              "requested_max_num_seqs": max_num_seqs}

    import vllm  # noqa: F401  its import resets the vllm loggers

    capture = _Capture()
    logging.getLogger("vllm").addHandler(capture)
    allowed = [tokenizer.encode(word, add_special_tokens=False)[0]
               for word in ("TRUE", "FALSE")]
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    try:
        state, boot = VLLMEngine().boot(
            replace(spec, max_num_seqs=max_num_seqs), allowed)
        record["boot"] = "started"
        record["capacity"] = {
            key: value for key, value in state["capacity"].items()
            if not key.startswith("initial_")}
        record["boot_s"] = boot["boot_s"]
        record["model_memory_bytes"] = _model_memory(state["client"])
    except Exception as error:    # vLLM names its limit in the message
        record["boot"] = f"{type(error).__name__}: {error}"
    record["vllm_log"] = capture.lines
    path = f"/results/ablations/{model}-vllm-limits-{max_num_seqs}.json"
    with open(path, "w") as out:
        json.dump(record, out, indent=2)
    results.commit()
    record["path"] = path
    return record


@app.local_entrypoint()
def main(model: str = "qwen3.5-27b-fp8", settings: str = "256,512,1024,2048"):
    values = [int(item) for item in settings.split(",")]
    calls = [probe.spawn(model, value) for value in values]
    for value, call in zip(values, calls):
        print(f"function call id: {call.object_id} (max_num_seqs={value})",
              flush=True)
    for value, call in zip(values, calls):
        record = call.get()
        print(json.dumps(record, indent=2), flush=True)
