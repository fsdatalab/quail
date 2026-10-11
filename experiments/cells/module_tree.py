"""List a loaded checkpoint's module tree, shapes, and layer forward code.

Loads the model through Quail's loader and records what a forward pass
written for it must name: the decoder layers and their children, every
weight's shape and dtype, the numeric attributes of each distinct
token mixer, and the source of the layer and mixer forward methods.

    uv run modal run experiments/cells/module_tree.py \
        --model qwen3.5-4b-bf16 2>&1 | tee /tmp/module_tree.log

The record is written to /results/ablations/<model>-module-tree.json on
the quail-results volume.
"""

import inspect
import json

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
SOURCE_LIMIT = 9000


def _attributes(module):
    return {name: value for name, value in vars(module).items()
            if isinstance(value, (int, float, bool, str))
            and not name.startswith("_")}


def _weights(module):
    return {name: [list(parameter.shape), str(parameter.dtype)]
            for name, parameter in module.named_parameters(recurse=False)}


def _tree(module, depth):
    record = {"class": type(module).__name__, "weights": _weights(module)}
    if depth:
        record["children"] = {name: _tree(child, depth - 1)
                              for name, child in module.named_children()}
    return record


def _methods(cls):
    """Return the source of every function the class body defines."""
    return {name: _source(function) for name, function in vars(cls).items()
            if inspect.isfunction(function)}


def _source(function):
    try:
        return inspect.getsource(function)[:SOURCE_LIMIT]
    except (OSError, TypeError) as error:
        return f"unavailable: {error}"


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def inspect_model(model: str) -> dict:
    """Load the model text-only and describe its modules."""
    import torch  # noqa: F401  vLLM needs it initialised before loading

    from quail.backends.quail.executor.model import load_model
    from quail.specs import MODELS

    spec = MODELS[model]
    loaded = load_model(spec.hf_name, spec.revision or None,
                        max_batched_tokens=4096, text_only=spec.vision_tower)
    layers = {name: module for name, module in loaded.named_modules()
              if type(module).__name__.endswith("DecoderLayer")}
    names = sorted(layers, key=lambda name: int(name.rsplit(".", 1)[1]))
    record = {"model": model, "class": type(loaded).__name__,
              "top": {name: type(child).__name__
                      for name, child in loaded.named_children()},
              "layer_names": names[:2] + names[-1:],
              "layer_classes": sorted({type(layers[n]).__name__
                                       for n in names}),
              "layer_count": len(names), "layers": {}, "mixers": {},
              "forward_source": {}}
    first_of_kind = {}
    for name in names:
        layer = layers[name]
        kind = {child: type(m).__name__
                for child, m in layer.named_children()}
        key = json.dumps(kind, sort_keys=True)
        first_of_kind.setdefault(key, name)
    for key, name in first_of_kind.items():
        layer = layers[name]
        record["layers"][name] = _tree(layer, 3)
        for child_name, child in layer.named_children():
            if "attn" in child_name or "mixer" in child_name:
                kind = type(child).__name__
                record["mixers"][kind] = _attributes(child)
                record["forward_source"][kind] = _source(child.forward)
        record["forward_source"][type(layer).__name__] = _source(layer.forward)
    record["method_source"] = {}
    for name in names[:1] + [n for n in names if n.endswith(".3")][:1]:
        layer = layers[name]
        for child_name, child in layer.named_modules():
            kind = type(child).__name__
            if kind in ("QwenGatedDeltaNetAttention", "Qwen3NextAttention",
                        "GemmaRMSNorm", "RMSNormGated", "MRotaryEmbedding"):
                record["method_source"][kind] = _methods(type(child))
    final = {name: _tree(child, 1) for name, child in loaded.named_modules()
             if name.endswith(".norm") and name.count(".") <= 2}
    record["final_norms"] = final
    path = f"/results/ablations/{model}-module-tree.json"
    with open(path, "w") as out:
        json.dump(record, out, indent=1)
    results.commit()
    record["path"] = path
    return record


@app.local_entrypoint()
def main(model: str = "qwen3.5-4b-bf16"):
    call = inspect_model.spawn(model)
    print(f"function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=1), flush=True)
