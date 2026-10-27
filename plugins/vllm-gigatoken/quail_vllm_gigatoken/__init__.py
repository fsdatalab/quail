"""A vLLM plugin that registers Quail's gigatoken tokenizer mode.

vLLM loads general plugins in each process it starts, so the engine
core can load the tokenizer the request backend names. Quail itself is
mounted as source in the GPU image, not installed, so its own entry
points are not visible there; this package is installed instead.
"""


def register() -> None:
    """Register the gigatoken tokenizer mode in this vLLM process."""
    from quail.baselines.vllm import register_gigatoken

    register_gigatoken()
