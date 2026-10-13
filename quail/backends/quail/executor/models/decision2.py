"""Loading a Decision 2.0 package's Qwen3 backbone through vLLM.

A Decision 2.0 package keeps its backbone in ``backbone/``, saved as a
bare Qwen3Model without the ``model.`` tensor prefix. vLLM loads it
through ``Decision2ForCausalLM`` in ``decision2_vllm``, which maps the
names; this module registers that class without importing vLLM at
import time.
"""

from __future__ import annotations

BACKBONE_ARCHITECTURE = "Decision2ForCausalLM"
BACKBONE_MODULE = "quail.backends.quail.executor.models.decision2_vllm"


def register_backbone() -> None:
    """Register the backbone class with vLLM's model registry."""
    from vllm import ModelRegistry

    if BACKBONE_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            BACKBONE_ARCHITECTURE, f"{BACKBONE_MODULE}:{BACKBONE_ARCHITECTURE}")
