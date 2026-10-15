"""The vLLM model class that loads a Decision 2.0 backbone.

vLLM imports this module by name after ``decision2.register_backbone()``.
"""

from __future__ import annotations

from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper


class Decision2ForCausalLM(Qwen3ForCausalLM):
    """Qwen3 that loads a Decision 2.0 backbone's tensor names.

    The backbone is saved as a bare Qwen3Model, without the `model.`
    prefix Qwen3ForCausalLM's parameters carry.
    """

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"": "model."})

    def load_weights(self, weights):
        loader = AutoWeightsLoader(
            self, skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings
                                 else None))
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
