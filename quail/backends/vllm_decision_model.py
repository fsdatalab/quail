"""The Decision 2.0 model class vLLM loads for the pooling runner.

vLLM imports this module by name after `vllm_decision.register()`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler.tokwise import (
    AllPool,
    TokenPooler,
    TokenPoolerHead,
)
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

from quail.backends.quail.executor.readout import DecisionHead
from quail.backends.vllm_decision import TASK, score_requests


class DecisionPoolerHead(TokenPoolerHead):
    """Option scores from the rows a decision head reads.

    See `score_requests`.

    Args:
        head: The model's DecisionHead.
    """

    def __init__(self, head: DecisionHead):
        super().__init__()
        self.head = head

    def get_supported_tasks(self):
        return {TASK}

    def forward_chunk(self, pooled_data, pooling_param):
        (scores,) = score_requests(torch, self.head, [pooled_data], [pooling_param])
        return scores

    def forward(self, pooled_data, pooling_metadata):
        return score_requests(torch, self.head, pooled_data,
                              pooling_metadata.pooling_params)


class Qwen3DecisionModel(Qwen3ForCausalLM, VllmModelForPooling):
    """The Qwen3 backbone with a decision head pooler.

    The output head is tied to the embeddings and holds no memory of
    its own; the pooler never calls it.
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        head = DecisionHead.load(torch, F, vllm_config.model_config.model)
        self.pooler = TokenPooler(pooling=AllPool(), head=DecisionPoolerHead(head))
