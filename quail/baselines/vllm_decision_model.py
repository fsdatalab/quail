"""The Decision 2.0 model class vLLM loads for the pooling runner.

vLLM imports this module by name after `vllm_decision.register()`.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.models.interfaces_base import VllmModelForPooling

from quail.backends.quail.executor.models.decision2_vllm import (
    Decision2ForCausalLM,
)
from quail.backends.quail.executor.readout import DecisionHead
from quail.baselines.vllm_decision import TASK, pool_decision_rows


class DecisionPooler(Pooler):
    """Option scores from the rows a decision head reads.

    See `pool_decision_rows`.

    Args:
        head: The model's DecisionHead.
    """

    def __init__(self, head: DecisionHead):
        super().__init__()
        self.head = head

    def get_supported_tasks(self):
        return {TASK}

    def forward(self, hidden_states, pooling_metadata):
        cursor = pooling_metadata.get_pooling_cursor()
        return pool_decision_rows(
            torch, self.head,
            torch.split(hidden_states, cursor.num_scheduled_tokens_cpu.tolist()),
            cursor.is_finished().tolist(), pooling_metadata.pooling_params,
            pooling_metadata.pooling_states)


class Qwen3DecisionModel(Decision2ForCausalLM, VllmModelForPooling):
    """The Qwen3 backbone with a decision head pooler.

    The output head is tied to the embeddings and holds no memory of
    its own; the pooler never calls it.
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # the head sits at the package root, beside backbone/
        self.pooler = DecisionPooler(DecisionHead.load(
            torch, F, Path(vllm_config.model_config.model).parent))
