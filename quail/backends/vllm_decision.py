"""Decision 2.0 checkpoints on vLLM's pooling runner.

The converted Qwen3 backbone runs as a vLLM pooling model under the
token classification task. vLLM's ALL pooling hands the pooler every
hidden row of a finished prompt; the pooler's head picks the rows at
the request's offsets (the end of each option block, then the prompt's
last row) and returns the decision head's fp32 option scores. The
backend reads an answer from the scores as the Quail readout does.

vLLM does not read the prefix cache for a token-level pooling request,
because a row served from the cache is never computed. A request
therefore computes its whole prompt.
"""

from __future__ import annotations

ARCHITECTURE = "Qwen3DecisionModel"
TASK = "token_classify"
OFFSETS_KEY = "decision_offsets"


def register() -> None:
    """Register the pooling model class with vLLM's model registry.

    The engine core process is forked after this call and inherits
    the registration.
    """
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCHITECTURE, "quail.backends.vllm_decision_model:Qwen3DecisionModel")


def pooling_params(offsets):
    """Build the parameters of one request whose head reads rows at offsets.

    Args:
        offsets: Distances before the last row: one per option, then 0.
    """
    from vllm import PoolingParams

    return PoolingParams(task=TASK,
                         extra_kwargs={OFFSETS_KEY: [int(o) for o in offsets]})


def score_requests(torch, head, rows, params) -> list:
    """Score each finished request's options from its hidden rows.

    Args:
        torch: The torch module.
        head: The model's DecisionHead.
        rows: Per request, every hidden row of its prompt, or None
            while it is still prefilling.
        params: Per request, its PoolingParams.

    Returns:
        Per request, float32 option scores, or None while it prefills.
        A request without offsets (vLLM's profiling run) gets one row's
        first value.
    """
    out = [None] * len(rows)
    groups = {}
    for index, (hidden, param) in enumerate(zip(rows, params)):
        if hidden is None:
            continue
        offsets = (param.extra_kwargs or {}).get(OFFSETS_KEY)
        if offsets is None:
            out[index] = hidden[-1, :1].float()
            continue
        last = hidden.shape[0] - 1
        groups.setdefault(tuple(offsets), []).append(
            (index, hidden[[last - offset for offset in offsets]]))
    for members in groups.values():
        picked = torch.stack([picked for _, picked in members])
        scores = head.scores(picked[:, :-1], picked[:, -1])
        for (index, _), row in zip(members, scores):
            out[index] = row
    return out


def decision_bit(output) -> int:
    """Read a yes/no answer: 1 when Yes scores above No."""
    scores = output.outputs.data
    return int(scores[1] > scores[0])


def choice_index(output) -> int:
    """Read the index of the highest-scoring option; ties go to the first."""
    return int(output.outputs.data.argmax())
