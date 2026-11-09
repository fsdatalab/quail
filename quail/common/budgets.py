"""Memory and time budgets of a model on a device, from Rust's cost model.

The planner prices plans with the same functions in
``crates/quail-planner/src/cost/budgets.rs``; this module reads them for the
GPU executor, so the two never disagree.
"""

import json
from dataclasses import asdict
from functools import lru_cache

import _quail_native

from quail.specs import DeviceSpec, ModelSpec

POOL_FRACTION = _quail_native.POOL_FRACTION
PAGE_TOKENS = _quail_native.PAGE_TOKENS
ACT_RESERVE_CHUNKS = _quail_native.ACT_RESERVE_CHUNKS
READOUT_BLOCK_ROWS = _quail_native.READOUT_BLOCK_ROWS
CHECKPOINT_PASS_QUESTIONS = _quail_native.CHECKPOINT_PASS_QUESTIONS
GRAPH_TOKEN_BUCKETS = tuple(_quail_native.GRAPH_TOKEN_BUCKETS)

KV_POOL_TOKENS_ENV = _quail_native.KV_POOL_TOKENS_ENV
parallel_copies = _quail_native.parallel_copies
kv_pool_cap_tokens = _quail_native.kv_pool_cap_tokens


def _spec_json(spec) -> str:
    return json.dumps(asdict(spec))


@lru_cache(maxsize=64)
def _budgets(model_json: str, device_json: str | None):
    return _quail_native.Budgets(model_json, device_json)


def budgets_of(model: ModelSpec, device: DeviceSpec | None = None):
    """Return the Rust ``Budgets`` of a model, on a device when given."""
    return _budgets(_spec_json(model),
                    None if device is None else _spec_json(device))


def pages(tokens: int, page_tokens: int = PAGE_TOKENS) -> int:
    """Return the pages that hold `tokens` tokens."""
    return _quail_native.pages(tokens, page_tokens)


def execution_pages(chunk_tokens: int, page_tokens: int = PAGE_TOKENS) -> int:
    """Return the KV pages kept free for two chunks of execution."""
    return _quail_native.execution_pages(chunk_tokens, page_tokens)


def chunk_budget(model: ModelSpec, device: DeviceSpec) -> int:
    """Return the tokens per forward pass."""
    return budgets_of(model, device).chunk_budget()


def kernel_index_cap(model: ModelSpec) -> int:
    """Return the most tokens one kernel call indexes."""
    return budgets_of(model).kernel_index_cap()


def arena_pages(model: ModelSpec, device: DeviceSpec,
                chunk_tokens: int | None = None,
                mean_document_tokens: float | None = None) -> tuple[int, int]:
    """Return the pages of the two KV pools: (every-token, sliding-layer).

    Raises:
        ValueError: The pool cannot hold a chunk's KV.
    """
    return budgets_of(model, device).arena_pages(chunk_tokens, mean_document_tokens)


def arena_tokens(model: ModelSpec, device: DeviceSpec,
                 chunk_tokens: int | None = None,
                 mean_document_tokens: float | None = None) -> int:
    """Return the tokens of document KV that can be resident at once."""
    return budgets_of(model, device).arena_tokens(chunk_tokens, mean_document_tokens)


def arena_bytes(model: ModelSpec, device: DeviceSpec, chunk_tokens: int) -> float:
    """Return the bytes left for KV after weights, activations, and graphs."""
    return budgets_of(model, device).arena_bytes(chunk_tokens)


def hybrid_arena_pages(model: ModelSpec, device: DeviceSpec, chunk_tokens: int,
                       mean_document_tokens: float | None) -> int:
    """Return a hybrid model's arena KV pages."""
    return budgets_of(model, device).hybrid_arena_pages(
        chunk_tokens, mean_document_tokens)


def hybrid_pool_pages(model: ModelSpec, device: DeviceSpec, chunk_tokens: int) -> int:
    """Return the pages a hybrid model shares between KV and checkpoints."""
    return budgets_of(model, device).hybrid_pool_pages(chunk_tokens)


def checkpoint_slots(model: ModelSpec, device: DeviceSpec, chunk_tokens: int,
                     kv_pages: int) -> int:
    """Return the checkpoints the pool holds beside an arena of `kv_pages`."""
    return budgets_of(model, device).checkpoint_slots(chunk_tokens, kv_pages)


def checkpoint_pages(model: ModelSpec, page_tokens: int = PAGE_TOKENS) -> float:
    """Return the KV pages of memory one document checkpoint takes."""
    return budgets_of(model).checkpoint_pages(page_tokens)


def checkpoint_pass_bytes(model: ModelSpec, questions: int, chunk_tokens: int) -> int:
    """Return the temporary bytes a pass of `questions` starts from checkpoints."""
    return budgets_of(model).checkpoint_pass_bytes(questions, chunk_tokens)


def checkpoint_memory_bytes(model: ModelSpec, chunk_tokens: int,
                            page_tokens: int = PAGE_TOKENS) -> float:
    """Return the workspace, execution KV, and one retained document's state."""
    return budgets_of(model).checkpoint_memory_bytes(chunk_tokens, page_tokens)


def chunk_memory_bytes(model: ModelSpec, chunk_tokens: int,
                       page_tokens: int = PAGE_TOKENS) -> float:
    """Return the resident weights, activations, and checkpoint reservations."""
    return budgets_of(model).chunk_memory_bytes(chunk_tokens, page_tokens)


def chunk_memory_bound(model: ModelSpec, device: DeviceSpec) -> int:
    """Return the largest chunk that fits its memory reservations, or zero."""
    return budgets_of(model, device).chunk_memory_bound()


def copy_verification_workspace_bytes(model: ModelSpec, chunk_tokens: int) -> int:
    """Return the readout workspace for copy verification."""
    return budgets_of(model).copy_verification_workspace_bytes(chunk_tokens)


def minimum_weight_gpus(model: ModelSpec, device: DeviceSpec) -> int:
    """Return the GPUs whose pooled memory holds the loaded weights."""
    return budgets_of(model, device).minimum_weight_gpus()


def sliding_kv_fits(model: ModelSpec, device: DeviceSpec,
                    chunk_tokens: int | None = None) -> bool:
    """Return whether the leftover arena holds one chunk's sliding KV."""
    return budgets_of(model, device).sliding_kv_fits(chunk_tokens)


def graph_token_buckets(model: ModelSpec, device: DeviceSpec) -> tuple[int, ...]:
    """Return the pass sizes captured as CUDA graphs; empty without graphs."""
    return tuple(budgets_of(model).graph_token_buckets())


def graph_memory_bytes(model: ModelSpec, device: DeviceSpec) -> int:
    """Return the memory the captured CUDA graphs hold."""
    return budgets_of(model).graph_memory_bytes()


def compute_knee(model: ModelSpec, device: DeviceSpec) -> float:
    """Return the tokens per pass where the projections become compute bound.

    Raises:
        ValueError: The projections never cross the roofline ridge.
    """
    return budgets_of(model, device).compute_knee()


def tree_attention_allowed(model: ModelSpec) -> bool:
    """Return whether the model may use the tree attention path."""
    return budgets_of(model).tree_attention_allowed()


def cheaper_attention_path(model: ModelSpec, device: DeviceSpec, *,
                           readers: float, reader_rows: float,
                           node_tokens: float) -> str:
    """Return "unified" or "tree" attention for readers of one shared node."""
    return budgets_of(model, device).cheaper_attention_path(
        readers, reader_rows, node_tokens)
