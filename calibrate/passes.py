"""Measure wall seconds of filter scans on a grid of sizes, for quail.cost.predict."""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pyarrow as pa

from quail.cost.budgets import chunk_budget
from quail.cost.sol import speed_of_light
from quail.cost.work import scan

# tokens of the filter instruction after each document
FILTER_TAIL_TOKENS = 51
WORDS_PER_DOCUMENT = (20, 60, 150, 300, 600, 1200, 2400)
NUM_DOCUMENTS = (1, 4, 16, 64, 256, 1024, 4096)
MAX_WORDS = 3_000_000
QUERY = ("SELECT d.id FROM documents d "
         "WHERE AI_FILTER(PROMPT('Is this text about a place? {0}', d.text))")


def grid() -> list[tuple[int, int]]:
    """Return every (words per document, num documents) point to measure.

    Points over MAX_WORDS words are left out to bound the GPU time.
    """
    return [(words, num) for words, num
            in itertools.product(WORDS_PER_DOCUMENT, NUM_DOCUMENTS)
            if words * num <= MAX_WORDS]


def synthetic_table(words_per_document: int, num_documents: int) -> pa.Table:
    """Return documents of a fixed length, each different from the others."""
    padding = " ".join(["filler"] * words_per_document)
    return pa.table({
        "id": [f"d{i}" for i in range(num_documents)],
        "text": [f"document {i} {padding}" for i in range(num_documents)],
    })


def record_from_metrics(model, device, words_per_document: int,
                        num_documents: int,
                        node_metrics: Mapping[str, Mapping]) -> dict:
    """Return one measurement from the filter nodes of a finished run.

    Args:
        model: The ModelSpec the run used.
        device: The DeviceSpec the run used.
        words_per_document: Words in each synthetic document.
        num_documents: Documents the filter read.
        node_metrics: Node id -> scalar NodeMetrics fields.

    Raises:
        ValueError: No node ran a forward pass.
    """
    nodes = [m for m in node_metrics.values() if m.get("chunks", 0) > 0]
    if not nodes:
        raise ValueError("no node ran a forward pass")
    fresh = sum(m["fresh_tokens"] for m in nodes)
    chunks = sum(m["chunks"] for m in nodes)
    request = fresh / num_documents
    work = scan(request - FILTER_TAIL_TOKENS, FILTER_TAIL_TOKENS) * num_documents
    chunk_tokens = chunk_budget(model, device)
    sol = speed_of_light(work, model, device, chunk_tokens).seconds
    return {
        "model": model.name, "device": device.name,
        "words_per_document": words_per_document,
        "num_documents": num_documents, "fresh_tokens": fresh, "chunks": chunks,
        "wall_seconds": sum(m["wall_s"] for m in nodes),
        "gpu_seconds": sum(m["gpu_s"] for m in nodes),
        "sol_seconds": sol, "chunk_tokens": chunk_tokens,
        "tokens": fresh / chunks, "pairs": work.pairs / chunks,
    }


def measure(make_session: Callable, words_per_document: int,
            num_documents: int) -> dict:
    """Run one filter scan on synthetic documents and return its record.

    Args:
        make_session: Returns a Session whose EngineConfig has
            gpu_timing=True. The session is closed after the run.
        words_per_document: Words in each synthetic document.
        num_documents: Documents the filter reads.
    """
    import quail

    with make_session() as session:
        session.register("documents", quail.DocumentProvider.from_table(
            synthetic_table(words_per_document, num_documents), id_col="id"))
        result = session.sql(QUERY).run()
        metrics = {node: vars(m) if not isinstance(m, Mapping) else m
                   for node, m in result.node_metrics.items()}
        return record_from_metrics(session.model, session.device,
                                   words_per_document, num_documents, metrics)


def save(path: Path, records: list[dict]) -> None:
    """Write measurements as JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
