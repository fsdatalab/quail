"""Check AI.MAP through the existing attention paths on Modal."""

from dataclasses import replace

import pyarrow as pa
import pytest

import quail
from quail.physical import AiMap

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")


def test_maps_agree_with_and_without_document_drafts():
    documents = pa.table({
        "id": [0, 1],
        "text": [
            "Alice borrowed 3 books and returned 1. "
            "Summary: Alice still has two books from the library.",
            "Bob borrowed 5 books and returned 2. "
            "Summary: Bob still has three books from the library.",
        ],
    })
    expected = {
        "d.id": [0, 1], "remaining": ["2", "3"],
        "summary": ["Alice still has two books from the library.",
                    "Bob still has three books from the library."],
    }
    for attention in ("tree", "unified"):
        with quail.Session(quail.EngineConfig(
                model="qwen3-4b-fp8", device="h100-sxm",
                attention=attention)) as session:
            session.register("documents", quail.DocumentProvider.from_table(
                documents, id_col="id"))
            for draft_tokens in (0, 15):
                query = (session.docs("documents").alias("d")
                         .ai_map("d.text", "How many books remain borrowed? "
                                 "Reply with the digit only.",
                                 name="remaining", max_answer_tokens=32)
                         .ai_map("d.text", "Copy the sentence after Summary: "
                                 "exactly. Return only that sentence.",
                                 name="summary", max_answer_tokens=32)
                         .select("d.id", "remaining", "summary"))
                plan = query.plan()
                plan = replace(plan, nodes=tuple(
                    replace(node, spec=replace(node.spec, draft_tokens=draft_tokens))
                    if isinstance(node, AiMap) else node for node in plan.nodes))
                result = query.run(plan=plan)
                actual = result.collect().sort_by("d.id").to_pydict()
                print(attention, draft_tokens, actual, result.report, flush=True)
                assert actual == expected
