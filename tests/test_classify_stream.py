"""A classification's stages inside the chain and join of its table."""

from types import SimpleNamespace

import numpy as np
from fakes import (
    DOC,
    FRAME,
    PARTNER,
    QUESTION,
    SETTINGS,
    FakeModel,
    cpu_arena,
    fake_pack,
    fake_pipeline,
    fake_torch,
)

from quail.backends.quail import QuailModelExecution
from quail.backends.quail.executor import loop
from quail.backends.quail.graph import execute_single_graph
from quail.builtins import built_in_registry
from quail.physical import (
    AiClassify,
    AiFilter,
    AiJoin,
    ClassifySpec,
    FilterStage,
    JoinStage,
    LabelFilter,
    PhysicalGraph,
    PortRef,
    Scan,
)
from quail.physical.base import input_ports
from quail.specs import DEVICES, MODELS

# the classification frame is written after the document like a join
# frame; the cue is the tail's last token
CLASSIFY_FRAME = 4500
CUE = 5000
LABEL_IDS = ((11,), (12,))


class LabelingModel(FakeModel):
    """Answer filter and join suffixes from truth, and label documents."""

    def __init__(self, filter_truth, join_truth, label_truth):
        super().__init__(filter_truth, join_truth)
        self.label_truth = label_truth

    def forward_chunk(self, chunk):
        rows = []
        for spec in chunk.specs:
            for suffix in spec["suffixes"]:
                head = suffix[0]
                if head == CUE:
                    # the cue row's log probabilities over the targets
                    wanted = self.label_truth[spec["key"][1]]
                    rows.append(np.asarray(
                        [-1.0 if index == wanted else -5.0
                         for index in range(len(LABEL_IDS))], np.float32))
                elif head >= FRAME:
                    rows.append(0)
                elif head >= PARTNER:
                    rows.append(self.join_truth[spec["key"]][head - PARTNER])
                else:
                    document = spec["key"][1]
                    rows.append(self.filter_truth[document][head - QUESTION])
            heads = [suffix[0] for suffix in spec["suffixes"]]
            if spec["prefix"] is not None and heads and heads[0] != QUESTION:
                raise AssertionError(
                    "a document packed its prefix past the filter stage")
        self.launched.append(chunk.specs)
        return rows


def fused_graph():
    """The r chain, its classification, a label filter, and a join with p."""
    spec = ClassifySpec(
        name="topic", aliases=("r",), query_template="", arguments=(),
        expected_inputs=14, estimated_seconds=0.0,
        prompt_token_parts=((), (CLASSIFY_FRAME, CUE)), labels=("a", "b"),
        label_token_ids=LABEL_IDS, scoring="trie_paths")
    nodes = [
        Scan(node_id="input:r", alias="r", input_id="r"),
        Scan(node_id="input:p", alias="p", input_id="p"),
        AiFilter(
            node_id="filter:r",
            inputs=input_ports((PortRef("input:r", "ids:r"),)),
            alias="r", arena_writes=True, pin_survivors=True,
            stages=(FilterStage(0, 1, 0, 0.8, 14 * 0.8),),
            question_token_ids=((QUESTION,),)),
        AiClassify(
            node_id="classify:r",
            inputs=input_ports((PortRef("filter:r", "ids:r"),)),
            backend_name="quail", model="qwen3-4b-fp8", spec=spec,
            pin_survivors=True),
        LabelFilter(
            node_id="label:r",
            inputs=input_ports((PortRef("classify:r", "scores"),)),
            score_name="topic", aliases=("r",), comparison="in",
            threshold=0.0, selectivity=0.5, written_pos=1, accepted=("a",)),
        AiJoin(
            node_id="group:0", anchor="r", anchor_resident="filter",
            inputs=input_ports((PortRef("label:r", "ids:r"),
                                PortRef("input:p", "ids:p"))),
            stages=(JoinStage(
                written_pos=2, exec_idx=0, anchor="r", partners=("p",),
                semantics="full", selectivity=0.5, expected_tuples=1,
                anchor_frame_tokens=1, pair_tail_tokens=0,
                anchor_resident="filter", tuple_tokens=0, pairs_from="",
                frame_token_ids=(FRAME,), label_token_ids=(("p", ()),),
                tail_token_ids=()),)),
    ]
    return PhysicalGraph(tuple(nodes), PortRef("group:0", "ids:r"))


def test_streamed_classification_labels_survivors_with_their_kv_resident(
        monkeypatch):
    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    n_docs, n_partners = 14, 4
    docs = {
        "r": [[DOC + d] * (10 + 2 * d) for d in range(n_docs)],
        "p": [[PARTNER + i] * 20 for i in range(n_partners)],
    }
    filter_truth = [[d % 3 != 0] for d in range(n_docs)]
    label_truth = [d % 2 for d in range(n_docs)]           # even: "a"
    join_truth = {("r", d): [(d + i) % 3 == 0 for i in range(n_partners)]
                  for d in range(n_docs)}
    model = LabelingModel(filter_truth, join_truth, label_truth)
    torch = fake_torch()
    arena = cpu_arena(64)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(
        model=MODELS["qwen3-4b-fp8"], gpu_index=0, gpu_count=1,
        device=DEVICES["h100-sxm"]))
    execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
    execution.bind_query(
        torch=torch,
        async_answers=SimpleNamespace(submit=lambda v: v, result=lambda v: v,
                                      dtype=None),
        answer_rows=object(), chunk_tokens=120)
    def submit(rows):
        # a frame entry's row is not read; it carries no log probabilities
        return np.asarray([row if hasattr(row, "shape") else np.zeros(2)
                           for row in rows], np.float32).reshape(-1, 1, 2)

    execution.state["label_readout"] = SimpleNamespace(
        targets=np.asarray([11, 12]), rows=1,
        dtype=np.dtype((np.float32, (1, 2))), submit=submit,
        result=lambda rows: rows)
    state = {
        "torch": torch, "arena": arena, "pipeline": pipeline,
        "model_execution": execution,
        "runtimes": built_in_registry().runtimes,
        "model_spec": MODELS["qwen3-4b-fp8"], "device": DEVICES["h100-sxm"],
        "chunk_tokens": 120, "docs": docs,
    }
    result = execute_single_graph(state, SETTINGS, fused_graph())
    assert not arena.accounting.owned

    passed = [d for d in range(n_docs) if filter_truth[d][0]]
    accepted = [d for d in passed if label_truth[d] == 0]
    matched = [d for d in accepted if any(join_truth[("r", d)])]
    outputs = result["_outputs"]
    metrics = result["node_metrics"]
    labels = outputs[PortRef("classify:r", "scores")]
    assert labels.column("r").to_pylist() == passed
    assert labels.column("topic").to_pylist() == [
        "ab"[label_truth[d]] for d in passed]
    assert outputs[PortRef("classify:r", "ids:r")].column("r").to_pylist() \
        == passed
    assert outputs[PortRef("label:r", "ids:r")].column("r").to_pylist() \
        == accepted
    assert outputs[PortRef("group:0", "ids:r")].column("r").to_pylist() \
        == matched
    # the chain packed every document once; the classification and the
    # join packed only frames and suffixes after resident KV
    prefill = sum(len(docs["r"][d]) for d in range(n_docs))
    assert metrics["filter:r"]["fresh_tokens"] == prefill + n_docs
    assert metrics["classify:r"]["fresh_tokens"] == 2 * len(passed)
    assert metrics["classify:r"]["evaluated_documents"] == len(passed)
    assert metrics["classify:r"]["label_tokens"] == len(passed)
    # some chunk held documents at the chain's stage beside documents
    # at the classification's or the join's
    heads = [{suffix[0] for spec in specs for suffix in spec["suffixes"]}
             for specs in model.launched]
    assert any(QUESTION in chunk and (CUE in chunk or chunk & {FRAME})
               or CLASSIFY_FRAME in chunk and FRAME in chunk
               for chunk in heads)
