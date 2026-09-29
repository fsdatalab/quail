"""A classification's stages inside the chain and join of its table."""

from types import SimpleNamespace

import numpy as np
import pytest
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
# two-token labels sharing a first token: the trie path [CUE, 11] is
# read at both rows, beside filter and join rows read at one
LABEL_IDS = ((11, 13), (11, 14))
TARGETS = [11, 13, 14]


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
                    # each row's log probabilities over the targets: the
                    # document's label token at that depth wins
                    wanted = LABEL_IDS[self.label_truth[spec["key"][1]]]
                    for depth in range(len(suffix)):
                        rows.append(np.asarray(
                            [-1.0 if token == wanted[depth] else -5.0
                             for token in TARGETS], np.float32))
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
        label_token_ids=LABEL_IDS, scoring="trie_paths", share_prefixes=True)
    nodes = [
        Scan(node_id="input:r", alias="r", input_id="r"),
        Scan(node_id="input:p", alias="p", input_id="p"),
        AiFilter(
            node_id="filter:r",
            inputs=input_ports((PortRef("input:r", "ids:r"),)),
            alias="r", arena_writes=True,
            stages=(FilterStage(0, 1, 0, 0.8, 14 * 0.8),),
            question_token_ids=((QUESTION,),)),
        AiClassify(
            node_id="classify:r",
            inputs=input_ports((PortRef("filter:r", "ids:r"),)),
            backend_name="quail", model="qwen3-4b-fp8", spec=spec),
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


def test_a_canvas_classification_runs_in_its_filter_chain_pipeline(monkeypatch):
    """A filter, then a diffusion model's classification: one run.

    Each survivor's denoising steps run over its resident KV, in
    chunks that also hold other documents' filter rows; the filter's
    fixed canvas reads no self-conditioning input.
    """
    torch = pytest.importorskip("torch")
    from dataclasses import replace

    from quail.backends.quail.executor import classify as classify_module
    from quail.execution.pipelines import build_pipelines
    from quail.specs import Denoising

    settings = Denoising(
        canvas_rows=4, max_steps=5, t_min=0.4, t_max=0.8, entropy_bound=0.1,
        confidence_threshold=0.005, stability_threshold=1, logit_softcap=30.0,
        stop_token_ids=(0,))
    spelling = {1: "a", 2: "b", 3: "\n", 5: "x", 6: " "}
    tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens: "".join(
        spelling.get(i, "") for i in ids))
    spec = ClassifySpec(
        name="topic", aliases=("r",), query_template="", arguments=(),
        expected_inputs=4, estimated_seconds=0.0,
        prompt_token_parts=((), (91, 92, 93)), labels=("a", "b"),
        label_token_ids=((1,), (2,)), scoring="canvas")
    nodes = [
        Scan(node_id="input:r", alias="r", input_id="r"),
        AiFilter(
            node_id="filter:r",
            inputs=input_ports((PortRef("input:r", "ids:r"),)),
            alias="r", arena_writes=True,
            stages=(FilterStage(0, 1, 0, 0.8, 4 * 0.8),),
            question_token_ids=((QUESTION,),)),
        AiClassify(
            node_id="classify:r",
            inputs=input_ports((PortRef("filter:r", "ids:r"),)),
            backend_name="quail", model="diffusion-gemma-26b-a4b-fp8",
            spec=spec),
    ]
    graph = PhysicalGraph(tuple(nodes), PortRef("classify:r", "scores"))
    chains = {pipeline.node_ids for pipeline in build_pipelines(graph).values()}
    assert chains == {("filter:r", "classify:r")}

    filter_truth = [[1], [0], [1], [1]]
    # a survivor's canvas settles on "a", "b", or a stop token first
    answers = {0: [[1, 0, 5, 5]] * 5, 2: [[5, 5, 5, 5]] + [[2, 0, 6, 6]] * 4,
               3: [[0, 5, 5, 5]] * 5}
    vocab = 8
    head = torch.eye(vocab, dtype=torch.bfloat16)
    packed = {}
    conditioned = {}

    def forward(chunk):
        rows = []
        canvas_rows = 0
        for index, entry in enumerate(chunk.specs):
            document = entry["key"][1]
            for suffix in entry["suffixes"]:
                if entry.get("canvas") is not None:
                    packed.setdefault(document, []).append(list(entry["canvas"]))
                    assert entry["read_all_rows"]
                    # the step reads the soft embedding the step before
                    # wrote, zero at its first step
                    soft = chunk.meta["canvas"]["conditioning"][
                        canvas_rows:canvas_rows + 4].float()
                    canvas_rows += 4
                    conditioned.setdefault(document, []).append(soft)
                    wanted = answers[document][len(packed[document]) - 1]
                    for token in wanted:
                        row = torch.full((vocab,), -1000.0)
                        row[token] = 1000.0
                        rows.append(row)
                else:
                    # a filter row on the fixed canvas: TRUE is any
                    # nonzero
                    assert suffix[0] == QUESTION
                    rows.append(torch.full(
                        (vocab,), float(filter_truth[document][0])))
        return torch.stack(rows)

    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    monkeypatch.setattr(torch.cuda, "Event", lambda **kw: SimpleNamespace(
        record=lambda: None, elapsed_time=lambda other: 2.0))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(classify_module, "full_output_head", lambda model: head)
    pipeline = fake_pipeline(forward_chunk=forward, canvas_ids=(7,),
                             tree_attention=False,
                             normalizer=torch.tensor(2.0, dtype=torch.bfloat16))
    arena = cpu_arena(64)
    model_spec = replace(MODELS["qwen3-4b-fp8"], name="tiny")
    execution = QuailModelExecution(SimpleNamespace(
        model=model_spec, gpu_index=0, gpu_count=1, device=DEVICES["h100-sxm"]))
    execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
    # the filter's readout takes its rows of the chunk's hidden rows
    execution.bind_query(
        torch=torch,
        async_answers=SimpleNamespace(submit=lambda v: [int(row[0]) for row in v],
                                      result=lambda v: v, dtype=None),
        answer_rows=object(), chunk_tokens=64)
    execution._state.update(
        model_spec=SimpleNamespace(name="tiny", vocab=vocab,
                                   denoising=settings),
        tokenizer=tokenizer)
    docs = {"r": [[DOC + d] * (3 + d) for d in range(4)]}
    state = {
        "torch": torch, "arena": arena, "pipeline": pipeline,
        "model_execution": execution,
        "runtimes": built_in_registry().runtimes,
        "model_spec": model_spec, "device": DEVICES["h100-sxm"],
        "chunk_tokens": 64, "docs": docs,
    }
    result = execute_single_graph(state, SETTINGS, graph)
    assert not arena.accounting.owned
    outputs = result["_outputs"]
    metrics = result["node_metrics"]
    assert outputs[PortRef("filter:r", "ids:r")].column("r").to_pylist() \
        == [0, 2, 3]
    labels = outputs[PortRef("classify:r", "scores")]
    # document 3 answers a stop token first and names no label
    assert labels.column("r").to_pylist() == [0, 2]
    assert labels.column("topic").to_pylist() == ["a", "b"]
    assert sorted(packed) == [0, 2, 3]
    assert [len(packed[d]) for d in (0, 2, 3)] == [2, 3, 2]
    # the first step reads zero conditioning; the next reads the step
    # before's probabilities times the embedding and the normalizer
    assert not conditioned[2][0].any()
    assert torch.allclose(conditioned[2][1], 2.0 * torch.eye(vocab)[[5, 5, 5, 5]])
    # the classification packed a frame, then a cue and a canvas per
    # step, over the survivors' resident KV: no second prefill
    steps = 2 + 3 + 2
    assert metrics["classify:r"]["fresh_tokens"] == 3 * 2 + steps * 5
    assert metrics["classify:r"]["evaluated_documents"] == 3
    assert metrics["filter:r"]["fresh_tokens"] == sum(map(len, docs["r"])) + 4


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
    def submit(rows, rows_per_answer=None):
        # (answers, rows, targets), padded; a frame entry's one row is
        # not read and carries no log probabilities
        rows_per_answer = rows_per_answer or [1] * len(rows)
        padded = np.full((len(rows_per_answer), 2, len(TARGETS)), np.nan,
                         np.float32)
        start = 0
        for answer, count in enumerate(rows_per_answer):
            for offset in range(count):
                row = rows[start + offset]
                padded[answer, offset] = (row if hasattr(row, "shape")
                                          else np.zeros(len(TARGETS)))
            start += count
        return padded

    execution.state["label_readout"] = SimpleNamespace(
        targets=np.asarray(TARGETS), rows=2,
        dtype=np.dtype((np.float32, (2, len(TARGETS)))), submit=submit,
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
        ("a", "b")[label_truth[d]] for d in passed]
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
    # a frame and the two-token path per survivor
    assert metrics["classify:r"]["fresh_tokens"] == 3 * len(passed)
    assert metrics["classify:r"]["evaluated_documents"] == len(passed)
    assert metrics["classify:r"]["label_tokens"] == 2 * len(passed)
    # some chunk held documents at the chain's stage beside documents
    # at the classification's or the join's
    heads = [{suffix[0] for spec in specs for suffix in spec["suffixes"]}
             for specs in model.launched]
    assert any(QUESTION in chunk and (CUE in chunk or chunk & {FRAME})
               or CLASSIFY_FRAME in chunk and FRAME in chunk
               for chunk in heads)


def chained_graph():
    """A decoded classification, its label filter, and a second classification."""
    first = ClassifySpec(
        name="topic", aliases=("r",), query_template="", arguments=(),
        expected_inputs=14, estimated_seconds=0.0,
        prompt_token_parts=((), (CLASSIFY_FRAME, CUE)), labels=("a", "b"),
        label_token_ids=LABEL_IDS, scoring="trie_decode")
    second = ClassifySpec(
        name="kind", aliases=("r",), query_template="", arguments=(),
        expected_inputs=7, estimated_seconds=0.0,
        prompt_token_parts=((), (CLASSIFY_FRAME, CUE)), labels=("a", "b"),
        label_token_ids=LABEL_IDS, scoring="trie_paths")
    nodes = [
        Scan(node_id="input:r", alias="r", input_id="r"),
        AiClassify(
            node_id="classify:r",
            inputs=input_ports((PortRef("input:r", "ids:r"),)),
            backend_name="quail", model="qwen3-4b-fp8", spec=first),
        LabelFilter(
            node_id="label:r",
            inputs=input_ports((PortRef("classify:r", "scores"),)),
            score_name="topic", aliases=("r",), comparison="in",
            threshold=0.0, selectivity=0.5, written_pos=1, accepted=("a",)),
        AiClassify(
            node_id="classify2:r",
            inputs=input_ports((PortRef("label:r", "scores"),)),
            backend_name="quail", model="qwen3-4b-fp8", spec=second),
    ]
    return PhysicalGraph(tuple(nodes), PortRef("classify2:r", "scores"))


def test_a_decoded_classification_hands_its_documents_on_through_the_label_filter(
        monkeypatch):
    from quail.backends.quail.executor import classify as classify_module
    from quail.execution.pipelines import build_pipelines

    graph = chained_graph()
    chains = {pipeline.node_ids for pipeline in build_pipelines(graph).values()}
    assert chains == {("classify:r", "label:r", "classify2:r")}

    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    n_docs = 14
    docs = {"r": [[DOC + d] * (10 + 2 * d) for d in range(n_docs)]}
    label_truth = [d % 2 for d in range(n_docs)]           # even: "a"

    class DecodingModel(LabelingModel):
        def forward_chunk(self, chunk):
            rows = []
            for spec in chunk.specs:
                for suffix in spec["suffixes"]:
                    if suffix[0] != CUE:
                        rows.append(0)
                        continue
                    wanted = LABEL_IDS[self.label_truth[spec["key"][1]]]
                    count = len(suffix) if spec.get("read_all_rows") else 1
                    for depth in range(count):
                        # the last row of a decode round wants the next
                        # token; every row of a path wants its own
                        want = wanted[len(suffix) - 1] if count == 1 \
                            else wanted[depth]
                        rows.append(np.asarray(
                            [-1.0 if token == want else -5.0
                             for token in TARGETS], np.float32))
            self.launched.append(chunk.specs)
            return rows

    model = DecodingModel([], {}, label_truth)
    torch = fake_torch()
    # the readout constructor is replaced below; its arguments are read
    torch.nn = SimpleNamespace(functional=None)
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

    class FakeReadout:
        # the two classifications read one row and two rows per answer
        def __init__(self, targets, rows):
            self.targets = np.asarray(targets)
            self.rows = rows
            self.dtype = np.dtype((np.float32, (rows, len(targets))))

        def submit(self, rows, rows_per_answer=None):
            rows_per_answer = rows_per_answer or [1] * len(rows)
            padded = np.full((len(rows_per_answer), self.rows,
                              len(self.targets)), np.nan, np.float32)
            start = 0
            for answer, count in enumerate(rows_per_answer):
                for offset in range(count):
                    row = rows[start + offset]
                    padded[answer, offset] = (
                        row if hasattr(row, "shape")
                        else np.zeros(len(self.targets)))
                start += count
            return padded

        def result(self, rows):
            return rows

    monkeypatch.setattr(classify_module, "full_output_head",
                        lambda model: SimpleNamespace(shape=(1, 1),
                                                      dtype="fake"))
    monkeypatch.setattr(
        classify_module, "AsyncLabelLogprobs",
        lambda torch, F, head, targets, rows, normalize: FakeReadout(
            targets, rows))
    state = {
        "torch": torch, "arena": arena, "pipeline": pipeline,
        "model_execution": execution,
        "runtimes": built_in_registry().runtimes,
        "model_spec": MODELS["qwen3-4b-fp8"], "device": DEVICES["h100-sxm"],
        "chunk_tokens": 120, "docs": docs,
    }
    result = execute_single_graph(state, SETTINGS, graph)
    assert not arena.accounting.owned
    outputs = result["_outputs"]
    metrics = result["node_metrics"]
    accepted = [d for d in range(n_docs) if label_truth[d] == 0]
    labels = outputs[PortRef("classify:r", "scores")]
    assert labels.column("topic").to_pylist() == [
        ("a", "b")[label_truth[d]] for d in range(n_docs)]
    assert outputs[PortRef("label:r", "ids:r")].column("r").to_pylist() \
        == accepted
    second = outputs[PortRef("classify2:r", "scores")]
    assert second.column("r").to_pylist() == accepted
    assert second.column("kind").to_pylist() == ["a"] * len(accepted)
    # the first label rides along, so a projection can name it
    assert second.column("topic").to_pylist() == ["a"] * len(accepted)
    # the first classification packed every document once; the second
    # packed only a frame and the label path per accepted document
    prefill = sum(len(docs["r"][d]) for d in range(n_docs))
    assert metrics["classify:r"]["fresh_tokens"] >= prefill
    assert metrics["classify2:r"]["fresh_tokens"] == 3 * len(accepted)
    assert metrics["classify2:r"]["evaluated_documents"] == len(accepted)
