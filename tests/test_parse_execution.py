"""AI.PARSE on the executor: a parse alone, and a parse feeding a filter."""

import random
from types import SimpleNamespace

from fakes import (
    DOC,
    QUESTION,
    SETTINGS,
    FakeModel,
    cpu_arena,
    fake_pack,
    fake_pipeline,
    fake_torch,
)

from quail.backends.quail import QuailModelExecution
from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.graph import execute_single_graph
from quail.builtins import built_in_registry
from quail.execution.pipelines import build_pipelines
from quail.physical import (
    AiFilter,
    FilterStage,
    Parse,
    PhysicalGraph,
    PortRef,
    Scan,
)
from quail.physical.base import input_ports
from quail.specs import DEVICES, MODELS

ROWS = 12


class FakeParser:
    """Parse row d as 'd:<length>' and yield rows out of order."""

    def __init__(self, lengths):
        self.lengths = lengths
        self.stats = {}
        self.parsed = []

    def parse(self, rows):
        rows = list(rows)
        for row, path_file, pages in reversed(rows):
            assert path_file == f"file-{row}" and pages == [1]
            self.parsed.append(row)
            yield row, f"{row}:{self.lengths[row]}"
        self.stats = {"pages_parsed": len(rows)}


def tokenize(text):
    row, length = (int(part) for part in text.split(":"))
    return [DOC + row] * length


def _graph(with_filter):
    nodes = [
        Scan(node_id="input:r", alias="r", input_id="r", predicates=()),
        Parse(node_id="parse:r",
              inputs=input_ports((PortRef("input:r", "ids:r"),)),
              alias="r", name="text", parser="fake", expected_rows=ROWS),
    ]
    root = PortRef("parse:r", "ids:r")
    if with_filter:
        nodes.append(AiFilter(
            node_id="filter:r",
            inputs=input_ports((PortRef("parse:r", "ids:r"),)),
            alias="r", stages=(FilterStage(0, 1, 0, 0.5, ROWS / 2),),
            question_token_ids=((QUESTION,),)))
        root = PortRef("filter:r", "ids:r")
    return PhysicalGraph(tuple(nodes), root)


def _run(monkeypatch, graph, chunk_tokens=120):
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    rng = random.Random(3)
    lengths = [rng.randrange(10, 40) for _ in range(ROWS)]
    truth = [[rng.randrange(2)] for _ in range(ROWS)]
    model = FakeModel(truth, {})
    parser = FakeParser(lengths)
    torch = fake_torch()
    arena = cpu_arena(64)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace())
    execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
    execution.bind_query(
        torch=torch,
        async_answers=SimpleNamespace(submit=lambda v: v, result=lambda v: v,
                                      dtype=None),
        answer_rows=object(), chunk_tokens=chunk_tokens)
    pages = [(f"file-{row}", [1]) for row in range(ROWS)]
    state = {
        "torch": torch, "arena": arena, "pipeline": pipeline,
        "model_execution": execution,
        "runtimes": built_in_registry().runtimes,
        "model_spec": MODELS["qwen3-4b-fp8"], "device": DEVICES["h100-sxm"],
        "chunk_tokens": chunk_tokens, "docs": {"r": pages},
        "parser": lambda node: parser, "tokenize": tokenize,
    }
    result = execute_single_graph(state, SETTINGS, graph)
    assert not arena.accounting.owned
    return result, model, parser, lengths, truth


def test_a_parse_alone_returns_every_rows_text(monkeypatch):
    result, _, parser, lengths, _ = _run(monkeypatch, _graph(False))
    parsed = result["_outputs"][PortRef("parse:r", "parsed")]
    assert parsed.column("r").to_pylist() == list(range(ROWS))
    assert parsed.column("text").to_pylist() == [
        f"{row}:{lengths[row]}" for row in range(ROWS)]
    assert result["node_metrics"]["parse:r"]["parsed_reader_tokens"] == sum(lengths)


def test_a_filter_reads_parsed_rows_in_rounds(monkeypatch):
    graph = _graph(True)
    pipeline = build_pipelines(graph)["parse:r"]
    assert [member.node_id for member in pipeline.members] == [
        "parse:r", "filter:r"]
    result, model, parser, lengths, truth = _run(monkeypatch, graph)
    survivors = result["_outputs"][PortRef("filter:r", "ids:r")]
    assert survivors.column("r").to_pylist() == [
        row for row in range(ROWS) if truth[row][0]]
    metrics = result["node_metrics"]["parse:r"]
    # 12 rows of 10 to 39 tokens in rounds of 120 tokens: the filter
    # starts before the last row is parsed
    assert metrics["parse_rounds"] > 1
    first_round = {spec["key"][1] for spec in model.launched[0][1]}
    assert parser.parsed[-1] not in first_round
    # the documents the filter packed are the parsed rows' tokens
    packed = {spec["key"][1]: spec["prefix"] for _, specs in model.launched
              for spec in specs if spec["prefix"] is not None}
    assert sorted(packed) == list(range(ROWS))
    for row, prefix in packed.items():
        assert list(prefix)[-lengths[row]:] == [DOC + row] * lengths[row]
