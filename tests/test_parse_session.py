"""AI.PARSE through a Session: planning, refusals, and returned text."""

from types import SimpleNamespace

import pytest
from fakes import cpu_arena, fake_pipeline, fake_torch, letter_tokens
from pdfs import write_pdf

import quail
from quail.backends.quail import QuailModelExecution
from quail.backends.quail.graph import execute_single_graph
from quail.execution.execute import execute_query
from quail.execution.types import PhysicalResponse
from quail.parsers import PdfiumParser
from quail.physical import Parse, Scan
from quail.saved_parses import SavedParses
from quail.specs import DEVICES, MODELS

SQL = ("SELECT p.path, p.page, AI.PARSE(p.document) AS text "
       "FROM pages p WHERE p.page = 2")


@pytest.fixture(autouse=True)
def saved_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("QUAIL_PARSE_DIR", str(tmp_path / "parses"))


@pytest.fixture
def folder(tmp_path):
    write_pdf(tmp_path / "a.pdf", ["first page", "second page of a"])
    write_pdf(tmp_path / "b.pdf", ["only page"])
    write_pdf(tmp_path / "c.pdf", ["one", ""])
    return tmp_path


def _session(folder, parser="pdfium", **config):
    session = quail.Session(
        quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm",
                           parser=parser, **config),
        tokenizer=letter_tokens)
    session.register("pages", quail.read_pdfs(str(folder / "*.pdf")))
    return session


def _execute(request, query):
    """Run the plan's executor graph on the CPU with the pdfium parser."""
    parser = PdfiumParser(workers=1)
    execution = QuailModelExecution(SimpleNamespace())
    arena = cpu_arena(16)
    execution.bind_loaded_model(model=object(), arena=arena,
                                pipeline=fake_pipeline())
    state = {
        "torch": fake_torch(), "arena": arena, "model_execution": execution,
        "runtimes": query.session.registry.runtimes,
        "model_spec": MODELS["qwen3-4b-fp8"], "device": DEVICES["h100-sxm"],
        "chunk_tokens": 120,
        "docs": {node.alias: request.inputs[node.input_id].documents
                 for node in query.plan().nodes if isinstance(node, Scan)},
        "columns": request.column_tables(),
        "parser": lambda node: parser, "tokenize": letter_tokens,
        "saved_parses": SavedParses(),
    }
    try:
        report = execute_single_graph(
            state, {**query.plan().settings, "pre_ids": [], "retention": {}},
            query.plan().graph)
    finally:
        parser.close()
    return PhysicalResponse(report.pop("_outputs"), report)


def test_parse_only_the_rows_the_scan_keeps_and_return_their_text(folder):
    query = _session(folder).sql(SQL)
    plan = query.plan()
    (parse,) = [node for node in plan.nodes if isinstance(node, Parse)]
    assert (parse.parser, parse.name, parse.lanes) == ("pdfium", "text", 0)
    assert plan.settings["parser"] == "pdfium"
    result = execute_query(query, physical_executor=lambda r: _execute(r, query))
    rows = result.collect().to_pylist()
    assert [(row["p.path"].rsplit("/", 1)[-1], row["p.page"], row["text"])
            for row in rows] == [("a.pdf", 2, "second page of a"), ("c.pdf", 2, "")]
    metrics = result.report["node_metrics"]["parse:p"]
    assert metrics["pages_parsed"] == 2 and metrics["parse_empty_pages"] == 1


def test_explain_shows_the_parse(folder):
    text = _session(folder).sql(SQL).explain()
    assert "SemanticParse: text = AI.PARSE(p.document, mode=text)" in text
    assert "parse" in text.lower()


def test_parse_estimates_lengths_from_stored_text(folder):
    session = _session(folder)
    lengths = session.estimate_parse_lengths("pages")
    # letter_tokens gives one token per character; five pages hold 45
    # characters
    assert len(lengths) == 5 and sum(lengths) == pytest.approx(45, abs=5)


@pytest.mark.parametrize("parser, config, message", [
    (None, {}, "AI.PARSE needs a parser"),
    ("pdfium", {"gpus": 2}, "one GPU"),
])
def test_refusals(folder, parser, config, message):
    plan = _session(folder, parser, **config).sql(SQL).plan()
    assert isinstance(plan, quail.planner.plan.Refusal)
    assert message in plan.reasons[0]


def test_layout_mode_is_refused_until_a_parser_supports_it(folder):
    plan = _session(folder).sql(
        "SELECT AI.PARSE(p.document, mode => 'layout') AS t FROM pages p").plan()
    assert "not supported by parser 'pdfium'" in plan.reasons[0]


def test_an_unknown_parser_is_refused(folder):
    with pytest.raises(quail.RefusalError, match="unknown parser"):
        _session(folder, "tesseract")


def test_an_ocr_parser_takes_memory_from_the_reader(folder):
    plain = quail.Session(
        quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm",
                           parser="lightonocr-3-0.8b"), tokenizer=letter_tokens)
    plain.register("pages", quail.read_pdfs(str(folder / "*.pdf")))
    plan = plain.sql(SQL).plan()
    (parse,) = [node for node in plan.nodes if isinstance(node, Parse)]
    # half the memory left after both models' weights, at 103 MB a lane
    assert 200 < parse.lanes <= 256
    assert 25 < plan.settings["parser_gib"] < 40


def test_a_limit_parses_only_the_rows_it_returns(folder):
    query = _session(folder).sql(
        "SELECT p.path, AI.PARSE(p.document) AS text FROM pages p LIMIT 2")
    assert query.plan().settings["parse_limit"] == 2
    result = execute_query(query, physical_executor=lambda r: _execute(r, query))
    assert result.count() == 2
    assert result.report["node_metrics"]["parse:p"]["pages_parsed"] == 2


def test_a_second_query_reads_the_saved_rows(folder):
    session = _session(folder)
    first = session.sql(SQL)
    (parse,) = [node for node in first.plan().nodes if isinstance(node, Parse)]
    assert parse.saved and parse.saved_rows == 0
    execute_query(first, physical_executor=lambda r: _execute(r, first))
    second = session.sql(SQL)
    (parse,) = [node for node in second.plan().nodes if isinstance(node, Parse)]
    assert parse.saved_rows == 2
    assert "saved results: 2 of the table's rows" in second.explain()
    result = execute_query(second, physical_executor=lambda r: _execute(r, second))
    metrics = result.report["node_metrics"]["parse:p"]
    assert metrics["rows_from_saved"] == 2 and "pages_parsed" not in metrics
    assert [row["text"] for row in result.collect().to_pylist()] == [
        "second page of a", ""]


def test_saved_results_can_be_turned_off(folder):
    session = _session(folder, save_parses=False)
    query = session.sql(SQL)
    (parse,) = [node for node in query.plan().nodes if isinstance(node, Parse)]
    assert not parse.saved
    execute_query(query, physical_executor=lambda r: _execute(r, query))
    assert not list(SavedParses().directory.glob("*/*.txt"))
