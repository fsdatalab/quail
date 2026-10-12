import pytest
from fakes import letter_tokens
from pdfs import write_pdf

from quail.catalog import Catalog
from quail.explain import logical_tree
from quail.frontend.sql import compile_sql
from quail.logical import (
    Alias,
    CompileError,
    ParseCall,
    Scan,
    SemanticFilter,
    SemanticParse,
)
from quail.pdf import read_pdfs
from quail.planner.logical_rules import push_down_projection


@pytest.fixture
def catalog(tmp_path):
    write_pdf(tmp_path / "a.pdf", ["Invoice for goods", "page two"])
    catalog = Catalog()
    catalog.register("pages", read_pdfs(str(tmp_path / "*.pdf")))
    return catalog


def _compile(catalog, sql, dialect="snowflake"):
    return compile_sql(sql, catalog, letter_tokens, dialect=dialect)


def _chain(plan):
    return [type(node).__name__ for node in plan.walk()]


def test_a_projected_parse_adds_its_column_above_the_scan(catalog):
    plan = _compile(catalog, "SELECT p.path, AI.PARSE(p.document) AS text "
                             "FROM pages p WHERE p.page = 1")
    assert _chain(plan) == ["Scan", "SemanticParse", "Project"]
    scan, parse = plan.walk()[:2]
    assert scan.column == "document" and len(scan.predicates) == 1
    assert parse.name == "text" and parse.call.mode == "text"
    assert plan.projection.columns[-1] == Alias(parse.call, "text")
    assert "SemanticParse: text = AI.PARSE(p.document, mode=text)" in logical_tree(plan)


def test_a_prompt_and_a_classification_read_the_parsed_text(catalog):
    plan = _compile(catalog, (
        "SELECT p.path, AI.CLASSIFY(AI.PARSE(p.document), ['goods', 'services'])"
        " AS kind FROM pages p"))
    assert _chain(plan) == ["Scan", "SemanticParse", "SemanticClassify", "Project"]
    # a parse no column returns gets a hidden name
    assert plan.walk()[1].name == "__parse_p"
    plan = _compile(catalog, (
        "SELECT p.path FROM pages p WHERE "
        "AI.IF(PROMPT('Is this an invoice? {0}', AI.PARSE(p.document)))"), "bq")
    assert _chain(plan) == ["Scan", "SemanticParse", "SemanticFilter", "Project"]
    assert isinstance(plan.walk()[2], SemanticFilter)


def test_one_parse_serves_the_prompt_and_the_projection(catalog):
    plan = _compile(catalog, (
        "SELECT AI_PARSE_DOCUMENT(p.document, {'mode': 'OCR'}) AS t FROM pages p "
        "WHERE AI_FILTER(PROMPT('x {0}', AI_PARSE_DOCUMENT(p.document, "
        "{'mode': 'OCR'})))"))
    parses = [node for node in plan.walk() if isinstance(node, SemanticParse)]
    assert [(node.name, node.call.mode) for node in parses] == [("t", "text")]


def test_projection_pushdown_loads_no_document_or_parsed_values(catalog):
    plan = _compile(catalog, "SELECT p.path, AI.PARSE(p.document) AS text "
                             "FROM pages p")
    rewritten = push_down_projection(plan.root)
    (scan,) = [node for node in _walk(rewritten) if isinstance(node, Scan)]
    assert scan.columns == ("path",)


def _walk(node):
    for child in node.children():
        yield from _walk(child)
    yield node


@pytest.mark.parametrize("sql, message", [
    ("SELECT p.document FROM pages p", "refers to PDF pages"),
    ("SELECT p.path FROM pages p WHERE AI_FILTER(PROMPT('x {0}', p.document))",
     "read it with AI.PARSE"),
    ("SELECT p.path FROM pages p WHERE AI.PARSE(p.document) = 'x'",
     "may be a PROMPT argument"),
    ("SELECT AI.PARSE(p.path) AS t FROM pages p", "reads the document column"),
    ("SELECT AI.PARSE(p.document) FROM pages p", "with an AS name"),
    ("SELECT AI.PARSE(p.document, mode => 'html') AS t FROM pages p",
     "mode must be one of"),
    ("SELECT AI.PARSE(p.document) AS t, AI.PARSE(p.document, mode => 'layout') "
     "AS u FROM pages p", "parsed in two modes"),
    ("SELECT AI_PARSE_DOCUMENT(p.document, {'mode': 'HTML'}) AS t FROM pages p",
     "options are"),
])
def test_document_columns_and_parse_calls_outside_their_places_are_refused(
        catalog, sql, message):
    with pytest.raises(CompileError, match=message):
        _compile(catalog, sql)


def test_select_star_leaves_out_the_document_column(catalog):
    plan = _compile(catalog, "SELECT *, AI.PARSE(p.document) AS t FROM pages p")
    names = [getattr(column, "column", getattr(column, "name", None))
             for column in plan.projection.columns]
    assert names == ["path", "page", "page_count", "t"]


def test_parse_call_validates_its_mode():
    with pytest.raises(CompileError):
        ParseCall(Scan("pages", "p", "document").output_schema()[0], "html").validate()
