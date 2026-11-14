"""Plan a region's operators with Rust's planner, as a session does."""

import json
import re
from dataclasses import asdict
from types import SimpleNamespace

import _quail_native

from quail.api.compile import DATAFUSION_DIALECTS
from quail.builtins import PHYSICAL_RULES
from quail.common.expressions import CompileError
from quail.physical import decode_plan_value
from quail.physical.lowering import Vocabulary
from quail.physical.plan import plan_from_json
from quail.specs import H100_SXM, QWEN3_4B_FP8


def node_kinds(plan):
    return [type(node).__name__ for node in plan.nodes]


def tokenizer(text):
    return text.split()


def placements(operators):
    """The column placements the planner prices, each with its rewrites.

    Args:
        operators: A region's operators as JSON.

    Returns:
        Each placement's operators, as JSON and read (``found``), and its
        rewrites.
    """
    return tuple(SimpleNamespace(operators=text, found=read_operators(text),
                                 rules=tuple(rules))
                 for text, rules in _quail_native.column_placements(operators))


def read_operators(operators):
    """A region's operators as Rust derives them, read for tests.

    Each operator is a namespace of its fields, a list is a tuple, and an
    expression is Quail's expression type, such as ColumnRef.
    """
    return _read(json.loads(operators))


def operators_of(query):
    """The operators Rust derives for a query's one model region, read."""
    return read_operators(query.operators)


def column_tests(found):
    """Each table's tests of answers, in the order its operators run them."""
    return [test for _, tests in found.column_filters for test in tests]


def row_tests(query):
    """The tests DataFusion runs on a region's rows, as explain() shows them.

    Each is the predicate of a FilterExec, or the filter of a join that
    adds tables' columns back, above the ModelRegionExec, with its
    columns named without their positions, outermost first. DataFusion's
    dynamic filters are left out.
    """
    above = query.explain().split("ModelRegionExec")[0]
    tests = []
    for line in above.splitlines():
        line = line.strip()
        if line.startswith("FilterExec: "):
            found = line.removeprefix("FilterExec: ").split(", projection=")[0]
        elif line.startswith("HashJoinExec: ") and ", filter=" in line:
            found = line.split(", filter=")[1].split(", projection=")[0]
        else:
            continue
        if "DynamicFilter" not in found:
            tests.append(re.sub(r"@\d+", "", found))
    return tests


def _read(value):
    if isinstance(value, dict) and "type" in value:
        return decode_plan_value(json.dumps(value))
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _read(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_read(item) for item in value)
    return value


def search(alternatives, fields, word_tokenizer=tokenizer):
    """The cheapest refined Quail plan of the alternatives, or the Refusal.

    Args:
        alternatives: Column placements, each with its plan and rewrites.
        fields: The planning context's fields, as Rust reads them.
        word_tokenizer: A callable from text to tokens; words are
            interned as integer ids.
    """
    vocabulary = Vocabulary(word_tokenizer)
    found, message = _quail_native.plan_operator(
        "search_plan", json.dumps(fields),
        json.dumps({"alternatives": [[json.loads(alternative.operators),
                                      list(alternative.rules)]
                                     for alternative in alternatives],
                    "rules": list(PHYSICAL_RULES)}),
        vocabulary.tokens, vocabulary.decoder())
    if message is not None:
        raise ValueError(message)
    found = json.loads(found)
    if "refusal" not in found:
        found["operators"] = alternatives[found["alternative"]].operators
        found["fields"] = fields
    return plan_from_json(found)


def _plan(operators, document_tokens, model=QWEN3_4B_FP8, gpus=1, order=None,
          attention=None):
    """Plan as a session would: the column placements, then the physical planner."""
    fields = {
        "model": asdict(model),
        "device": asdict(H100_SXM),
        "gpus": gpus, "canvas_draws": 4,
        "document_tokens": {alias: [int(value) for value in lengths]
                            for alias, lengths in document_tokens.items()},
        "operators": json.loads(operators), "attention": attention, "order": order,
    }
    return search(placements(operators), fields)


class _CompilingEngine:
    """An engine that plans model regions and runs none."""

    def __init__(self, model):
        self.turn, self.layout = tuple(model.turn), model.prompt_layout
        self.rules = None

    def run_region(self, *_request):
        raise RuntimeError("compiling a query runs no model region")


def compile_sql(sql, catalog, dialect="snowflake", model=QWEN3_4B_FP8):
    """Plan SQL against a catalog's schemas; return its one region's operators.

    The operators are JSON, as Rust derives them from the region's plan.
    """
    import pyarrow as pa

    def refuse(_columns):
        raise RuntimeError("compiling a query scans no table")

    native = _quail_native.NativeSession(_CompilingEngine(model))
    for name, provider in catalog.providers.items():
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, provider.schema()):
            pass
        native.register(name, sink.getvalue().to_pybytes(), refuse,
                        getattr(provider, "id_col", None) or None)
    (region,), _ = native.region_plans(sql, tuple(model.turn), model.prompt_layout,
                                       DATAFUSION_DIALECTS[dialect])
    found, message, _ = region
    if message is not None:
        raise CompileError(message)
    return json.loads(found)["operators"]
