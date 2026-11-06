"""Plan a logical plan with Rust's planner, as a session does."""

import json
from types import SimpleNamespace

import _quail_native

from quail.api.compile import compile_with_datafusion
from quail.builtins import PHYSICAL_RULES
from quail.common.expressions import CompileError
from quail.physical.codec import built_in_codecs, encode_value, untagged
from quail.physical.lowering import Vocabulary
from quail.physical.plan import plan_from_json
from quail.specs import H100_SXM, QWEN3_4B_FP8


def node_kinds(plan):
    return [type(node).__name__ for node in plan.nodes]


def tokenizer(text):
    return text.split()


def placements(logical):
    """The column placements the planner prices, each with its rewrites."""
    found, message = _quail_native.column_placements(logical)
    if message is not None:
        raise CompileError(message)
    return tuple(SimpleNamespace(plan=text, rules=tuple(rules))
                 for text, rules in found)


def _optimize(logical, document_tokens, model=QWEN3_4B_FP8, **kwargs):
    """Return a plan with one column placement and its rewrites."""
    candidate, = placements(logical)
    return candidate.plan, candidate.rules


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
        json.dumps({"alternatives": [[json.loads(alternative.plan),
                                      list(alternative.rules)]
                                     for alternative in alternatives],
                    "rules": list(PHYSICAL_RULES)}),
        vocabulary.tokens, vocabulary.decoder())
    if message is not None:
        raise ValueError(message)
    found = json.loads(found)
    if "refusal" not in found:
        found["logical"] = alternatives[found["alternative"]].plan
        found["fields"] = fields
    codecs = {codec.type_name: codec for codec in built_in_codecs()}
    return plan_from_json(found, codecs)


def _plan(logical, document_tokens, model=QWEN3_4B_FP8, gpus=1, order=None,
          attention=None):
    """Plan as a session would: the column placements, then the physical planner."""
    fields = {
        "model": untagged(encode_value(model)),
        "device": untagged(encode_value(H100_SXM)),
        "gpus": gpus, "canvas_draws": 4,
        "document_tokens": {alias: [int(value) for value in lengths]
                            for alias, lengths in document_tokens.items()},
        "plan": json.loads(logical), "attention": attention, "order": order,
    }
    return search(placements(logical), fields)


def compile_sql(sql, catalog, dialect="snowflake", model=QWEN3_4B_FP8):
    """Compile SQL against a catalog, as a session on the model does."""
    return compile_with_datafusion(sql, catalog, model, dialect=dialect)
