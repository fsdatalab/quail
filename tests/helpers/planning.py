"""Plan as a session would: the column placements, then the physical planner."""

from quail.api.compile import compile_with_datafusion
from quail.builtins import built_in_registry
from quail.common.tokens import TokenLengths
from quail.planner import native
from quail.planner.optimize import plan_query
from quail.specs import H100_SXM, QWEN3_4B_FP8


def node_kinds(plan):
    return [type(node).__name__ for node in plan.nodes]


def tokenizer(text):
    return text.split()


def _optimize(logical, document_tokens, model=QWEN3_4B_FP8, **kwargs):
    """Return a plan with one column placement and its rewrites."""
    candidate, = native.column_placements(logical)
    return candidate.plan, candidate.rules


def placements(logical):
    """The column placements the planner prices, each with its rewrites."""
    return native.column_placements(logical)


def _plan(logical, document_tokens, model=QWEN3_4B_FP8, **kwargs):
    """Plan as a session would: the tree rewrites, then the physical planner."""
    return plan_query(logical, model=model, device=H100_SXM,
                      registry=built_in_registry(),
                      document_tokens=token_lengths(document_tokens),
                      tokenizer=tokenizer, **kwargs)


def token_lengths(document_tokens) -> dict:
    """Return each alias's counts as the TokenLengths the planner reads."""
    return {alias: lengths if isinstance(lengths, TokenLengths)
            else TokenLengths(lengths)
            for alias, lengths in document_tokens.items()}


def compile_sql(sql, catalog, dialect="snowflake", model=QWEN3_4B_FP8):
    """Compile SQL against a catalog, as a session on the model does."""
    return compile_with_datafusion(sql, catalog, model, dialect=dialect)
