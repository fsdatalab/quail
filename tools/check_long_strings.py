"""Flag string literals longer than a limit, however many lines they span.

Ruff's line-length rule sees one line at a time, so a prompt or message
built from several short adjacent literals, or a triple-quoted block,
slips past it. This check measures the whole literal after the parser
joins adjacent pieces. Docstrings are exempt. An f-string counts only
its literal text.

    uv run python tools/check_long_strings.py [--limit 200] [paths...]

Exits 1 when any literal is over the limit. Long text belongs in a data
file next to the code that reads it.
"""

import argparse
import ast
import sys
from pathlib import Path

DEFAULT_PATHS = ("quail", "tests", "experiments", "reports", "tools")
DEFAULT_LIMIT = 200


def _docstring_ids(tree):
    ids = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            ids.add(id(body[0].value))
    return ids


def long_strings(path, limit):
    """Yield (line, length) for every over-limit literal in one file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    docstrings = _docstring_ids(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            length = len(node.value)
        elif isinstance(node, ast.JoinedStr):
            length = sum(len(part.value) for part in node.values
                         if isinstance(part, ast.Constant))
        else:
            continue
        if length > limit:
            yield node.lineno, length


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="*", default=DEFAULT_PATHS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help="longest allowed literal, in characters")
    args = parser.parse_args(argv)
    files = []
    for entry in args.paths:
        entry = Path(entry)
        files.extend(sorted(entry.rglob("*.py")) if entry.is_dir()
                     else [entry])
    findings = 0
    for path in files:
        for line, length in long_strings(path, args.limit):
            findings += 1
            print(f"{path}:{line}: string literal of {length} characters "
                  f"(limit {args.limit})")
    if findings:
        print(f"{findings} string literals over {args.limit} characters",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
