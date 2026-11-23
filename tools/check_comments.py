"""Flag inline and block comments in agent-maintained source files.

Quail's source trees (`quail/` and `crates/`) keep no inline or block
comments. Linter pragmas (`# noqa`, `# type:`, `# pragma:`, `# pyright:`,
`# pylint:`, `# ruff:`, `# fmt:`, `# isort:`, `# mypy:`), shebangs, and
license headers (`SPDX-`, `Copyright`) are exempt. Python docstrings are
exempt.

    uv run --no-sync python tools/check_comments.py [paths...]

Exits 1 when any disallowed comment is found.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from pathlib import Path

import tree_sitter_rust
from tree_sitter import Language, Parser

DEFAULT_PATHS = ("quail", "crates")
ALLOWED_PYTHON = re.compile(
    r"^#(\!|\s*(noqa|type:|pragma:|pyright:|pylint:|ruff:|fmt:|isort:|"
    r"mypy:|SPDX-|Copyright))",
    re.IGNORECASE,
)
ALLOWED_RUST = re.compile(
    r"^(/[/!*]+\s*)(SPDX-|Copyright)",
    re.IGNORECASE,
)


def _rust_parser() -> Parser:
    return Parser(Language(tree_sitter_rust.language()))


def _python_comments(path: Path) -> list[tuple[int, str]]:
    source = path.read_text(encoding="utf-8")
    findings = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        if ALLOWED_PYTHON.match(token.string):
            continue
        findings.append((token.start[0], token.string.strip()))
    return findings


def _rust_comments(path: Path, parser: Parser) -> list[tuple[int, str]]:
    raw = path.read_bytes()
    tree = parser.parse(raw)
    findings = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in ("line_comment", "block_comment"):
            text = raw[node.start_byte:node.end_byte].decode("utf-8", "replace")
            if not ALLOWED_RUST.match(text.strip()):
                first = text.splitlines()[0].strip()
                findings.append((node.start_point[0] + 1, first))
            continue
        for child in reversed(node.children):
            stack.append(child)
    return findings


def find_comments(paths: list[str] | tuple[str, ...]) -> list[str]:
    """Return formatted findings for every disallowed comment under paths."""
    parser = _rust_parser()
    findings = []
    for entry in paths:
        root = Path(entry)
        if root.is_file():
            files = [root]
        else:
            files = sorted(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix in (".py", ".rs")
            )
        for path in files:
            if path.suffix == ".py":
                items = _python_comments(path)
            elif path.suffix == ".rs":
                items = _rust_comments(path, parser)
            else:
                continue
            for lineno, text in items:
                preview = text if len(text) <= 80 else text[:77] + "..."
                findings.append(f"{path}:{lineno}: {preview}")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", default=list(DEFAULT_PATHS))
    arguments = parser.parse_args(argv)
    findings = find_comments(arguments.paths)
    for finding in findings:
        print(finding)
    if findings:
        print(f"{len(findings)} disallowed comments", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
