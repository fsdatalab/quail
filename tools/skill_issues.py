"""Measure the source against the rules of PufferLib's SKILL_ISSUES.md.

Joseph Suarez's refactoring algorithm shortens source without golfing:
inline each function used once, replace error handling with asserts,
flatten nesting, merge functions with their callers, then apply a
style guide. The script counts what each step removes, so a refactor
can drive each count down while the tests hold behavior:

    uv run --no-sync python tools/skill_issues.py --json base.json
    uv run --no-sync python tools/skill_issues.py --compare base.json
    uv run --no-sync python tools/skill_issues.py --list single_use_functions

The source is `quail/` and the Rust under `crates/*/src`, without test
code. Tests, benchmarks, tools, and docs count only as callers. Python
is parsed with `ast` and `tokenize`, and Rust with tree-sitter; no
metric matches text. Rust references come from rust-analyzer's SCIP
index, which resolves paths, methods, and macro expansions; the index
is cached under `target/skill-issues/` by a hash of its inputs. Python
references are resolved through scopes, imports, classes, and `self`.
A reference the script cannot resolve, such as a method called on an
untyped value or a name in a string, counts toward every function of
that name, so a count of one is never a guess.

A function whose callers the script cannot see is left out of the
reference metrics and counted under `excluded_functions` with the
reason: a trait implementation, a pyo3 entry point, a Triton kernel,
a public item of a published crate or of `quail.__all__`, a function
behind a framework decorator, an abstract method, or a public method
of a class that another package's code reaches: through a base class,
an argument, an attribute it is stored in, or its name in a string.
Rust functions named in an attribute's string or a `macro_rules!` body
count those names as references.

The same tree gives the same output, byte for byte.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import subprocess
import sys
import tokenize
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIPPED_DIRECTORIES = {".venv", "target", "__pycache__", "node_modules", ".git",
                       "results", ".pytest_cache", ".ruff_cache"}
TEST_DIRECTORIES = {"tests", "examples", "benches"}
PUBLISHED_CRATES = {"quail-planner", "quail-datafusion", "quail"}

# metric -> (Joseph's step, what one unit is); a lower count is better
METRICS = {
    "code_tokens": ("size", "tokens outside comments and docstrings"),
    "code_lines": ("size", "lines that hold a code token"),
    "files": ("size", "source files"),
    "single_use_functions": ("1 inline", "functions with one reference"),
    "test_only_functions": ("1 inline", "functions only tests reference"),
    "unused_functions": ("1 inline", "functions nothing references"),
    "error_guards": ("2 assert", "an if (or let-else) whose body only raises"),
    "error_sites": ("2 assert", "raise, Err(...), bail! and *_err! macros"),
    "error_handlers": ("2 assert", "except clauses, map_err, ok_or, context"),
    "error_propagations": ("2 assert", "Rust ? operators"),
    "guard_lookups": ("2 assert", "hasattr, and getattr with a default"),
    "depth_excess": ("3 nesting", "per code line, the blocks past depth 2"),
    "deep_functions": ("3 nesting", "functions with a line past depth 3"),
    "pass_through_functions": ("4 merge", "a body that only forwards"),
    "unused_parameters": ("4 merge", "defaulted parameters no call passes"),
    "constant_arguments": ("4 merge", "parameters every call passes alike"),
    "lines_over_100": ("5 style", "lines over the hard limit"),
    "lines_over_80": ("5 style", "lines over the soft limit"),
    "one_line_blocks": ("5 style", "if, for, while with the body inline"),
    "divider_comments": ("5 style", "comments with ---, ===, or ###"),
    "aligned_continuations": ("5 style", "lines aligned to an open bracket"),
    "bare_blocks": ("5 style", "Rust blocks that only open a scope"),
    "functions": ("info", "function definitions"),
    "excluded_functions": ("info", "functions with callers out of sight"),
    "comment_lines": ("info", "lines with only a comment"),
    "docstring_lines": ("info", "docstring and doc comment lines"),
    "asserts": ("info", "assert, expect, unwrap, panic"),
    "max_depth": ("info", "the deepest line in any function"),
}
LANGUAGES = ("python", "rust")
DIVIDER = re.compile(r"([-=#*~_])\1\1")
TRANSPARENT_DECORATORS = {
    "property", "staticmethod", "classmethod", "cached_property", "cache",
    "lru_cache", "contextmanager", "abstractmethod", "override",
    "no_grad", "inference_mode", "setter", "getter", "deleter", "wraps",
}
# bases whose class calls no method of a subclass by name
INERT_BASES = {"object", "Protocol", "Generic", "ABC", "NamedTuple",
               "TypedDict", "Enum", "IntEnum", "StrEnum", "Exception",
               "ValueError", "RuntimeError", "TypeError", "KeyError"}


class Report:
    """Counts per metric, language, and file, and the items behind them."""

    def __init__(self):
        self.counts = defaultdict(int)
        self.items = []

    def add(self, metric, language, path, line=0, name="", detail="",
            amount=1, listed=True):
        key = (metric, language, path)
        if metric == "max_depth":
            self.counts[key] = max(self.counts[key], amount)
        else:
            self.counts[key] += amount
        if listed:
            self.items.append({"metric": metric, "language": language,
                               "path": path, "line": line, "name": name,
                               "detail": detail})

    def to_dict(self) -> dict:
        metrics = {}
        for (metric, language, path), value in sorted(self.counts.items()):
            metrics.setdefault(metric, {}).setdefault(language, {})[path] = value
        items = sorted(self.items, key=lambda item: (
            item["metric"], item["path"], item["line"], item["name"],
            item["detail"]))
        return {"metrics": metrics, "items": items}


def totals(data: dict, under: str = "") -> dict:
    """Return metric -> language -> total over the paths under a prefix."""
    out = {}
    for metric in METRICS:
        for language in LANGUAGES:
            values = [value for path, value in
                      data["metrics"].get(metric, {}).get(language, {}).items()
                      if path.startswith(under)]
            out.setdefault(metric, {})[language] = max(values, default=0) \
                if metric == "max_depth" else sum(values)
    return out


def source_files(suffix):
    """Yield (relative path, category) of every file with the suffix.

    The category is "source" for measured code, "test" for tests and
    examples, and "caller" for everything else that may call the source.
    """
    for path in sorted(ROOT.rglob(f"*{suffix}")):
        parts = path.relative_to(ROOT).parts
        if SKIPPED_DIRECTORIES & set(parts):
            continue
        relative = "/".join(parts)
        if TEST_DIRECTORIES & set(parts):
            yield relative, "test"
        elif suffix == ".py":
            yield relative, "source" if parts[0] == "quail" else "caller"
        else:
            yield relative, "source" if parts[0] == "crates" else "caller"


class Function:
    """One function definition and the references to it."""

    def __init__(self, language, key, name, path, line, end_line):
        self.language, self.key, self.name = language, key, name
        self.path, self.line, self.end_line = path, line, end_line
        self.excluded = ""
        # (category, path, line, resolved, call or None)
        self.references = []

    def record(self, report):
        """Add the function to the reference metrics."""
        report.add("functions", self.language, self.path, listed=False)
        if self.excluded:
            report.add("excluded_functions", self.language, self.path,
                       self.line, self.name, self.excluded)
            return
        size = f"{self.end_line - self.line + 1} lines"
        outside_tests = [reference for reference in self.references
                         if reference[0] != "test"]
        if not self.references:
            report.add("unused_functions", self.language, self.path,
                       self.line, self.name, size)
        elif not outside_tests:
            report.add("test_only_functions", self.language, self.path,
                       self.line, self.name, size)
        elif len(self.references) == 1 and self.references[0][3]:
            _, path, line, _, _ = self.references[0]
            report.add("single_use_functions", self.language, self.path,
                       self.line, self.name, f"{size}, used at {path}:{line}")


# Python


def _docstring(node):
    body = getattr(node, "body", None)
    if body and isinstance(body[0], ast.Expr) \
            and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[0]
    return None


def python_syntax(report, path, source, tree):
    """Count the size and style metrics of one Python file."""
    lines = source.splitlines()
    docstrings, docstring_rows = set(), set()
    for node in ast.walk(tree):
        found = _docstring(node) if isinstance(node, (
            ast.Module, ast.ClassDef, ast.FunctionDef,
            ast.AsyncFunctionDef)) else None
        if found is not None:
            docstrings.add((found.lineno, found.col_offset))
            docstring_rows.update(range(found.lineno, found.end_lineno + 1))
    code_rows, comment_rows, sequence = set(), set(), []
    num_tokens, depth, start = 0, 0, None
    skipped = {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
               tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        # an f-string is one token, as Rust's format string is
        if token.type == tokenize.FSTRING_START:
            depth += 1
            start = start if depth > 1 else token.start
            continue
        if token.type == tokenize.FSTRING_END:
            depth -= 1
            if depth == 0:
                num_tokens += 1
                code_rows.update(range(start[0], token.end[0] + 1))
                sequence.append(("f", start, token.end))
            continue
        if depth or token.type in skipped:
            continue
        if token.type == tokenize.COMMENT:
            comment_rows.add(token.start[0])
            if DIVIDER.search(token.string.lstrip("#")):
                report.add("divider_comments", "python", path, token.start[0])
            continue
        if token.type == tokenize.STRING and token.start in docstrings:
            continue
        num_tokens += 1
        code_rows.update(range(token.start[0], token.end[0] + 1))
        sequence.append((token.string, token.start, token.end))
    report.add("files", "python", path, listed=False)
    report.add("code_tokens", "python", path, amount=num_tokens, listed=False)
    report.add("code_lines", "python", path, amount=len(code_rows),
               listed=False)
    report.add("comment_lines", "python", path,
               amount=len(comment_rows - code_rows), listed=False)
    report.add("docstring_lines", "python", path,
               amount=len(docstring_rows - code_rows), listed=False)
    for row, line in enumerate(lines, 1):
        if len(line) > 100:
            report.add("lines_over_100", "python", path, row)
        if len(line) > 80:
            report.add("lines_over_80", "python", path, row, listed=False)
    # a continuation line aligned to a bracket that has code after it
    brackets, last_row = [], 0
    for position, (string, (row, column), end) in enumerate(sequence):
        if row > last_row and brackets and string not in (")", "]", "}"):
            open_row, open_column, hanging = brackets[-1]
            if not hanging and column == open_column + 1:
                report.add("aligned_continuations", "python", path, row)
        if string in ("(", "[", "{"):
            following = sequence[position + 1][1][0] \
                if position + 1 < len(sequence) else row
            brackets.append((row, column, following > row))
        elif string in (")", "]", "}") and brackets:
            brackets.pop()
        last_row = end[0]
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While)) \
                and node.body[0].lineno == node.lineno:
            report.add("one_line_blocks", "python", path, node.lineno)
        if isinstance(node, ast.If) and len(node.body) == 1 \
                and isinstance(node.body[0], ast.Raise):
            report.add("error_guards", "python", path, node.lineno)
        if isinstance(node, ast.Raise):
            report.add("error_sites", "python", path, node.lineno,
                       listed=False)
        if isinstance(node, ast.ExceptHandler):
            report.add("error_handlers", "python", path, node.lineno)
        if isinstance(node, ast.Assert):
            report.add("asserts", "python", path, listed=False)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and (node.func.id == "hasattr" or (
                    node.func.id == "getattr" and len(node.args) == 3)):
            report.add("guard_lookups", "python", path, node.lineno)
    python_nesting(report, path, tree, lines, code_rows)


def python_nesting(report, path, tree, lines, code_rows):
    """Count each function's lines by the blocks around them."""
    def visit(statements, depth, depths):
        for statement in statements:
            for row in range(statement.lineno, statement.end_lineno + 1):
                depths[row] = depth
            if isinstance(statement, ast.If):
                visit(statement.body, depth + 1, depths)
                elif_ = (len(statement.orelse) == 1
                         and isinstance(statement.orelse[0], ast.If)
                         and lines[statement.orelse[0].lineno - 1]
                         .lstrip().startswith("elif"))
                visit(statement.orelse, depth if elif_ else depth + 1, depths)
            elif isinstance(statement, ast.Match):
                for case in statement.cases:
                    visit(case.body, depth + 2, depths)
            elif isinstance(statement, (ast.Try, ast.TryStar)):
                for body in (statement.body, statement.orelse,
                             statement.finalbody,
                             *(handler.body for handler in statement.handlers)):
                    visit(body, depth + 1, depths)
            else:
                for field in ("body", "orelse"):
                    body = getattr(statement, field, None)
                    if isinstance(body, list):
                        visit(body, depth + 1, depths)

    def functions(node, inside):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and not inside:
                yield child
                yield from functions(child, True)
            else:
                yield from functions(child, inside)

    for function in functions(tree, False):
        depths = {}
        visit(function.body, 0, depths)
        rows = [depths[row] for row in sorted(depths) if row in code_rows]
        for depth in rows:
            if depth > 2:
                report.add("depth_excess", "python", path, amount=depth - 2,
                           listed=False)
        deepest = max(rows, default=0)
        report.add("max_depth", "python", path, amount=deepest, listed=False)
        if deepest > 3:
            report.add("deep_functions", "python", path, function.lineno,
                       function.name, f"depth {deepest}")


def _parameters(node):
    arguments = node.args
    return [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
            *([arguments.vararg] if arguments.vararg else []),
            *([arguments.kwarg] if arguments.kwarg else [])]


def _is_path(node, parameters):
    """Whether the node reads attributes off a parameter, two or more deep."""
    hops = 0
    while isinstance(node, ast.Attribute):
        node, hops = node.value, hops + 1
    return hops >= 2 and isinstance(node, ast.Name) and node.id in parameters


def _forwards(value, parameters):
    if _is_path(value, parameters):
        return True
    if not isinstance(value, ast.Call):
        return False
    arguments = [*value.args, *(keyword.value for keyword in value.keywords)]
    if not arguments:
        return isinstance(value.func, ast.Attribute) and (
            _is_path(value.func, parameters) or (
                isinstance(value.func.value, ast.Name)
                and value.func.value.id in parameters))
    return all(isinstance(argument, ast.Name) and argument.id in parameters
               for argument in (argument.value
                                if isinstance(argument, ast.Starred)
                                else argument for argument in arguments))


def _decorator_name(decorator):
    node = decorator.func if isinstance(decorator, ast.Call) else decorator
    return node.attr if isinstance(node, ast.Attribute) else getattr(
        node, "id", "")


def _abstract(node):
    body = node.body[1:] if _docstring(node) else node.body
    return not body or all(
        isinstance(statement, ast.Pass)
        or (isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant))
        or (isinstance(statement, ast.Raise) and "NotImplementedError"
            in ast.unparse(statement))
        for statement in body)


class Scope:
    """The names one module, class, or function body binds."""

    def __init__(self, kind, parent, module, key):
        self.kind, self.parent, self.module, self.key = kind, parent, module, key
        self.bindings = defaultdict(list)


class PythonIndex:
    """Every Python definition, with each reference resolved or by name."""

    def __init__(self):
        self.functions = {}
        self.by_name = defaultdict(list)
        self.classes = {}
        self.class_names = defaultdict(list)
        self.resolving = set()
        self.modules = {}
        self.scopes = {}
        self.trees = {}

    def load(self):
        for path, category in source_files(".py"):
            source = (ROOT / path).read_text()
            tree = ast.parse(source, filename=path)
            parts = path[:-3].split("/")
            if parts[-1] == "__init__":
                parts = parts[:-1]
            names = [".".join(parts)]
            if parts[0] == "tests" and len(parts) > 1:
                names.append(".".join(parts[1:]))
            for name in names:
                self.modules[name] = tree
            self.trees[path] = (tree, category, names[0], source)
            self._bind(tree, Scope("module", None, names[0], names[0]),
                       path, category)
        self.packages = {name.split(".")[0] for name in self.modules}

    def _bind(self, node, scope, path, category):
        """Bind each name a body defines, then the bodies inside it."""
        self.scopes[node] = scope
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.Lambda)):
            for parameter in _parameters(node):
                scope.bindings[parameter.arg].append(("local",))
            parameters = _parameters(node)
            decorators = {_decorator_name(decorator)
                          for decorator in getattr(node, "decorator_list", ())}
            if scope.parent.kind == "class" and parameters \
                    and "staticmethod" not in decorators:
                kind = "class" if "classmethod" in decorators else "instance"
                scope.bindings[parameters[0].arg] = [(kind, scope.parent.key)]
        pending = list(ast.iter_child_nodes(node))
        while pending:
            child = pending.pop()
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                for decorator in child.decorator_list:
                    pending.append(decorator)
                key = f"{scope.key}.{child.name}"
                if isinstance(child, ast.ClassDef):
                    scope.bindings[child.name].append(("class", key))
                    self.classes[key] = {"node": child, "scope": scope,
                                         "methods": {}, "path": path}
                    self.class_names[child.name].append(key)
                    pending.extend(child.bases)
                    self._bind(child, Scope("class", scope, scope.module, key),
                               path, category)
                else:
                    scope.bindings[child.name].append(("def", key))
                    self._define(child, key, scope, path, category)
                    pending.extend(child.args.defaults)
                    pending.extend(default for default in child.args.kw_defaults
                                   if default is not None)
                    self._bind(child, Scope("function", scope, scope.module,
                                            key), path, category)
                continue
            if isinstance(child, ast.Lambda):
                self._bind(child, Scope("function", scope, scope.module,
                                        f"{scope.key}.<lambda>"), path,
                           category)
                continue
            if isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp,
                                  ast.GeneratorExp)):
                pending.extend(ast.iter_child_nodes(child))
                continue
            if isinstance(child, ast.Import):
                for alias in child.names:
                    if alias.asname:
                        scope.bindings[alias.asname].append(
                            ("module", alias.name))
                    else:
                        head = alias.name.split(".")[0]
                        scope.bindings[head].append(("module", head))
            elif isinstance(child, ast.ImportFrom):
                base = child.module or ""
                if child.level:
                    package = scope.module.split(".")
                    if not path.endswith("__init__.py"):
                        package = package[:-1]
                    package = package[:len(package) - child.level + 1]
                    base = ".".join([*package, *([base] if base else [])])
                for alias in child.names:
                    scope.bindings[alias.asname or alias.name].append(
                        ("symbol", base, alias.name))
            elif isinstance(child, ast.Assign) and len(child.targets) == 1 \
                    and isinstance(child.targets[0], ast.Name) \
                    and isinstance(child.value, ast.Call):
                # x = Class(...) makes x an instance, if Class is ours
                scope.bindings[child.targets[0].id].append(
                    ("assigned", child.value, scope))
                pending.append(child.value)
                continue
            elif isinstance(child, ast.Name) and isinstance(
                    child.ctx, (ast.Store, ast.Del)):
                scope.bindings[child.id].append(("local",))
            elif isinstance(child, (ast.Global, ast.Nonlocal)):
                for name in child.names:
                    scope.bindings[name].append(("outer",))
            elif isinstance(child, ast.ExceptHandler) and child.name:
                scope.bindings[child.name].append(("local",))
            pending.extend(ast.iter_child_nodes(child))

    def _define(self, node, key, scope, path, category):
        function = Function("python", key, node.name, path, node.lineno,
                            node.end_lineno)
        function.node, function.category = node, category
        decorators = {_decorator_name(decorator)
                      for decorator in node.decorator_list}
        function.decorators = decorators
        if node.name.startswith("__") and node.name.endswith("__"):
            function.excluded = "dunder"
        elif decorators & {"jit", "autotune"}:
            # a GPU kernel cannot be inlined into its Python caller
            function.excluded = "kernel"
        elif decorators - TRANSPARENT_DECORATORS:
            function.excluded = "decorator " + ", ".join(
                sorted(decorators - TRANSPARENT_DECORATORS))
        elif _abstract(node):
            function.excluded = "abstract"
        if scope.kind == "class":
            self.classes[scope.key]["methods"][node.name] = key
        function.owner = scope.key if scope.kind == "class" else None
        self.functions[key] = function
        self.by_name[node.name].append(function)

    def symbol(self, module, name, seen=0):
        """Resolve a module's name to a binding, following re-exports."""
        if seen > 20:
            return None
        tree = self.modules.get(module)
        if tree is None:
            return None
        found = self.scopes[tree].bindings.get(name, [])
        if len(found) == 1:
            binding = found[0]
            if binding[0] == "symbol":
                return self.symbol(binding[1], binding[2], seen + 1) or (
                    ("module", f"{binding[1]}.{binding[2]}")
                    if f"{binding[1]}.{binding[2]}" in self.modules else None)
            if binding[0] == "module":
                return binding
            if binding[0] in ("def", "class"):
                return binding
            return None
        if not found and f"{module}.{name}" in self.modules:
            return ("module", f"{module}.{name}")
        return ("many", found) if found else None

    def lookup(self, scope, name):
        """Resolve a bare name read in a scope, by Python's scoping rules."""
        first = True
        while scope is not None:
            if scope.kind != "class" or first:
                found = scope.bindings.get(name)
                if found:
                    if found == [("outer",)]:
                        scope, first = scope.parent, False
                        continue
                    return self.settle(found)
            scope, first = scope.parent, False
        return None

    def settle(self, found):
        if len(found) > 1:
            return ("many", found)
        binding = found[0]
        if binding[0] in ("symbol", "module") and binding[1].split(".")[0] \
                not in self.packages:
            # numpy.zeros can only be numpy's
            return ("external",)
        if binding[0] == "symbol":
            return self.symbol(binding[1], binding[2]) or (
                ("module", f"{binding[1]}.{binding[2]}")
                if f"{binding[1]}.{binding[2]}" in self.modules else None)
        if binding[0] == "module":
            return binding if binding[1] in self.modules else None
        if binding[0] == "assigned":
            # x = x(...) would resolve itself forever
            if id(binding[1]) in self.resolving:
                return ("local",)
            self.resolving.add(id(binding[1]))
            found = self.resolve(binding[1], binding[2])
            self.resolving.discard(id(binding[1]))
            return found if found and found[0] == "instance" else ("local",)
        return binding

    def method(self, class_key, name, seen=None):
        """Find a method in a class or its project bases; None when absent."""
        seen = seen or set()
        if class_key in seen or class_key not in self.classes:
            return None
        seen.add(class_key)
        found = self.classes[class_key]["methods"].get(name)
        if found:
            return found
        for base in self.bases(class_key):
            if base is not None:
                found = self.method(base, name, seen)
                if found:
                    return found
        return None

    def bases(self, class_key):
        """Each base of a class: its key, or None for an outside class."""
        info = self.classes[class_key]
        out = []
        for base in info["node"].bases:
            binding = self.resolve(base, info["scope"])
            out.append(binding[1] if binding and binding[0] == "class"
                       else None)
        return out

    def resolve(self, node, scope):
        """Resolve an expression to a binding, or None when unknown."""
        if isinstance(node, ast.Name):
            return self.lookup(scope, node.id)
        if isinstance(node, ast.Attribute):
            base = self.resolve(node.value, scope)
            if base is None or base[0] == "external":
                return base
            if base[0] == "module":
                return self.symbol(base[1], node.attr)
            if base[0] in ("class", "instance"):
                key = self.method(base[1], node.attr)
                return ("def", key) if key else None
            return None
        if isinstance(node, ast.Call):
            binding = self.resolve(node.func, scope)
            if binding and binding[0] == "class":
                return ("instance", binding[1])
        return None

    def subclasses(self):
        children = defaultdict(set)
        for key in self.classes:
            for base in self.bases(key):
                if base is not None:
                    children[base].add(key)
        return children

    def reference(self, function, category, path, line, resolved, call=None):
        function.references.append((category, path, line, resolved, call))

    def by_text(self, name, category, path, line):
        for function in self.by_name.get(name, ()):
            self.reference(function, category, path, line, False)

    def link(self):
        """Resolve every name, attribute, and string in every file."""
        children = self.subclasses()

        def overrides(class_key, name):
            pending, out = list(children.get(class_key, ())), []
            while pending:
                key = pending.pop()
                if name in self.classes[key]["methods"]:
                    out.append(self.classes[key]["methods"][name])
                pending.extend(children.get(key, ()))
            return out

        escaped = set()
        for path, (tree, category, module, _) in self.trees.items():
            category = "test" if category == "test" else "source"
            parents = {}
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    parents[child] = node

            def enclosing(node):
                # the scope a node is read in: decorators, defaults, and
                # bases are read in the scope around their definition
                child, node = node, parents.get(node)
                while node is not None:
                    if node in self.scopes:
                        if isinstance(node, (ast.FunctionDef,
                                             ast.AsyncFunctionDef)) and (
                                child in node.decorator_list
                                or child in node.args.defaults
                                or child in node.args.kw_defaults
                                or child is node.returns
                                or child is node.args):
                            child, node = node, parents.get(node)
                            continue
                        if isinstance(node, ast.ClassDef) and (
                                child in node.bases
                                or child in node.decorator_list
                                or child in node.keywords):
                            child, node = node, parents.get(node)
                            continue
                        return self.scopes[node]
                    child, node = node, parents.get(node)
                return self.scopes[tree]

            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                        and node.value.isidentifier():
                    self.by_text(node.value, category, path, node.lineno)
                    escaped.update(self.class_names.get(node.value, ()))
                    continue
                if category != "test":
                    escaped.update(self._escapes(node, enclosing))
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    binding = self.lookup(enclosing(node), node.id)
                    attribute = None
                elif isinstance(node, ast.Attribute) and isinstance(
                        node.ctx, ast.Load):
                    attribute = node.attr
                    base = self.resolve(node.value, enclosing(node))
                    binding = None
                    if base is not None and base[0] == "module":
                        binding = self.symbol(base[1], node.attr)
                    elif base is not None and base[0] in ("class", "instance"):
                        key = self.method(base[1], node.attr)
                        if key is not None:
                            binding = ("def", key)
                            for other in overrides(base[1], node.attr):
                                self.reference(self.functions[other], category,
                                               path, node.lineno, False)
                        base = ("receiver", base[0], base[1])
                    if binding is None:
                        if base is not None and base[0] in ("module",
                                                            "external"):
                            continue
                        self.by_text(node.attr, category, path, node.lineno)
                        continue
                else:
                    continue
                if binding is None:
                    continue
                parent = parents.get(node)
                call = parent if isinstance(parent, ast.Call) \
                    and parent.func is node else None
                if binding[0] == "many":
                    for found in binding[1]:
                        if found[0] == "def":
                            self.reference(self.functions[found[1]], category,
                                           path, node.lineno, False)
                elif binding[0] == "def":
                    function = self.functions[binding[1]]
                    offset = None
                    if call is not None:
                        offset = self._offset(function, node, attribute,
                                              enclosing(node))
                    self.reference(function, category, path, node.lineno,
                                   True, (call, offset) if call is not None
                                   else None)
                elif binding[0] == "class" and call is not None:
                    init = self.classes[binding[1]]["methods"].get("__init__")
                    if init:
                        self.reference(self.functions[init], category, path,
                                       node.lineno, True, (call, 1))
        self._mark_public(escaped)

    def _escapes(self, node, enclosing):
        """Our classes a node hands to outside code, which calls by name."""
        if isinstance(node, ast.Call):
            outside = self.resolve(node.func, enclosing(node))
            values = [*node.args, *(keyword.value for keyword in node.keywords)]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value:
            targets = node.targets if isinstance(node, ast.Assign) \
                else [node.target]
            outside = next((self.resolve(target.value, enclosing(node))
                            for target in targets
                            if isinstance(target, ast.Attribute)), None)
            values = [node.value]
        else:
            return []
        if outside is None or outside[0] != "external":
            return []
        out = []
        for value in values:
            found = self.resolve(value, enclosing(node))
            if found and found[0] in ("class", "instance"):
                out.append(found[1])
        return out

    def _offset(self, function, node, attribute, scope):
        """How many leading parameters a call skips: self or cls."""
        if function.owner is None or "staticmethod" in function.decorators:
            return 0
        if attribute is None:
            return None
        base = self.resolve(node.value, scope)
        if base is None:
            return None
        if base[0] == "instance" or "classmethod" in function.decorators:
            return 1
        return 0

    def _mark_public(self, escaped):
        """Exclude what outside code may call: exports, framework methods."""
        for key in sorted(escaped):
            for method in self._class_methods(key):
                function = self.functions[method]
                if not function.name.startswith("_") and not function.excluded:
                    function.excluded = "reaches outside code"
        exported = set()
        tree = self.modules.get("quail")
        for node in tree.body if tree else ():
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets):
                exported = {element.value for element in node.value.elts}
        for name in exported:
            binding = self.symbol("quail", name)
            if binding and binding[0] == "def":
                self.functions[binding[1]].excluded = "public"
            if binding and binding[0] == "class":
                for key in self._class_methods(binding[1]):
                    function = self.functions[key]
                    if not function.name.startswith("_") \
                            and not function.excluded:
                        function.excluded = "public"
        for key, info in self.classes.items():
            outside = [ast.unparse(base) for base, found in
                       zip(info["node"].bases, self.bases(key))
                       if found is None and ast.unparse(base).split(".")[-1]
                       .split("[")[0] not in INERT_BASES]
            if not outside:
                continue
            for name, method in info["methods"].items():
                function = self.functions[method]
                if not name.startswith("_") and not function.excluded:
                    function.excluded = "outside base " + outside[0]

    def _class_methods(self, class_key, seen=None):
        seen = seen or set()
        if class_key in seen or class_key not in self.classes:
            return []
        seen.add(class_key)
        out = list(self.classes[class_key]["methods"].values())
        for base in self.bases(class_key):
            if base is not None:
                out.extend(self._class_methods(base, seen))
        return out

    def record(self, report):
        for key in sorted(self.functions):
            function = self.functions[key]
            if function.category != "source":
                continue
            function.record(report)
            node = function.node
            parameters = {parameter.arg for parameter in _parameters(node)}
            body = node.body[1:] if _docstring(node) else node.body
            if not function.excluded and len(body) == 1 \
                    and isinstance(body[0], (ast.Return, ast.Expr)) \
                    and body[0].value is not None \
                    and _forwards(body[0].value, parameters):
                report.add("pass_through_functions", "python", function.path,
                           function.line, function.name)
            if function.excluded and function.name != "__init__":
                continue
            self._arguments(report, function)

    def _arguments(self, report, function):
        """Count the parameters no call passes, or every call passes alike."""
        calls = [reference[4] for reference in function.references]
        if not calls or any(call is None or call[1] is None
                            for call in calls):
            return
        node = function.node.args
        positional = [*node.posonlyargs, *node.args]
        skipped = 1 if function.owner is not None and \
            "staticmethod" not in function.decorators else 0
        positional = positional[skipped:]
        defaults = {parameter.arg for parameter in
                    positional[len(positional) - len(node.defaults):]} \
            if node.defaults else set()
        defaults |= {parameter.arg for parameter, default in
                     zip(node.kwonlyargs, node.kw_defaults)
                     if default is not None}
        names = [parameter.arg for parameter in [*positional, *node.kwonlyargs]]
        passed = defaultdict(list)
        for call, offset in calls:
            values = {}
            arguments = call.args
            if offset == 0 and skipped:
                arguments = arguments[1:]
            if any(isinstance(argument, ast.Starred) for argument in arguments) \
                    or any(keyword.arg is None for keyword in call.keywords):
                return
            for parameter, argument in zip(positional, arguments):
                values[parameter.arg] = argument
            for keyword in call.keywords:
                values[keyword.arg] = keyword.value
            for name in names:
                passed[name].append(values.get(name))
        for name in names:
            given = passed[name]
            if name in defaults and all(value is None for value in given):
                report.add("unused_parameters", "python", function.path,
                           function.line, f"{function.name}({name})")
            elif len(given) >= 2 and all(isinstance(value, ast.Constant)
                                         for value in given) \
                    and len({repr(value.value) for value in given}) == 1:
                report.add("constant_arguments", "python", function.path,
                           function.line, f"{function.name}({name})",
                           f"always {given[0].value!r}")


def measure_python(report):
    """Count every Python metric over quail/."""
    index = PythonIndex()
    index.load()
    for path, (tree, category, _, source) in index.trees.items():
        if category == "source":
            python_syntax(report, path, source, tree)
    # Rust calls Python methods by name: call_method1("name", ...)
    for path, category in source_files(".rs"):
        text = (ROOT / path).read_text()
        for match in re.finditer(r'"([A-Za-z_][A-Za-z0-9_]*)"', text):
            index.by_text(match.group(1), "test" if category == "test"
                          else "source", path,
                          text.count("\n", 0, match.start()) + 1)
    # entry points: "package.module:function" in any TOML file
    for path, _ in source_files(".toml"):
        text = (ROOT / path).read_text()
        for match in re.finditer(r'"[\w.]+:(\w+)"', text):
            index.by_text(match.group(1), "source", path,
                          text.count("\n", 0, match.start()) + 1)
    index.link()
    index.record(report)


# Rust

ATOMIC = {"string_literal", "raw_string_literal", "char_literal",
          "line_comment", "block_comment"}
ERROR_HANDLERS = {"map_err", "ok_or", "ok_or_else", "context", "with_context"}
ASSERT_MACROS = {"assert", "assert_eq", "assert_ne", "debug_assert",
                 "debug_assert_eq", "debug_assert_ne", "panic", "unreachable",
                 "todo", "unimplemented"}
PYO3 = re.compile(r"#\[(pyfunction|pymethods|pymodule|pyclass|getter|setter|"
                  r"new|staticmethod|classmethod|pyo3)\b")
TEST_ATTRIBUTE = re.compile(r"#\[(test|\w+::test|cfg\(.*\btest\b.*\))\]")


def _parser():
    import tree_sitter
    import tree_sitter_rust
    return tree_sitter.Parser(tree_sitter.Language(tree_sitter_rust.language()))


def _walk(node):
    pending = [node]
    while pending:
        node = pending.pop()
        yield node
        if node.type not in ATOMIC:
            pending.extend(reversed(node.children))


def _attributes(node):
    out, sibling = [], node.prev_sibling
    while sibling is not None and sibling.type in (
            "attribute_item", "line_comment", "block_comment"):
        if sibling.type == "attribute_item":
            out.append(sibling)
        sibling = sibling.prev_sibling
    return out


def _text(node):
    return node.text.decode()


def _error_return(node):
    """Whether a statement only returns an error."""
    if node.type == "expression_statement" and node.named_child_count == 1:
        node = node.named_children[0]
    if node.type == "return_expression" and node.named_child_count == 1:
        node = node.named_children[0]
    if node.type == "call_expression":
        name = _text(node.child_by_field_name("function"))
        return name in ("Err", "Result::Err")
    if node.type == "macro_invocation":
        name = _text(node.child_by_field_name("macro"))
        return name in ("bail", "ensure") or name.endswith("_err")
    return False


def _named(block):
    return [child for child in block.named_children
            if child.type not in ("line_comment", "block_comment")]


class RustFile:
    """One parsed Rust file: its test rows, import rows, and functions."""

    def __init__(self, parser, path, category):
        self.path, self.category = path, category
        self.source = (ROOT / path).read_bytes()
        self.tree = parser.parse(self.source)
        self.test_rows, self.use_rows = set(), set()
        for node in _walk(self.tree.root_node):
            if node.type == "use_declaration":
                self.use_rows.update(range(node.start_point.row,
                                           node.end_point.row + 1))
            if not node.type.endswith("_item") \
                    or node.type == "attribute_item":
                continue
            attributes = _attributes(node)
            if any(TEST_ATTRIBUTE.fullmatch(_text(attribute))
                   and "not(" not in _text(attribute)
                   for attribute in attributes):
                self.test_rows.update(range(attributes[-1].start_point.row,
                                            node.end_point.row + 1))

    def is_test(self, row):
        return self.category == "test" or row in self.test_rows


def rust_syntax(report, file):
    """Count the size, error, nesting, and style metrics of one file."""
    path = file.path
    code_rows, comment_rows, docstring_rows = set(), set(), set()
    num_tokens = 0
    for node in _walk(file.tree.root_node):
        row = node.start_point.row
        if file.is_test(row):
            continue
        if node.type in ("line_comment", "block_comment"):
            text = _text(node)
            rows = range(row, node.end_point.row + 1)
            if text.startswith(("///", "//!", "/**", "/*!")):
                docstring_rows.update(rows)
            else:
                comment_rows.update(rows)
                if DIVIDER.search(text.lstrip("/*!")):
                    report.add("divider_comments", "rust", path, row + 1)
            continue
        if node.child_count == 0 or node.type in ATOMIC:
            if node.end_byte > node.start_byte:
                num_tokens += 1
                code_rows.update(range(row, node.end_point.row + 1))
            continue
        parent = node.parent
        statement = parent is not None and (
            parent.type == "expression_statement" or (
                parent.type == "block" and node.type.endswith("expression")))
        if node.type in ("if_expression", "for_expression", "while_expression",
                         "loop_expression") and statement \
                and row == node.end_point.row:
            body = node.child_by_field_name("consequence") \
                or node.child_by_field_name("body")
            if body is not None and _named(body):
                report.add("one_line_blocks", "rust", path, row + 1)
        if node.type == "if_expression":
            body = _named(node.child_by_field_name("consequence"))
            if len(body) == 1 and _error_return(body[0]):
                report.add("error_guards", "rust", path, row + 1)
        if node.type == "let_declaration":
            alternative = node.child_by_field_name("alternative")
            if alternative is not None and len(_named(alternative)) == 1 \
                    and _error_return(_named(alternative)[0]):
                report.add("error_guards", "rust", path, row + 1)
        if node.type == "block" and parent is not None and (
                parent.type == "block" or (
                    parent.type == "expression_statement"
                    and parent.parent is not None
                    and parent.parent.type == "block")):
            report.add("bare_blocks", "rust", path, row + 1)
        if node.type == "try_expression":
            report.add("error_propagations", "rust", path, row + 1,
                       listed=False)
        if node.type == "call_expression":
            function = node.child_by_field_name("function")
            if _text(function) in ("Err", "Result::Err"):
                report.add("error_sites", "rust", path, row + 1, listed=False)
            if function.type == "field_expression":
                field = _text(function.child_by_field_name("field"))
                if field in ERROR_HANDLERS:
                    report.add("error_handlers", "rust", path, row + 1)
                if field in ("expect", "unwrap"):
                    report.add("asserts", "rust", path, listed=False)
        if node.type == "macro_invocation":
            name = _text(node.child_by_field_name("macro")).split("::")[-1]
            if name in ("bail", "ensure") or name.endswith("_err"):
                report.add("error_sites", "rust", path, row + 1, listed=False)
            if name in ASSERT_MACROS:
                report.add("asserts", "rust", path, listed=False)
    report.add("files", "rust", path, listed=False)
    report.add("code_tokens", "rust", path, amount=num_tokens, listed=False)
    report.add("code_lines", "rust", path, amount=len(code_rows), listed=False)
    report.add("comment_lines", "rust", path,
               amount=len(comment_rows - code_rows), listed=False)
    report.add("docstring_lines", "rust", path,
               amount=len(docstring_rows - code_rows), listed=False)
    for row, line in enumerate(file.source.decode().splitlines()):
        if file.is_test(row):
            continue
        if len(line) > 100:
            report.add("lines_over_100", "rust", path, row + 1)
        if len(line) > 80:
            report.add("lines_over_80", "rust", path, row + 1, listed=False)
    rust_nesting(report, file, code_rows)


def rust_functions(file):
    """Yield each function item outside test code, outermost first."""
    for node in _walk(file.tree.root_node):
        if node.type == "function_item" and not file.is_test(
                node.start_point.row):
            yield node


def rust_nesting(report, file, code_rows):
    """Count each outermost function's lines by the braces around them."""
    for function in rust_functions(file):
        ancestor = function.parent
        while ancestor is not None and ancestor.type != "function_item":
            ancestor = ancestor.parent
        if ancestor is not None:
            continue
        depths = {}
        pending = [(function.child_by_field_name("body"), -1)]
        while pending:
            node, depth = pending.pop()
            if node.type in ("block", "match_block"):
                depth += 1
            if node.child_count == 0 or node.type in ATOMIC:
                depths.setdefault(node.start_point.row, depth)
                continue
            pending.extend((child, depth) for child in reversed(node.children))
        rows = [depth for row, depth in depths.items() if row in code_rows]
        for depth in rows:
            if depth > 2:
                report.add("depth_excess", "rust", file.path,
                           amount=depth - 2, listed=False)
        deepest = max(rows, default=0)
        report.add("max_depth", "rust", file.path, amount=deepest,
                   listed=False)
        if deepest > 3:
            name = _text(function.child_by_field_name("name"))
            report.add("deep_functions", "rust", file.path,
                       function.start_point.row + 1, name, f"depth {deepest}")


def _rust_forwards(function):
    parameters = {"self"}
    for parameter in function.child_by_field_name("parameters").named_children:
        pattern = parameter.child_by_field_name("pattern")
        if pattern is not None and pattern.type == "identifier":
            parameters.add(_text(pattern))
    body = _named(function.child_by_field_name("body"))
    if len(body) != 1:
        return False
    value = body[0]
    if value.type == "expression_statement" and value.named_child_count == 1:
        value = value.named_children[0]
    while value.type == "reference_expression":
        value = value.child_by_field_name("value")

    def path(node, hops=0):
        while node.type == "field_expression":
            node, hops = node.child_by_field_name("value"), hops + 1
        return hops, node.type in ("identifier", "self") \
            and _text(node) in parameters

    if value.type == "field_expression":
        hops, rooted = path(value)
        return rooted and hops >= 2
    if value.type != "call_expression":
        return False
    arguments = _named(value.child_by_field_name("arguments"))
    callee = value.child_by_field_name("function")
    if not arguments:
        return callee.type == "field_expression" and path(callee)[1]
    for argument in arguments:
        while argument.type == "reference_expression":
            argument = argument.child_by_field_name("value")
        if argument.type not in ("identifier", "self") \
                or _text(argument) not in parameters:
            return False
    return True


def _exclusion(file, function):
    """Why the script cannot see a function's callers, or ""."""
    name = _text(function.child_by_field_name("name"))
    attributes = " ".join(_text(attribute) for attribute in
                          _attributes(function))
    container = function.parent.parent if function.parent is not None \
        and function.parent.type == "declaration_list" else None
    if container is not None and container.type == "impl_item":
        if container.child_by_field_name("trait") is not None:
            return "trait implementation"
        attributes += " " + " ".join(_text(attribute) for attribute in
                                     _attributes(container))
    if PYO3.search(attributes):
        return "pyo3 entry point"
    if name == "main" or "no_mangle" in attributes \
            or any(child.type == "function_modifiers" and "extern" in
                   _text(child) for child in function.children):
        return "entry point"
    crate = file.path.split("/")[1]
    visibility = next((child for child in function.children
                       if child.type == "visibility_modifier"), None)
    if crate in PUBLISHED_CRATES and visibility is not None \
            and _text(visibility) == "pub":
        return "public"
    return ""


def _varint(data, position):
    shift = result = 0
    while True:
        byte = data[position]
        position += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return result, position


def _fields(data):
    """Yield (field number, value) of a protobuf message."""
    position = 0
    while position < len(data):
        key, position = _varint(data, position)
        wire = key & 7
        if wire == 0:
            value, position = _varint(data, position)
        elif wire == 2:
            size, position = _varint(data, position)
            value, position = data[position:position + size], position + size
        else:
            size = 8 if wire == 1 else 4
            value, position = data[position:position + size], position + size
        yield key >> 3, value


def scip_index() -> bytes:
    """Return rust-analyzer's SCIP index of the workspace, built once."""
    inputs = hashlib.sha256()
    version = subprocess.run(["rust-analyzer", "--version"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
    inputs.update(version.stdout.encode())
    for path in sorted([*ROOT.glob("crates/**/*.rs"),
                        *ROOT.glob("crates/**/Cargo.toml"),
                        ROOT / "Cargo.toml", ROOT / "Cargo.lock",
                        ROOT / "rust-toolchain.toml"]):
        if "target" in path.relative_to(ROOT).parts or not path.exists():
            continue
        inputs.update(str(path.relative_to(ROOT)).encode())
        inputs.update(path.read_bytes())
    cached = ROOT / "target" / "skill-issues" / f"{inputs.hexdigest()}.scip"
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        print("indexing Rust with rust-analyzer (about 3 minutes)",
              file=sys.stderr)
        partial = cached.with_suffix(".partial")
        subprocess.run(["rust-analyzer", "scip", ".", "--output",
                        str(partial)], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        partial.rename(cached)
    return cached.read_bytes()


def measure_rust(report, references=True):
    """Count every Rust metric over crates/*/src."""
    parser = _parser()
    files = {path: RustFile(parser, path, category)
             for path, category in source_files(".rs")
             if path.startswith("crates/")}
    for file in files.values():
        if file.category == "source":
            rust_syntax(report, file)
    functions = []
    for file in files.values():
        if file.category != "source":
            continue
        for node in rust_functions(file):
            name = node.child_by_field_name("name")
            function = Function("rust", "", _text(name), file.path,
                                node.start_point.row + 1,
                                node.end_point.row + 1)
            function.excluded = _exclusion(file, node)
            function.row = name.start_point.row
            functions.append(function)
            if not function.excluded and _rust_forwards(node):
                report.add("pass_through_functions", "rust", file.path,
                           function.line, function.name)
    if not references:
        for function in functions:
            report.add("functions", "rust", function.path, listed=False)
        return
    definitions = defaultdict(list)
    occurrences = defaultdict(list)
    for number, document in _fields(scip_index()):
        if number != 2:
            continue
        path, found = "", []
        for field, value in _fields(document):
            if field == 1:
                path = value.decode()
            elif field == 2:
                occurrence = dict(_fields(value))
                position, start = 0, []
                while position < len(occurrence.get(1, b"")):
                    value, position = _varint(occurrence[1], position)
                    start.append(value)
                found.append((occurrence.get(2, b"").decode(),
                              occurrence.get(3, 0), start[0]))
        for symbol, roles, row in found:
            if roles & 1:
                definitions[symbol].append((path, row))
            else:
                occurrences[symbol].append((path, row))
    # names rust-analyzer does not resolve: a function named in an
    # attribute's string, as serde(default = "name"), or in a macro_rules body
    named = defaultdict(list)
    for file in files.values():
        for node in _walk(file.tree.root_node):
            if node.type == "attribute_item":
                for match in re.finditer(r'"(?:[\w:]*::)?(\w+)"', _text(node)):
                    named[match.group(1)].append((file, node.start_point.row))
            elif node.type == "macro_definition":
                for inner in _walk(node):
                    if inner.type == "identifier":
                        named[_text(inner)].append((file, inner.start_point.row))
    by_place = defaultdict(list)
    for symbol, places in definitions.items():
        for place in places:
            by_place[place].append(symbol)
    for function in functions:
        if not function.excluded:
            symbols = [symbol for symbol in
                       by_place.get((function.path, function.row), ())
                       if symbol.endswith(f"{function.name}().")]
            if len(symbols) != 1:
                function.excluded = "not indexed"
            elif len(definitions[symbols[0]]) > 1:
                function.excluded = "shares a symbol"
            else:
                for path, row in sorted(occurrences[symbols[0]]):
                    file = files.get(path)
                    if file is None or row in file.use_rows:
                        continue
                    category = "test" if file.is_test(row) else "source"
                    function.references.append(
                        (category, path, row + 1, True, None))
            for file, row in named.get(function.name, ()):
                category = "test" if file.is_test(row) else "source"
                function.references.append(
                    (category, file.path, row + 1, False, None))
        function.record(report)


def measure(references=True) -> dict:
    report = Report()
    measure_python(report)
    measure_rust(report, references)
    return report.to_dict()


def print_table(now, base=None, under=""):
    current = totals(now, under)
    before = totals(base, under) if base else None
    header = f"{'metric':24} {'step':10}"
    for language in (*LANGUAGES, "total"):
        header += f" {language:>16}" if before else f" {language:>9}"
    print(header)
    worse = []
    for metric, (step, _) in METRICS.items():
        row = f"{metric:24} {step:10}"
        for language in (*LANGUAGES, "total"):
            def value(source, metric=metric, language=language):
                if language != "total":
                    return source[metric][language]
                combine = max if metric == "max_depth" else sum
                return combine(source[metric][name] for name in LANGUAGES)
            if before is None:
                row += f" {value(current):>9}"
                continue
            delta = value(current) - value(before)
            row += f" {value(current):>9} {delta:>+6}" if delta \
                else f" {value(current):>9} {'':>6}"
            if language != "total" and delta > 0 and step != "info":
                worse.append(f"{metric} ({language})")
        print(row)
    return worse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--json", help="write the full measurement here")
    parser.add_argument("--compare", help="a --json file to compare with; "
                        "exits 1 when a metric got worse")
    parser.add_argument("--list", metavar="METRIC", choices=sorted(METRICS),
                        help="print the items behind a metric")
    parser.add_argument("--under", default="",
                        help="count only the files under this path prefix")
    parser.add_argument("--no-references", action="store_true",
                        help="skip the Rust index; Rust reference metrics "
                        "read 0")
    arguments = parser.parse_args(argv)
    data = measure(references=not arguments.no_references)
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(data, indent=1,
                                                   sort_keys=True) + "\n")
    if arguments.list:
        for item in data["items"]:
            if item["metric"] == arguments.list \
                    and item["path"].startswith(arguments.under):
                print(f"{item['path']}:{item['line']} {item['name']} "
                      f"{item['detail']}".rstrip())
        return 0
    base = json.loads(Path(arguments.compare).read_text()) \
        if arguments.compare else None
    worse = print_table(data, base, arguments.under)
    if worse:
        print("worse: " + ", ".join(worse), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
