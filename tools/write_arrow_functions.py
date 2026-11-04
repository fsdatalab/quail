"""Write crates/quail-planner/src/arrow_functions.rs from pyarrow's function list.

Run from the repository root after changing the pyarrow version:

    uv run python tools/write_arrow_functions.py
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

PATH = Path("crates/quail-planner/src/arrow_functions.rs")
HEADER_TEXT = (
    "//! The functions of Arrow compute, as pyarrow {version} lists them.\n//!\n"
    "//! `tools/write_arrow_functions.py` writes this file; a test in\n"
    "//! `tests/planner/test_logical_builder.py` checks it against pyarrow.\n\n"
    "/// (name, kind, whether `pyarrow.compute` has an attribute of that name).\n"
    "pub const ARROW_FUNCTIONS: &[(&str, &str, bool)] = &[\n")


def arrow_functions_text() -> str:
    """Return the Rust file listing every Arrow compute function."""
    rows = []
    for name in sorted(pc.list_functions()):
        kind = pc.get_function(name).kind
        attribute = "true" if hasattr(pc, name) else "false"
        rows.append(f'    ("{name}", "{kind}", {attribute}),')
    head = HEADER_TEXT.format(version=pa.__version__)
    return head + "\n".join(rows) + "\n];\n"


def main() -> None:
    """Write the file."""
    PATH.write_text(arrow_functions_text())


if __name__ == "__main__":
    main()
