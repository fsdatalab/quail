"""Tests for tools/check_comments.py."""

from __future__ import annotations

import textwrap

from tools import check_comments


def test_python_checker_flags_comments_and_keeps_docstrings_and_pragmas(tmp_path):
    file = tmp_path / "sample.py"
    source_text = textwrap.dedent("""\
        #!/usr/bin/env python3
        # SPDX-License-Identifier: MIT
        \"\"\"Module docstring stays.\"\"\"


        def compute(value: int) -> int:
            \"\"\"Return value plus one.\"\"\"
            # inline comment fails
            result = value + 1  # noqa: F841
            return result  # trailing comment fails
        """)
    file.write_text(source_text)
    findings = check_comments.find_comments([str(file)])
    assert findings == [
        f"{file}:8: # inline comment fails",
        f"{file}:10: # trailing comment fails",
    ]


def test_rust_checker_flags_comments_and_ignores_strings(tmp_path):
    file = tmp_path / "sample.rs"
    source_text = textwrap.dedent("""\
        // SPDX-License-Identifier: MIT
        //! module doc fails
        /// item doc fails
        fn main() {
            let url = "https://example.com /* not a comment */";
            let raw = r#"// still a string"#;
            // line comment fails
            /* block comment fails */
            let _ = (url, raw);
        }
        """)
    file.write_text(source_text)
    findings = check_comments.find_comments([str(file)])
    assert findings == [
        f"{file}:2: //! module doc fails",
        f"{file}:3: /// item doc fails",
        f"{file}:7: // line comment fails",
        f"{file}:8: /* block comment fails */",
    ]


def test_main_returns_zero_on_clean_tree_and_one_on_comment(tmp_path):
    clean = tmp_path / "clean.py"
    clean.write_text("\"\"\"Clean module.\"\"\"\n\nx = 1\n")
    assert check_comments.main([str(clean)]) == 0
    dirty = tmp_path / "dirty.rs"
    dirty.write_text("fn f() {\n    // comment\n}\n")
    assert check_comments.main([str(dirty)]) == 1
