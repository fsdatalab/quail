import shutil
import textwrap

import pytest

from tools import skill_issues


def _measure(tmp_path, monkeypatch, files, references=False):
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(textwrap.dedent(text))
    monkeypatch.setattr(skill_issues, "ROOT", tmp_path)
    return skill_issues.measure(references=references)


def _items(data, metric):
    return [f"{item['path']}:{item['line']} {item['name']} {item['detail']}"
            .rstrip() for item in data["items"] if item["metric"] == metric]


def _total(data, metric, language="python"):
    return skill_issues.totals(data)[metric][language]


def test_a_helper_called_once_is_single_use_and_one_called_twice_is_not(
        tmp_path, monkeypatch):
    source_text = {"quail/a.py": """\
        def once():
            return 1


        def twice():
            return 2


        def run():
            return once() + twice() + twice()
        """, "bench/run.py": """\
        from quail.a import run

        run()
        """}
    data = _measure(tmp_path, monkeypatch, source_text)
    assert _items(data, "single_use_functions") == [
        "quail/a.py:1 once 2 lines, used at quail/a.py:10",
        "quail/a.py:9 run 2 lines, used at bench/run.py:3"]
    assert _items(data, "unused_functions") == []


def test_a_method_called_on_an_unknown_value_counts_toward_each_method_of_its_name(
        tmp_path, monkeypatch):
    source_text = {"quail/a.py": """\
        class A:
            def step(self):
                return self.helper()

            def helper(self):
                return 1


        class B:
            def step(self):
                return 2


        def run(thing):
            return thing.step()
        """, "bench/run.py": """\
        from quail.a import run

        run(None)
        """}
    data = _measure(tmp_path, monkeypatch, source_text)
    # self.helper() resolves; thing.step() could be either step
    assert _items(data, "single_use_functions") == [
        "quail/a.py:5 helper 2 lines, used at quail/a.py:3",
        "quail/a.py:14 run 2 lines, used at bench/run.py:3"]
    assert _items(data, "unused_functions") == []


def test_a_name_in_a_string_or_rust_counts_as_a_reference(
        tmp_path, monkeypatch):
    source_text = {"quail/a.py": """\
        def looked_up():
            return 1


        def from_rust():
            return 2


        def unused():
            return 3


        HANDLER = getattr(None, "looked_up", None)
        """, "crates/x/src/lib.rs": """\
        fn call(worker: &Worker) { worker.call_method0("from_rust"); }
        """}
    data = _measure(tmp_path, monkeypatch, source_text)
    assert _items(data, "unused_functions") == ["quail/a.py:9 unused 2 lines"]
    assert _items(data, "single_use_functions") == []


def test_a_function_only_tests_call_is_test_only(tmp_path, monkeypatch):
    source_text = {"quail/a.py": """\
        def checked():
            return 1
        """, "tests/test_a.py": """\
        from quail.a import checked


        def test_checked():
            assert checked() == 1
        """}
    data = _measure(tmp_path, monkeypatch, source_text)
    assert _items(data, "test_only_functions") == [
        "quail/a.py:1 checked 2 lines"]


def test_size_skips_comments_and_docstrings_and_counts_an_fstring_once(
        tmp_path, monkeypatch):
    source_text = {"quail/a.py": '''\
        """A module."""

        # a comment
        def f(x):
            """A docstring."""
            return f"{x} and {x + 1}"
        '''}
    data = _measure(tmp_path, monkeypatch, source_text)
    # def f ( x ) : return <f-string>
    assert _total(data, "code_tokens") == 8
    assert _total(data, "code_lines") == 2
    assert _total(data, "comment_lines") == 1
    assert _total(data, "docstring_lines") == 2


def test_an_elif_sits_at_the_depth_of_its_if(tmp_path, monkeypatch):
    source_text = {"quail/a.py": """\
        def f(a, b, c):
            for x in a:
                if b:
                    with c:
                        return x
                elif c:
                    return 1
            return 0
        """}
    data = _measure(tmp_path, monkeypatch, source_text)
    assert _total(data, "max_depth") == 3
    assert _total(data, "depth_excess") == 1
    assert _items(data, "deep_functions") == []


def test_python_error_and_style_checks_find_each_case(tmp_path, monkeypatch):
    source_text = {"quail/a.py": """\
        # ---- section ----
        def f(value, other):
            if value is None: raise ValueError("no value")
            if not hasattr(value, "x"):
                raise TypeError("no x")
            try:
                return g(value,
                         other)
            except KeyError:
                return g(
                    value, other)


        def g(value, other):
            return value
        """}
    data = _measure(tmp_path, monkeypatch, source_text)
    assert _items(data, "divider_comments") == ["quail/a.py:1"]
    assert _items(data, "one_line_blocks") == ["quail/a.py:3"]
    assert _items(data, "error_guards") == ["quail/a.py:3", "quail/a.py:4"]
    assert _items(data, "guard_lookups") == ["quail/a.py:4"]
    assert _items(data, "error_handlers") == ["quail/a.py:9"]
    assert _total(data, "error_sites") == 2
    assert _items(data, "aligned_continuations") == ["quail/a.py:8"]


def test_call_sites_show_unused_parameters_and_constant_arguments(
        tmp_path, monkeypatch):
    source_text = {"quail/a.py": """\
        def scaled(value, factor, verbose=False):
            return value * factor


        def run(values):
            return [scaled(values[0], 2), scaled(values[1], factor=2)]


        def forward(value, factor):
            return scaled(value, factor)
        """, "bench/run.py": """\
        from quail.a import forward, run

        run([1, 2])
        forward(1, 3)
        """}
    data = _measure(tmp_path, monkeypatch, source_text)
    # forward passes factor through, so only verbose is never passed
    assert _items(data, "unused_parameters") == ["quail/a.py:1 scaled(verbose)"]
    assert _items(data, "constant_arguments") == []
    assert _items(data, "pass_through_functions") == ["quail/a.py:9 forward"]
    source_text = {"quail/a.py": """\
        def scaled(value, factor, verbose=False):
            return value * factor


        def run(values):
            return [scaled(values[0], 2), scaled(values[1], factor=2)]
        """, "bench/run.py": """\
        from quail.a import run

        run([1, 2])
        """}
    data = _measure(tmp_path, monkeypatch, source_text)
    # without forward as a caller, every call passes factor=2
    assert _items(data, "constant_arguments") == [
        "quail/a.py:1 scaled(factor) always 2"]


RUST_TEXT = """\
    pub fn exported() -> u32 { helper(1) }

    fn helper(x: u32) -> u32 {
        // ===== divider =====
        if x == 0 { return Err(Error::Zero); }
        let Some(y) = lookup(x) else {
            return Err(Error::Missing);
        };
        {
            let z = y?;
        }
        for a in 0..x {
            match a {
                1 => {
                    if a > 0 {
                        return a;
                    }
                }
                _ => {}
            }
        }
        y.map_err(wrap).unwrap()
    }

    fn forward(&self, x: u32) -> u32 {
        self.inner.helper(x)
    }

    #[cfg(test)]
    mod tests {
        fn only_in_tests() { if true { return Err(1); } }
    }
    """


def test_rust_checks_skip_test_code_and_find_each_case(tmp_path, monkeypatch):
    data = _measure(tmp_path, monkeypatch,
                    {"crates/x/src/lib.rs": RUST_TEXT})
    assert _items(data, "divider_comments") == ["crates/x/src/lib.rs:4"]
    assert _items(data, "error_guards") == [
        "crates/x/src/lib.rs:5", "crates/x/src/lib.rs:6"]
    assert _items(data, "one_line_blocks") == ["crates/x/src/lib.rs:5"]
    assert _items(data, "bare_blocks") == ["crates/x/src/lib.rs:9"]
    assert _items(data, "error_handlers") == ["crates/x/src/lib.rs:22"]
    assert _items(data, "pass_through_functions") == [
        "crates/x/src/lib.rs:25 forward"]
    assert _total(data, "error_sites", "rust") == 2
    assert _total(data, "error_propagations", "rust") == 1
    assert _total(data, "asserts", "rust") == 1
    # for { match { arm { if { return } } } }
    assert _items(data, "deep_functions") == [
        "crates/x/src/lib.rs:3 helper depth 4"]
    assert _total(data, "code_lines", "rust") == 24
    assert _total(data, "comment_lines", "rust") == 1


@pytest.mark.skipif(shutil.which("rust-analyzer") is None
                    or shutil.which("cargo") is None,
                    reason="needs rust-analyzer")
def test_rust_references_come_from_the_rust_analyzer_index(
        tmp_path, monkeypatch):
    source_text = {
        "Cargo.toml": """\
            [workspace]
            members = ["crates/quail"]
            resolver = "2"
            """,
        "rust-toolchain.toml": (skill_issues.ROOT
                                / "rust-toolchain.toml").read_text(),
        "crates/quail/Cargo.toml": """\
            [package]
            name = "quail"
            version = "0.1.0"
            edition = "2024"
            """,
        "crates/quail/src/lib.rs": """\
            mod inner;
            use inner::twice;

            pub fn api() -> u32 { once() + twice() + twice() }

            fn once() -> u32 { 1 }

            fn unused() {}

            pub(crate) fn tested() -> u32 { 3 }

            #[cfg(test)]
            mod tests {
                #[test]
                fn it() { assert_eq!(super::tested(), 3); }
            }
            """,
        "crates/quail/src/inner.rs": """\
            pub(crate) fn twice() -> u32 { 2 }
            """}
    data = _measure(tmp_path, monkeypatch, source_text, references=True)
    assert _items(data, "single_use_functions") == [
        "crates/quail/src/lib.rs:6 once 1 lines, used at "
        "crates/quail/src/lib.rs:4"]
    assert _items(data, "unused_functions") == [
        "crates/quail/src/lib.rs:8 unused 1 lines"]
    assert _items(data, "test_only_functions") == [
        "crates/quail/src/lib.rs:10 tested 1 lines"]
    assert _items(data, "excluded_functions") == [
        "crates/quail/src/lib.rs:4 api public"]
