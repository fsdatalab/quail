//! The tests that embed Python, in one program: each test program links
//! its own copy of DataFusion, about 300 MB.

// links DataFusion and Quail from the shared library
use quail_dylib as _;

mod caller_routing;
mod optional_attributes;
mod session;

use std::path::PathBuf;
use std::sync::Once;

use quail_python::native::_quail_native;

/// The repository's root.
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Start Python, once, with `_quail_native` built from this workspace as
/// a built-in module, so no test depends on the module `uv sync` last
/// installed.
fn start_python() {
    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        if std::env::var_os("VIRTUAL_ENV").is_none() {
            // SAFETY: set before Python starts, while no other thread reads it
            unsafe { std::env::set_var("VIRTUAL_ENV", root().join(".venv")) };
        }
        // the table of built-in modules is read when Python starts
        pyo3::append_to_inittab!(_quail_native);
        quail_python::start_python().expect("Python starts");
    });
}
