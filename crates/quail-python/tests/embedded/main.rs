use quail_dylib as _;

mod caller_routing;
mod foreign_table;
mod optional_attributes;
mod session;

use std::path::PathBuf;
use std::sync::Once;

use quail_python::native::_quail_native;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn start_python() {
    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        if std::env::var_os("VIRTUAL_ENV").is_none() {
            unsafe { std::env::set_var("VIRTUAL_ENV", root().join(".venv")) };
        }
        pyo3::append_to_inittab!(_quail_native);
        quail_python::start_python().expect("Python starts");
    });
}
