//! Record libpython's directory in this package's binaries, so its
//! tests start the interpreter without `LD_LIBRARY_PATH`.

fn main() {
    let config = pyo3_build_config::get();
    if let Some(directory) = config.lib_dir() {
        println!("cargo:rustc-link-arg-tests=-Wl,-rpath,{directory}");
    }
}
