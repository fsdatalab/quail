//! Quail and its dependencies as one shared library.
//!
//! A test program or example that names this crate links DataFusion,
//! Arrow, and Quail from the library, not into its own file. Only
//! dev-dependencies name it: the Python module and `quail-server` hold
//! their own copy, so each ships as one file.

pub use quail;
