//! Quail's DataFusion extension in a DataFusion session. The modules
//! share one test binary, which links DataFusion once.

// links DataFusion and Quail from the shared library
use quail_dylib as _;

mod claim;
mod dynamic_filter;
mod extension;
mod streaming;
