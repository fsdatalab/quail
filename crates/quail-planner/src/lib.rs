//! Quail's planner of one model region. It reads the region's operators,
//! which `quail_datafusion` derives from DataFusion's plan, and chooses a
//! physical plan for the GPU. It depends on neither DataFusion nor Python.
//!
//! - [`operators`], [`expressions`], [`facts`]: a region's operators and
//!   what the planner reads from them.
//! - [`physical`]: the physical plan types the executor runs.
//! - [`cost`]: model and device specs, and the cost model.
//! - [`inputs`]: tokenized document columns and prefix statistics.
//! - [`search`], [`optimize`], [`pricing`], [`physical_rules`]: the choice
//!   of a plan: operator order, join order, prices, and graph rewrites.
//! - [`prompts`], [`lowering`]: prompt text and its token ids.
//! - [`explain`], [`sol`], [`request_plan`]: plan text, speed-of-light
//!   estimates, and the plans of the request baselines.

pub mod budgets;
pub mod cost;
pub mod inputs;
pub mod physical;

pub mod constants;
pub mod context;
pub mod error;
pub mod expressions;
pub mod operators;
pub mod plan;
pub mod pricing;
pub mod prompts;
pub mod specs;
pub mod topology;

pub use error::{CompileError, ErrorKind, Refusal, compile_error};
