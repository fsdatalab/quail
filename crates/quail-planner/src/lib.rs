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

pub mod cost;
pub mod inputs;
pub mod physical;

pub mod classify;
pub mod columns;
pub mod context;
pub mod costs;
pub mod edits;
pub mod explain;
pub mod expressions;
pub mod facts;
pub mod graph;
pub mod lowering;
pub mod operators;
pub mod optimize;
pub mod physical_rules;
pub mod placements;
pub mod plan_statistics;
pub mod pricing;
pub mod prompts;
pub mod request_order;
pub mod request_plan;
pub mod results;
pub mod retention;
pub mod scores;
pub mod search;
pub mod sol;
pub mod stages;
pub mod statistics;
pub mod validation;

pub use expressions::CompileError;
