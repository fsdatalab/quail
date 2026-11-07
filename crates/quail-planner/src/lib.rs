//! Quail's planner, ported from `quail/planner` and the deleted
//! `quail/logical` package.
//!
//! | module | Python |
//! |---|---|
//! | `expressions` | the helpers of `quail/common/expressions.py` |
//! | `context` | `PlanningContext` |
//! | `costs` | the filter costs and join specs of `quail/planner/operators` |
//! | `columns` | the column planners of `quail/planner/operators` |
//! | `classify` | `quail/planner/operators/classify.py` |
//! | `lowering` | `quail/planner/prompt_lowering.py`, with a tokenizer callback |
//! | `operators` | a region's operators, grouped by kind, and its scans |
//! | `plan_statistics` | `quail/planner/search/statistics.py` |
//! | `search` | the step order, join order, and anchor search of `quail/planner/search` |
//! | `scores` | `quail/planner/operators/score.py` |
//! | `statistics` | `quail/planner/statistics` |
//! | `prompts` | the prompt binding of `quail/common/prompts.py` |
//! | `arrow_functions` | `pyarrow.compute`'s function list |

pub mod arrow_functions;
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
pub mod statistics;
pub mod validation;

pub use expressions::CompileError;
