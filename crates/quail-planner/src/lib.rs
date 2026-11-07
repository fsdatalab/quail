//! Quail's planner, ported from `quail/planner` and the deleted
//! `quail/logical` package.
//!
//! | module | Python |
//! |---|---|
//! | `logical` | the logical node types and `LogicalPlan` |
//! | `expressions` | the helpers of `quail/common/expressions.py` |
//! | `context` | `PlanningContext` |
//! | `costs` | the filter costs and join specs of `quail/planner/operators` |
//! | `columns` | the column planners of `quail/planner/operators` |
//! | `classify` | `quail/planner/operators/classify.py` |
//! | `builder` | building a logical plan from a bound region |
//! | `nodes` | what each logical node reads and returns, and its checks |
//! | `lowering` | `quail/planner/prompt_lowering.py`, with a tokenizer callback |
//! | `operators` | a plan's operators, grouped by kind |
//! | `plan_statistics` | `quail/planner/search/statistics.py` |
//! | `search` | the step order, join order, and anchor search of `quail/planner/search` |
//! | `scores` | `quail/planner/operators/score.py` |
//! | `statistics` | `quail/planner/statistics` |
//! | `rules` | the logical rules of `quail/planner/rules` and `logical_alternatives` |
//! | `prompts` | the prompt binding of `quail/common/prompts.py` |
//! | `arrow_functions` | `pyarrow.compute`'s function list |

pub mod arrow_functions;
pub mod builder;
pub mod classify;
pub mod columns;
pub mod context;
pub mod costs;
pub mod edits;
pub mod explain;
pub mod expressions;
pub mod facts;
pub mod graph;
pub mod logical;
pub mod lowering;
pub mod nodes;
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
pub mod rules;
pub mod scores;
pub mod search;
pub mod sol;
pub mod statistics;
pub mod validation;

pub use expressions::CompileError;
pub use logical::{LogicalNode, LogicalPlan};
