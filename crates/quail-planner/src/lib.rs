//! Quail's planner, ported from `quail/planner` and `quail/logical`.
//!
//! | module | Python |
//! |---|---|
//! | `logical` | the node types of `quail/logical/nodes` and `LogicalPlan` |
//! | `expressions` | the helpers of `quail/common/expressions.py` |
//! | `context` | `PlanningContext` |
//! | `costs` | the filter costs and join specs of `quail/planner/operators` |
//! | `columns` | the column planners of `quail/planner/operators` |
//! | `classify` | `quail/planner/operators/classify.py` |
//! | `builder` | `quail/logical/naive.py`, `placement.py`, `builder.py` |
//! | `nodes` | the methods of the node classes in `quail/logical/nodes` |
//! | `lowering` | `quail/planner/prompt_lowering.py`, with a tokenizer callback |
//! | `operators` | `LogicalPlan.operators` in `quail/logical/plan.py` |
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
pub mod expressions;
pub mod graph;
pub mod logical;
pub mod lowering;
pub mod nodes;
pub mod operators;
pub mod plan_statistics;
pub mod pricing;
pub mod prompts;
pub mod results;
pub mod retention;
pub mod rules;
pub mod scores;
pub mod search;
pub mod statistics;
pub mod validation;

pub use expressions::CompileError;
pub use logical::{LogicalNode, LogicalPlan, read_logical_plan, write_logical_plan};
