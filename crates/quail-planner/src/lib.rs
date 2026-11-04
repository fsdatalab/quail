//! Quail's planner, ported from `quail/planner` and `quail/logical`.
//!
//! | module | Python |
//! |---|---|
//! | `logical` | the node types of `quail/logical/nodes` and `LogicalPlan` |
//! | `expressions` | the helpers of `quail/common/expressions.py` |
//! | `context` | `PlanningContext` |
//! | `columns` | the column planners of `quail/planner/operators` |
//! | `builder` | `quail/logical/naive.py`, `placement.py`, `builder.py` |
//! | `nodes` | the methods of the node classes in `quail/logical/nodes` |
//! | `lowering` | `quail/planner/prompt_lowering.py`, with a tokenizer callback |
//! | `operators` | `LogicalPlan.operators` in `quail/logical/plan.py` |
//! | `rules` | the logical rules of `quail/planner/rules` and `logical_alternatives` |
//! | `prompts` | the prompt binding of `quail/common/prompts.py` |
//! | `arrow_functions` | `pyarrow.compute`'s function list |

pub mod arrow_functions;
pub mod builder;
pub mod columns;
pub mod context;
pub mod expressions;
pub mod logical;
pub mod lowering;
pub mod nodes;
pub mod operators;
pub mod prompts;
pub mod rules;

pub use expressions::CompileError;
pub use logical::{LogicalNode, LogicalPlan, read_logical_plan, write_logical_plan};
