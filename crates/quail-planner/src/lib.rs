//! Quail's planner, ported from `quail/planner` and `quail/logical`.
//!
//! | module | Python |
//! |---|---|
//! | `logical` | the node types of `quail/logical/nodes` and `LogicalPlan` |
//! | `expressions` | the helpers of `quail/common/expressions.py` |
//! | `builder` | `quail/logical/naive.py`, `placement.py`, `builder.py` |
//! | `nodes` | the methods of the node classes in `quail/logical/nodes` |
//! | `arrow_functions` | `pyarrow.compute`'s function list |

pub mod arrow_functions;
pub mod builder;
pub mod expressions;
pub mod logical;
pub mod nodes;

pub use expressions::CompileError;
pub use logical::{LogicalNode, LogicalPlan, read_logical_plan, write_logical_plan};
