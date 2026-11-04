//! Quail's planner, ported from `quail/planner` and `quail/logical`.
//!
//! | module | Python |
//! |---|---|
//! | `logical` | the node types of `quail/logical/nodes` and `LogicalPlan` |

pub mod logical;

pub use logical::{LogicalNode, LogicalPlan, read_logical_plan, write_logical_plan};
