//! Plan a region: validate its operators, build the physical graph in
//! the chosen order, refine it with the physical rules, cut it into
//! stages, and say what it returns; the plan of a request backend; and
//! the text of `explain()`.

pub mod explain;
pub mod facts;
pub mod graph;
pub mod optimize;
pub mod physical_rules;
pub mod placements;
pub mod request_order;
pub mod request_plan;
pub mod results;
pub mod retention;
pub mod stages;
pub mod validation;
