//! Print each model region's DataFusion plan for the queries in a JSON file.
//!
//! The file is a list of `{"id", "sql", "tables": {name: [[column, type]]}}`.
//! Run with `cargo run --example region_plans -- <file>`.

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::{Result, tree_node::TreeNode, tree_node::TreeNodeRecursion};
use datafusion::datasource::MemTable;
use datafusion::logical_expr::LogicalPlan;
use quail_datafusion::{ModelRegion, RegionRunner, session_context};

#[derive(Debug)]
struct NoRows;

impl RegionRunner for NoRows {
    fn run(
        &self,
        _region: &ModelRegion,
        _inputs: Vec<quail_datafusion::InputStream>,
        _fetch: Option<usize>,
        _hint: &quail_datafusion::RegionHint,
    ) -> Result<quail_datafusion::RegionBatches> {
        Ok(Box::new(std::iter::empty()))
    }
}

fn arrow_type(name: &str) -> DataType {
    match name {
        "i64" | "i32" => DataType::Int64,
        "fp64" | "fp32" => DataType::Float64,
        "bool" => DataType::Boolean,
        _ => DataType::Utf8,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&arguments[1])?;
    let queries: serde_json::Value = serde_json::from_str(&text).unwrap();
    for query in queries.as_array().unwrap() {
        let context = session_context(Arc::new(NoRows));
        for (name, columns) in query["tables"].as_object().unwrap() {
            let fields: Vec<Field> = columns
                .as_array()
                .unwrap()
                .iter()
                .map(|column| {
                    Field::new(
                        column[0].as_str().unwrap(),
                        arrow_type(column[1].as_str().unwrap()),
                        true,
                    )
                })
                .collect();
            let table = MemTable::try_new(Arc::new(Schema::new(fields)), vec![vec![]])?;
            context.register_table(name.as_str(), Arc::new(table))?;
        }
        println!("== {}", query["id"].as_str().unwrap());
        let plan = match context.sql(query["sql"].as_str().unwrap()).await {
            Err(error) => {
                println!("BIND ERROR: {error}");
                continue;
            }
            Ok(frame) => frame.into_optimized_plan()?,
        };
        plan.apply(|node| {
            if let LogicalPlan::Extension(extension) = node
                && let Some(region) = extension.node.as_any().downcast_ref::<ModelRegion>()
            {
                println!("{}", region.plan.display_indent());
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
    }
    Ok(())
}
