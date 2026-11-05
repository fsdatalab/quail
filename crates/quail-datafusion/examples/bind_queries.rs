//! Bind each query in a JSON file with DataFusion and report the result.
//!
//! The file is a list of `{"id", "sql", "tables": {name: [[column, type]]}}`.
//! `tools/dump_check_queries.py` writes the file. Run with
//! `cargo run --example bind_queries -- <file> [dialect]`.

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::{Result, tree_node::TreeNode, tree_node::TreeNodeRecursion};
use datafusion::datasource::MemTable;
use datafusion::logical_expr::LogicalPlan;
use quail_datafusion::{ModelRegion, RegionRunner, session_context, sql};

#[derive(Debug)]
struct NoRows;

impl RegionRunner for NoRows {
    fn run(
        &self,
        _region: &ModelRegion,
        _inputs: &[quail_datafusion::InputRows],
        _fetch: Option<usize>,
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

fn count_regions(plan: &LogicalPlan) -> usize {
    let mut count = 0;
    plan.apply(|node| {
        if let LogicalPlan::Extension(extension) = node
            && extension.node.as_any().is::<ModelRegion>()
        {
            count += 1;
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    count
}

#[tokio::main]
async fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&arguments[1])?;
    let dialect = arguments.get(2).cloned();
    let queries: serde_json::Value = serde_json::from_str(&text).unwrap();
    for query in queries.as_array().unwrap() {
        let context = session_context(Arc::new(NoRows));
        if let Some(dialect) = &dialect {
            context
                .sql(&format!("SET datafusion.sql_parser.dialect = '{dialect}'"))
                .await?;
        }
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
        let id = query["id"].as_str().unwrap();
        let outcome = match sql(&context, query["sql"].as_str().unwrap()).await {
            Err(error) => format!("BIND ERROR: {error}"),
            Ok(frame) => match frame.into_optimized_plan() {
                Err(error) => format!("OPTIMIZE ERROR: {error}"),
                Ok(plan) => format!(
                    "ok, {} region(s), root is region: {}",
                    count_regions(&plan),
                    matches!(plan, LogicalPlan::Extension(_))
                ),
            },
        };
        println!("{id}\t{}", outcome.replace('\n', " "));
    }
    Ok(())
}
