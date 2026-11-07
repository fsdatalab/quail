//! The join above a model region that adds its tables' columns back.
//!
//! A model region returns, for each row, the id column of each table it
//! reads and its answers, not the tables' other columns. [`join_back`]
//! splits a claimed part's SELECT list into the region's, the id
//! columns and the answers, and the join above it: for each table whose
//! other columns the SELECT list reads, an inner join of the region's
//! rows with the table on its id, then the SELECT list's columns as the
//! part named them.
//!
//! A table's id is its primary key. An input DataFusion computes for
//! the region without one, such as a CTE that groups rows, has
//! [`ROW_COLUMN`]. The join reads an input as the region does, through
//! the same [`crate::SharedInput`], so its rows are computed once; it scans an
//! identified table the region reads by name again.
//!
//! For example, `SELECT r.id, r.title, AI.CLASSIFY(...) AS topic FROM
//! reviews r` becomes a region returning `r.id` and `topic`, joined with
//! `reviews` on `id` for `r.title`.

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, JoinType, Result, TableReference};
use datafusion::datasource::source_as_provider;
use datafusion::logical_expr::{
    Expr, LogicalPlan, LogicalPlanBuilder, Projection, SubqueryAlias, TableScan, TableScanBuilder,
};

use crate::ai::AiNode;
use crate::extension::{as_node, extension_node};
use crate::identified::IdentifiedTable;
use crate::shared::ROW_COLUMN;

/// Where a table a region reads comes from.
#[derive(Debug, Clone)]
enum Source {
    /// An identified table the region reads by name.
    Named(TableScan),
    /// The region's input of this name.
    Input(String),
}

/// One table the region's rows hold the id of, and the columns the
/// join above adds back from it.
#[derive(Debug, Clone)]
struct Joined {
    alias: TableReference,
    key: String,
    source: Source,
    columns: Vec<String>,
}

/// The join DataFusion runs above a region, as the module explains.
#[derive(Debug, Default)]
pub(crate) struct JoinBack {
    joined: Vec<Joined>,
    /// The part's SELECT list, as the region's columns or the joined
    /// tables' columns, under the names the part gave them.
    columns: Vec<(Option<TableReference>, String)>,
    /// Whether the region's SELECT list is the part's.
    unchanged: bool,
}

impl JoinBack {
    /// The names of the inputs the join reads.
    pub(crate) fn inputs(&self) -> BTreeSet<String> {
        self.joined
            .iter()
            .filter(|joined| !joined.columns.is_empty())
            .filter_map(|joined| match &joined.source {
                Source::Input(name) => Some(name.clone()),
                Source::Named(_) => None,
            })
            .collect()
    }

    /// The join over `region`, reading each input from `inputs`, the
    /// region's inputs by name.
    ///
    /// # Errors
    ///
    /// A join or projection does not plan.
    pub(crate) fn build(
        self,
        region: LogicalPlan,
        inputs: &[(String, LogicalPlan)],
    ) -> Result<LogicalPlan> {
        if self.unchanged {
            return Ok(region);
        }
        let joined: Vec<&Joined> = self
            .joined
            .iter()
            .filter(|joined| !joined.columns.is_empty())
            .collect();
        // each joined table's id leaves the region under a name of its
        // own, as a region over two tables returns two ids of one name,
        // such as `r.id` and `p.id`, and DataFusion 55.2's dynamic
        // filters, when a session turns them on, read a column by name
        // (tests/dynamic_filter.rs)
        let renamed = region
            .schema()
            .iter()
            .map(|(qualifier, field)| {
                let column = Expr::Column(Column::new(qualifier.cloned(), field.name()));
                match id_of(&joined, qualifier, field.name()) {
                    Some(alias) => column.alias(key_name(alias)),
                    None => column,
                }
            })
            .collect();
        let mut plan = LogicalPlan::Projection(Projection::try_new(renamed, Arc::new(region))?);
        for joined in &joined {
            let table = match &joined.source {
                Source::Named(scan) => {
                    let schema = source_as_provider(&scan.source)?.schema();
                    let projection = std::iter::once(&joined.key)
                        .chain(&joined.columns)
                        .map(|name| schema.index_of(name))
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    LogicalPlanBuilder::scan(
                        scan.table_name.clone(),
                        Arc::clone(&scan.source),
                        Some(projection),
                    )?
                    .alias(joined.alias.clone())?
                    .build()?
                }
                Source::Input(name) => {
                    let Some((_, input)) = inputs.iter().find(|(found, _)| found == name) else {
                        return datafusion::common::internal_err!("no region input {name}");
                    };
                    let column =
                        |name: &str| Expr::Column(Column::new(Some(joined.alias.clone()), name));
                    let read = std::iter::once(&joined.key)
                        .chain(&joined.columns)
                        .map(|name| column(name))
                        .collect();
                    let table = LogicalPlan::SubqueryAlias(SubqueryAlias::try_new(
                        Arc::new(input.clone()),
                        joined.alias.clone(),
                    )?);
                    LogicalPlan::Projection(Projection::try_new(read, Arc::new(table))?)
                }
            };
            // DataFusion builds its hash table on the side its statistics
            // say is smaller, and on the region's rows when it cannot tell
            plan = LogicalPlanBuilder::from(plan)
                .join(
                    table,
                    JoinType::Inner,
                    (
                        vec![Column::from_name(key_name(&joined.alias))],
                        vec![Column::new(Some(joined.alias.clone()), &joined.key)],
                    ),
                    None,
                )?
                .build()?;
        }
        // a joined table's id comes from the table, as the region's rows
        // hold it under another name
        let exprs: Vec<Expr> = self
            .columns
            .into_iter()
            .map(|(qualifier, name)| Expr::Column(Column::new(qualifier, name)))
            .collect();
        Ok(LogicalPlan::Projection(Projection::try_new(
            exprs,
            Arc::new(plan),
        )?))
    }
}

/// The alias of the joined table whose id is the column `qualifier.name`.
fn id_of<'a>(
    joined: &[&'a Joined],
    qualifier: Option<&TableReference>,
    name: &str,
) -> Option<&'a TableReference> {
    joined
        .iter()
        .find(|joined| qualifier == Some(&joined.alias) && name == joined.key)
        .map(|joined| &joined.alias)
}

/// The name the region's rows give a joined table's id above the
/// region; the table holds the id under its own name.
fn key_name(alias: &TableReference) -> String {
    format!("__quail_key_{}", alias.table())
}

/// Split a part whose root is its SELECT list into the region, which
/// returns the id column of each table it reads and its answers, and the
/// join above it, as the module explains.
///
/// # Errors
///
/// A rebuilt operator does not plan.
pub(crate) fn join_back(
    plan: LogicalPlan,
    input_names: &[String],
) -> Result<(LogicalPlan, JoinBack)> {
    let LogicalPlan::Projection(root) = &plan else {
        return Ok((
            plan,
            JoinBack {
                unchanged: true,
                ..JoinBack::default()
            },
        ));
    };
    let mut relation = root.input.as_ref().clone();
    let answers = answer_names(&relation);
    let mut joined: Vec<Joined> = Vec::new();
    for (qualifier, _) in relation.schema().iter() {
        let Some(alias) = qualifier else { continue };
        if joined.iter().any(|found| found.alias == *alias) {
            continue;
        }
        let Some(scan) = leaf(&relation, alias) else {
            continue;
        };
        let Some(key) = key_of(&scan)? else { continue };
        let source = if source_as_provider(&scan.source)?
            .downcast_ref::<IdentifiedTable>()
            .is_some()
        {
            Source::Named(scan.clone())
        } else if input_names
            .iter()
            .any(|name| name == scan.table_name.table())
        {
            Source::Input(scan.table_name.table().to_string())
        } else {
            continue;
        };
        joined.push(Joined {
            alias: alias.clone(),
            key,
            source,
            columns: Vec::new(),
        });
    }
    for found in &joined {
        let key = Column::new(Some(found.alias.clone()), &found.key);
        if !relation.schema().has_column(&key) {
            relation = pass_up(relation, &found.alias, &found.key)?;
        }
    }
    // the SELECT list's columns of a joined table other than its id move
    // above the region
    let mut kept: Vec<Expr> = Vec::new();
    let mut columns = Vec::new();
    for (expr, (qualifier, field)) in root.expr.iter().zip(root.schema.iter()) {
        let moved = match expr {
            Expr::Column(column) if !answers.contains(&column.name) => joined
                .iter_mut()
                .find(|found| column.relation.as_ref() == Some(&found.alias))
                .filter(|found| found.key != column.name && readable(found, &column.name)),
            _ => None,
        };
        match moved {
            Some(found) => {
                if !found.columns.contains(&field.name().to_string()) {
                    found.columns.push(field.name().to_string());
                }
                columns.push((qualifier.cloned(), field.name().clone()));
            }
            None => {
                kept.push(expr.clone());
                columns.push((qualifier.cloned(), field.name().clone()));
            }
        }
    }
    let unchanged_relation = relation == *root.input;
    for found in &joined {
        let key = Expr::Column(Column::new(Some(found.alias.clone()), &found.key));
        if !kept.contains(&key) {
            kept.push(key);
        }
    }
    let unchanged = unchanged_relation && kept == root.expr;
    let region = LogicalPlan::Projection(Projection::try_new(kept, Arc::new(relation))?);
    Ok((
        region,
        JoinBack {
            joined,
            columns,
            unchanged,
        },
    ))
}

/// Whether the join can read `name` from a joined table: a column of
/// the table itself, not one an operator inside the region computes.
fn readable(joined: &Joined, name: &str) -> bool {
    let schema = match &joined.source {
        Source::Named(scan) => source_as_provider(&scan.source).map(|provider| provider.schema()),
        Source::Input(_) => return name != ROW_COLUMN,
    };
    schema.is_ok_and(|schema| schema.index_of(name).is_ok())
}

/// The names of the columns the region's `AiNode`s add.
fn answer_names(plan: &LogicalPlan) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let _ = plan.apply(|node| {
        if let Some(ai) = as_node::<AiNode>(node) {
            found.extend(ai.fields.iter().map(|field| field.name().clone()));
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

/// The scan of the table the region reads as `alias`: under that alias,
/// or under its own name.
fn leaf(plan: &LogicalPlan, alias: &TableReference) -> Option<TableScan> {
    let mut found = None;
    let _ = plan.apply(|node| {
        match node {
            LogicalPlan::SubqueryAlias(subquery) if subquery.alias == *alias => {
                if let LogicalPlan::TableScan(scan) = subquery.input.as_ref() {
                    found = Some(scan.clone());
                }
                return Ok(TreeNodeRecursion::Stop);
            }
            LogicalPlan::SubqueryAlias(_) => return Ok(TreeNodeRecursion::Jump),
            LogicalPlan::TableScan(scan) if scan.table_name == *alias => {
                found = Some(scan.clone());
                return Ok(TreeNodeRecursion::Stop);
            }
            _ => {}
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

/// A table's id column: its one-column primary key, or [`ROW_COLUMN`].
fn key_of(scan: &TableScan) -> Result<Option<String>> {
    let provider = source_as_provider(&scan.source)?;
    let schema = provider.schema();
    if schema.index_of(ROW_COLUMN).is_ok() {
        return Ok(Some(ROW_COLUMN.to_string()));
    }
    let key = provider.constraints().and_then(|constraints| {
        constraints.iter().find_map(|constraint| match constraint {
            datafusion::common::Constraint::PrimaryKey(indices) if indices.len() == 1 => {
                Some(schema.field(indices[0]).name().clone())
            }
            _ => None,
        })
    });
    Ok(key)
}

/// `plan` returning `alias.key` too: the table's scan reads it, and each
/// operator above passes it on.
fn pass_up(plan: LogicalPlan, alias: &TableReference, key: &str) -> Result<LogicalPlan> {
    let column = Column::new(Some(alias.clone()), key);
    plan.transform_up(|node| match node {
        LogicalPlan::TableScan(scan) if scan.table_name == *alias => Ok(Transformed::yes(
            LogicalPlan::TableScan(reading(scan, key)?),
        )),
        LogicalPlan::SubqueryAlias(subquery) if subquery.alias == *alias => {
            let LogicalPlan::TableScan(scan) = subquery.input.as_ref().clone() else {
                return Ok(Transformed::no(LogicalPlan::SubqueryAlias(subquery)));
            };
            Ok(Transformed::yes(LogicalPlan::SubqueryAlias(
                SubqueryAlias::try_new(
                    Arc::new(LogicalPlan::TableScan(reading(scan, key)?)),
                    subquery.alias,
                )?,
            )))
        }
        LogicalPlan::Projection(projection)
            if projection.input.schema().has_column(&column)
                && !projection.schema.has_column(&column) =>
        {
            let mut exprs = projection.expr.clone();
            exprs.push(Expr::Column(column.clone()));
            Ok(Transformed::yes(LogicalPlan::Projection(
                Projection::try_new(exprs, Arc::clone(&projection.input))?,
            )))
        }
        LogicalPlan::Extension(_) if as_node::<AiNode>(&node).is_some() => {
            let ai = as_node::<AiNode>(&node).expect("an AiNode").clone();
            let input = ai.input.as_ref().clone();
            Ok(Transformed::yes(extension_node(ai.with_input(input)?)))
        }
        other => other.recompute_schema().map(Transformed::yes),
    })
    .map(|done| done.data)
}

/// `scan` reading the column `key` too.
fn reading(scan: TableScan, key: &str) -> Result<TableScan> {
    let index = source_as_provider(&scan.source)?.schema().index_of(key)?;
    match scan.projection.clone() {
        Some(mut projection) if !projection.contains(&index) => {
            projection.push(index);
            TableScanBuilder::new(scan.table_name, scan.source)
                .with_projection(Some(projection))
                .with_filters(scan.filters)
                .with_fetch(scan.fetch)
                .build()
        }
        _ => Ok(scan),
    }
}
