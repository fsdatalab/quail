use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result, TableReference};
use datafusion::datasource::{MemTable, provider_as_source, source_as_provider};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{
    Expr, LogicalPlan, LogicalPlanBuilder, Projection, TableScan, TableScanBuilder,
    UserDefinedLogicalNodeCore,
};

use super::calls_ai;
use super::identified::IdentifiedTable;
use crate::ai::AiNode;
use crate::functions::{is_ai_call, is_quail_call};
use crate::plan::as_node;
use crate::shared::{ROW_COLUMN, row_field};

pub(super) fn precompute_plain_expressions(plan: LogicalPlan) -> Result<LogicalPlan> {
    let mut tables: Vec<TableReference> = Vec::new();
    plan.apply(|node| {
        if let LogicalPlan::SubqueryAlias(alias) = node
            && !calls_ai(node)?
            && single_scan(node).is_some()
        {
            tables.push(alias.alias.clone());
            return Ok(TreeNodeRecursion::Jump);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    if tables.is_empty() {
        return Ok(plan);
    }
    let mut reserved = std::collections::BTreeSet::new();
    plan.apply(|node| {
        reserved.extend(
            node.schema()
                .fields()
                .iter()
                .map(|field| field.name().clone()),
        );
        Ok(TreeNodeRecursion::Continue)
    })?;
    let mut computed: Vec<(TableReference, Expr, String)> = Vec::new();
    let mut find = |expr: &Expr| {
        let _ = expr.apply(|inner| {
            let Some(table) = plain_table(inner) else {
                return Ok(TreeNodeRecursion::Continue);
            };
            let field = matches!(inner, Expr::ScalarFunction(call) if call.name() == "get_field");
            if matches!(inner, Expr::Column(_) | Expr::Literal(..) | Expr::Alias(_))
                || field
                || !tables.contains(&table)
                || inner.exists(|found| {
                    Ok(matches!(
                        found,
                        Expr::AggregateFunction(_) | Expr::WindowFunction(_)
                    ))
                })?
            {
                return Ok(TreeNodeRecursion::Continue);
            }
            if !computed.iter().any(|(_, found, _)| found == inner) {
                let mut index = computed.len();
                let name = loop {
                    let candidate = format!("__quail_expr_{index}");
                    if reserved.insert(candidate.clone()) {
                        break candidate;
                    }
                    index += 1;
                };
                computed.push((table, inner.clone(), name));
            }
            Ok(TreeNodeRecursion::Jump)
        });
    };
    plan.apply(|node| {
        if let LogicalPlan::SubqueryAlias(alias) = node
            && tables.contains(&alias.alias)
        {
            return Ok(TreeNodeRecursion::Jump);
        }
        for expr in node.expressions() {
            if let LogicalPlan::Filter(_) = node {
                for term in split_conjunction(&expr) {
                    if plain_table(term).is_none() {
                        find(term);
                    }
                }
            } else {
                find(&expr);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    if computed.is_empty() {
        return Ok(plan);
    }
    let replace_inner = |expr: Expr| -> Result<Transformed<Expr>> {
        expr.transform_down(
            |inner| match computed.iter().find(|(_, found, _)| *found == inner) {
                Some((table, _, name)) => Ok(Transformed::new(
                    Expr::Column(Column::new(Some(table.clone()), name)),
                    true,
                    TreeNodeRecursion::Jump,
                )),
                None => Ok(Transformed::no(inner)),
            },
        )
    };
    let replace = |expr: Expr, filter: bool| -> Result<Transformed<Expr>> {
        if !filter {
            return replace_inner(expr);
        }
        let mut changed = false;
        let terms = split_conjunction(&expr)
            .into_iter()
            .cloned()
            .map(|term| {
                if plain_table(&term).is_some() {
                    return Ok(term);
                }
                let found = replace_inner(term)?;
                changed |= found.transformed;
                Ok(found.data)
            })
            .collect::<Result<Vec<_>>>()?;
        let joined = conjunction(terms).unwrap_or(datafusion::logical_expr::lit(true));
        Ok(if changed {
            Transformed::yes(joined)
        } else {
            Transformed::no(expr)
        })
    };
    plan.transform_up(|node| {
        if let LogicalPlan::SubqueryAlias(alias) = &node
            && tables.contains(&alias.alias)
        {
            let added: Vec<Expr> = computed
                .iter()
                .filter(|(table, _, _)| *table == alias.alias)
                .map(|(_, expr, name)| expr.clone().alias(name))
                .collect();
            if added.is_empty() {
                return Ok(Transformed::no(node));
            }
            let table = alias.alias.clone();
            let mut exprs: Vec<Expr> = node
                .schema()
                .columns()
                .into_iter()
                .map(Expr::Column)
                .collect();
            exprs.extend(added);
            let projected = LogicalPlanBuilder::from(node)
                .project(exprs)?
                .alias(table)?
                .build()?;
            return Ok(Transformed::yes(projected));
        }
        let is_projection = matches!(node, LogicalPlan::Projection(_));
        let is_filter = matches!(node, LogicalPlan::Filter(_));
        let mut changed = false;
        let exprs = node
            .expressions()
            .into_iter()
            .map(|expr| {
                let name = expr.schema_name().to_string();
                let found = replace(expr, is_filter)?;
                changed |= found.transformed;
                Ok(
                    if is_projection && found.transformed && !matches!(found.data, Expr::Alias(_)) {
                        found.data.alias(name)
                    } else {
                        found.data
                    },
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let inputs: Vec<LogicalPlan> = node.inputs().into_iter().cloned().collect();
        let rebuilt = node.with_new_exprs(exprs, inputs)?;
        Ok(Transformed::new(
            rebuilt,
            changed,
            TreeNodeRecursion::Continue,
        ))
    })
    .map(|done| done.data)
}

fn plain_table(term: &Expr) -> Option<TableReference> {
    let special = term
        .exists(|inner| Ok(is_quail_call(inner) || is_subquery_expr(inner)))
        .unwrap_or(true);
    if special {
        return None;
    }
    let mut tables = term
        .column_refs()
        .into_iter()
        .map(|column| column.relation.clone());
    let first = tables.next()??;
    tables
        .all(|table| table.as_ref() == Some(&first))
        .then_some(first)
}

fn is_subquery_expr(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Exists(_) | Expr::InSubquery(_) | Expr::ScalarSubquery(_)
    )
}

fn single_scan(plan: &LogicalPlan) -> Option<TableScan> {
    let mut scans = Vec::new();
    plan.apply(|node| {
        if crate::plan::as_node::<crate::shared::SharedInput>(node).is_some() {
            return Ok(TreeNodeRecursion::Jump);
        }
        if let LogicalPlan::TableScan(scan) = node {
            scans.push(scan.clone());
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .ok()?;
    match <[TableScan; 1]>::try_from(scans) {
        Ok([scan]) => Some(scan),
        Err(_) => None,
    }
}

fn read_by_name(plan: &LogicalPlan) -> bool {
    let scan = match plan {
        LogicalPlan::TableScan(scan) => scan,
        LogicalPlan::SubqueryAlias(alias) => match alias.input.as_ref() {
            LogicalPlan::TableScan(scan) => scan,
            _ => return false,
        },
        _ => return false,
    };
    scan.filters.is_empty()
        && scan.fetch.is_none()
        && source_as_provider(&scan.source)
            .is_ok_and(|provider| provider.downcast_ref::<IdentifiedTable>().is_some())
}

pub(super) fn numbered(plan: &LogicalPlan) -> Vec<String> {
    let mut found = Vec::new();
    let _ = plan.apply(|node| {
        if let LogicalPlan::TableScan(scan) = node
            && scan
                .source
                .schema()
                .field_with_name(ROW_COLUMN)
                .is_ok_and(|field| {
                    field
                        .metadata()
                        .get("quail.row_identity")
                        .is_some_and(|value| value == "true")
                })
        {
            found.push(scan.table_name.table().to_string());
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

fn placeholder_schema(
    schema: &datafusion::arrow::datatypes::Schema,
    numbered: bool,
) -> Arc<datafusion::arrow::datatypes::Schema> {
    let mut fields: Vec<_> = schema.fields().iter().cloned().collect();
    if numbered {
        fields.push(Arc::new(row_field()));
    }
    Arc::new(datafusion::arrow::datatypes::Schema::new(fields))
}

fn plain_input(plan: &LogicalPlan) -> Result<Option<TableScan>> {
    if read_by_name(plan) || calls_ai(plan)? {
        return Ok(None);
    }
    Ok(single_scan(plan))
}

pub(super) fn split_inputs(plan: LogicalPlan) -> Result<Split> {
    let mut inputs: Vec<(String, LogicalPlan)> = Vec::new();
    let merged = merged_aliases(&plan);
    let mut root = true;
    let plan = plan
        .transform_down(|node| {
            let at_root = std::mem::take(&mut root);
            let cut = if let Some(scan) = plain_input(&node)? {
                Some(cut_input(node.clone(), &scan, &mut inputs)?)
            } else if let LogicalPlan::TableScan(scan) = &node {
                Some(table_input(scan, &mut inputs)?)
            } else {
                None
            };
            if let Some(cut) = cut {
                return Ok(cut);
            }
            let LogicalPlan::SubqueryAlias(alias) = &node else {
                return Ok(Transformed::no(node));
            };
            let groups = alias.input.exists(|inner| {
                Ok(matches!(
                    inner,
                    LogicalPlan::Aggregate(_) | LogicalPlan::Window(_)
                ))
            })?;
            let table = matches!(alias.input.as_ref(), LogicalPlan::TableScan(_));
            let staged = !at_root && !table && !merged.contains(&alias.alias.table().to_string());
            if !groups && !staged {
                return Ok(Transformed::no(node));
            }
            let (mut name, body) = match alias.input.as_ref() {
                LogicalPlan::SubqueryAlias(cte) => {
                    (cte.alias.table().to_string(), cte.input.as_ref().clone())
                }
                body => (alias.alias.table().to_string(), body.clone()),
            };
            let read_before = inputs
                .iter()
                .any(|(found, plan)| *found == name && *plan == body);
            if !read_before && inputs.iter().any(|(found, _)| *found == name) {
                name = format!("{name}_{}", inputs.len());
            }
            let schema = placeholder_schema(body.schema().as_arrow(), true);
            let Ok(table) = MemTable::try_new(schema, vec![vec![]]) else {
                return Ok(Transformed::no(node));
            };
            let read =
                LogicalPlanBuilder::scan(name.as_str(), provider_as_source(Arc::new(table)), None)?
                    .alias(alias.alias.clone())?
                    .build()?;
            if !read_before {
                inputs.push((name, body));
            }
            Ok(Transformed::new(read, true, TreeNodeRecursion::Jump))
        })?
        .data;
    Ok((plan, inputs))
}

pub(super) type Split = (LogicalPlan, Vec<(String, LogicalPlan)>);

fn cut_input(
    node: LogicalPlan,
    scan: &TableScan,
    inputs: &mut Vec<(String, LogicalPlan)>,
) -> Result<Transformed<LogicalPlan>> {
    let schema = node.schema();
    let qualifiers: Vec<_> = schema
        .iter()
        .map(|(qualifier, _)| qualifier.cloned())
        .collect();
    let Some(Some(qualifier)) = qualifiers.first().cloned() else {
        return Ok(Transformed::no(node));
    };
    if qualifiers
        .iter()
        .any(|found| found.as_ref() != Some(&qualifier))
    {
        return Ok(Transformed::no(node));
    }
    if qualifier
        .table()
        .contains(quail_planner::operators::INPUT_SEPARATOR)
    {
        return Ok(Transformed::no(node));
    }
    let name = format!(
        "{}{}{}",
        scan.table_name.table(),
        quail_planner::operators::INPUT_SEPARATOR,
        qualifier.table()
    );
    let provider = source_as_provider(&scan.source)?;
    let key = provider.constraints().and_then(|constraints| {
        constraints.iter().find_map(|constraint| match constraint {
            datafusion::common::Constraint::PrimaryKey(indices) if indices.len() == 1 => {
                Some(indices[0])
            }
            _ => None,
        })
    });
    let key = key.map(|index| (index, provider.schema().field(index).name().clone()));
    let node = match &key {
        Some((index, name)) => with_key(node, *index, name)?,
        None => node,
    };
    let key = key.map(|(_, name)| name);
    let fields = placeholder_schema(node.schema().as_arrow(), key.is_none());
    let mut placeholder = MemTable::try_new(Arc::clone(&fields), vec![vec![]])?;
    if let Some(index) = key.and_then(|column| fields.index_of(&column).ok()) {
        placeholder =
            placeholder.with_constraints(datafusion::common::Constraints::new_unverified(vec![
                datafusion::common::Constraint::PrimaryKey(vec![index]),
            ]));
    }
    let read = LogicalPlanBuilder::scan(
        name.as_str(),
        provider_as_source(Arc::new(placeholder)),
        None,
    )?
    .alias(qualifier)?
    .build()?;
    inputs.push((name, node));
    Ok(Transformed::new(read, true, TreeNodeRecursion::Jump))
}

fn with_key(plan: LogicalPlan, index: usize, name: &str) -> Result<LogicalPlan> {
    plan.transform_up(|node| match node {
        LogicalPlan::TableScan(scan) => match scan.projection.clone() {
            Some(mut projection) if !projection.contains(&index) => {
                projection.push(index);
                let read = TableScanBuilder::new(scan.table_name, scan.source)
                    .with_projection(Some(projection))
                    .with_filters(scan.filters)
                    .with_fetch(scan.fetch)
                    .build()?;
                Ok(Transformed::yes(LogicalPlan::TableScan(read)))
            }
            _ => Ok(Transformed::no(LogicalPlan::TableScan(scan))),
        },
        LogicalPlan::Projection(projection) => {
            let found: Vec<Column> = projection
                .input
                .schema()
                .iter()
                .filter(|(_, field)| field.name() == name)
                .map(|(qualifier, field)| Column::new(qualifier.cloned(), field.name()))
                .collect();
            match <[Column; 1]>::try_from(found) {
                Ok([column]) if !projection.schema.has_column(&column) => {
                    let mut exprs = projection.expr.clone();
                    exprs.push(Expr::Column(column));
                    Ok(Transformed::yes(LogicalPlan::Projection(
                        Projection::try_new(exprs, Arc::clone(&projection.input))?,
                    )))
                }
                _ => Ok(Transformed::no(LogicalPlan::Projection(projection))),
            }
        }
        other => other.recompute_schema().map(Transformed::yes),
    })
    .map(|done| done.data)
}

fn table_input(
    scan: &TableScan,
    inputs: &mut Vec<(String, LogicalPlan)>,
) -> Result<Transformed<LogicalPlan>> {
    let provider = source_as_provider(&scan.source)?;
    if provider.downcast_ref::<IdentifiedTable>().is_some() {
        return Ok(Transformed::no(LogicalPlan::TableScan(scan.clone())));
    }
    let name = scan.table_name.table().to_string();
    let schema = provider.schema();
    let keyed = provider.constraints().is_some_and(|constraints| {
        constraints.iter().any(|constraint| {
            matches!(constraint, datafusion::common::Constraint::PrimaryKey(indices) if indices.len() == 1)
        })
    });
    let mut placeholder = MemTable::try_new(placeholder_schema(&schema, !keyed), vec![vec![]])?;
    let mut projection = scan.projection.clone();
    if keyed && let Some(constraints) = provider.constraints() {
        placeholder = placeholder.with_constraints(constraints.clone());
    } else if let Some(read) = &mut projection {
        read.push(schema.fields().len());
    }
    let read = LogicalPlanBuilder::scan_with_filters(
        scan.table_name.clone(),
        provider_as_source(Arc::new(placeholder)),
        projection,
        scan.filters.clone(),
    )?
    .build()?;
    if !inputs.iter().any(|(found, _)| *found == name) {
        let body =
            LogicalPlanBuilder::scan(scan.table_name.clone(), Arc::clone(&scan.source), None)?
                .build()?;
        inputs.push((name, body));
    }
    Ok(Transformed::new(read, true, TreeNodeRecursion::Jump))
}

fn subquery_body(alias: &datafusion::logical_expr::SubqueryAlias) -> (String, &LogicalPlan) {
    match alias.input.as_ref() {
        LogicalPlan::SubqueryAlias(cte) => (cte.alias.table().to_string(), cte.input.as_ref()),
        body => (alias.alias.table().to_string(), body),
    }
}

fn merged_aliases(plan: &LogicalPlan) -> Vec<String> {
    let mut readers: std::collections::BTreeMap<String, usize> = Default::default();
    let _ = plan.apply(|node| {
        if let LogicalPlan::SubqueryAlias(alias) = node
            && !matches!(
                alias.input.as_ref(),
                LogicalPlan::TableScan(_) | LogicalPlan::SubqueryAlias(_)
            )
        {
            *readers.entry(alias.alias.table().to_string()).or_default() += 1;
        }
        Ok(TreeNodeRecursion::Continue)
    });
    let mut merged = Vec::new();
    let _ = plan.apply(|node| {
        let LogicalPlan::Join(join) = node else {
            return Ok(TreeNodeRecursion::Continue);
        };
        for side in [join.left.as_ref(), join.right.as_ref()] {
            let LogicalPlan::SubqueryAlias(alias) = side else {
                continue;
            };
            let (name, body) = subquery_body(alias);
            let (mut base, mut calls_model) = match body {
                LogicalPlan::Projection(projection) => (
                    projection.input.as_ref(),
                    projection
                        .expr
                        .iter()
                        .any(|expr| expr.exists(|inner| Ok(is_ai_call(inner))).unwrap_or(false)),
                ),
                other => (other, false),
            };
            loop {
                base = match base {
                    LogicalPlan::Filter(filter) => filter.input.as_ref(),
                    LogicalPlan::Extension(_) => match as_node::<AiNode>(base) {
                        Some(node) => {
                            calls_model |= !node.is_predicate() || node.is_join();
                            node.inputs()[0]
                        }
                        None => break,
                    },
                    _ => break,
                };
            }
            let is_table = match base {
                LogicalPlan::SubqueryAlias(found) => {
                    let mut scanned = found.input.as_ref();
                    while let LogicalPlan::Filter(filter) = scanned {
                        scanned = filter.input.as_ref();
                    }
                    matches!(scanned, LogicalPlan::TableScan(_))
                }
                _ => false,
            };
            if is_table && readers.get(&name).copied().unwrap_or(0) <= 1 && !calls_model {
                merged.push(alias.alias.table().to_string());
                if name != alias.alias.table() {
                    merged.push(name);
                }
            }
        }
        Ok(TreeNodeRecursion::Continue)
    });
    merged
}
