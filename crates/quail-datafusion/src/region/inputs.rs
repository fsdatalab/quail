//! Prepare the table and subquery inputs that DataFusion computes.
//! Replace each input with a scan, and restore it after lowering.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result, TableReference};
use datafusion::datasource::{MemTable, provider_as_source, source_as_provider};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{
    Expr, Filter, LogicalPlan, LogicalPlanBuilder, Projection, SubqueryAlias, TableScan,
    TableScanBuilder,
};

use super::claim::{calls_ai, merged_aliases};
use super::identified::IdentifiedTable;
use super::shared::{ROW_COLUMN, row_field};
use crate::functions::is_quail_call;

/// Move each plain filter term about one table down to that table's
/// alias, where it becomes part of the table's input.
///
/// A plain term calls no model and has no subquery. It moves down only
/// through filters and inner joins, which keep a row whatever happens
/// to the other tables' rows; a term about two tables stays.
pub(super) fn push_plain_terms(plan: LogicalPlan) -> Result<LogicalPlan> {
    plan.transform_down(|node| {
        let LogicalPlan::Filter(filter) = &node else {
            return Ok(Transformed::no(node));
        };
        let mut kept = Vec::new();
        let mut moved: Vec<(TableReference, Expr)> = Vec::new();
        for term in split_conjunction(&filter.predicate) {
            match plain_table(term) {
                Some(table) if reaches_alias(&filter.input, &table) => {
                    moved.push((table, term.clone()))
                }
                _ => kept.push(term.clone()),
            }
        }
        if moved.is_empty() {
            return Ok(Transformed::no(node));
        }
        let mut input = filter.input.as_ref().clone();
        for (table, term) in moved {
            input = place_term(input, &table, term)?;
        }
        let placed = match conjunction(kept) {
            Some(predicate) => LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(input))?),
            None => input,
        };
        Ok(Transformed::yes(placed))
    })
    .map(|pushed| pushed.data)
}

/// Compute each plain expression of a region that reads one table as a
/// column of that table's input, so DataFusion evaluates it.
///
/// A plain expression is not a column, calls no model, aggregate,
/// window, or subquery, and reads columns of one table alias whose part
/// of the plan calls no model. The alias's part gains a projection with
/// the expression as `__quail_expr_<n>`, and the region reads that
/// column; a SELECT item keeps its name.
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
    // (table, expression, column name), in the order first seen
    let mut computed: Vec<(TableReference, Expr, String)> = Vec::new();
    let mut find = |expr: &Expr| {
        let _ = expr.apply(|inner| {
            let Some(table) = plain_table(inner) else {
                return Ok(TreeNodeRecursion::Continue);
            };
            // a struct field is a column path the region reads as it is
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
                // a whole plain term is the input's filter; only a model
                // call's arguments are computed
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
    // a filter's whole plain terms stay as written; they are the input's
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
        let joined = conjunction(terms).expect("a term");
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
                // a SELECT item keeps the name it had
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

/// The one table a plain term reads, or None for a term that calls a
/// model, has a subquery, or reads several tables or none.
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

/// Whether `plan` reaches the alias `table` through filters and inner
/// joins only, and the alias's subtree calls no model.
fn reaches_alias(plan: &LogicalPlan, table: &TableReference) -> bool {
    match plan {
        // a term moves only to a table DataFusion can compute as an input
        LogicalPlan::SubqueryAlias(alias) => {
            alias.alias == *table && !calls_ai(plan).unwrap_or(true)
        }
        LogicalPlan::Filter(filter) => reaches_alias(&filter.input, table),
        LogicalPlan::Join(join) if join.join_type == datafusion::logical_expr::JoinType::Inner => {
            reaches_alias(&join.left, table) || reaches_alias(&join.right, table)
        }
        _ => false,
    }
}

/// `plan` with `term` as a filter directly above the alias `table`.
fn place_term(plan: LogicalPlan, table: &TableReference, term: Expr) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::SubqueryAlias(ref alias) if alias.alias == *table => {
            Ok(LogicalPlan::Filter(Filter::try_new(term, Arc::new(plan))?))
        }
        LogicalPlan::Filter(filter) => {
            let input = place_term(filter.input.as_ref().clone(), table, term)?;
            Ok(LogicalPlan::Filter(Filter::try_new(
                filter.predicate,
                Arc::new(input),
            )?))
        }
        LogicalPlan::Join(join) => {
            let (left, right) = if reaches_alias(&join.left, table) {
                (
                    place_term(join.left.as_ref().clone(), table, term)?,
                    join.right.as_ref().clone(),
                )
            } else {
                (
                    join.left.as_ref().clone(),
                    place_term(join.right.as_ref().clone(), table, term)?,
                )
            };
            LogicalPlan::Join(join.clone())
                .with_new_exprs(LogicalPlan::Join(join).expressions(), vec![left, right])
        }
        other => Ok(other),
    }
}

fn is_subquery_expr(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Exists(_) | Expr::InSubquery(_) | Expr::ScalarSubquery(_)
    )
}

/// The one table a subtree reads, or None when it reads several or none.
fn single_scan(plan: &LogicalPlan) -> Option<TableScan> {
    let mut scans = Vec::new();
    plan.apply(|node| {
        // Shared rows have their own identity, not the source table's key.
        if crate::extension::as_node::<super::shared::SharedInput>(node).is_some() {
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

/// Whether a subtree is a table the runner reads by name: an identified
/// table's scan, alone or under its alias.
fn read_by_name(plan: &LogicalPlan) -> bool {
    let scan = match plan {
        LogicalPlan::TableScan(scan) => scan,
        LogicalPlan::SubqueryAlias(alias) => match alias.input.as_ref() {
            LogicalPlan::TableScan(scan) => scan,
            _ => return false,
        },
        _ => return false,
    };
    // a filter or limit DataFusion moved into the scan is the input's
    scan.filters.is_empty()
        && scan.fetch.is_none()
        && source_as_provider(&scan.source)
            .is_ok_and(|provider| provider.downcast_ref::<IdentifiedTable>().is_some())
}

/// The inputs a region's plan reads with [`ROW_COLUMN`], by name: those
/// without an id column of their own.
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

/// The schema of a placeholder for a region input: the input's columns,
/// then [`ROW_COLUMN`] when `numbered`, as [`super::shared::SharedInput`] returns them.
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

/// Whether DataFusion computes this part of a region as an input: it
/// calls no model, reads one table, and is more than an identified table
/// the runner reads by name.
fn plain_input(plan: &LogicalPlan) -> Result<Option<TableScan>> {
    // DataFusion computes a part's subqueries too, such as a filter
    // `stars > (SELECT AVG(stars) ...)` its optimizer moved to the table
    if read_by_name(plan) || calls_ai(plan)? {
        return Ok(None);
    }
    Ok(single_scan(plan))
}

/// Take out of a region's plan each part DataFusion computes as an
/// input: a part that calls no model and reads one table, such as a
/// table's plain filters, each CTE or FROM subquery that groups or
/// windows rows, and each one the region does not read through
/// ([`merged_aliases`]). The plan reads each input as a table of its name.
///
/// Returns the plan, each input's table name and plan, and the parts the
/// inputs replaced.
pub(super) fn split_inputs(plan: LogicalPlan) -> Result<Split> {
    let mut inputs: Vec<(String, LogicalPlan)> = Vec::new();
    let mut reads: Vec<Read> = Vec::new();
    let merged = merged_aliases(&plan);
    let mut root = true;
    let plan = plan
        .transform_down(|node| {
            // a subquery at the root is the region itself
            let at_root = std::mem::take(&mut root);
            let cut = if let Some(scan) = plain_input(&node)? {
                Some(cut_input(node.clone(), &scan, &mut inputs)?)
            } else if let LogicalPlan::TableScan(scan) = &node {
                Some(table_input(scan, &mut inputs)?)
            } else {
                None
            };
            if let Some(cut) = cut {
                if cut.transformed {
                    reads.push(Read {
                        read: cut.data.clone(),
                        original: node,
                        body: None,
                    });
                }
                return Ok(cut);
            }
            let LogicalPlan::SubqueryAlias(alias) = &node else {
                return Ok(Transformed::no(node));
            };
            // a subquery the region does not read through runs before the
            // query that reads it, as an input: its own model region when it
            // calls a model, or DataFusion's when it groups or windows
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
            // `FROM clauses a` reads the CTE `clauses` under the alias `a`;
            // every read of one CTE is one input, which runs once
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
            // a CTE's rows have no id column; the region numbers them
            let schema = placeholder_schema(body.schema().as_arrow(), true);
            let Ok(table) = MemTable::try_new(schema, vec![vec![]]) else {
                return Ok(Transformed::no(node));
            };
            let read =
                LogicalPlanBuilder::scan(name.as_str(), provider_as_source(Arc::new(table)), None)?
                    .alias(alias.alias.clone())?
                    .build()?;
            reads.push(Read {
                read: read.clone(),
                original: node.clone(),
                body: Some(name.clone()),
            });
            if !read_before {
                inputs.push((name, body));
            }
            Ok(Transformed::new(read, true, TreeNodeRecursion::Jump))
        })?
        .data;
    Ok((plan, inputs, reads))
}

/// A region's plan, each input's table name and plan, and the parts of
/// the region the inputs replaced, as [`split_inputs`] returns them.
pub(super) type Split = (LogicalPlan, Vec<(String, LogicalPlan)>, Vec<Read>);

/// A part of a region that [`split_inputs`] replaced with a scan of an
/// input.
#[derive(Debug)]
pub(super) struct Read {
    /// The scan that replaced it.
    read: LogicalPlan,
    /// The part it replaced.
    original: LogicalPlan,
    /// The input that is the body of a subquery it replaced, if it is one.
    body: Option<String>,
}

/// `plan` with each scan of an input that [`split_inputs`] made replaced
/// by the part it replaced again, a subquery over its claimed body.
pub(super) fn read_inputs(
    plan: LogicalPlan,
    reads: &[Read],
    claimed: &[(String, LogicalPlan)],
) -> Result<LogicalPlan> {
    plan.transform_down(|node| {
        let Some(found) = reads.iter().find(|read| read.read == node) else {
            return Ok(Transformed::no(node));
        };
        let body = found
            .body
            .as_ref()
            .and_then(|name| claimed.iter().find(|(input, _)| input == name));
        let original = match (body, &found.original) {
            (Some((_, body)), LogicalPlan::SubqueryAlias(alias)) => {
                let inner = match alias.input.as_ref() {
                    LogicalPlan::SubqueryAlias(cte) => LogicalPlan::SubqueryAlias(
                        SubqueryAlias::try_new(Arc::new(body.clone()), cte.alias.clone())?,
                    ),
                    _ => body.clone(),
                };
                LogicalPlan::SubqueryAlias(SubqueryAlias::try_new(
                    Arc::new(inner),
                    alias.alias.clone(),
                )?)
            }
            _ => found.original.clone(),
        };
        Ok(Transformed::new(original, true, TreeNodeRecursion::Jump))
    })
    .map(|done| done.data)
}

/// Replace `node`, a plain part of a region that reads `scan`'s table,
/// with a scan of an input DataFusion computes from it.
///
/// The input is named `<table>__<alias>`; its id column stays a primary
/// key when the part returns it.
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
    // the input's name ends with the alias after the last separator, which
    // an alias with the separator in it would make ambiguous; the region
    // then reads the table itself
    if qualifier
        .table()
        .contains(quail_planner::operators::INPUT_SEPARATOR)
    {
        return Ok(Transformed::no(node));
    }
    // not the table's own name: two aliases of one table read different rows
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
    // the region reads each document by its id, which DataFusion's
    // projection pruning drops when no operator reads it
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

/// `plan`, a part that reads one table, returning the table's column
/// `name` at `index` too: each scan reads it, and each `Projection`
/// passes it on.
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

/// A scan of a table without an identity, as a scan of a region input of
/// the table's name: DataFusion scans the table and passes its rows. A
/// table read twice, as in a self join, is one input.
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
    // a table without an id column has its rows numbered
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
