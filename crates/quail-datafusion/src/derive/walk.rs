//! Read the plan in input order and track each operator's source tables.
//! Read each derived table's column as the expression that defines it.

use super::binding::unsupported;
use crate::ai::{AiNode, Fetch, Semantics};
use crate::extension::as_node;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::{Expr, Join, JoinType, LogicalPlan};
use quail_planner::error::CompileError;
use std::collections::BTreeSet;

/// `plan` with a SELECT list at its root: DataFusion removes one that
/// returns its input's columns as they are.
pub(super) fn with_select_list(plan: &LogicalPlan) -> LogicalPlan {
    if matches!(plan, LogicalPlan::Projection(_)) {
        return plan.clone();
    }
    LogicalPlan::Projection(datafusion::logical_expr::Projection::new_from_schema(
        std::sync::Arc::new(plan.clone()),
        std::sync::Arc::clone(plan.schema()),
    ))
}

// ---------------------------------------------------------------- the SELECT

/// The SELECT list of a region and the relation below it.
pub(super) struct Top<'a> {
    pub(super) projection: Vec<Expr>,
    pub(super) relation: &'a LogicalPlan,
}

/// Read the SELECT list at a region's root.
///
/// # Errors
///
/// The region has no SELECT list at its root, such as a LIMIT, ORDER
/// BY, or DISTINCT, which DataFusion runs above the region.
pub(super) fn read_top(plan: &LogicalPlan) -> Result<Top<'_>, CompileError> {
    let LogicalPlan::Projection(projection) = plan else {
        return unsupported("A model region needs a SELECT list.");
    };
    Ok(Top {
        projection: projection.expr.clone(),
        relation: projection.input.as_ref(),
    })
}

/// The global fetch of the region's last `AiNode`, which a LIMIT became
/// (`Derive`); None, in the outer Option, when another
/// node has one, as the region's operators cannot place a LIMIT there.
pub(super) fn read_fetch(top: &Top<'_>) -> Option<Option<i64>> {
    let mut last = top.relation;
    while let LogicalPlan::Projection(_) | LogicalPlan::SubqueryAlias(_) = last {
        last = last.inputs()[0];
    }
    let last = as_node::<AiNode>(last);
    let mut fetch = None;
    let mut elsewhere = false;
    let _ = top.relation.apply(|plan| {
        if let Some(node) = as_node::<AiNode>(plan)
            && let Some(Fetch::Global(rows)) = &node.fetch
        {
            match last {
                Some(found) if std::ptr::eq(found, node) => fetch = Some(*rows as i64),
                _ => elsewhere = true,
            }
        }
        Ok(TreeNodeRecursion::Continue)
    });
    (!elsewhere).then_some(fetch)
}

/// A table the region scans.
#[derive(Debug, Clone)]
pub(super) struct Table {
    pub(super) alias: String,
    pub(super) provider: String,
    pub(super) schema: SchemaRef,
    /// Whether an EXISTS gate reads it, rather than FROM or JOIN.
    pub(super) gate: bool,
}

/// What one operator of the plan is.
#[derive(Debug)]
pub(super) enum Kind<'a> {
    Scan(usize),
    Join(&'a Join),
    Ai(&'a AiNode),
    /// A projection, or a SubqueryAlias of a derived table.
    PassThrough,
}

/// One operator of the plan, as the walk meets it.
#[derive(Debug)]
pub(super) struct Visit<'a> {
    pub(super) kind: Kind<'a>,
    /// The ids of its inputs.
    pub(super) inputs: Vec<usize>,
    /// The id of the operator it is the input of.
    pub(super) parent: Option<usize>,
    /// The aliases of the tables it reads, in scan order.
    pub(super) tables: Vec<String>,
    /// Whether an AI join or gate is at or below it.
    pub(super) ai_join: bool,
}

/// The region's operators, children first, and its tables.
#[derive(Debug, Default)]
pub(super) struct Walk<'a> {
    /// Each operator by its id.
    visits: Vec<Option<Visit<'a>>>,
    /// The ids, children first.
    order: Vec<usize>,
    pub(super) tables: Vec<Table>,
    /// (alias, its columns' expressions) of each derived table the
    /// binder merges into the query that reads it.
    merged: Vec<(String, Vec<(String, Expr)>)>,
    /// The binder's refusal of the FROM and JOIN shape, if any.
    pub(super) refusal: Option<CompileError>,
}

impl<'a> Walk<'a> {
    pub(super) fn visit(&mut self, plan: &'a LogicalPlan) -> usize {
        let id = self.visits.len();
        self.visits.push(None);
        let (kind, inputs) = match plan {
            LogicalPlan::Projection(projection) => {
                (Kind::PassThrough, vec![self.visit(&projection.input)])
            }
            LogicalPlan::Join(join) => {
                if join.join_type != JoinType::Inner {
                    self.refuse(unsupported(&format!(
                        "A {} JOIN must run in DataFusion.",
                        join.join_type
                    )));
                }
                let left = self.visit(&join.left);
                let right = self.visit(&join.right);
                (Kind::Join(join), vec![left, right])
            }
            LogicalPlan::SubqueryAlias(alias) => match self.table(plan) {
                Some(table) => (Kind::Scan(table), Vec::new()),
                None => {
                    let definitions = derived_columns(&alias.input);
                    let input = self.visit(&alias.input);
                    self.merged
                        .push((alias.alias.table().to_string(), definitions));
                    (Kind::PassThrough, vec![input])
                }
            },
            LogicalPlan::TableScan(_) => match self.table(plan) {
                Some(table) => (Kind::Scan(table), Vec::new()),
                None => unreachable!("a table scan reads a table"),
            },
            other => match as_node::<AiNode>(other) {
                Some(node) => (Kind::Ai(node), vec![self.visit(&node.input)]),
                // the claim puts no other operator in a region
                None => {
                    self.refuse(unsupported(
                        "These operators must run outside the model region.",
                    ));
                    (Kind::PassThrough, Vec::new())
                }
            },
        };
        for input in &inputs {
            if let Some(visit) = self.visits[*input].as_mut() {
                visit.parent = Some(id);
            }
        }
        let mut tables: Vec<String> = Vec::new();
        let mut ai_join = false;
        for input in &inputs {
            let below = self.visits[*input].as_ref().expect("a visited input");
            tables.extend(below.tables.iter().cloned());
            ai_join |= below.ai_join;
        }
        if let Kind::Scan(table) = &kind {
            tables.push(self.tables[*table].alias.clone());
        }
        if let Kind::Ai(node) = &kind {
            ai_join |= node.semantics != Semantics::Rows
                || (node.is_predicate() && self.call_tables(&node.call).len() > 1);
            if node.semantics != Semantics::Rows
                && let Kind::Join(_) = &self.get(inputs[0]).kind
            {
                // the gate's table is the right input of its join
                let gated = self.get(self.get(inputs[0]).inputs[1]).tables.clone();
                for table in &mut self.tables {
                    if gated.contains(&table.alias) {
                        table.gate = true;
                    }
                }
            }
        }
        if let Kind::Join(_) = &kind
            && self.get(inputs[1]).tables.len() != 1
        {
            self.refuse(unsupported(
                "A model join needs one registered provider on its right side.",
            ));
        }
        self.visits[id] = Some(Visit {
            kind,
            inputs,
            parent: None,
            tables,
            ai_join,
        });
        self.order.push(id);
        id
    }

    fn refuse(&mut self, refusal: Result<(), CompileError>) {
        if self.refusal.is_none() {
            self.refusal = refusal.err();
        }
    }

    /// Record the table a scan reads; None for a derived table.
    fn table(&mut self, plan: &'a LogicalPlan) -> Option<usize> {
        let (alias, below) = match plan {
            LogicalPlan::SubqueryAlias(alias) => {
                (alias.alias.table().to_string(), alias.input.as_ref())
            }
            other => (String::new(), other),
        };
        let LogicalPlan::TableScan(scan) = below else {
            return None;
        };
        let provider = scan.table_name.table().to_string();
        let alias = if alias.is_empty() {
            provider.clone()
        } else {
            alias
        };
        let index = self.tables.len();
        self.tables.push(Table {
            alias: alias.clone(),
            provider: provider.clone(),
            schema: scan.source.schema(),
            gate: false,
        });
        Some(index)
    }

    /// Whether an EXISTS gate reads the table `alias`.
    pub(super) fn is_gate(&self, alias: &str) -> bool {
        self.tables
            .iter()
            .any(|table| table.alias == alias && table.gate)
    }

    pub(super) fn get(&self, id: usize) -> &Visit<'a> {
        self.visits[id].as_ref().expect("a visited operator")
    }

    /// The ids of the operators, children first.
    pub(super) fn ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.order.iter().copied()
    }

    /// The tables an expression reads, through the derived tables merged
    /// into the query.
    pub(super) fn call_tables(&self, expr: &Expr) -> BTreeSet<String> {
        self.through(expr)
            .column_refs()
            .into_iter()
            .filter_map(|column| {
                column
                    .relation
                    .as_ref()
                    .map(|table| table.table().to_string())
            })
            .collect()
    }

    /// `expr` with each merged derived table's column as the expression
    /// that defines it, so it reads the tables' columns and the answer
    /// columns the `AiNode`s add.
    pub(super) fn through(&self, expr: &Expr) -> Expr {
        expr.clone()
            .transform_up(|node| {
                let Expr::Column(column) = &node else {
                    return Ok(Transformed::no(node));
                };
                if let Some(relation) = &column.relation
                    && let Some((_, columns)) = self
                        .merged
                        .iter()
                        .rev()
                        .find(|(alias, _)| alias == relation.table())
                    && let Some((_, definition)) =
                        columns.iter().find(|(name, _)| *name == column.name)
                {
                    return Ok(Transformed::yes(self.through(definition)));
                }
                Ok(Transformed::no(node))
            })
            .map(|found| found.data)
            .unwrap_or_else(|_| expr.clone())
    }

    /// A SELECT list item read through the derived tables, under the
    /// name the region returns it by.
    pub(super) fn selected(&self, expr: &Expr) -> Expr {
        let (inner, name) = match expr {
            Expr::Alias(alias) => (alias.expr.as_ref(), Some(alias.name.clone())),
            other => (other, None),
        };
        let read = self.through(inner);
        if read == *inner {
            return expr.clone();
        }
        let name = name.unwrap_or_else(|| inner.qualified_name().1);
        match &read {
            Expr::Column(column) if column.name == name && !matches!(expr, Expr::Alias(_)) => read,
            _ => read.alias(name),
        }
    }
}

/// Each output column of a derived table and its expression: a SELECT
/// list's, or, where DataFusion removed one that passes its input on,
/// the input's columns.
fn derived_columns(plan: &LogicalPlan) -> Vec<(String, Expr)> {
    match plan {
        LogicalPlan::Projection(projection) => projection
            .expr
            .iter()
            .map(|expr| (expr.qualified_name().1, expr.clone().unalias()))
            .collect(),
        _ => plan
            .schema()
            .columns()
            .into_iter()
            .map(|column| (column.name.clone(), Expr::Column(column)))
            .collect(),
    }
}
