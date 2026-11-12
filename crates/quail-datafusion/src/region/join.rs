//! Restore outer AI joins from matching pairs and positional row identities.

use std::collections::HashSet;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result, TableReference, internal_err};
use datafusion::logical_expr::{Expr, Join, JoinType, LogicalPlan, LogicalPlanBuilder, Projection};

use super::shared::{ROW_COLUMN, SharedInput};
use crate::extension::extension_node;
use crate::functions::is_ai_call;

/// Lower an outer AI ON predicate through an inner matching relation.
///
/// `lower` recursively lowers the inputs and the inner matching relation. The
/// latter may use the general expression fallback; only restoring unmatched
/// rows needs this special rewrite. Inner joins and joins without AI in ON are
/// left unchanged. Call this only after the existing lowering path declines.
pub(super) fn lower_outer_join(
    plan: &LogicalPlan,
    number: usize,
    mut lower: impl FnMut(LogicalPlan) -> Result<LogicalPlan>,
) -> Result<Option<LogicalPlan>> {
    let LogicalPlan::Join(join) = plan else {
        return Ok(None);
    };
    if join.join_type == JoinType::Inner {
        return Ok(None);
    }
    let mut has_ai = false;
    for expr in plan.expressions() {
        has_ai |= expr.exists(|inner| Ok(is_ai_call(inner)))?;
    }
    if !has_ai {
        return Ok(None);
    }

    let left_input = lower(join.left.as_ref().clone())?;
    let right_input = lower(join.right.as_ref().clone())?;
    let mut names = Names::default();
    for input in [plan, &left_input, &right_input] {
        names.reserve(input)?;
    }
    let left = Side::new(
        left_input,
        names.fresh(&format!("__quail_outer_l{number}")),
        &mut names,
    )?;
    let right = Side::new(
        right_input,
        names.fresh(&format!("__quail_outer_r{number}")),
        &mut names,
    )?;
    let left_id = names.fresh("__quail_outer_left_id");
    let right_id = names.fresh("__quail_outer_right_id");
    let pairs_alias = TableReference::bare(names.fresh(&format!("__quail_outer_pairs{number}")));
    let rewrite = |expr: Expr| -> Result<Expr> {
        Ok(expr
            .transform_up(|expr| match expr {
                Expr::Column(column) => {
                    let Some(read) = left.column(&column).or_else(|| right.column(&column)) else {
                        return internal_err!("outer AI join cannot resolve column {column}");
                    };
                    Ok(Transformed::yes(Expr::Column(read)))
                }
                other => Ok(Transformed::no(other)),
            })?
            .data)
    };
    let on = join
        .on
        .iter()
        .map(|(a, b)| Ok((rewrite(a.clone())?, rewrite(b.clone())?)))
        .collect::<Result<Vec<_>>>()?;
    let filter = join.filter.clone().map(rewrite).transpose()?;
    let matches = LogicalPlan::Join(Join::try_new(
        Arc::new(left.rows.clone()),
        Arc::new(right.rows.clone()),
        on,
        filter,
        JoinType::Inner,
        join.join_constraint,
        join.null_equality,
        false,
    )?);
    let pairs = LogicalPlanBuilder::from(matches)
        .project(vec![
            Expr::Column(left.id()).alias(&left_id),
            Expr::Column(right.id()).alias(&right_id),
        ])?
        .build()?;
    let matches = lower(pairs)?;
    // The source aliases also occur in the restoring joins. Keep them behind
    // the pair projection so optimizer inlining cannot expose them twice.
    let matches = extension_node(SharedInput::try_new(matches, false)?);
    let matches = LogicalPlanBuilder::from(matches)
        .alias(pairs_alias.clone())?
        .build()?;
    let matched_left = Column::new(Some(pairs_alias.clone()), left_id);
    let matched_right = Column::new(Some(pairs_alias), right_id);

    // Each occurrence has a non-null positional id, even when its content id
    // is duplicated or NULL. The first join restores left-only rows; their
    // NULL pair-right id cannot match a right row in the second join. RIGHT
    // needs no left-only rows, whereas FULL preserves both unmatched sides.
    let restored = match join.join_type {
        // a semi, anti, or mark join returns one side's rows, each kept,
        // dropped, or marked by whether it has a matching pair
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => {
            LogicalPlanBuilder::from(left.rows.clone())
                .join(
                    matches,
                    join.join_type,
                    (vec![left.id()], vec![matched_left]),
                    None,
                )?
                .build()?
        }
        JoinType::RightSemi | JoinType::RightAnti | JoinType::RightMark => {
            LogicalPlanBuilder::from(matches)
                .join(
                    right.rows.clone(),
                    join.join_type,
                    (vec![matched_right], vec![right.id()]),
                    None,
                )?
                .build()?
        }
        _ => {
            let first = if join.join_type == JoinType::Right {
                JoinType::Inner
            } else {
                JoinType::Left
            };
            LogicalPlanBuilder::from(left.rows.clone())
                .join(matches, first, (vec![left.id()], vec![matched_left]), None)?
                .join(
                    right.rows.clone(),
                    join.join_type,
                    (vec![matched_right], vec![right.id()]),
                    None,
                )?
                .build()?
        }
    };
    let exprs = join
        .schema
        .columns()
        .into_iter()
        .map(|column| {
            let read = match left.column(&column).or_else(|| right.column(&column)) {
                Some(read) => read,
                // a mark join's mark is the restoring join's own, under the
                // restoring join's qualifier; every other column is renamed
                None => match restored
                    .schema()
                    .columns()
                    .into_iter()
                    .find(|found| found.name == column.name)
                {
                    Some(found) => found,
                    None => {
                        return internal_err!("outer AI join cannot restore column {column}");
                    }
                },
            };
            Ok(Expr::Column(read).alias_qualified(column.relation, column.name))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(LogicalPlan::Projection(
        Projection::try_new_with_schema(exprs, Arc::new(restored), Arc::clone(&join.schema))?,
    )))
}

struct Side {
    alias: TableReference,
    columns: Vec<Column>,
    renamed: Vec<String>,
    id: String,
    rows: LogicalPlan,
}

impl Side {
    fn new(input: LogicalPlan, name: String, names: &mut Names) -> Result<Self> {
        let columns = input.schema().columns();
        // A side can itself be a join with duplicate names, or already contain
        // __quail_row. Rename every column before adding a positional identity.
        let renamed: Vec<_> = columns
            .iter()
            .map(|_| names.fresh("__quail_outer_column"))
            .collect();
        let exprs = columns
            .iter()
            .zip(&renamed)
            .map(|(column, name)| Expr::Column(column.clone()).alias(name))
            .collect();
        let input = LogicalPlan::Projection(Projection::try_new(exprs, Arc::new(input))?);
        let numbered = extension_node(SharedInput::try_new(input, true)?);
        let mut exprs: Vec<_> = renamed
            .iter()
            .map(|name| Expr::Column(Column::from_name(name)))
            .collect();
        let id = names.fresh("__quail_outer_id");
        exprs.push(Expr::Column(Column::from_name(ROW_COLUMN)).alias(&id));
        let alias = TableReference::bare(name);
        let renamed_rows = LogicalPlanBuilder::from(numbered).project(exprs)?.build()?;
        // A staged region input gets its own __quail_row. Keep the old name
        // hidden behind an optimizer boundary after renaming the identity.
        let shared = extension_node(SharedInput::try_new(renamed_rows, false)?);
        let rows = LogicalPlanBuilder::from(shared)
            .alias(alias.clone())?
            .build()?;
        Ok(Self {
            alias,
            columns,
            renamed,
            id,
            rows,
        })
    }

    fn column(&self, column: &Column) -> Option<Column> {
        self.columns
            .iter()
            .position(|found| found == column)
            .map(|index| Column::new(Some(self.alias.clone()), self.renamed[index].clone()))
    }

    fn id(&self) -> Column {
        Column::new(Some(self.alias.clone()), self.id.clone())
    }
}

/// Internal aliases and columns must not shadow user names, including names
/// behind a projection that optimizer rules might expose later.
#[derive(Default)]
struct Names {
    used: HashSet<String>,
}

impl Names {
    fn reserve(&mut self, plan: &LogicalPlan) -> Result<()> {
        plan.apply(|node| {
            for (qualifier, field) in node.schema().iter() {
                self.used.insert(field.name().clone());
                if let Some(qualifier) = qualifier {
                    self.used.insert(qualifier.to_string());
                    self.used.insert(qualifier.table().to_owned());
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(())
    }

    fn fresh(&mut self, base: &str) -> String {
        let mut candidate = base.to_owned();
        let mut suffix = 0;
        while !self.used.insert(candidate.clone()) {
            suffix += 1;
            candidate = format!("{base}_{suffix}");
        }
        candidate
    }
}
