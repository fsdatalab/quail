use std::collections::HashSet;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result, TableReference, internal_err};
use datafusion::logical_expr::{Expr, Join, JoinType, LogicalPlan, LogicalPlanBuilder, Projection};

use crate::functions::is_ai_call;
use crate::plan::extension_node;
use crate::shared::{ROW_COLUMN, SharedInput};

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
    let matches = extension_node(SharedInput::try_new(matches, false)?);
    let matches = LogicalPlanBuilder::from(matches)
        .alias(pairs_alias.clone())?
        .build()?;
    let matched_left = Column::new(Some(pairs_alias.clone()), left_id);
    let matched_right = Column::new(Some(pairs_alias), right_id);

    let restored = match join.join_type {
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
