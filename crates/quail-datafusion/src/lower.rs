//! The `LowerAiCalls` analyzer rule: AI calls as Quail's plan nodes.
//!
//! DataFusion binds a query's AI calls as scalar functions inside its
//! own operators. The rule rewrites each model region so the model's
//! work is in plan nodes of its own, and DataFusion's operators hold
//! only the relational work:
//!
//! - A `Filter` keeps its other terms; each `AI.IF` term, or comparison
//!   of an `AI.SCORE` call, goes into a [`SemanticFilter`] above it, or
//!   a [`SemanticJoin`] when the term reads two tables.
//! - An inner `Join` keeps its other ON terms; each `AI.IF` term goes
//!   into a [`SemanticJoin`] above it.
//! - A `Projection` of `AI.CLASSIFY(...) AS name`, or of `AI.EXTRACT`,
//!   `AI.MAP`, or `AI.SAE_FEATURES`, reads the column `name` that a
//!   [`SemanticColumn`] below it adds.
//!
//! An EXISTS subquery, a label test, and a call anywhere else stay as
//! DataFusion bound them.

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{
    Expr, ExprSchemable, Extension, Filter, Join, JoinType, LogicalPlan, Operator, Projection,
    UserDefinedLogicalNode,
};
use datafusion::optimizer::AnalyzerRule;

use crate::region::ModelRegion;
use crate::semantic::{SemanticColumn, SemanticFilter, SemanticJoin, Written};

/// Rewrite each model region's AI calls into Quail's plan nodes, as the
/// module explains.
#[derive(Debug, Default)]
pub struct LowerAiCalls;

impl AnalyzerRule for LowerAiCalls {
    fn name(&self) -> &str {
        "lower_ai_calls"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        Ok(plan
            .transform_up(|node| {
                if let LogicalPlan::Extension(extension) = &node
                    && let Some(region) = extension.node.as_any().downcast_ref::<ModelRegion>()
                {
                    let mut region = region.clone();
                    region.plan = lower_ai_calls(region.bound.clone())?;
                    return Ok(Transformed::yes(extension_node(region)));
                }
                Ok(Transformed::no(node))
            })?
            .data)
    }
}

fn extension_node(node: impl UserDefinedLogicalNode + 'static) -> LogicalPlan {
    LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    })
}

fn function_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::ScalarFunction(call) => Some(call.name()),
        _ => None,
    }
}

/// The term inside a `HINT`, or the term itself.
fn hinted(expr: &Expr) -> &Expr {
    match expr {
        Expr::ScalarFunction(call) if call.name() == "hint" && !call.args.is_empty() => {
            hinted(&call.args[0])
        }
        other => other,
    }
}

/// Whether a term is a predicate the model answers: an `AI.IF` call or a
/// comparison of an `AI.SCORE` call.
fn is_asked(term: &Expr) -> bool {
    match hinted(term) {
        Expr::BinaryExpr(binary)
            if matches!(
                binary.op,
                Operator::Eq
                    | Operator::NotEq
                    | Operator::Lt
                    | Operator::LtEq
                    | Operator::Gt
                    | Operator::GtEq
            ) =>
        {
            function_name(&binary.left) == Some("ai_score")
                || function_name(&binary.right) == Some("ai_score")
        }
        other => function_name(other) == Some("ai_filter"),
    }
}

/// The tables a term reads.
fn tables(term: &Expr) -> BTreeSet<String> {
    term.column_refs()
        .into_iter()
        .filter_map(|column| column.relation.as_ref().map(ToString::to_string))
        .collect()
}

fn is_column_call(expr: &Expr) -> bool {
    matches!(
        function_name(expr),
        Some("ai_classify" | "ai_extract" | "ai_map" | "ai_sae_features")
    )
}

/// A predicate's asked terms with their positions, and the rest, each
/// in written order, and how many terms it has.
fn split_asked(predicate: &Expr) -> (Vec<(usize, Expr)>, Vec<Expr>, usize) {
    let terms = split_conjunction(predicate);
    let num_terms = terms.len();
    let (mut asked, mut rest) = (Vec::new(), Vec::new());
    for (position, term) in terms.into_iter().enumerate() {
        if is_asked(term) {
            asked.push((position, term.clone()));
        } else {
            rest.push(term.clone());
        }
    }
    (asked, rest, num_terms)
}

/// Stack one node per asked term on `input`, merging the one-table
/// terms of a run on the same table into one `SemanticFilter`.
fn stack_asked(
    mut input: LogicalPlan,
    asked: Vec<(usize, Expr)>,
    num_terms: usize,
    in_on: bool,
) -> LogicalPlan {
    let mut run: Vec<(usize, Expr)> = Vec::new();
    let mut run_table: Option<BTreeSet<String>> = None;
    let close = |input: LogicalPlan, run: &mut Vec<(usize, Expr)>| {
        if run.is_empty() {
            return input;
        }
        let (positions, predicates) = std::mem::take(run).into_iter().unzip();
        extension_node(SemanticFilter {
            input: Arc::new(input),
            predicates,
            stop_key: Vec::new(),
            written: Written {
                positions,
                num_terms,
            },
        })
    };
    for (position, term) in asked {
        let read = tables(&term);
        if !in_on && read.len() < 2 {
            if run_table.as_ref() != Some(&read) {
                input = close(input, &mut run);
                run_table = Some(read);
            }
            run.push((position, term));
            continue;
        }
        input = close(input, &mut run);
        run_table = None;
        input = extension_node(SemanticJoin {
            input: Arc::new(input),
            predicate: term,
            semantics: "full".to_string(),
            in_on,
            written: Written {
                positions: vec![position],
                num_terms,
            },
        });
    }
    close(input, &mut run)
}

fn lower_filter(filter: &Filter) -> Result<Transformed<LogicalPlan>> {
    let unchanged = || Ok(Transformed::no(LogicalPlan::Filter(filter.clone())));
    let (asked, rest, num_terms) = split_asked(&filter.predicate);
    if asked.is_empty() {
        return unchanged();
    }
    let input = match conjunction(rest) {
        // a term DataFusion refuses keeps its operator, for DataFusion to refuse
        Some(predicate) => match Filter::try_new(predicate, Arc::clone(&filter.input)) {
            Ok(kept) => LogicalPlan::Filter(kept),
            Err(_) => return unchanged(),
        },
        None => filter.input.as_ref().clone(),
    };
    Ok(Transformed::yes(stack_asked(
        input, asked, num_terms, false,
    )))
}

fn lower_join(join: &Join) -> Result<Transformed<LogicalPlan>> {
    let Some(predicate) = &join.filter else {
        return Ok(Transformed::no(LogicalPlan::Join(join.clone())));
    };
    let (asked, rest, num_terms) = split_asked(predicate);
    if asked.is_empty() || join.join_type != JoinType::Inner {
        return Ok(Transformed::no(LogicalPlan::Join(join.clone())));
    }
    let mut kept = join.clone();
    kept.filter = conjunction(rest);
    Ok(Transformed::yes(stack_asked(
        LogicalPlan::Join(kept),
        asked,
        num_terms,
        true,
    )))
}

fn lower_projection(projection: &Projection) -> Result<Transformed<LogicalPlan>> {
    let mut input = projection.input.as_ref().clone();
    let mut exprs = Vec::new();
    let mut changed = false;
    for expr in &projection.expr {
        if let Expr::Alias(alias) = expr
            && alias.relation.is_none()
            && alias.metadata.is_none()
            && is_column_call(&alias.expr)
            && !input.schema().has_column_with_unqualified_name(&alias.name)
        {
            let (_, field) = alias.expr.to_field(input.schema())?;
            let field = field.as_ref().clone().with_name(alias.name.clone());
            input = extension_node(SemanticColumn::try_new(
                Arc::new(input),
                alias.expr.as_ref().clone(),
                alias.name.clone(),
                vec![field],
            )?);
            exprs.push(Expr::Column(Column::from_name(alias.name.clone())));
            changed = true;
        } else {
            exprs.push(expr.clone());
        }
    }
    if !changed {
        return Ok(Transformed::no(LogicalPlan::Projection(projection.clone())));
    }
    Ok(Transformed::yes(LogicalPlan::Projection(
        Projection::try_new(exprs, Arc::new(input))?,
    )))
}

/// Rewrite a region's AI calls into Quail's plan nodes, as the module
/// explains.
///
/// # Errors
///
/// A rewritten operator's schema does not fit DataFusion's rules.
pub fn lower_ai_calls(plan: LogicalPlan) -> Result<LogicalPlan> {
    Ok(plan
        .transform_up(|node| match &node {
            LogicalPlan::Filter(filter) => lower_filter(filter),
            LogicalPlan::Join(join) => lower_join(join),
            LogicalPlan::Projection(projection) => lower_projection(projection),
            _ => Ok(Transformed::no(node)),
        })?
        .data)
}
