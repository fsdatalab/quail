//! The `PushDownProjectionIntoAiNodes` optimizer rule: an `AiNode` returns
//! only the added columns the projection above it reads, and an
//! extraction asks only the fields the query reads.
//!
//! An `AiNode`'s projection `P` names the added columns it returns. A
//! projection `Q` above the node that reads only some of them needs only
//! those:
//!
//! ```text
//! π_Q(AI(R; P)) ⇒ π_Q(AI(R; P = Q[P]))
//! ```
//!
//! `Q[P]` is the columns of `P` that `Q` reads. For example, a SELECT
//! list over a FROM subquery that returns an extraction reads the
//! extraction's answer but not its spans: the node then returns the
//! answer alone. The node always returns its answer: a column node's
//! answer is the reason the model is asked. The node's filter and tests
//! still read every added column, as `P` applies after them.
//!
//! An `AI.EXTRACT` asks one question per field of its answer `a`, each
//! in a prompt of its own, so leaving out a field changes no other
//! field's answer. When every use of `a` and its spans in the plan reads
//! named fields, `a.f` or `a_span.f`, the call asks only those fields:
//!
//! ```text
//! π_{a.f}(AI_{extract(d, {f, g}) → a}(R)) ⇒ π_{a.f}(AI_{extract(d, {f}) → a}(R))
//! ```
//!
//! A use is any expression of any operator above, the node's own filter
//! and tests included, followed through projections that pass the column
//! on or rename it. A use that reads `a` whole, such as `SELECT a`, a
//! comparison of `a`, or the query's own result columns, keeps every
//! field; so does an operator whose columns the rule does not follow by
//! name, such as a `UNION` or an `UNNEST`.
//!
//! DataFusion's `OptimizeProjections` prunes the columns a node reads
//! from its input (`AiNode::necessary_children_exprs`), but not the
//! columns the node adds or the fields of a struct column.

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{DFSchema, Result};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::{
    Expr, ExprSchemable, LogicalPlan, Projection, UserDefinedLogicalNodeCore,
};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::SPAN_SUFFIX;
use crate::ai::{AiNode, Fetch, Semantics};
use crate::derive::unfolded;
use crate::extension::{as_node, extension_node};
use crate::functions::function_name;
use crate::rules::lower::quail_function;

/// Make each `AiNode` under a projection return only the added columns
/// the projection reads, and each extraction ask only the fields the
/// query reads, as the module explains.
#[derive(Debug, Default)]
pub struct PushDownProjectionIntoAiNodes;

impl OptimizerRule for PushDownProjectionIntoAiNodes {
    fn name(&self) -> &str {
        "push_down_projection_into_ai_nodes"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let pushed = plan.transform_up(push_down)?;
        let transformed = pushed.transformed;
        let mut pruned = prune_extract_fields(pushed.data)?;
        pruned.transformed |= transformed;
        Ok(pruned)
    }
}

fn push_down(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Projection(projection) = &plan else {
        return Ok(Transformed::no(plan));
    };
    let Some(node) = as_node::<AiNode>(&projection.input) else {
        return Ok(Transformed::no(plan));
    };
    if node.semantics != Semantics::Rows || node.is_predicate() {
        return Ok(Transformed::no(plan));
    }
    let read: Vec<String> = projection
        .expr
        .iter()
        .flat_map(|expr| expr.column_refs())
        .filter(|column| column.relation.is_none())
        .map(|column| column.name.clone())
        .collect();
    let returned: Vec<usize> = match &node.projection {
        Some(positions) => positions.clone(),
        None => (0..node.fields.len()).collect(),
    };
    let kept: Vec<usize> = returned
        .iter()
        .copied()
        .filter(|index| *index == 0 || read.contains(node.fields[*index].name()))
        .collect();
    if kept.len() == returned.len() {
        return Ok(Transformed::no(plan));
    }
    let mut pruned = node.clone();
    pruned.projection = Some(kept);
    let input = extension_node(pruned.checked()?);
    Ok(Transformed::yes(LogicalPlan::Projection(
        Projection::try_new(projection.expr.clone(), Arc::new(input))?,
    )))
}

/// `plan` with each extraction asking only the fields the plan reads,
/// as the module explains.
fn prune_extract_fields(mut plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let mut changed = false;
    for (call, answers) in extractions(&plan)? {
        let Some(fields) = question_fields(&call) else {
            continue;
        };
        let Some(read) = read_fields(&plan, &answers)? else {
            continue;
        };
        let kept: Vec<usize> = fields
            .iter()
            .enumerate()
            .filter(|(_, field)| read.contains(*field))
            .map(|(index, _)| index)
            .collect();
        if kept.is_empty() || kept.len() == fields.len() {
            continue;
        }
        let pruned = with_fields(&call, &kept);
        plan = replace_call(plan, &call, &pruned)?;
        changed = true;
    }
    Ok(Transformed::new_transformed(plan, changed))
}

/// Each extraction call of the plan's `AiNode`s, its literals unfolded,
/// with the answer columns of the nodes that make it.
fn extractions(plan: &LogicalPlan) -> Result<Vec<(Expr, Vec<String>)>> {
    let mut found: Vec<(Expr, Vec<String>)> = Vec::new();
    plan.apply_with_subqueries(|node| {
        if let Some(ai) = as_node::<AiNode>(node)
            && ai.function() == "ai_extract"
            && ai.semantics == Semantics::Rows
        {
            let call = unfolded(ai.call.clone());
            match found.iter_mut().find(|(known, _)| *known == call) {
                Some((_, answers)) => answers.push(ai.answer.clone()),
                None => found.push((call, vec![ai.answer.clone()])),
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(found)
}

/// The field names of an unfolded extraction call's questions argument,
/// in order; None when it is not a `named_struct` of named fields.
fn question_fields(call: &Expr) -> Option<Vec<String>> {
    let Expr::ScalarFunction(found) = call else {
        return None;
    };
    let Some(Expr::ScalarFunction(questions)) = found.args.get(1) else {
        return None;
    };
    if questions.name() != "named_struct" || questions.args.len() % 2 != 0 {
        return None;
    }
    questions
        .args
        .chunks(2)
        .map(|pair| match &pair[0] {
            Expr::Literal(value, _) => value.try_as_str().flatten().map(ToString::to_string),
            _ => None,
        })
        .collect()
}

/// An unfolded extraction call asking only the questions at `kept`.
fn with_fields(call: &Expr, kept: &[usize]) -> Expr {
    let Expr::ScalarFunction(found) = call else {
        unreachable!("an extraction call")
    };
    let Some(Expr::ScalarFunction(questions)) = found.args.get(1) else {
        unreachable!("a questions argument")
    };
    let pairs: Vec<Expr> = kept
        .iter()
        .flat_map(|index| questions.args[2 * index..2 * index + 2].to_vec())
        .collect();
    let mut args = found.args.clone();
    args[1] = Expr::ScalarFunction(ScalarFunction::new_udf(Arc::clone(&questions.func), pairs));
    Expr::ScalarFunction(ScalarFunction::new_udf(Arc::clone(&found.func), args))
}

/// The fields of the columns named `answers` and their spans that the
/// plan reads; None when it reads one of them whole.
fn read_fields(plan: &LogicalPlan, answers: &[String]) -> Result<Option<BTreeSet<String>>> {
    let mut names: Vec<String> = answers
        .iter()
        .flat_map(|answer| [answer.clone(), format!("{answer}{SPAN_SUFFIX}")])
        .collect();
    // a projection that renames a column passes it on under the new name
    loop {
        let mut renamed = Vec::new();
        plan.apply_with_subqueries(|node| {
            if let LogicalPlan::Projection(projection) = node {
                for expr in &projection.expr {
                    if let Expr::Alias(alias) = expr
                        && let Expr::Column(column) = alias.expr.as_ref()
                        && names.contains(&column.name)
                        && !names.contains(&alias.name)
                        && !renamed.contains(&alias.name)
                    {
                        renamed.push(alias.name.clone());
                    }
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        if renamed.is_empty() {
            break;
        }
        names.extend(renamed);
    }
    let mut reads = Reads {
        names: &names,
        fields: BTreeSet::new(),
        whole: holds_any(plan.schema(), &names),
    };
    plan.apply_with_subqueries(|node| {
        reads.node(node);
        Ok(if reads.whole {
            TreeNodeRecursion::Stop
        } else {
            TreeNodeRecursion::Continue
        })
    })?;
    Ok((!reads.whole).then_some(reads.fields))
}

/// Whether a schema has a column named one of `names`.
fn holds_any(schema: &DFSchema, names: &[String]) -> bool {
    schema
        .fields()
        .iter()
        .any(|field| names.contains(field.name()))
}

/// The uses of the columns `names` that [`read_fields`] has found.
struct Reads<'a> {
    names: &'a [String],
    /// The fields read by name.
    fields: BTreeSet<String>,
    /// Whether a use reads a column whole.
    whole: bool,
}

impl Reads<'_> {
    /// Note the uses of one operator.
    fn node(&mut self, node: &LogicalPlan) {
        match node {
            LogicalPlan::Projection(projection) => {
                for expr in &projection.expr {
                    if !self.passes_on(expr) {
                        self.expr(expr);
                    }
                }
            }
            LogicalPlan::Extension(_) => match as_node::<AiNode>(node) {
                Some(ai) => {
                    let exprs = ai
                        .expressions()
                        .into_iter()
                        .chain(ai.filter.clone())
                        .chain(ai.hint_test.clone())
                        .chain(ai.tests.iter().map(|test| test.test.clone()));
                    for expr in exprs {
                        self.expr(&expr);
                    }
                    if let Some(Fetch::PerKey { key, .. }) = &ai.fetch
                        && key.iter().any(|name| self.names.contains(name))
                    {
                        self.whole = true;
                    }
                }
                None => self.inputs_hold(node),
            },
            LogicalPlan::Filter(_)
            | LogicalPlan::Sort(_)
            | LogicalPlan::Limit(_)
            | LogicalPlan::Join(_)
            | LogicalPlan::Aggregate(_)
            | LogicalPlan::Window(_)
            | LogicalPlan::Repartition(_)
            | LogicalPlan::Distinct(_)
            | LogicalPlan::SubqueryAlias(_)
            | LogicalPlan::TableScan(_)
            | LogicalPlan::Values(_)
            | LogicalPlan::EmptyRelation(_) => {
                for expr in node.expressions() {
                    self.expr(&expr);
                }
            }
            // the columns of the rest are not followed by name
            _ => self.inputs_hold(node),
        }
    }

    /// Whether a SELECT list item passes a column on, as it is or renamed.
    fn passes_on(&self, expr: &Expr) -> bool {
        let inner = match expr {
            Expr::Alias(alias) => alias.expr.as_ref(),
            other => other,
        };
        matches!(inner, Expr::Column(column) if self.names.contains(&column.name))
    }

    /// Note an operator whose inputs' columns are all read whole.
    fn inputs_hold(&mut self, node: &LogicalPlan) {
        if node
            .inputs()
            .iter()
            .any(|input| holds_any(input.schema(), self.names))
        {
            self.whole = true;
        }
    }

    /// Note the uses in one expression.
    fn expr(&mut self, expr: &Expr) {
        let _ = expr.apply(|found| {
            if let Some(field) = self.field_read(found) {
                self.fields.insert(field);
                return Ok(TreeNodeRecursion::Jump);
            }
            let subquery = match found {
                Expr::Column(column) | Expr::OuterReferenceColumn(_, column) => {
                    self.whole |= self.names.contains(&column.name);
                    None
                }
                Expr::ScalarSubquery(subquery) => Some(&subquery.subquery),
                Expr::InSubquery(found) => Some(&found.subquery.subquery),
                Expr::Exists(found) => Some(&found.subquery.subquery),
                _ => None,
            };
            if let Some(plan) = subquery {
                self.whole |= holds_any(plan.schema(), self.names);
            }
            Ok(TreeNodeRecursion::Continue)
        });
    }

    /// The field `expr` reads of one of the columns by name, if it is
    /// `get_field(column, 'field', ...)`.
    fn field_read(&self, expr: &Expr) -> Option<String> {
        let Expr::ScalarFunction(call) = expr else {
            return None;
        };
        if call.name() != "get_field" {
            return None;
        }
        match call.args.as_slice() {
            [Expr::Column(column), Expr::Literal(field, _), ..]
                if self.names.contains(&column.name) =>
            {
                field.try_as_str().flatten().map(ToString::to_string)
            }
            _ => None,
        }
    }
}

/// `plan` with each AI call `old`, an unfolded call, as `new` in every
/// `AiNode`, an extraction's spans call of the same arguments included,
/// and each node that makes `new` adding the columns of its types.
pub(crate) fn replace_call(plan: LogicalPlan, old: &Expr, new: &Expr) -> Result<LogicalPlan> {
    let Expr::ScalarFunction(old) = old else {
        unreachable!("an AI call")
    };
    let Expr::ScalarFunction(new) = new else {
        unreachable!("an AI call")
    };
    let extraction = old.name() == "ai_extract";
    let rewrite = |expr: Expr| -> Result<Expr> {
        expr.transform_down(|found| {
            let Expr::ScalarFunction(call) = &found else {
                return Ok(Transformed::no(found));
            };
            let span = extraction && call.name() == "ai_extract_span";
            if !(span || call.name() == old.name())
                || call
                    .args
                    .iter()
                    .cloned()
                    .map(unfolded)
                    .ne(old.args.iter().cloned())
            {
                return Ok(Transformed::no(found));
            }
            Ok(Transformed::new(
                Expr::ScalarFunction(ScalarFunction::new_udf(
                    Arc::clone(&call.func),
                    new.args.clone(),
                )),
                true,
                TreeNodeRecursion::Jump,
            ))
        })
        .map(|done| done.data)
    };
    plan.transform_up_with_subqueries(|node| {
        let Some(ai) = as_node::<AiNode>(&node) else {
            return node.recompute_schema().map(Transformed::yes);
        };
        let mut ai = ai.map_expressions(rewrite)?;
        if function_name(&ai.call) == Some("ai_extract")
            && let Expr::ScalarFunction(call) = &ai.call
            && call.args == new.args
        {
            let schema = ai.input.schema();
            let spans = Expr::ScalarFunction(ScalarFunction::new_udf(
                quail_function("ai_extract_span"),
                call.args.clone(),
            ));
            for (index, typed) in [&ai.call, &spans].into_iter().enumerate() {
                if let Some(field) = ai.fields.get(index) {
                    let (_, found) = typed.to_field(schema)?;
                    ai.fields[index] = found.as_ref().clone().with_name(field.name());
                }
            }
        }
        Ok(Transformed::yes(extension_node(ai.checked()?)))
    })
    .map(|done| done.data)
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{DataType, Field, Fields, Schema};
    use datafusion::datasource::empty::EmptyTable;
    use datafusion::datasource::provider_as_source;
    use datafusion::logical_expr::{LogicalPlanBuilder, col, lit};
    use datafusion::optimizer::{Optimizer, OptimizerContext};

    use super::*;

    /// An extraction of `reviews.body`, adding `e` and `e_span`.
    fn extraction() -> LogicalPlan {
        extraction_of(&["who"])
    }

    /// An extraction of `reviews.body` asking one question per field,
    /// adding `e` and `e_span`.
    fn extraction_of(fields: &[&str]) -> LogicalPlan {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("body", DataType::Utf8, true),
        ]));
        let reviews = LogicalPlanBuilder::scan(
            "reviews",
            provider_as_source(Arc::new(EmptyTable::new(schema))),
            None,
        )
        .unwrap()
        .build()
        .unwrap();
        let questions = fields
            .iter()
            .flat_map(|field| [lit(*field), lit(format!("{field}?"))])
            .collect();
        let call = Expr::ScalarFunction(ScalarFunction::new_udf(
            crate::functions::quail_functions()
                .into_iter()
                .find(|found| found.name() == "ai_extract")
                .unwrap(),
            vec![
                col("reviews.body"),
                datafusion::functions::core::expr_fn::named_struct(questions),
            ],
        ));
        let answer = Fields::from(
            fields
                .iter()
                .map(|field| Field::new(*field, DataType::Utf8, true))
                .collect::<Vec<_>>(),
        );
        let node = AiNode::try_new(
            Arc::new(reviews),
            call,
            "e".into(),
            vec![
                Field::new("e", DataType::Struct(answer.clone()), true),
                Field::new("e_span", DataType::Struct(answer), true),
            ],
        )
        .unwrap();
        extension_node(node)
    }

    fn pushed(plan: LogicalPlan) -> String {
        Optimizer::with_rules(vec![Arc::new(PushDownProjectionIntoAiNodes)])
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap()
            .display_indent()
            .to_string()
    }

    #[test]
    fn a_node_returns_only_the_added_columns_the_projection_reads() {
        let plan = LogicalPlanBuilder::from(extraction())
            .project(vec![col("reviews.id"), col("e")])
            .unwrap()
            .build()
            .unwrap();
        let text = pushed(plan);
        assert!(text.contains("returns: [e]"), "{text}");
    }

    #[test]
    fn a_node_keeps_the_columns_the_projection_reads_and_its_answer() {
        let spans = LogicalPlanBuilder::from(extraction())
            .project(vec![col("reviews.id"), col("e"), col("e_span")])
            .unwrap()
            .build()
            .unwrap();
        assert!(!pushed(spans).contains("returns:"));
        let ids = LogicalPlanBuilder::from(extraction())
            .project(vec![col("reviews.id")])
            .unwrap()
            .build()
            .unwrap();
        assert!(pushed(ids).contains("returns: [e]"));
    }

    fn field(column: &str, name: &str) -> Expr {
        datafusion::functions::core::expr_fn::get_field(col(column), name)
    }

    #[test]
    fn an_extraction_asks_only_the_fields_the_query_reads() {
        let plan = LogicalPlanBuilder::from(extraction_of(&["who", "when", "where"]))
            .filter(field("e", "when").is_not_null())
            .unwrap()
            .project(vec![col("reviews.id"), field("e", "who").alias("who")])
            .unwrap()
            .project(vec![col("who")])
            .unwrap()
            .build()
            .unwrap();
        let text = pushed(plan);
        assert!(
            text.contains(
                r#"named_struct(Utf8("who"), Utf8("who?"), Utf8("when"), Utf8("when?")))"#
            ),
            "{text}"
        );
    }

    #[test]
    fn an_extraction_follows_a_renamed_answer_to_its_fields() {
        let plan = LogicalPlanBuilder::from(extraction_of(&["who", "when"]))
            .project(vec![col("reviews.id"), col("e").alias("f")])
            .unwrap()
            .project(vec![field("f", "when").alias("when")])
            .unwrap()
            .build()
            .unwrap();
        let text = pushed(plan);
        assert!(
            text.contains(r#"named_struct(Utf8("when"), Utf8("when?")))"#),
            "{text}"
        );
    }

    #[test]
    fn an_extraction_read_whole_asks_every_field() {
        let whole = LogicalPlanBuilder::from(extraction_of(&["who", "when"]))
            .project(vec![col("reviews.id"), col("e")])
            .unwrap()
            .build()
            .unwrap();
        assert!(pushed(whole).contains(r#"Utf8("who"), Utf8("who?"), Utf8("when")"#));
        let compared = LogicalPlanBuilder::from(extraction_of(&["who", "when"]))
            .filter(col("e").is_not_null())
            .unwrap()
            .project(vec![field("e", "who").alias("who")])
            .unwrap()
            .build()
            .unwrap();
        assert!(pushed(compared).contains(r#"Utf8("who"), Utf8("who?"), Utf8("when")"#));
    }
}
