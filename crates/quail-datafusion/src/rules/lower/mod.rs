mod extract;
mod outer_join;

use quail_planner::constants::SPAN_SUFFIX;
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result, internal_err, plan_err};
use datafusion::logical_expr::expr_rewriter::strip_outer_reference;
use datafusion::logical_expr::simplify::SimplifyContext;
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{
    Expr, ExprSchemable, Filter, Join, JoinType, LogicalPlan, Operator, Projection,
};
use datafusion::optimizer::analyzer::type_coercion::TypeCoercionRewriter;
use datafusion::optimizer::simplify_expressions::ExprSimplifier;
use datafusion::optimizer::{AnalyzerRule, OptimizerConfig, OptimizerRule};

use crate::ai::{AiNode, Semantics, Written, WrittenPredicate};
use crate::functions::column_call_kind;
use crate::functions::{function_name, is_ai_call, quail_function};
use crate::plan::{as_node, extension_node};

#[derive(Debug, Default)]
pub struct LowerAiCalls;

impl AnalyzerRule for LowerAiCalls {
    fn name(&self) -> &str {
        "lower_ai_calls"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        extract::lower_plan(plan)
    }
}

impl OptimizerRule for LowerAiCalls {
    fn name(&self) -> &str {
        "lower_ai_calls"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        if !calls_ai_function(&plan)? {
            return Ok(Transformed::no(plan));
        }
        let lowered = extract::lower_plan(plan.clone())?;
        Ok(if lowered == plan {
            Transformed::no(plan)
        } else {
            Transformed::yes(lowered)
        })
    }
}

fn hinted(expr: &Expr) -> &Expr {
    match expr {
        Expr::ScalarFunction(call) if call.name() == "hint" && !call.args.is_empty() => {
            hinted(&call.args[0])
        }
        other => other,
    }
}

fn unhint(expr: &Expr) -> (Expr, Option<Expr>) {
    match expr {
        Expr::ScalarFunction(call) if call.name() == "hint" && !call.args.is_empty() => {
            let (term, inner) = unhint(&call.args[0]);
            (term, call.args.get(1).cloned().or(inner))
        }
        other => (other.clone(), None),
    }
}

fn calls_ai_function(plan: &LogicalPlan) -> Result<bool> {
    plan.exists(|node| {
        if as_node::<AiNode>(node).is_some() {
            return Ok(false);
        }
        for expr in node.expressions() {
            let found = expr.exists(|inner| match inner {
                Expr::Exists(exists) => calls_ai_function(&exists.subquery.subquery),
                Expr::InSubquery(found) => calls_ai_function(&found.subquery.subquery),
                Expr::ScalarSubquery(found) => calls_ai_function(&found.subquery),
                other => Ok(is_ai_call(other)),
            })?;
            if found {
                return Ok(true);
            }
        }
        Ok(false)
    })
}

fn is_score_test(term: &Expr) -> bool {
    matches!(term, Expr::BinaryExpr(binary)
    if matches!(
        binary.op,
        Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
    ) && match (function_name(&binary.left), function_name(&binary.right)) {
        (Some("ai_score"), _) => !binary.right.exists(|found| Ok(crate::is_ai_call(found))).unwrap_or(true),
        (_, Some("ai_score")) => !binary.left.exists(|found| Ok(crate::is_ai_call(found))).unwrap_or(true),
        _ => false,
    })
}

fn filter_calls(term: &Expr) -> Option<Vec<Expr>> {
    let mut calls: Vec<Expr> = Vec::new();
    let mut other = false;
    let _ = term.apply(|found| {
        if matches!(function_name(found), Some("ai_filter" | "ai_score")) {
            if !calls.contains(found) {
                calls.push(found.clone());
            }
            return Ok(TreeNodeRecursion::Jump);
        }
        other = other
            || crate::is_ai_call(found)
            || matches!(
                found,
                Expr::Exists(_) | Expr::InSubquery(_) | Expr::ScalarSubquery(_)
            );
        Ok(TreeNodeRecursion::Continue)
    });
    (!other && !calls.is_empty()).then_some(calls)
}

fn is_asked(term: &Expr) -> bool {
    let term = hinted(term);
    function_name(term) == Some("ai_filter") || is_score_test(term)
}

pub(crate) fn is_join_predicate(term: &Expr) -> bool {
    is_score_test(hinted(term)) || filter_calls(hinted(term)).is_some()
}

type Terms = Vec<(usize, Expr)>;

fn split_asked(predicate: &Expr) -> (Terms, Terms, usize) {
    let terms = split_conjunction(predicate);
    let num_terms = terms.len();
    let (mut asked, mut rest) = (Vec::new(), Vec::new());
    for (position, term) in terms.into_iter().enumerate() {
        if is_join_predicate(term) {
            asked.push((position, term.clone()));
        } else {
            rest.push((position, term.clone()));
        }
    }
    (asked, rest, num_terms)
}

fn replace_call(term: Expr, call: &Expr, with: &Expr) -> Result<Expr> {
    Ok(term
        .transform_down(|found| {
            if found != *call {
                return Ok(Transformed::no(found));
            }
            Ok(Transformed::new(
                with.clone(),
                true,
                TreeNodeRecursion::Jump,
            ))
        })?
        .data)
}

fn answer_node(
    input: LogicalPlan,
    call: &Expr,
    (answer, index): (String, usize),
    written: &Written,
) -> Result<AiNode> {
    AiNode::answering(
        input,
        call.clone(),
        answer,
        Written {
            index,
            ..written.clone()
        },
    )
}

fn asked_node(
    input: LogicalPlan,
    term: &Expr,
    (answer, index): (String, usize),
    semantics: Semantics,
    written: Written,
) -> Result<LogicalPlan> {
    let (term, hint) = unhint(term);
    let read = Expr::Column(Column::from_name(answer.clone()));
    let (call, inline_predicate) = match term {
        Expr::BinaryExpr(mut binary) if function_name(&binary.left) == Some("ai_score") => {
            let call = std::mem::replace(binary.left.as_mut(), read);
            (call, Expr::BinaryExpr(binary))
        }
        Expr::BinaryExpr(mut binary) if function_name(&binary.right) == Some("ai_score") => {
            let call = std::mem::replace(binary.right.as_mut(), read);
            (call, Expr::BinaryExpr(binary))
        }
        call => (call, read),
    };
    let node = AiNode::answering(input, call, answer, Written { index, ..written })?;
    Ok(extension_node(
        node.hinted(hint).tested(inline_predicate, semantics)?,
    ))
}

fn stack_asked(
    mut input: LogicalPlan,
    asked: Terms,
    (num_terms, clause): (usize, usize),
    on: Option<&str>,
    columns: &mut Columns,
) -> Result<LogicalPlan> {
    for (position, term) in asked {
        let written = Written {
            position,
            num_terms,
            clause,
            index: 0,
            on: on.map(str::to_string),
        };
        let Some(calls) = filter_calls(hinted(&term)) else {
            input = asked_node(
                input,
                &term,
                columns.answer_name(),
                Semantics::Rows,
                written,
            )?;
            continue;
        };
        let (mut test, hint) = unhint(&term);
        for (number, call) in calls.iter().enumerate() {
            let (mut answer, index) = columns.answer_name();
            if columns.selects(call) {
                answer = columns.name_for(call, input.schema());
            }
            let read = Expr::Column(Column::from_name(answer.clone()));
            let mut node = answer_node(input, call, (answer, index), &written)?;
            if number + 1 == calls.len() {
                node = node.hinted(hint.clone());
            }
            input = extension_node(node);
            test = replace_call(test, call, &read)?;
        }
        input = LogicalPlan::Filter(Filter::try_new(test, Arc::new(input))?);
    }
    Ok(input)
}

#[derive(Debug, Default)]
pub(crate) struct Columns {
    selected: Vec<(Expr, String)>,
    reserved: std::collections::BTreeSet<String>,
    num_unnamed: usize,
    num_answers: usize,
    num_clauses: usize,
}

impl Columns {
    fn answer_name(&mut self) -> (String, usize) {
        loop {
            self.num_answers += 1;
            let name = format!("__quail_answer_{}", self.num_answers);
            if self.reserved.insert(name.clone()) {
                return (name, self.num_answers);
            }
        }
    }

    fn selects(&self, call: &Expr) -> bool {
        self.selected.iter().any(|(found, _)| found == call)
    }

    fn take_selected_name(&mut self, name: &str) -> String {
        self.reserved.insert(name.to_string());
        name.to_string()
    }

    fn clause(&mut self) -> usize {
        self.num_clauses += 1;
        self.num_clauses
    }

    fn name_for(&mut self, call: &Expr, input: &datafusion::common::DFSchema) -> String {
        match self.selected.iter().find(|(found, _)| found == call) {
            Some((_, name)) if !names_input(input, name) => name.clone(),
            _ => self.unnamed(),
        }
    }

    fn unnamed(&mut self) -> String {
        loop {
            self.num_unnamed += 1;
            let name = format!("__quail_column_{}", self.num_unnamed);
            if self.reserved.insert(name.clone()) {
                return name;
            }
        }
    }
}

pub(crate) fn names_input(input: &datafusion::common::DFSchema, name: &str) -> bool {
    input.has_column_with_unqualified_name(name)
        || input
            .iter()
            .any(|(qualifier, _)| qualifier.is_some_and(|found| found.table() == name))
}

fn selected_calls(plan: &LogicalPlan) -> Vec<(Expr, String)> {
    let mut found = Vec::new();
    let _ = plan.apply(|node| {
        if let LogicalPlan::Projection(projection) = node {
            for expr in &projection.expr {
                if let Expr::Alias(alias) = expr
                    && (column_call_kind(&alias.expr).is_some()
                        || matches!(function_name(&alias.expr), Some("ai_filter" | "ai_score")))
                {
                    found.push((alias.expr.as_ref().clone(), alias.name.clone()));
                }
            }
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

fn is_selectivity_hint(arg: &Expr) -> bool {
    match arg {
        Expr::ScalarFunction(call) if call.name() == "named_struct" => {
            call.args.len() == 2
                && matches!(&call.args[0], Expr::Literal(value, _)
                    if value.try_as_str().flatten() == Some("selectivity"))
        }
        _ => false,
    }
}

fn base_call(call: &Expr) -> Expr {
    let Expr::ScalarFunction(found) = call else {
        return call.clone();
    };
    let mut args = found.args.clone();
    if args.last().is_some_and(is_selectivity_hint) {
        args.pop();
    }
    let function = if found.name() == "ai_extract_span" {
        quail_function("ai_extract")
    } else {
        Arc::clone(&found.func)
    };
    Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
        function, args,
    ))
}

fn is_span_call(expr: &Expr) -> bool {
    function_name(expr) == Some("ai_extract_span")
}

fn column_calls(term: &Expr) -> Vec<Expr> {
    let mut found: Vec<Expr> = Vec::new();
    let _ = term.apply(|expr| {
        if column_call_kind(expr).is_some() || is_span_call(expr) {
            let call = base_call(expr);
            if !found.contains(&call) {
                found.push(call);
            }
            return Ok(TreeNodeRecursion::Jump);
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

fn added_column(plan: &LogicalPlan, call: &Expr) -> Option<String> {
    fn search(node: &LogicalPlan, call: &Expr) -> Option<String> {
        if let Some(column) = as_node::<AiNode>(node)
            && column.answers_every_row(call)
        {
            return Some(column.answer().to_string());
        }
        let inputs: Vec<&LogicalPlan> = match node {
            LogicalPlan::Join(_)
            | LogicalPlan::Filter(_)
            | LogicalPlan::Projection(_)
            | LogicalPlan::Sort(_)
            | LogicalPlan::Window(_)
            | LogicalPlan::Limit(_)
            | LogicalPlan::Repartition(_)
            | LogicalPlan::Extension(_) => node.inputs(),
            _ => Vec::new(),
        };
        inputs.into_iter().find_map(|input| search(input, call))
    }
    search(plan, call).filter(|name| plan.schema().has_column_with_unqualified_name(name))
}

fn add_column(input: LogicalPlan, call: Expr, name: &str) -> Result<LogicalPlan> {
    let schema = input.schema();
    let field = |expr: &Expr, name: String| -> Result<Field> {
        let (_, field) = expr.to_field(schema)?;
        Ok(field.as_ref().clone().with_name(name))
    };
    let mut fields = vec![field(&call, name.to_string())?];
    if function_name(&call) == Some("ai_extract") {
        let Expr::ScalarFunction(found) = &call else {
            return internal_err!("an AI call is a function call");
        };
        let spans = Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
            quail_function("ai_extract_span"),
            found.args.clone(),
        ));
        fields.push(field(&spans, format!("{name}{SPAN_SUFFIX}"))?);
    }
    Ok(extension_node(AiNode::try_new(
        Arc::new(input),
        call,
        name.to_string(),
        fields,
    )?))
}

fn read_columns(term: Expr, columns: &[(Expr, String)]) -> Result<Expr> {
    Ok(term
        .transform_down(|expr| {
            if !(column_call_kind(&expr).is_some() || is_span_call(&expr)) {
                return Ok(Transformed::no(expr));
            }
            let call = base_call(&expr);
            let Some((_, name)) = columns.iter().find(|(found, _)| *found == call) else {
                return Ok(Transformed::no(expr));
            };
            let name = if is_span_call(&expr) {
                format!("{name}{SPAN_SUFFIX}")
            } else {
                name.clone()
            };
            Ok(Transformed::new(
                Expr::Column(Column::from_name(name)),
                true,
                TreeNodeRecursion::Jump,
            ))
        })?
        .data)
}

fn exists_gate(term: &Expr) -> Option<(LogicalPlan, Expr, bool)> {
    let Expr::Exists(exists) = hinted(term) else {
        return None;
    };
    let LogicalPlan::Projection(projection) = exists.subquery.subquery.as_ref() else {
        return None;
    };
    let LogicalPlan::Filter(filter) = projection.input.as_ref() else {
        return None;
    };
    let LogicalPlan::SubqueryAlias(alias) = filter.input.as_ref() else {
        return None;
    };
    if !matches!(alias.input.as_ref(), LogicalPlan::TableScan(_)) || !is_asked(&filter.predicate) {
        return None;
    }
    let predicate = strip_outer_reference(filter.predicate.clone());
    Some((filter.input.as_ref().clone(), predicate, exists.negated))
}

fn lower_filter(filter: &Filter, columns: &mut Columns) -> Result<Transformed<LogicalPlan>> {
    let (asked, rest, num_terms) = split_asked(&filter.predicate);
    let (gates, rest): (Terms, Terms) = rest
        .into_iter()
        .partition(|(_, term)| exists_gate(term).is_some());
    let (tested, plain): (Terms, Terms) = rest
        .into_iter()
        .partition(|(_, term)| !column_calls(term).is_empty());
    if asked.is_empty() && tested.is_empty() && gates.is_empty() {
        return Ok(Transformed::no(LogicalPlan::Filter(filter.clone())));
    }
    let mut input = match conjunction(plain.into_iter().map(|(_, term)| term)) {
        Some(predicate) => match boolean_filter(predicate, &filter.input)? {
            Some(kept) => LogicalPlan::Filter(kept),
            None => return Ok(Transformed::no(LogicalPlan::Filter(filter.clone()))),
        },
        None => filter.input.as_ref().clone(),
    };
    let clause = columns.clause();
    input = stack_asked(input, asked, (num_terms, clause), None, columns)?;
    for (position, gate) in &gates {
        let (table, predicate, negated) = exists_gate(gate).expect("a gate");
        let joined = datafusion::logical_expr::LogicalPlanBuilder::from(input)
            .cross_join(table)?
            .build()?;
        input = asked_node(
            joined,
            &predicate,
            columns.answer_name(),
            if negated {
                Semantics::Anti
            } else {
                Semantics::Semi
            },
            Written {
                position: *position,
                num_terms,
                clause,
                index: 0,
                on: None,
            },
        )?;
    }
    if tested.is_empty() {
        return Ok(Transformed::yes(input));
    }
    let mut added: Vec<(Expr, String)> = Vec::new();
    for call in tested.iter().flat_map(|(_, term)| column_calls(term)) {
        if added.iter().any(|(found, _)| *found == call) {
            continue;
        }
        let name = match added_column(&input, &call) {
            Some(name) => name,
            None => {
                let name = columns.name_for(&call, input.schema());
                input = add_column(input, call.clone(), &name)?;
                name
            }
        };
        added.push((call, name));
    }
    let mut tests = Vec::new();
    let mut written = Vec::new();
    for (position, term) in tested {
        let test = read_columns(term.clone(), &added)?;
        written.push(WrittenPredicate {
            predicate: simplified(test.clone(), &input),
            written: term,
            position,
            clause,
        });
        tests.push(test);
    }
    let names: Vec<String> = added.iter().map(|(_, name)| name.clone()).collect();
    input = with_written_predicates(input, &names, &written)?;
    match boolean_filter(
        conjunction(tests).unwrap_or(datafusion::logical_expr::lit(true)),
        &Arc::new(input),
    )? {
        Some(test) => Ok(Transformed::yes(LogicalPlan::Filter(test))),
        None => Ok(Transformed::no(LogicalPlan::Filter(filter.clone()))),
    }
}

fn simplified(test: Expr, input: &LogicalPlan) -> Expr {
    let Ok(coerced) = test
        .clone()
        .rewrite(&mut TypeCoercionRewriter::new(input.schema()))
    else {
        return test;
    };
    let context = SimplifyContext::builder()
        .with_schema(Arc::clone(input.schema()))
        .build();
    ExprSimplifier::new(context)
        .simplify(coerced.data)
        .unwrap_or(test)
}

fn with_written_predicates(
    plan: LogicalPlan,
    names: &[String],
    predicates: &[WrittenPredicate],
) -> Result<LogicalPlan> {
    Ok(plan
        .transform_down(|node| {
            if matches!(node, LogicalPlan::SubqueryAlias(_)) {
                return Ok(Transformed::new(node, false, TreeNodeRecursion::Jump));
            }
            let Some(column) = as_node::<AiNode>(&node) else {
                return Ok(Transformed::no(node));
            };
            if column.is_predicate() || !names.iter().any(|name| name == column.answer()) {
                return Ok(Transformed::no(node));
            }
            Ok(Transformed::yes(extension_node(
                column.clone().with_written_predicates(predicates),
            )))
        })?
        .data)
}

fn boolean_filter(predicate: Expr, input: &Arc<LogicalPlan>) -> Result<Option<Filter>> {
    let Ok(data_type) = predicate.get_type(input.schema()) else {
        return Ok(None);
    };
    if !matches!(data_type, DataType::Boolean | DataType::Null) {
        return plan_err!("a test is true or false, got {data_type}");
    }
    Ok(Filter::try_new(predicate, Arc::clone(input)).ok())
}

fn lower_join(join: &Join, columns: &mut Columns) -> Result<Transformed<LogicalPlan>> {
    let Some(predicate) = &join.filter else {
        return Ok(Transformed::no(LogicalPlan::Join(join.clone())));
    };
    let is_or = |term: &Expr| {
        matches!(hinted(term), Expr::BinaryExpr(found) if found.op == Operator::Or)
            && is_join_predicate(term)
    };
    let (ors, on): (Vec<&Expr>, Vec<&Expr>) = split_conjunction(predicate)
        .into_iter()
        .partition(|term| is_or(term));
    if let Some(ors) = conjunction(ors.into_iter().cloned())
        && join.join_type == JoinType::Inner
    {
        let mut kept = join.clone();
        kept.filter = conjunction(on.into_iter().cloned());
        let above = Filter::try_new(ors, Arc::new(LogicalPlan::Join(kept)))?;
        return lower_filter(&above, columns);
    }
    let (asked, rest, num_terms) = split_asked(predicate);
    if asked.is_empty() || join.join_type != JoinType::Inner {
        return Ok(Transformed::no(LogicalPlan::Join(join.clone())));
    }
    let joined = join
        .right
        .schema()
        .iter()
        .find_map(|(qualifier, _)| qualifier.map(|found| found.table().to_string()))
        .unwrap_or_default();
    let clause = columns.clause();
    let mut kept = join.clone();
    kept.filter = conjunction(rest.into_iter().map(|(_, term)| term));
    Ok(Transformed::yes(stack_asked(
        LogicalPlan::Join(kept),
        asked,
        (num_terms, clause),
        Some(&joined),
        columns,
    )?))
}

fn combine_prompt_documents(
    prompt: &mut datafusion::logical_expr::expr::ScalarFunction,
) -> Result<()> {
    use datafusion::logical_expr::lit;
    if prompt.args.len() <= 2 {
        return Ok(());
    }
    let Some(Expr::Literal(value, _)) = prompt.args.first() else {
        return Ok(());
    };
    let Some(template) = value.try_as_str().flatten() else {
        return Ok(());
    };
    quail_planner::prompts::check_placeholders(template, prompt.args.len() - 1)
        .map_err(|error| datafusion::common::DataFusionError::Plan(error.to_string()))?;
    let mut question = template.to_string();
    let mut documents = Vec::new();
    for (index, document) in prompt.args.iter().skip(1).enumerate() {
        question = question.replace(&format!("{{{index}}}"), &format!("DOCUMENT {index}"));
        documents.push(lit(format!("\n\nDOCUMENT {index}:\n")));
        documents.push(document.clone());
    }
    let absent = prompt
        .args
        .iter()
        .skip(1)
        .cloned()
        .map(Expr::is_null)
        .reduce(Expr::or)
        .unwrap_or(datafusion::logical_expr::lit(false));
    let combined = datafusion::functions::string::expr_fn::concat(documents);
    let document =
        datafusion::logical_expr::when(absent, lit(datafusion::common::ScalarValue::Utf8(None)))
            .otherwise(combined)?;
    prompt.args = vec![lit(format!("{{0}}\n\n{question}")), document];
    Ok(())
}

pub(crate) fn precompute_prompt_arguments(projection: Projection) -> Result<Projection> {
    let mut computed: Vec<(Expr, String)> = Vec::new();
    let mut compute = |argument: &mut Expr| {
        if matches!(argument, Expr::Column(_) | Expr::OuterReferenceColumn(..)) {
            return;
        }
        let name = match computed.iter().find(|(found, _)| found == argument) {
            Some((_, name)) => name.clone(),
            None => {
                let mut index = computed.len();
                let name = loop {
                    let name = format!("__quail_argument_{index}");
                    if !projection
                        .input
                        .schema()
                        .has_column_with_unqualified_name(&name)
                        && !computed.iter().any(|(_, found)| *found == name)
                    {
                        break name;
                    }
                    index += 1;
                };
                computed.push((argument.clone(), name.clone()));
                name
            }
        };
        *argument = Expr::Column(Column::from_name(name));
    };
    let exprs = projection
        .expr
        .iter()
        .map(|expr| {
            let mapped = expr
                .clone()
                .transform_down(|expr| {
                    if !crate::functions::is_ai_call(&expr) {
                        return Ok(Transformed::no(expr));
                    }
                    let Expr::ScalarFunction(mut call) = expr else {
                        return internal_err!("an AI call is a function call");
                    };
                    let document_call = matches!(
                        call.name(),
                        "ai_classify"
                            | "ai_extract"
                            | "ai_extract_span"
                            | "ai_map"
                            | "ai_sae_features"
                    );
                    if let Some(argument) = call.args.first_mut() {
                        if let Expr::ScalarFunction(prompt) = argument
                            && prompt.name() == "prompt"
                        {
                            combine_prompt_documents(prompt)?;
                            for document in prompt.args.iter_mut().skip(1) {
                                compute(document);
                            }
                        } else if document_call {
                            compute(argument);
                        }
                    }
                    Ok(Transformed::new(
                        Expr::ScalarFunction(call),
                        true,
                        TreeNodeRecursion::Jump,
                    ))
                })?
                .data;
            mapped.alias_if_changed(expr.schema_name().to_string())
        })
        .collect::<Result<Vec<_>>>()?;
    if computed.is_empty() {
        return Ok(projection);
    }
    let mut input: Vec<Expr> = projection
        .input
        .schema()
        .columns()
        .into_iter()
        .map(Expr::Column)
        .collect();
    input.extend(computed.into_iter().map(|(expr, name)| expr.alias(name)));
    let input = LogicalPlan::Projection(Projection::try_new(input, projection.input)?);
    Projection::try_new(exprs, Arc::new(input))
}

fn lower_projection(
    projection: &Projection,
    columns: &mut Columns,
) -> Result<Transformed<LogicalPlan>> {
    let mut input = projection.input.as_ref().clone();
    let mut exprs = Vec::new();
    let mut changed = false;
    for original in &projection.expr {
        let named;
        let expr = if crate::functions::is_ai_call(original) {
            named = original.clone().alias(original.schema_name().to_string());
            &named
        } else {
            original
        };
        let Expr::Alias(alias) = expr else {
            exprs.push(expr.clone());
            continue;
        };
        if !(column_call_kind(&alias.expr).is_some()
            || is_span_call(&alias.expr)
            || matches!(function_name(&alias.expr), Some("ai_filter" | "ai_score")))
            || alias.relation.is_some()
            || alias.metadata.is_some()
        {
            exprs.push(expr.clone());
            continue;
        }
        let call = base_call(&alias.expr);
        let name = match added_column(&input, &call) {
            Some(name) => name,
            None => {
                let name = if is_span_call(&alias.expr) {
                    alias
                        .name
                        .strip_suffix(SPAN_SUFFIX)
                        .unwrap_or(&alias.name)
                        .to_string()
                } else if columns.selects(&call)
                    || input.schema().has_column_with_unqualified_name(&alias.name)
                {
                    columns.name_for(&call, input.schema())
                } else {
                    columns.take_selected_name(&alias.name)
                };
                let name = if names_input(input.schema(), &name) {
                    columns.unnamed()
                } else {
                    name
                };
                input = add_column(input, call.clone(), &name)?;
                name
            }
        };
        let column = if is_span_call(&alias.expr) {
            format!("{name}{SPAN_SUFFIX}")
        } else {
            name
        };
        let read = Expr::Column(Column::from_name(column.clone()));
        exprs.push(if column == alias.name {
            read
        } else {
            read.alias(alias.name.clone())
        });
        changed = true;
    }
    if !changed {
        return Ok(Transformed::no(LogicalPlan::Projection(projection.clone())));
    }
    Ok(Transformed::yes(LogicalPlan::Projection(
        Projection::try_new(exprs, Arc::new(input))?,
    )))
}

impl Columns {
    pub(crate) fn of(plan: &LogicalPlan) -> Result<Self> {
        let mut reserved = std::collections::BTreeSet::new();
        plan.apply_with_subqueries(|node| {
            reserved.extend(
                node.schema()
                    .fields()
                    .iter()
                    .map(|field| field.name().clone()),
            );
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(Self {
            selected: selected_calls(plan),
            reserved,
            ..Self::default()
        })
    }
}

pub(crate) fn lower_node(
    node: LogicalPlan,
    columns: &mut Columns,
) -> Result<Transformed<LogicalPlan>> {
    match &node {
        LogicalPlan::Filter(filter) => lower_filter(filter, columns),
        LogicalPlan::Join(join) => lower_join(join, columns),
        LogicalPlan::Projection(projection) => lower_projection(projection, columns),
        _ => Ok(Transformed::no(node)),
    }
}

pub fn lower_ai_calls(plan: LogicalPlan) -> Result<LogicalPlan> {
    let mut columns = Columns::of(&plan)?;
    Ok(plan
        .transform_up(|node| lower_node(node, &mut columns))?
        .data)
}
