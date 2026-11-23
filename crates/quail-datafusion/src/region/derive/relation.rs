use super::binding::{AnswerColumn, Binder, negated_inward, unsupported};
use super::walk::{Kind, Walk};
use super::{ByAlias, push};
use crate::ai::{AiNode, Semantics, WrittenTest};
use crate::functions::is_quail_call;
use crate::functions::options::{string_literal, unwrap_hint};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use quail_planner::error::{CompileError, compile_error};
use quail_planner::expressions::{
    asked_call, call_aliases, column_alias_of, equality_text, repr_list, repr_str,
};
use quail_planner::physical::{Asked, Equality, FilterPredicate, ModelCall, ValueExpression};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
pub(super) struct Relation {
    pub(super) filters: ByAlias<FilterPredicate>,
    pub(super) joins: BTreeMap<usize, JoinSpec>,
    pub(super) label_tests: ByAlias<(String, Value, Option<f64>)>,
    pub(super) conditions: Vec<(ValueExpression, Option<f64>)>,
    pub(super) equalities: Vec<(String, Vec<Equality>)>,
    pub(super) join_filters: Vec<(String, Vec<Expr>)>,
    claimed: Vec<String>,
    pub(super) tables: Vec<String>,
    pub(super) unplaced: bool,
    answers: Vec<(String, Option<f64>)>,
    or_terms: BTreeMap<String, OrTerm>,
}

#[derive(Clone, Default)]
struct OrTerm {
    plain: Option<Expr>,
    negated: bool,
    disjunction: Option<String>,
    short_circuit: bool,
}

fn is_column_equality(term: &Expr) -> bool {
    matches!(term, Expr::BinaryExpr(binary)
        if binary.op == datafusion::logical_expr::Operator::Eq
            && matches!((binary.left.as_ref(), binary.right.as_ref()),
                (Expr::Column(left), Expr::Column(right)) if left.relation != right.relation))
}

pub(crate) fn is_join_filter(term: &Expr) -> bool {
    let tables = |expr: &Expr| -> Option<BTreeSet<String>> {
        expr.column_refs()
            .into_iter()
            .map(|column| column.relation.as_ref().map(ToString::to_string))
            .collect()
    };
    if let Expr::BinaryExpr(binary) = term
        && binary.op == Operator::Eq
        && let (Some(left), Some(right)) = (tables(&binary.left), tables(&binary.right))
        && left.len() == 1
        && right.len() == 1
        && left != right
    {
        return false;
    }
    let special = term
        .exists(|inner| {
            Ok(is_quail_call(inner)
                || matches!(
                    inner,
                    Expr::Exists(_) | Expr::InSubquery(_) | Expr::ScalarSubquery(_)
                ))
        })
        .unwrap_or(true);
    !special && tables(term).is_some_and(|found| found.len() == 2)
}

impl Relation {
    pub(super) fn in_or(&self, answer: &str) -> bool {
        self.or_terms.contains_key(answer)
    }

    pub(super) fn bind(
        walk: &Walk<'_>,
        binder: &mut Binder,
        names: &BTreeMap<String, String>,
    ) -> Result<Self, CompileError> {
        let mut relation = Self::default();
        for table in walk.joined_tables() {
            binder.add_table(&table.alias, &table.provider, table.schema.clone())?;
            relation.tables.push(table.alias.clone());
        }
        bind_answers(walk, binder, names)?;
        for id in walk.ids() {
            let Kind::Join(join) = &walk.get(id).kind else {
                continue;
            };
            let right = &walk.get(walk.get(id).inputs[1]).tables;
            if right.iter().any(|alias| walk.is_gate(alias)) {
                continue;
            }
            let joined = right.last().cloned().unwrap_or_default();
            let mut terms: Vec<Expr> = join
                .on
                .iter()
                .map(|(left, right)| left.clone().eq(right.clone()))
                .collect();
            terms.extend(join.filter.iter().flat_map(split_conjunction).cloned());
            let (mut equalities, mut filters) = (Vec::new(), Vec::new());
            for term in terms {
                let term = walk.through(&term);
                if is_column_equality(&term) {
                    equalities.push(binder.bind_equality(&term, &joined)?);
                } else if is_join_filter(&term) {
                    filters.push(term);
                } else {
                    relation.bind_term(binder, &term, None)?;
                }
            }
            if !filters.is_empty() {
                relation.join_filters.insert(0, (joined.clone(), filters));
            }
            if !equalities.is_empty() {
                relation.equalities.insert(0, (joined, equalities));
            }
        }
        for id in walk.ids() {
            let Kind::Ai(node) = &walk.get(id).kind else {
                continue;
            };
            let Some(parts) = node
                .or_parts()
                .filter(|_| node.function() == "ai_filter" && node.semantics() == Semantics::Rows)
            else {
                continue;
            };
            let disjunction = (!parts.others.is_empty()).then(|| node.answer().to_string());
            let short_circuit = node.guard().is_some_and(|guard| {
                guard.column_refs().iter().any(|column| {
                    parts
                        .others
                        .iter()
                        .any(|(answer, _)| *answer == column.name)
                })
            });
            let own = (node.answer().to_string(), parts.negated);
            for (answer, negated) in parts.others.into_iter().chain([own]) {
                relation.or_terms.insert(
                    answer,
                    OrTerm {
                        plain: parts.plain.clone(),
                        negated,
                        disjunction: disjunction.clone(),
                        short_circuit,
                    },
                );
            }
        }
        let mut asked: Vec<(usize, &AiNode)> = walk
            .ids()
            .filter_map(|id| match &walk.get(id).kind {
                Kind::Ai(node) if node.is_predicate() || node.semantics() != Semantics::Rows => {
                    Some((id, *node))
                }
                _ => None,
            })
            .collect();
        asked.sort_by_key(|(_, node)| {
            node.written()
                .map_or((true, usize::MAX, usize::MAX), |written| match written.on {
                    Some(_) => (false, written.index, written.index),
                    None => (true, written.position, written.index),
                })
        });
        let (on, rest): (Vec<_>, Vec<_>) = asked.into_iter().partition(|(_, node)| node.in_on());
        for (_, node) in &on {
            let written = node.written().expect("an ON term's place");
            let num_asked = on
                .iter()
                .filter(|(_, other)| {
                    other.written().map(|found| found.clause) == Some(written.clause)
                })
                .count();
            if num_asked > 1 {
                return unsupported(&format!(
                    "The {num_asked} model predicates in JOIN {} must run in separate regions.",
                    written.on.as_deref().unwrap_or_default()
                ));
            }
        }
        for (id, node) in on {
            relation.bind_asked(walk, binder, id, node)?;
        }
        for (id, node) in rest {
            relation.bind_asked(walk, binder, id, node)?;
        }
        let mut tests: Vec<WrittenTest> = Vec::new();
        let mut terms: Vec<Expr> = Vec::new();
        for id in walk.ids() {
            let Kind::Ai(node) = &walk.get(id).kind else {
                continue;
            };
            for test in node.tests() {
                if !tests
                    .iter()
                    .any(|found| (found.clause, found.position) == (test.clause, test.position))
                {
                    tests.push(test);
                }
            }
            if !node.is_predicate()
                && node.semantics() == Semantics::Rows
                && let Some(filter) = node.filter()
            {
                terms.extend(split_conjunction(&filter).into_iter().cloned());
            }
        }
        terms.retain(|term| {
            !tests
                .iter()
                .any(|test| split_conjunction(&test.test).contains(&term))
        });
        for term in terms {
            relation.bind_term(binder, &walk.through(&term), None)?;
        }
        tests.sort_by_key(|test| (test.clause, test.position));
        for test in tests {
            let (written, hint) = unwrap_hint(&test.written)?;
            let hint = match hint {
                Some(hint) => Some(hint),
                None => binder.label_selectivity(&written)?,
            };
            relation.bind_term(binder, &walk.through(&test.test), hint)?;
        }
        Ok(relation)
    }

    fn bind_asked(
        &mut self,
        walk: &Walk<'_>,
        binder: &mut Binder,
        id: usize,
        node: &AiNode,
    ) -> Result<(), CompileError> {
        let join = node.in_on().then_some(true);
        let call = walk.through(&node.call());
        if node.semantics() != Semantics::Rows {
            let input = walk.get(walk.get(id).inputs[0]);
            let Kind::Join(_) = &input.kind else {
                self.unplaced = true;
                return Ok(());
            };
            let inner = walk.get(input.inputs[1]).tables.last().cloned();
            let table = inner
                .as_deref()
                .and_then(|alias| walk.scanned(alias))
                .expect("a gated table");
            binder.add_table(&table.alias, &table.provider, table.schema.clone())?;
            let (call, options) =
                binder.bind_ai_filter_call(&call, &["selectivity", "anchor"], Some(true))?;
            let anti = node.semantics() == Semantics::Anti;
            let spec = gate(
                &call,
                &table.alias,
                anti,
                options.anchor,
                options.selectivity,
            )?;
            self.joins.insert(id, spec);
            return Ok(());
        }
        let filter = node.filter().filter(|_| node.function() == "ai_filter");
        let term = match (filter, self.or_terms.get(node.answer())) {
            (_, Some(term)) => term.clone(),
            (Some(filter), None) => match node.answer_term(&filter) {
                Some(negated) => OrTerm {
                    negated,
                    ..OrTerm::default()
                },
                None => {
                    return unsupported(
                        "DataFusion must evaluate this condition over the model's Boolean answer.",
                    );
                }
            },
            (None, None) => OrTerm::default(),
        };
        let OrTerm {
            plain,
            negated,
            disjunction,
            short_circuit,
        } = term;
        let in_or = plain.is_some() || disjunction.is_some();
        let alternative = match &plain {
            Some(plain) => Some(binder.bind_condition(&walk.through(plain))?),
            None => None,
        };
        let guard = node.guard().and_then(|guard| {
            conjunction(
                split_conjunction(&guard)
                    .into_iter()
                    .filter(|term| {
                        !term
                            .column_refs()
                            .iter()
                            .any(|column| self.or_terms.contains_key(&column.name))
                    })
                    .cloned(),
            )
        });
        let guard = match &guard {
            Some(guard) => Some(binder.bind_condition(&walk.through(guard))?),
            None => None,
        };
        let answer = (node.function() == "ai_filter" && node.filter().is_none() && !in_or)
            .then(|| node.answer().to_string());
        let (predicate, options) = if node.function() == "ai_score" {
            let term = walk
                .through(&node.filter().expect("a test"))
                .transform_up(|found| {
                    Ok(match &found {
                        Expr::Column(column)
                            if column.relation.is_none() && column.name == node.answer() =>
                        {
                            Transformed::yes(call.clone())
                        }
                        _ => Transformed::no(found),
                    })
                })
                .map_err(|error| CompileError::new(error.to_string()))?
                .data;
            let (test, options) = binder.bind_ai_score_test(&term, &["selectivity"], join)?;
            (Asked::Compare(test), options)
        } else {
            let (call, options) =
                binder.bind_ai_filter_call(&call, &["selectivity", "anchor"], join)?;
            (Asked::Call(call), options)
        };
        let read = call_aliases(asked_call(&predicate));
        if read.len() == 1 {
            if options.anchor.is_some() {
                return compile_error(
                    "anchor is a join option; a one-provider AI_FILTER takes only selectivity",
                );
            }
            let guard_selectivity = node
                .guard()
                .map(|found| crate::rules::guards::guard_selectivity(&found, &self.answers));
            if let Some(answer) = &answer {
                self.answers.push((answer.clone(), options.selectivity));
            }
            let selectivity = options
                .selectivity
                .map(|yes| if negated { 1.0 - yes } else { yes });
            let filter = FilterPredicate {
                expression: predicate,
                selectivity,
                guard,
                alternative,
                answer,
                guard_selectivity,
                negated,
                disjunction,
                short_circuit,
            };
            push(&mut self.filters, &read[0], filter);
            return Ok(());
        }
        if in_or || guard.is_some() || answer.is_some() || negated {
            return unsupported(&format!(
                "DataFusion must evaluate {} over joined model results.",
                walk.through(&node.filter().unwrap_or_else(|| node.call()))
            ));
        }
        let spec = JoinSpec {
            predicate,
            semantics: "full",
            selectivity: options.selectivity,
            anchor: options.anchor,
            on: Vec::new(),
            filter: None,
        };
        let (claimed, equalities) = (&mut self.claimed, &mut self.equalities);
        let (mut spec, filters) = join_spec(
            spec,
            &read,
            &self.tables,
            claimed,
            equalities,
            &mut self.join_filters,
        )?;
        if let Some(filter) = conjunction(filters) {
            spec.filter = Some(binder.bind_condition(&negated_inward(filter, false))?);
        }
        self.joins.insert(id, spec);
        Ok(())
    }

    fn bind_term(
        &mut self,
        binder: &mut Binder,
        term: &Expr,
        hint: Option<f64>,
    ) -> Result<(), CompileError> {
        let (term, written) = unwrap_hint(term)?;
        let (term, hint) = (&term, hint.or(written));
        if let Some((answer, items)) = label_test(term)
            && let Some(found) = binder.answer(answer).filter(|found| found.kind == "label")
        {
            let Some(alias) = found.alias.clone() else {
                return unsupported("DataFusion must evaluate this label test over joined rows.");
            };
            let (name, selectivity) = (found.name.clone(), hint);
            let mut accepted = Vec::new();
            for item in items {
                match string_literal(item) {
                    Some(text) => accepted.push(Value::String(text)),
                    None => {
                        return unsupported(
                            "DataFusion must compare this label with the other expression.",
                        );
                    }
                }
            }
            let mut unique = accepted.clone();
            unique.sort_by_key(|value| value.to_string());
            unique.dedup();
            if unique.len() != accepted.len() {
                return unsupported("DataFusion must evaluate a label test with repeated values.");
            }
            push(
                &mut self.label_tests,
                &alias,
                (name, Value::Array(accepted), selectivity),
            );
            return Ok(());
        }
        let term = negated_inward(term.clone(), false);
        let parts = match hint {
            Some(_) => vec![&term],
            None => split_conjunction(&term),
        };
        for part in parts {
            self.conditions.push((binder.bind_condition(part)?, hint));
        }
        Ok(())
    }
}

fn label_test(expr: &Expr) -> Option<(&str, Vec<&Expr>)> {
    fn answer(expr: &Expr) -> Option<&str> {
        match expr {
            Expr::Column(column) if column.relation.is_none() => Some(column.name.as_str()),
            _ => None,
        }
    }
    match expr {
        Expr::InList(InList {
            expr,
            list,
            negated: false,
        }) => Some((answer(expr)?, list.iter().collect())),
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Or,
            right,
        }) => {
            let (column, mut labels) = label_test(left)?;
            let (other, more) = label_test(right)?;
            if column != other {
                return None;
            }
            labels.extend(more);
            Some((column, labels))
        }
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Eq,
            right,
        }) => match (answer(left), answer(right)) {
            (Some(column), None) => Some((column, vec![right.as_ref()])),
            (None, Some(column)) => Some((column, vec![left.as_ref()])),
            _ => None,
        },
        _ => None,
    }
}

fn bind_answers(
    walk: &Walk<'_>,
    binder: &mut Binder,
    names: &BTreeMap<String, String>,
) -> Result<(), CompileError> {
    for id in walk.ids() {
        let Kind::Ai(node) = &walk.get(id).kind else {
            continue;
        };
        if node.semantics() != Semantics::Rows {
            continue;
        }
        let name = names
            .get(node.answer())
            .cloned()
            .unwrap_or_else(|| node.answer().to_string());
        let mut found = AnswerColumn {
            answer: node.answer().to_string(),
            name: name.clone(),
            kind: "boolean",
            alias: None,
            calls: Vec::new(),
            fields: Vec::new(),
        };
        let returned_score = node.function() == "ai_score" && !node.returned_fields().is_empty();
        if node.is_predicate() && !returned_score {
            let tables: Vec<String> = walk.call_tables(&node.call()).into_iter().collect();
            if node.function() != "ai_filter" || node.filter().is_some() || tables.len() != 1 {
                continue;
            }
            found.alias = tables.into_iter().next();
            found.name = node.answer().to_string();
            binder.add_answer(found);
            continue;
        }
        let call = walk.through(&node.call());
        if node.function() == "ai_score" {
            found.kind = "score";
            found.calls = vec![(binder.bind_ai_value(&call)?, name)];
            binder.add_answer(found);
            continue;
        }
        let Some(kind) = node.column_kind() else {
            continue;
        };
        found.kind = kind;
        match kind {
            "extract" => {
                let (fields, members) = binder.bind_ai_extract(&call, &name)?;
                found.fields = fields.fields;
                for member in members {
                    if let ValueExpression::Alias(alias) = member
                        && let ValueExpression::ModelCall(call) = *alias.expression
                    {
                        found.calls.push((call, alias.name));
                    }
                }
            }
            _ => found.calls = vec![(binder.bind_column_call(&call, kind)?, name)],
        }
        let read = call_aliases(&found.calls[0].0);
        if let [alias] = read.as_slice() {
            found.alias = Some(alias.clone());
        }
        binder.add_answer(found);
    }
    Ok(())
}
fn repr_tuple(items: &[String]) -> String {
    let parts: Vec<String> = items.iter().map(|item| repr_str(item)).collect();
    match parts.as_slice() {
        [one] => format!("({one},)"),
        _ => format!("({})", parts.join(", ")),
    }
}

fn gate(
    call: &ModelCall,
    inner: &str,
    anti: bool,
    anchor: Option<String>,
    selectivity: Option<f64>,
) -> Result<JoinSpec, CompileError> {
    let aliases = call_aliases(call);
    let outer = match aliases.as_slice() {
        [first, second] if first == inner => second.clone(),
        [first, second] if second == inner => first.clone(),
        _ => {
            return unsupported("DataFusion must evaluate this EXISTS over the model results.");
        }
    };
    if let Some(anchor) = anchor
        && anchor != outer
    {
        return compile_error(format!(
            "exists/anti always anchor on the outer table {} - the gate applies to its \
             documents - got anchor {}",
            repr_str(&outer),
            repr_str(&anchor)
        ));
    }
    Ok(JoinSpec {
        predicate: Asked::Call(call.clone()),
        semantics: if anti { "anti" } else { "exists" },
        selectivity,
        anchor: Some(outer),
        on: Vec::new(),
        filter: None,
    })
}

fn join_spec(
    mut spec: JoinSpec,
    aliases: &[String],
    tables: &[String],
    claimed: &mut Vec<String>,
    equalities: &mut Vec<(String, Vec<Equality>)>,
    join_filters: &mut Vec<(String, Vec<Expr>)>,
) -> Result<(JoinSpec, Vec<Expr>), CompileError> {
    let outside: Vec<String> = aliases
        .iter()
        .filter(|alias| !tables.contains(alias))
        .cloned()
        .collect();
    if !outside.is_empty() {
        let mut sorted: Vec<String> = tables.to_vec();
        sorted.sort();
        sorted.dedup();
        return compile_error(format!(
            "the join prompt references {}, which are not the FROM table or JOINed tables of \
             this query ({})",
            repr_list(&outside),
            repr_list(&sorted)
        ));
    }
    if let Some(anchor) = &spec.anchor
        && !aliases.contains(anchor)
    {
        return compile_error(format!(
            "anchor {} is not a table of this join ({})",
            repr_str(anchor),
            repr_tuple(aliases)
        ));
    }
    let mut filters: Vec<Expr> = Vec::new();
    for alias in &tables[1..] {
        if aliases.contains(alias) && !claimed.contains(alias) {
            claimed.push(alias.clone());
            if let Some(index) = equalities.iter().position(|(found, _)| found == alias) {
                spec.on.extend(equalities.remove(index).1);
            }
            if let Some(index) = join_filters.iter().position(|(found, _)| found == alias) {
                filters.extend(join_filters.remove(index).1);
            }
        }
    }
    if (!spec.on.is_empty() || !filters.is_empty()) && aliases.len() != 2 {
        return unsupported(
            "DataFusion must evaluate join conditions involving more than two model inputs.",
        );
    }
    for condition in &spec.on {
        let sides = [&condition.left, &condition.right].map(column_alias_of);
        if !sides.iter().all(|side| aliases.contains(side)) {
            return unsupported(
                "DataFusion must evaluate a join condition that reads other model inputs.",
            );
        }
    }
    let read_elsewhere = filters.iter().any(|term| {
        term.column_refs().iter().any(|column| {
            column
                .relation
                .as_ref()
                .is_none_or(|table| !aliases.contains(&table.to_string()))
        })
    });
    if read_elsewhere {
        return unsupported(
            "DataFusion must evaluate a join condition that reads other model inputs.",
        );
    }
    Ok((spec, filters))
}

pub(super) fn check_joins(
    equalities: &[(String, Vec<Equality>)],
    join_filters: &[(String, Vec<Expr>)],
    tables: &[String],
    joins: &[&JoinSpec],
    projected: &[(&ModelCall, String)],
) -> Result<(), CompileError> {
    let scores: Vec<&ModelCall> = projected
        .iter()
        .filter(|(call, _)| call.kind == "score")
        .map(|(call, _)| *call)
        .collect();
    if let Some((alias, conditions)) = equalities.first() {
        let first = equality_text(&conditions[0]);
        if scores
            .iter()
            .any(|score| call_aliases(score).contains(alias))
        {
            return unsupported(
                "DataFusion must join the rows before this projected pair score is computed.",
            );
        }
        return unsupported(&format!(
            "JOIN {alias} ON {first} has no AI predicate over its pairs."
        ));
    }
    if let Some((alias, terms)) = join_filters.first() {
        return unsupported(&format!(
            "JOIN {alias} ON {} has no AI predicate over its pairs.",
            terms[0]
        ));
    }
    let mut predicates: Vec<BTreeSet<String>> = joins
        .iter()
        .filter(|join| join.semantics == "full")
        .map(|join| {
            call_aliases(asked_call(&join.predicate))
                .into_iter()
                .collect()
        })
        .collect();
    predicates.extend(
        scores
            .iter()
            .map(|score| call_aliases(score))
            .filter(|aliases| aliases.len() > 1)
            .map(|aliases| aliases.into_iter().collect()),
    );
    let (root, joined) = (&tables[0], &tables[1..]);
    let uncovered: Vec<String> = joined
        .iter()
        .filter(|alias| !predicates.iter().any(|found| found.contains(*alias)))
        .cloned()
        .collect();
    if !uncovered.is_empty() {
        return unsupported(&format!(
            "Joined tables {} have no AI predicate over their pairs.",
            repr_list(&uncovered)
        ));
    }
    let mut reached: BTreeSet<String> = BTreeSet::from([root.clone()]);
    let mut grew = true;
    while grew {
        grew = false;
        for found in &predicates {
            if !found.is_disjoint(&reached) && !found.is_subset(&reached) {
                reached.extend(found.iter().cloned());
                grew = true;
            }
        }
    }
    let disconnected: BTreeSet<&String> = joined
        .iter()
        .filter(|alias| !reached.contains(*alias))
        .collect();
    if !disconnected.is_empty() {
        let disconnected: Vec<String> = disconnected.into_iter().cloned().collect();
        return unsupported(&format!(
            "DataFusion must join the disconnected model inputs {} to {}.",
            repr_list(&disconnected),
            repr_str(root)
        ));
    }
    Ok(())
}

pub(super) fn check_calls(
    filters: &ByAlias<FilterPredicate>,
    joins: &[&JoinSpec],
    projected: &[(&ModelCall, String)],
) -> Result<(), CompileError> {
    let scores: Vec<bool> = filters
        .iter()
        .flat_map(|(_, predicates)| predicates.iter().map(|found| &found.expression))
        .chain(joins.iter().map(|join| &join.predicate))
        .map(|asked| asked_call(asked).kind == "score")
        .chain(
            projected
                .iter()
                .filter(|(call, _)| call.kind == "score")
                .map(|_| true),
        )
        .collect();
    if scores.contains(&true) && scores.contains(&false) {
        return unsupported("Scores and generative model calls must run in separate regions.");
    }
    Ok(())
}
#[derive(Debug, Clone)]
pub(super) struct JoinSpec {
    pub(super) predicate: Asked,
    pub(super) semantics: &'static str,
    pub(super) selectivity: Option<f64>,
    pub(super) anchor: Option<String>,
    pub(super) on: Vec<Equality>,
    pub(super) filter: Option<ValueExpression>,
}
