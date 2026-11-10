//! Bind tables, join predicates, and WHERE terms in written order.
//! Check which tables each AI join and gate reads.

use super::binding::{AnswerColumn, Binder, negated_inward, string_literal, unsupported};
use super::walk::{Kind, Walk};
use super::{ByAlias, push};
use crate::ai::{AiNode, Semantics, WrittenTest};
use crate::functions::column_call_kind;
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

/// The relation's terms, bound in the order the query wrote them, each
/// where it runs.
#[derive(Default)]
pub(super) struct Relation {
    /// alias -> its AI filters.
    pub(super) filters: ByAlias<FilterPredicate>,
    /// Each AI join and gate, by its node.
    pub(super) joins: BTreeMap<usize, JoinSpec>,
    /// alias -> its tests of a label: the label's column, the labels the
    /// test accepts, and the test's selectivity.
    pub(super) label_tests: ByAlias<(String, Value, Option<f64>)>,
    /// The conditions that ask no model, and their HINT's selectivity.
    pub(super) conditions: Vec<(ValueExpression, Option<f64>)>,
    /// alias -> the ON equalities of the JOIN of that table that no AI
    /// join has claimed, last JOIN first.
    pub(super) equalities: Vec<(String, Vec<Equality>)>,
    /// The JOINed tables whose ON equalities an AI join has claimed.
    claimed: Vec<String>,
    /// The FROM table, then the JOINed tables.
    pub(super) tables: Vec<String>,
    /// Whether a gate is not over a join, which the operators cannot
    /// read.
    pub(super) unplaced: bool,
    /// The answer of each AI filter that keeps every row, and its
    /// selectivity hint, for the guards that read it.
    answers: Vec<(String, Option<f64>)>,
    /// By answer, the `AI.IF` nodes that are terms of an OR.
    or_terms: BTreeMap<String, OrTerm>,
}

/// One `AI.IF` term of an OR.
#[derive(Clone, Default)]
struct OrTerm {
    /// The OR's terms that ask no model, if it has any.
    plain: Option<Expr>,
    /// Whether the OR reads the call's answer as `NOT`.
    negated: bool,
    /// The OR's name, when it has more than one `AI.IF`.
    disjunction: Option<String>,
    /// Whether the OR's last call is guarded on another call's answer:
    /// each call then asks only the rows no earlier call passed.
    short_circuit: bool,
}

/// Whether a term is `a.x = b.y`, an equality of two tables' columns.
fn is_column_equality(term: &Expr) -> bool {
    matches!(term, Expr::BinaryExpr(binary)
        if binary.op == datafusion::logical_expr::Operator::Eq
            && matches!((binary.left.as_ref(), binary.right.as_ref()),
                (Expr::Column(left), Expr::Column(right)) if left.relation != right.relation))
}

impl Relation {
    /// Bind the relation's tables and terms in the order the query wrote
    /// them: the tables, each JOIN's ON, the ON AI predicates, then WHERE.
    ///
    /// `names` maps an answer column to the name the SELECT list returns
    /// it by.
    pub(super) fn bind(
        walk: &Walk<'_>,
        binder: &mut Binder,
        names: &BTreeMap<String, String>,
    ) -> Result<Self, CompileError> {
        let mut relation = Self::default();
        for table in walk.tables.iter().filter(|table| !table.gate) {
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
            let mut equalities = Vec::new();
            for term in terms {
                let term = walk.through(&term);
                // DataFusion pushes a WHERE test of the two sides into the
                // join; an equality of their columns is the join's key
                if is_column_equality(&term) {
                    equalities.push(binder.bind_equality(&term, &joined)?);
                } else {
                    relation.bind_term(binder, &term, None)?;
                }
            }
            if !equalities.is_empty() {
                relation.equalities.insert(0, (joined, equalities));
            }
        }
        for id in walk.ids() {
            let Kind::Ai(node) = &walk.get(id).kind else {
                continue;
            };
            let Some(parts) = crate::rules::or_guards::or_parts(node)
                .filter(|_| node.function() == "ai_filter" && node.semantics == Semantics::Rows)
            else {
                continue;
            };
            // the OR is named by its last call's answer
            let disjunction = (!parts.others.is_empty()).then(|| node.answer.clone());
            let short_circuit = node.guard.as_ref().is_some_and(|guard| {
                guard.column_refs().iter().any(|column| {
                    parts
                        .others
                        .iter()
                        .any(|(answer, _)| *answer == column.name)
                })
            });
            let own = (node.answer.clone(), parts.negated);
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
                Kind::Ai(node) if node.is_predicate() || node.semantics != Semantics::Rows => {
                    Some((id, *node))
                }
                _ => None,
            })
            .collect();
        // the ON predicates by their JOINs, then WHERE's by their terms
        asked.sort_by_key(|(_, node)| {
            node.written
                .as_ref()
                .map_or((true, usize::MAX, usize::MAX), |written| match written.on {
                    Some(_) => (false, written.index, written.index),
                    None => (true, written.position, written.index),
                })
        });
        let (on, rest): (Vec<_>, Vec<_>) = asked.into_iter().partition(|(_, node)| node.in_on());
        // one question per JOIN: its pairs are asked once
        for (_, node) in &on {
            let written = node.written.as_ref().expect("an ON term's place");
            let num_asked = on
                .iter()
                .filter(|(_, other)| {
                    other.written.as_ref().map(|found| found.clause) == Some(written.clause)
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
        // each column node's tests: the terms no written test holds, in
        // the plan's order, then the written tests in written order
        let mut tests: Vec<&WrittenTest> = Vec::new();
        let mut terms: Vec<&Expr> = Vec::new();
        for id in walk.ids() {
            let Kind::Ai(node) = &walk.get(id).kind else {
                continue;
            };
            for test in &node.tests {
                if !tests
                    .iter()
                    .any(|found| (found.clause, found.position) == (test.clause, test.position))
                {
                    tests.push(test);
                }
            }
            if !node.is_predicate() && node.semantics == Semantics::Rows {
                terms.extend(node.filter.iter().flat_map(split_conjunction));
            }
        }
        // DataFusion may write one test as several terms
        terms.retain(|term| {
            !tests
                .iter()
                .any(|test| split_conjunction(&test.test).contains(term))
        });
        for term in terms {
            relation.bind_term(binder, &walk.through(term), None)?;
        }
        tests.sort_by_key(|test| (test.clause, test.position));
        for test in tests {
            // the lowering keeps a call's selectivity option on the test
            // as written, not on the node's call
            let (written, hint) = binder.unwrap_hint(&test.written)?;
            let hint = match hint {
                Some(hint) => Some(hint),
                None => binder.label_selectivity(&written)?,
            };
            relation.bind_term(binder, &walk.through(&test.test), hint)?;
        }
        Ok(relation)
    }

    /// Bind an AI filter, AI join, or gate.
    fn bind_asked(
        &mut self,
        walk: &Walk<'_>,
        binder: &mut Binder,
        id: usize,
        node: &AiNode,
    ) -> Result<(), CompileError> {
        let join = node.in_on().then_some(true);
        let call = walk.through(&node.call);
        if node.semantics != Semantics::Rows {
            let input = walk.get(walk.get(id).inputs[0]);
            let Kind::Join(_) = &input.kind else {
                self.unplaced = true;
                return Ok(());
            };
            let inner = walk.get(input.inputs[1]).tables.last().cloned();
            let table = walk
                .tables
                .iter()
                .find(|table| Some(&table.alias) == inner.as_ref())
                .expect("a gated table");
            binder.add_table(&table.alias, &table.provider, table.schema.clone())?;
            let (call, options) =
                binder.bind_ai_filter_call(&call, &["selectivity", "anchor"], Some(true))?;
            let anti = node.semantics == Semantics::Anti;
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
        // an AI.IF in an OR passes the rows where the OR's plain terms are TRUE
        let filter = node
            .filter
            .as_ref()
            .filter(|_| node.function() == "ai_filter");
        let term = match (filter, self.or_terms.get(&node.answer)) {
            (_, Some(term)) => term.clone(),
            (Some(filter), None) => match node.answer_term(filter) {
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
        // a guard on another call's answer is the OR's short circuit
        let guard = node.guard.as_ref().and_then(|guard| {
            conjunction(
                split_conjunction(guard)
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
        let answer = (node.function() == "ai_filter" && node.filter.is_none() && !in_or)
            .then(|| node.answer.clone());
        let (predicate, options) = if node.function() == "ai_score" {
            // the test compares the score's call with a threshold
            let term = walk
                .through(node.filter.as_ref().expect("a test"))
                .transform_up(|found| {
                    Ok(match &found {
                        Expr::Column(column)
                            if column.relation.is_none() && column.name == node.answer =>
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
                .guard
                .as_ref()
                .map(|found| crate::rules::guards::guard_selectivity(found, &self.answers));
            if let Some(answer) = &answer {
                self.answers.push((answer.clone(), options.selectivity));
            }
            // the call's selectivity is the fraction the model answers yes for
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
                walk.through(node.filter.as_ref().unwrap_or(&node.call))
            ));
        }
        let spec = JoinSpec {
            predicate,
            semantics: "full",
            selectivity: options.selectivity,
            anchor: options.anchor,
            on: Vec::new(),
        };
        let (claimed, equalities) = (&mut self.claimed, &mut self.equalities);
        let spec = join_spec(spec, &read, &self.tables, claimed, equalities)?;
        self.joins.insert(id, spec);
        Ok(())
    }

    /// Bind a term that asks no model: a test of one label, or a plain
    /// condition, one per part of an AND unless `hint`, a HINT's
    /// selectivity, estimates the whole term.
    fn bind_term(
        &mut self,
        binder: &mut Binder,
        term: &Expr,
        hint: Option<f64>,
    ) -> Result<(), CompileError> {
        let (term, written) = binder.unwrap_hint(term)?;
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

/// The answer column and tested values of an `=` or `IN` test of one
/// unqualified column, or None.
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
        // DataFusion writes a short IN list as an OR of equalities
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

/// Bind the call of each `AiNode` that adds columns, in the plan's
/// order, and record the columns with `binder`; an AI filter that keeps
/// every row adds its Boolean answer.
fn bind_answers(
    walk: &Walk<'_>,
    binder: &mut Binder,
    names: &BTreeMap<String, String>,
) -> Result<(), CompileError> {
    for id in walk.ids() {
        let Kind::Ai(node) = &walk.get(id).kind else {
            continue;
        };
        if node.semantics != Semantics::Rows {
            continue;
        }
        let name = names.get(&node.answer).unwrap_or(&node.answer).clone();
        let mut found = AnswerColumn {
            answer: node.answer.clone(),
            name: name.clone(),
            kind: "boolean",
            alias: None,
            calls: Vec::new(),
            fields: Vec::new(),
        };
        if node.is_predicate() {
            let tables: Vec<String> = walk.call_tables(&node.call).into_iter().collect();
            if node.function() != "ai_filter" || node.filter.is_some() || tables.len() != 1 {
                continue;
            }
            found.alias = tables.into_iter().next();
            found.name = node.answer.clone();
            binder.add_answer(found);
            continue;
        }
        let call = walk.through(&node.call);
        if node.function() == "ai_score" {
            found.kind = "score";
            found.calls = vec![(binder.bind_ai_value(&call)?, name)];
            binder.add_answer(found);
            continue;
        }
        let Some(kind) = column_call_kind(&node.call) else {
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
/// A list of names as Python writes a tuple of them.
fn repr_tuple(items: &[String]) -> String {
    let parts: Vec<String> = items.iter().map(|item| repr_str(item)).collect();
    match parts.as_slice() {
        [one] => format!("({one},)"),
        _ => format!("({})", parts.join(", ")),
    }
}

/// An EXISTS or NOT EXISTS gate over the table `inner`.
///
/// # Errors
///
/// The call does not read the inner table and one outer table, or the
/// anchor is not the outer table.
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
    })
}

/// An AI join over `aliases`, with the ON equalities of each JOINed
/// table it is the first to read.
///
/// # Errors
///
/// The predicate reads a table outside FROM and JOIN, its anchor is not
/// one of its tables, or an ON equality reads a table it does not.
fn join_spec(
    mut spec: JoinSpec,
    aliases: &[String],
    tables: &[String],
    claimed: &mut Vec<String>,
    equalities: &mut Vec<(String, Vec<Equality>)>,
) -> Result<JoinSpec, CompileError> {
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
    for alias in &tables[1..] {
        if aliases.contains(alias) && !claimed.contains(alias) {
            claimed.push(alias.clone());
            if let Some(index) = equalities.iter().position(|(found, _)| found == alias) {
                spec.on.extend(equalities.remove(index).1);
            }
        }
    }
    if !spec.on.is_empty() && aliases.len() != 2 {
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
    Ok(spec)
}

/// Refuse JOINs that no AI predicate covers, or that do not connect to
/// the FROM table.
pub(super) fn check_joins(
    equalities: &[(String, Vec<Equality>)],
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

/// Refuse AI.SCORE beside generative predicates.
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
/// An AI join or gate, before it is placed.
#[derive(Debug, Clone)]
pub(super) struct JoinSpec {
    pub(super) predicate: Asked,
    pub(super) semantics: &'static str,
    pub(super) selectivity: Option<f64>,
    pub(super) anchor: Option<String>,
    pub(super) on: Vec<Equality>,
}
