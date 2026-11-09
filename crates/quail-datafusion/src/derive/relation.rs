//! Bind tables, join predicates, and WHERE terms in written order.
//! Check which tables each AI join and gate reads.

use super::binding::{Binder, label_test, unsupported};
use super::walk::{Answer, Kind, Walk};
use super::{ByAlias, push};
use crate::ai::{AiNode, Semantics};
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::utils::split_conjunction;
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
    /// alias -> its tests of a label: the classification, the labels it
    /// accepts, and the test's selectivity.
    pub(super) label_tests: ByAlias<(ModelCall, Value, Option<f64>)>,
    /// The conditions that ask no model, and their HINT's selectivity.
    pub(super) conditions: Vec<(ValueExpression, Option<f64>)>,
    /// alias -> the ON equalities of the JOIN of that table that no AI
    /// join has claimed, last JOIN first.
    pub(super) equalities: Vec<(String, Vec<Equality>)>,
    /// The JOINed tables whose ON equalities an AI join has claimed.
    claimed: Vec<String>,
    /// The FROM table, then the JOINed tables.
    pub(super) tables: Vec<String>,
    /// Whether a term the lowering left in place binds, which the
    /// operators cannot read.
    pub(super) unplaced: bool,
    /// The answer of each AI filter that keeps every row, and its
    /// selectivity hint, for the guards that read it.
    answers: Vec<(String, Option<f64>)>,
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
    pub(super) fn bind(
        walk: &Walk<'_>,
        binder: &mut Binder,
        answers: &[Answer],
    ) -> Result<Self, CompileError> {
        let mut relation = Self::default();
        for table in walk.tables.iter().filter(|table| !table.gate) {
            binder.add_table(&table.alias, &table.provider, table.schema.clone())?;
            relation.tables.push(table.alias.clone());
        }
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
                let term = walk.written(&term, answers);
                // DataFusion pushes a WHERE test of the two sides into the
                // join; an equality of their columns is the join's key
                if is_column_equality(&term) {
                    equalities.push(binder.bind_equality(&term, &joined)?);
                } else {
                    relation.bind_where_term(binder, &term)?;
                }
            }
            if !equalities.is_empty() {
                relation.equalities.insert(0, (joined, equalities));
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
                return compile_error(format!(
                    "JOIN {} has {num_asked} AI predicates in ON; ask one question per JOIN",
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
        for term in walk.where_terms(answers) {
            relation.bind_where_term(binder, &term)?;
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
        let call = walk.written(&node.call, &[]);
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
        // an AI.IF in an OR passes the rows where the OR's other terms are TRUE
        let alternative = match &node.filter {
            Some(Expr::Column(column))
                if node.function() == "ai_filter"
                    && column.relation.is_none()
                    && column.name == node.answer =>
            {
                None
            }
            Some(filter)
                if node.function() == "ai_filter" && split_conjunction(filter).len() == 1 =>
            {
                match crate::rules::or_guards::or_terms(node) {
                    Some(rest) => {
                        let rest = walk.filter_answers(&rest, binder);
                        Some(binder.bind_condition(&rest)?)
                    }
                    None => {
                        self.unplaced = true;
                        return Ok(());
                    }
                }
            }
            _ => None,
        };
        let guard = match &node.guard {
            Some(guard) => {
                let guard = walk.filter_answers(guard, binder);
                Some(binder.bind_condition(&guard)?)
            }
            None => None,
        };
        let answer =
            (node.function() == "ai_filter" && node.filter.is_none()).then(|| node.answer.clone());
        let (predicate, options) = if node.function() == "ai_score" {
            let answer = Answer {
                name: node.answer.clone(),
                call,
            };
            let term = walk.written(node.filter.as_ref().expect("a test"), &[answer]);
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
            let filter = FilterPredicate {
                expression: predicate,
                selectivity: options.selectivity,
                guard,
                alternative,
                answer,
                guard_selectivity,
            };
            push(&mut self.filters, &read[0], filter);
            return Ok(());
        }
        if alternative.is_some() || guard.is_some() || answer.is_some() {
            return unsupported(&format!(
                "{}, an OR of an AI join and other tests,",
                walk.written(node.filter.as_ref().unwrap_or(&node.call), &[])
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

    /// Bind a WHERE term that is no AI filter: a plain condition, a test
    /// of a label, or a test of another column operator's answer.
    fn bind_where_term(&mut self, binder: &mut Binder, term: &Expr) -> Result<(), CompileError> {
        if let Some(bound) = binder.bind_plain_where_term(term)? {
            self.conditions.extend(bound);
            return Ok(());
        }
        let (unhinted, hint) = binder.unwrap_hint(term)?;
        if let Some((classify, items)) = label_test(&unhinted) {
            let (call, labels, selectivity) = binder.bind_label_test(classify, &items, hint)?;
            let read = call_aliases(&call);
            if read.len() != 1 {
                return compile_error(format!(
                    "a filter on a label tests a one-document classification; this one \
                     classifies pairs of {}",
                    repr_tuple(&read)
                ));
            }
            push(&mut self.label_tests, &read[0], (call, labels, selectivity));
            return Ok(());
        }
        // the binder's refusal of the term, or a term it reads that the
        // lowering left in place
        binder.refuse_where_term(term)?;
        self.unplaced = true;
        Ok(())
    }
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
            return compile_error(
                "the EXISTS predicate must reference the inner provider and exactly one outer \
                 provider",
            );
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
        return compile_error(format!(
            "join conditions are supported on two-table AI predicates; this one names {}",
            repr_tuple(aliases)
        ));
    }
    for condition in &spec.on {
        let sides = [&condition.left, &condition.right].map(column_alias_of);
        if !sides.iter().all(|side| aliases.contains(side)) {
            return compile_error(format!(
                "join condition {} names a table the AI predicate over {} does not",
                equality_text(condition),
                repr_tuple(aliases)
            ));
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
            return compile_error(format!(
                "JOIN {alias} ON {first} projects a pair AI.SCORE; a projected pair score runs \
                 over CROSS JOIN, and an ON equality is supported when the score is compared \
                 in ON"
            ));
        }
        return unsupported(&format!(
            "JOIN {alias} ON {first} without an AI predicate over its pairs"
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
            "JOINed tables {} with no AI predicate over their pairs",
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
        return compile_error(format!(
            "join predicates do not connect {} to the FROM table {}: the predicates' tables \
             must form one connected graph with it",
            repr_list(&disconnected),
            repr_str(root)
        ));
    }
    Ok(())
}

/// Refuse AI.SCORE beside generative predicates, and a query that asks
/// no model.
pub(super) fn check_calls(
    filters: &ByAlias<FilterPredicate>,
    joins: &[&JoinSpec],
    label_tests: &ByAlias<(ModelCall, Value, Option<f64>)>,
    projected: &[(&ModelCall, String)],
    tested: &[(ModelCall, String)],
    allow_relational: bool,
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
        return compile_error(
            "AI.SCORE cannot be mixed with generative AI predicates in one query",
        );
    }
    let asks = !(joins.is_empty()
        && filters.is_empty()
        && label_tests.is_empty()
        && projected.is_empty()
        && tested.is_empty());
    if !(allow_relational || asks) {
        return compile_error(
            "the query has no AI predicate; a plain scan belongs in the database the ids came \
             from",
        );
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
