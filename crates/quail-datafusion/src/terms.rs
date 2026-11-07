//! Placing a region's bound ON and WHERE terms into each table's
//! operators, the joins, and the tests of result rows.
//!
//! [`crate::derive`] binds the terms in written order and hands each to
//! [`RelationTerms`]: a condition that asks no model, an AI filter, a
//! test of a label, an AI join, or an EXISTS gate. [`RelationTerms::content`]
//! names each table's column operators and places each condition at its
//! scan, after the operator that computes the columns it reads, or on
//! the result rows.

use std::collections::{BTreeMap, BTreeSet};

use quail_physical::{
    Asked, ColumnRef, Condition, Connective, Equality, FilterPredicate, GetField, ModelCall, Not,
    RegularPredicate, ScalarFunction, SortExpression, SortKey, TaggedModelCall, ValueExpression,
};
use quail_planner::expressions::{
    CompileError, SPAN_SUFFIX, aliased_call, asked_call, call_aliases, column_alias_of,
    column_name, column_references, compile_error, condition_text, conjuncts, equality_text,
    normalize, reference_name, repr_list, repr_str, tests, validate_condition,
};
use quail_planner::operators::{Scan, ScanColumn, Schema};
use serde_json::Value;

// ---------------------------------------------------------------- input

/// Values kept per table alias, in the order the aliases first appear.
pub(crate) type ByAlias<T> = Vec<(String, Vec<T>)>;

/// A test of a label: the classification, the accepted labels, and its selectivity.
type LabelTest = (ModelCall, Value, Option<f64>);

// ---------------------------------------------------------------- placement

/// The parts of a query's conditions, by where they run.
#[derive(Debug, Default)]
pub(crate) struct PlacedConditions {
    pub(crate) scans: Vec<(String, Vec<Condition>)>,
    pub(crate) computed: ByAlias<(Condition, Option<f64>)>,
    pub(crate) rows: Vec<Condition>,
}

/// Add `value` to `key`'s values, keeping keys in first use order.
pub(crate) fn push<K: PartialEq, V>(map: &mut Vec<(K, Vec<V>)>, key: K, value: V) {
    match map.iter_mut().find(|(found, _)| *found == key) {
        Some((_, values)) => values.push(value),
        None => map.push((key, vec![value])),
    }
}

fn lookup<'a, K: PartialEq, V>(map: &'a [(K, Vec<V>)], key: &K) -> &'a [V] {
    map.iter()
        .find(|(found, _)| found == key)
        .map(|(_, values)| values.as_slice())
        .unwrap_or(&[])
}

/// Split conditions at AND and place each part where its columns exist.
///
/// `computed` maps (alias, column) to (kind, name) of the operator
/// computing it.
pub(crate) fn place_conditions(
    conditions: Vec<(Condition, Option<f64>)>,
    computed: &BTreeMap<(String, String), (String, String)>,
    unnested: &[String],
) -> Result<PlacedConditions, CompileError> {
    let mut placed = PlacedConditions::default();
    for (condition, selectivity) in conditions {
        let condition = normalize(condition);
        let parts = if selectivity.is_none() {
            conjuncts(&condition)
        } else {
            vec![condition]
        };
        for part in parts {
            validate_condition(&part)?;
            let references: Vec<ColumnRef> = tests(&part)
                .into_iter()
                .flat_map(|test| column_references(&test.expression))
                .collect();
            let aliases: BTreeSet<String> =
                references.iter().map(|found| found.alias.clone()).collect();
            let owners: BTreeSet<Option<(String, String)>> = references
                .iter()
                .map(|found| {
                    computed
                        .get(&(found.alias.clone(), found.column.clone()))
                        .cloned()
                })
                .collect();
            let owned: BTreeSet<&(String, String)> = owners.iter().flatten().collect();
            let kinds: BTreeSet<&String> = owned.iter().map(|(kind, _)| kind).collect();
            let unnested_alias = aliases.iter().any(|alias| unnested.contains(alias));
            let only_none = owners.len() == 1 && owners.contains(&None);
            if aliases.len() == 1 && only_none && !unnested_alias {
                if selectivity.is_some() {
                    return compile_error(format!(
                        "HINT estimates a test of a label or an answer; {} tests a source \
                         column, applied at the scan",
                        condition_text(&part)
                    ));
                }
                let alias = aliases.into_iter().next().expect("one alias");
                push(&mut placed.scans, alias, part);
            } else if aliases.len() == 1 && !unnested_alias && kinds.len() == 1 {
                let alias = aliases.into_iter().next().expect("one alias");
                push(&mut placed.computed, alias, (part, selectivity));
            } else {
                if kinds.contains(&"label".to_string()) && owned.len() > 1 {
                    return compile_error(format!(
                        "{} tests a label and another operator's column; test them in \
                         separate conditions",
                        condition_text(&part)
                    ));
                }
                if selectivity.is_some() {
                    return compile_error(format!(
                        "HINT estimates a test that runs before a model call; {} runs on the \
                         result rows, after every model call",
                        condition_text(&part)
                    ));
                }
                placed.rows.push(part);
            }
        }
    }
    Ok(placed)
}

fn rewrite_reference(
    expression: &ValueExpression,
    members: &BTreeMap<String, (String, String)>,
) -> ValueExpression {
    match expression {
        ValueExpression::ColumnRef(found) if members.contains_key(&found.column) => {
            let (owner, name) = &members[&found.column];
            ValueExpression::ColumnRef(ColumnRef {
                alias: owner.clone(),
                provider: String::new(),
                column: name.clone(),
            })
        }
        ValueExpression::GetField(field) => ValueExpression::GetField(GetField {
            input: Box::new(rewrite_reference(&field.input, members)),
            field: field.field.clone(),
        }),
        ValueExpression::ScalarFunction(function) => {
            ValueExpression::ScalarFunction(ScalarFunction {
                function: function.function.clone(),
                args: function
                    .args
                    .iter()
                    .map(|argument| rewrite_reference(argument, members))
                    .collect(),
            })
        }
        ValueExpression::Alias(alias) => ValueExpression::Alias(quail_physical::Alias {
            expression: Box::new(rewrite_reference(&alias.expression, members)),
            name: alias.name.clone(),
        }),
        other => other.clone(),
    }
}

/// Rewrite a condition to read result rows, where struct columns hold answers.
pub(crate) fn result_row_condition(
    condition: &Condition,
    packing: &[(String, Vec<(String, String)>)],
) -> Condition {
    let members: BTreeMap<String, (String, String)> = packing
        .iter()
        .flat_map(|(owner, names)| {
            names
                .iter()
                .map(move |(name, member)| (member.clone(), (owner.clone(), name.clone())))
        })
        .collect();
    rewrite_condition(condition, &members)
}

fn rewrite_condition(
    condition: &Condition,
    members: &BTreeMap<String, (String, String)>,
) -> Condition {
    match condition {
        Condition::Compare(test) => Condition::Compare(Box::new(RegularPredicate {
            expression: rewrite_reference(&test.expression, members),
            comparison: test.comparison.clone(),
            value: test.value.clone(),
        })),
        Condition::Not(not) => Condition::Not(Not {
            operand: Box::new(rewrite_condition(&not.operand, members)),
        }),
        Condition::And(connective) => Condition::And(Connective {
            operands: connective
                .operands
                .iter()
                .map(|operand| rewrite_condition(operand, members))
                .collect(),
        }),
        Condition::Or(connective) => Condition::Or(Connective {
            operands: connective
                .operands
                .iter()
                .map(|operand| rewrite_condition(operand, members))
                .collect(),
        }),
    }
}

/// The result row columns a rewritten condition reads, by name.
pub(crate) fn result_columns_read(condition: &Condition) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for test in tests(condition) {
        for reference in column_references(&test.expression) {
            let name = reference_name(&reference, true);
            if !found.contains(&name) {
                found.push(name);
            }
        }
    }
    found
}

// ---------------------------------------------------------------- operators

/// A join predicate, before assembly.
#[derive(Debug, Clone)]
pub(crate) struct JoinSpec {
    pub(crate) predicate: Asked,
    pub(crate) semantics: String,
    pub(crate) selectivity: Option<f64>,
    pub(crate) anchor: Option<String>,
    pub(crate) on: Vec<Equality>,
}

/// One table's operators.
#[derive(Debug, Default, Clone)]
pub(crate) struct ScanOperators {
    pub(crate) predicates: Vec<FilterPredicate>,
    pub(crate) labels: Vec<(ModelCall, String)>,
    pub(crate) regular_predicates: Vec<Condition>,
    pub(crate) extracts: Vec<(ModelCall, String)>,
    pub(crate) computed_filters: Vec<(Condition, Option<f64>)>,
    pub(crate) sae_features: Vec<(ModelCall, String)>,
    pub(crate) maps: Vec<(ModelCall, String)>,
}

fn check_labels(condition: &Condition, calls: &[(String, ModelCall)]) -> Result<(), CompileError> {
    for test in tests(condition) {
        let references = column_references(&test.expression);
        if !["=", "in"].contains(&test.comparison.as_str()) || references.len() != 1 {
            continue;
        }
        let name = &references[0].column;
        let Some((_, call)) = calls.iter().find(|(found, _)| found == name) else {
            continue;
        };
        let values = match &test.value {
            Value::Array(items) => items.clone(),
            other => vec![other.clone()],
        };
        let labels: Vec<Value> = match &call.labels {
            Value::Array(items) => items.clone(),
            _ => Vec::new(),
        };
        let mut unknown: Vec<String> = values
            .iter()
            .filter(|value| !labels.contains(value))
            .filter_map(|value| value.as_str().map(str::to_string))
            .collect();
        unknown.sort();
        unknown.dedup();
        if !unknown.is_empty() {
            let written: Vec<String> = labels
                .iter()
                .filter_map(|label| label.as_str().map(str::to_string))
                .collect();
            return compile_error(format!(
                "the filter on {} lists {}, which are not among its labels {}",
                repr_str(name),
                repr_list(&unknown),
                repr_list(&written)
            ));
        }
    }
    Ok(())
}

/// The source columns the row tests read that the SELECT list does not
/// return, which it adds for them.
pub(crate) fn row_test_columns(
    row_tests: &[Condition],
    unnested: &BTreeSet<String>,
    source: &[ValueExpression],
    returned: &[String],
) -> Vec<ValueExpression> {
    let mut present: BTreeSet<String> = returned.iter().cloned().collect();
    present.extend(source.iter().map(|column| column_name(column, true)));
    let mut extra = Vec::new();
    for test in row_tests {
        for found in tests(test) {
            for reference in column_references(&found.expression) {
                let name = reference_name(&reference, true);
                if !reference.provider.is_empty()
                    && !unnested.contains(&reference.alias)
                    && !present.contains(&name)
                {
                    present.insert(name);
                    extra.push(ValueExpression::ColumnRef(reference));
                }
            }
        }
    }
    extra
}

/// Refuse a DISTINCT or ORDER BY that compares AI.SAE_FEATURES lists.
///
/// # Errors
///
/// The query compares a feature list that is not unnested.
pub(crate) fn check_feature_lists(
    columns: &[ValueExpression],
    order: &[SortKey],
    distinct: bool,
    unnested: &BTreeSet<String>,
) -> Result<(), CompileError> {
    let names: BTreeSet<String> = columns
        .iter()
        .filter(|column| aliased_call(column, &["features"]).is_some())
        .map(|column| column_name(column, true))
        .filter(|name| !unnested.contains(name))
        .collect();
    if names.is_empty() {
        return Ok(());
    }
    if distinct {
        return compile_error("DISTINCT does not compare AI.SAE_FEATURES list columns");
    }
    for key in order {
        let refused = match &key.expression {
            SortExpression::Value(expression) => {
                let named = matches!(expression, ValueExpression::Alias(alias)
                    if names.contains(&alias.name));
                named
                    || aliased_call(expression, &["features"]).is_some()
                    || matches!(expression, ValueExpression::ModelCall(call)
                        if call.kind == "features")
            }
        };
        if refused {
            return compile_error("ORDER BY does not compare AI.SAE_FEATURES list columns");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- relation

fn gate(
    call: &ModelCall,
    inner: &str,
    anti: bool,
    anchor: Option<&String>,
    selectivity: Option<f64>,
) -> Result<JoinSpec, CompileError> {
    let inner = inner.to_string();
    let aliases = call_aliases(call);
    if aliases.len() != 2 || !aliases.contains(&inner) {
        return compile_error(
            "the EXISTS predicate must reference the inner provider and exactly one outer \
             provider",
        );
    }
    let outer = aliases
        .iter()
        .find(|alias| **alias != inner)
        .cloned()
        .expect("an outer alias");
    if let Some(anchor) = anchor
        && *anchor != outer
    {
        return compile_error(format!(
            "exists/anti always anchor on the outer table {} - the gate applies to its \
             documents - got anchor {}",
            repr_str(&outer),
            repr_str(anchor)
        ));
    }
    Ok(JoinSpec {
        predicate: Asked::Call(call.clone()),
        semantics: if anti { "anti" } else { "exists" }.to_string(),
        selectivity,
        anchor: Some(outer),
        on: Vec::new(),
    })
}

/// A list of names as Python writes a tuple of them.
pub(crate) fn repr_tuple(items: &[String]) -> String {
    let parts: Vec<String> = items.iter().map(|item| repr_str(item)).collect();
    if parts.len() == 1 {
        format!("({},)", parts[0])
    } else {
        format!("({})", parts.join(", "))
    }
}

#[allow(clippy::too_many_arguments)]
fn join_spec(
    predicate: Asked,
    selectivity: Option<f64>,
    anchor: Option<String>,
    aliases: &[String],
    base: &str,
    joined: &[String],
    claimed: &mut Vec<String>,
    equalities: &mut Vec<(String, Vec<Equality>)>,
) -> Result<JoinSpec, CompileError> {
    let mut allowed: Vec<String> = vec![base.to_string()];
    allowed.extend(joined.iter().cloned());
    let outside: Vec<String> = aliases
        .iter()
        .filter(|alias| !allowed.contains(alias))
        .cloned()
        .collect();
    if !outside.is_empty() {
        let mut sorted = allowed.clone();
        sorted.sort();
        sorted.dedup();
        return compile_error(format!(
            "the join prompt references {}, which are not the FROM table or JOINed tables of \
             this query ({})",
            repr_list(&outside),
            repr_list(&sorted)
        ));
    }
    if let Some(anchor) = &anchor
        && !aliases.contains(anchor)
    {
        return compile_error(format!(
            "anchor {} is not a table of this join ({})",
            repr_str(anchor),
            repr_tuple(aliases)
        ));
    }
    let news: Vec<String> = joined
        .iter()
        .filter(|alias| aliases.contains(alias) && !claimed.contains(alias))
        .cloned()
        .collect();
    claimed.extend(news.iter().cloned());
    let mut on = Vec::new();
    for alias in &news {
        if let Some(index) = equalities.iter().position(|(found, _)| found == alias) {
            on.extend(equalities.remove(index).1);
        }
    }
    if !on.is_empty() && aliases.len() != 2 {
        return compile_error(format!(
            "join conditions are supported on two-table AI predicates; this one names {}",
            repr_tuple(aliases)
        ));
    }
    for condition in &on {
        let sides = [
            column_alias_of(&condition.left),
            column_alias_of(&condition.right),
        ];
        if !sides.iter().all(|side| aliases.contains(side)) {
            return compile_error(format!(
                "join condition {} names a table the AI predicate over {} does not",
                equality_text(condition),
                repr_tuple(aliases)
            ));
        }
    }
    Ok(JoinSpec {
        predicate,
        semantics: "full".to_string(),
        selectivity,
        anchor,
        on,
    })
}

fn check_joins(
    equalities: &[(String, Vec<Equality>)],
    root: &str,
    joined: &[String],
    joins: &[JoinSpec],
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
        return compile_error(format!(
            "JOIN {alias} ON {first} without an AI predicate over its pairs {}",
            quail_planner::expressions::UNSUPPORTED
        ));
    }
    if joined.is_empty() {
        return Ok(());
    }
    let mut predicates: Vec<BTreeSet<String>> = joins
        .iter()
        .filter(|join| join.semantics == "full")
        .map(|join| {
            asked_call(&join.predicate)
                .prompt
                .args
                .iter()
                .map(quail_planner::expressions::argument_alias)
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
    let uncovered: Vec<String> = joined
        .iter()
        .filter(|alias| !predicates.iter().any(|found| found.contains(*alias)))
        .cloned()
        .collect();
    if !uncovered.is_empty() {
        return compile_error(format!(
            "JOINed tables {} with no AI predicate over their pairs {}",
            repr_list(&uncovered),
            quail_planner::expressions::UNSUPPORTED
        ));
    }
    let mut reached: BTreeSet<String> = [root.to_string()].into_iter().collect();
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
    let mut disconnected: Vec<String> = joined
        .iter()
        .filter(|alias| !reached.contains(*alias))
        .cloned()
        .collect();
    disconnected.sort();
    if !disconnected.is_empty() {
        return compile_error(format!(
            "join predicates do not connect {} to the FROM table {}: the predicates' tables \
             must form one connected graph with it",
            repr_list(&disconnected),
            repr_str(root)
        ));
    }
    Ok(())
}

fn is_score(asked: &Asked) -> bool {
    asked_call(asked).kind == "score"
}

/// Each table's operators, the joins, and the row tests of one SELECT's
/// relation, before they are put in order.
#[derive(Debug, Clone)]
pub(crate) struct RelationContent {
    /// Each table with its operators, in FROM and JOIN order.
    pub(crate) tables: Vec<TableContent>,
    pub(crate) joins: Vec<JoinSpec>,
    /// Classifications of joined rows, with their names.
    pub(crate) joined_classifications: Vec<(ModelCall, String)>,
    /// Tests of more than one table's columns, on the result rows.
    pub(crate) rows: Vec<Condition>,
}

/// One table and its operators.
#[derive(Debug, Clone)]
pub(crate) struct TableContent {
    pub(crate) alias: String,
    pub(crate) provider: String,
    pub(crate) column: ScanColumn,
    pub(crate) schema: Option<Schema>,
    pub(crate) operators: ScanOperators,
}

/// The ON and WHERE terms of one SELECT's relation, bound and taken in
/// written order, before each table's operators are named and placed.
#[derive(Debug)]
pub(crate) struct RelationTerms {
    /// The Scan of each table in FROM and JOIN order, then each gate's.
    scans: Vec<Scan>,
    /// alias -> the ON equalities of the JOIN of that table, last JOIN first.
    equalities: Vec<(String, Vec<Equality>)>,
    joined: Vec<String>,
    base: String,
    filters: Vec<(String, Vec<FilterPredicate>)>,
    joins: Vec<JoinSpec>,
    label_tests: ByAlias<LabelTest>,
    conditions: Vec<(Condition, Option<f64>)>,
    claimed: Vec<String>,
}

impl RelationTerms {
    /// No terms yet over `scans`, the tables in FROM and JOIN order,
    /// whose JOINs have `equalities`, last JOIN first.
    pub(crate) fn new(scans: Vec<Scan>, equalities: Vec<(String, Vec<Equality>)>) -> Self {
        let joined: Vec<String> = scans
            .iter()
            .skip(1)
            .map(|scan| scan.alias.clone())
            .collect();
        let base = scans[0].alias.clone();
        Self {
            scans,
            equalities,
            joined,
            base,
            filters: Vec::new(),
            joins: Vec::new(),
            label_tests: Vec::new(),
            conditions: Vec::new(),
            claimed: Vec::new(),
        }
    }

    /// Take an EXISTS or NOT EXISTS gate over the table `relation` scans.
    ///
    /// # Errors
    ///
    /// The call does not read the inner table and one outer table, or
    /// the anchor is not the outer table.
    pub(crate) fn gate(
        &mut self,
        call: &ModelCall,
        relation: Scan,
        anti: bool,
        anchor: Option<&String>,
        selectivity: Option<f64>,
    ) -> Result<(), CompileError> {
        let spec = gate(call, &relation.alias, anti, anchor, selectivity);
        self.scans.push(relation);
        self.joins.push(spec?);
        Ok(())
    }

    /// Take a condition that asks no model.
    pub(crate) fn condition(&mut self, condition: Condition, selectivity: Option<f64>) {
        self.conditions.push((condition, selectivity));
    }

    /// Take a model call or a comparison of one: an AI filter, a label
    /// test, or an AI join.
    ///
    /// # Errors
    ///
    /// The call does not fit the query's tables.
    pub(crate) fn asked(
        &mut self,
        predicate: Asked,
        selectivity: Option<f64>,
        anchor: Option<String>,
    ) -> Result<(), CompileError> {
        let call = asked_call(&predicate).clone();
        if let Asked::Compare(test) = &predicate
            && call.kind == "label"
        {
            let aliases = call_aliases(&call);
            if aliases.len() != 1 {
                return compile_error(format!(
                    "a filter on a label tests a one-document classification; this one \
                     classifies pairs of {}",
                    repr_tuple(&aliases)
                ));
            }
            push(
                &mut self.label_tests,
                aliases[0].clone(),
                (call.clone(), test.value.clone(), selectivity),
            );
            return Ok(());
        }
        let aliases = call_aliases(&call);
        if aliases.len() == 1 {
            if anchor.is_some() {
                return compile_error(
                    "anchor is a join option; a one-provider AI_FILTER takes only \
                     selectivity",
                );
            }
            push(
                &mut self.filters,
                aliases[0].clone(),
                FilterPredicate {
                    expression: predicate,
                    selectivity,
                },
            );
        } else {
            self.joins.push(join_spec(
                predicate,
                selectivity,
                anchor,
                &aliases,
                &self.base,
                &self.joined,
                &mut self.claimed,
                &mut self.equalities,
            )?);
        }
        Ok(())
    }

    /// Name each table's operators and place the conditions.
    ///
    /// `columns` is the SELECT list, `tested` the (call, name) of each
    /// column operator only a test reads, `schemas` each provider's
    /// schema, and `unnested` the aliases of unnested lists.
    ///
    /// # Errors
    ///
    /// A part does not fit the query's tables, or the query has no AI
    /// predicate where it needs one.
    pub(crate) fn content(
        self,
        columns: &[ValueExpression],
        tested: &[(TaggedModelCall, String)],
        schemas: &BTreeMap<String, Option<Schema>>,
        unnested: &[String],
        allow_relational: bool,
    ) -> Result<RelationContent, CompileError> {
        let Self {
            scans,
            equalities,
            joined,
            base,
            filters,
            joins,
            label_tests,
            mut conditions,
            ..
        } = self;
        let projected: Vec<(&ModelCall, String)> = columns
            .iter()
            .filter_map(|column| {
                let call = aliased_call(column, &[])?;
                Some((call, column_name(column, true)))
            })
            .collect();
        check_joins(&equalities, &base, &joined, &joins, &projected)?;
        let tested: Vec<(ModelCall, String)> = tested
            .iter()
            .map(|(TaggedModelCall::Call(call), name)| (call.clone(), name.clone()))
            .collect();
        check_calls(
            &filters,
            &joins,
            &label_tests,
            &projected,
            &tested,
            allow_relational,
        )?;
        let (named, labels) = label_names(&label_tests, &filters, &projected, &tested);
        let mut computing: BTreeMap<&str, ByAlias<(ModelCall, String)>> = BTreeMap::new();
        for kind in ["extract", "features", "map"] {
            computing.insert(kind, Vec::new());
        }
        let entries = projected
            .iter()
            .map(|(call, name)| ((*call).clone(), name.clone()))
            .chain(tested.iter().cloned());
        for (call, name) in entries {
            if let Some(by_alias) = computing.get_mut(call.kind.as_str()) {
                let alias = call_aliases(&call)[0].clone();
                let entry = (call.clone(), name.clone());
                match by_alias.iter_mut().find(|(found, _)| *found == alias) {
                    Some((_, calls)) => {
                        if !calls.contains(&entry) {
                            calls.push(entry);
                        }
                    }
                    None => by_alias.push((alias, vec![entry])),
                }
            }
        }
        let providers: BTreeMap<String, String> = scans
            .iter()
            .map(|scan| (scan.alias.clone(), scan.provider.clone()))
            .collect();
        for (alias, found) in &label_tests {
            for (call, accepted, selectivity) in found {
                let name = labels
                    .iter()
                    .find(|(owner, _)| owner == alias)
                    .and_then(|(_, calls)| calls.iter().find(|(found, _)| found == call))
                    .map(|(_, name)| name.clone())
                    .expect("a label column name");
                conditions.push((
                    Condition::Compare(Box::new(RegularPredicate {
                        expression: ValueExpression::ColumnRef(ColumnRef {
                            alias: alias.clone(),
                            provider: providers[alias].clone(),
                            column: name,
                        }),
                        comparison: "in".to_string(),
                        value: python_tuple(accepted),
                    })),
                    *selectivity,
                ));
            }
        }
        let mut computed: BTreeMap<(String, String), (String, String)> = BTreeMap::new();
        for (alias, calls) in &computing["extract"] {
            for (_, name) in calls {
                for column in [name.clone(), format!("{name}{SPAN_SUFFIX}")] {
                    computed.insert(
                        (alias.clone(), column),
                        ("extract".to_string(), name.clone()),
                    );
                }
            }
        }
        for kind in ["features", "map"] {
            for (alias, calls) in &computing[kind] {
                for (_, name) in calls {
                    computed.insert(
                        (alias.clone(), name.clone()),
                        (kind.to_string(), name.clone()),
                    );
                }
            }
        }
        for (alias, named_calls) in &labels {
            for (_, name) in named_calls {
                computed.insert(
                    (alias.clone(), name.clone()),
                    ("label".to_string(), name.clone()),
                );
            }
        }
        let placed = place_conditions(conditions, &computed, unnested)?;
        let tables = scans
            .iter()
            .map(|scan| {
                let of = |kind: &str| -> Vec<(ModelCall, String)> {
                    lookup(&computing[kind], &scan.alias).to_vec()
                };
                TableContent {
                    alias: scan.alias.clone(),
                    provider: scan.provider.clone(),
                    column: scan.column.clone(),
                    schema: schemas.get(&scan.provider).cloned().flatten(),
                    operators: ScanOperators {
                        predicates: lookup(&filters, &scan.alias).to_vec(),
                        labels: lookup(&labels, &scan.alias).to_vec(),
                        regular_predicates: lookup(&placed.scans, &scan.alias).to_vec(),
                        extracts: of("extract"),
                        computed_filters: lookup(&placed.computed, &scan.alias).to_vec(),
                        sae_features: of("features"),
                        maps: of("map"),
                    },
                }
            })
            .collect::<Vec<TableContent>>();
        for table in &tables {
            let calls: Vec<(String, ModelCall)> = table
                .operators
                .labels
                .iter()
                .map(|(call, name)| (name.clone(), call.clone()))
                .collect();
            for (condition, _) in &table.operators.computed_filters {
                check_labels(condition, &calls)?;
            }
        }
        Ok(RelationContent {
            tables,
            joins,
            joined_classifications: named
                .into_iter()
                .filter(|(call, _)| call_aliases(call).len() == 2)
                .collect(),
            rows: placed.rows,
        })
    }
}

/// Python's `tuple(value)`: a list stays, a string becomes its characters.
fn python_tuple(value: &Value) -> Value {
    match value {
        Value::String(text) => {
            Value::Array(text.chars().map(|c| Value::String(c.to_string())).collect())
        }
        other => other.clone(),
    }
}

type LabelNames = (
    Vec<(ModelCall, String)>,
    Vec<(String, Vec<(ModelCall, String)>)>,
);

/// Name each classification that SELECT returns or WHERE tests.
fn label_names(
    label_tests: &[(String, Vec<LabelTest>)],
    filters: &[(String, Vec<FilterPredicate>)],
    projected: &[(&ModelCall, String)],
    tested: &[(ModelCall, String)],
) -> LabelNames {
    let mut named: Vec<(ModelCall, String)> = Vec::new();
    for (call, name) in projected {
        if call.kind == "label" {
            set_entry(&mut named, (*call).clone(), name.clone(), true);
        }
    }
    let mut labels: Vec<(String, Vec<(ModelCall, String)>)> = Vec::new();
    for (alias, found) in label_tests {
        let asks = lookup(filters, alias).len();
        for (index, (call, _, _)) in found.iter().enumerate() {
            let name = named
                .iter()
                .find(|(owner, _)| owner == call)
                .map(|(_, name)| name.clone())
                .or_else(|| {
                    tested
                        .iter()
                        .rev()
                        .find(|(owner, _)| owner == call)
                        .map(|(_, name)| name.clone())
                })
                .unwrap_or_else(|| format!("__label_{alias}_{}", asks + index));
            set_default(&mut labels, alias, call, name);
        }
    }
    let mut every = named.clone();
    for (call, name) in tested {
        set_entry(&mut every, call.clone(), name.clone(), true);
    }
    for (call, name) in every {
        let aliases = call_aliases(&call);
        if call.kind == "label" && aliases.len() == 1 {
            set_default(&mut labels, &aliases[0], &call, name);
        }
    }
    (named, labels)
}

/// Python's `dict[key] = value`, keeping the first position.
fn set_entry(map: &mut Vec<(ModelCall, String)>, key: ModelCall, value: String, replace: bool) {
    match map.iter_mut().find(|(found, _)| *found == key) {
        Some(entry) => {
            if replace {
                entry.1 = value;
            }
        }
        None => map.push((key, value)),
    }
}

/// Python's `labels.setdefault(alias, {}).setdefault(call, name)`.
fn set_default(
    labels: &mut Vec<(String, Vec<(ModelCall, String)>)>,
    alias: &str,
    call: &ModelCall,
    name: String,
) {
    let position = match labels.iter().position(|(found, _)| found == alias) {
        Some(position) => position,
        None => {
            labels.push((alias.to_string(), Vec::new()));
            labels.len() - 1
        }
    };
    set_entry(&mut labels[position].1, call.clone(), name, false);
}

fn check_calls(
    filters: &[(String, Vec<FilterPredicate>)],
    joins: &[JoinSpec],
    label_tests: &[(String, Vec<LabelTest>)],
    projected: &[(&ModelCall, String)],
    tested: &[(ModelCall, String)],
    allow_relational: bool,
) -> Result<(), CompileError> {
    let mut flags: Vec<bool> = filters
        .iter()
        .flat_map(|(_, predicates)| predicates.iter())
        .map(|predicate| is_score(&predicate.expression))
        .collect();
    flags.extend(joins.iter().map(|join| is_score(&join.predicate)));
    flags.extend(
        projected
            .iter()
            .filter(|(call, _)| call.kind == "score")
            .map(|_| true),
    );
    if flags.iter().any(|flag| *flag) && !flags.iter().all(|flag| *flag) {
        return compile_error(
            "AI.SCORE cannot be mixed with generative AI predicates in one query",
        );
    }
    if !(allow_relational
        || !joins.is_empty()
        || !filters.is_empty()
        || !label_tests.is_empty()
        || !projected.is_empty()
        || !tested.is_empty())
    {
        return compile_error(
            "the query has no AI predicate; a plain scan belongs in the database the ids came \
             from",
        );
    }
    Ok(())
}
