//! Placing a bound query's conditions and building its logical plan, as
//! `quail/logical/naive.py`, `placement.py`, and `builder.py` do.
//!
//! The input is what the SQL front end binds: the scans in FROM and
//! JOIN order, the ON equalities, the ON and WHERE parts in written
//! order, and the arguments of the plan's result nodes.

use std::collections::{BTreeMap, BTreeSet};

use quail_physical::{
    AggregateCall, Asked, ColumnRef, Condition, Connective, Equality, ExtractStruct,
    FilterPredicate, GetField, HavingTest, ModelCall, Not, Projected, RegularPredicate,
    ScalarFunction, SortExpression, SortKey, TaggedModelCall, ValueExpression, WindowCall,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::expressions::{
    CompileError, SPAN_SUFFIX, aliased_call, call_aliases, column_name, column_references,
    compile_error, condition_columns, condition_text, conjuncts, normalize, reference_name,
    repr_list, repr_str, tests, validate_condition,
};
use crate::logical::{
    Aggregate, Filter, Join, LogicalNode, LogicalPlan, Project, ResultNode, Scan, ScanColumn,
    Schema, SemanticColumn, SemanticFilter, SemanticJoin, Unnest, Window,
};
use crate::nodes::{
    asked_call, equality_text, output_schema, readable_columns, result_columns, validate_node,
    validate_plan,
};

// ---------------------------------------------------------------- input

/// Values kept per table alias, in the order the aliases first appear.
pub type ByAlias<T> = Vec<(String, Vec<T>)>;

/// A test of a label: the classification, the accepted labels, and its selectivity.
type LabelTest = (ModelCall, Value, Option<f64>);

/// A part of a condition with the options written on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hint {
    pub condition: Box<Part>,
    pub selectivity: Option<f64>,
    pub anchor: Option<String>,
}

/// An EXISTS or NOT EXISTS gate: some row of a table makes a model call true.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Exists {
    pub call: TaggedModelCall,
    /// The Scan of the subquery's table.
    pub relation: LogicalNode,
    pub anti: bool,
}

/// One ON or WHERE part as written.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Part {
    #[serde(rename = "quail.logical.naive.Hint")]
    Hint(Hint),
    #[serde(rename = "quail.logical.naive.Exists")]
    Exists(Box<Exists>),
    #[serde(rename = "quail.common.expressions.ModelCall")]
    Call(ModelCall),
    #[serde(rename = "quail.common.expressions.RegularPredicate")]
    Compare(Box<RegularPredicate>),
    #[serde(rename = "quail.common.expressions.Not")]
    Not(Not),
    #[serde(rename = "quail.common.expressions.And")]
    And(Connective),
    #[serde(rename = "quail.common.expressions.Or")]
    Or(Connective),
}

/// The arguments of `place_relation`, with the relation tree split into
/// its scans, ON equalities, and parts.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlaceRelationArguments {
    /// The Scan of each table in FROM and JOIN order.
    pub scans: Vec<LogicalNode>,
    /// alias -> the ON equalities of the JOIN of that table, last JOIN first.
    pub equalities: Vec<(String, Vec<Equality>)>,
    pub parts: Vec<Part>,
    pub columns: Vec<ValueExpression>,
    /// provider -> its schema.
    pub schemas: BTreeMap<String, Option<Schema>>,
    pub unnested: Vec<String>,
    /// (model call, column name) of each column operator a test reads.
    pub tested: Vec<(TaggedModelCall, String)>,
    pub allow_relational: bool,
}

/// GROUP BY keys, the aggregates over each group, and the HAVING tests.
#[derive(Debug, Clone, Deserialize)]
pub struct Aggregation {
    pub keys: Vec<String>,
    pub aggregates: Vec<AggregateCall>,
    pub output: Vec<String>,
    pub having: Vec<HavingTest>,
}

/// The arguments of `LogicalPlanBuilder.project`, other than the row tests.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectArguments {
    pub columns: Vec<ValueExpression>,
    pub limit: Option<i64>,
    pub order: Vec<SortKey>,
    pub offset: i64,
    pub distinct: bool,
    pub aggregation: Option<Aggregation>,
    pub structs: Vec<ExtractStruct>,
    pub windows: Vec<WindowCall>,
    pub output: Vec<String>,
    pub unnests: Vec<Vec<String>>,
    pub unnest_outer: Vec<bool>,
}

// ---------------------------------------------------------------- placement

/// The parts of a query's conditions, by where they run.
#[derive(Debug, Default)]
pub struct PlacedConditions {
    pub scans: Vec<(String, Vec<Condition>)>,
    pub computed: ByAlias<(Condition, Option<f64>)>,
    pub rows: Vec<Condition>,
}

fn push<K: PartialEq, V>(map: &mut Vec<(K, Vec<V>)>, key: K, value: V) {
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
pub fn place_conditions(
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
pub fn result_row_condition(
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
pub fn result_columns_read(condition: &Condition) -> Vec<String> {
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

// ---------------------------------------------------------------- builder

/// A join predicate, before assembly.
#[derive(Debug, Clone)]
pub struct JoinSpec {
    /// The newly joined table aliases.
    pub aliases: Vec<String>,
    pub predicate: Asked,
    pub semantics: String,
    pub selectivity: Option<f64>,
    pub anchor: Option<String>,
    pub on: Vec<Equality>,
}

/// One table's operators, as `LogicalPlanBuilder.add_scan` takes them.
#[derive(Debug, Default)]
pub struct ScanOperators {
    pub predicates: Vec<FilterPredicate>,
    pub labels: Vec<(ModelCall, String)>,
    pub regular_predicates: Vec<Condition>,
    pub extracts: Vec<(ModelCall, String)>,
    pub computed_filters: Vec<(Condition, Option<f64>)>,
    pub sae_features: Vec<(ModelCall, String)>,
    pub maps: Vec<(ModelCall, String)>,
}

/// Builds the logical nodes of one SELECT.
#[derive(Debug, Default)]
pub struct LogicalPlanBuilder {
    nodes: Vec<(String, LogicalNode)>,
    root: Option<LogicalNode>,
}

fn tagged(call: &ModelCall) -> TaggedModelCall {
    TaggedModelCall::Call(call.clone())
}

fn column_node(kind: &str, input: LogicalNode, call: &ModelCall, name: &str) -> LogicalNode {
    let column = SemanticColumn {
        input: Box::new(input),
        call: tagged(call),
        name: name.to_string(),
    };
    match kind {
        "label" => LogicalNode::SemanticClassify(column),
        "extract" => LogicalNode::SemanticExtract(column),
        "features" => LogicalNode::SemanticSaeFeatures(column),
        _ => LogicalNode::SemanticMap(column),
    }
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

impl LogicalPlanBuilder {
    /// Add one table with its filters, classifications, and extractions.
    pub fn add_scan(
        &mut self,
        alias: &str,
        provider: &str,
        column: &ScanColumn,
        operators: ScanOperators,
        schema: Option<Schema>,
    ) -> Result<(), CompileError> {
        if self.nodes.iter().any(|(found, _)| found == alias) {
            return compile_error(format!("duplicate table alias {}", repr_str(alias)));
        }
        let mut node = LogicalNode::Scan(Scan {
            provider: provider.to_string(),
            alias: alias.to_string(),
            column: column.clone(),
            columns: Vec::new(),
            predicates: operators.regular_predicates,
            schema,
        });
        if !operators.predicates.is_empty() {
            node = LogicalNode::SemanticFilter(SemanticFilter {
                input: Box::new(node),
                predicates: operators.predicates,
                stop_key: Vec::new(),
            });
        }
        let calls: Vec<(String, ModelCall)> = operators
            .labels
            .iter()
            .map(|(call, name)| (name.clone(), call.clone()))
            .collect();
        let call_names: BTreeSet<&String> = calls.iter().map(|(name, _)| name).collect();
        let label_filters: Vec<(Condition, Option<f64>)> = operators
            .computed_filters
            .iter()
            .filter(|(condition, _)| {
                condition_columns(condition)
                    .iter()
                    .any(|name| call_names.contains(name))
            })
            .cloned()
            .collect();
        let mut classified: Vec<ModelCall> = Vec::new();
        for (condition, selectivity) in &label_filters {
            for name in condition_columns(condition) {
                if let Some((_, call)) = calls.iter().find(|(found, _)| *found == name)
                    && !classified.contains(call)
                {
                    node = column_node("label", node, call, &name);
                    classified.push(call.clone());
                }
            }
            check_labels(condition, &calls)?;
            node = LogicalNode::Filter(Filter {
                input: Box::new(node),
                condition: condition.clone(),
                selectivity: *selectivity,
            });
        }
        for (call, name) in &operators.labels {
            if !classified.contains(call) {
                node = column_node("label", node, call, name);
                classified.push(call.clone());
            }
        }
        let mut remaining: Vec<(Condition, Option<f64>)> = operators
            .computed_filters
            .iter()
            .filter(|test| !label_filters.contains(test))
            .cloned()
            .collect();
        let answers: BTreeSet<String> = [
            &operators.extracts,
            &operators.sae_features,
            &operators.maps,
        ]
        .iter()
        .flat_map(|calls| calls.iter())
        .flat_map(|(_, name)| [name.clone(), format!("{name}{SPAN_SUFFIX}")])
        .collect();
        let mut done: BTreeSet<String> = BTreeSet::new();
        for (kind, calls_of_kind) in [
            ("extract", &operators.extracts),
            ("features", &operators.sae_features),
            ("map", &operators.maps),
        ] {
            for (call, name) in calls_of_kind {
                node = column_node(kind, node, call, name);
                done.extend(crate::nodes::added_columns(&node));
                let mut index = 0;
                while index < remaining.len() {
                    let read: BTreeSet<String> = condition_columns(&remaining[index].0)
                        .into_iter()
                        .filter(|column| answers.contains(column))
                        .collect();
                    if read.is_subset(&done) {
                        let (predicate, selectivity) = remaining.remove(index);
                        node = LogicalNode::Filter(Filter {
                            input: Box::new(node),
                            condition: predicate,
                            selectivity,
                        });
                    } else {
                        index += 1;
                    }
                }
            }
        }
        if let Some((condition, _)) = remaining.first() {
            return compile_error(format!(
                "the filter {} tests a column the query does not return",
                condition_text(condition)
            ));
        }
        if self.root.is_none() {
            self.root = Some(node.clone());
        }
        self.nodes.push((alias.to_string(), node));
        Ok(())
    }

    fn table(&self, alias: &str) -> LogicalNode {
        self.nodes
            .iter()
            .find(|(found, _)| found == alias)
            .map(|(_, node)| node.clone())
            .expect("a table added before its join")
    }

    /// Classify the rows of the join the call reads.
    pub fn add_classify(&mut self, call: &ModelCall, name: &str) -> Result<(), CompileError> {
        let aliases: BTreeSet<String> = call_aliases(call).into_iter().collect();
        let Some(root) = self.root.take().filter(|_| aliases.len() == 2) else {
            return compile_error(
                "a classification of joined rows reads one document from each side of a join",
            );
        };
        fn insert(
            node: LogicalNode,
            aliases: &BTreeSet<String>,
            call: &ModelCall,
            name: &str,
        ) -> Result<LogicalNode, CompileError> {
            match node {
                LogicalNode::SemanticJoin(join)
                    if call_aliases(asked_call(&join.predicate))
                        .into_iter()
                        .collect::<BTreeSet<_>>()
                        == *aliases =>
                {
                    Ok(column_node(
                        "label",
                        LogicalNode::SemanticJoin(join),
                        call,
                        name,
                    ))
                }
                LogicalNode::SemanticJoin(mut join) => {
                    join.input = Box::new(insert(*join.input, aliases, call, name)?);
                    Ok(LogicalNode::SemanticJoin(join))
                }
                LogicalNode::SemanticClassify(mut column) => {
                    column.input = Box::new(insert(*column.input, aliases, call, name)?);
                    Ok(LogicalNode::SemanticClassify(column))
                }
                LogicalNode::Join(mut join) => {
                    join.left = Box::new(insert(*join.left, aliases, call, name)?);
                    Ok(LogicalNode::Join(join))
                }
                _ => {
                    let written = call_aliases(call);
                    compile_error(format!(
                        "the classification of {} x {} joined rows needs a join of the two",
                        repr_str(&written[0]),
                        repr_str(&written[1])
                    ))
                }
            }
        }
        self.root = Some(insert(root, &aliases, call, name)?);
        Ok(())
    }

    /// Join each new table onto the tree, then ask the prompt.
    pub fn add_join(&mut self, join: &JoinSpec) -> Result<(), CompileError> {
        let Some(mut root) = self.root.clone() else {
            return compile_error("a logical join needs an input table");
        };
        for alias in &join.aliases {
            root = LogicalNode::Join(Join {
                left: Box::new(root),
                right: Box::new(self.table(alias)),
                on: Vec::new(),
            });
        }
        let present: BTreeSet<String> = output_schema(&root)
            .into_iter()
            .map(|field| field.alias)
            .collect();
        for condition in &join.on {
            let sides = [
                crate::nodes::column_alias_of(&condition.left),
                crate::nodes::column_alias_of(&condition.right),
            ];
            if !sides.iter().all(|side| present.contains(side)) {
                return compile_error(format!(
                    "join condition {} names a table this join does not bring in ({}); put \
                     it on the JOIN that introduces the table",
                    equality_text(condition),
                    repr_list(&join.aliases)
                ));
            }
        }
        if !join.on.is_empty()
            && let LogicalNode::Join(found) = &mut root
        {
            found.on = join.on.clone();
        }
        self.root = Some(LogicalNode::SemanticJoin(SemanticJoin {
            input: Box::new(root),
            predicate: join.predicate.clone(),
            semantics: join.semantics.clone(),
            selectivity: join.selectivity,
            anchor: join.anchor.clone(),
        }));
        Ok(())
    }

    /// Add one relational cross join without an AI predicate.
    pub fn add_cross_join(&mut self, alias: &str) -> Result<(), CompileError> {
        let Some(root) = self.root.take() else {
            return compile_error("a logical join needs an input table");
        };
        self.root = Some(LogicalNode::Join(Join {
            left: Box::new(root),
            right: Box::new(self.table(alias)),
            on: Vec::new(),
        }));
        Ok(())
    }

    /// Build the plan's result nodes over the projected columns.
    pub fn project(
        &self,
        arguments: &ProjectArguments,
        row_tests: &[Condition],
    ) -> Result<LogicalPlan, CompileError> {
        let Some(root) = self.root.clone() else {
            return compile_error("a logical plan needs an input table");
        };
        let columns = &arguments.columns;
        let aggregation = arguments.aggregation.as_ref();
        if columns.is_empty() && aggregation.is_none() {
            return compile_error("Project needs at least one column");
        }
        if !arguments.structs.is_empty() && aggregation.is_some() {
            return compile_error(
                "GROUP BY does not read AI.EXTRACT struct columns; extract each field as its \
                 own column",
            );
        }
        if !arguments.structs.is_empty() && arguments.distinct {
            return compile_error(
                "DISTINCT does not compare AI.EXTRACT struct columns; extract each field as \
                 its own column",
            );
        }
        let unnested: BTreeSet<String> = arguments.unnests.iter().flatten().cloned().collect();
        check_feature_lists(
            columns,
            &arguments.order,
            arguments.distinct,
            aggregation,
            &unnested,
        )?;
        let reads_unnested = |expression: &ValueExpression| {
            column_references(expression)
                .iter()
                .any(|found| unnested.contains(&found.alias))
        };
        let mut expanded: Vec<(String, ValueExpression)> = Vec::new();
        for column in columns {
            if reads_unnested(column) {
                let name = column_name(column, true);
                match expanded.iter_mut().find(|(found, _)| *found == name) {
                    Some(entry) => entry.1 = column.clone(),
                    None => expanded.push((name, column.clone())),
                }
            }
        }
        let mut source: Vec<ValueExpression> = columns
            .iter()
            .filter(|column| {
                let name = column_name(column, true);
                !expanded.iter().any(|(found, _)| *found == name)
            })
            .cloned()
            .collect();
        let project = |source: &[ValueExpression], input: LogicalNode| {
            LogicalNode::Project(Project {
                input: Box::new(input),
                columns: source.iter().cloned().map(Projected::Expression).collect(),
                structs: arguments.structs.clone(),
            })
        };
        let written = project(&source, root.clone());
        let returned = result_columns(&written);
        let packing = crate::nodes::packing(&arguments.structs);
        let row_tests: Vec<Condition> = row_tests
            .iter()
            .map(|test| result_row_condition(test, &packing))
            .collect();
        let has_rows = !arguments.unnests.is_empty() || !row_tests.is_empty();
        if has_rows && aggregation.is_none() {
            for key in &arguments.order {
                if let SortExpression::Value(expression) = &key.expression
                    && !column_references(expression).is_empty()
                    && !reads_unnested(expression)
                    && !source.contains(expression)
                {
                    source.push(expression.clone());
                }
            }
        }
        source.extend(row_test_columns(&row_tests, &unnested, &source, &returned));
        let outer = if arguments.unnest_outer.is_empty() {
            vec![false; arguments.unnests.len()]
        } else {
            arguments.unnest_outer.clone()
        };
        let mut node = unnested_node(
            project(&source, root),
            &arguments.unnests,
            row_tests,
            &outer,
        )?;
        if has_rows {
            let selected: Vec<String> = if aggregation.is_some() || !arguments.windows.is_empty() {
                columns
                    .iter()
                    .map(|column| column_name(column, true))
                    .filter(|name| !unnested.contains(name))
                    .collect()
            } else if !arguments.unnests.is_empty() {
                arguments.output.clone()
            } else {
                returned
            };
            node = LogicalNode::Project(Project {
                input: Box::new(node),
                columns: selected
                    .into_iter()
                    .map(
                        |name| match expanded.iter().find(|(found, _)| *found == name) {
                            Some((_, column)) => Projected::Expression(column.clone()),
                            None => Projected::Name(name),
                        },
                    )
                    .collect(),
                structs: Vec::new(),
            });
        }
        if let Some(aggregation) = aggregation {
            node = LogicalNode::Aggregate(Aggregate {
                input: Box::new(node),
                keys: aggregation.keys.clone(),
                aggregates: aggregation.aggregates.clone(),
                output: aggregation.output.clone(),
                having: aggregation.having.clone(),
            });
        }
        if !arguments.windows.is_empty() {
            node = LogicalNode::Window(Window {
                input: Box::new(node),
                calls: arguments.windows.clone(),
                output: arguments.output.clone(),
            });
        }
        if arguments.limit.is_some()
            || !arguments.order.is_empty()
            || arguments.offset != 0
            || arguments.distinct
        {
            node = LogicalNode::Result(ResultNode {
                input: Box::new(node),
                limit: arguments.limit,
                order: arguments.order.clone(),
                offset: arguments.offset,
                distinct: arguments.distinct,
            });
        }
        let plan = LogicalPlan { root: node };
        validate_plan(&plan)?;
        Ok(plan)
    }
}

fn row_test_columns(
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

fn unnested_node(
    node: LogicalNode,
    unnests: &[Vec<String>],
    row_tests: Vec<Condition>,
    outer: &[bool],
) -> Result<LogicalNode, CompileError> {
    let mut remaining = row_tests;
    let place = |node: LogicalNode, remaining: &mut Vec<Condition>| {
        let available = readable_columns(&node);
        let mut node = node;
        let mut index = 0;
        while index < remaining.len() {
            if result_columns_read(&remaining[index])
                .iter()
                .all(|name| available.contains(name))
            {
                node = LogicalNode::Filter(Filter {
                    input: Box::new(node),
                    condition: remaining.remove(index),
                    selectivity: None,
                });
            } else {
                index += 1;
            }
        }
        node
    };
    let mut node = place(node, &mut remaining);
    if outer.len() != unnests.len() {
        panic!("zip() arguments have different lengths");
    }
    for (lists, keep_empty) in unnests.iter().zip(outer) {
        node = LogicalNode::Unnest(Unnest {
            input: Box::new(node),
            columns: lists.clone(),
            outer: *keep_empty,
        });
        validate_node(&node)?;
        node = place(node, &mut remaining);
    }
    if let Some(test) = remaining.first() {
        let available = readable_columns(&node);
        let mut missing: Vec<String> = result_columns_read(test)
            .into_iter()
            .filter(|name| !available.contains(name))
            .collect();
        missing.sort();
        missing.dedup();
        return compile_error(format!(
            "{} reads {}, which no result row holds",
            condition_text(test),
            repr_list(&missing)
        ));
    }
    Ok(node)
}

fn check_feature_lists(
    columns: &[ValueExpression],
    order: &[SortKey],
    distinct: bool,
    aggregation: Option<&Aggregation>,
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
    if aggregation.is_some() || distinct {
        return compile_error("GROUP BY and DISTINCT do not compare AI.SAE_FEATURES list columns");
    }
    for key in order {
        let refused = match &key.expression {
            SortExpression::Aggregate(aggregate) => names.contains(&aggregate.name),
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

/// The placed tables and joins, and what the result rows still test.
pub struct PlacedRelation {
    pub builder: LogicalPlanBuilder,
    pub rows: Vec<Condition>,
}

/// A part without its options: a model call, or a condition.
enum Body {
    Exists(Box<Exists>),
    Call(ModelCall),
    Condition(Condition),
}

fn body_of(part: Part) -> (Body, Option<f64>, Option<String>) {
    match part {
        Part::Hint(hint) => {
            let (body, _, _) = body_of(*hint.condition);
            (body, hint.selectivity, hint.anchor)
        }
        Part::Exists(exists) => (Body::Exists(exists), None, None),
        Part::Call(call) => (Body::Call(call), None, None),
        Part::Compare(test) => (Body::Condition(Condition::Compare(test)), None, None),
        Part::Not(not) => (Body::Condition(Condition::Not(not)), None, None),
        Part::And(connective) => (Body::Condition(Condition::And(connective)), None, None),
        Part::Or(connective) => (Body::Condition(Condition::Or(connective)), None, None),
    }
}

fn scan_of(node: &LogicalNode) -> &Scan {
    match node {
        LogicalNode::Scan(scan) => scan,
        _ => panic!("a relation part names a Scan"),
    }
}

fn asks_model(condition: &Condition) -> Option<&ModelCall> {
    match condition {
        Condition::Compare(test) => match &test.expression {
            ValueExpression::ModelCall(call) => Some(call),
            _ => None,
        },
        _ => None,
    }
}

fn gate(
    exists: &Exists,
    anchor: Option<&String>,
    selectivity: Option<f64>,
) -> Result<JoinSpec, CompileError> {
    let inner = scan_of(&exists.relation).alias.clone();
    let TaggedModelCall::Call(call) = &exists.call;
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
        aliases: vec![inner],
        predicate: Asked::Call(call.clone()),
        semantics: if exists.anti { "anti" } else { "exists" }.to_string(),
        selectivity,
        anchor: Some(outer),
        on: Vec::new(),
    })
}

fn repr_tuple(items: &[String]) -> String {
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
            crate::nodes::column_alias_of(&condition.left),
            crate::nodes::column_alias_of(&condition.right),
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
        aliases: news,
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
            "JOIN {alias} ON {first} has no AI predicate over its pairs; a plain join belongs \
             in the database the ids came from"
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
                .map(crate::expressions::argument_alias)
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
            "JOINed tables {} appear in no join predicate: every JOINed table must appear in \
             at least one AI predicate on a JOIN's ON or as a WHERE term",
            repr_list(&uncovered)
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

/// Place each part of a bound relation and build the logical plan below it.
pub fn place_relation(arguments: PlaceRelationArguments) -> Result<PlacedRelation, CompileError> {
    let mut scans: Vec<Scan> = arguments
        .scans
        .iter()
        .map(|node| scan_of(node).clone())
        .collect();
    let mut equalities = arguments.equalities.clone();
    let joined: Vec<String> = scans
        .iter()
        .skip(1)
        .map(|scan| scan.alias.clone())
        .collect();
    let base = scans[0].alias.clone();
    let mut filters: Vec<(String, Vec<FilterPredicate>)> = Vec::new();
    let mut joins: Vec<JoinSpec> = Vec::new();
    let mut label_tests: ByAlias<LabelTest> = Vec::new();
    let mut conditions: Vec<(Condition, Option<f64>)> = Vec::new();
    let mut claimed: Vec<String> = Vec::new();
    for part in arguments.parts {
        let (body, selectivity, anchor) = body_of(part);
        match body {
            Body::Exists(exists) => {
                scans.push(scan_of(&exists.relation).clone());
                joins.push(gate(&exists, anchor.as_ref(), selectivity)?);
            }
            Body::Condition(condition) if asks_model(&condition).is_none() => {
                conditions.push((condition, selectivity));
            }
            body => {
                let predicate = match body {
                    Body::Call(call) => Asked::Call(call),
                    Body::Condition(Condition::Compare(test)) => Asked::Compare(*test),
                    _ => unreachable!("a model call or a comparison of one"),
                };
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
                        &mut label_tests,
                        aliases[0].clone(),
                        (call.clone(), test.value.clone(), selectivity),
                    );
                    continue;
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
                        &mut filters,
                        aliases[0].clone(),
                        FilterPredicate {
                            expression: predicate,
                            selectivity,
                        },
                    );
                } else {
                    joins.push(join_spec(
                        predicate,
                        selectivity,
                        anchor,
                        &aliases,
                        &base,
                        &joined,
                        &mut claimed,
                        &mut equalities,
                    )?);
                }
            }
        }
    }
    let projected: Vec<(&ModelCall, String)> = arguments
        .columns
        .iter()
        .filter_map(|column| {
            let call = aliased_call(column, &[])?;
            Some((call, column_name(column, true)))
        })
        .collect();
    check_joins(&equalities, &base, &joined, &joins, &projected)?;
    let tested: Vec<(ModelCall, String)> = arguments
        .tested
        .iter()
        .map(|(TaggedModelCall::Call(call), name)| (call.clone(), name.clone()))
        .collect();
    check_calls(
        &filters,
        &joins,
        &label_tests,
        &projected,
        &tested,
        arguments.allow_relational,
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
    let placed = place_conditions(conditions, &computed, &arguments.unnested)?;
    let mut builder = LogicalPlanBuilder::default();
    for scan in &scans {
        let of = |kind: &str| -> Vec<(ModelCall, String)> {
            lookup(&computing[kind], &scan.alias).to_vec()
        };
        builder.add_scan(
            &scan.alias,
            &scan.provider,
            &scan.column,
            ScanOperators {
                predicates: lookup(&filters, &scan.alias).to_vec(),
                labels: lookup(&labels, &scan.alias).to_vec(),
                regular_predicates: lookup(&placed.scans, &scan.alias).to_vec(),
                extracts: of("extract"),
                computed_filters: lookup(&placed.computed, &scan.alias).to_vec(),
                sae_features: of("features"),
                maps: of("map"),
            },
            arguments.schemas.get(&scan.provider).cloned().flatten(),
        )?;
    }
    for join in &joins {
        builder.add_join(join)?;
    }
    for alias in &joined {
        if !claimed.contains(alias) {
            builder.add_cross_join(alias)?;
        }
    }
    for (call, name) in &named {
        if call_aliases(call).len() == 2 {
            builder.add_classify(call, name)?;
        }
    }
    Ok(PlacedRelation {
        builder,
        rows: placed.rows,
    })
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

/// Place a bound SELECT's relation and build its logical plan.
pub fn build_select(
    place: PlaceRelationArguments,
    project: &ProjectArguments,
) -> Result<LogicalPlan, CompileError> {
    let placed = place_relation(place)?;
    placed.builder.project(project, &placed.rows)
}
