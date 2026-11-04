//! What each logical node reads and returns, and its checks, as the
//! node classes of `quail/logical/nodes` define them.

use std::collections::{BTreeMap, BTreeSet};

use quail_physical::{
    ColumnRef, Condition, ExtractStruct, ModelCall, Projected, SortExpression, TaggedModelCall,
    ValueExpression,
};

use crate::expressions::{
    CompileError, PROBABILITIES_SUFFIX, SPAN_SUFFIX, aggregate_text, call_aliases, column_name,
    column_references, compile_error, condition_aliases, condition_references, condition_text,
    projected_call, projected_name, reference_name, repr_list, repr_str, repr_value, sort_name,
    tests, validate_aggregate, validate_call, validate_condition, validate_expression,
    validate_tree,
};
use crate::logical::{
    Aggregate, Join, LogicalNode, LogicalPlan, Project, ResultNode, Scan, ScanColumn,
    SemanticColumn, Unnest, Window,
};

/// The struct fields of one feature list, as `FEATURE_LIST` holds them.
pub const FEATURE_FIELDS: [&str; 2] = ["feature", "strength"];

/// The model call a column operator asks.
pub fn column_call(node: &SemanticColumn) -> &ModelCall {
    let TaggedModelCall::Call(call) = &node.call;
    call
}

/// The predicate of a semantic filter or join as a model call.
pub fn asked_call(asked: &quail_physical::Asked) -> &ModelCall {
    match asked {
        quail_physical::Asked::Call(call) => call,
        quail_physical::Asked::Compare(test) => match &test.expression {
            ValueExpression::ModelCall(call) => call,
            _ => panic!("a comparison that asks no model"),
        },
    }
}

/// The class name, call kind, and SQL function of a column operator.
pub fn column_kind(node: &LogicalNode) -> Option<(&'static str, &'static str, &'static str)> {
    match node {
        LogicalNode::SemanticClassify(_) => Some(("SemanticClassify", "label", "AI.CLASSIFY")),
        LogicalNode::SemanticExtract(_) => Some(("SemanticExtract", "extract", "AI.EXTRACT")),
        LogicalNode::SemanticMap(_) => Some(("SemanticMap", "map", "AI.MAP")),
        LogicalNode::SemanticSaeFeatures(_) => {
            Some(("SemanticSaeFeatures", "features", "AI.SAE_FEATURES"))
        }
        _ => None,
    }
}

/// The column operator a node is, if it is one.
pub fn as_column(node: &LogicalNode) -> Option<&SemanticColumn> {
    match node {
        LogicalNode::SemanticClassify(column)
        | LogicalNode::SemanticExtract(column)
        | LogicalNode::SemanticMap(column)
        | LogicalNode::SemanticSaeFeatures(column) => Some(column),
        _ => None,
    }
}

/// The columns a column operator adds, its answer first.
pub fn added_columns(node: &LogicalNode) -> Vec<String> {
    let column = as_column(node).expect("a column operator");
    match node {
        LogicalNode::SemanticClassify(_) if column_call(column).probabilities => vec![
            column.name.clone(),
            format!("{}{PROBABILITIES_SUFFIX}", column.name),
        ],
        LogicalNode::SemanticExtract(_) => {
            vec![column.name.clone(), format!("{}{SPAN_SUFFIX}", column.name)]
        }
        _ => vec![column.name.clone()],
    }
}

/// The node's inputs.
pub fn children(node: &LogicalNode) -> Vec<&LogicalNode> {
    match node {
        LogicalNode::Scan(_) => Vec::new(),
        LogicalNode::Join(join) => vec![&join.left, &join.right],
        LogicalNode::WithCte(with) => with
            .ctes
            .iter()
            .map(|(_, node)| node)
            .chain(std::iter::once(with.input.as_ref()))
            .collect(),
        other => vec![input(other)],
    }
}

/// The one input of a node that has one.
///
/// # Panics
///
/// The node is a Scan, a Join, or a WithCTE.
pub fn input(node: &LogicalNode) -> &LogicalNode {
    match node {
        LogicalNode::SemanticFilter(found) => &found.input,
        LogicalNode::Filter(found) => &found.input,
        LogicalNode::SemanticJoin(found) => &found.input,
        LogicalNode::Project(found) => &found.input,
        LogicalNode::Aggregate(found) => &found.input,
        LogicalNode::Window(found) => &found.input,
        LogicalNode::Result(found) => &found.input,
        LogicalNode::Unnest(found) => &found.input,
        LogicalNode::WithCte(found) => &found.input,
        LogicalNode::SemanticClassify(found)
        | LogicalNode::SemanticExtract(found)
        | LogicalNode::SemanticMap(found)
        | LogicalNode::SemanticSaeFeatures(found) => &found.input,
        LogicalNode::Scan(_) | LogicalNode::Join(_) => panic!("the node has no single input"),
    }
}

/// The node and every node below it, children first.
pub fn subtree(node: &LogicalNode) -> Vec<&LogicalNode> {
    let mut found = Vec::new();
    for child in children(node) {
        found.extend(subtree(child));
    }
    found.push(node);
    found
}

fn reference(alias: &str, provider: &str, column: &str) -> ColumnRef {
    ColumnRef {
        alias: alias.to_string(),
        provider: provider.to_string(),
        column: column.to_string(),
    }
}

/// The names of the source columns a document column reads.
pub fn source_columns(column: &ScanColumn) -> Vec<String> {
    match column {
        ScanColumn::Unnamed(name) => vec![name.clone()],
        ScanColumn::Expression(expression) => {
            let mut found: Vec<String> = Vec::new();
            for reference in column_references(expression) {
                if !found.contains(&reference.column) {
                    found.push(reference.column);
                }
            }
            found
        }
    }
}

fn has_column(scan: &Scan) -> bool {
    !matches!(&scan.column, ScanColumn::Unnamed(name) if name.is_empty())
}

/// The columns a node returns, as references.
pub fn output_schema(node: &LogicalNode) -> Vec<ColumnRef> {
    match node {
        LogicalNode::Scan(scan) => {
            let sources = if has_column(scan) {
                source_columns(&scan.column)
            } else {
                scan.schema
                    .as_ref()
                    .map(|schema| schema.names())
                    .unwrap_or_default()
            };
            let mut fields: Vec<ColumnRef> = sources
                .iter()
                .map(|source| reference(&scan.alias, &scan.provider, source))
                .collect();
            fields.extend(
                scan.columns
                    .iter()
                    .filter(|column| !sources.contains(column))
                    .map(|column| reference(&scan.alias, &scan.provider, column)),
            );
            fields
        }
        LogicalNode::Join(join) => {
            let mut fields = output_schema(&join.left);
            for field in output_schema(&join.right) {
                if !fields.contains(&field) {
                    fields.push(field);
                }
            }
            fields
        }
        LogicalNode::Project(project) => {
            let mut fields: Vec<(String, ColumnRef)> = Vec::new();
            let mut put = |name: String, value: ColumnRef| {
                if let Some(entry) = fields.iter_mut().find(|(key, _)| *key == name) {
                    entry.1 = value;
                } else {
                    fields.push((name, value));
                }
            };
            for field in output_schema(&project.input) {
                put(reference_name(&field, true), field);
            }
            for column in &project.columns {
                if let Projected::Expression(ValueExpression::ColumnRef(found)) = column {
                    put(reference_name(found, true), found.clone());
                }
            }
            result_columns(node)
                .iter()
                .map(|name| {
                    fields
                        .iter()
                        .find(|(key, _)| key == name)
                        .map(|(_, field)| field.clone())
                        .unwrap_or_else(|| reference("", "", name))
                })
                .collect()
        }
        LogicalNode::Aggregate(Aggregate { input, output, .. })
        | LogicalNode::Window(Window { input, output, .. }) => named_schema(input, output),
        LogicalNode::Unnest(unnest) => {
            let types = list_types(&unnest.input);
            let mut fields = Vec::new();
            for column in output_schema(&unnest.input) {
                let name = reference_name(&column, true);
                if unnest.columns.contains(&name) {
                    fields.extend(types[&name].iter().map(|field| reference(&name, "", field)));
                } else {
                    fields.push(column);
                }
            }
            fields
        }
        column if as_column(column).is_some() => {
            let found = as_column(column).expect("a column operator");
            let mut fields = output_schema(&found.input);
            let alias = column_alias(found);
            let provider = fields
                .iter()
                .find(|field| field.alias == alias)
                .map(|field| field.provider.clone())
                .unwrap_or_default();
            for name in added_columns(column) {
                fields.push(reference(&alias, &provider, &name));
            }
            fields
        }
        other => output_schema(input(other)),
    }
}

fn named_schema(input: &LogicalNode, output: &[String]) -> Vec<ColumnRef> {
    let fields = output_schema(input);
    output
        .iter()
        .map(|name| {
            fields
                .iter()
                .rev()
                .find(|field| reference_name(field, true) == *name)
                .cloned()
                .unwrap_or_else(|| reference("", "", name))
        })
        .collect()
}

/// The table alias a column operator's answer columns belong to.
pub fn column_alias(node: &SemanticColumn) -> String {
    call_aliases(column_call(node))
        .into_iter()
        .next()
        .unwrap_or_default()
}

/// Every struct column of a Project and its member columns, answers and spans.
pub fn packing(structs: &[ExtractStruct]) -> Vec<(String, Vec<(String, String)>)> {
    let mut found = Vec::new();
    for found_struct in structs {
        for suffix in ["", SPAN_SUFFIX] {
            found.push((
                format!("{}{suffix}", found_struct.name),
                found_struct
                    .fields
                    .iter()
                    .map(|field| {
                        (
                            field.clone(),
                            format!("{}.{field}{suffix}", found_struct.name),
                        )
                    })
                    .collect(),
            ));
        }
    }
    found
}

/// Result column names after packing members into struct columns.
pub fn packed_columns(
    columns: impl IntoIterator<Item = String>,
    packing: &[(String, Vec<(String, String)>)],
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for column in columns {
        let name = packing
            .iter()
            .find(|(_, fields)| fields.iter().any(|(_, member)| *member == column))
            .map(|(name, _)| name.clone())
            .unwrap_or(column);
        if !out.contains(&name) {
            out.push(name);
        }
    }
    out
}

/// The names of the result columns of a node above the Project, or of it.
///
/// # Panics
///
/// The node reads documents rather than result rows.
pub fn result_columns(node: &LogicalNode) -> Vec<String> {
    match node {
        LogicalNode::Project(project) => packed_columns(
            project.columns.iter().map(projected_name),
            &packing(&project.structs),
        ),
        LogicalNode::Filter(filter) => result_columns(&filter.input),
        LogicalNode::Result(result) => result_columns(&result.input),
        LogicalNode::Aggregate(aggregate) => aggregate.output.clone(),
        LogicalNode::Window(window) => window.output.clone(),
        LogicalNode::Unnest(unnest) => {
            let fields = unnest_fields(unnest);
            result_columns(&unnest.input)
                .into_iter()
                .flat_map(|column| match fields.get(&column) {
                    Some(names) => names.clone(),
                    None => vec![column],
                })
                .collect()
        }
        _ => panic!("the node returns documents, not result rows"),
    }
}

/// The field columns each list column of an Unnest becomes.
pub fn unnest_fields(unnest: &Unnest) -> BTreeMap<String, Vec<String>> {
    let types = list_types(&unnest.input);
    unnest
        .columns
        .iter()
        .map(|column| {
            let names = types
                .get(column)
                .map(|fields| {
                    fields
                        .iter()
                        .map(|field| format!("{column}.{field}"))
                        .collect()
                })
                .unwrap_or_default();
            (column.clone(), names)
        })
        .collect()
}

/// Each list column a node above the Project returns, with its struct fields.
pub fn list_types(node: &LogicalNode) -> BTreeMap<String, Vec<String>> {
    match node {
        LogicalNode::Project(project) => project
            .columns
            .iter()
            .filter_map(|column| {
                projected_call(column, &["features"])?;
                Some((
                    projected_name(column),
                    FEATURE_FIELDS
                        .iter()
                        .map(|field| field.to_string())
                        .collect(),
                ))
            })
            .collect(),
        LogicalNode::Filter(filter) => list_types(&filter.input),
        LogicalNode::Unnest(unnest) => list_types(&unnest.input)
            .into_iter()
            .filter(|(name, _)| !unnest.columns.contains(name))
            .collect(),
        _ => panic!("the node returns no list columns"),
    }
}

/// Whether the node reads result rows: an Unnest, or a Filter above an
/// Unnest or the Project.
pub fn above_projection(node: &LogicalNode) -> bool {
    let mut node = node;
    while let LogicalNode::Filter(filter) = node {
        node = &filter.input;
        if matches!(node, LogicalNode::Project(_)) {
            return true;
        }
    }
    matches!(node, LogicalNode::Unnest(_))
}

/// Whether the node is the Project or an Unnest chain above it.
pub fn reads_projection(node: &LogicalNode) -> bool {
    matches!(node, LogicalNode::Project(_)) || above_projection(node)
}

/// The Project at or below a node above it.
pub fn projection_below(node: &LogicalNode) -> &Project {
    let mut node = node;
    loop {
        match node {
            LogicalNode::Project(project) => return project,
            other => node = input(other),
        }
    }
}

/// The names a condition on the node's result rows may read.
pub fn readable_columns(node: &LogicalNode) -> BTreeSet<String> {
    let mut names: BTreeSet<String> = result_columns(node).into_iter().collect();
    for (name, fields) in packing(&projection_below(node).structs) {
        if names.contains(&name) {
            for (field, _) in fields {
                names.insert(format!("{name}.{field}"));
            }
        }
    }
    names
}

/// (alias, column) pairs the one-table column operators below compute.
pub fn computed_columns(node: &LogicalNode) -> BTreeSet<(String, String)> {
    let mut found = BTreeSet::new();
    for below in subtree(node) {
        if let Some(column) = as_column(below)
            && call_aliases(column_call(column)).len() == 1
        {
            for name in added_columns(below) {
                found.insert((column_alias(column), name));
            }
        }
    }
    found
}

/// The source columns of the table an alias names, below a node.
pub fn scan_columns(node: &LogicalNode, alias: &str) -> BTreeSet<String> {
    let mut node = node;
    loop {
        if let LogicalNode::Scan(scan) = node {
            if scan.alias != alias {
                return BTreeSet::new();
            }
            return scan
                .schema
                .as_ref()
                .map(|schema| schema.names().into_iter().collect())
                .unwrap_or_default();
        }
        let next = children(node).into_iter().find(|child| {
            output_schema(child)
                .iter()
                .any(|field| field.alias == alias)
        });
        match next {
            Some(child) => node = child,
            None => return BTreeSet::new(),
        }
    }
}

fn aliases_of(node: &LogicalNode) -> BTreeSet<String> {
    output_schema(node)
        .into_iter()
        .map(|field| field.alias)
        .collect()
}

fn sorted_list(names: impl IntoIterator<Item = String>) -> String {
    let mut names: Vec<String> = names.into_iter().collect();
    names.sort();
    repr_list(&names)
}

/// Check that an expression asks the model yes or no for every row.
pub fn validate_predicate(asked: &quail_physical::Asked) -> Result<(), CompileError> {
    match asked {
        quail_physical::Asked::Compare(test) => crate::expressions::validate_test(test),
        quail_physical::Asked::Call(call) => {
            validate_call(call)?;
            match call.kind.as_str() {
                "score" => compile_error(
                    "an AI.SCORE call is a predicate only when compared with a threshold",
                ),
                "label" => compile_error(
                    "an AI.CLASSIFY label is tested by a Filter on its column, not by a \
                     SemanticFilter",
                ),
                "extract" => {
                    compile_error("an AI.EXTRACT answer is a result column, not a predicate")
                }
                "features" => {
                    compile_error("an AI.SAE_FEATURES answer is a result column, not a predicate")
                }
                "map" => compile_error("an AI.MAP answer is a result column, not a predicate"),
                _ => Ok(()),
            }
        }
    }
}

fn validate_scan(scan: &Scan) -> Result<(), CompileError> {
    if scan.provider.is_empty()
        || scan.alias.is_empty()
        || (!has_column(scan) && scan.schema.is_none())
    {
        return compile_error("Scan needs a provider, alias, and column");
    }
    if let ScanColumn::Expression(expression) = &scan.column {
        validate_expression(expression)?;
        if column_references(expression)
            .iter()
            .any(|found| found.alias != scan.alias)
        {
            return compile_error(format!(
                "Scan {} reads the document column {} of another table",
                repr_str(&scan.alias),
                column_name(expression, true)
            ));
        }
    }
    let mut seen = BTreeSet::new();
    if !scan.columns.iter().all(|column| seen.insert(column)) {
        let listed: Vec<String> = scan.columns.iter().map(|column| repr_str(column)).collect();
        let tuple = if listed.len() == 1 {
            format!("({},)", listed[0])
        } else {
            format!("({})", listed.join(", "))
        };
        return compile_error(format!(
            "Scan {} lists a column twice: {tuple}",
            repr_str(&scan.alias)
        ));
    }
    for predicate in &scan.predicates {
        validate_condition(predicate)?;
        if condition_aliases(predicate) != [scan.alias.clone()] {
            return compile_error(format!(
                "predicate {} is on the Scan of {}",
                condition_text(predicate),
                repr_str(&scan.alias)
            ));
        }
    }
    Ok(())
}

fn validate_filter(input: &LogicalNode, condition: &Condition) -> Result<(), CompileError> {
    validate_condition(condition)?;
    let references = condition_references(condition);
    if reads_projection(input) {
        let returned = readable_columns(input);
        for found in &references {
            let name = reference_name(found, true);
            if !returned.contains(&name) {
                return compile_error(format!(
                    "a Filter on result rows tests a column the rows below return, got {name}"
                ));
            }
        }
        return Ok(());
    }
    let computed = computed_columns(input);
    for found in &references {
        if !computed.contains(&(found.alias.clone(), found.column.clone()))
            && !scan_columns(input, &found.alias).contains(&found.column)
        {
            return compile_error(format!(
                "a Filter tests a column a one-table column operator below it computes; \
                 nothing below computes {}.{}",
                found.alias, found.column
            ));
        }
    }
    Ok(())
}

fn validate_join(join: &Join) -> Result<(), CompileError> {
    let left = aliases_of(&join.left);
    let right = aliases_of(&join.right);
    for condition in &join.on {
        let sides: BTreeSet<String> = [
            column_alias_of(&condition.left),
            column_alias_of(&condition.right),
        ]
        .into_iter()
        .collect();
        if sides.is_disjoint(&left) || sides.is_disjoint(&right) || sides.len() != 2 {
            return compile_error(format!(
                "join condition {} must name one table on each side of the join ({} and {})",
                equality_text(condition),
                sorted_list(left),
                sorted_list(right)
            ));
        }
    }
    Ok(())
}

/// The one table alias a value expression of an equality reads.
pub fn column_alias_of(expression: &ValueExpression) -> String {
    column_references(expression)
        .into_iter()
        .next()
        .map(|found| found.alias)
        .unwrap_or_default()
}

/// The text of an equality, as `str(Equality)`.
pub fn equality_text(equality: &quail_physical::Equality) -> String {
    format!(
        "{} = {}",
        column_name(&equality.left, true),
        column_name(&equality.right, true)
    )
}

fn validate_column(node: &LogicalNode) -> Result<(), CompileError> {
    let column = as_column(node).expect("a column operator");
    let (class, kind, function) = column_kind(node).expect("a column operator");
    let call = column_call(column);
    validate_call(call)?;
    if call.kind != kind {
        return compile_error(format!("{class} needs an {function} call"));
    }
    if let LogicalNode::SemanticExtract(_) = node {
        let parts: Vec<&str> = column.name.split('.').collect();
        if parts.len() != 2 || parts.iter().any(|part| part.is_empty()) {
            return compile_error(format!(
                "an extraction is named after its struct column and field, as \
                 'struct.field', got {}",
                repr_str(&column.name)
            ));
        }
    } else if column.name.is_empty() || column.name.contains('.') {
        return compile_error(format!(
            "{function} needs a column name without a dot, got {}",
            repr_str(&column.name)
        ));
    }
    let aliases = call_aliases(call);
    let allowed: &[usize] = if kind == "label" { &[1, 2] } else { &[1] };
    if !allowed.contains(&aliases.len()) {
        let tail = if allowed.contains(&2) {
            ", or one from each side of a join"
        } else {
            " column"
        };
        return compile_error(format!("{function} reads one document{tail}"));
    }
    let schema = output_schema(&column.input);
    let present: BTreeSet<String> = schema.iter().map(|field| field.alias.clone()).collect();
    let missing: Vec<String> = aliases
        .iter()
        .filter(|alias| !present.contains(*alias))
        .cloned()
        .collect();
    if !missing.is_empty() {
        return compile_error(format!(
            "{function} reads {}, which its input does not produce ({})",
            sorted_list(missing),
            sorted_list(present)
        ));
    }
    let alias = column_alias(column);
    let taken: BTreeSet<&String> = schema
        .iter()
        .filter(|field| field.alias == alias)
        .map(|field| &field.column)
        .collect();
    for name in added_columns(node) {
        if taken.contains(&name) {
            return compile_error(format!("the name {} is already used", repr_str(&name)));
        }
    }
    if let LogicalNode::SemanticClassify(_) = node
        && aliases.len() == 2
    {
        let wanted: BTreeSet<&String> = aliases.iter().collect();
        let joined = subtree(&column.input).into_iter().any(|below| {
            matches!(below, LogicalNode::SemanticJoin(join)
                if call_aliases(asked_call(&join.predicate)).iter().collect::<BTreeSet<_>>() == wanted)
        });
        if !joined {
            return compile_error(format!(
                "the classification of {} x {} joined rows needs a join of the two",
                repr_str(&aliases[0]),
                repr_str(&aliases[1])
            ));
        }
    }
    Ok(())
}

fn validate_project(node: &LogicalNode, project: &Project) -> Result<(), CompileError> {
    let computed: Vec<(&ModelCall, &String)> = subtree(&project.input)
        .into_iter()
        .filter_map(as_column)
        .map(|column| (column_call(column), &column.name))
        .collect();
    let _ = node;
    for column in &project.columns {
        let expression = match column {
            Projected::Name(name) => {
                if !result_columns(&project.input).contains(name) {
                    return compile_error(format!(
                        "Project reads an unknown column {}",
                        repr_str(name)
                    ));
                }
                continue;
            }
            Projected::Expression(expression) => expression,
        };
        validate_expression(expression)?;
        if let (Some(call), ValueExpression::Alias(alias)) =
            (projected_call(column, &[]), expression)
            && call.kind != "score"
            && !computed
                .iter()
                .any(|(found, name)| *found == call && **name == alias.name)
        {
            return compile_error(format!(
                "the {} column {} needs the column operator that computes it below the \
                 Project",
                call.kind,
                repr_str(&alias.name)
            ));
        }
    }
    let names: Vec<String> = project.columns.iter().map(projected_name).collect();
    if names.iter().collect::<BTreeSet<_>>().len() != names.len() {
        return compile_error(format!(
            "projection names must be unique, got {}",
            repr_list(&names)
        ));
    }
    validate_structs(project)
}

fn validate_structs(project: &Project) -> Result<(), CompileError> {
    let extracted: Vec<(String, &ModelCall)> = project
        .columns
        .iter()
        .filter_map(|column| {
            let call = projected_call(column, &["extract"])?;
            Some((projected_name(column), call))
        })
        .collect();
    let mut members: Vec<String> = Vec::new();
    for found in &project.structs {
        validate_extract_struct(found)?;
        let mut aliases: BTreeSet<Vec<String>> = BTreeSet::new();
        for field in &found.fields {
            let member = format!("{}.{field}", found.name);
            let Some((_, call)) = extracted.iter().find(|(name, _)| *name == member) else {
                return compile_error(format!(
                    "the struct column {} needs the extraction {} in the projection",
                    repr_str(&found.name),
                    repr_str(&member)
                ));
            };
            aliases.insert(call_aliases(call));
            members.push(member);
        }
        if aliases.len() != 1 {
            return compile_error(format!(
                "the fields of {} read one document column",
                repr_str(&found.name)
            ));
        }
    }
    for (name, _) in &extracted {
        if !members.contains(name) {
            return compile_error(format!(
                "the extraction {} is a field of no struct column",
                repr_str(name)
            ));
        }
    }
    let mut all: Vec<String> = project
        .columns
        .iter()
        .map(projected_name)
        .filter(|name| !members.contains(name))
        .collect();
    all.extend(packing(&project.structs).into_iter().map(|(name, _)| name));
    if all.iter().collect::<BTreeSet<_>>().len() != all.len() {
        return compile_error(format!(
            "projection names must be unique, got {}",
            repr_list(&all)
        ));
    }
    Ok(())
}

fn validate_extract_struct(found: &ExtractStruct) -> Result<(), CompileError> {
    if found.name.is_empty() || found.name.contains('.') {
        return compile_error(format!(
            "an extraction needs a column name without a dot, got {}",
            repr_str(&found.name)
        ));
    }
    if found.fields.is_empty() {
        return compile_error(format!(
            "AI.EXTRACT {} needs at least one field",
            repr_str(&found.name)
        ));
    }
    for field in &found.fields {
        if field.is_empty() || field.contains('.') {
            return compile_error(format!(
                "an AI.EXTRACT field needs a name without a dot, got {}",
                repr_str(field)
            ));
        }
    }
    if found.fields.iter().collect::<BTreeSet<_>>().len() != found.fields.len() {
        return compile_error(format!(
            "AI.EXTRACT {} names a field twice: {}",
            repr_str(&found.name),
            repr_list(&found.fields)
        ));
    }
    Ok(())
}

fn validate_aggregate_node(aggregate: &Aggregate) -> Result<(), CompileError> {
    if !reads_projection(&aggregate.input) {
        return compile_error("Aggregate needs projected input columns");
    }
    let projected = result_columns(&aggregate.input);
    for key in &aggregate.keys {
        if !projected.contains(key) {
            return compile_error(format!(
                "GROUP BY {} is not a projected column",
                repr_str(key)
            ));
        }
    }
    let mut names = aggregate.keys.clone();
    for call in &aggregate.aggregates {
        validate_aggregate(call)?;
        if let Some(argument) = &call.argument
            && !projected.contains(argument)
        {
            return compile_error(format!(
                "{} reads a column the projection does not have",
                aggregate_text(call)
            ));
        }
        names.push(call.name.clone());
    }
    if names.iter().collect::<BTreeSet<_>>().len() != names.len() {
        return compile_error(format!(
            "GROUP BY keys and aggregate names must be unique, got {}",
            repr_list(&names)
        ));
    }
    if aggregate.output.is_empty() {
        return compile_error("an aggregation returns at least one column");
    }
    for name in &aggregate.output {
        if !names.contains(name) {
            return compile_error(format!(
                "{} is neither a GROUP BY key nor an aggregate",
                repr_str(name)
            ));
        }
    }
    if aggregate.output.iter().collect::<BTreeSet<_>>().len() != aggregate.output.len() {
        let listed: Vec<String> = aggregate.output.iter().map(|name| repr_str(name)).collect();
        return compile_error(format!(
            "output names must be unique, got ({})",
            listed.join(", ")
        ));
    }
    const HAVING: [&str; 6] = ["=", "<>", "<", "<=", ">", ">="];
    for test in &aggregate.having {
        validate_aggregate(&test.aggregate)?;
        if !HAVING.contains(&test.comparison.as_str()) {
            return compile_error(format!(
                "HAVING compares with one of ('=', '<>', '<', '<=', '>', '>='), got {}",
                repr_str(&test.comparison)
            ));
        }
        if !matches!(test.value, serde_json::Value::Number(_)) {
            return compile_error("HAVING compares an aggregate with a number");
        }
        if !aggregate.aggregates.contains(&test.aggregate) {
            return compile_error(format!(
                "HAVING {} {} {} tests an aggregate the aggregation does not compute",
                test.aggregate.name,
                test.comparison,
                repr_value(&test.value)
            ));
        }
    }
    Ok(())
}

fn validate_window(window: &Window) -> Result<(), CompileError> {
    if !(reads_projection(&window.input) || matches!(*window.input, LogicalNode::Aggregate(_))) {
        return compile_error("Window needs projected or aggregated input columns");
    }
    let projected = result_columns(&window.input);
    let mut names = projected.clone();
    const FUNCTIONS: [&str; 5] = ["row_number", "lag", "lead", "sum", "count"];
    for call in &window.calls {
        if !FUNCTIONS.contains(&call.function.as_str()) {
            return compile_error(format!(
                "unsupported window function {}; supported: {}",
                repr_str(&call.function),
                FUNCTIONS.join(", ")
            ));
        }
        if call.function == "row_number" && call.argument.is_some() {
            return compile_error("row_number() takes no column");
        }
        if ["lag", "lead", "sum"].contains(&call.function.as_str()) && call.argument.is_none() {
            return compile_error(format!("{}() needs a column", call.function));
        }
        if ["lag", "lead"].contains(&call.function.as_str()) && call.order.is_empty() {
            return compile_error(format!(
                "{}() needs an ORDER BY in its window",
                call.function
            ));
        }
        let read = call
            .argument
            .iter()
            .chain(call.partition.iter())
            .chain(call.order.iter().map(|(name, _, _)| name));
        for column in read {
            if !projected.contains(column) {
                return compile_error(format!(
                    "the window {} reads {}, which is not a projected column",
                    repr_str(&call.name),
                    repr_str(column)
                ));
            }
        }
        names.push(call.name.clone());
    }
    if names.iter().collect::<BTreeSet<_>>().len() != names.len() {
        return compile_error(format!(
            "window names must be unique, got {}",
            repr_list(&names)
        ));
    }
    for name in &window.output {
        if !names.contains(name) {
            return compile_error(format!(
                "the window output {} is neither a column nor a window",
                repr_str(name)
            ));
        }
    }
    Ok(())
}

fn validate_result(result: &ResultNode) -> Result<(), CompileError> {
    let input = result.input.as_ref();
    if !(reads_projection(input)
        || matches!(input, LogicalNode::Aggregate(_) | LogicalNode::Window(_)))
    {
        return compile_error("Result needs a projection, aggregation, or window");
    }
    if result.limit.is_some_and(|limit| limit <= 0) {
        return compile_error("LIMIT must be a positive integer");
    }
    if result.offset < 0 {
        return compile_error("OFFSET must be a nonnegative integer");
    }
    let projected: BTreeSet<String> = result_columns(input).into_iter().collect();
    let projection = projection_below(input);
    let grouped = matches!(input, LogicalNode::Aggregate(_))
        || matches!(input, LogicalNode::Window(window)
            if matches!(*window.input, LogicalNode::Aggregate(_)));
    for key in &result.order {
        match &key.expression {
            SortExpression::Aggregate(aggregate) => validate_aggregate(aggregate)?,
            SortExpression::Value(value) => validate_tree(value)?,
        }
        let name = sort_name(&key.expression);
        if let SortExpression::Value(value @ ValueExpression::Alias(_)) = &key.expression
            && !projection
                .columns
                .contains(&Projected::Expression(value.clone()))
            && !projected.contains(&name)
        {
            return compile_error(format!(
                "ORDER BY {} names an expression the projection does not return",
                repr_str(&name)
            ));
        }
        if grouped && !projected.contains(&name) {
            return compile_error(format!(
                "ORDER BY {} with GROUP BY names a key or an aggregate in the SELECT list",
                repr_str(&name)
            ));
        }
        if result.distinct && !projected.contains(&name) {
            return compile_error(format!(
                "ORDER BY {} with DISTINCT needs the column in the SELECT list",
                repr_str(&name)
            ));
        }
    }
    Ok(())
}

fn validate_unnest(unnest: &Unnest) -> Result<(), CompileError> {
    if unnest.columns.is_empty() {
        return compile_error("UNNEST needs at least one list column");
    }
    if unnest.columns.iter().collect::<BTreeSet<_>>().len() != unnest.columns.len() {
        let listed: Vec<String> = unnest.columns.iter().map(|name| repr_str(name)).collect();
        let tuple = if listed.len() == 1 {
            format!("({},)", listed[0])
        } else {
            format!("({})", listed.join(", "))
        };
        return compile_error(format!("UNNEST names a list twice, got {tuple}"));
    }
    let types = list_types(&unnest.input);
    for column in &unnest.columns {
        if !types.contains_key(column) {
            return compile_error(format!(
                "UNNEST reads a list column the rows below return, got {}; the list columns \
                 are {}",
                repr_str(column),
                sorted_list(types.keys().cloned())
            ));
        }
    }
    let names = result_columns(&LogicalNode::Unnest(unnest.clone()));
    if names.iter().collect::<BTreeSet<_>>().len() != names.len() {
        let listed: Vec<String> = names.iter().map(|name| repr_str(name)).collect();
        return compile_error(format!(
            "UNNEST field columns must not repeat a column name, got ({})",
            listed.join(", ")
        ));
    }
    Ok(())
}

/// Check one node, as its Python class's `validate`.
pub fn validate_node(node: &LogicalNode) -> Result<(), CompileError> {
    match node {
        LogicalNode::Scan(scan) => validate_scan(scan),
        LogicalNode::SemanticFilter(filter) => {
            if filter.predicates.is_empty() {
                return compile_error("SemanticFilter needs at least one predicate");
            }
            for predicate in &filter.predicates {
                validate_predicate(&predicate.expression)?;
                if call_aliases(asked_call(&predicate.expression)).len() != 1 {
                    return compile_error("a SemanticFilter predicate reads one table");
                }
            }
            Ok(())
        }
        LogicalNode::Filter(filter) => validate_filter(&filter.input, &filter.condition),
        LogicalNode::Join(join) => validate_join(join),
        LogicalNode::SemanticJoin(join) => {
            if !["full", "exists", "anti"].contains(&join.semantics.as_str()) {
                return compile_error(format!(
                    "unknown join semantics {}",
                    repr_str(&join.semantics)
                ));
            }
            validate_predicate(&join.predicate)?;
            let present = aliases_of(&join.input);
            let missing: Vec<String> = call_aliases(asked_call(&join.predicate))
                .into_iter()
                .filter(|alias| !present.contains(alias))
                .collect();
            if !missing.is_empty() {
                return compile_error(format!(
                    "the join predicate reads {}, which its input does not produce ({})",
                    sorted_list(missing),
                    sorted_list(present)
                ));
            }
            Ok(())
        }
        LogicalNode::Project(project) => validate_project(node, project),
        LogicalNode::Aggregate(aggregate) => validate_aggregate_node(aggregate),
        LogicalNode::Window(window) => validate_window(window),
        LogicalNode::Result(result) => validate_result(result),
        LogicalNode::Unnest(unnest) => validate_unnest(unnest),
        LogicalNode::WithCte(with) => {
            let names: BTreeSet<&String> = with.ctes.iter().map(|(name, _)| name).collect();
            if names.len() != with.ctes.len() {
                return compile_error("CTE names must be unique within a query");
            }
            Ok(())
        }
        column => validate_column(column),
    }
}

/// Every node of a plan, each child before its parent.
pub fn walk(plan: &LogicalPlan) -> Vec<&LogicalNode> {
    subtree(&plan.root)
}

/// Check every node of a plan.
pub fn validate_plan(plan: &LogicalPlan) -> Result<(), CompileError> {
    for node in walk(plan) {
        validate_node(node)?;
    }
    Ok(())
}

/// The conditions's tests that read a model call.
pub fn tested_calls(condition: &Condition) -> Vec<&ModelCall> {
    tests(condition)
        .into_iter()
        .filter_map(|test| match &test.expression {
            ValueExpression::ModelCall(call) => Some(call),
            _ => None,
        })
        .collect()
}
