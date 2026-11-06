//! What a region's run returns besides its rows: each model operator's
//! answers, the documents still in play, the measured selectivity of
//! each predicate, and the run's report.
//!
//! [`collect_answers`] reads the executor's output relations by port;
//! [`execution_report`] reads the backend's measurements.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, Int32Array, RecordBatch};
use arrow::compute::{cast, filter_record_batch, is_not_null, take};
use arrow::datatypes::{DataType, Int32Type};
use quail_planner::LogicalPlan;
use quail_planner::expressions::call_aliases;
use quail_planner::nodes::asked_call;
use quail_planner::operators::operators;
use serde::Deserialize;
use serde_json::{Map, Value, json};

/// The answer columns a port exposes for reporting.
#[derive(Debug, Clone, Deserialize)]
pub struct AnswerPort {
    /// "classifies", "scores", "extracts", "features", or "maps".
    pub kind: String,
    pub name: String,
    pub columns: Vec<String>,
}

/// One output port of the executed graph.
#[derive(Debug, Clone, Deserialize)]
pub struct OutputPort {
    /// The node's registered type name, such as `quail.ai_join`.
    pub node_type: String,
    pub node: String,
    pub port: String,
    /// The port's value type, such as "rows" or "filter_answers".
    pub value_type: String,
    pub answer: Option<AnswerPort>,
}

/// What [`collect_answers`] reads besides the output relations.
#[derive(Debug, Clone, Deserialize)]
pub struct AnswerRequest {
    /// The region's logical plan, as JSON.
    pub logical: String,
    /// Every output port, in the graph's topological order.
    pub ports: Vec<OutputPort>,
    /// alias -> the number of documents its scan read.
    pub num_documents: BTreeMap<String, i64>,
    /// alias -> each document's row in the registered table, for a scan
    /// that reads some of a table's rows.
    #[serde(default)]
    pub source_rows: BTreeMap<String, Vec<i32>>,
}

/// Which operator an answer table belongs to.
#[derive(Debug, Clone, PartialEq)]
pub enum AnswerKey {
    /// An answer column, by its output name.
    Name(String),
    /// A filter predicate: its alias and written position.
    Filter(String, i64),
    /// A join predicate, by written position.
    Join(i64),
}

/// The answers, survivors, and stages of one run.
#[derive(Debug)]
pub struct Answers {
    /// (kind, key, table): kind is "filters", "joins", or an answer
    /// port's kind.
    pub tables: Vec<(String, AnswerKey, RecordBatch)>,
    /// alias -> the documents still in play after the last operator.
    pub survivors: BTreeMap<String, Int32Array>,
    /// written position -> the rows a full join kept.
    pub true_joins: BTreeMap<i64, RecordBatch>,
    /// One record per filter and join predicate, with its measured
    /// selectivity.
    pub stages: Vec<Value>,
}

type Error = String;

fn arrow_error(error: arrow::error::ArrowError) -> Error {
    error.to_string()
}

/// Read each model operator's answers, the surviving documents, and each
/// predicate's measured selectivity from a run's output relations.
///
/// Answer tables and survivors name each document by its row in the
/// registered table when `source_rows` holds its alias.
///
/// # Errors
///
/// A filter or join predicate has no answer relation, or one lacks the
/// metadata or columns that name its predicate.
pub fn collect_answers(
    request: &AnswerRequest,
    outputs: &HashMap<(String, String), RecordBatch>,
) -> Result<Answers, Error> {
    let logical: LogicalPlan =
        quail_planner::read_logical_plan(&request.logical).map_err(|error| error.to_string())?;
    let found = operators(&logical);
    // alias -> its AI.IF predicates' selectivities, then its column tests'
    let mut filters: Vec<(String, Vec<Option<f64>>)> = Vec::new();
    for (alias, predicates) in &found.filters {
        filters.push((
            alias.clone(),
            predicates.iter().map(|found| found.selectivity).collect(),
        ));
    }
    for (alias, tests) in &found.column_filters {
        let selectivities = tests.iter().map(|test| test.selectivity);
        match filters.iter_mut().find(|(known, _)| known == alias) {
            Some((_, known)) => known.extend(selectivities),
            None => filters.push((alias.clone(), selectivities.collect())),
        }
    }

    let mut tables = Vec::new();
    let mut filter_relations: Vec<(String, Vec<RecordBatch>)> = Vec::new();
    let mut join_relations: BTreeMap<i64, RecordBatch> = BTreeMap::new();
    let mut survivors: BTreeMap<String, Int32Array> = found
        .scans
        .iter()
        .map(|scan| {
            let count = request.num_documents.get(&scan.alias).copied().unwrap_or(0);
            (scan.alias.clone(), (0..count as i32).collect())
        })
        .collect();
    for port in &request.ports {
        let Some(table) = outputs.get(&(port.node.clone(), port.port.clone())) else {
            continue;
        };
        if let Some(answer) = &port.answer {
            let mut selected = select(table, &answer.columns)?;
            if answer.kind == "classifies"
                && let Ok(index) = selected.schema().index_of(&answer.name)
            {
                let valid = is_not_null(selected.column(index)).map_err(arrow_error)?;
                selected = filter_record_batch(&selected, &valid).map_err(arrow_error)?;
            }
            tables.push((
                answer.kind.clone(),
                AnswerKey::Name(answer.name.clone()),
                selected,
            ));
        }
        let metadata = table.schema().metadata().clone();
        match port.value_type.as_str() {
            "filter_answers" => {
                let alias = metadata
                    .get("quail.alias")
                    .ok_or("a filter answer relation needs quail.alias metadata")?;
                match filter_relations
                    .iter_mut()
                    .find(|(known, _)| known == alias)
                {
                    Some((_, found)) => found.push(table.clone()),
                    None => filter_relations.push((alias.clone(), vec![table.clone()])),
                }
            }
            "join_answers" => {
                let position: i64 = metadata
                    .get("quail.written_position")
                    .ok_or("a join answer relation needs quail.written_position metadata")?
                    .parse()
                    .map_err(|_| "a join answer relation's written position is not a number")?;
                if join_relations.insert(position, table.clone()).is_some() {
                    return Err(format!("duplicate join answer relation {position}"));
                }
            }
            _ => {}
        }
        if let Some(alias) = port.port.strip_prefix("rows:")
            && let Ok(index) = table.schema().index_of(alias)
        {
            survivors.insert(alias.to_string(), document_ids(table.column(index))?);
        }
    }

    let mut missing: Vec<&String> = filters
        .iter()
        .map(|(alias, _)| alias)
        .filter(|alias| !filter_relations.iter().any(|(known, _)| known == *alias))
        .collect();
    missing.sort();
    if !missing.is_empty() {
        return Err(format!(
            "execution response is missing filter answer relations for {missing:?}"
        ));
    }
    let missing: Vec<i64> = (0..found.joins.len() as i64)
        .filter(|position| !join_relations.contains_key(position))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "execution response is missing join answer relations for {missing:?}"
        ));
    }

    let mut stages = Vec::new();
    for (alias, relations) in &filter_relations {
        let table = concat(relations)?;
        let positions = int_column(&table, 1)?;
        // predicates in the order their first answers appear
        let mut order: Vec<i64> = Vec::new();
        for position in positions.iter().flatten() {
            if !order.contains(&position) {
                order.push(position);
            }
        }
        let provided = filters
            .iter()
            .find(|(known, _)| known == alias)
            .map(|(_, found)| found.clone())
            .unwrap_or_default();
        for position in 0..provided.len() as i64 {
            if !order.contains(&position) {
                order.push(position);
            }
        }
        for (index, position) in order.into_iter().enumerate() {
            let mask: BooleanArray = positions
                .iter()
                .map(|found| Some(found == Some(position)))
                .collect();
            let stage = filter_record_batch(&table, &mask).map_err(arrow_error)?;
            let evaluated = stage.num_rows();
            let passed = true_count(&stage);
            stages.push(json!({
                "op": "filter", "alias": alias, "stage": index,
                "written_position": position,
                "provided_selectivity": provided.get(position as usize).copied().flatten(),
                "observed_selectivity": rounded(passed as f64 / evaluated.max(1) as f64),
                "evaluated": evaluated,
            }));
            tables.push((
                "filters".to_string(),
                AnswerKey::Filter(alias.clone(), position),
                stage,
            ));
        }
    }

    // joins in the order the planned joins and barriers run them, then the rest
    let mut order: Vec<i64> = request
        .ports
        .iter()
        .filter(|port| matches!(port.node_type.as_str(), "quail.ai_join" | "quail.barrier"))
        .filter_map(|port| port.port.strip_prefix("join_answers:")?.parse().ok())
        .collect();
    for position in join_relations.keys() {
        if !order.contains(position) {
            order.push(*position);
        }
    }
    let mut true_joins = BTreeMap::new();
    for position in order {
        let Some(table) = join_relations.get(&position) else {
            continue;
        };
        let join = &found.joins[position as usize];
        let aliases = call_aliases(asked_call(&join.predicate));
        let metadata = table.schema().metadata().clone();
        let mut missing: Vec<&str> = ["quail.anchor", "quail.semantics"]
            .into_iter()
            .filter(|key| !metadata.contains_key(*key))
            .collect();
        missing.sort_unstable();
        if !missing.is_empty() {
            return Err(format!(
                "a join answer relation is missing metadata {missing:?}"
            ));
        }
        let anchor = metadata["quail.anchor"].clone();
        let semantics = metadata["quail.semantics"].clone();
        let mut missing: Vec<&String> = aliases
            .iter()
            .filter(|alias| table.schema().index_of(alias).is_err())
            .collect();
        missing.sort();
        if !missing.is_empty() {
            return Err(format!(
                "a join answer relation is missing alias columns {missing:?}"
            ));
        }
        let partners: Vec<&String> = aliases.iter().filter(|alias| **alias != anchor).collect();
        let tuples = table.num_rows();
        stages.push(json!({
            "op": "join", "written_position": position, "anchor": anchor,
            "partners": partners, "semantics": semantics,
            "provided_selectivity": join.selectivity,
            "observed_selectivity": rounded(true_count(table) as f64 / tuples.max(1) as f64),
            "tuples": tuples,
        }));
        tables.push((
            "joins".to_string(),
            AnswerKey::Join(position),
            table.clone(),
        ));
        if semantics == "full" {
            true_joins.insert(position, true_answer_rows(table)?);
        }
    }

    let mut answers = Answers {
        tables,
        survivors,
        true_joins,
        stages,
    };
    to_source_rows(&mut answers, &request.source_rows)?;
    Ok(answers)
}

/// Number each document by its row in the registered table, for each
/// alias `source_rows` holds.
fn to_source_rows(
    answers: &mut Answers,
    source_rows: &BTreeMap<String, Vec<i32>>,
) -> Result<(), Error> {
    if source_rows.is_empty() {
        return Ok(());
    }
    let mappings: BTreeMap<&String, Int32Array> = source_rows
        .iter()
        .map(|(alias, rows)| (alias, Int32Array::from(rows.clone())))
        .collect();
    let renumbered = |table: &RecordBatch| -> Result<RecordBatch, Error> {
        let mut columns: Vec<ArrayRef> = table.columns().to_vec();
        for (alias, mapping) in &mappings {
            if let Ok(index) = table.schema().index_of(alias) {
                columns[index] = take(mapping, &columns[index], None).map_err(arrow_error)?;
            }
        }
        RecordBatch::try_new(table.schema(), columns).map_err(arrow_error)
    };
    for (_, _, table) in &mut answers.tables {
        *table = renumbered(table)?;
    }
    for table in answers.true_joins.values_mut() {
        *table = renumbered(table)?;
    }
    for (alias, survivors) in &mut answers.survivors {
        if let Some(mapping) = mappings.get(alias) {
            let taken = take(mapping, survivors, None).map_err(arrow_error)?;
            *survivors = taken.as_primitive::<Int32Type>().clone();
        }
    }
    Ok(())
}

/// The report of one region's run, from its plan and the backend's
/// measurements.
pub fn execution_report(
    backend: &str,
    estimated_seconds: f64,
    order_rule: Option<&str>,
    metrics: &Map<String, Value>,
    coordinator_wall: f64,
) -> Value {
    let get = |key: &str| metrics.get(key).cloned().unwrap_or(Value::Null);
    let mut report = Map::new();
    report.insert(
        "backend".into(),
        metrics
            .get("backend")
            .cloned()
            .unwrap_or_else(|| backend.into()),
    );
    for key in [
        "wall_s",
        "boot_s",
        "boot_kind",
        "boot",
        "fresh_tokens",
        "cached_tokens",
    ] {
        report.insert(key.into(), get(key));
    }
    report.insert("estimated_seconds".into(), estimated_seconds.into());
    report.insert(
        "coordinator_wall_s".into(),
        ((coordinator_wall * 100.0).round() / 100.0).into(),
    );
    report.insert("stages".into(), json!([]));
    report.insert("peak_gib".into(), get("peak_gib"));
    report.insert("order_rule".into(), order_rule.into());
    report.insert(
        "executed_join_plan".into(),
        metrics
            .get("executed_join_plan")
            .cloned()
            .unwrap_or_else(|| json!([])),
    );
    report.insert("kv_manager".into(), get("kv_manager"));
    report.insert(
        "node_metrics".into(),
        metrics
            .get("node_metrics")
            .cloned()
            .unwrap_or_else(|| json!({})),
    );
    report.insert("backend_metrics".into(), get("backend_metrics"));
    for key in [
        "gpu_s",
        "chunks",
        "memory_trace",
        "memory_trace_started",
        "memory_events",
        "memory_capacity",
        "memory_bytes",
        "memory_workers",
        "evaluated_documents",
        "evaluated_document_pairs",
        "documents_per_second",
        "document_pairs_per_second",
        "usd_per_query",
    ] {
        if let Some(value) = metrics.get(key) {
            report.insert(key.into(), value.clone());
        }
    }
    Value::Object(report)
}

fn select(table: &RecordBatch, columns: &[String]) -> Result<RecordBatch, Error> {
    let indices = columns
        .iter()
        .map(|name| table.schema().index_of(name).map_err(arrow_error))
        .collect::<Result<Vec<_>, _>>()?;
    table.project(&indices).map_err(arrow_error)
}

fn concat(tables: &[RecordBatch]) -> Result<RecordBatch, Error> {
    arrow::compute::concat_batches(&tables[0].schema(), tables).map_err(arrow_error)
}

/// A column of integers, as int64.
fn int_column(table: &RecordBatch, index: usize) -> Result<arrow::array::Int64Array, Error> {
    let column = cast(table.column(index), &DataType::Int64).map_err(arrow_error)?;
    Ok(column.as_primitive::<arrow::datatypes::Int64Type>().clone())
}

/// The number of true answers in an answer table: its last column.
fn true_count(table: &RecordBatch) -> usize {
    table
        .columns()
        .last()
        .and_then(|column| column.as_boolean_opt())
        .map_or(0, BooleanArray::true_count)
}

/// The rows of a join answer table whose answer is true, without the
/// answer column.
fn true_answer_rows(table: &RecordBatch) -> Result<RecordBatch, Error> {
    let answers = table
        .columns()
        .last()
        .and_then(|column| column.as_boolean_opt())
        .ok_or("answer table has no answer column")?
        .clone();
    let selected = filter_record_batch(table, &answers).map_err(arrow_error)?;
    let kept: Vec<usize> = (0..table.num_columns() - 1).collect();
    let selected = selected.project(&kept).map_err(arrow_error)?;
    let mut metadata = table.schema().metadata().clone();
    metadata.insert("quail.kind".into(), "join_answers".into());
    let schema = Arc::new(selected.schema().as_ref().clone().with_metadata(metadata));
    selected.with_schema(schema).map_err(arrow_error)
}

/// Document positions as one int32 array.
fn document_ids(column: &ArrayRef) -> Result<Int32Array, Error> {
    if column.null_count() > 0 {
        return Err("a document id is null".into());
    }
    let ids = cast(column, &DataType::Int32).map_err(arrow_error)?;
    Ok(ids.as_primitive::<Int32Type>().clone())
}

fn rounded(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::rounded;

    #[test]
    fn a_fraction_rounds_to_four_places() {
        assert_eq!(rounded(2.0 / 3.0), 0.6667);
        assert_eq!(rounded(0.0), 0.0);
    }
}
