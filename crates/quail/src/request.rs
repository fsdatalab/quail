//! The tables a planned region sends its workers: each scan's
//! documents, its documents' text, and the value columns the executor
//! reads.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, Float32Array, Float64Array, Int32Array, RecordBatch, new_empty_array,
};
use arrow::compute::{cast, concat};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::row::{RowConverter, SortField};
use quail_datafusion::derive::expressions::PhysicalValue;
use quail_planner::constants::NULL_DOCUMENT;
use quail_planner::error::CompileError;
use quail_planner::expressions::column_references;
use quail_planner::inputs::Documents;
use quail_planner::operators::{Operators, ScanColumn};
use quail_planner::physical::{PhysicalGraph, PhysicalNode};
use serde_json::Value;

use crate::shared::Table;

/// The column of a table the plan deduplicates that holds each row's
/// first row with the same document.
pub const FIRST_DOCUMENT: &str = "__first_document";

/// The column of a token store that holds each document's token ids.
const TOKEN_IDS: &str = "__quail_token_ids";

/// The value column that holds a scan's image documents.
const IMAGES: &str = "__quail_images";

/// What a region reads of one scan: its alias, its token store, and
/// the columns of its rows.
pub type Scanned = (String, Table, Table);

fn arrow_error(error: arrow::error::ArrowError) -> CompileError {
    CompileError::new(error.to_string())
}

/// One column of a table as its chunks, one per batch, so a column of
/// many images reaches the executor without a copy.
type Chunks = Vec<ArrayRef>;

/// One column of a table, as its chunks: one empty chunk for a table of
/// no batches.
fn column_chunks(table: &Table, name: &str) -> Option<Chunks> {
    let (schema, batches) = table;
    let index = schema.index_of(name).ok()?;
    Some(match batches.is_empty() {
        true => vec![new_empty_array(schema.field(index).data_type())],
        false => (batches.iter())
            .map(|batch| Arc::clone(batch.column(index)))
            .collect(),
    })
}

/// One column of a table, as one array, for a key that compares rows
/// across batches.
fn column(table: &Table, name: &str) -> Result<Option<ArrayRef>, CompileError> {
    let Some(chunks) = column_chunks(table, name) else {
        return Ok(None);
    };
    concat(&chunks.iter().map(AsRef::as_ref).collect::<Vec<_>>())
        .map(Some)
        .map_err(arrow_error)
}

/// One array split into chunks of `lengths` rows, without a copy.
fn split(values: &ArrayRef, lengths: &[usize]) -> Chunks {
    let mut offset = 0;
    (lengths.iter())
        .map(|length| {
            let chunk = values.slice(offset, *length);
            offset += length;
            chunk
        })
        .collect()
}

/// A table of one batch per chunk; every column has the same chunks.
fn table_of(columns: Vec<(String, Chunks)>) -> Result<Table, CompileError> {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, chunks)| Field::new(name, chunks[0].data_type().clone(), true))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let num_batches = columns.first().map_or(1, |(_, chunks)| chunks.len());
    let batches = (0..num_batches)
        .map(|index| {
            let arrays = (columns.iter())
                .map(|(_, chunks)| Arc::clone(&chunks[index]))
                .collect();
            RecordBatch::try_new(Arc::clone(&schema), arrays).map_err(arrow_error)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((schema, batches))
}

/// Values as keys that group by numeric equality: a dictionary is
/// decoded, a signed zero is 0, and every NaN is one key.
fn equality_keys(values: ArrayRef) -> Result<ArrayRef, CompileError> {
    let values = match values.data_type() {
        DataType::Dictionary(_, value) => cast(&values, value).map_err(arrow_error)?,
        _ => values,
    };
    Ok(match values.data_type() {
        DataType::Float64 => {
            let keys: Float64Array = values
                .as_primitive::<arrow::datatypes::Float64Type>()
                .unary(|value| {
                    if value == 0.0 {
                        0.0
                    } else if value.is_nan() {
                        f64::NAN
                    } else {
                        value
                    }
                });
            Arc::new(keys)
        }
        DataType::Float32 => {
            let keys: Float32Array = values
                .as_primitive::<arrow::datatypes::Float32Type>()
                .unary(|value| {
                    if value == 0.0 {
                        0.0
                    } else if value.is_nan() {
                        f32::NAN
                    } else {
                        value
                    }
                });
            Arc::new(keys)
        }
        _ => values,
    })
}

/// Each row's first row with the same tokens and the same values in
/// each of `names`.
fn first_documents(
    tokens: &Table,
    columns: &Table,
    names: &[String],
) -> Result<ArrayRef, CompileError> {
    let lists = DataType::LargeList(Arc::new(Field::new_list_field(DataType::Int32, true)));
    let index = (tokens.0)
        .index_of(TOKEN_IDS)
        .map_err(|_| CompileError::new("a token store has no token ids"))?;
    let documents = (tokens.1.iter())
        .map(|batch| {
            let ids = cast(batch.column(index), &lists).map_err(arrow_error)?;
            Ok(ids.as_list::<i64>().clone())
        })
        .collect::<Result<Vec<_>, CompileError>>()?;
    let values = names
        .iter()
        .map(|name| {
            let found = column(columns, name)?
                .ok_or_else(|| CompileError::new(format!("no column {name}")))?;
            equality_keys(found)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let first = quail_planner::inputs::first_documents(&Documents::new(&documents), &values)
        .map_err(arrow_error)?;
    Ok(Arc::new(Int32Array::from_iter_values(
        first.into_iter().map(|row| row as i32),
    )))
}

/// A scan's documents, evaluated over each batch of its columns, when
/// the columns hold every column the document reads.
fn document_values(document: &ScanColumn, columns: &Table) -> Result<Option<Chunks>, CompileError> {
    let expression = match document {
        ScanColumn::Unnamed(name) => return Ok(column_chunks(columns, name)),
        ScanColumn::Expression(expression) => expression,
    };
    let held = |name: &str| columns.0.index_of(name).is_ok();
    if !(column_references(expression).iter()).all(|reference| held(&reference.column)) {
        return Ok(None);
    }
    let built = PhysicalValue::new(expression)?;
    let chunks = (columns.1.iter())
        .map(|batch| built.evaluate(batch, false))
        .collect::<Result<Vec<_>, _>>()?;
    // a scan of no rows has no batch, and its text is no rows
    if chunks.is_empty() {
        return Ok(Some(vec![new_empty_array(&DataType::Utf8)]));
    }
    Ok(Some(chunks))
}

/// The text of a scan's documents, when the scan's columns hold every
/// column the document reads; None for image documents.
fn texts(document: &ScanColumn, columns: &Table) -> Result<Option<Chunks>, CompileError> {
    Ok(document_values(document, columns)?
        .filter(|chunks| !crate::images::holds_images(chunks[0].data_type())))
}

/// The tables a region's workers read, by name: `documents:<alias>`
/// for each scan of the plan, `texts:<alias>` where the scan's columns
/// hold its documents' text, and `columns:<alias>` with the value
/// columns the executor reads.
///
/// A table the plan deduplicates carries [`FIRST_DOCUMENT`], and a
/// table with a NULL document carries [`NULL_DOCUMENT`].
///
/// # Errors
///
/// The plan does not decode, a scan of the plan is not in `scans`, or
/// a column the executor reads is missing.
pub fn request_tables(
    envelope: &Value,
    operators: &Operators,
    scans: &[Scanned],
) -> Result<Vec<(String, Table)>, CompileError> {
    let decode = |error: serde_json::Error| CompileError::new(error.to_string());
    let graph: PhysicalGraph = serde_json::from_value(envelope["graph"].clone()).map_err(decode)?;
    let deduplicated: BTreeMap<String, Vec<String>> = match envelope["settings"].get("deduplicate")
    {
        Some(found) => serde_json::from_value(found.clone()).map_err(decode)?,
        None => BTreeMap::new(),
    };
    let facts = quail_planner::plan::facts::region_facts(operators)?;
    let read: BTreeMap<String, Vec<String>> =
        serde_json::from_value(facts["value_columns"].clone()).map_err(decode)?;
    let scanned = |alias: &str| {
        (scans.iter())
            .find(|(known, _, _)| known == alias)
            .ok_or_else(|| CompileError::new(format!("no rows for the scan of {alias}")))
    };
    let mut tables = Vec::new();
    for node in &graph.nodes {
        let PhysicalNode::Scan(scan) = node else {
            continue;
        };
        let alias = &scan.attributes.alias;
        let (_, tokens, columns) = scanned(alias)?;
        tables.push((format!("documents:{alias}"), tokens.clone()));
        let document = (operators.scans.iter()).find(|found| &found.alias == alias);
        if let Some(text) = document
            .map(|found| texts(&found.column, columns))
            .transpose()?
            .flatten()
        {
            tables.push((
                format!("texts:{alias}"),
                table_of(vec![("text".into(), text)])?,
            ));
        }
    }
    for (alias, _, columns) in scans {
        let names = read.get(alias);
        let rows = deduplicated.get(alias);
        let mut values = Vec::new();
        for name in names.into_iter().flatten() {
            let found = column_chunks(columns, name)
                .ok_or_else(|| CompileError::new(format!("no column {name}")))?;
            values.push((name.clone(), found));
        }
        if let Some(rows) = rows {
            let (_, tokens, _) = scanned(alias)?;
            let lengths: Vec<usize> = match columns.1.is_empty() {
                true => vec![0],
                false => columns.1.iter().map(RecordBatch::num_rows).collect(),
            };
            let first = first_documents(tokens, columns, rows)?;
            values.push((FIRST_DOCUMENT.into(), split(&first, &lengths)));
        }
        // the executor reads an image document's pixels from the first
        // binary column: the document's values, such as p.u['image']
        let document = (operators.scans.iter()).find(|found| &found.alias == alias);
        if let Some(images) = document
            .map(|found| document_values(&found.column, columns))
            .transpose()?
            .flatten()
            .filter(|found| crate::images::holds_images(found[0].data_type()))
        {
            values.retain(|(name, _)| name != IMAGES);
            values.insert(0, (IMAGES.into(), images));
        }
        if let Some(absent) = column_chunks(columns, NULL_DOCUMENT) {
            values.push((NULL_DOCUMENT.into(), absent));
        }
        if names.is_some() || rows.is_some() || !values.is_empty() {
            tables.push((format!("columns:{alias}"), table_of(values)?));
        }
    }
    Ok(tables)
}

/// By alias, lists of each document's group, for the stages' hash: a
/// document goes to the GPU of the smallest document it is grouped
/// with ([`crate::stages`]).
///
/// One list holds each document's first document, for a table the
/// plan deduplicates; one holds each AI filter's stop key groups, so a
/// key's documents reach one GPU.
///
/// # Errors
///
/// A table the plan groups has no columns, or lacks a grouped column.
pub fn stage_groupings(
    graph: &PhysicalGraph,
    deduplicated: &[String],
    tables: &[(String, Table)],
) -> Result<HashMap<String, Vec<Vec<i64>>>, CompileError> {
    let columns = |alias: &str| {
        let name = format!("columns:{alias}");
        (tables.iter())
            .find(|(known, _)| *known == name)
            .map(|(_, table)| table)
            .ok_or_else(|| CompileError::new(format!("{alias} has no value columns")))
    };
    let held = |alias: &str, name: &str| {
        column(columns(alias)?, name)?.ok_or_else(|| CompileError::new(format!("no column {name}")))
    };
    let mut groupings: HashMap<String, Vec<Vec<i64>>> = HashMap::new();
    for alias in deduplicated {
        let first = cast(&held(alias, FIRST_DOCUMENT)?, &DataType::Int64).map_err(arrow_error)?;
        let first = first.as_primitive::<arrow::datatypes::Int64Type>();
        groupings
            .entry(alias.clone())
            .or_default()
            .push(first.values().to_vec());
    }
    for node in &graph.nodes {
        let PhysicalNode::AiFilter(filter) = node else {
            continue;
        };
        let (alias, key) = (&filter.attributes.alias, &filter.attributes.stop_key);
        if key.is_empty() {
            continue;
        }
        let values = key
            .iter()
            .map(|name| equality_keys(held(alias, name)?))
            .collect::<Result<Vec<_>, _>>()?;
        let fields = (values.iter())
            .map(|values| SortField::new(values.data_type().clone()))
            .collect();
        let converter = RowConverter::new(fields).map_err(arrow_error)?;
        let rows = converter.convert_columns(&values).map_err(arrow_error)?;
        // a key's group is the first document with its values
        let mut first: HashMap<&[u8], i64> = HashMap::new();
        let groups = (rows.iter().enumerate())
            .map(|(document, row)| *first.entry(row.data()).or_insert(document as i64))
            .collect();
        groupings.entry(alias.clone()).or_default().push(groups);
    }
    Ok(groupings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equality_keys_join_signed_zeros_and_nans() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(-0.0),
            Some(0.0),
            Some(f64::NAN),
            Some(-f64::NAN),
            None,
            Some(1.5),
        ]));
        let keys = equality_keys(values).unwrap();
        let keys = keys.as_primitive::<arrow::datatypes::Float64Type>();
        assert_eq!(keys.value(0).to_bits(), 0.0f64.to_bits());
        assert_eq!(keys.value(2).to_bits(), keys.value(3).to_bits());
        assert!(keys.is_null(4));
        assert_eq!(keys.value(5), 1.5);
    }
}
