//! A model region's input as the executor reads it, and the hint over
//! the rows it reads.
//!
//! [`InputReader`] hands the executor one region input batch by batch,
//! as DataFusion produces it, so the region's GPU loop starts before the
//! input ends. [`InputHint`] says whether each row the executor reads may
//! still pass the tests above the region.

use std::collections::HashMap;
use std::sync::mpsc::TryRecvError;
use std::sync::{Arc, Mutex};

use datafusion::arrow::array::{ArrayRef, BooleanArray, Int32Array, RecordBatch};
use datafusion::arrow::compute::{concat_batches, filter_record_batch};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::{DataFusionError, Result};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use quail_datafusion::expressions::PhysicalValue;
use quail_datafusion::{InputBatches, RegionColumns, RegionHint};
use quail_planner::operators::ScanColumn;

use crate::planning::{Planner, document_texts};
use crate::pyarrow::{array_to_pyarrow, to_pyarrow};

/// The rows of each table a hint reads, by table name, in the order the
/// executor numbers them.
pub(crate) type HintRows = Arc<Mutex<HashMap<String, Vec<RecordBatch>>>>;

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// A region's hint over the rows of the tables the executor reads:
/// whether each row may still pass the tests above the region, as they
/// tighten while the region runs.
#[pyclass(name = "InputHint", module = "_quail_native", frozen)]
pub(crate) struct InputHint {
    hint: RegionHint,
    columns: RegionColumns,
    rows: HintRows,
}

impl InputHint {
    pub(crate) fn new(hint: RegionHint, columns: RegionColumns, rows: HintRows) -> Self {
        Self {
            hint,
            columns,
            rows,
        }
    }
}

#[pymethods]
impl InputHint {
    /// A number that changes whenever a test tightens.
    fn generation(&self) -> u64 {
        self.hint.generation()
    }

    /// The tables the hint holds rows of.
    fn tables(&self) -> Vec<String> {
        self.rows
            .lock()
            .expect("the hint rows lock")
            .keys()
            .cloned()
            .collect()
    }

    /// Whether each row of `table` read so far, in the order the
    /// executor numbers them, may still pass every test.
    fn keep(&self, table: &str) -> PyResult<Vec<bool>> {
        let rows = self.rows.lock().expect("the hint rows lock");
        let Some(first) = rows.get(table).and_then(|batches| batches.first()) else {
            return Ok(Vec::new());
        };
        let all = concat_batches(&first.schema(), &rows[table]).map_err(runtime_error)?;
        self.hint
            .keep(&self.columns, table, &all)
            .map_err(runtime_error)
    }
}

/// Keep the rows of `batch`, rows of `table`, that may still pass every
/// test of `hint`.
pub(crate) fn hinted_rows(
    hint: &RegionHint,
    columns: &RegionColumns,
    table: &str,
    batch: &RecordBatch,
) -> Result<RecordBatch> {
    let keep = BooleanArray::from(hint.keep(columns, table, batch)?);
    Ok(filter_record_batch(batch, &keep)?)
}

/// What an [`InputReader`] has read of its input.
struct Reading {
    batches: InputBatches,
    ended: bool,
    /// The rows handed on so far, the next row's id.
    num_rows: usize,
}

/// How an [`InputReader`] tokenizes its documents.
pub(crate) struct Documents {
    pub(crate) planner: Arc<Planner>,
    pub(crate) column: ScanColumn,
    /// Whether each token's character range is read too, for AI.EXTRACT.
    pub(crate) with_offsets: bool,
}

/// One region input the executor reads batch by batch, as DataFusion
/// produces it.
///
/// Each read hands on the next rows: their ids, numbered from 0 in the
/// order read; their documents' tokens; and the input columns the
/// executor reads. A row the region's hint rejects as it arrives gets no
/// id.
#[pyclass(name = "InputReader", module = "_quail_native", frozen)]
pub(crate) struct InputReader {
    reading: Mutex<Reading>,
    documents: Documents,
    /// The input columns handed on with each row.
    columns: SchemaRef,
    /// The hint, when it reads this input's rows.
    hint: Option<(RegionHint, RegionColumns, HintRows)>,
    /// The table name the region reads the input by.
    #[pyo3(get)]
    table: String,
}

impl InputReader {
    pub(crate) fn new(
        table: String,
        batches: InputBatches,
        documents: Documents,
        columns: SchemaRef,
        hint: Option<(RegionHint, RegionColumns, HintRows)>,
    ) -> Self {
        if let Some((_, _, rows)) = &hint {
            rows.lock()
                .expect("the hint rows lock")
                .insert(table.clone(), Vec::new());
        }
        Self {
            reading: Mutex::new(Reading {
                batches,
                ended: false,
                num_rows: 0,
            }),
            documents,
            columns,
            hint,
            table,
        }
    }

    /// The table name the region reads the input by.
    pub(crate) fn table_name(&self) -> String {
        self.table.clone()
    }

    /// The next batch of the input, or None: with `block`, None once the
    /// input has ended; without, also while no batch has arrived.
    fn receive(&self, py: Python<'_>, block: bool) -> Result<Option<RecordBatch>> {
        let mut reading = self.reading.lock().expect("the reading lock");
        if reading.ended {
            return Ok(None);
        }
        let received = match block {
            true => {
                let batches = &mut reading.batches;
                py.detach(move || batches.recv().ok())
            }
            false => match reading.batches.try_recv() {
                Ok(batch) => Some(batch),
                Err(TryRecvError::Empty) => return Ok(None),
                Err(TryRecvError::Disconnected) => None,
            },
        };
        match received {
            Some(batch) => batch.map(Some),
            None => {
                reading.ended = true;
                Ok(None)
            }
        }
    }

    /// Number a batch's rows the hint keeps, and record them for the hint.
    fn number(&self, batch: RecordBatch) -> Result<(Int32Array, RecordBatch)> {
        let batch = match &self.hint {
            Some((hint, columns, rows)) => {
                let kept = hinted_rows(hint, columns, &self.table, &batch)?;
                rows.lock()
                    .expect("the hint rows lock")
                    .entry(self.table.clone())
                    .or_default()
                    .push(kept.clone());
                kept
            }
            None => batch,
        };
        let mut reading = self.reading.lock().expect("the reading lock");
        let start = reading.num_rows;
        reading.num_rows += batch.num_rows();
        let ids = (start..reading.num_rows).map(|id| id as i32).collect();
        Ok((ids, batch))
    }

    /// The documents' tokens, and each token's character range when read.
    fn tokens(&self, py: Python<'_>, batch: &RecordBatch) -> Result<(ArrayRef, Option<ArrayRef>)> {
        let documents = &self.documents;
        let built = match &documents.column {
            ScanColumn::Expression(expression) => Some(
                PhysicalValue::new(expression).map_err(|error| DataFusionError::Plan(error.0))?,
            ),
            ScanColumn::Unnamed(_) => None,
        };
        let texts = document_texts(&documents.column, built.as_ref(), batch)
            .map_err(|error| DataFusionError::Plan(error.0))?;
        let tokenizer = documents.planner.tokenizer();
        py.detach(|| match documents.with_offsets {
            true => {
                let (tokens, ranges) = tokenizer.encode_batch_with_offsets(&texts)?;
                Ok((
                    Arc::new(tokens) as ArrayRef,
                    Some(Arc::new(ranges) as ArrayRef),
                ))
            }
            false => Ok((Arc::new(tokenizer.encode_batch(&texts)?) as ArrayRef, None)),
        })
    }

    /// The columns the executor reads of a batch's rows.
    fn handed_columns(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let schema = batch.schema();
        let mut arrays: Vec<ArrayRef> = Vec::new();
        for field in self.columns.fields() {
            arrays.push(Arc::clone(batch.column(schema.index_of(field.name())?)));
        }
        Ok(RecordBatch::try_new_with_options(
            Arc::clone(&self.columns),
            arrays,
            &datafusion::arrow::array::RecordBatchOptions::new()
                .with_row_count(Some(batch.num_rows())),
        )?)
    }
}

#[pymethods]
impl InputReader {
    /// The input columns handed on with each row, as a pyarrow Schema.
    #[getter]
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_pyarrow(py, Arc::clone(&self.columns), Vec::new())?.getattr("schema")
    }

    /// Whether the input has ended and every row was handed on.
    #[getter]
    fn ended(&self) -> bool {
        self.reading.lock().expect("the reading lock").ended
    }

    /// The next rows of the input, as (ids, tokens, character ranges or
    /// None, columns): ids an int32 pyarrow Array, tokens a large list
    /// of int32 token ids per row, and columns a pyarrow Table.
    ///
    /// Returns None when the input has ended, or, unless `block`, when
    /// no rows have arrived yet; with `block`, waits for either.
    fn next<'py>(&self, py: Python<'py>, block: bool) -> PyResult<Option<Bound<'py, PyTuple>>> {
        loop {
            let Some(batch) = self.receive(py, block).map_err(runtime_error)? else {
                return Ok(None);
            };
            let (ids, batch) = self.number(batch).map_err(runtime_error)?;
            if batch.num_rows() == 0 {
                continue;
            }
            let (tokens, ranges) = self.tokens(py, &batch).map_err(runtime_error)?;
            let columns = self.handed_columns(&batch).map_err(runtime_error)?;
            let ids: ArrayRef = Arc::new(ids);
            let ranges = match ranges {
                Some(ranges) => Some(array_to_pyarrow(py, &ranges)?),
                None => None,
            };
            return Ok(Some(PyTuple::new(
                py,
                [
                    array_to_pyarrow(py, &ids)?,
                    array_to_pyarrow(py, &tokens)?,
                    ranges.unwrap_or_else(|| py.None().into_bound(py)),
                    to_pyarrow(py, columns.schema(), vec![columns])?,
                ],
            )?));
        }
    }
}
