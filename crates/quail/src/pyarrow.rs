//! Arrow batches to and from pyarrow over the same buffers, through the
//! Arrow C stream interface.

use std::ffi::CStr;

use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use datafusion::arrow::array::{RecordBatch, RecordBatchIterator};
use datafusion::arrow::datatypes::SchemaRef;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};

/// The capsule name of the Arrow C stream interface.
const ARROW_STREAM: &CStr = c"arrow_array_stream";

/// Hand batches to Python as a pyarrow Table over the same buffers.
///
/// # Errors
///
/// pyarrow is not importable.
pub fn to_pyarrow<'py>(
    py: Python<'py>,
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
) -> PyResult<Bound<'py, PyAny>> {
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    let stream = FFI_ArrowArrayStream::new(Box::new(reader));
    // pyarrow takes the stream; the capsule's destructor releases it
    // only when pyarrow did not
    let capsule = PyCapsule::new_with_value(py, stream, ARROW_STREAM)?;
    py.import("pyarrow")?
        .getattr("RecordBatchReader")?
        .call_method1("_import_from_c_capsule", (capsule,))?
        .call_method0("read_all")
}

/// Take a Python object's Arrow stream, such as a pyarrow Table's or
/// RecordBatchReader's, as a reader over the same buffers.
///
/// # Errors
///
/// The object has no Arrow C stream.
pub fn stream_from_pyarrow(value: &Bound<'_, PyAny>) -> PyResult<ArrowArrayStreamReader> {
    let capsule = value.call_method0("__arrow_c_stream__")?;
    let capsule = capsule.cast::<PyCapsule>()?;
    let pointer = capsule.pointer_checked(Some(ARROW_STREAM))?;
    // the reader takes the stream and leaves the capsule's released
    unsafe { ArrowArrayStreamReader::from_raw(pointer.as_ptr().cast()) }
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

/// Read a Python object's Arrow stream into batches over the same buffers.
///
/// # Errors
///
/// The object has no Arrow C stream, or a batch does not read.
pub fn from_pyarrow(value: &Bound<'_, PyAny>) -> PyResult<Vec<RecordBatch>> {
    stream_from_pyarrow(value)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}
