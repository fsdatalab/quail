use std::ffi::CStr;

use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use datafusion::arrow::array::{Array, ArrayRef, RecordBatch, RecordBatchIterator};
use datafusion::arrow::datatypes::SchemaRef;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};

const ARROW_STREAM: &CStr = c"arrow_array_stream";
const ARROW_ARRAY: &CStr = c"arrow_array";
const ARROW_SCHEMA: &CStr = c"arrow_schema";

pub fn to_pyarrow<'py>(
    py: Python<'py>,
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
) -> PyResult<Bound<'py, PyAny>> {
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    let stream = FFI_ArrowArrayStream::new(Box::new(reader));
    let capsule = PyCapsule::new_with_value(py, stream, ARROW_STREAM)?;
    py.import("pyarrow")?
        .getattr("RecordBatchReader")?
        .call_method1("_import_from_c_capsule", (capsule,))?
        .call_method0("read_all")
}

pub fn array_to_pyarrow<'py>(py: Python<'py>, array: &ArrayRef) -> PyResult<Bound<'py, PyAny>> {
    let (data, schema) = arrow::ffi::to_ffi(&array.to_data())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let schema: FFI_ArrowSchema = schema;
    let data: FFI_ArrowArray = data;
    let schema = PyCapsule::new_with_value(py, schema, ARROW_SCHEMA)?;
    let data = PyCapsule::new_with_value(py, data, ARROW_ARRAY)?;
    py.import("pyarrow")?
        .getattr("Array")?
        .call_method1("_import_from_c_capsule", (schema, data))
}

pub fn stream_from_pyarrow(value: &Bound<'_, PyAny>) -> PyResult<ArrowArrayStreamReader> {
    let capsule = value.call_method0("__arrow_c_stream__")?;
    let capsule = capsule.cast::<PyCapsule>()?;
    let pointer = capsule.pointer_checked(Some(ARROW_STREAM))?;
    unsafe { ArrowArrayStreamReader::from_raw(pointer.as_ptr().cast()) }
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

pub fn from_pyarrow(value: &Bound<'_, PyAny>) -> PyResult<Vec<RecordBatch>> {
    stream_from_pyarrow(value)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}
