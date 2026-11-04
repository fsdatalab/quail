//! The Rust cost model, exposed to Python so tests can compare it with
//! Python's.

use std::collections::BTreeMap;
use std::path::Path;

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use quail_cost::quantity::{Quantity, numpy_sum};
use quail_cost::specs::{DeviceSpec, ModelSpec, read_devices, read_models};
use quail_cost::work::{self, Work};
use quail_cost::{dense_decoder_cost, sol};

/// A number, or a list of numbers, from Python.
fn quantity(value: &Bound<'_, PyAny>) -> PyResult<Quantity> {
    if let Ok(number) = value.extract::<f64>() {
        return Ok(Quantity::Scalar(number));
    }
    Ok(Quantity::Array(value.extract::<Vec<f64>>()?))
}

fn to_python<'py>(py: Python<'py>, value: &Quantity) -> PyResult<Bound<'py, PyAny>> {
    match value {
        Quantity::Scalar(number) => Ok(number.into_pyobject(py)?.into_any()),
        Quantity::Array(numbers) => Ok(numbers.clone().into_pyobject(py)?.into_any()),
    }
}

/// One component's name, FLOPs, and bytes moved.
type ComponentCounts<'py> = (String, Bound<'py, PyAny>, Bound<'py, PyAny>);

const WORK_FIELDS: [&str; 9] = [
    "tokens",
    "pairs",
    "kv_written",
    "kv_read",
    "sliding_pairs",
    "sliding_kv_read",
    "state_written",
    "state_read",
    "sae_encoded",
];

fn work_from(fields: &Bound<'_, PyDict>) -> PyResult<Work> {
    let get = |name: &str| -> PyResult<Quantity> {
        match fields.get_item(name)? {
            Some(value) => quantity(&value),
            None => Ok(Quantity::Scalar(0.0)),
        }
    };
    Ok(Work {
        tokens: get("tokens")?,
        pairs: get("pairs")?,
        kv_written: get("kv_written")?,
        kv_read: get("kv_read")?,
        sliding_pairs: get("sliding_pairs")?,
        sliding_kv_read: get("sliding_kv_read")?,
        state_written: get("state_written")?,
        state_read: get("state_read")?,
        sae_encoded: get("sae_encoded")?,
    })
}

fn work_to<'py>(py: Python<'py>, work: &Work) -> PyResult<Bound<'py, PyDict>> {
    let fields = PyDict::new(py);
    let values = [
        &work.tokens,
        &work.pairs,
        &work.kv_written,
        &work.kv_read,
        &work.sliding_pairs,
        &work.sliding_kv_read,
        &work.state_written,
        &work.state_read,
        &work.sae_encoded,
    ];
    for (name, value) in WORK_FIELDS.iter().zip(values) {
        fields.set_item(*name, to_python(py, value)?)?;
    }
    Ok(fields)
}

/// The Rust cost model over the specs in one directory.
#[pyclass(name = "CostModel", module = "_quail_datafusion")]
pub struct CostModel {
    models: BTreeMap<String, ModelSpec>,
    devices: BTreeMap<String, DeviceSpec>,
}

impl CostModel {
    fn model(&self, name: &str) -> PyResult<&ModelSpec> {
        self.models
            .get(name)
            .ok_or_else(|| PyKeyError::new_err(name.to_string()))
    }

    fn device(&self, name: &str) -> PyResult<&DeviceSpec> {
        self.devices
            .get(name)
            .ok_or_else(|| PyKeyError::new_err(name.to_string()))
    }
}

#[pymethods]
impl CostModel {
    /// Read the model and device specs under `data`.
    #[new]
    fn new(data: &str) -> PyResult<Self> {
        let data = Path::new(data);
        let error = |error: quail_cost::specs::SpecError| PyValueError::new_err(error.0);
        Ok(Self {
            models: read_models(&data.join("models")).map_err(error)?,
            devices: read_devices(&data.join("devices")).map_err(error)?,
        })
    }

    /// The names of the models read.
    fn model_names(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// The names of the devices read.
    fn device_names(&self) -> Vec<String> {
        self.devices.keys().cloned().collect()
    }

    /// A model's derived properties, by the names ModelSpec gives them.
    fn model_properties<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
        let model = self.model(name)?;
        let properties = PyDict::new(py);
        properties.set_item("reuses_kv", model.reuses_kv())?;
        properties.set_item("attention_layers", model.attention_layers())?;
        properties.set_item("gdn_layers", model.gdn_layers())?;
        properties.set_item("short_conv_layers", model.short_conv_layers())?;
        properties.set_item("state_layers", model.state_layers())?;
        properties.set_item("conv_state_channels", model.conv_state_channels())?;
        properties.set_item("conv_channels", model.conv_channels())?;
        properties.set_item("gdn_state_bytes", model.gdn_state_bytes())?;
        properties.set_item("conv_state_bytes", model.conv_state_bytes())?;
        properties.set_item("checkpoint_bytes", model.checkpoint_bytes())?;
        properties.set_item("kv_shapes", model.kv_shapes())?;
        properties.set_item("widest_projection", model.widest_projection())?;
        properties.set_item("kappa", model.kappa())?;
        properties.set_item("kv_elements_per_token", model.kv_elements_per_token())?;
        properties.set_item("kappa_sliding", model.kappa_sliding())?;
        properties.set_item("kappa_full", model.kappa_full())?;
        properties.set_item("W_mem", model.W_mem())?;
        properties.set_item("head_mem_bytes", model.head_mem_bytes())?;
        properties.set_item("W_resident", model.W_resident())?;
        properties.set_item("act_per_token", model.act_per_token())?;
        properties.set_item("intermediate", model.intermediate())?;
        properties.set_item("dense_params", dense_decoder_cost::dense_params(model))?;
        properties.set_item(
            "gdn_core_flops_per_token",
            dense_decoder_cost::gdn_core_flops_per_token(model),
        )?;
        properties.set_item(
            "conv_flops_per_token",
            dense_decoder_cost::conv_flops_per_token(model),
        )?;
        properties.set_item(
            "attention_pair_flops",
            dense_decoder_cost::attention_pair_flops(model),
        )?;
        Ok(properties)
    }

    /// One device's GEMM rate for a precision.
    fn gemm_flops(&self, device: &str, precision: &str) -> PyResult<f64> {
        Ok(self.device(device)?.gemm_flops(precision))
    }

    /// The work of one `quail.common.work` function: scan, ask, stream,
    /// packed, or decode. `other` is the suffix, the suffixes, the
    /// questions, or the rounds.
    #[pyo3(signature = (kind, prefix, other, window=0))]
    fn work<'py>(
        &self,
        py: Python<'py>,
        kind: &str,
        prefix: &Bound<'py, PyAny>,
        other: &Bound<'py, PyAny>,
        window: i64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let prefix = quantity(prefix)?;
        let counted = match kind {
            "scan" => work::scan(&prefix, &quantity(other)?, window),
            "ask" => work::ask(&prefix, &quantity(other)?, window),
            "stream" => {
                let suffixes = other
                    .try_iter()?
                    .map(|item| quantity(&item?))
                    .collect::<PyResult<Vec<_>>>()?;
                work::stream(&prefix, &suffixes, window)
            }
            "packed" => work::packed(&prefix, &other.extract::<Vec<f64>>()?, window),
            "decode" => work::decode(&prefix, &other.extract::<Vec<f64>>()?, window),
            other => return Err(PyValueError::new_err(format!("no work function {other}"))),
        };
        work_to(py, &counted)
    }

    /// The decoder components of some work: (name, FLOPs, bytes).
    fn components<'py>(
        &self,
        py: Python<'py>,
        work: &Bound<'py, PyDict>,
        model: &str,
        passes: f64,
        rows: f64,
        head_rows: i64,
    ) -> PyResult<Vec<ComponentCounts<'py>>> {
        let components = dense_decoder_cost::dense_decoder_components(
            &work_from(work)?,
            self.model(model)?,
            passes,
            rows,
            head_rows,
        );
        components
            .iter()
            .map(|component| {
                Ok((
                    component.name.to_string(),
                    to_python(py, &component.flops)?,
                    to_python(py, &component.bytes_moved)?,
                ))
            })
            .collect()
    }

    /// `quail.common.sol.unrounded_seconds`.
    fn unrounded_seconds(
        &self,
        work: &Bound<'_, PyDict>,
        model: &str,
        device: &str,
        chunk_tokens: i64,
        rows: f64,
        head_rows: i64,
    ) -> PyResult<f64> {
        Ok(sol::unrounded_seconds(
            &work_from(work)?,
            self.model(model)?,
            self.device(device)?,
            chunk_tokens,
            rows,
            head_rows,
        ))
    }

    /// `quail.common.sol.compute_seconds`.
    fn compute_seconds(
        &self,
        work: &Bound<'_, PyDict>,
        model: &str,
        device: &str,
        peak: bool,
    ) -> PyResult<f64> {
        Ok(sol::compute_seconds(
            &work_from(work)?,
            self.model(model)?,
            self.device(device)?,
            peak,
        ))
    }

    /// `quail.common.sol.speed_of_light`: passes, seconds, compute
    /// seconds, memory seconds, and bytes moved.
    fn speed_of_light(
        &self,
        work: &Bound<'_, PyDict>,
        model: &str,
        device: &str,
        chunk_tokens: i64,
        rows: f64,
        min_passes: i64,
    ) -> PyResult<(i64, f64, f64, f64, f64)> {
        let estimate = sol::speed_of_light(
            &work_from(work)?,
            self.model(model)?,
            self.device(device)?,
            chunk_tokens,
            rows,
            min_passes,
        );
        Ok((
            estimate.passes,
            estimate.seconds(),
            estimate.compute(),
            estimate.memory(),
            estimate.bytes_moved(),
        ))
    }

    /// `float(np.sum(values))`, as the Rust port rounds it.
    #[staticmethod]
    fn numpy_sum(values: Vec<f64>) -> f64 {
        numpy_sum(&values)
    }
}
