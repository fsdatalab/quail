//! The Rust cost model, exposed to Python for tests.

use std::collections::BTreeMap;
use std::path::Path;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use quail_cost::classify::{self, ClassifyInputs};
use quail_cost::extract::{self, ExtractInputs};
use quail_cost::labels;
use quail_cost::quantity::{Quantity, numpy_sum};
use quail_cost::specs::{DeviceSpec, ModelSpec, read_devices, read_models};
use quail_cost::statistics::AliasStats;
use quail_cost::work::Work;
use quail_cost::{budgets, dense_decoder_cost, packing, sol};

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

/// Read nodes, fed nodes, and each label's decode round tokens.
type LabelNodes = (Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<Vec<i64>>);

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
#[pyclass(name = "CostModel", module = "_quail_native")]
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

    /// `arena_pages`; raises ValueError where Python's does.
    #[pyo3(signature = (model, device, chunk_tokens=None, mean_document_tokens=None))]
    fn arena_pages(
        &self,
        model: &str,
        device: &str,
        chunk_tokens: Option<i64>,
        mean_document_tokens: Option<f64>,
    ) -> PyResult<(i64, i64)> {
        budgets::arena_pages(
            self.model(model)?,
            self.device(device)?,
            chunk_tokens,
            mean_document_tokens,
        )
        .map_err(PyValueError::new_err)
    }

    /// `checkpoint_slots`.
    fn checkpoint_slots(
        &self,
        model: &str,
        device: &str,
        chunk_tokens: i64,
        kv_pages: i64,
    ) -> PyResult<i64> {
        Ok(budgets::checkpoint_slots(
            self.model(model)?,
            self.device(device)?,
            chunk_tokens,
            kv_pages,
        ))
    }

    /// `summarize_alias`: the summary's fields, from an int64 buffer of
    /// document token lengths.
    #[staticmethod]
    fn summarize_alias<'py>(
        py: Python<'py>,
        lengths: PyBuffer<i64>,
        window: i64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let values = lengths.to_vec(py)?;
        let stats = AliasStats::summarize(&values, window);
        let fields = PyDict::new(py);
        fields.set_item("count", stats.count)?;
        fields.set_item("total", stats.total)?;
        fields.set_item("squared", stats.squared)?;
        fields.set_item("maximum", stats.maximum)?;
        fields.set_item("window", stats.window)?;
        fields.set_item("short_counts", stats.short_counts)?;
        Ok(fields)
    }

    /// `choose_pass_groups`; raises ValueError where Python's does.
    #[allow(clippy::too_many_arguments)]
    fn choose_pass_groups(
        &self,
        question_tokens: Vec<i64>,
        selectivities: Vec<f64>,
        rows: f64,
        state_tokens: f64,
        longest_state: i64,
        model: &str,
        device: &str,
        chunk_tokens: i64,
        set_size: usize,
        header: i64,
    ) -> PyResult<Vec<usize>> {
        packing::choose_pass_groups(
            &question_tokens,
            &selectivities,
            rows,
            state_tokens,
            longest_state,
            self.model(model)?,
            self.device(device)?,
            chunk_tokens,
            set_size,
            header,
        )
        .map_err(PyValueError::new_err)
    }

    /// The label trie helpers: the read nodes, the fed nodes, and the
    /// decode round tokens, as sorted lists.
    #[staticmethod]
    fn label_nodes(labels: Vec<Vec<i64>>, starts: Vec<Vec<bool>>) -> PyResult<LabelNodes> {
        let read = labels::read_nodes(&labels, &starts).map_err(PyValueError::new_err)?;
        let fed = labels::fed_nodes(&read);
        let rounds = labels::decode_round_tokens(&labels).map_err(PyValueError::new_err)?;
        Ok((
            read.into_iter().collect(),
            fed.into_iter().collect(),
            rounds,
        ))
    }

    /// `classify::estimate`: seconds, passes, the work's
    /// fields, suffix tokens, and rounds.
    #[allow(clippy::too_many_arguments)]
    fn classify_estimate<'py>(
        &self,
        py: Python<'py>,
        scoring: &str,
        expected_rows: f64,
        head_tokens: i64,
        frame_tokens: i64,
        labels: Vec<Vec<i64>>,
        lengths: Vec<f64>,
        shared: Vec<f64>,
        chunk: i64,
        model: &str,
        device: &str,
        resident: bool,
        draws: i64,
        starts: Vec<Vec<bool>>,
        gpus: i64,
    ) -> PyResult<(f64, f64, Bound<'py, PyDict>, f64, usize)> {
        let inputs = ClassifyInputs {
            head_tokens,
            frame_tokens,
            expected_rows,
            lengths: &lengths,
            shared: &shared,
            chunk,
            model: self.model(model)?,
            device: self.device(device)?,
            resident,
            gpus,
        };
        let cost = classify::estimate(&inputs, scoring, &labels, draws, &starts)
            .map_err(PyValueError::new_err)?;
        Ok((
            cost.seconds,
            cost.passes,
            work_to(py, &cost.work)?,
            cost.suffix_tokens,
            cost.rounds,
        ))
    }

    /// `extract::estimate`: seconds, passes, the work's
    /// fields, suffix tokens, and rows.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (expected_rows, lengths, shared, head_tokens, tail_tokens, chunk, model, device, resident, max_answer_tokens, draft_tokens, draft_policy, round_cost_per_token, gpus, answer_tokens))]
    fn extract_estimate<'py>(
        &self,
        py: Python<'py>,
        expected_rows: f64,
        lengths: Vec<f64>,
        shared: Vec<f64>,
        head_tokens: i64,
        tail_tokens: i64,
        chunk: i64,
        model: &str,
        device: &str,
        resident: bool,
        max_answer_tokens: i64,
        draft_tokens: i64,
        draft_policy: String,
        round_cost_per_token: f64,
        gpus: i64,
        answer_tokens: i64,
    ) -> PyResult<(f64, f64, Bound<'py, PyDict>, f64, f64)> {
        let inputs = ExtractInputs {
            head_tokens,
            tail_tokens,
            window: 0,
            resident,
            max_answer_tokens,
            draft_tokens,
            draft_policy,
            round_cost_per_token,
            answer_tokens,
            recurrent: false,
        };
        let cost = extract::estimate(
            expected_rows,
            &lengths,
            &shared,
            &inputs,
            chunk,
            self.model(model)?,
            self.device(device)?,
            gpus,
        )
        .map_err(PyValueError::new_err)?;
        Ok((
            cost.seconds,
            cost.passes,
            work_to(py, &cost.work)?,
            cost.suffix_tokens,
            cost.rows,
        ))
    }

    /// `extract::document_work`: each document's work fields, fed
    /// tokens, and readout rows.
    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[pyo3(signature = (lengths, shared, head_tokens, tail_tokens, window, resident, max_answer_tokens, draft_tokens, draft_policy, round_cost_per_token, answer_tokens, recurrent))]
    fn extract_document_work<'py>(
        py: Python<'py>,
        lengths: Vec<f64>,
        shared: Vec<f64>,
        head_tokens: i64,
        tail_tokens: i64,
        window: i64,
        resident: bool,
        max_answer_tokens: i64,
        draft_tokens: i64,
        draft_policy: String,
        round_cost_per_token: f64,
        answer_tokens: i64,
        recurrent: bool,
    ) -> PyResult<Vec<(Bound<'py, PyDict>, f64, f64)>> {
        let inputs = ExtractInputs {
            head_tokens,
            tail_tokens,
            window,
            resident,
            max_answer_tokens,
            draft_tokens,
            draft_policy,
            round_cost_per_token,
            answer_tokens,
            recurrent,
        };
        extract::document_work(&lengths, &shared, &inputs)
            .map_err(PyValueError::new_err)?
            .iter()
            .map(|(work, fed, rows)| Ok((work_to(py, work)?, *fed, *rows)))
            .collect()
    }

    /// `extract::table_work`: the table's work fields, fed tokens, and
    /// readout rows.
    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[pyo3(signature = (lengths, shared, head_tokens, tail_tokens, window, resident, max_answer_tokens, draft_tokens, draft_policy, round_cost_per_token, answer_tokens, recurrent))]
    fn extract_table_work<'py>(
        py: Python<'py>,
        lengths: Vec<f64>,
        shared: Vec<f64>,
        head_tokens: i64,
        tail_tokens: i64,
        window: i64,
        resident: bool,
        max_answer_tokens: i64,
        draft_tokens: i64,
        draft_policy: String,
        round_cost_per_token: f64,
        answer_tokens: i64,
        recurrent: bool,
    ) -> PyResult<(Bound<'py, PyDict>, f64, f64)> {
        let inputs = ExtractInputs {
            head_tokens,
            tail_tokens,
            window,
            resident,
            max_answer_tokens,
            draft_tokens,
            draft_policy,
            round_cost_per_token,
            answer_tokens,
            recurrent,
        };
        let (work, fed, rows) =
            extract::table_work(&lengths, &shared, &inputs).map_err(PyValueError::new_err)?;
        Ok((work_to(py, &work)?, fed, rows))
    }

    /// `extract::survival`: the chance an answer of mean `answer_tokens`
    /// has more than `tokens` tokens after its first.
    #[staticmethod]
    fn extract_survival(answer_tokens: i64, tokens: f64) -> f64 {
        extract::survival(answer_tokens, tokens)
    }

    /// `extract::expected_rounds` with `draft_tokens` per round after
    /// the chosen token.
    #[staticmethod]
    fn extract_rounds(answer_tokens: i64, max_answer_tokens: i64, draft_tokens: i64) -> f64 {
        extract::expected_rounds(answer_tokens, max_answer_tokens, 1 + draft_tokens)
    }

    /// `float(np.sum(values))`, as the Rust port rounds it.
    #[staticmethod]
    fn numpy_sum(values: Vec<f64>) -> f64 {
        numpy_sum(&values)
    }
}
