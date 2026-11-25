use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use quail_planner::budgets;
use quail_planner::cost::decode::DecodeCost;
use quail_planner::cost::{answers, draft_policy, retention};
use quail_planner::specs::{DeviceSpec, ModelSpec};

#[pyclass(name = "Budgets", module = "_quail_native", frozen)]
pub struct Budgets {
    model: ModelSpec,
    device: Option<DeviceSpec>,
}

impl Budgets {
    fn device(&self) -> PyResult<&DeviceSpec> {
        self.device
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("this budget needs a device"))
    }
}

#[pymethods]
impl Budgets {
    #[new]
    #[pyo3(signature = (model, device=None))]
    fn new(model: &str, device: Option<&str>) -> PyResult<Self> {
        let read = |error: serde_json::Error| PyValueError::new_err(error.to_string());
        Ok(Self {
            model: serde_json::from_str(model).map_err(read)?,
            device: device.map(serde_json::from_str).transpose().map_err(read)?,
        })
    }

    #[pyo3(signature = (chunk_tokens, writes_answers=true))]
    fn arena_bytes(&self, chunk_tokens: i64, writes_answers: bool) -> PyResult<f64> {
        Ok(budgets::arena_bytes(
            &self.model,
            self.device()?,
            chunk_tokens,
            writes_answers,
        ))
    }

    #[pyo3(signature = (chunk_tokens=None, mean_document_tokens=None, writes_answers=true))]
    fn arena_tokens(
        &self,
        chunk_tokens: Option<i64>,
        mean_document_tokens: Option<f64>,
        writes_answers: bool,
    ) -> PyResult<i64> {
        budgets::arena_tokens(
            &self.model,
            self.device()?,
            chunk_tokens,
            mean_document_tokens,
            writes_answers,
        )
        .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (chunk_tokens, mean_document_tokens=None, writes_answers=true))]
    fn hybrid_arena_pages(
        &self,
        chunk_tokens: i64,
        mean_document_tokens: Option<f64>,
        writes_answers: bool,
    ) -> PyResult<i64> {
        budgets::hybrid_arena_pages(
            &self.model,
            self.device()?,
            chunk_tokens,
            mean_document_tokens,
            writes_answers,
        )
        .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (chunk_tokens, writes_answers=true))]
    fn hybrid_pool_pages(&self, chunk_tokens: i64, writes_answers: bool) -> PyResult<i64> {
        Ok(budgets::hybrid_pool_pages(
            &self.model,
            self.device()?,
            chunk_tokens,
            writes_answers,
        ))
    }

    fn cheaper_attention_path(
        &self,
        readers: f64,
        reader_rows: f64,
        node_tokens: f64,
    ) -> PyResult<&'static str> {
        Ok(budgets::cheaper_attention_path(
            &self.model,
            self.device()?,
            readers,
            reader_rows,
            node_tokens,
        ))
    }

    #[pyo3(signature = (chunk_tokens=None))]
    fn sliding_kv_fits(&self, chunk_tokens: Option<i64>) -> PyResult<bool> {
        Ok(budgets::sliding_kv_fits(
            &self.model,
            self.device()?,
            chunk_tokens,
        ))
    }

    fn minimum_weight_gpus(&self) -> PyResult<i64> {
        Ok(budgets::minimum_weight_gpus(&self.model, self.device()?))
    }

    fn chunk_memory_bound(&self) -> PyResult<i64> {
        Ok(budgets::chunk_memory_bound(&self.model, self.device()?))
    }

    fn chunk_memory_bytes(&self, chunk_tokens: i64, page_tokens: i64) -> f64 {
        budgets::chunk_memory_bytes(&self.model, chunk_tokens, page_tokens)
    }

    fn checkpoint_memory_bytes(&self, chunk_tokens: i64, page_tokens: i64) -> f64 {
        budgets::checkpoint_memory_bytes(&self.model, chunk_tokens, page_tokens)
    }

    fn checkpoint_pass_bytes(&self, questions: i64, chunk_tokens: i64) -> i64 {
        budgets::checkpoint_pass_bytes(&self.model, questions, chunk_tokens)
    }

    fn checkpoint_pages(&self, page_tokens: i64) -> f64 {
        budgets::checkpoint_pages(&self.model, page_tokens)
    }

    fn copy_verification_workspace_bytes(&self, chunk_tokens: i64) -> i64 {
        budgets::copy_verification_workspace_bytes(&self.model, chunk_tokens)
    }

    fn chunk_budget(&self) -> PyResult<i64> {
        Ok(budgets::chunk_budget(&self.model, self.device()?))
    }

    fn kernel_index_cap(&self) -> i64 {
        budgets::kernel_index_cap(&self.model)
    }

    #[pyo3(signature = (chunk_tokens=None, mean_document_tokens=None, writes_answers=true))]
    fn arena_pages(
        &self,
        chunk_tokens: Option<i64>,
        mean_document_tokens: Option<f64>,
        writes_answers: bool,
    ) -> PyResult<(i64, i64)> {
        budgets::arena_pages(
            &self.model,
            self.device()?,
            chunk_tokens,
            mean_document_tokens,
            writes_answers,
        )
        .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (chunk_tokens, kv_pages, writes_answers=true))]
    fn checkpoint_slots(
        &self,
        chunk_tokens: i64,
        kv_pages: i64,
        writes_answers: bool,
    ) -> PyResult<i64> {
        Ok(budgets::checkpoint_slots(
            &self.model,
            self.device()?,
            chunk_tokens,
            kv_pages,
            writes_answers,
        ))
    }

    fn compute_knee(&self) -> PyResult<f64> {
        budgets::compute_knee(&self.model, self.device()?).map_err(PyValueError::new_err)
    }

    fn graph_token_buckets(&self) -> Vec<i64> {
        budgets::graph_token_buckets(&self.model)
    }

    fn graph_memory_bytes(&self) -> i64 {
        budgets::graph_memory_bytes(&self.model)
    }

    fn tree_attention_allowed(&self) -> bool {
        budgets::tree_attention_allowed(&self.model)
    }

    #[pyo3(signature = (peak=false))]
    fn coefficients<'py>(&self, py: Python<'py>, peak: bool) -> PyResult<Bound<'py, PyDict>> {
        let values = retention::coefficients(&self.model, self.device()?, peak);
        let found = PyDict::new(py);
        found.set_item("linear_seconds", values.linear_seconds)?;
        found.set_item("pair_seconds", values.pair_seconds)?;
        found.set_item("sliding_pair_seconds", values.sliding_pair_seconds)?;
        found.set_item("sliding_window", values.sliding_window)?;
        if let Some(pages) = values.checkpoint_pages {
            found.set_item("checkpoint_pages", pages)?;
        }
        Ok(found)
    }

    fn decode_seconds(&self) -> PyResult<(f64, f64, f64, f64)> {
        let cost = DecodeCost::from_specs(&self.model, self.device()?);
        Ok((
            cost.weight_seconds,
            cost.kv_seconds,
            cost.sliding_kv_seconds,
            cost.state_seconds,
        ))
    }
}

#[pyfunction]
pub fn execution_pages(chunk_tokens: i64, page_tokens: i64) -> i64 {
    budgets::execution_pages(chunk_tokens, page_tokens)
}

#[pyfunction]
pub fn pages(tokens: i64, page_tokens: i64) -> i64 {
    budgets::pages(tokens, page_tokens)
}

#[pyfunction]
pub fn parallel_copies(gpus: i64, num_documents: f64) -> f64 {
    budgets::parallel_copies(gpus, num_documents)
}

#[pyfunction]
pub fn retention_pages(arena_tokens: i64, chunk_tokens: i64, page_tokens: i64) -> i64 {
    retention::retention_pages(arena_tokens, chunk_tokens, page_tokens)
}

fn le_bytes<'py, T: Copy>(
    py: Python<'py>,
    values: &[T],
    write: fn(T) -> [u8; 8],
) -> Bound<'py, PyBytes> {
    let bytes: Vec<u8> = values.iter().flat_map(|value| write(*value)).collect();
    PyBytes::new(py, &bytes)
}

fn rows<T: Clone>(values: Vec<T>, width: usize) -> Vec<Vec<T>> {
    if width == 0 {
        return Vec::new();
    }
    values.chunks(width).map(<[T]>::to_vec).collect()
}

#[pyfunction]
pub fn prior_survival(py: Python<'_>, expected_tokens: f64, limit: usize) -> Bound<'_, PyBytes> {
    le_bytes(
        py,
        &answers::prior_survival(expected_tokens, limit),
        f64::to_le_bytes,
    )
}

#[pyfunction]
pub fn survival_rows<'py>(
    py: Python<'py>,
    counts: PyBuffer<i64>,
    events: PyBuffer<i64>,
    prior: PyBuffer<f64>,
    limits: PyBuffer<i64>,
    width: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let limits: Vec<usize> = limits
        .to_vec(py)?
        .into_iter()
        .map(|limit| limit.max(0) as usize)
        .collect();
    let curves = answers::survival_rows(
        &rows(counts.to_vec(py)?, width),
        &rows(events.to_vec(py)?, width),
        &rows(prior.to_vec(py)?, width),
        &limits,
    );
    Ok(le_bytes(py, &curves.concat(), f64::to_le_bytes))
}

#[pyfunction]
pub fn remaining_tokens<'py>(
    py: Python<'py>,
    survival: PyBuffer<f64>,
) -> PyResult<Bound<'py, PyBytes>> {
    let remaining = answers::remaining_tokens(&survival.to_vec(py)?);
    Ok(le_bytes(py, &remaining, f64::to_le_bytes))
}

#[pyfunction]
pub fn live_buckets<'py>(
    py: Python<'py>,
    num_streams: PyBuffer<i64>,
) -> PyResult<Bound<'py, PyBytes>> {
    let buckets: Vec<i64> = num_streams
        .to_vec(py)?
        .into_iter()
        .map(|count| draft_policy::live_bucket(count) as i64)
        .collect();
    Ok(le_bytes(py, &buckets, i64::to_le_bytes))
}

#[pyfunction]
pub fn source_odds<'py>(
    py: Python<'py>,
    num_streams: PyBuffer<i64>,
    trials: PyBuffer<i64>,
    accepts: PyBuffer<i64>,
) -> PyResult<Bound<'py, PyBytes>> {
    let (trials, accepts) = (trials.to_vec(py)?, accepts.to_vec(py)?);
    let odds: Vec<f64> = num_streams
        .to_vec(py)?
        .into_iter()
        .map(|count| draft_policy::source_odds(count, &trials, &accepts))
        .collect();
    Ok(le_bytes(py, &odds, f64::to_le_bytes))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn draft_gains<'py>(
    py: Python<'py>,
    cap: usize,
    limit: usize,
    survival: PyBuffer<f64>,
    written: PyBuffer<i64>,
    num_streams: PyBuffer<i64>,
    trials: PyBuffer<i64>,
    accepts: PyBuffer<i64>,
) -> PyResult<Bound<'py, PyBytes>> {
    let gains = draft_policy::draft_gains(
        cap,
        limit,
        &survival.to_vec(py)?,
        &written.to_vec(py)?,
        &num_streams.to_vec(py)?,
        &trials.to_vec(py)?,
        &accepts.to_vec(py)?,
    );
    Ok(le_bytes(py, &gains.concat(), f64::to_le_bytes))
}

#[pyfunction]
pub fn fit_drafts<'py>(
    py: Python<'py>,
    gains: PyBuffer<f64>,
    cap: usize,
    drafts: PyBuffer<i64>,
    room: i64,
) -> PyResult<Bound<'py, PyBytes>> {
    let fitted = draft_policy::fit_drafts(&rows(gains.to_vec(py)?, cap), &drafts.to_vec(py)?, room);
    Ok(le_bytes(py, &fitted, i64::to_le_bytes))
}

#[pyfunction]
pub fn kv_pool_cap_tokens() -> Option<i64> {
    budgets::kv_pool_cap_tokens()
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn draft_lengths<'py>(
    py: Python<'py>,
    cap: usize,
    limit: usize,
    survival: PyBuffer<f64>,
    written: PyBuffer<i64>,
    num_streams: PyBuffer<i64>,
    rho: PyBuffer<f64>,
    trials: PyBuffer<i64>,
    accepts: PyBuffer<i64>,
) -> PyResult<Bound<'py, PyBytes>> {
    let lengths: Vec<i64> = draft_policy::draft_lengths(
        cap,
        limit,
        &survival.to_vec(py)?,
        &written.to_vec(py)?,
        &num_streams.to_vec(py)?,
        &rho.to_vec(py)?,
        &trials.to_vec(py)?,
        &accepts.to_vec(py)?,
    )
    .into_iter()
    .map(|length| length as i64)
    .collect();
    Ok(le_bytes(py, &lengths, i64::to_le_bytes))
}

#[pyfunction]
pub fn model_properties(model: &str) -> PyResult<String> {
    let model: ModelSpec =
        serde_json::from_str(model).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(serde_json::json!({
        "reuses_kv": model.reuses_kv(),
        "attention_layers": model.attention_layers(),
        "gdn_layers": model.gdn_layers(),
        "short_conv_layers": model.short_conv_layers(),
        "state_layers": model.state_layers(),
        "conv_state_channels": model.conv_state_channels(),
        "conv_channels": model.conv_channels(),
        "gdn_state_bytes": model.gdn_state_bytes(),
        "conv_state_bytes": model.conv_state_bytes(),
        "checkpoint_bytes": model.checkpoint_bytes(),
        "kv_shapes": model.kv_shapes(),
        "widest_projection": model.widest_projection(),
        "kappa": model.kappa(),
        "kv_elements_per_token": model.kv_elements_per_token(),
        "kappa_sliding": model.kappa_sliding(),
        "kappa_full": model.kappa_full(),
        "W_mem": model.W_mem(),
        "head_mem_bytes": model.head_mem_bytes(),
        "W_resident": model.W_resident(),
        "act_per_token": model.act_per_token(),
        "intermediate": model.intermediate(),
    })
    .to_string())
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Budgets>()?;
    for function in [
        wrap_pyfunction!(model_properties, module)?,
        wrap_pyfunction!(execution_pages, module)?,
        wrap_pyfunction!(pages, module)?,
        wrap_pyfunction!(parallel_copies, module)?,
        wrap_pyfunction!(retention_pages, module)?,
        wrap_pyfunction!(prior_survival, module)?,
        wrap_pyfunction!(survival_rows, module)?,
        wrap_pyfunction!(remaining_tokens, module)?,
        wrap_pyfunction!(live_buckets, module)?,
        wrap_pyfunction!(source_odds, module)?,
        wrap_pyfunction!(draft_lengths, module)?,
        wrap_pyfunction!(draft_gains, module)?,
        wrap_pyfunction!(fit_drafts, module)?,
        wrap_pyfunction!(kv_pool_cap_tokens, module)?,
    ] {
        module.add_function(function)?;
    }
    module.add("READOUT_BLOCK_ROWS", budgets::READOUT_BLOCK_ROWS)?;
    module.add("PAGE_TOKENS", budgets::PAGE_TOKENS)?;
    module.add("POOL_FRACTION", budgets::POOL_FRACTION)?;
    module.add("KV_POOL_TOKENS_ENV", budgets::KV_POOL_TOKENS_ENV)?;
    module.add("ACT_RESERVE_CHUNKS", budgets::ACT_RESERVE_CHUNKS)?;
    module.add(
        "CHECKPOINT_PASS_QUESTIONS",
        budgets::CHECKPOINT_PASS_QUESTIONS,
    )?;
    module.add("GRAPH_TOKEN_BUCKETS", budgets::graph_token_buckets_all())?;
    module.add("LIVE_BUCKETS", draft_policy::LIVE_BUCKETS.to_vec())?;
    Ok(())
}
