//! The cost model's budgets and run-time curves, as the GPU executor
//! reads them: the same functions the planner prices plans with.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use quail_cost::decode::DecodeCost;
use quail_cost::specs::{DeviceSpec, ModelSpec};
use quail_cost::{answers, budgets, draft_policy, retention};

/// One model's memory and time budgets, on one device when given.
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
    /// Read the model and device specs, as JSON without type tags.
    #[new]
    #[pyo3(signature = (model, device=None))]
    fn new(model: &str, device: Option<&str>) -> PyResult<Self> {
        let read = |error: serde_json::Error| PyValueError::new_err(error.to_string());
        Ok(Self {
            model: serde_json::from_str(model).map_err(read)?,
            device: device.map(serde_json::from_str).transpose().map_err(read)?,
        })
    }

    /// Bytes left for KV after weights, activations, and CUDA graphs.
    fn arena_bytes(&self, chunk_tokens: i64) -> PyResult<f64> {
        Ok(budgets::arena_bytes(
            &self.model,
            self.device()?,
            chunk_tokens,
        ))
    }

    /// Tokens of document KV that can be resident at once.
    #[pyo3(signature = (chunk_tokens=None, mean_document_tokens=None))]
    fn arena_tokens(
        &self,
        chunk_tokens: Option<i64>,
        mean_document_tokens: Option<f64>,
    ) -> PyResult<i64> {
        budgets::arena_tokens(
            &self.model,
            self.device()?,
            chunk_tokens,
            mean_document_tokens,
        )
        .map_err(PyValueError::new_err)
    }

    /// A hybrid model's arena KV pages.
    #[pyo3(signature = (chunk_tokens, mean_document_tokens=None))]
    fn hybrid_arena_pages(
        &self,
        chunk_tokens: i64,
        mean_document_tokens: Option<f64>,
    ) -> PyResult<i64> {
        budgets::hybrid_arena_pages(
            &self.model,
            self.device()?,
            chunk_tokens,
            mean_document_tokens,
        )
        .map_err(PyValueError::new_err)
    }

    /// Pages a hybrid model shares between retained KV and checkpoints.
    fn hybrid_pool_pages(&self, chunk_tokens: i64) -> PyResult<i64> {
        Ok(budgets::hybrid_pool_pages(
            &self.model,
            self.device()?,
            chunk_tokens,
        ))
    }

    /// "unified" or "tree" attention for the readers of one shared node.
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

    /// Whether the leftover arena holds one chunk's sliding KV.
    #[pyo3(signature = (chunk_tokens=None))]
    fn sliding_kv_fits(&self, chunk_tokens: Option<i64>) -> PyResult<bool> {
        Ok(budgets::sliding_kv_fits(
            &self.model,
            self.device()?,
            chunk_tokens,
        ))
    }

    /// GPUs whose pooled memory holds the loaded weights.
    fn minimum_weight_gpus(&self) -> PyResult<i64> {
        Ok(budgets::minimum_weight_gpus(&self.model, self.device()?))
    }

    /// The largest chunk that fits its memory reservations, or zero.
    fn chunk_memory_bound(&self) -> PyResult<i64> {
        Ok(budgets::chunk_memory_bound(&self.model, self.device()?))
    }

    /// Resident weights, activation memory, and checkpoint reservations.
    fn chunk_memory_bytes(&self, chunk_tokens: i64, page_tokens: i64) -> f64 {
        budgets::chunk_memory_bytes(&self.model, chunk_tokens, page_tokens)
    }

    /// Workspace, execution KV, and one retained document's KV and state.
    fn checkpoint_memory_bytes(&self, chunk_tokens: i64, page_tokens: i64) -> f64 {
        budgets::checkpoint_memory_bytes(&self.model, chunk_tokens, page_tokens)
    }

    /// Temporary bytes a pass of `questions` starts from checkpoints.
    fn checkpoint_pass_bytes(&self, questions: i64, chunk_tokens: i64) -> i64 {
        budgets::checkpoint_pass_bytes(&self.model, questions, chunk_tokens)
    }

    /// KV pages of memory one document checkpoint takes.
    fn checkpoint_pages(&self, page_tokens: i64) -> f64 {
        budgets::checkpoint_pages(&self.model, page_tokens)
    }

    /// Readout workspace for copy verification, capped by one chunk.
    fn copy_verification_workspace_bytes(&self, chunk_tokens: i64) -> i64 {
        budgets::copy_verification_workspace_bytes(&self.model, chunk_tokens)
    }

    /// Tokens per forward pass.
    fn chunk_budget(&self) -> PyResult<i64> {
        Ok(budgets::chunk_budget(&self.model, self.device()?))
    }

    /// The most tokens one kernel call indexes.
    fn kernel_index_cap(&self) -> i64 {
        budgets::kernel_index_cap(&self.model)
    }

    /// Pages of the two KV pools: (every-token pool, sliding-layer pool).
    ///
    /// Raises ValueError when the pool cannot hold a chunk's KV.
    #[pyo3(signature = (chunk_tokens=None, mean_document_tokens=None))]
    fn arena_pages(
        &self,
        chunk_tokens: Option<i64>,
        mean_document_tokens: Option<f64>,
    ) -> PyResult<(i64, i64)> {
        budgets::arena_pages(
            &self.model,
            self.device()?,
            chunk_tokens,
            mean_document_tokens,
        )
        .map_err(PyValueError::new_err)
    }

    /// Checkpoints the pool holds beside an arena of `kv_pages`.
    fn checkpoint_slots(&self, chunk_tokens: i64, kv_pages: i64) -> PyResult<i64> {
        Ok(budgets::checkpoint_slots(
            &self.model,
            self.device()?,
            chunk_tokens,
            kv_pages,
        ))
    }

    /// Tokens per pass past which a pass is compute bound.
    fn compute_knee(&self) -> PyResult<f64> {
        budgets::compute_knee(&self.model, self.device()?).map_err(PyValueError::new_err)
    }

    /// The pass sizes captured as CUDA graphs.
    fn graph_token_buckets(&self) -> Vec<i64> {
        budgets::graph_token_buckets(&self.model)
    }

    /// Memory the captured CUDA graphs hold.
    fn graph_memory_bytes(&self) -> i64 {
        budgets::graph_memory_bytes(&self.model)
    }

    /// Whether the model may use the tree attention path.
    fn tree_attention_allowed(&self) -> bool {
        budgets::tree_attention_allowed(&self.model)
    }

    /// Seconds per token and per attention pair a pass spends:
    /// `linear_seconds`, `pair_seconds`, `sliding_pair_seconds`,
    /// `sliding_window`, and `checkpoint_pages` when the model has them.
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

    /// Seconds a decode pass spends reading the weights, a token's full
    /// KV, a token's sliding KV, and one recurrent state.
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

/// KV pages kept free for two chunks of execution.
#[pyfunction]
pub fn execution_pages(chunk_tokens: i64, page_tokens: i64) -> i64 {
    budgets::execution_pages(chunk_tokens, page_tokens)
}

/// Pages that hold `tokens` tokens.
#[pyfunction]
pub fn pages(tokens: i64, page_tokens: i64) -> i64 {
    budgets::pages(tokens, page_tokens)
}

/// Model copies worth running for `num_documents` documents on `gpus` GPUs.
#[pyfunction]
pub fn parallel_copies(gpus: i64, num_documents: f64) -> f64 {
    budgets::parallel_copies(gpus, num_documents)
}

/// KV pages a retained document may hold beside a chunk's execution.
#[pyfunction]
pub fn retention_pages(arena_tokens: i64, chunk_tokens: i64, page_tokens: i64) -> i64 {
    retention::retention_pages(arena_tokens, chunk_tokens, page_tokens)
}

/// The prior survival curve of an answer expected to run `expected_tokens`.
#[pyfunction]
pub fn prior_survival(expected_tokens: f64, limit: usize) -> Vec<f64> {
    answers::prior_survival(expected_tokens, limit)
}

/// Kaplan-Meier survival curves, one per row, starting from each prior.
#[pyfunction]
pub fn survival_rows(
    counts: Vec<Vec<i64>>,
    events: Vec<Vec<i64>>,
    prior: Vec<Vec<f64>>,
    limits: Vec<usize>,
) -> Vec<Vec<f64>> {
    answers::survival_rows(&counts, &events, &prior, &limits)
}

/// The survival curve of one question's answers so far.
#[pyfunction]
#[pyo3(signature = (lengths, written, expected_tokens, limit, prior=None))]
pub fn answer_survival(
    lengths: Vec<i64>,
    written: Vec<i64>,
    expected_tokens: f64,
    limit: usize,
    prior: Option<Vec<f64>>,
) -> Vec<f64> {
    answers::answer_survival(&lengths, &written, expected_tokens, limit, prior)
}

/// The expected tokens still to write after each written length.
#[pyfunction]
pub fn remaining_tokens(survival: Vec<f64>) -> Vec<f64> {
    answers::remaining_tokens(&survival)
}

/// The live-stream bucket of one count of live positions.
#[pyfunction]
pub fn live_bucket(num_streams: i64) -> usize {
    draft_policy::live_bucket(num_streams)
}

/// The odds a copy source is right, from the trials and accepts of its bucket.
#[pyfunction]
pub fn source_odds(num_streams: i64, trials: Vec<i64>, accepts: Vec<i64>) -> f64 {
    draft_policy::source_odds(num_streams, &trials, &accepts)
}

/// Each answer's copy draft length.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn draft_lengths(
    cap: usize,
    limit: usize,
    survival: Vec<f64>,
    written: Vec<i64>,
    num_streams: Vec<i64>,
    rho: Vec<f64>,
    trials: Vec<i64>,
    accepts: Vec<i64>,
) -> Vec<usize> {
    draft_policy::draft_lengths(
        cap,
        limit,
        &survival,
        &written,
        &num_streams,
        &rho,
        &trials,
        &accepts,
    )
}

/// Add the budget functions and constants to the native module.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Budgets>()?;
    for function in [
        wrap_pyfunction!(execution_pages, module)?,
        wrap_pyfunction!(pages, module)?,
        wrap_pyfunction!(parallel_copies, module)?,
        wrap_pyfunction!(retention_pages, module)?,
        wrap_pyfunction!(prior_survival, module)?,
        wrap_pyfunction!(survival_rows, module)?,
        wrap_pyfunction!(answer_survival, module)?,
        wrap_pyfunction!(remaining_tokens, module)?,
        wrap_pyfunction!(live_bucket, module)?,
        wrap_pyfunction!(source_odds, module)?,
        wrap_pyfunction!(draft_lengths, module)?,
    ] {
        module.add_function(function)?;
    }
    module.add("READOUT_BLOCK_ROWS", budgets::READOUT_BLOCK_ROWS)?;
    module.add("PAGE_TOKENS", budgets::PAGE_TOKENS)?;
    module.add("POOL_FRACTION", budgets::POOL_FRACTION)?;
    module.add("ACT_RESERVE_CHUNKS", budgets::ACT_RESERVE_CHUNKS)?;
    module.add(
        "CHECKPOINT_PASS_QUESTIONS",
        budgets::CHECKPOINT_PASS_QUESTIONS,
    )?;
    module.add("GRAPH_TOKEN_BUCKETS", budgets::graph_token_buckets_all())?;
    module.add("LIVE_BUCKETS", draft_policy::LIVE_BUCKETS.to_vec())?;
    Ok(())
}
