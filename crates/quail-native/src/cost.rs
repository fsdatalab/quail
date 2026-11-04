//! The Rust cost model, exposed to Python so tests can compare it with
//! Python's.

use std::collections::BTreeMap;
use std::path::Path;

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use quail_cost::classify::{self, ClassifyInputs};
use quail_cost::decode::DecodeCost;
use quail_cost::draft_policy::DraftPolicy;
use quail_cost::labels;
use quail_cost::quantity::{Quantity, numpy_sum};
use quail_cost::sae_features::SaeFeaturesCost;
use quail_cost::score::ScoreCost;
use quail_cost::specs::{DeviceSpec, ModelSpec, read_devices, read_models};
use quail_cost::statistics::AliasStats;
use quail_cost::work::{self, Work};
use quail_cost::{
    answers, budgets, dense_decoder_cost, draft_policy, filters, packing, retention, sol,
};

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

/// An `AliasStats` from its fields: count, total, squared, maximum,
/// window, and short_counts.
fn alias_stats(fields: &Bound<'_, PyDict>) -> PyResult<AliasStats> {
    let get = |name: &str| -> PyResult<Bound<'_, PyAny>> {
        fields
            .get_item(name)?
            .ok_or_else(|| PyKeyError::new_err(name.to_string()))
    };
    Ok(AliasStats {
        count: get("count")?.extract()?,
        total: get("total")?.extract()?,
        squared: get("squared")?.extract()?,
        maximum: get("maximum")?.extract()?,
        window: get("window")?.extract()?,
        short_counts: get("short_counts")?
            .extract::<Vec<Vec<i64>>>()?
            .into_iter()
            .map(|pair| (pair[0], pair[1]))
            .collect(),
    })
}

/// A scaled `AliasStats`: count, total, squared, and short_counts.
type ScaledStats = (i64, i64, i128, Vec<(i64, i64)>);
/// An `AliasStats`'s mean, variance, window pairs, window reads, and
/// its scaling.
type StatsValues = (f64, f64, f64, f64, ScaledStats);

/// Read nodes, fed nodes, and each label's decode round tokens.
type LabelNodes = (Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<Vec<i64>>);

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

    /// The budgets of one model on one device that take no chunk size.
    fn model_budgets<'py>(
        &self,
        py: Python<'py>,
        model: &str,
        device: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (model, device) = (self.model(model)?, self.device(device)?);
        let values = PyDict::new(py);
        values.set_item("chunk_budget", budgets::chunk_budget(model, device))?;
        values.set_item(
            "chunk_memory_bound",
            budgets::chunk_memory_bound(model, device),
        )?;
        values.set_item("kernel_index_cap", budgets::kernel_index_cap(model))?;
        values.set_item(
            "minimum_weight_gpus",
            budgets::minimum_weight_gpus(model, device),
        )?;
        values.set_item("graph_memory_bytes", budgets::graph_memory_bytes(model))?;
        values.set_item("graph_token_buckets", budgets::graph_token_buckets(model))?;
        values.set_item(
            "tree_attention_allowed",
            budgets::tree_attention_allowed(model),
        )?;
        values.set_item(
            "sliding_kv_fits",
            budgets::sliding_kv_fits(model, device, None),
        )?;
        values.set_item(
            "compute_knee",
            budgets::compute_knee(model, device).map_err(PyValueError::new_err)?,
        )?;
        Ok(values)
    }

    /// The budgets of one model on one device at one chunk size.
    fn chunk_budgets<'py>(
        &self,
        py: Python<'py>,
        model: &str,
        device: &str,
        chunk_tokens: i64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (model, device) = (self.model(model)?, self.device(device)?);
        let page_tokens = device.page_tokens;
        let values = PyDict::new(py);
        values.set_item(
            "arena_bytes",
            budgets::arena_bytes(model, device, chunk_tokens),
        )?;
        values.set_item(
            "checkpoint_memory_bytes",
            budgets::checkpoint_memory_bytes(model, chunk_tokens, page_tokens),
        )?;
        values.set_item(
            "chunk_memory_bytes",
            budgets::chunk_memory_bytes(model, chunk_tokens, page_tokens),
        )?;
        values.set_item(
            "copy_verification_workspace_bytes",
            budgets::copy_verification_workspace_bytes(model, chunk_tokens),
        )?;
        values.set_item(
            "checkpoint_pass_bytes",
            budgets::checkpoint_pass_bytes(model, budgets::CHECKPOINT_PASS_QUESTIONS, chunk_tokens),
        )?;
        values.set_item(
            "hybrid_pool_pages",
            budgets::hybrid_pool_pages(model, device, chunk_tokens),
        )?;
        values.set_item(
            "checkpoint_pages",
            budgets::checkpoint_pages(model, page_tokens),
        )?;
        Ok(values)
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

    /// `cheaper_attention_path`.
    fn cheaper_attention_path(
        &self,
        model: &str,
        device: &str,
        readers: f64,
        reader_rows: f64,
        node_tokens: f64,
    ) -> PyResult<&'static str> {
        Ok(budgets::cheaper_attention_path(
            self.model(model)?,
            self.device(device)?,
            readers,
            reader_rows,
            node_tokens,
        ))
    }

    /// `parallel_copies`.
    #[staticmethod]
    fn parallel_copies(gpus: i64, num_documents: f64) -> f64 {
        budgets::parallel_copies(gpus, num_documents)
    }

    /// `prior_survival`.
    #[staticmethod]
    fn prior_survival(expected_tokens: f64, limit: usize) -> Vec<f64> {
        answers::prior_survival(expected_tokens, limit)
    }

    /// `answer_survival` with the geometric prior.
    #[staticmethod]
    fn answer_survival(
        lengths: Vec<i64>,
        written: Vec<i64>,
        expected_tokens: f64,
        limit: usize,
    ) -> Vec<f64> {
        answers::answer_survival(&lengths, &written, expected_tokens, limit, None)
    }

    /// `remaining_tokens` of a survival curve.
    #[staticmethod]
    fn remaining_tokens(survival: Vec<f64>) -> Vec<f64> {
        answers::remaining_tokens(&survival)
    }

    /// `live_buckets` of one count of live positions.
    #[staticmethod]
    fn live_bucket(num_streams: i64) -> usize {
        draft_policy::live_bucket(num_streams)
    }

    /// `source_odds` of one count of live positions.
    #[staticmethod]
    fn source_odds(num_streams: i64, trials: Vec<i64>, accepts: Vec<i64>) -> f64 {
        draft_policy::source_odds(num_streams, &trials, &accepts)
    }

    /// `draft_lengths`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn draft_lengths(
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

    /// `DraftPolicy(cap, max_answer_tokens, expected).curve()`.
    #[staticmethod]
    fn draft_policy_curve(cap: usize, max_answer_tokens: usize, expected: f64) -> Vec<f64> {
        DraftPolicy::new(cap, max_answer_tokens, expected).curve()
    }

    /// `retention.coefficients`, as the dict Python returns.
    fn coefficients<'py>(
        &self,
        py: Python<'py>,
        model: &str,
        device: &str,
        peak: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let values = retention::coefficients(self.model(model)?, self.device(device)?, peak);
        let config = PyDict::new(py);
        config.set_item("linear_seconds", values.linear_seconds)?;
        config.set_item("pair_seconds", values.pair_seconds)?;
        config.set_item("sliding_pair_seconds", values.sliding_pair_seconds)?;
        config.set_item("sliding_window", values.sliding_window)?;
        if let Some(pages) = values.checkpoint_pages {
            config.set_item("checkpoint_pages", pages)?;
        }
        Ok(config)
    }

    /// `RetentionPolicy(**coefficients(model, device)).prefill_seconds`.
    fn prefill_seconds(&self, model: &str, device: &str, tokens: f64) -> PyResult<f64> {
        Ok(
            retention::coefficients(self.model(model)?, self.device(device)?, false)
                .prefill_seconds(tokens),
        )
    }

    /// `retention_pages`.
    #[staticmethod]
    fn retention_pages(arena_tokens: i64, chunk_tokens: i64, page_tokens: i64) -> i64 {
        retention::retention_pages(arena_tokens, chunk_tokens, page_tokens)
    }

    /// `DecodeCost.from_specs(model, device)`'s weight, KV, and state seconds.
    fn decode_cost(&self, model: &str, device: &str) -> PyResult<(f64, f64, f64, f64)> {
        let cost = DecodeCost::from_specs(self.model(model)?, self.device(device)?);
        Ok((
            cost.weight_seconds,
            cost.kv_seconds,
            cost.sliding_kv_seconds,
            cost.state_seconds,
        ))
    }

    /// `DecodeCost.from_specs(model, device).lower_bound_seconds(work, passes)`.
    fn decode_lower_bound_seconds(
        &self,
        model: &str,
        device: &str,
        work: &Bound<'_, PyDict>,
        passes: f64,
    ) -> PyResult<f64> {
        let cost = DecodeCost::from_specs(self.model(model)?, self.device(device)?);
        Ok(cost.lower_bound_seconds(&work_from(work)?, passes))
    }

    /// An `AliasStats`'s mean, variance, window pairs and reads at an
    /// offset, and the fields of its scaling by a fraction.
    #[staticmethod]
    fn alias_stats_values(
        fields: &Bound<'_, PyDict>,
        offset: f64,
        fraction: f64,
    ) -> PyResult<StatsValues> {
        let stats = alias_stats(fields)?;
        let (pairs, reads) = if stats.window != 0 {
            (stats.window_pairs(offset), stats.window_reads(offset))
        } else {
            (0.0, 0.0)
        };
        let scaled = stats.scale(fraction);
        Ok((
            stats.mean(),
            stats.variance(),
            pairs,
            reads,
            (
                scaled.count,
                scaled.total,
                scaled.squared,
                scaled.short_counts,
            ),
        ))
    }

    /// `filter_cost`.
    #[allow(clippy::too_many_arguments)]
    fn filter_cost(
        &self,
        question_tokens: i64,
        mean_tokens: f64,
        model: &str,
        device: &str,
        chunk_tokens: i64,
        first: bool,
    ) -> PyResult<f64> {
        Ok(filters::filter_cost(
            question_tokens,
            mean_tokens,
            self.model(model)?,
            self.device(device)?,
            chunk_tokens,
            first,
        ))
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

    /// `ScoreCost(...).estimate(expected_rows, prefix_groups=...)`: the
    /// work's fields and the seconds.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (token_lengths, summaries, mean_tokens, model, device, chunk_tokens, capacity, gpus, draws, expected_rows, prefix_groups=None))]
    fn score_estimate<'py>(
        &self,
        py: Python<'py>,
        token_lengths: Vec<i64>,
        summaries: Vec<Bound<'py, PyDict>>,
        mean_tokens: f64,
        model: &str,
        device: &str,
        chunk_tokens: i64,
        capacity: i64,
        gpus: i64,
        draws: i64,
        expected_rows: f64,
        prefix_groups: Option<f64>,
    ) -> PyResult<(Bound<'py, PyDict>, f64)> {
        let cost = ScoreCost {
            token_lengths,
            summaries: summaries
                .iter()
                .map(alias_stats)
                .collect::<PyResult<Vec<_>>>()?,
            mean_tokens,
            model: self.model(model)?.clone(),
            device: self.device(device)?.clone(),
            chunk_tokens,
            capacity,
            gpus,
            draws,
        };
        let (work, seconds) = cost.estimate(expected_rows, prefix_groups);
        Ok((work_to(py, &work)?, seconds))
    }

    /// `SaeFeaturesCost(...).estimate(expected_rows, resident)`.
    #[allow(clippy::too_many_arguments)]
    fn sae_features_estimate<'py>(
        &self,
        py: Python<'py>,
        token_lengths: Vec<i64>,
        summary: &Bound<'py, PyDict>,
        pooling: String,
        model: &str,
        device: &str,
        chunk_tokens: i64,
        gpus: i64,
        max_pooling: bool,
        expected_rows: f64,
        resident: bool,
    ) -> PyResult<(Bound<'py, PyDict>, f64)> {
        let cost = SaeFeaturesCost {
            token_lengths,
            summary: alias_stats(summary)?,
            pooling,
            model: self.model(model)?.clone(),
            device: self.device(device)?.clone(),
            chunk_tokens,
            gpus,
            max_pooling,
        };
        let (work, seconds) = cost.estimate(expected_rows, resident);
        Ok((work_to(py, &work)?, seconds))
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

    /// `quail.cost.classify.estimate`: seconds, passes, the work's
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

    /// `float(np.sum(values))`, as the Rust port rounds it.
    #[staticmethod]
    fn numpy_sum(values: Vec<f64>) -> f64 {
        numpy_sum(&values)
    }
}
