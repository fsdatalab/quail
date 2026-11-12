//! The model region runner, and the tables whose rows a function
//! returns.
//!
//! An [`Executor`] plans each model region and runs it on Quail's GPU
//! workers ([`crate::workers`]). An executor made over a
//! [`RegionEngine`] calls the engine instead, for a backend that is not
//! Quail's own.

use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use async_trait::async_trait;
use datafusion::arrow::array::{ArrayRef, BooleanArray, RecordBatch};
use datafusion::arrow::compute::cast;
use datafusion::arrow::compute::{concat_batches, filter_record_batch};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{Constraints, DataFusionError, Result};
use datafusion::datasource::TableType;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::physical_plan::ExecutionPlan;
use std::collections::{BTreeMap, HashMap};

use quail_datafusion::{
    IdentifiedTable, InputRows, InputStream, ModelRegion, RegionBatches, RegionColumns, RegionHint,
    RegionRunner, SharedInput, collect_inputs, derive, quail_column_names,
};
use quail_planner::operators::Operators;
use quail_planner::prompts::Turn;
use tokio::sync::mpsc::UnboundedSender;

use crate::EngineConfig;
use crate::planning::{PlannedRegion, Planner, ScanInputs, Settings, TableRows, outcome_json};
use crate::workers::Workers;

/// A region for a [`RegionEngine`] to run.
pub struct EngineRegion<'a> {
    pub region: &'a ModelRegion,
    /// The rows of each input the region reads, by table name.
    pub inputs: &'a [InputRows],
    /// The region as Rust planned it, with what its scans read; None
    /// for a region Rust did not plan.
    pub planned: Option<PlannedRegion>,
    /// The region's operators and DataFusion plan, as [`region_json`]
    /// writes them, for a region Rust did not plan.
    pub payload: Option<String>,
    /// The most rows DataFusion reads.
    pub fetch: Option<usize>,
    /// The tests above the region, by which the engine may skip rows.
    pub hint: Option<InputHint>,
    /// Where the engine sends rows as they finish.
    pub sink: RegionSink,
}

/// Runs regions in place of Quail's GPU workers: a backend that is not
/// Quail's own, or a test's stand-in.
pub trait RegionEngine: Send + Sync + std::fmt::Debug {
    /// Run a region. The engine sends rows to the region's sink as they
    /// finish, or returns them, or both; each row once.
    ///
    /// # Errors
    ///
    /// The region fails.
    fn run(&self, region: EngineRegion<'_>) -> Result<Vec<RecordBatch>>;
}

/// What runs a region's model calls.
enum Runs {
    Engine(Arc<dyn RegionEngine>),
    /// Quail's GPU workers.
    Workers(Arc<Mutex<Workers>>),
}

impl std::fmt::Debug for Runs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Runs::Engine(_) => "Engine",
            Runs::Workers(_) => "Workers",
        })
    }
}

/// Plans each model region and runs it: on Quail's GPU workers, or by
/// calling a [`RegionEngine`] it was made over.
///
/// It is the [`RegionRunner`] a [`crate::Session`] runs model regions
/// on. To add Quail to a DataFusion session of your own, register
/// `QuailExtension::new(Arc::new(Executor::open(&config)?))`.
pub struct Executor {
    runs: Runs,
    /// The optimizer rules each model region runs, by name, in order.
    pub(crate) rules: Vec<String>,
    /// The model's chat-turn text and prompt layout, which each region's
    /// prompts are bound in.
    pub(crate) prompting: (Turn, String),
    /// While a query runs: the thread that called DataFusion, and where
    /// a worker sends it a region to run.
    caller: Mutex<Option<(ThreadId, UnboundedSender<RegionCall>)>>,
    /// Own the caller route for the whole query, including error cleanup.
    execution: Mutex<()>,
    /// The planner and configuration Rust plans each region with, then
    /// calls the engine's `execute`; None calls `run_region`.
    pub(crate) planning: Option<(Arc<Planner>, Settings)>,
    /// What each region of the running query ran on the workers.
    regions: Mutex<Vec<RegionRan>>,
    /// A plan's envelope to run in place of the planner's, for a query
    /// of one region.
    edited: Mutex<Option<serde_json::Value>>,
    /// Called once the workers are ready to run a region.
    ready: Mutex<Option<Ready>>,
}

/// Called once the workers have the model loaded for a region.
pub type Ready = Arc<dyn Fn() + Send + Sync>;

impl std::fmt::Debug for Executor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Executor")
            .field("runs", &self.runs)
            .field("rules", &self.rules)
            .finish_non_exhaustive()
    }
}

/// One region's run on the GPU workers.
#[derive(Debug)]
pub struct RegionRan {
    /// The planned region, as [`outcome_json`] writes it, with its
    /// logical plan, planning seconds, and region id.
    pub planned: String,
    /// What each scan read.
    pub scans: Vec<ScanInputs>,
    /// The rows of each input the region read, by table name.
    pub inputs: Vec<InputRows>,
    /// The region's rows, with Quail's names for its columns.
    pub rows: RecordBatch,
    /// What each worker ran: its counters and its nodes' outputs.
    pub ran: Vec<crate::workers::Ran>,
    /// Each worker's boot record.
    pub boots: Vec<serde_json::Value>,
    /// Seconds the region took on the workers.
    pub seconds: f64,
    /// Seconds from the planned region to its last row: the workers'
    /// boot and the run.
    pub total_seconds: f64,
}

impl Executor {
    /// Open Quail's executor for `config`: start a GPU worker per GPU
    /// and ask one what Quail plans the configuration's regions with.
    ///
    /// A worker is the command `QUAIL_WORKER` names, or `python -m
    /// quail.worker` with the Python of the virtual environment
    /// `VIRTUAL_ENV` names, or `python3`.
    ///
    /// # Errors
    ///
    /// A worker does not start, or Quail refuses the configuration.
    pub fn open(config: &EngineConfig) -> Result<Self> {
        let mut workers = Workers::bind(config.gpus)?;
        workers.spawn(&worker_command())?;
        workers.connect()?;
        Self::on_workers(Arc::new(Mutex::new(workers)), config)
    }

    /// An executor for `config` that runs model regions on `workers`,
    /// which are connected; it asks one what Quail plans with.
    ///
    /// # Errors
    ///
    /// A worker refuses the configuration, or what it describes does
    /// not read.
    pub fn on_workers(workers: Arc<Mutex<Workers>>, config: &EngineConfig) -> Result<Self> {
        let described = workers
            .lock()
            .expect("the workers lock")
            .describe(&serde_json::json!({
                "model": config.model, "device": config.device,
                "gpus": config.gpus, "backend": config.backend,
            }))?;
        Self::described(workers, &described, None)
    }

    /// An executor that runs model regions on `workers`, planning with
    /// what a worker described: the model's chat turn and prompt
    /// layout, the rules, the tokenizer file, and the planning settings.
    ///
    /// `planner` is a planner for the described tokenizer, which keeps
    /// the tables it tokenized for later queries; None makes a new one.
    ///
    /// # Errors
    ///
    /// The description does not read.
    pub fn described(
        workers: Arc<Mutex<Workers>>,
        described: &serde_json::Value,
        planner: Option<Arc<Planner>>,
    ) -> Result<Self> {
        let read = |error: serde_json::Error| DataFusionError::External(Box::new(error));
        let field = |name: &str| described[name].clone();
        let turn: (String, String) = serde_json::from_value(field("turn")).map_err(read)?;
        let layout: String = serde_json::from_value(field("layout")).map_err(read)?;
        let rules: Vec<String> = serde_json::from_value(field("rules")).map_err(read)?;
        let tokenizer: String = serde_json::from_value(field("tokenizer_file")).map_err(read)?;
        let settings: Settings = serde_json::from_value(field("settings")).map_err(read)?;
        let planner = match planner {
            Some(planner) => planner,
            None => Arc::new(
                Planner::new(std::path::Path::new(&tokenizer))
                    .map_err(quail_datafusion::external)?,
            ),
        };
        Ok(Self {
            runs: Runs::Workers(workers),
            rules,
            prompting: (turn, layout),
            caller: Mutex::new(None),
            execution: Mutex::new(()),
            planning: Some((planner, settings)),
            regions: Mutex::new(Vec::new()),
            edited: Mutex::new(None),
            ready: Mutex::new(None),
        })
    }

    /// An executor that runs model regions on `engine`.
    ///
    /// `prompting` is the model's chat-turn text and prompt layout,
    /// which each region's prompts are bound in. `rules` names the
    /// optimizer rules each model region runs, in order, or is None for
    /// the built-in rules. With `planning`, a planner and the
    /// configuration it plans for, Rust plans each region and hands the
    /// engine the plan; without it the engine plans.
    pub fn with_engine(
        engine: Arc<dyn RegionEngine>,
        prompting: (Turn, String),
        rules: Option<Vec<String>>,
        planning: Option<(Arc<Planner>, Settings)>,
    ) -> Self {
        Self {
            runs: Runs::Engine(engine),
            rules: rules.unwrap_or_else(quail_datafusion::default_rule_names),
            prompting,
            caller: Mutex::new(None),
            execution: Mutex::new(()),
            planning,
            regions: Mutex::new(Vec::new()),
            edited: Mutex::new(None),
            ready: Mutex::new(None),
        }
    }
}

/// Clears the route before releasing exclusive execution ownership.
/// Keeping this outside the query future also covers early errors and unwinding.
pub(crate) struct CallerGuard<'a> {
    executor: &'a Executor,
    _execution: std::sync::MutexGuard<'a, ()>,
}

impl Drop for CallerGuard<'_> {
    fn drop(&mut self) {
        *self
            .executor
            .caller
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = None;
    }
}

impl Executor {
    /// Run `envelope`, an edited plan's, in place of the plan the
    /// planner makes for the next region; None runs the planner's.
    pub fn edit_plan(&self, envelope: Option<serde_json::Value>) {
        *self.edited.lock().expect("the edited lock") = envelope;
    }

    /// Call `ready` each time the workers have the model loaded for a
    /// region, before the region runs; None calls nothing.
    pub fn when_ready(&self, ready: Option<Ready>) {
        *self.ready.lock().expect("the ready lock") = ready;
    }

    /// What the planner's explain reads of a planned region, as
    /// [`outcome_json`] writes the region: a
    /// `quail_planner::explain::ExplainRequest` as JSON.
    pub(crate) fn explain_request(
        &self,
        planned: &serde_json::Value,
        verbose: bool,
    ) -> Result<serde_json::Value> {
        use serde_json::{Value, json};
        let read = |error: serde_json::Error| DataFusionError::External(Box::new(error));
        let operators: Value =
            serde_json::from_str(planned["operators"].as_str().unwrap_or("null")).map_err(read)?;
        let mut request = json!({
            "logical": planned["logical"], "operators": operators, "verbose": verbose,
        });
        if let Some((_, settings)) = &self.planning {
            request["model_spec"] = settings.model.clone();
            request["usd_per_hour"] = settings.device["usd_per_hour"].clone();
            let rules = (self.rules.iter().map(|rule| (rule, "logical"))).chain(
                settings
                    .physical_rules
                    .iter()
                    .map(|rule| (rule, "physical")),
            );
            request["rules"] = json!(rules.collect::<Vec<_>>());
        }
        if let Some(refusal) = planned.get("refusal") {
            request["refusal"] = refusal.clone();
            return Ok(request);
        }
        let envelope = &planned["envelope"];
        let settings: Vec<(String, Value)> = envelope["settings"]
            .as_object()
            .map(|settings| settings.clone().into_iter().collect())
            .unwrap_or_default();
        let estimates: Vec<(String, Value)> = match &planned["estimates"] {
            Value::Object(estimates) => estimates.clone().into_iter().collect(),
            _ => Vec::new(),
        };
        request["physical"] = json!({
            "backend": envelope["backend"], "model": envelope["model"],
            "device": envelope["device"], "gpus": envelope["gpus"],
            "settings": settings, "shown_settings": settings,
            "graph": envelope["graph"], "estimates": estimates,
        });
        Ok(request)
    }

    /// What the planner's explain reads of a region that ran: its
    /// [`Executor::explain_request`] with the run's report and each
    /// node's measured metrics.
    ///
    /// # Errors
    ///
    /// The planned region or a worker's results do not read.
    pub(crate) fn analyzed_request(
        &self,
        region: &RegionRan,
        verbose: bool,
    ) -> Result<serde_json::Value> {
        use serde_json::{Value, json};
        let read = |error: serde_json::Error| DataFusionError::External(Box::new(error));
        let planned: Value = serde_json::from_str(&region.planned).map_err(read)?;
        let mut request = self.explain_request(&planned, verbose)?;
        let envelope = &planned["envelope"];
        let graph: quail_planner::physical::PhysicalGraph =
            serde_json::from_value(envelope["graph"].clone()).map_err(read)?;
        let usd_per_hour = (self.planning.as_ref())
            .and_then(|(_, settings)| settings.device["usd_per_hour"].as_f64());
        let (mut measured, _) = crate::report::region_report(
            &graph,
            &region.ran,
            region.seconds,
            usd_per_hour,
            envelope["gpus"].as_i64().unwrap_or(1),
            envelope["settings"]["gpu_timing"]
                .as_bool()
                .unwrap_or(false),
        )?;
        // the root's rows are the region's
        let root = graph.root.node_id.clone();
        measured["node_metrics"][&root]["output_rows"] = json!(region.rows.num_rows());
        let seconds = |boot: &Value| boot["boot_s"].as_f64().unwrap_or(0.0);
        let slowest =
            (region.boots.iter()).max_by(|one, two| seconds(one).total_cmp(&seconds(two)));
        if let Some(boot) = slowest {
            measured["boot_s"] = boot["boot_s"].clone();
            measured["boot_kind"] = boot["kind"].clone();
            measured["boot"] = boot.clone();
        }
        let report = crate::results::execution_report(
            envelope["backend"].as_str().unwrap_or_default(),
            planned["estimated_seconds"].as_f64().unwrap_or(0.0),
            envelope["settings"]["order_rule"].as_str(),
            measured.as_object().expect("a report"),
            region.total_seconds,
        );
        let metrics: serde_json::Map<String, Value> = (report["node_metrics"].as_object())
            .into_iter()
            .flatten()
            .map(|(node, fields)| {
                let pairs: Vec<Value> = (fields.as_object().into_iter().flatten())
                    .map(|(field, value)| json!([field, value]))
                    .collect();
                (node.clone(), Value::Array(pairs))
            })
            .collect();
        request["run"] = json!({"report": report, "metrics": metrics});
        Ok(request)
    }

    /// What each region of the last query ran on the workers, in the
    /// order the regions ran; the executor keeps none of it.
    pub fn take_regions(&self) -> Vec<RegionRan> {
        std::mem::take(&mut *self.regions.lock().expect("the regions lock"))
    }

    pub(crate) fn enter(&self, sender: UnboundedSender<RegionCall>) -> Result<CallerGuard<'_>> {
        let thread = std::thread::current().id();
        if self
            .caller
            .lock()
            .expect("the caller lock")
            .as_ref()
            .is_some_and(|(owner, _)| *owner == thread)
        {
            return Err(DataFusionError::Execution(
                "an executor cannot run a reentrant query".into(),
            ));
        }
        let execution = self.execution.lock().map_err(|_| {
            DataFusionError::Execution("the executor's previous query panicked".into())
        })?;
        *self.caller.lock().expect("the caller lock") = Some((thread, sender));
        self.regions.lock().expect("the regions lock").clear();
        Ok(CallerGuard {
            executor: self,
            _execution: execution,
        })
    }
}

/// The command that starts a GPU worker, before its socket and GPU
/// index.
fn worker_command() -> Vec<String> {
    if let Ok(command) = std::env::var("QUAIL_WORKER") {
        return command.split_whitespace().map(str::to_string).collect();
    }
    let python = match std::env::var_os("VIRTUAL_ENV") {
        Some(root) => std::path::PathBuf::from(root)
            .join("bin/python")
            .to_string_lossy()
            .into_owned(),
        None => "python3".to_string(),
    };
    vec![python, "-m".into(), "quail.worker".into()]
}

/// A region a DataFusion worker hands to the calling thread to run.
pub(crate) struct RegionCall {
    pub(crate) region: ModelRegion,
    pub(crate) inputs: Vec<InputStream>,
    pub(crate) fetch: Option<usize>,
    pub(crate) hint: RegionHint,
    pub(crate) rows: RowSender,
}

/// One message of a region's rows: a batch, a failure, or the end.
#[derive(Debug)]
pub(crate) enum RegionRows {
    Batch(Result<RecordBatch>),
    Done,
}

/// Where a running region sends its rows, in order.
pub(crate) type RowSender = std::sync::mpsc::Sender<RegionRows>;

/// The rows a region sends, read as DataFusion's iterator of batches.
///
/// A sender dropped before the end, such as when the query stopped,
/// reads as a failure.
struct ReceivedRows {
    receiver: std::sync::mpsc::Receiver<RegionRows>,
    ended: bool,
}

impl Iterator for ReceivedRows {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ended {
            return None;
        }
        match self.receiver.recv() {
            Ok(RegionRows::Batch(batch)) => Some(batch),
            Ok(RegionRows::Done) => {
                self.ended = true;
                None
            }
            Err(_) => {
                self.ended = true;
                Some(Err(DataFusionError::Execution(
                    "the region did not finish".into(),
                )))
            }
        }
    }
}

/// A region's hint over the rows of the tables the executor reads:
/// whether each row may still pass the tests above the region, as they
/// tighten while the region runs.
pub struct InputHint {
    hint: RegionHint,
    columns: RegionColumns,
    /// Each table's rows, by table name, in the order the executor reads
    /// them.
    rows: HashMap<String, RecordBatch>,
    /// The region's columns, which the tests read by position.
    schema: SchemaRef,
}

impl InputHint {
    /// The region column an operator's answer is in, and the field of
    /// it for an extraction's `struct.field`; None when no one column
    /// has the name.
    fn answer_column(&self, answer: &str) -> Option<(usize, Option<String>)> {
        let named = |name: &str| {
            let mut found =
                (self.schema.fields().iter().enumerate()).filter(|(_, field)| field.name() == name);
            match (found.next(), found.next()) {
                (Some((index, _)), None) => Some(index),
                _ => None,
            }
        };
        if let Some(index) = named(answer) {
            return Some((index, None));
        }
        let (name, field) = answer.rsplit_once('.')?;
        Some((named(name)?, Some(field.to_string())))
    }
}

impl InputHint {
    /// Answer a worker's question about the hint, as a worker's
    /// `ServedHint` asks it: the hint's generation, a table's kept
    /// rows, whether a test reads an answer, or whether the tests
    /// reject every answer that starts with a prefix.
    pub fn answer(&self, question: &serde_json::Value) -> Result<crate::workers::Answer> {
        use crate::workers::Answer;
        let text = |name: &str| question[name].as_str().unwrap_or_default();
        Ok(match text("hint") {
            "generation" => Answer::Json(self.hint.generation().into()),
            "keep" => {
                let keep: ArrayRef = Arc::new(BooleanArray::from(self.keep(text("table"))?));
                let batch = RecordBatch::try_from_iter([("keep", keep)])?;
                Answer::Table((batch.schema(), vec![batch]))
            }
            "reads_answer" => Answer::Json(self.reads_answer(text("answer")).into()),
            _ => {
                let completion = &question["completion"];
                let answers = quail_datafusion::derive::partial::Completions {
                    prefix: text("prefix").to_string(),
                    complete: completion["complete"].as_bool().unwrap_or(false),
                    may_be_null: completion["may_be_null"].as_bool().unwrap_or(false),
                    tails: serde_json::from_value(completion["tails"].clone()).unwrap_or(None),
                    max_chars: (completion["max_chars"].as_u64()).map(|count| count as usize),
                };
                Answer::Json(self.rejects_answer(text("answer"), &answers)?.into())
            }
        })
    }
}

impl InputHint {
    /// A number that changes whenever a test tightens.
    pub fn generation(&self) -> u64 {
        self.hint.generation()
    }

    /// The tables the hint holds rows of.
    pub fn tables(&self) -> Vec<String> {
        self.rows.keys().cloned().collect()
    }

    /// Whether each row of `table`, in the order the executor reads them,
    /// may still pass every test.
    ///
    /// # Errors
    ///
    /// A test does not evaluate.
    pub fn keep(&self, table: &str) -> Result<Vec<bool>> {
        match self.rows.get(table) {
            Some(rows) => self.hint.keep(&self.columns, table, rows),
            None => Ok(Vec::new()),
        }
    }

    /// Whether a test reads the answer column `answer`, an operator's
    /// name such as `vendor` or `contract.law`, and no other column.
    pub fn reads_answer(&self, answer: &str) -> bool {
        self.answer_column(answer)
            .is_some_and(|(index, _)| self.hint.reads_answer(index))
    }

    /// Whether some test, as the tests are now, is TRUE for none of
    /// `answers`, the complete answers `answer` may still be.
    ///
    /// # Errors
    ///
    /// A test does not evaluate.
    pub fn rejects_answer(
        &self,
        answer: &str,
        answers: &quail_datafusion::derive::partial::Completions,
    ) -> Result<bool> {
        match self.answer_column(answer) {
            Some((index, field)) => self.hint.rejects_answer(index, field.as_deref(), answers),
            None => Ok(false),
        }
    }
}

/// Where a region's engine sends each batch of finished rows while the
/// region runs.
pub struct RegionSink {
    rows: RowSender,
    names: Vec<String>,
    schema: SchemaRef,
}

impl RegionSink {
    /// Send a batch of rows on to DataFusion, in the region's schema.
    ///
    /// # Errors
    ///
    /// The batch lacks a column of the region.
    pub fn send(&self, batch: RecordBatch) -> Result<()> {
        let batch = conform(batch, &self.names, &self.schema)?;
        // the receiver is gone once DataFusion stops reading
        let _ = self.rows.send(RegionRows::Batch(Ok(batch)));
        Ok(())
    }
}

impl Executor {
    /// One region's operators and DataFusion plan.
    fn request(&self, region: &ModelRegion) -> Result<(Operators, LogicalPlan)> {
        let (turn, layout) = &self.prompting;
        derive::derive_region(region, turn, layout).map_err(quail_datafusion::external)
    }

    /// Plan a region in Rust, or None when the engine plans it: a region that
    /// asks no model. `derived` is the region's operators and DataFusion
    /// plan; the plan carries the DataFusion plan's text on to the engine,
    /// beside what its scans read.
    fn plan_here(
        &self,
        tables: &HashMap<String, TableRows>,
        derived: &(Operators, LogicalPlan),
        fetch: Option<usize>,
    ) -> Result<Option<PlannedRegion>> {
        let Some((planner, settings)) = &self.planning else {
            return Ok(None);
        };
        let mut operators = derived.0.clone();
        let logical = derived.1.display_indent().to_string();
        if let Some(fetch) = fetch {
            operators = operators.limited(fetch as i64);
        }
        // the engine plans a region that asks no model
        let asks_model = !(operators.filters.is_empty()
            && operators.joins.is_empty()
            && operators.classifies.is_empty()
            && operators.extracts.is_empty()
            && operators.sae_features.is_empty()
            && operators.maps.is_empty()
            && operators.projections.is_empty());
        if !asks_model {
            return Ok(None);
        }
        // planning reads, and tokenizes, each input's documents first
        let started = std::time::Instant::now();
        let compile = |error: quail_planner::error::CompileError| quail_datafusion::external(error);
        let (outcome, scans) = planner
            .plan(&operators, tables, settings)
            .map_err(compile)?;
        let mut planned = outcome_json(&outcome).map_err(compile)?;
        if planned.get("envelope").is_some() {
            if let Some(edited) = self.edited.lock().expect("the edited lock").take() {
                planned["envelope"] = edited;
            }
            if settings.gpu_timing {
                planned["envelope"]["settings"]["gpu_timing"] = true.into();
            }
        }
        planned["logical"] = logical.into();
        planned["planning_s"] = started.elapsed().as_secs_f64().into();
        Ok(Some((planned.to_string(), scans)))
    }

    /// Plan a region without running it, over the rows of its inputs that
    /// call no model; an input that calls a model plans as empty.
    ///
    /// Returns the region as [`region_json`] writes it, and the planned
    /// region or None when the engine plans it.
    pub(crate) fn plan_only(
        &self,
        region: &ModelRegion,
        inputs: &[InputRows],
    ) -> Result<(String, Option<PlannedRegion>)> {
        let derived = self.request(region)?;
        let sorted = inputs
            .iter()
            .zip(&region.inputs)
            .map(|((name, batches), input)| Ok((name.clone(), in_source_order(input, batches)?)))
            .collect::<Result<Vec<InputRows>>>()?;
        let planned = if self.planning.is_some() {
            let tables = planning_tables(region, &sorted, &derived.0)?;
            self.plan_here(&tables, &derived, None)?
        } else {
            None
        };
        Ok((
            with_region_id(region_json(&derived.0, &derived.1)?, region.id)?,
            planned,
        ))
    }

    /// Run a call's region on this thread and send its rows, then the end.
    pub(crate) fn serve(&self, call: RegionCall) {
        let found = self.run_here(
            &call.region,
            call.inputs,
            call.fetch,
            &call.hint,
            &call.rows,
        );
        if let Err(error) = found {
            let _ = call.rows.send(RegionRows::Batch(Err(error)));
        }
        let _ = call.rows.send(RegionRows::Done);
    }

    /// Run the region on this thread, sending its rows to `rows` as the
    /// executor finishes them.
    ///
    /// The region runs once, as one plan over every row of its inputs.
    /// With a hint (see [`input_hint`]), it first drops each input row a
    /// test rejects, and hands the executor the hint to drop rows by as
    /// the tests tighten.
    fn run_here(
        &self,
        region: &ModelRegion,
        inputs: Vec<InputStream>,
        fetch: Option<usize>,
        hint: &RegionHint,
        rows: &RowSender,
    ) -> Result<()> {
        let derived = self.request(region)?;
        let inputs = collect_inputs(inputs)?;
        let mut sorted;
        let mut held = None;
        let (planned, inputs) = if self.planning.is_some() {
            sorted = inputs
                .iter()
                .zip(&region.inputs)
                .map(|((name, batches), input)| {
                    Ok((name.clone(), in_source_order(input, batches)?))
                })
                .collect::<Result<Vec<InputRows>>>()?;
            let mut tables = planning_tables(region, &sorted, &derived.0)?;
            held = input_hint(region, &derived.0, fetch, hint, &mut sorted, &mut tables)?;
            (self.plan_here(&tables, &derived, fetch)?, sorted.as_slice())
        } else {
            (None, inputs.as_slice())
        };
        let schema = Arc::clone(region.plan.schema());
        let names = quail_column_names(&region.plan);
        let sink = RegionSink {
            rows: rows.clone(),
            names: names.clone(),
            schema: Arc::clone(schema.inner()),
        };
        // Carry the logical occurrence's identity, not callback completion order.
        let planned = planned
            .map(|(payload, scans)| {
                Ok::<_, DataFusionError>((with_region_id(payload, region.id)?, scans))
            })
            .transpose()?;
        // the engine reads a region it plans as JSON
        let payload = match &planned {
            None => Some(with_region_id(
                region_json(&derived.0, &derived.1)?,
                region.id,
            )?),
            Some(_) => None,
        };
        let engine = match &self.runs {
            Runs::Engine(engine) => engine,
            Runs::Workers(workers) => {
                let (found, scans) = planned.ok_or_else(|| {
                    DataFusionError::NotImplemented(
                        "a region that asks no model does not run on workers".into(),
                    )
                })?;
                let conformed = move |batch| conform(batch, &names, schema.inner());
                let mut source_rows = BTreeMap::new();
                for ((name, batches), input) in inputs.iter().zip(&region.inputs) {
                    if let Some(positions) = source_positions(name, input, batches)? {
                        source_rows.insert(name.clone(), positions);
                    }
                }
                let ran = run_on_workers(
                    workers,
                    found,
                    scans,
                    inputs.to_vec(),
                    &source_rows,
                    held,
                    conformed,
                    rows,
                    self.ready.lock().expect("the ready lock").clone(),
                )?;
                self.regions.lock().expect("the regions lock").push(ran);
                return Ok(());
            }
        };
        let returned = engine.run(EngineRegion {
            region,
            inputs,
            planned,
            payload,
            fetch,
            hint: held,
            sink,
        })?;
        for batch in returned {
            let batch = conform(batch, &names, schema.inner())?;
            if rows.send(RegionRows::Batch(Ok(batch))).is_err() {
                break;
            }
        }
        Ok(())
    }
}

/// Run a region Rust planned on the executor's workers, sending its
/// rows, as `conformed` gives them the region's schema, to `rows`;
/// returns what the region ran.
///
/// `found` is the planned region as [`outcome_json`] writes it, and
/// `scans` what its scans read. `source_rows` gives, by input name,
/// each row's position in the registered table. Each worker boots for
/// the plan first; a worker that has the model loaded takes no time.
#[allow(clippy::too_many_arguments)]
fn run_on_workers(
    workers: &Mutex<Workers>,
    found: String,
    scans: Vec<ScanInputs>,
    inputs: Vec<InputRows>,
    source_rows: &BTreeMap<String, Vec<i32>>,
    hint: Option<InputHint>,
    conformed: impl Fn(RecordBatch) -> Result<RecordBatch> + Send + Sync + 'static,
    rows: &RowSender,
    ready: Option<Ready>,
) -> Result<RegionRan> {
    let read = |error: serde_json::Error| DataFusionError::External(Box::new(error));
    let plan: serde_json::Value = serde_json::from_str(&found).map_err(read)?;
    if let Some(refusal) = plan.get("refusal") {
        return Err(DataFusionError::External(
            format!("RefusalError: {refusal}").into(),
        ));
    }
    let envelope = &plan["envelope"];
    let operators_json = plan["operators"].as_str().unwrap_or_default();
    let operators: Operators = serde_json::from_str(operators_json).map_err(read)?;
    let scanned: Vec<crate::request::Scanned> = scans
        .iter()
        .map(|scan| {
            let tokens = (
                Arc::clone(scan.tokens.schema()),
                scan.tokens.record_batches().to_vec(),
            );
            (scan.alias.clone(), tokens, scan.columns.clone())
        })
        .collect();
    // the workers and the answers name a scan by its alias
    let by_alias: BTreeMap<String, Vec<i32>> = (operators.scans.iter())
        .filter_map(|scan| Some((scan.alias.clone(), source_rows.get(&scan.provider)?.clone())))
        .collect();
    let planned = crate::region_run::Planned {
        envelope,
        operators: &operators,
        operators_json,
        scans: &scanned,
        source_rows: &by_alias,
    };
    let settings = &envelope["settings"];
    let setup = serde_json::json!({
        "model": envelope["model"], "device": envelope["device"],
        "backend": envelope["backend"], "gpus": envelope["gpus"],
        "chunk_tokens": settings["chunk_tokens"], "true_ids": settings["true_ids"],
        "false_ids": settings["false_ids"], "arena_pages": settings["arena_pages"],
        // the worker finds the model's files and imports no extension
        "model_path": null, "extensions": [],
    });
    let finished = Arc::new(Mutex::new(planned.finished_rows()?));
    let hinted = hint.as_ref().map(|hint| {
        (operators.scans.iter())
            .filter(|scan| hint.rows.contains_key(&scan.provider))
            .map(|scan| (scan.alias.clone(), scan.provider.clone()))
            .collect()
    });
    let listening = crate::region_run::Listening {
        hinted,
        progress: false,
        answers: false,
    };
    let sender = rows.clone();
    let on_rows = Arc::new(move |batch: RecordBatch| {
        // the receiver is gone once DataFusion stops reading
        let _ = sender.send(RegionRows::Batch(Ok(conformed(batch)?)));
        Ok(())
    });
    let ask: crate::workers::Ask = Box::new(move |question| match &hint {
        Some(hint) if question.get("hint").is_some() => hint.answer(question),
        _ => Ok(crate::workers::Answer::Json(serde_json::Value::Null)),
    });
    let mut workers = workers.lock().expect("the workers lock");
    let entered = std::time::Instant::now();
    let boots = workers.boot(&setup)?;
    if let Some(ready) = ready {
        ready();
    }
    let started = std::time::Instant::now();
    let ran =
        crate::region_run::run_region(&mut workers, &planned, &listening, &finished, on_rows, ask)?;
    let seconds = started.elapsed().as_secs_f64();
    let rows = (finished.lock().expect("the rows lock"))
        .rows()
        .map_err(crate::workers::failure)?;
    Ok(RegionRan {
        planned: found,
        scans,
        inputs,
        rows,
        ran,
        boots,
        seconds,
        total_seconds: entered.elapsed().as_secs_f64(),
    })
}

impl RegionRunner for Executor {
    /// Run the region on the thread that called DataFusion, so the GPU
    /// process's per-thread state is set up once, not on each worker.
    fn run(
        &self,
        region: &ModelRegion,
        inputs: Vec<InputStream>,
        fetch: Option<usize>,
        hint: &RegionHint,
    ) -> Result<RegionBatches> {
        let caller = self.caller.lock().expect("the caller lock").clone();
        let (rows, receiver) = std::sync::mpsc::channel();
        let call = RegionCall {
            region: region.clone(),
            inputs,
            fetch,
            hint: hint.clone(),
            rows,
        };
        match caller {
            Some((thread, sender)) if thread != std::thread::current().id() => sender
                .send(call)
                .map_err(|_| DataFusionError::Execution("the query ended".into()))?,
            _ => self.serve(call),
        }
        Ok(Box::new(ReceivedRows {
            receiver,
            ended: false,
        }))
    }
}

/// The hint over the rows of each table a test of `hint` reads, or None
/// when the region takes none. Each row of `inputs` a test rejects is
/// dropped first.
///
/// A fetch or a stop key keeps the first rows: dropping one changes which
/// rows those are, so such a region takes no hint.
fn input_hint(
    region: &ModelRegion,
    operators: &Operators,
    fetch: Option<usize>,
    hint: &RegionHint,
    inputs: &mut [InputRows],
    tables: &mut HashMap<String, TableRows>,
) -> Result<Option<InputHint>> {
    if hint.is_empty()
        || fetch.is_some()
        || operators.fetch.is_some()
        || !operators.stop_keys.is_empty()
    {
        return Ok(None);
    }
    let mut columns = RegionColumns::of(region);
    // Named tables are projected before materialization. Hint source positions
    // refer to their registered schemas, so translate them to these batches.
    for (name, provider) in identified_scans(region)? {
        if let Some(rows) = tables.get(&name) {
            for column in &mut columns.0 {
                if let quail_datafusion::RegionColumn::Source(table, index) = column
                    && *table == name
                {
                    *index = rows
                        .schema
                        .index_of(provider.schema().field(*index).name())?;
                }
            }
            for (table, index) in &mut columns.1 {
                if *table == name {
                    *index = rows
                        .schema
                        .index_of(provider.schema().field(*index).name())?;
                }
            }
        }
    }
    let mut rows = HashMap::new();
    for table in hint.tables(&columns) {
        let found = match inputs.iter().position(|(name, _)| *name == table) {
            Some(input) => {
                let schema = Arc::new(region.inputs[input].schema().as_arrow().clone());
                let all = concat_batches(&schema, &inputs[input].1)?;
                let keep = BooleanArray::from(hint.keep(&columns, &table, &all)?);
                let kept = filter_record_batch(&all, &keep)?;
                inputs[input].1 = vec![kept.clone()];
                if let Some(rows) = tables.get_mut(&table) {
                    // The logical input still describes the whole table. Do
                    // not reuse or populate its cache after a runtime hint.
                    rows.identity = None;
                    rows.batches = vec![kept.clone()];
                }
                Some(kept)
            }
            None => tables
                .get(&table)
                .map(|rows| concat_batches(&rows.schema, &rows.batches))
                .transpose()?,
        };
        if let Some(found) = found {
            rows.insert(table, found);
        }
    }
    // a test of an answer alone holds no table's rows; the decoder
    // reads it while the answer is written
    let answers = columns.0.iter().enumerate().any(|(index, column)| {
        matches!(column, quail_datafusion::RegionColumn::Answer)
            && hint.tests.iter().any(|test| {
                datafusion::physical_expr::utils::collect_columns(test)
                    .iter()
                    .any(|read| read.index() == index)
            })
    });
    Ok((!rows.is_empty() || answers).then(|| InputHint {
        hint: hint.clone(),
        columns,
        rows,
        schema: Arc::new(region.plan.schema().as_arrow().clone()),
    }))
}

/// Each identified table a region's plan reads by name, by table name.
fn identified_scans(region: &ModelRegion) -> Result<Vec<(String, Arc<dyn TableProvider>)>> {
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::datasource::source_as_provider;
    use datafusion::logical_expr::LogicalPlan;
    let mut found: Vec<(String, Arc<dyn TableProvider>)> = Vec::new();
    region.plan.apply(|node| {
        if let LogicalPlan::TableScan(scan) = node
            && let Ok(provider) = source_as_provider(&scan.source)
            && provider.downcast_ref::<IdentifiedTable>().is_some()
        {
            let name = scan.table_name.table().to_string();
            if !found.iter().any(|(known, _)| *known == name) {
                found.push((name, provider));
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(found)
}

/// A region as JSON: `operators`, its operators as JSON text, and
/// `logical`, DataFusion's plan of the region as `display_indent` writes
/// it.
///
/// # Errors
///
/// The operators do not write.
pub fn region_json(
    operators: &Operators,
    plan: &datafusion::logical_expr::LogicalPlan,
) -> Result<String> {
    let operators =
        crate::planning::operators_json(operators).map_err(quail_datafusion::external)?;
    // calls a typo apart; calls that differ only in spelling are refused
    let warnings: Vec<String> = quail_datafusion::derive::similar::similar_calls(plan)
        .iter()
        .filter(|found| found.likeness == quail_datafusion::derive::similar::Likeness::Typo)
        .map(quail_datafusion::derive::similar::Similar::message)
        .collect();
    Ok(serde_json::json!({
        "operators": operators,
        "logical": plan.display_indent().to_string(),
        "warnings": warnings,
    })
    .to_string())
}

/// Add the logical occurrence's ID to either a region or its planned envelope.
fn with_region_id(payload: String, id: Option<usize>) -> Result<String> {
    let mut payload: serde_json::Value = serde_json::from_str(&payload)
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
    payload["region_id"] = serde_json::json!(id);
    Ok(payload.to_string())
}

/// An input's rows as one batch, and each row's position in the
/// registered table the input holds rows of, found by the table's id
/// column; None for an input that is no such table's rows.
fn source_lookup(
    input: &datafusion::logical_expr::LogicalPlan,
    batches: &[RecordBatch],
) -> Result<Option<(RecordBatch, datafusion::arrow::array::UInt64Array)>> {
    use datafusion::arrow::datatypes::DataType;
    use datafusion::common::Constraint;
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::datasource::source_as_provider;
    use datafusion::logical_expr::LogicalPlan;
    let mut source = None;
    input.apply(|node| {
        if let LogicalPlan::TableScan(scan) = node
            && let Ok(provider) = source_as_provider(&scan.source)
            && provider.downcast_ref::<IdentifiedTable>().is_some()
        {
            source = Some(provider);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    let Some(provider) = source else {
        return Ok(None);
    };
    let table = provider
        .downcast_ref::<IdentifiedTable>()
        .expect("an identified table");
    let Some(scanned) = table.inner().downcast_ref::<ScannedTable>() else {
        return Ok(None);
    };
    let key = scanned
        .constraints
        .iter()
        .find_map(|constraint| match constraint {
            Constraint::PrimaryKey(keys) if keys.len() == 1 => Some(keys[0]),
            _ => None,
        });
    let Some(key) = key.map(|index| scanned.schema.field(index).name().clone()) else {
        return Ok(None);
    };
    let Some(first) = batches.first() else {
        return Ok(None);
    };
    let rows = concat_batches(&first.schema(), batches)?;
    let Ok(column) = rows.schema().index_of(&key) else {
        return Ok(None);
    };
    let texts = |values: &ArrayRef| -> Result<Vec<Option<String>>> {
        let values = cast(values, &DataType::Utf8)?;
        let values = values
            .as_any()
            .downcast_ref::<datafusion::arrow::array::StringArray>()
            .expect("a string array");
        Ok(values
            .iter()
            .map(|value| value.map(str::to_string))
            .collect())
    };
    let mut positions: HashMap<String, u64> = HashMap::new();
    let mut next = 0u64;
    for batch in scanned.column_rows(&key)? {
        let column = batch.columns().first().ok_or_else(|| {
            DataFusionError::Execution(format!("the table's scan returned no column {key}"))
        })?;
        for value in texts(column)? {
            if let Some(value) = value {
                positions.entry(value).or_insert(next);
            }
            next += 1;
        }
    }
    let found = texts(rows.column(column))?
        .into_iter()
        .map(|value| value.and_then(|value| positions.get(&value).copied()))
        .collect();
    Ok(Some((rows, found)))
}

/// An input's rows in the order of the registered table it holds rows of,
/// found by the table's id column; the rows unchanged for any other input.
///
/// DataFusion scans and filters in parallel, so the rows reach the region
/// in no fixed order.
fn in_source_order(
    input: &datafusion::logical_expr::LogicalPlan,
    batches: &[RecordBatch],
) -> Result<Vec<RecordBatch>> {
    use datafusion::arrow::compute::{sort_to_indices, take_record_batch};
    let Some((rows, found)) = source_lookup(input, batches)? else {
        return Ok(batches.to_vec());
    };
    let order = sort_to_indices(&found, None, None)?;
    Ok(vec![take_record_batch(&rows, &order)?])
}

/// Each row's position in the registered table, for an input named
/// `<table>@<alias>` that holds some of a registered table's rows; None
/// for any other input. A row the table does not hold is -1.
fn source_positions(
    name: &str,
    input: &datafusion::logical_expr::LogicalPlan,
    batches: &[RecordBatch],
) -> Result<Option<Vec<i32>>> {
    if !name.contains(quail_planner::operators::INPUT_SEPARATOR) {
        return Ok(None);
    }
    Ok(source_lookup(input, batches)?.map(|(_, found)| {
        (found.iter())
            .map(|position| position.map_or(-1, |position| position as i32))
            .collect()
    }))
}

/// Materialize each named table once, projecting the union of columns all
/// placements read. Keep these same batches for runtime hints and planning.
fn planning_tables(
    region: &ModelRegion,
    inputs: &[InputRows],
    operators: &Operators,
) -> Result<HashMap<String, TableRows>> {
    let required = crate::planning::required_columns(operators);
    let columns = RegionColumns::of(region);
    let mut tables = HashMap::new();
    for (name, provider) in identified_scans(region)? {
        let table = provider
            .downcast_ref::<IdentifiedTable>()
            .expect("an identified table");
        let Some(scanned) = table.inner().downcast_ref::<ScannedTable>() else {
            continue;
        };
        let mut names = required.get(&name).cloned().unwrap_or_default();
        // The hint reads passed-through columns by source position. Include
        // them and primary keys even if no physical alternative reads them.
        for column in &columns.0 {
            if let quail_datafusion::RegionColumn::Source(source, index) = column
                && *source == name
            {
                names.push(scanned.schema.field(*index).name().clone());
            }
        }
        for constraint in scanned.constraints.iter() {
            if let datafusion::common::Constraint::PrimaryKey(indices) = constraint {
                names.extend(
                    indices
                        .iter()
                        .map(|index| scanned.schema.field(*index).name().clone()),
                );
            }
        }
        let mut indices = names
            .iter()
            .map(|name| scanned.schema.index_of(name))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        indices.sort_unstable();
        indices.dedup();
        let schema = Arc::new(scanned.schema.project(&indices)?);
        let names = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        tables.insert(
            name,
            TableRows {
                identity: Some(table.identity().to_string()),
                schema,
                batches: scanned.scan_columns(names)?,
            },
        );
    }
    for ((name, batches), input) in inputs.iter().zip(&region.inputs) {
        tables.insert(
            name.clone(),
            TableRows {
                identity: whole_table_identity(input),
                schema: Arc::new(input.schema().as_arrow().clone()),
                batches: batches.clone(),
            },
        );
    }
    Ok(tables)
}

/// The content identity of unchanged base columns in source order. Computed
/// or renamed columns must not share token stores with another projection of
/// the same table, even when both plans choose the same generated name.
fn whole_table_identity(input: &datafusion::logical_expr::LogicalPlan) -> Option<String> {
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::datasource::source_as_provider;
    use datafusion::logical_expr::LogicalPlan;
    let mut identity = None;
    let mut whole = true;
    let mut scans = 0;
    let _ = input.apply(|node| {
        match node {
            LogicalPlan::TableScan(scan) => {
                scans += 1;
                whole &= scan.filters.is_empty() && scan.fetch.is_none();
                identity = source_as_provider(&scan.source).ok().and_then(|provider| {
                    let table = provider.downcast_ref::<IdentifiedTable>()?;
                    let scanned = table.inner().downcast_ref::<ScannedTable>()?;
                    // in_source_order restores only inputs retaining the source
                    // primary key. Otherwise partition order is not stable.
                    let ordered = scanned.constraints.iter().any(|constraint| {
                        matches!(constraint, datafusion::common::Constraint::PrimaryKey(keys)
                            if keys.len() == 1 && input.schema().as_arrow()
                                .index_of(scanned.schema.field(keys[0]).name()).is_ok())
                    });
                    ordered.then(|| table.identity().to_string())
                });
            }
            LogicalPlan::SubqueryAlias(_) => {}
            LogicalPlan::Projection(projection) => {
                whole &= projection
                    .expr
                    .iter()
                    .all(|expr| matches!(expr, Expr::Column(_)));
            }
            LogicalPlan::Extension(extension)
                if extension
                    .node
                    .as_any()
                    .downcast_ref::<SharedInput>()
                    .is_some_and(|shared| !shared.numbered) => {}
            _ => whole = false,
        }
        Ok(TreeNodeRecursion::Continue)
    });
    if whole && scans == 1 { identity } else { None }
}

/// Give a batch of Quail's rows the region's schema: each region
/// column is the Quail column of the same name, cast to its type.
///
/// A column Quail names differently, such as `r.details.body` for
/// DataFusion's `r.details[body]`, is the one in the same position when
/// both list the same number of columns, as both follow the SELECT list.
fn conform(batch: RecordBatch, names: &[String], schema: &SchemaRef) -> Result<RecordBatch> {
    let positional = batch.num_columns() == names.len();
    let columns = names
        .iter()
        .zip(schema.fields())
        .enumerate()
        .map(|(index, (name, field))| {
            let found = batch
                .column_by_name(name)
                .or_else(|| positional.then(|| batch.column(index)));
            let column = found.ok_or_else(|| {
                let returned: Vec<_> = batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect();
                DataFusionError::Execution(format!(
                    "Quail returned columns {returned:?}, without the region's column {name}"
                ))
            })?;
            Ok(cast(column, field.data_type())?)
        })
        .collect::<Result<Vec<ArrayRef>>>()?;
    Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
}

/// Returns the named columns of a table's rows.
pub type Scan = Arc<dyn Fn(Vec<String>) -> Result<Vec<RecordBatch>> + Send + Sync>;

/// A table whose rows a function returns on each scan.
///
/// Model regions read it through the planner, which keeps its token
/// counts, so DataFusion scans it only for operators outside every
/// region.
pub struct ScannedTable {
    pub(crate) schema: SchemaRef,
    pub(crate) scan: Scan,
    /// The id column as a primary key, which DataFusion's rules read.
    pub(crate) constraints: Constraints,
}

impl std::fmt::Debug for ScannedTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScannedTable")
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl ScannedTable {
    /// One column of the table's rows.
    fn column_rows(&self, name: &str) -> Result<Vec<RecordBatch>> {
        self.scan_columns(vec![name.to_string()])
    }

    fn scan_columns(&self, columns: Vec<String>) -> Result<Vec<RecordBatch>> {
        (self.scan)(columns)
    }
}

#[async_trait]
impl TableProvider for ScannedTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn constraints(&self) -> Option<&Constraints> {
        Some(&self.constraints)
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = match projection {
            Some(indices) => Arc::new(self.schema.project(indices)?),
            None => Arc::clone(&self.schema),
        };
        let columns: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
        let batches = self.scan_columns(columns)?;
        // one partition per thread DataFusion runs, each a slice of the
        // rows, so the operators above read the table in parallel
        let num_partitions = state.config().target_partitions().max(1);
        let mut partitions: Vec<Vec<RecordBatch>> = vec![Vec::new(); num_partitions];
        for batch in batches {
            let size = batch.num_rows().div_ceil(num_partitions).max(1);
            let mut offset = 0;
            let mut index = 0;
            while offset < batch.num_rows() {
                let length = size.min(batch.num_rows() - offset);
                partitions[index % num_partitions].push(batch.slice(offset, length));
                offset += length;
                index += 1;
            }
        }
        Ok(MemorySourceConfig::try_new_exec(&partitions, schema, None)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};

    #[test]
    fn hint_filtered_inputs_lose_their_whole_table_cache_identity() {
        use datafusion::arrow::array::Int64Array;
        use datafusion::common::ScalarValue;
        use datafusion::datasource::{MemTable, provider_as_source};
        use datafusion::logical_expr::{LogicalPlanBuilder, Operator};
        use datafusion::physical_expr::expressions::{BinaryExpr, Column, Literal};

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let source = MemTable::try_new(Arc::clone(&schema), vec![vec![batch.clone()]]).unwrap();
        let plan = LogicalPlanBuilder::scan("input", provider_as_source(Arc::new(source)), None)
            .unwrap()
            .build()
            .unwrap();
        let mut region = ModelRegion::new(plan.clone());
        region.inputs = vec![plan];
        region.input_names = vec!["input".into()];
        for threshold in [0, 1, 3] {
            let mut inputs = vec![("input".into(), vec![batch.clone()])];
            let mut tables = HashMap::from([(
                "input".into(),
                TableRows {
                    identity: Some("whole-table".into()),
                    schema: Arc::clone(&schema),
                    batches: vec![batch.clone()],
                },
            )]);
            let hint = RegionHint {
                tests: vec![Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("id", 0)),
                    Operator::Gt,
                    Arc::new(Literal::new(ScalarValue::Int64(Some(threshold)))),
                ))],
            };
            input_hint(
                &region,
                &Operators::default(),
                None,
                &hint,
                &mut inputs,
                &mut tables,
            )
            .unwrap()
            .unwrap();
            let rows = &tables["input"];
            assert!(rows.identity.is_none());
            assert_eq!(rows.batches, inputs[0].1);
            assert_eq!(rows.batches[0].num_rows(), (3 - threshold) as usize);
        }
    }

    #[test]
    fn executor_rows_missing_a_required_column_return_an_error() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "answer",
            DataType::Utf8,
            true,
        )]));
        let batch = RecordBatch::new_empty(Arc::new(Schema::empty()));
        let error = conform(batch, &["answer".into()], &schema).unwrap_err();
        assert!(matches!(error, DataFusionError::Execution(_)));
        assert!(
            error
                .to_string()
                .contains("without the region's column answer")
        );
    }
}
