//! Quail: AI functions in SQL, on DataFusion.
//!
//! A [`Session`] runs SQL in DataFusion with Quail's AI functions. Each
//! part of a query that calls the model is one model region, which
//! Quail's executor plans and runs; DataFusion runs every other
//! operator. The executor is Quail's Python package, which a Rust
//! program runs in an embedded Python interpreter. Python's
//! `quail.Session` is this same `Session`.
//!
//! ```ignore
//! let config = quail::EngineConfig::new("qwen3-4b-fp8", "h100-sxm");
//! let session = quail::Session::new(&config)?;
//! session.register_table("reviews", reviews)?;
//! let batches = session.sql(
//!     "SELECT r.id FROM reviews r \
//!      WHERE AI.IF(PROMPT('Does the review discuss the ending?\n\n{0}', r.body))",
//! )?;
//! ```

mod engine;
pub mod planning;
pub mod pyarrow;
pub mod results;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::TableProvider;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{Constraint, Constraints, DataFusionError, Result};
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use pyo3::prelude::*;
use quail_datafusion::{IdentifiedTable, InputRows, ModelRegion, RegionRunner, quail_schema};
use tokio::runtime::Runtime;
use tokio::sync::mpsc::unbounded_channel;

pub use crate::engine::region_json;
use crate::engine::{PythonRunner, PythonTable, RegionCall};

/// The model and hardware a [`Session`] runs on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineConfig {
    /// A model name Quail knows, such as `qwen3-4b-fp8`.
    pub model: String,
    /// A device name Quail knows, such as `h100-sxm`.
    pub device: String,
    /// The number of GPUs.
    pub gpus: usize,
    /// The execution backend, `quail` by default.
    pub backend: String,
}

impl EngineConfig {
    /// A configuration of `model` on one `device` GPU, with the `quail`
    /// backend.
    pub fn new(model: impl Into<String>, device: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            device: device.into(),
            gpus: 1,
            backend: "quail".to_string(),
        }
    }
}

/// A DataFusion session with Quail's AI functions, whose model regions
/// run on Quail's executor.
///
/// `sql` blocks until the query finishes; call it outside an async
/// runtime, or from `spawn_blocking`. Each model region runs on the
/// thread that called `sql`.
pub struct Session {
    context: SessionContext,
    /// Taken when the session drops, to shut down without blocking.
    runtime: Option<Runtime>,
    runner: Arc<PythonRunner>,
    /// Whether this session opened its engine, and closes it.
    owns_engine: bool,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").finish_non_exhaustive()
    }
}

fn python_failure(error: PyErr) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}

/// Add the site-packages of the virtual environment `VIRTUAL_ENV`
/// names, which an embedded interpreter does not read by itself.
fn add_virtual_env(py: Python<'_>) -> PyResult<()> {
    let Some(root) = std::env::var_os("VIRTUAL_ENV").map(PathBuf::from) else {
        return Ok(());
    };
    let Ok(entries) = std::fs::read_dir(root.join("lib")) else {
        return Ok(());
    };
    let site = py.import("site")?;
    for entry in entries.flatten() {
        let packages = entry.path().join("site-packages");
        if packages.is_dir() {
            site.call_method1("addsitedir", (packages.to_string_lossy().to_string(),))?;
        }
    }
    Ok(())
}

/// Start Python in this process if it has not started, and add the
/// virtual environment `VIRTUAL_ENV` names to its path.
///
/// # Errors
///
/// The environment's site-packages do not load.
pub fn start_python() -> Result<()> {
    Python::initialize();
    Python::attach(add_virtual_env).map_err(python_failure)
}

impl Session {
    /// Open a session on `config`, starting Python if this process has
    /// not, and loading Quail's executor from the `quail` package.
    ///
    /// The interpreter finds `quail` on its own path, or in the virtual
    /// environment `VIRTUAL_ENV` names.
    ///
    /// # Errors
    ///
    /// Python cannot import `quail`, or Quail refuses the configuration.
    pub fn new(config: &EngineConfig) -> Result<Self> {
        start_python()?;
        let engine = Python::attach(|py| {
            let engine = py
                .import("quail.api.engine")?
                .getattr("Engine")?
                .call_method1(
                    "open",
                    (
                        config.model.as_str(),
                        config.device.as_str(),
                        config.gpus,
                        config.backend.as_str(),
                    ),
                )?;
            Ok::<_, PyErr>(engine.unbind())
        })
        .map_err(python_failure)?;
        let mut session = Self::with_engine(engine)?;
        session.owns_engine = true;
        Ok(session)
    }

    /// A session whose model regions run on `engine`, a Python object
    /// with `run_region(kind, payload, tables, fetch)` and the
    /// attributes `turn`, `layout`, and `rules`.
    ///
    /// `run_region` takes ("plan", a region's operators and DataFusion
    /// plan as [`region_json`] writes them) or ("sql", the region's SQL)
    /// for a region whose shape Quail's operators do not read, the rows of each input the region reads by table name, and
    /// the region's row limit, and returns a pyarrow Table or
    /// RecordBatchReader. `turn` and `layout` are the model's chat-turn
    /// text and prompt layout, or None to send every region as SQL.
    /// `rules` names the optimizer rules each model region runs, in
    /// order, or is None for the built-in rules.
    ///
    /// # Errors
    ///
    /// An attribute is missing or a rule name is unknown.
    /// When the engine has a `tokenizer_file` attribute, the path of the
    /// model's `tokenizer.json`, Rust plans each region with it.
    pub fn with_engine(engine: Py<PyAny>) -> Result<Self> {
        let tokenizer_file = Python::attach(|py| match engine.bind(py).getattr("tokenizer_file") {
            Ok(found) => found.extract::<Option<String>>(),
            Err(_) => Ok(None),
        })
        .map_err(python_failure)?;
        let planner = tokenizer_file
            .map(|path| planning::Planner::new(std::path::Path::new(&path)))
            .transpose()
            .map_err(|error| DataFusionError::Plan(format!("CompileError: {}", error.0)))?
            .map(Arc::new);
        Self::with_planner(engine, planner)
    }

    /// A session as [`Session::with_engine`] describes, whose regions
    /// `planner` plans in Rust when the engine has a `settings`
    /// attribute, the configuration as JSON, and the engine's `execute`
    /// runs; without a planner every region goes to `run_region`.
    ///
    /// # Errors
    ///
    /// An attribute is missing or malformed, or a rule name is unknown.
    pub fn with_planner(
        engine: Py<PyAny>,
        planner: Option<Arc<planning::Planner>>,
    ) -> Result<Self> {
        let (prompting, rules, settings) = Python::attach(|py| {
            let bound = engine.bind(py);
            let turn: Option<(String, String)> = bound.getattr("turn")?.extract()?;
            let layout: Option<String> = bound.getattr("layout")?.extract()?;
            let rules: Option<Vec<String>> = bound.getattr("rules")?.extract()?;
            let settings: Option<String> = match bound.getattr("settings") {
                Ok(found) => found.extract()?,
                Err(_) => None,
            };
            Ok::<_, PyErr>((turn.zip(layout), rules, settings))
        })
        .map_err(python_failure)?;
        let planning = match (planner, settings) {
            (Some(planner), Some(settings)) => Some((
                planner,
                serde_json::from_str::<planning::Settings>(&settings)
                    .map_err(|error| DataFusionError::External(Box::new(error)))?,
            )),
            _ => None,
        };
        let regions = match rules {
            Some(rules) => {
                for name in &rules {
                    quail_datafusion::region_rule(name)?;
                }
                quail_datafusion::OptimizeModelRegions::new(rules)
            }
            None => quail_datafusion::OptimizeModelRegions::default(),
        };
        let runner = Arc::new(PythonRunner {
            engine,
            prompting,
            caller: Mutex::new(None),
            planning,
        });
        let context = quail_datafusion::session_context_with_rules(
            Arc::clone(&runner) as Arc<dyn RegionRunner>,
            regions,
        );
        Ok(Self {
            context,
            runtime: Some(Runtime::new()?),
            runner,
            owns_engine: false,
        })
    }

    /// The DataFusion context, for registering tables or functions.
    pub fn context(&self) -> &SessionContext {
        &self.context
    }

    /// Register a table under `name`.
    ///
    /// DataFusion scans it, and a model region that reads it receives
    /// its rows.
    ///
    /// # Errors
    ///
    /// DataFusion refuses the name.
    pub fn register_table(&self, name: &str, table: Arc<dyn TableProvider>) -> Result<()> {
        self.context.register_table(name, table)?;
        Ok(())
    }

    /// Register a table whose rows `scan(columns)` returns as a pyarrow
    /// Table, the same rows whenever it is scanned, which `identity`
    /// names. `id_column` names the column whose values are unique.
    ///
    /// # Errors
    ///
    /// DataFusion refuses the name.
    pub fn register_python_table(
        &self,
        name: &str,
        schema: SchemaRef,
        scan: Py<PyAny>,
        id_column: Option<&str>,
        identity: &str,
    ) -> Result<()> {
        let keys: Vec<usize> = id_column
            .and_then(|column| schema.index_of(column).ok())
            .into_iter()
            .collect();
        let constraints = if keys.is_empty() {
            Constraints::default()
        } else {
            Constraints::new_unverified(vec![Constraint::PrimaryKey(keys)])
        };
        let table = PythonTable {
            schema,
            scan,
            constraints,
        };
        self.register_table(
            name,
            Arc::new(IdentifiedTable::new(Arc::new(table), identity)),
        )
    }

    /// Run a query and return its rows, with each column named as
    /// Quail names it, such as `r.id`.
    ///
    /// # Errors
    ///
    /// The query does not plan, or a region or operator fails.
    pub fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        Ok(self.sql_in_dialect(query, None)?.1)
    }

    /// Run a query read in `dialect`, a DataFusion dialect name such as
    /// "BigQuery" ("Snowflake" by default), and return its schema and
    /// rows.
    ///
    /// # Errors
    ///
    /// The query does not plan, or a region or operator fails.
    pub fn sql_in_dialect(
        &self,
        query: &str,
        dialect: Option<&str>,
    ) -> Result<(SchemaRef, Vec<RecordBatch>)> {
        let runner = Arc::clone(&self.runner);
        self.runtime().block_on(async {
            let (sender, mut calls) = unbounded_channel::<RegionCall>();
            *runner.caller.lock().expect("the caller lock") =
                Some((std::thread::current().id(), sender));
            let query = async {
                let frame = quail_datafusion::sql_in_dialect(&self.context, query, dialect).await?;
                let schema = quail_schema(frame.schema());
                // renaming the columns keeps each column's buffers
                let batches = frame
                    .collect()
                    .await?
                    .into_iter()
                    .map(|batch| {
                        RecordBatch::try_new(Arc::clone(&schema), batch.columns().to_vec())
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok::<_, DataFusionError>((schema, batches))
            };
            tokio::pin!(query);
            // DataFusion's workers run the operators; this thread runs
            // each model region they hand it
            let done = loop {
                tokio::select! {
                    done = &mut query => break done,
                    Some(call) = calls.recv() => {
                        let rows = runner.run_here(&call.region, &call.inputs, call.fetch);
                        let _ = call.rows.send(rows);
                    }
                }
            };
            *runner.caller.lock().expect("the caller lock") = None;
            done
        })
    }

    /// Return DataFusion's optimized logical plan for a query, with each
    /// model region as one `ModelRegion` node.
    ///
    /// # Errors
    ///
    /// The query does not plan.
    pub fn explain(&self, query: &str, dialect: Option<&str>) -> Result<String> {
        Ok(self
            .optimized_plan(query, dialect)?
            .display_indent()
            .to_string())
    }

    /// Return DataFusion's optimized logical plan for a query.
    ///
    /// # Errors
    ///
    /// The query does not plan.
    pub fn optimized_plan(&self, query: &str, dialect: Option<&str>) -> Result<LogicalPlan> {
        self.runtime().block_on(async {
            quail_datafusion::sql_in_dialect(&self.context, query, dialect)
                .await?
                .into_optimized_plan()
        })
    }

    /// Return each model region of a query, in plan order.
    ///
    /// # Errors
    ///
    /// The query does not plan.
    pub fn regions(&self, query: &str, dialect: Option<&str>) -> Result<Vec<ModelRegion>> {
        let plan = self.optimized_plan(query, dialect)?;
        let mut regions = Vec::new();
        plan.apply(|node| {
            if let LogicalPlan::Extension(extension) = node
                && let Some(region) = extension.node.as_any().downcast_ref::<ModelRegion>()
            {
                regions.push(region.clone());
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(regions)
    }
}

impl Session {
    /// Compute the inputs of a query's model regions that call no model,
    /// each as its name, schema, and rows, region by region, so a region
    /// can be planned on its real rows before the query runs.
    ///
    /// # Errors
    ///
    /// The query does not plan, or an input fails.
    #[allow(clippy::type_complexity)]
    pub fn region_inputs(
        &self,
        query: &str,
        dialect: Option<&str>,
    ) -> Result<Vec<Vec<(String, SchemaRef, Vec<RecordBatch>)>>> {
        let regions = self.regions(query, dialect)?;
        self.runtime().block_on(async {
            let mut found = Vec::new();
            for region in regions {
                let mut inputs = Vec::new();
                for (name, input) in region.input_names.iter().zip(&region.inputs) {
                    let calls_model = input.exists(|node| {
                        Ok(matches!(node, LogicalPlan::Extension(extension)
                            if extension.node.as_any().downcast_ref::<ModelRegion>().is_some()))
                    })?;
                    if calls_model {
                        continue;
                    }
                    let frame =
                        datafusion::dataframe::DataFrame::new(self.context.state(), input.clone());
                    let schema = Arc::new(input.schema().as_arrow().clone());
                    inputs.push((name.clone(), schema, frame.collect().await?));
                }
                found.push(inputs);
            }
            Ok(found)
        })
    }
}

impl Session {
    /// Plan each model region of a query without running it, region by
    /// region: (its kind, the region as [`region_json`] writes it or its
    /// SQL, its planned region as JSON when Rust plans it). A region's inputs that call no model
    /// are computed first, so it plans on their real rows.
    ///
    /// # Errors
    ///
    /// The query does not plan, an input fails, or a region does not bind.
    #[allow(clippy::type_complexity)]
    pub fn plans(
        &self,
        query: &str,
        dialect: Option<&str>,
    ) -> Result<Vec<(&'static str, String, Option<String>)>> {
        let regions = self.regions(query, dialect)?;
        let computed = self.region_inputs(query, dialect)?;
        regions
            .iter()
            .zip(computed)
            .map(|(region, computed)| {
                let inputs: Vec<InputRows> = region
                    .input_names
                    .iter()
                    .zip(&region.inputs)
                    .map(|(name, input)| {
                        let batches = computed
                            .iter()
                            .find(|(found, _, _)| found == name)
                            .map(|(_, _, batches)| batches.clone())
                            .unwrap_or_else(|| {
                                vec![RecordBatch::new_empty(Arc::new(
                                    input.schema().as_arrow().clone(),
                                ))]
                            });
                        (name.clone(), batches)
                    })
                    .collect();
                self.runner.plan_only(region, &inputs)
            })
            .collect()
    }
}

impl Session {
    fn runtime(&self) -> &Runtime {
        self.runtime
            .as_ref()
            .expect("a session's runtime until it drops")
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Python may free a session while another one runs a query, inside
        // that one's runtime, where a blocking shutdown is not allowed
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
        if self.owns_engine {
            Python::attach(|py| {
                // a close that fails leaves nothing this session can release
                let _ = self.runner.engine.call_method0(py, "close");
            });
        }
    }
}
