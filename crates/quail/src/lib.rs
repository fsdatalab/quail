//! Quail: AI functions in SQL, on DataFusion.
//!
//! Quail adds AI functions to DataFusion and runs the model work of a
//! query on the GPU:
//!
//! - `quail_datafusion::QuailExtension` is the DataFusion extension: the
//!   AI functions, Quail's optimizer rules, and model regions. It hands
//!   each model region to a `RegionRunner`.
//! - [`Executor`] is Quail's `RegionRunner`. It plans each model region
//!   and runs it on Quail's GPU workers: one process per GPU, which
//!   Quail starts and which loads the model ([`workers`]).
//! - [`Session`] is a DataFusion session with the extension registered
//!   and an [`Executor`] as its runner, and a blocking `sql()`.
//!
//! ```ignore
//! let config = quail::EngineConfig::new("qwen3-4b-fp8", "h100-sxm");
//! let session = quail::Session::new(&config)?;
//! session.register_table("reviews", reviews)?;
//! let batches = session.sql(
//!     "SELECT r.id FROM reviews r \
//!      WHERE AI.IF(PROMPT('Does the review discuss the ending?\n\n{0}', r.body))",
//! )?;
//!
//! // or add Quail to a DataFusion session of your own
//! let executor = Arc::new(quail::Executor::open(&config)?);
//! let state = quail_datafusion::QuailExtension::new(executor).register(state)?;
//! ```

mod executor;
pub mod finished;
pub mod planning;
pub mod region_run;
pub mod request;
pub mod results;
pub mod shared;
pub mod stages;
pub mod workers;

use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::TableProvider;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{Constraint, Constraints, DataFusionError, Result};
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use quail_datafusion::{
    IdentifiedTable, InputRows, ModelRegion, RegionRunner, SharedInput, quail_schema,
};
use tokio::runtime::Runtime;
use tokio::sync::mpsc::unbounded_channel;

use crate::executor::RegionCall;
pub use crate::executor::{
    EngineRegion, Executor, InputHint, Ready, RegionEngine, RegionRan, RegionSink, Scan,
    ScannedTable, region_json,
};

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
    runner: Arc<Executor>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").finish_non_exhaustive()
    }
}

impl Session {
    /// Open a session on `config`: Quail's [`Executor`] for `config`, as
    /// [`Executor::open`] opens it, with a DataFusion session that runs
    /// every model region on it.
    ///
    /// # Errors
    ///
    /// A GPU worker does not start, or Quail refuses the configuration.
    pub fn new(config: &EngineConfig) -> Result<Self> {
        Self::with_executor(Arc::new(Executor::open(config)?))
    }

    /// A session that runs each model region on `executor`, with the
    /// optimizer rules the executor's engine names.
    ///
    /// SQL calls sharing an executor, even across sessions, are serialized.
    /// Each model region runs on its query's calling thread. Nested SQL on
    /// the same executor from that thread returns an error rather than
    /// deadlocking or replacing the outer query's caller route.
    ///
    /// # Errors
    ///
    /// A rule name is unknown.
    pub fn with_executor(executor: Arc<Executor>) -> Result<Self> {
        let context = quail_datafusion::session_context_with_rules(
            Arc::clone(&executor) as Arc<dyn RegionRunner>,
            executor.rules.clone(),
        )?;
        Ok(Self {
            context,
            runtime: Some(Runtime::new()?),
            runner: executor,
        })
    }

    /// Run an edited plan's envelope in place of the planner's plan
    /// for the next model region, as [`Executor::edit_plan`] does.
    pub fn edit_plan(&self, envelope: Option<serde_json::Value>) {
        self.runner.edit_plan(envelope);
    }

    /// Call `ready` each time the workers have the model loaded for a
    /// region, as [`Executor::when_ready`] does.
    pub fn when_ready(&self, ready: Option<Ready>) {
        self.runner.when_ready(ready);
    }

    /// What each model region of the last query ran on the workers, in
    /// the order the regions ran; the session keeps none of it.
    pub fn take_regions(&self) -> Vec<RegionRan> {
        self.runner.take_regions()
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
    /// DataFusion refuses the name, or it has
    /// `quail_planner::operators::INPUT_SEPARATOR` in it.
    pub fn register_table(&self, name: &str, table: Arc<dyn TableProvider>) -> Result<()> {
        let separator = quail_planner::operators::INPUT_SEPARATOR;
        if name.contains(separator) {
            return datafusion::common::plan_err!(
                "table name {name:?} has {separator:?}, which Quail keeps for the \
                 inputs of model regions"
            );
        }
        self.context.register_table(name, table)?;
        Ok(())
    }

    /// Register a table whose rows `scan(columns)` returns, the same
    /// rows whenever it is scanned, which `identity` names. `id_column`
    /// names the column whose values are unique.
    ///
    /// # Errors
    ///
    /// DataFusion refuses the name.
    pub fn register_scanned_table(
        &self,
        name: &str,
        schema: SchemaRef,
        scan: Scan,
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
        let table = ScannedTable {
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
    /// rows. Concurrent calls sharing this executor wait their turn.
    ///
    /// # Errors
    ///
    /// The query does not plan, a region or operator fails, or the caller
    /// tries to run nested SQL on the same executor.
    pub fn sql_in_dialect(
        &self,
        query: &str,
        dialect: Option<&str>,
    ) -> Result<(SchemaRef, Vec<RecordBatch>)> {
        let runner = Arc::clone(&self.runner);
        let (sender, mut calls) = unbounded_channel::<RegionCall>();
        // An executor can be shared by sessions, but its workers and
        // caller route belong to one query at a time. Acquire before block_on
        // so waiting callers do not occupy a runtime worker.
        let _caller = runner.enter(sender)?;
        self.runtime().block_on(async {
            let context = self.context.clone();
            let (query, dialect) = (query.to_string(), dialect.map(str::to_string));
            // a worker runs the query, so the operators above a region
            // read its rows while this thread runs the region
            let query = self.runtime().spawn(async move {
                let frame = frame(&context, &query, dialect.as_deref()).await?;
                let (state, plan) = frame.into_parts();
                let frame = match plan {
                    // SET, PREPARE, and related statements update this session,
                    // not a temporary copy of its state.
                    LogicalPlan::Statement(_) => context.execute_logical_plan(plan).await?,
                    // DDL needs DataFusion's execution dispatcher: collecting a
                    // DataFrame containing CreateMemoryTable does not create it.
                    // The per-query state preserves dialect settings and shares
                    // the session's catalog, where created tables and views live.
                    LogicalPlan::Ddl(_) => {
                        SessionContext::new_with_state(state)
                            .execute_logical_plan(plan)
                            .await?
                    }
                    _ => datafusion::dataframe::DataFrame::new(state, plan),
                };
                let schema = quail_schema(frame.logical_plan());
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
            });
            tokio::pin!(query);
            // DataFusion's workers run the operators; this thread runs
            // each model region they hand it
            loop {
                tokio::select! {
                    done = &mut query => break done
                        .map_err(|error| DataFusionError::External(Box::new(error)))
                        .and_then(|done| done),
                    Some(call) = calls.recv() => runner.serve(call),
                }
            }
        })
    }

    /// Return DataFusion's physical plan for a query, with each model
    /// region as one `ModelRegionExec` operator.
    ///
    /// # Errors
    ///
    /// The query does not plan.
    pub fn explain(&self, query: &str, dialect: Option<&str>) -> Result<String> {
        self.runtime().block_on(async {
            let plan = frame(&self.context, query, dialect)
                .await?
                .create_physical_plan()
                .await?;
            let text = datafusion::physical_plan::displayable(plan.as_ref())
                .indent(false)
                .to_string();
            Ok(text.trim_end().to_string())
        })
    }

    /// Return DataFusion's optimized logical plan for a query, with its
    /// model regions claimed.
    ///
    /// # Errors
    ///
    /// The query does not plan.
    pub fn optimized_plan(&self, query: &str, dialect: Option<&str>) -> Result<LogicalPlan> {
        let optimized = self.runtime().block_on(async {
            frame(&self.context, query, dialect)
                .await?
                .into_optimized_plan()
        })?;
        quail_datafusion::claim_model_regions(optimized, &self.context.state())
    }

    /// Return each model region of a query, in plan order.
    ///
    /// # Errors
    ///
    /// The query does not plan.
    pub fn regions(&self, query: &str, dialect: Option<&str>) -> Result<Vec<ModelRegion>> {
        let plan = self.optimized_plan(query, dialect)?;
        let mut regions = Vec::new();
        // a shared input's rows are computed once, however often it is read
        let mut shared = Vec::new();
        plan.apply_with_subqueries(|node| {
            if let LogicalPlan::Extension(extension) = node {
                if let Some(region) = extension.node.as_any().downcast_ref::<ModelRegion>() {
                    regions.push(region.clone());
                }
                if let Some(input) = extension.node.as_any().downcast_ref::<SharedInput>() {
                    if shared.contains(&input.key) {
                        return Ok(TreeNodeRecursion::Jump);
                    }
                    shared.push(input.key);
                }
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
    /// region: (the region as [`region_json`] writes it, its planned
    /// region as JSON when Rust plans it). A region's inputs that call no
    /// model are computed first, so it plans on their real rows.
    ///
    /// # Errors
    ///
    /// The query does not plan, an input fails, or a region does not bind.
    #[allow(clippy::type_complexity)]
    pub fn plans(
        &self,
        query: &str,
        dialect: Option<&str>,
    ) -> Result<Vec<(String, Option<planning::PlannedRegion>)>> {
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
    }
}

/// DataFusion's plan of `query`, read in `dialect` (a DataFusion dialect
/// name, such as "BigQuery") or the session's dialect.
///
/// # Errors
///
/// The dialect is unknown, or the query does not plan.
async fn frame(
    context: &SessionContext,
    query: &str,
    dialect: Option<&str>,
) -> Result<datafusion::dataframe::DataFrame> {
    let mut state = context.state();
    if let Some(name) = dialect {
        let dialect: datafusion::config::Dialect = name.parse()?;
        let options = state.config_mut().options_mut();
        options.sql_parser.dialect = dialect;
        if matches!(dialect, datafusion::config::Dialect::BigQuery) {
            // BigQuery sorts NULLs first in ascending order
            options.sql_parser.default_null_ordering = "nulls_min".to_string();
        }
    }
    let dialect = state.config().options().sql_parser.dialect;
    let statement = match state.sql_to_statement(query, &dialect) {
        // Keep Snowflake's expression semantics, but accept standard syntax
        // such as aggregate FILTER that its parser does not implement.
        Err(original) if matches!(dialect, datafusion::config::Dialect::Snowflake) => state
            .sql_to_statement(query, &datafusion::config::Dialect::Generic)
            .map_err(|_| original)?,
        parsed => parsed?,
    };
    let plan = state.statement_to_plan(statement).await?;
    Ok(datafusion::dataframe::DataFrame::new(state, plan))
}
