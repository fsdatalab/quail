#[cfg(test)]
use quail_dylib as _;

mod bigquery;
mod executor;
pub mod finished;
mod images;
pub mod planning;
pub mod region_run;
pub mod report;
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
use datafusion::logical_expr::{LogicalPlan, Statement};
use datafusion::prelude::SessionContext;
use futures::StreamExt;
use quail_datafusion::{
    IdentifiedTable, InputRows, ModelRegion, RegionRunner, SharedInput, quail_schema,
};
use tokio::runtime::Runtime;
use tokio::sync::mpsc::unbounded_channel;

use crate::executor::RegionCall;

pub fn runtime() -> std::io::Result<Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(quail_datafusion::PLANNING_STACK_BYTES)
        .build()
}
pub use quail_datafusion as extension;
pub use quail_planner as planner;

pub use crate::images::{ImageSizing, image_size};

pub use crate::executor::{
    EngineRegion, Executor, InputHint, Observer, RegionEngine, RegionEvent, RegionRan, RegionSink,
    ScannedTable, region_json,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineConfig {
    pub model: String,
    pub device: String,
    pub gpus: usize,
    pub backend: String,
}

impl EngineConfig {
    pub fn new(model: impl Into<String>, device: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            device: device.into(),
            gpus: 1,
            backend: "quail".to_string(),
        }
    }
}

pub struct Session {
    context: SessionContext,
    base: Option<SessionContext>,
    runtime: Option<Runtime>,
    runner: Arc<Executor>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").finish_non_exhaustive()
    }
}

pub fn register_table(
    context: &SessionContext,
    name: &str,
    table: Arc<dyn TableProvider>,
) -> Result<()> {
    let separator = quail_planner::operators::INPUT_SEPARATOR;
    if name.contains(separator) {
        return datafusion::common::plan_err!(
            "table name {name:?} has {separator:?}, which Quail keeps for the \
             inputs of model regions"
        );
    }
    context.register_table(name, table)?;
    Ok(())
}

pub fn register_scanned_table(
    context: &SessionContext,
    name: &str,
    table: Arc<dyn TableProvider>,
    id_column: Option<&str>,
    identity: &str,
) -> Result<()> {
    let schema = table.schema();
    let keys: Vec<usize> = id_column
        .and_then(|column| schema.index_of(column).ok())
        .into_iter()
        .collect();
    let constraints = if keys.is_empty() {
        Constraints::default()
    } else {
        Constraints::new_unverified(vec![Constraint::PrimaryKey(keys)])
    };
    let state = context.state();
    let state = datafusion::execution::SessionStateBuilder::new()
        .with_config(state.config().clone())
        .with_runtime_env(Arc::clone(state.runtime_env()))
        .with_default_features()
        .build();
    let table = ScannedTable::new(table, state, constraints);
    register_table(
        context,
        name,
        Arc::new(IdentifiedTable::new(Arc::new(table), identity)),
    )
}

impl Session {
    pub fn new(config: &EngineConfig) -> Result<Self> {
        Self::with_executor(Arc::new(Executor::open(config)?))
    }

    pub fn with_executor(executor: Arc<Executor>) -> Result<Self> {
        let context = quail_datafusion::session_context_with_rules(
            Arc::clone(&executor) as Arc<dyn RegionRunner>,
            executor.rules.clone(),
        )?;
        Ok(Self {
            context,
            base: None,
            runtime: Some(runtime()?),
            runner: executor,
        })
    }

    pub fn with_executor_on(executor: Arc<Executor>, base: SessionContext) -> Result<Self> {
        let state =
            quail_datafusion::QuailExtension::new(Arc::clone(&executor) as Arc<dyn RegionRunner>)
                .with_logical_rules(executor.rules.clone())?
                .register(base.state())?;
        Ok(Self {
            context: SessionContext::new_with_state(state),
            base: Some(base),
            runtime: Some(runtime()?),
            runner: executor,
        })
    }

    pub fn observe(&self, observer: Option<Observer>, answers: bool) {
        self.runner.observe(observer, answers);
    }

    pub fn interrupt(&self) {
        self.runner.interrupt();
    }

    pub fn resume(&self) {
        self.runner.resume();
    }

    pub fn progress(&self) -> Option<serde_json::Value> {
        self.runner.progress()
    }

    pub fn take_regions<T, E>(
        &self,
        each: impl FnMut(RegionRan) -> std::result::Result<T, E>,
    ) -> std::result::Result<Vec<T>, E> {
        self.runner.take_regions(each)
    }

    pub fn discard_regions(&self) {
        self.runner.discard_regions();
    }

    pub fn context(&self) -> &SessionContext {
        &self.context
    }

    pub fn register_table(&self, name: &str, table: Arc<dyn TableProvider>) -> Result<()> {
        register_table(&self.context, name, table)
    }

    pub fn register_scanned_table(
        &self,
        name: &str,
        table: Arc<dyn TableProvider>,
        id_column: Option<&str>,
        identity: &str,
    ) -> Result<()> {
        register_scanned_table(&self.context, name, table, id_column, identity)
    }

    pub fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        Ok(self.sql_in_dialect(query, None)?.1)
    }

    pub fn sql_in_dialect(
        &self,
        query: &str,
        dialect: Option<&str>,
    ) -> Result<(SchemaRef, Vec<RecordBatch>)> {
        let (sender, received) = std::sync::mpsc::channel();
        self.query_run()
            .run(query.to_string(), dialect.map(str::to_string), &sender)?;
        drop(sender);
        let mut schema = None;
        let mut batches = Vec::new();
        for streamed in received {
            match streamed {
                Streamed::Schema(found) => schema = Some(found),
                Streamed::Batch(batch) => batches.push(batch),
                Streamed::Failed(error) => return Err(error),
                Streamed::Done => {}
            }
        }
        let schema = schema
            .ok_or_else(|| DataFusionError::Execution("the query returned no schema".into()))?;
        Ok((schema, batches))
    }

    pub fn sql_stream(&self, query: &str, dialect: Option<&str>) -> Result<RowStream> {
        let (sender, received) = std::sync::mpsc::channel();
        let run = self.query_run();
        let (query, dialect) = (query.to_string(), dialect.map(str::to_string));
        std::thread::Builder::new()
            .name("quail-query".into())
            .stack_size(quail_datafusion::PLANNING_STACK_BYTES)
            .spawn(move || {
                if let Err(error) = run.run(query, dialect, &sender) {
                    let _ = sender.send(Streamed::Failed(error));
                }
            })?;
        match received.recv() {
            Ok(Streamed::Schema(schema)) => Ok(RowStream {
                schema,
                received,
                ended: false,
            }),
            Ok(Streamed::Failed(error)) => Err(error),
            _ => Err(DataFusionError::Execution(
                "the query ended before its schema".into(),
            )),
        }
    }

    fn query_run(&self) -> QueryRun {
        QueryRun {
            context: self.context.clone(),
            base: self.base.clone(),
            runtime: self.runtime().handle().clone(),
            runner: Arc::clone(&self.runner),
        }
    }

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

    pub fn optimized_plan(&self, query: &str, dialect: Option<&str>) -> Result<LogicalPlan> {
        let optimized = self.runtime().block_on(async {
            frame(&self.context, query, dialect)
                .await?
                .into_optimized_plan()
        })?;
        quail_datafusion::claim_model_regions(optimized, &self.context.state())
    }

    pub fn regions(&self, query: &str, dialect: Option<&str>) -> Result<Vec<ModelRegion>> {
        let plan = self.optimized_plan(query, dialect)?;
        let mut regions = Vec::new();
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
    pub fn explain_regions(
        &self,
        query: &str,
        dialect: Option<&str>,
        verbose: bool,
    ) -> Result<Vec<String>> {
        let read = |error: serde_json::Error| DataFusionError::External(Box::new(error));
        self.plans(query, dialect)?
            .into_iter()
            .map(|(region, planned)| {
                let Some((planned, _)) = planned else {
                    return Ok(region);
                };
                let planned: serde_json::Value = serde_json::from_str(&planned).map_err(read)?;
                let request = self.runner.explain_request(&planned, verbose)?;
                let request = serde_json::from_value(request).map_err(read)?;
                quail_planner::plan::explain::explain_text(&request)
                    .map_err(DataFusionError::Execution)
            })
            .collect()
    }

    pub fn explain_analyze(
        &self,
        query: &str,
        dialect: Option<&str>,
        verbose: bool,
    ) -> Result<(SchemaRef, Vec<RecordBatch>, Vec<String>)> {
        let read = |error: serde_json::Error| DataFusionError::External(Box::new(error));
        let (schema, batches) = self.sql_in_dialect(query, dialect)?;
        let explained = self.take_regions(|region| {
            let request = self.runner.analyzed_request(&region, verbose)?;
            let request = serde_json::from_value(request).map_err(read)?;
            quail_planner::plan::explain::explain_text(&request).map_err(DataFusionError::Execution)
        })?;
        Ok((schema, batches, explained))
    }

    pub fn region_report(
        &self,
        region: &RegionRan,
    ) -> Result<(serde_json::Value, Vec<shared::Table>)> {
        self.runner.region_report(region)
    }

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
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

enum Streamed {
    Schema(SchemaRef),
    Batch(RecordBatch),
    Failed(DataFusionError),
    Done,
}

pub struct RowStream {
    schema: SchemaRef,
    received: std::sync::mpsc::Receiver<Streamed>,
    ended: bool,
}

impl std::fmt::Debug for RowStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RowStream").finish_non_exhaustive()
    }
}

impl RowStream {
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl Iterator for RowStream {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ended {
            return None;
        }
        let found = match self.received.recv() {
            Ok(Streamed::Batch(batch)) => return Some(Ok(batch)),
            Ok(Streamed::Failed(error)) => Some(Err(error)),
            Ok(Streamed::Done) => None,
            Ok(Streamed::Schema(_)) => Some(Err(DataFusionError::Internal(
                "a query sent its schema twice".into(),
            ))),
            Err(_) => Some(Err(DataFusionError::Execution(
                "the query stopped before its end".into(),
            ))),
        };
        self.ended = true;
        found
    }
}

struct QueryRun {
    context: SessionContext,
    base: Option<SessionContext>,
    runtime: tokio::runtime::Handle,
    runner: Arc<Executor>,
}

impl QueryRun {
    fn run(
        self,
        query: String,
        dialect: Option<String>,
        sent: &std::sync::mpsc::Sender<Streamed>,
    ) -> Result<()> {
        let Self {
            context,
            base,
            runtime,
            runner,
        } = self;
        let (sender, mut calls) = unbounded_channel::<RegionCall>();
        let _caller = runner.enter(sender)?;
        let mut stopped = runner.stopped();
        let mut failed = None;
        let out = sent.clone();
        runtime.block_on(async {
            let query = runtime.spawn(async move {
                let frame = frame(&context, &query, dialect.as_deref()).await?;
                let (state, plan) = frame.into_parts();
                let frame = match plan {
                    LogicalPlan::Statement(statement) => {
                        if let Some(base) = &base
                            && matches!(
                                statement,
                                Statement::SetVariable(_) | Statement::ResetVariable(_)
                            )
                        {
                            base.execute_logical_plan(LogicalPlan::Statement(statement.clone()))
                                .await?;
                        }
                        context
                            .execute_logical_plan(LogicalPlan::Statement(statement))
                            .await?
                    }
                    LogicalPlan::Ddl(_) => {
                        SessionContext::new_with_state(state)
                            .execute_logical_plan(plan)
                            .await?
                    }
                    _ => datafusion::dataframe::DataFrame::new(state, plan),
                };
                let dialect = frame
                    .task_ctx()
                    .session_config()
                    .options()
                    .sql_parser
                    .dialect;
                let schema = quail_schema(frame.logical_plan(), &query, dialect);
                let mut batches = frame.execute_stream().await?;
                if out.send(Streamed::Schema(Arc::clone(&schema))).is_err() {
                    return Ok(());
                }
                while let Some(batch) = batches.next().await {
                    let batch =
                        RecordBatch::try_new(Arc::clone(&schema), batch?.columns().to_vec())?;
                    if out.send(Streamed::Batch(batch)).is_err() {
                        return Ok(());
                    }
                }
                Ok::<_, DataFusionError>(())
            });
            tokio::pin!(query);
            loop {
                tokio::select! {
                    biased;
                    done = &mut query => break done
                        .map_err(|error| DataFusionError::External(Box::new(error)))
                        .and_then(|done| done),
                    Some(call) = calls.recv() => match &failed {
                        None => failed = runner.serve(call).err(),
                        Some(error) => call.refuse(error),
                    },
                    Ok(_) = stopped.wait_for(|stopped| *stopped) => {
                        query.abort();
                        break Err(DataFusionError::Execution("the query was stopped".into()));
                    }
                }
            }
        })?;
        let _ = sent.send(Streamed::Done);
        Ok(())
    }
}

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
            options.sql_parser.default_null_ordering = "nulls_min".to_string();
            options.sql_parser.enable_ident_normalization = false;
        }
    }
    let dialect = state.config().options().sql_parser.dialect;
    let mut statement = match state.sql_to_statement(query, &dialect) {
        Err(original) if matches!(dialect, datafusion::config::Dialect::Snowflake) => state
            .sql_to_statement(query, &datafusion::config::Dialect::Generic)
            .map_err(|_| original)?,
        parsed => parsed?,
    };
    if matches!(dialect, datafusion::config::Dialect::BigQuery) {
        bigquery::rewrite(&mut statement);
    }
    let plan = state.statement_to_plan(statement).await?;
    Ok(datafusion::dataframe::DataFrame::new(state, plan))
}
