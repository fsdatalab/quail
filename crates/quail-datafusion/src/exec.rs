//! `ModelRegionExec` and the planners that build it and `SharedInputExec`.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::Session;
use datafusion::common::config::ConfigOptions;
use datafusion::common::runtime::SpawnedTask;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{Result, internal_err};
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execute_stream;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::filter_pushdown::{
    ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation, PushedDown,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    PlanProperties, ReplaceChildrenOptions,
};
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use futures::channel::{mpsc, oneshot};
use futures::{StreamExt, stream};

use crate::region::ModelRegion;
use crate::region_hint::RegionHint;
use crate::shared::{SharedInput, SharedInputExec, SharedRowsByKey};

/// The rows of one region input: its table name and its batches.
pub type InputRows = (String, Vec<RecordBatch>);

/// A region's rows, in the region's schema, as the runner produces them.
pub type RegionBatches = Box<dyn Iterator<Item = Result<RecordBatch>> + Send>;

/// One region input's batches, as DataFusion computes them: `recv`
/// waits for the next batch, `try_recv` does not, and both fail once the
/// input has ended.
pub type InputBatches = std::sync::mpsc::Receiver<Result<RecordBatch>>;

/// One region input: its table name and its batches as they arrive.
pub type InputStream = (String, InputBatches);

/// Plans and runs one model region, returning its rows as they come.
///
/// Quail's executor implements this trait, in the same process or
/// behind Quail Server. `ModelRegionExec` calls it on a thread that may
/// block, and hands DataFusion each batch as the iterator yields it, so
/// the operators above the region read rows while the region runs.
pub trait RegionRunner: fmt::Debug + Send + Sync {
    /// Plan and run the region over its inputs' rows, which arrive as
    /// DataFusion computes them. Each input holds at least one batch, or
    /// has ended, when the call starts. `fetch` is the most rows
    /// DataFusion reads, when a LIMIT above the region says so; a
    /// runner may stop the model once it has that many. The region may
    /// drop each row where a test of `hint` is not TRUE.
    fn run(
        &self,
        region: &ModelRegion,
        inputs: Vec<InputStream>,
        fetch: Option<usize>,
        hint: &RegionHint,
    ) -> Result<RegionBatches>;
}

/// The physical operator for one model region.
#[derive(Debug)]
pub struct ModelRegionExec {
    region: ModelRegion,
    runner: Arc<dyn RegionRunner>,
    /// The physical plans of the region's inputs, in order.
    inputs: Vec<Arc<dyn ExecutionPlan>>,
    /// The most rows DataFusion reads, from its limit pushdown.
    fetch: Option<usize>,
    /// The dynamic filters the operators above handed the region.
    hint: RegionHint,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl ModelRegionExec {
    /// Create the operator for a region, run by the given runner over the
    /// rows of `inputs`, the physical plans of the region's inputs.
    pub fn new(
        region: ModelRegion,
        runner: Arc<dyn RegionRunner>,
        inputs: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Self {
        let schema = Arc::clone(region.plan.schema().inner());
        let properties = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            region,
            runner,
            inputs,
            fetch: None,
            hint: RegionHint::default(),
            schema,
            properties: Arc::new(properties),
        }
    }

    /// The operator with the same region, hint, and fetch over other
    /// inputs.
    fn with_inputs(&self, inputs: Vec<Arc<dyn ExecutionPlan>>) -> Self {
        let mut exec = Self::new(self.region.clone(), Arc::clone(&self.runner), inputs);
        exec.fetch = self.fetch;
        exec.hint = self.hint.clone();
        exec
    }

    /// The region this operator runs.
    pub fn region(&self) -> &ModelRegion {
        &self.region
    }
}

impl DisplayAs for ModelRegionExec {
    fn fmt_as(&self, format: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match format {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "ModelRegionExec")?;
                let mut details = Vec::new();
                if let Some(fetch) = self.fetch.or(self.region.fetch) {
                    details.push(format!("fetch={fetch}"));
                }
                if !self.hint.is_empty() {
                    let tests: Vec<String> = self
                        .hint
                        .tests
                        .iter()
                        .map(|test| test.to_string())
                        .collect();
                    details.push(format!("hint=[{}]", tests.join(", ")));
                }
                if details.is_empty() {
                    return Ok(());
                }
                write!(f, ": {}", details.join(", "))
            }
            DisplayFormatType::TreeRender => write!(f, ""),
        }
    }
}

impl ExecutionPlan for ModelRegionExec {
    fn name(&self) -> &'static str {
        "ModelRegionExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inputs.iter().collect()
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        // `execute` reads each input as one stream
        vec![false; self.inputs.len()]
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        let mut exec = self.with_inputs(self.inputs.clone());
        exec.fetch = limit;
        Some(Arc::new(exec))
    }

    fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        // the operator that made a dynamic filter fills it only when an
        // operator below holds it
        for test in &self.hint.tests {
            if f(test)? == TreeNodeRecursion::Stop {
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    }

    fn handle_child_pushdown_result(
        &self,
        phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        // the operators above still apply every filter: the region only
        // may drop the rows a filter rejects
        let filters = vec![PushedDown::No; child_pushdown_result.parent_filters.len()];
        let dynamic: Vec<Arc<dyn PhysicalExpr>> = child_pushdown_result
            .parent_filters
            .into_iter()
            .map(|found| found.filter)
            .filter(|test| {
                test.exists(|expr| Ok(expr.downcast_ref::<DynamicFilterPhysicalExpr>().is_some()))
                    .unwrap_or(false)
            })
            .collect();
        let propagation = FilterPushdownPropagation::with_parent_pushdown_result(filters);
        if phase != FilterPushdownPhase::Post || dynamic.is_empty() {
            return Ok(propagation);
        }
        let mut exec = self.with_inputs(self.inputs.clone());
        exec.hint.tests.extend(dynamic);
        Ok(propagation.with_updated_node(Arc::new(exec)))
    }

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(self.with_inputs(children)))
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!("ModelRegionExec has one partition, got {partition}");
        }
        let region = self.region.clone();
        let runner = Arc::clone(&self.runner);
        let inputs = self.inputs.clone();
        let fetch = self.fetch;
        let hint = self.hint.clone();
        let (sender, receiver) = mpsc::unbounded::<Result<RecordBatch>>();
        // the region starts when DataFusion first reads it: a hash join
        // reads its probe side once its build side, and so its filter, is
        // complete
        let start = move || {
            SpawnedTask::spawn(async move {
                // each input's batches go to a channel the runner reads; the
                // tasks live as long as this one
                let mut readers = Vec::new();
                let mut streams: Vec<InputStream> = Vec::new();
                let mut first = Vec::new();
                for (name, input) in region.input_names.iter().zip(inputs) {
                    let (batches, received) = std::sync::mpsc::channel();
                    let (arrived, waiting) = oneshot::channel::<()>();
                    let context = Arc::clone(&context);
                    readers.push(SpawnedTask::spawn(async move {
                        let mut arrived = Some(arrived);
                        match execute_stream(input, context) {
                            Ok(mut found) => {
                                while let Some(batch) = found.next().await {
                                    if batches.send(batch).is_err() {
                                        break;
                                    }
                                    if let Some(arrived) = arrived.take() {
                                        let _ = arrived.send(());
                                    }
                                }
                            }
                            Err(error) => {
                                let _ = batches.send(Err(error));
                            }
                        }
                    }));
                    streams.push((name.clone(), received));
                    first.push(waiting);
                }
                // a region starts once each input has a batch or has ended: an
                // input that is another region's rows then runs first, on the
                // same thread
                for waiting in first {
                    let _ = waiting.await;
                }
                // the runner blocks while the region runs; Quail's hands the
                // region to the thread that called DataFusion
                let run = SpawnedTask::spawn_blocking(move || {
                    let batches = match runner.run(&region, streams, fetch, &hint) {
                        Ok(batches) => batches,
                        Err(error) => {
                            let _ = sender.unbounded_send(Err(error));
                            return;
                        }
                    };
                    for batch in (Limited {
                        batches,
                        remaining: fetch,
                    }) {
                        // the receiver is gone once DataFusion stops reading
                        if sender.unbounded_send(batch).is_err() {
                            return;
                        }
                    }
                });
                let _ = run.join().await;
                drop(readers);
            })
        };
        // the task lives as long as the stream: dropping the stream ends it
        let batches =
            stream::once(async move { (receiver, start()) }).flat_map(|(receiver, task)| {
                stream::unfold((receiver, task), |(mut receiver, task)| async move {
                    let batch = receiver.next().await?;
                    Some((batch, (receiver, task)))
                })
            });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            batches,
        )))
    }
}

/// A region's batches cut off after `remaining` rows, when DataFusion
/// pushed a limit into the region.
struct Limited {
    batches: RegionBatches,
    remaining: Option<usize>,
}

impl Iterator for Limited {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == Some(0) {
            return None;
        }
        let batch = match self.batches.next()? {
            Ok(batch) => batch,
            Err(error) => return Some(Err(error)),
        };
        Some(Ok(match &mut self.remaining {
            Some(remaining) => {
                let rows = batch.num_rows().min(*remaining);
                *remaining -= rows;
                batch.slice(0, rows)
            }
            None => batch,
        }))
    }
}

/// The `ExtensionPlanner` that turns a `ModelRegion` into a
/// `ModelRegionExec`, and a `SharedInput` into a `SharedInputExec` that
/// shares its rows with the other readers of its key.
#[derive(Debug)]
pub struct ModelRegionPlanner {
    runner: Arc<dyn RegionRunner>,
    shared: SharedRowsByKey,
}

impl ModelRegionPlanner {
    /// Create the planner; every region it plans runs on `runner`.
    pub fn new(runner: Arc<dyn RegionRunner>) -> Self {
        Self {
            runner,
            shared: SharedRowsByKey::default(),
        }
    }
}

#[async_trait]
impl ExtensionPlanner for ModelRegionPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session: &dyn Session,
        _planning_ctx: &PhysicalPlanningContext,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if let Some(shared) = node.as_any().downcast_ref::<SharedInput>() {
            let Some(input) = physical_inputs.first() else {
                return internal_err!("SharedInput reads one input");
            };
            return Ok(Some(Arc::new(SharedInputExec::new(
                Arc::clone(input),
                shared.numbered,
                self.shared.rows(shared.key),
                Arc::new(node.schema().as_arrow().clone()),
            ))));
        }
        let Some(region) = node.as_any().downcast_ref::<ModelRegion>() else {
            return Ok(None);
        };
        Ok(Some(Arc::new(ModelRegionExec::new(
            region.clone(),
            Arc::clone(&self.runner),
            physical_inputs.to_vec(),
        ))))
    }
}

/// DataFusion's default query planner, with `ModelRegionPlanner` added.
#[derive(Debug)]
pub struct QuailQueryPlanner {
    runner: Arc<dyn RegionRunner>,
}

impl QuailQueryPlanner {
    /// Create the query planner; every region runs on `runner`.
    pub fn new(runner: Arc<dyn RegionRunner>) -> Self {
        Self { runner }
    }
}

#[async_trait]
impl QueryPlanner for QuailQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let planner = DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(
            ModelRegionPlanner::new(Arc::clone(&self.runner)),
        )]);
        planner.create_physical_plan(logical_plan, session).await
    }
}
