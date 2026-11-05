//! `ModelRegionExec` and the planners that build it.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::Session;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DataFusionError, Result, internal_err};
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    PlanProperties, ReplaceChildrenOptions,
};
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use futures::stream;

use crate::region::ModelRegion;

/// The rows of one region input: its table name and its batches.
pub type InputRows = (String, Vec<RecordBatch>);

/// Plans and runs one model region, returning its rows.
///
/// Quail's executor implements this trait. The call blocks until the
/// region's rows are ready.
pub trait RegionRunner: fmt::Debug + Send + Sync {
    /// Plan and run the region over its inputs' rows, returning rows in
    /// the region's schema.
    fn run(&self, region: &ModelRegion, inputs: &[InputRows]) -> Result<Vec<RecordBatch>>;
}

/// The physical operator for one model region.
#[derive(Debug)]
pub struct ModelRegionExec {
    region: ModelRegion,
    runner: Arc<dyn RegionRunner>,
    /// The physical plans of the region's inputs, in order.
    inputs: Vec<Arc<dyn ExecutionPlan>>,
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
        let schema = Arc::clone(region.bound.schema().inner());
        let properties = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Self {
            region,
            runner,
            inputs,
            schema,
            properties: Arc::new(properties),
        }
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
                write!(f, "ModelRegionExec")
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

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            self.region.clone(),
            Arc::clone(&self.runner),
            children,
        )))
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
        // the region runs on the thread that polls the stream: under a
        // current-thread runtime, the thread that called DataFusion, so a
        // runner's per-thread state (Python's, the GPU process's) is reused
        let rows = async move {
            let mut rows: Vec<InputRows> = Vec::new();
            for (name, input) in region.input_names.iter().zip(inputs) {
                let batches =
                    datafusion::physical_plan::collect(input, Arc::clone(&context)).await?;
                rows.push((name.clone(), batches));
            }
            let batches = runner.run(&region, &rows)?;
            Ok::<_, DataFusionError>(stream::iter(batches.into_iter().map(Ok)))
        };
        let batches = futures::TryStreamExt::try_flatten(stream::once(rows));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            batches,
        )))
    }
}

/// The `ExtensionPlanner` that turns a `ModelRegion` into a
/// `ModelRegionExec`.
#[derive(Debug)]
pub struct ModelRegionPlanner {
    runner: Arc<dyn RegionRunner>,
}

impl ModelRegionPlanner {
    /// Create the planner; every region it plans runs on `runner`.
    pub fn new(runner: Arc<dyn RegionRunner>) -> Self {
        Self { runner }
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
