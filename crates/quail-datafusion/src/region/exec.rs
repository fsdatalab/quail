use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::Session;
use datafusion::common::config::ConfigOptions;
use datafusion::common::runtime::SpawnedTask;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{DataFusionError, Result, exec_err, internal_err};
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion::physical_expr::utils::collect_columns;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::distribution_requirements::InputDistributionRequirements;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType, EvaluationType};
use datafusion::physical_plan::filter_pushdown::{
    ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation, PushedDown,
};
use datafusion::physical_plan::limit::LimitStream;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::{
    RecordBatchReceiverStreamBuilder, RecordBatchStreamAdapter,
};
use datafusion::physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, Distribution, ExecutionPlan,
    Partitioning, PlanProperties, ReplaceChildrenOptions,
};
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use futures::SinkExt;
use futures::channel::oneshot;
use futures::{StreamExt, stream};

use crate::region::ModelRegion;
use crate::region::region_hint::RegionHint;
use crate::shared::{SharedInput, SharedInputExec, SharedRowsByKey};

pub type InputRows = (String, Vec<RecordBatch>);

pub type RegionBatches = Box<dyn Iterator<Item = Result<RecordBatch>> + Send>;

pub type InputBatches = Box<dyn Iterator<Item = Result<RecordBatch>> + Send>;

pub type InputStream = (String, InputBatches);

pub fn collect_inputs(inputs: Vec<InputStream>) -> Result<Vec<InputRows>> {
    inputs
        .into_iter()
        .map(|(name, batches)| Ok((name, batches.collect::<Result<Vec<_>>>()?)))
        .collect()
}

pub trait RegionRunner: fmt::Debug + Send + Sync {
    fn run(
        &self,
        region: &ModelRegion,
        inputs: Vec<InputStream>,
        fetch: Option<usize>,
        hint: &RegionHint,
    ) -> Result<RegionBatches>;
}

#[derive(Debug)]
pub struct ModelRegionExec {
    region: ModelRegion,
    runner: Arc<dyn RegionRunner>,
    inputs: Vec<Arc<dyn ExecutionPlan>>,
    fetch: Option<usize>,
    hint: RegionHint,
    filter_columns: Option<BTreeSet<String>>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl ModelRegionExec {
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
        )
        .with_evaluation_type(EvaluationType::Eager);
        Self {
            region,
            runner,
            inputs,
            fetch: None,
            hint: RegionHint::default(),
            filter_columns: None,
            schema,
            properties: Arc::new(properties),
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    fn with_inputs(&self, inputs: Vec<Arc<dyn ExecutionPlan>>) -> Self {
        let mut exec = Self::new(self.region.clone(), Arc::clone(&self.runner), inputs);
        exec.fetch = self.fetch;
        exec.hint = self.hint.clone();
        exec.filter_columns = self.filter_columns.clone();
        exec
    }

    pub(crate) fn with_filter_columns(
        &self,
        columns: BTreeSet<String>,
        inputs: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Self {
        let mut exec = self.with_inputs(inputs);
        exec.filter_columns = Some(columns);
        exec
    }

    pub fn region(&self) -> &ModelRegion {
        &self.region
    }
}

impl DisplayAs for ModelRegionExec {
    fn fmt_as(&self, format: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match format {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "ModelRegionExec: ")?;
                let mut details = Vec::new();
                if let Some(fetch) = self.fetch {
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
                details.push(self.region.summary());
                write!(f, "{}", details.join(", "))
            }
            DisplayFormatType::TreeRender => write!(f, "{}", self.region.summary()),
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

    fn input_distribution_requirements(&self) -> InputDistributionRequirements {
        InputDistributionRequirements::new(vec![Distribution::SinglePartition; self.inputs.len()])
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
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
        let filters = vec![PushedDown::No; child_pushdown_result.parent_filters.len()];
        let dynamic: Vec<Arc<dyn PhysicalExpr>> = child_pushdown_result
            .parent_filters
            .into_iter()
            .map(|found| found.filter)
            .filter(|test| {
                test.exists(|expr| Ok(expr.downcast_ref::<DynamicFilterPhysicalExpr>().is_some()))
                    .unwrap_or(false)
            })
            .filter(|test| {
                self.filter_columns.as_ref().is_some_and(|safe| {
                    collect_columns(test)
                        .iter()
                        .all(|column| safe.contains(column.name()))
                })
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
        let schema = Arc::clone(&self.schema);
        let start = move || {
            let mut builder = RecordBatchReceiverStreamBuilder::new(schema, BUFFERED_BATCHES);
            let mut streams: Vec<InputStream> = Vec::new();
            let mut first = Vec::new();
            for (name, input) in region.input_names.iter().zip(inputs) {
                let (mut batches, received) =
                    futures::channel::mpsc::channel::<Result<RecordBatch>>(BUFFERED_BATCHES);
                let (arrived, waiting) = oneshot::channel::<()>();
                let context = Arc::clone(&context);
                let ended = Arc::new(AtomicBool::new(false));
                let finished = Arc::clone(&ended);
                builder.spawn(async move {
                    let mut arrived = Some(arrived);
                    let mut found = match input.execute(0, context) {
                        Ok(found) => found,
                        Err(error) => {
                            let _ = batches.send(Err(error)).await;
                            return Ok(());
                        }
                    };
                    while let Some(batch) = found.next().await {
                        if batches.send(batch).await.is_err() {
                            break;
                        }
                        if let Some(arrived) = arrived.take() {
                            let _ = arrived.send(());
                        }
                    }
                    finished.store(true, Ordering::Release);
                    Ok(())
                });
                streams.push((name.clone(), Box::new(BlockingReceiver { received, ended })));
                first.push(waiting);
            }
            let sender = builder.tx();
            builder.spawn(async move {
                for waiting in first {
                    let _ = waiting.await;
                }
                SpawnedTask::spawn_blocking(move || {
                    let batches = runner.run(&region, streams, fetch, &hint)?;
                    for batch in batches {
                        if sender
                            .blocking_send(batch.and_then(with_struct_nulls))
                            .is_err()
                        {
                            break;
                        }
                    }
                    Ok(())
                })
                .join_unwind()
                .await
                .map_err(|error| DataFusionError::ExecutionJoin(Box::new(error)))?
            });
            builder.build()
        };
        let batches = stream::once(async move { start() }).flatten();
        let batches = Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            batches,
        ));
        Ok(Box::pin(LimitStream::new(
            batches,
            0,
            self.fetch,
            BaselineMetrics::new(&self.metrics, partition),
        )))
    }
}

const BUFFERED_BATCHES: usize = 2;

fn with_struct_nulls(batch: RecordBatch) -> Result<RecordBatch> {
    use datafusion::arrow::array::{Array, ArrayRef, StructArray, make_array};
    use datafusion::arrow::buffer::NullBuffer;

    fn normalize(array: &ArrayRef) -> Result<ArrayRef> {
        let Some(values) = array.as_any().downcast_ref::<StructArray>() else {
            return Ok(Arc::clone(array));
        };
        let columns = values
            .columns()
            .iter()
            .map(|column| {
                let nulls = NullBuffer::union(values.nulls(), column.nulls());
                let data = column.to_data().into_builder().nulls(nulls).build()?;
                normalize(&make_array(data))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(StructArray::try_new(
            values.fields().clone(),
            columns,
            values.nulls().cloned(),
        )?))
    }

    RecordBatch::try_new(
        batch.schema(),
        batch
            .columns()
            .iter()
            .map(normalize)
            .collect::<Result<_>>()?,
    )
    .map_err(Into::into)
}

struct BlockingReceiver {
    received: futures::channel::mpsc::Receiver<Result<RecordBatch>>,
    ended: Arc<AtomicBool>,
}

impl Iterator for BlockingReceiver {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        match futures::executor::block_on(self.received.next()) {
            None if !self.ended.swap(true, Ordering::AcqRel) => Some(exec_err!(
                "the query was cancelled while a model region read its input"
            )),
            found => found,
        }
    }
}

#[derive(Debug)]
pub struct ModelRegionPlanner {
    runner: Arc<dyn RegionRunner>,
    shared: SharedRowsByKey,
}

impl ModelRegionPlanner {
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

#[derive(Debug)]
pub struct QuailQueryPlanner {
    runner: Arc<dyn RegionRunner>,
}

impl QuailQueryPlanner {
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
        let claimed = crate::claim_model_regions(logical_plan.clone(), session)?;
        planner.create_physical_plan(&claimed, session).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Array, Int32Array, StructArray};
    use datafusion::arrow::buffer::NullBuffer;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};

    #[test]
    fn null_structs_do_not_expose_child_storage_to_sql() -> Result<()> {
        let span = StructArray::new(
            vec![Arc::new(Field::new("start", DataType::Int32, true))].into(),
            vec![Arc::new(Int32Array::from(vec![0]))],
            Some(NullBuffer::from(vec![false])),
        );
        let schema = Arc::new(Schema::new(vec![Field::new(
            "span",
            span.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(span)])?;
        let normalized = with_struct_nulls(batch)?;
        let span = normalized
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert!(span.is_null(0));
        assert!(span.column(0).is_null(0));
        Ok(())
    }
}
