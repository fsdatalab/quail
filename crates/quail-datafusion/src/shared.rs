use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use datafusion::arrow::array::{ArrayRef, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, SchemaRef};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{
    DFSchema, DFSchemaRef, DataFusionError, Result, TableReference, internal_err,
};
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execute_stream;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::filter_pushdown::{FilterDescription, FilterPushdownPhase};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    PlanProperties, ReplaceChildrenOptions,
};
use futures::lock::Mutex as AsyncMutex;
use futures::{StreamExt, stream};

pub const ROW_COLUMN: &str = "__quail_row";

pub(crate) fn row_field() -> Field {
    Field::new(ROW_COLUMN, DataType::Int64, false).with_metadata(HashMap::from([(
        "quail.row_identity".to_string(),
        "true".to_string(),
    )]))
}

static NEXT_KEY: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct SharedInput {
    pub key: u64,
    pub input: LogicalPlan,
    pub numbered: bool,
    scope: Option<TableReference>,
    schema: DFSchemaRef,
}

impl SharedInput {
    pub fn try_new(input: LogicalPlan, numbered: bool) -> Result<Self> {
        let key = NEXT_KEY.fetch_add(1, Ordering::Relaxed);
        Self::with_key(key, input, numbered, None)
    }

    fn with_key(
        key: u64,
        input: LogicalPlan,
        numbered: bool,
        scope: Option<TableReference>,
    ) -> Result<Self> {
        let mut schema = input.schema().as_ref().clone();
        if numbered {
            let row = DFSchema::from_unqualified_fields(vec![row_field()].into(), HashMap::new())?;
            schema.merge(&row);
            if schema.fields().len() != input.schema().fields().len() + 1 {
                return internal_err!("the input already has a column {ROW_COLUMN}");
            }
        }
        if let Some(scope) = &scope {
            schema = schema.replace_qualifier(scope.clone());
        }
        Ok(Self {
            key,
            input,
            numbered,
            scope,
            schema: Arc::new(schema),
        })
    }
}

impl PartialEq for SharedInput {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
            && self.numbered == other.numbered
            && self.scope == other.scope
            && self.input == other.input
    }
}

impl Eq for SharedInput {}

impl Hash for SharedInput {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
        self.numbered.hash(state);
        self.scope.hash(state);
    }
}

impl PartialOrd for SharedInput {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (self == other).then_some(std::cmp::Ordering::Equal)
    }
}

impl UserDefinedLogicalNodeCore for SharedInput {
    fn name(&self) -> &str {
        "SharedInput"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SharedInput: {}", self.key)?;
        if self.numbered {
            write!(f, ", numbered")?;
        }
        Ok(())
    }

    fn with_exprs_and_inputs(&self, _exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        let Ok([input]) = <[LogicalPlan; 1]>::try_from(inputs) else {
            return internal_err!("SharedInput reads one input");
        };
        let own = self.input.schema();
        let added = input
            .schema()
            .columns()
            .iter()
            .any(|column| !own.has_column(column));
        let input = if added {
            let kept: Vec<Expr> = input
                .schema()
                .columns()
                .into_iter()
                .filter(|column| own.has_column(column))
                .map(Expr::Column)
                .collect();
            datafusion::logical_expr::LogicalPlanBuilder::from(input)
                .project(kept)?
                .build()?
        } else {
            input
        };
        Self::with_key(self.key, input, self.numbered, self.scope.clone())
    }
}

type Rows = AsyncMutex<Option<(Vec<RecordBatch>, MemoryReservation)>>;

pub(crate) type SharedRows = Arc<Rows>;

#[derive(Debug, Default)]
pub(crate) struct SharedRowsByKey {
    found: Mutex<HashMap<u64, Weak<Rows>>>,
}

impl SharedRowsByKey {
    pub(crate) fn rows(&self, key: u64) -> SharedRows {
        let mut found = self.found.lock().expect("the shared rows lock");
        found.retain(|_, rows| rows.strong_count() > 0);
        if let Some(rows) = found.get(&key).and_then(Weak::upgrade) {
            return rows;
        }
        let rows: SharedRows = Arc::new(AsyncMutex::new(None));
        found.insert(key, Arc::downgrade(&rows));
        rows
    }
}

#[derive(Debug)]
pub struct SharedInputExec {
    input: Arc<dyn ExecutionPlan>,
    numbered: bool,
    rows: SharedRows,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl SharedInputExec {
    pub(crate) fn new(
        input: Arc<dyn ExecutionPlan>,
        numbered: bool,
        rows: SharedRows,
        schema: SchemaRef,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Self {
            input,
            numbered,
            rows,
            schema,
            properties: Arc::new(properties),
        }
    }
}

impl DisplayAs for SharedInputExec {
    fn fmt_as(&self, format: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match format {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "SharedInputExec")?;
                if self.numbered {
                    write!(f, ": numbered")?;
                }
                Ok(())
            }
            DisplayFormatType::TreeRender => write!(f, ""),
        }
    }
}

impl ExecutionPlan for SharedInputExec {
    fn name(&self) -> &'static str {
        "SharedInputExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterDescription> {
        Ok(FilterDescription::all_unsupported(
            &parent_filters,
            &self.children(),
        ))
    }

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let Ok([input]) = <[Arc<dyn ExecutionPlan>; 1]>::try_from(children) else {
            return internal_err!("SharedInputExec reads one input");
        };
        Ok(Arc::new(Self::new(
            input,
            self.numbered,
            Arc::clone(&self.rows),
            Arc::clone(&self.schema),
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
            return internal_err!("SharedInputExec has one partition, got {partition}");
        }
        let input = Arc::clone(&self.input);
        let rows = Arc::clone(&self.rows);
        let numbered = self.numbered;
        let schema = Arc::clone(&self.schema);
        let batches = async move {
            let mut found = rows.lock().await;
            if found.is_none() {
                let reservation =
                    MemoryConsumer::new("SharedInputExec").register(context.memory_pool());
                let mut input = execute_stream(input, context)?;
                let mut batches = Vec::new();
                while let Some(batch) = input.next().await {
                    let batch = batch?;
                    reservation.try_grow(batch.get_array_memory_size())?;
                    batches.push(batch);
                }
                if numbered {
                    batches = with_positions(batches, &schema)?;
                    let held: usize = batches.iter().map(RecordBatch::get_array_memory_size).sum();
                    reservation.try_resize(held)?;
                }
                *found = Some((batches, reservation));
            }
            let batches = found
                .as_ref()
                .map(|(batches, _)| batches.clone())
                .unwrap_or_default();
            Ok::<_, DataFusionError>(stream::iter(batches.into_iter().map(Ok)))
        };
        let batches = futures::TryStreamExt::try_flatten(stream::once(batches));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            batches,
        )))
    }
}

fn with_positions(batches: Vec<RecordBatch>, schema: &SchemaRef) -> Result<Vec<RecordBatch>> {
    let mut next = 0i64;
    batches
        .into_iter()
        .map(|batch| {
            let rows = batch.num_rows() as i64;
            let positions: ArrayRef = Arc::new(Int64Array::from_iter_values(next..next + rows));
            next += rows;
            let mut columns = batch.columns().to_vec();
            columns.push(positions);
            Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
        })
        .collect()
}
