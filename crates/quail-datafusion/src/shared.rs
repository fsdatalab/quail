//! `SharedInput`: a region input that the region and the join above it
//! read once.
//!
//! A model region returns the id column of each table it reads and its
//! answers; DataFusion joins each table back on its id above the region
//! ([`crate::join_back`]). When DataFusion computes a table's rows for
//! the region, such as a table's plain filters or a CTE that groups
//! rows, the region and the join read the same `SharedInput`: its rows
//! are computed once, by whichever reads first, and both read those
//! rows. An input without an id column gets one, [`ROW_COLUMN`], each
//! row's position among those rows.

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use datafusion::arrow::array::{ArrayRef, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, SchemaRef};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DFSchema, DFSchemaRef, DataFusionError, Result, internal_err};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::filter_pushdown::{FilterDescription, FilterPushdownPhase};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    PlanProperties, ReplaceChildrenOptions,
};
use futures::lock::Mutex as AsyncMutex;
use futures::stream;

/// The id column of a region input without one: each row's position
/// among the input's rows, from zero.
pub const ROW_COLUMN: &str = "__quail_row";

/// The field of [`ROW_COLUMN`].
pub(crate) fn row_field() -> Field {
    Field::new(ROW_COLUMN, DataType::Int64, false)
}

static NEXT_KEY: AtomicU64 = AtomicU64::new(0);

/// A region input whose rows every reader of the same `key` shares.
#[derive(Debug, Clone)]
pub struct SharedInput {
    /// Names the rows: each `SharedInput` of one key reads the same rows.
    pub key: u64,
    pub input: LogicalPlan,
    /// Whether the rows get [`ROW_COLUMN`].
    pub numbered: bool,
    schema: DFSchemaRef,
}

impl SharedInput {
    /// A new shared input over `input`, numbering its rows when
    /// `numbered`.
    ///
    /// # Errors
    ///
    /// The input already has a column named [`ROW_COLUMN`].
    pub fn try_new(input: LogicalPlan, numbered: bool) -> Result<Self> {
        let key = NEXT_KEY.fetch_add(1, Ordering::Relaxed);
        Self::with_key(key, input, numbered)
    }

    fn with_key(key: u64, input: LogicalPlan, numbered: bool) -> Result<Self> {
        let mut schema = input.schema().as_ref().clone();
        if numbered {
            let row = DFSchema::from_unqualified_fields(vec![row_field()].into(), HashMap::new())?;
            schema.merge(&row);
            if schema.fields().len() != input.schema().fields().len() + 1 {
                return internal_err!("the input already has a column {ROW_COLUMN}");
            }
        }
        Ok(Self {
            key,
            input,
            numbered,
            schema: Arc::new(schema),
        })
    }
}

impl PartialEq for SharedInput {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.numbered == other.numbered && self.input == other.input
    }
}

impl Eq for SharedInput {}

impl Hash for SharedInput {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
        self.numbered.hash(state);
    }
}

impl PartialOrd for SharedInput {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.key.cmp(&other.key))
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
        Self::with_key(self.key, input, self.numbered)
    }
}

/// The rows of one shared input, once its first reader has computed them.
type Rows = AsyncMutex<Option<Vec<RecordBatch>>>;

/// The rows of one shared input, as each of its readers holds them.
pub(crate) type SharedRows = Arc<Rows>;

/// The rows of each shared input a planner has planned, by key, while a
/// reader holds them.
#[derive(Debug, Default)]
pub(crate) struct SharedRowsByKey {
    found: Mutex<HashMap<u64, Weak<Rows>>>,
}

impl SharedRowsByKey {
    /// The rows of the shared input `key`, shared with every other
    /// reader of it still planned.
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

/// The physical operator of a [`SharedInput`]: one partition of the
/// input's rows, computed by the first of its readers.
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
        // every reader gets the rows the first computes, so no reader's
        // filter, such as a join's dynamic filter, may reach the input
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
                let batches = datafusion::physical_plan::collect(input, context).await?;
                *found = Some(if numbered {
                    with_positions(batches, &schema)?
                } else {
                    batches
                });
            }
            let batches = found.clone().unwrap_or_default();
            Ok::<_, DataFusionError>(stream::iter(batches.into_iter().map(Ok)))
        };
        let batches = futures::TryStreamExt::try_flatten(stream::once(batches));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            batches,
        )))
    }
}

/// `batches` with [`ROW_COLUMN`] added: each row's position among them.
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
