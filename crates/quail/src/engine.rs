//! The model region runner that calls Quail's Python executor, and the
//! tables whose rows a Python provider returns.

use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use async_trait::async_trait;
use datafusion::arrow::array::{ArrayRef, RecordBatch};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{Constraints, DataFusionError, Result};
use datafusion::datasource::TableType;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use std::collections::HashMap;

use quail_datafusion::{
    IdentifiedTable, InputRows, ModelRegion, RegionBatches, RegionRunner, bind, quail_column_names,
    region_sql,
};
use quail_planner::prompts::Turn;
use tokio::sync::mpsc::UnboundedSender;

use crate::planning::{DOCUMENT_EXPRESSION, Planner, Settings, TableRows, planned_json};
use crate::pyarrow::{from_pyarrow, stream_from_pyarrow, to_pyarrow};

/// Runs each model region by calling the engine's
/// `run_region(kind, payload, tables, fetch)`, which returns the rows
/// as a pyarrow Table or RecordBatchReader.
#[derive(Debug)]
pub(crate) struct PythonRunner {
    pub(crate) engine: Py<PyAny>,
    /// The model's chat-turn text and prompt layout, to bind regions in
    /// Rust; None sends each region's SQL instead.
    pub(crate) prompting: Option<(Turn, String)>,
    /// While a query runs: the thread that called DataFusion, and where
    /// a worker sends it a region to run.
    pub(crate) caller: Mutex<Option<(ThreadId, UnboundedSender<RegionCall>)>>,
    /// The planner and configuration Rust plans each region with, then
    /// calls the engine's `execute`; None calls `run_region`.
    pub(crate) planning: Option<(Arc<Planner>, Settings)>,
}

/// A region a DataFusion worker hands to the calling thread to run.
#[derive(Debug)]
pub(crate) struct RegionCall {
    pub(crate) region: ModelRegion,
    pub(crate) inputs: Vec<InputRows>,
    pub(crate) fetch: Option<usize>,
    pub(crate) rows: std::sync::mpsc::Sender<Result<RegionBatches>>,
}

impl PythonRunner {
    /// The request for one region: ("plan", its logical plan as JSON),
    /// or ("sql", its SQL) for a region the Rust binder does not read.
    fn request(&self, region: &ModelRegion) -> Result<(&'static str, String)> {
        let Some((turn, layout)) = &self.prompting else {
            return Ok(("sql", region_sql(region)?));
        };
        match bind::bind_model_region(region, turn, layout) {
            Ok(plan) => Ok((
                "plan",
                quail_planner::write_logical_plan(&plan)
                    .map_err(|error| DataFusionError::External(Box::new(error)))?,
            )),
            Err(error) if error.0.contains(bind::UNSUPPORTED) => Ok(("sql", region_sql(region)?)),
            Err(error) => Err(DataFusionError::Plan(format!("CompileError: {}", error.0))),
        }
    }

    /// Plan a bound region in Rust, or None when Python plans it: a
    /// region Rust refuses, or whose document is an expression.
    fn plan_here(
        &self,
        region: &ModelRegion,
        inputs: &[InputRows],
        payload: &str,
        fetch: Option<usize>,
    ) -> Result<Option<String>> {
        let Some((planner, settings)) = &self.planning else {
            return Ok(None);
        };
        let mut logical = quail_planner::read_logical_plan(payload)
            .map_err(|error| DataFusionError::External(Box::new(error)))?;
        if let Some(fetch) = fetch {
            logical.root = with_limit(logical.root, fetch as i64);
        }
        // Python plans a region that asks no model
        let operators = quail_planner::operators::operators(&logical);
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
        let mut tables = named_tables(region)?;
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
        match planner.plan(logical, &tables, settings) {
            Ok(Ok(planned)) => Ok(Some(
                planned_json(&planned)
                    .map_err(|error| DataFusionError::Plan(format!("CompileError: {}", error.0)))?
                    .to_string(),
            )),
            Ok(Err(refused)) => Ok(Some(crate::planning::refused_json(&refused).to_string())),
            Err(error) if error.0.starts_with(DOCUMENT_EXPRESSION) => Ok(None),
            Err(error) => Err(DataFusionError::Plan(format!("CompileError: {}", error.0))),
        }
    }

    /// Run the region on this thread.
    pub(crate) fn run_here(
        &self,
        region: &ModelRegion,
        inputs: &[InputRows],
        fetch: Option<usize>,
    ) -> Result<RegionBatches> {
        let (kind, payload) = self.request(region)?;
        let sorted;
        let (planned, inputs) = if kind == "plan" && self.planning.is_some() {
            sorted = inputs
                .iter()
                .zip(&region.inputs)
                .map(|((name, batches), input)| {
                    Ok((name.clone(), in_source_order(input, batches)?))
                })
                .collect::<Result<Vec<InputRows>>>()?;
            (
                self.plan_here(region, &sorted, &payload, fetch)?,
                sorted.as_slice(),
            )
        } else {
            (None, inputs)
        };
        let reader = Python::attach(|py| {
            // each input as a pyarrow Table, by the table name the region reads
            let mut tables = Vec::new();
            for ((name, batches), input) in inputs.iter().zip(&region.inputs) {
                let schema = Arc::new(input.schema().as_arrow().clone());
                tables.push((name.clone(), to_pyarrow(py, schema, batches.clone())?));
            }
            let tables = tables.into_py_dict(py)?;
            let answer = match &planned {
                Some(found) => self.engine.call_method1(py, "execute", (found, tables))?,
                None => {
                    self.engine
                        .call_method1(py, "run_region", (kind, payload, tables, fetch))?
                }
            };
            stream_from_pyarrow(answer.bind(py))
        })
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
        let schema = Arc::clone(region.bound.schema());
        let names = quail_column_names(&schema);
        // DataFusion reads each batch as the Python stream yields it
        Ok(Box::new(
            reader.map(move |batch| conform(batch?, &names, schema.inner())),
        ))
    }
}

impl RegionRunner for PythonRunner {
    /// Run the region on the thread that called DataFusion, so the GPU
    /// process's per-thread state is set up once, not on each worker.
    fn run(
        &self,
        region: &ModelRegion,
        inputs: &[InputRows],
        fetch: Option<usize>,
    ) -> Result<RegionBatches> {
        let caller = self.caller.lock().expect("the caller lock").clone();
        match caller {
            Some((thread, sender)) if thread != std::thread::current().id() => {
                let (rows, answer) = std::sync::mpsc::channel();
                let call = RegionCall {
                    region: region.clone(),
                    inputs: inputs.to_vec(),
                    fetch,
                    rows,
                };
                sender
                    .send(call)
                    .map_err(|_| DataFusionError::Execution("the query ended".into()))?;
                answer
                    .recv()
                    .map_err(|_| DataFusionError::Execution("the region did not run".into()))?
            }
            _ => self.run_here(region, inputs, fetch),
        }
    }
}

/// A region's plan that returns at most `fetch` rows, when DataFusion
/// passes a LIMIT above the region down to it, so the model can stop
/// once enough rows pass.
fn with_limit(root: quail_planner::LogicalNode, fetch: i64) -> quail_planner::LogicalNode {
    use quail_planner::LogicalNode;
    match root {
        LogicalNode::Result(mut result) => {
            result.limit = Some(result.limit.map_or(fetch, |limit| limit.min(fetch)));
            LogicalNode::Result(result)
        }
        other => LogicalNode::Result(quail_planner::logical::ResultNode {
            input: Box::new(other),
            limit: Some(fetch),
            order: Vec::new(),
            offset: 0,
            distinct: false,
        }),
    }
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
    use datafusion::arrow::compute::{concat_batches, sort_to_indices, take_record_batch};
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
        return Ok(batches.to_vec());
    };
    let table = provider
        .downcast_ref::<IdentifiedTable>()
        .expect("an identified table");
    let Some(python) = table.inner().downcast_ref::<PythonTable>() else {
        return Ok(batches.to_vec());
    };
    let key = python
        .constraints
        .iter()
        .find_map(|constraint| match constraint {
            Constraint::PrimaryKey(keys) if keys.len() == 1 => Some(keys[0]),
            _ => None,
        });
    let Some(key) = key.map(|index| python.schema.field(index).name().clone()) else {
        return Ok(batches.to_vec());
    };
    let Some(first) = batches.first() else {
        return Ok(Vec::new());
    };
    let rows = concat_batches(&first.schema(), batches)?;
    let Ok(column) = rows.schema().index_of(&key) else {
        return Ok(batches.to_vec());
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
    for batch in python.column_rows(&key)? {
        for value in texts(batch.column(0))? {
            if let Some(value) = value {
                positions.entry(value).or_insert(next);
            }
            next += 1;
        }
    }
    let found: datafusion::arrow::array::UInt64Array = texts(rows.column(column))?
        .into_iter()
        .map(|value| value.and_then(|value| positions.get(&value).copied()))
        .collect();
    let order = sort_to_indices(&found, None, None)?;
    Ok(vec![take_record_batch(&rows, &order)?])
}

/// The rows and identity of each identified table a region reads by
/// name, by table name.
fn named_tables(region: &ModelRegion) -> Result<HashMap<String, TableRows>> {
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::datasource::source_as_provider;
    use datafusion::logical_expr::LogicalPlan;
    let mut found: Vec<(String, Arc<dyn TableProvider>)> = Vec::new();
    region.bound.apply(|node| {
        if let LogicalPlan::TableScan(scan) = node
            && let Ok(provider) = source_as_provider(&scan.source)
            && provider.downcast_ref::<IdentifiedTable>().is_some()
        {
            found.push((scan.table_name.table().to_string(), provider));
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    let mut tables = HashMap::new();
    for (name, provider) in found {
        let table = provider
            .downcast_ref::<IdentifiedTable>()
            .expect("an identified table");
        let Some(python) = table.inner().downcast_ref::<PythonTable>() else {
            continue;
        };
        tables.insert(
            name,
            TableRows {
                identity: Some(table.identity().to_string()),
                schema: Arc::clone(&python.schema),
                batches: python.rows()?,
            },
        );
    }
    Ok(tables)
}

/// The content identity of an input that is a whole table with one,
/// read without filters.
fn whole_table_identity(input: &datafusion::logical_expr::LogicalPlan) -> Option<String> {
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::datasource::source_as_provider;
    use datafusion::logical_expr::LogicalPlan;
    let mut identity = None;
    let mut whole = true;
    let _ = input.apply(|node| {
        match node {
            LogicalPlan::TableScan(scan) => {
                whole &= scan.filters.is_empty() && scan.fetch.is_none();
                identity = source_as_provider(&scan.source).ok().and_then(|provider| {
                    provider
                        .downcast_ref::<IdentifiedTable>()
                        .map(|table| table.identity().to_string())
                });
            }
            LogicalPlan::SubqueryAlias(_) | LogicalPlan::Projection(_) => {}
            _ => whole = false,
        }
        Ok(TreeNodeRecursion::Continue)
    });
    if whole { identity } else { None }
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

/// A table whose rows a Python function returns on each scan.
///
/// Model regions read it through the engine, which keeps its token
/// counts, so DataFusion scans it only for operators outside every
/// region.
#[derive(Debug)]
pub(crate) struct PythonTable {
    pub(crate) schema: SchemaRef,
    pub(crate) scan: Py<PyAny>,
    /// The id column as a primary key, which DataFusion's rules read.
    pub(crate) constraints: Constraints,
}

impl PythonTable {
    /// Every row of the table.
    pub(crate) fn rows(&self) -> Result<Vec<RecordBatch>> {
        let columns: Vec<String> = self
            .schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        self.scan_columns(columns)
    }

    /// One column of the table's rows.
    fn column_rows(&self, name: &str) -> Result<Vec<RecordBatch>> {
        self.scan_columns(vec![name.to_string()])
    }

    fn scan_columns(&self, columns: Vec<String>) -> Result<Vec<RecordBatch>> {
        Python::attach(|py| {
            let answer = self.scan.call1(py, (columns,))?;
            from_pyarrow(answer.bind(py))
        })
        .map_err(|error| DataFusionError::External(Box::new(error)))
    }
}

#[async_trait]
impl TableProvider for PythonTable {
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
        let batches = Python::attach(|py| {
            let answer = self.scan.call1(py, (columns,))?;
            from_pyarrow(answer.bind(py))
        })
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
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
