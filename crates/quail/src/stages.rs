//! Run a region's stages on several GPUs, one DataFusion partition per
//! GPU.
//!
//! [`quail_planner::stages`] cuts the region's graph into stages when
//! the region is planned; this module runs the stages the plan holds.
//! Partition p of every stage runs on the worker of GPU p, a process
//! that holds every document and the GPU's KV
//! ([`crate::workers`]); this module sends it the stage's input rows
//! and reads back the rows the stage writes. DataFusion moves the rows
//! between stages: a hashed input goes through a `RepartitionExec` on the key's ids, so a
//! document reaches the same GPU in every stage that reads it by its
//! key; a broadcast input goes through a `CoalescePartitionsExec`,
//! collected once and handed whole to every partition.
//!
//! While a stage runs, its worker sends each table it finishes as soon
//! as it has it, and may ask the region a question, such as whether the
//! tests above the region still keep a row; the region answers before
//! the stage goes on.
//!
//! A stage's partition starts once all of its inputs have ended, and a
//! worker runs one stage at a time; the stages form a DAG, so every
//! call a worker receives can finish.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use datafusion::arrow::array::{Int32Array, RecordBatch};
use datafusion::arrow::compute::{TakeOptions, take};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::Result;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::expressions::col;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::{ExecutionPlan, Partitioning, collect, common};
use datafusion::prelude::SessionContext;
use futures::future::{BoxFuture, FutureExt, Shared, try_join_all};
use quail_planner::physical::{PhysicalGraph, PhysicalNode};
use quail_planner::stages::Stage;

/// The column a keyed table's rows are hashed by, which no partition
/// reads.
const KEY: &str = "quail.key";

use crate::shared::Table;
use crate::workers::{Connection, Region, failure};

/// What one partition of a stage wrote: each output port's rows, in the
/// order of the stage's outputs.
type Written = Arc<Vec<Table>>;

/// A future many readers await; its error is the failure's text.
type Once<T> = Shared<BoxFuture<'static, std::result::Result<T, String>>>;

/// One stage input, ready for its partitions.
#[derive(Clone)]
enum Ready {
    /// Partition p reads output partition p.
    Hashed(Arc<dyn ExecutionPlan>),
    /// Every partition reads every row.
    Whole(Arc<Table>),
}

/// A region's stages running on its GPUs' workers.
struct Staged {
    stages: Vec<Stage>,
    workers: Vec<Arc<Mutex<Connection>>>,
    /// Scan node id -> its alias and number of documents.
    scans: HashMap<String, (String, usize)>,
    /// Alias -> each document's key, for a table hashed by a key other
    /// than its ids.
    keys: HashMap<String, Int32Array>,
    region: Arc<Region>,
    task: Arc<TaskContext>,
    runs: Mutex<HashMap<(usize, usize), Once<Written>>>,
    inputs: Mutex<HashMap<(usize, usize), Once<Ready>>>,
}

impl Staged {
    fn partitions(&self) -> usize {
        self.workers.len()
    }

    /// The run of one stage's partition, started on first use.
    fn run(self: &Arc<Self>, stage: usize, partition: usize) -> Once<Written> {
        let mut runs = self.runs.lock().expect("the runs lock");
        let staged = Arc::clone(self);
        runs.entry((stage, partition))
            .or_insert_with(|| {
                async move {
                    staged
                        .run_partition(stage, partition)
                        .await
                        .map_err(|error| error.to_string())
                }
                .boxed()
                .shared()
            })
            .clone()
    }

    async fn run_partition(self: Arc<Self>, stage: usize, partition: usize) -> Result<Written> {
        let mut tables = Vec::new();
        for index in 0..self.stages[stage].inputs.len() {
            let ready = self.input(stage, index).await.map_err(failure)?;
            tables.push(match ready {
                Ready::Hashed(plan) => {
                    let schema = plan.schema();
                    let batches =
                        common::collect(plan.execute(partition, Arc::clone(&self.task))?).await?;
                    unkeyed(&schema, batches)?
                }
                Ready::Whole(table) => (*table).clone(),
            });
        }
        let worker = Arc::clone(&self.workers[partition]);
        let region = Arc::clone(&self.region);
        let outputs = tokio::task::spawn_blocking(move || {
            (worker.lock().expect("the worker lock")).call(stage, &tables, &region)
        })
        .await
        .map_err(failure)??;
        if outputs.len() != self.stages[stage].outputs.len() {
            return Err(failure(format!(
                "stage {stage} returned {} outputs, expected {}",
                outputs.len(),
                self.stages[stage].outputs.len()
            )));
        }
        Ok(Arc::new(outputs))
    }

    /// One stage input, collected once its writers have all ended.
    fn input(self: &Arc<Self>, stage: usize, index: usize) -> Once<Ready> {
        let mut inputs = self.inputs.lock().expect("the inputs lock");
        let staged = Arc::clone(self);
        inputs
            .entry((stage, index))
            .or_insert_with(|| {
                async move {
                    staged
                        .ready(stage, index)
                        .await
                        .map_err(|error| error.to_string())
                }
                .boxed()
                .shared()
            })
            .clone()
    }

    async fn ready(self: Arc<Self>, stage: usize, index: usize) -> Result<Ready> {
        let key = &self.stages[stage].key;
        let input = &self.stages[stage].inputs[index];
        let (schema, partitions) = match input.stage {
            None => {
                let (alias, count) = self.scans.get(&input.node_id).ok_or_else(|| {
                    failure(format!("{:?} is neither a scan nor a stage", input.node_id))
                })?;
                let ids = Int32Array::from_iter_values(0..*count as i32);
                let schema = Arc::new(Schema::new(vec![Field::new(alias, DataType::Int32, false)]));
                let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(ids)])?;
                (schema, vec![vec![batch]])
            }
            Some(from) => {
                let wanted = (input.node_id.clone(), input.port.clone());
                let output = self
                    .stages
                    .get(from)
                    .ok_or_else(|| failure(format!("stage {from} does not exist")))?
                    .outputs
                    .iter()
                    .position(|found| *found == wanted)
                    .ok_or_else(|| failure(format!("stage {from} does not write {wanted:?}")))?;
                let runs = (0..self.partitions()).map(|partition| self.run(from, partition));
                let written = try_join_all(runs).await.map_err(failure)?;
                // a GPU that wrote no rows may type a column apart from
                // the others: its empty batches are left out
                let rows = |batches: &Vec<RecordBatch>| -> Vec<RecordBatch> {
                    batches
                        .iter()
                        .filter(|batch| batch.num_rows() > 0)
                        .cloned()
                        .collect()
                };
                let tables = written
                    .iter()
                    .map(|tables| {
                        tables.get(output).ok_or_else(|| {
                            failure(format!("stage {from} returned no output {output}"))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let first = tables
                    .first()
                    .ok_or_else(|| failure("a stage has no output partitions"))?;
                let schema = tables
                    .iter()
                    .find(|(_, batches)| !rows(batches).is_empty())
                    .unwrap_or(first)
                    .0
                    .clone();
                let partitions = tables
                    .iter()
                    .map(|(_, batches)| rows(batches))
                    .collect::<Vec<_>>();
                (schema, partitions)
            }
        };
        if !input.hashed {
            let memory = MemorySourceConfig::try_new_exec(&partitions, Arc::clone(&schema), None)?;
            let whole = CoalescePartitionsExec::new(memory);
            let batches = collect(Arc::new(whole), Arc::clone(&self.task)).await?;
            return Ok(Ready::Whole(Arc::new((schema, batches))));
        }
        let (schema, partitions, column) = match self.keys.get(key) {
            Some(keys) => {
                let (schema, partitions) = keyed(&schema, partitions, key, keys)?;
                (schema, partitions, KEY)
            }
            None => (schema, partitions, key.as_str()),
        };
        let memory = MemorySourceConfig::try_new_exec(&partitions, Arc::clone(&schema), None)?;
        let hash = Partitioning::Hash(vec![col(column, &schema)?], self.partitions());
        Ok(Ready::Hashed(Arc::new(RepartitionExec::try_new(
            memory, hash,
        )?)))
    }
}

/// Rows with each row's key beside its id, to hash by.
fn keyed(
    schema: &SchemaRef,
    partitions: Vec<Vec<RecordBatch>>,
    alias: &str,
    keys: &Int32Array,
) -> Result<(SchemaRef, Vec<Vec<RecordBatch>>)> {
    let index = schema.index_of(alias)?;
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new(KEY, DataType::Int32, true));
    let keyed_schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    let partitions = partitions
        .into_iter()
        .map(|batches| {
            batches
                .into_iter()
                .map(|batch| {
                    let mut columns = batch.columns().to_vec();
                    let ids = batch
                        .columns()
                        .get(index)
                        .ok_or_else(|| failure(format!("the rows have no key column {alias}")))?;
                    columns.push(take(
                        keys,
                        ids.as_ref(),
                        Some(TakeOptions { check_bounds: true }),
                    )?);
                    Ok(RecordBatch::try_new(Arc::clone(&keyed_schema), columns)?)
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((keyed_schema, partitions))
}

/// Rows without the key column [`keyed`] added.
fn unkeyed(schema: &SchemaRef, batches: Vec<RecordBatch>) -> Result<Table> {
    let Ok(index) = schema.index_of(KEY) else {
        return Ok((Arc::clone(schema), batches));
    };
    let kept: Vec<usize> = (0..schema.fields().len()).filter(|i| *i != index).collect();
    let schema = Arc::new(schema.project(&kept)?);
    let batches = batches
        .iter()
        .map(|batch| Ok(batch.project(&kept)?))
        .collect::<Result<Vec<_>>>()?;
    Ok((schema, batches))
}

/// Each document's key, by alias, for a table hashed by more than its
/// ids.
///
/// `groupings` gives, by alias, lists of each document's group: a stop
/// key's group goes to one GPU, so its per-key stop sees the whole
/// group, and copies of one text go with the copy that is asked. A
/// document's key is the smallest id it is grouped with, through every
/// list alike.
fn grouped_keys(
    groupings: HashMap<String, Vec<Vec<i64>>>,
    num_documents: &HashMap<String, usize>,
) -> Result<HashMap<String, Int32Array>> {
    fn root(parent: &mut [usize], mut document: usize) -> usize {
        while parent[document] != document {
            parent[document] = parent[parent[document]];
            document = parent[document];
        }
        document
    }
    groupings
        .into_iter()
        .map(|(alias, lists)| {
            let count = *num_documents
                .get(&alias)
                .ok_or_else(|| failure(format!("no scan reads {alias:?}")))?;
            let mut parent: Vec<usize> = (0..count).collect();
            for groups in lists {
                let mut first: HashMap<i64, usize> = HashMap::new();
                for (document, group) in groups.into_iter().enumerate().take(count) {
                    let other = *first.entry(group).or_insert(document);
                    let (left, right) = (root(&mut parent, document), root(&mut parent, other));
                    parent[left.max(right)] = left.min(right);
                }
            }
            let keys = (0..count).map(|document| root(&mut parent, document) as i32);
            Ok((alias, Int32Array::from_iter_values(keys)))
        })
        .collect()
}

/// Run a region's stages, as the plan cut them, on `workers`, one per
/// GPU, each of which has begun the region.
///
/// `graph` holds the region's compute nodes; `groupings` gives, by
/// alias, the groups [`grouped_keys`] reads for a table whose rows go to
/// a GPU by a key other than their id. Each table a stage finishes goes
/// to `region.report`, one at a time, and each question a worker asks
/// to `region.ask`.
///
/// # Errors
///
/// A grouping names no scan, or a stage fails.
pub fn run_stages(
    graph: &PhysicalGraph,
    cut: Vec<Stage>,
    groupings: HashMap<String, Vec<Vec<i64>>>,
    workers: &[Arc<Mutex<Connection>>],
    region: Region,
) -> Result<()> {
    if !cut.is_empty() && workers.is_empty() {
        return Err(failure("a stage has no GPU workers"));
    }
    let scans = graph
        .nodes
        .iter()
        .filter_map(|node| match node {
            PhysicalNode::Scan(scan) => Some(scan),
            _ => None,
        })
        .map(|scan| {
            let count = i32::try_from(scan.attributes.num_documents)
                .ok()
                .filter(|count| *count >= 0)
                .ok_or_else(|| {
                    failure(format!(
                        "scan {:?} has an invalid document count {}",
                        scan.id, scan.attributes.num_documents
                    ))
                })?;
            Ok((
                scan.id.clone(),
                (scan.attributes.alias.clone(), count as usize),
            ))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    let keys = grouped_keys(groupings, &scans.values().cloned().collect())?;
    let staged = Arc::new(Staged {
        stages: cut,
        workers: workers.to_vec(),
        scans,
        keys,
        region: Arc::new(Region {
            report: {
                // the region's rows take one table at a time
                let (report, one) = (region.report, Mutex::new(()));
                Box::new(move |table| {
                    let _held = one.lock().expect("the report lock");
                    report(table)
                })
            },
            ask: region.ask,
        }),
        task: SessionContext::new().task_ctx(),
        runs: Mutex::new(HashMap::new()),
        inputs: Mutex::new(HashMap::new()),
    });
    // the caller may be inside another runtime, where a runtime cannot
    // block: the stages run on a thread of their own
    let thread = {
        let staged = Arc::clone(&staged);
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Runtime::new()?;
            runtime.block_on(async move {
                let runs = (0..staged.stages.len()).flat_map(|stage| {
                    (0..staged.partitions()).map(move |partition| (stage, partition))
                });
                let runs = runs.map(|(stage, partition)| staged.run(stage, partition));
                try_join_all(runs).await.map(|_| ()).map_err(failure)
            })
        })
    };
    thread
        .join()
        .unwrap_or_else(|_| Err(failure("the stages' thread panicked")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyed_rows_reject_invalid_document_ids() {
        let schema = Arc::new(Schema::new(vec![Field::new("r", DataType::Int32, false)]));
        let keys = Int32Array::from(vec![0]);
        for id in [-1, 1] {
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from(vec![id]))],
            )
            .unwrap();
            assert!(keyed(&schema, vec![vec![batch]], "r", &keys).is_err());
        }
    }
}
