//! Run a region's stages on several GPUs, one DataFusion partition per
//! GPU.
//!
//! [`quail_planner::stages`] cuts the region's graph into stages.
//! Partition p of every stage runs on the server of GPU p, a process
//! that holds every document and the GPU's KV; this module sends it the
//! stage's input rows and reads back the rows the stage writes, as Arrow
//! IPC over a Unix socket. DataFusion moves the rows between stages: a
//! hashed input goes through a `RepartitionExec` on the key's ids, so a
//! document reaches the same GPU in every stage that reads it by its
//! key; a broadcast input goes through a `CoalescePartitionsExec`,
//! collected once and handed whole to every partition.
//!
//! A stage's partition starts once all of its inputs have ended, and a
//! server runs one stage at a time; the stages form a DAG, so every
//! call a server receives can finish.

use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use datafusion::arrow::array::{Int32Array, RecordBatch};
use datafusion::arrow::compute::take;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::expressions::col;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::{ExecutionPlan, Partitioning, collect, common};
use datafusion::prelude::SessionContext;
use futures::future::{BoxFuture, FutureExt, Shared, try_join_all};
use quail_physical::{PhysicalGraph, PhysicalNode};
use quail_planner::stages::{Stage, stages};

/// The column a keyed table's rows are hashed by, which no partition
/// reads.
const KEY: &str = "quail.key";

/// A table's schema and batches.
pub type Table = (SchemaRef, Vec<RecordBatch>);

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

/// The connection to one GPU's server.
struct Server {
    reader: BufReader<UnixStream>,
    writer: BufWriter<UnixStream>,
}

/// What a server sends back for one stage.
struct Reply {
    outputs: Vec<Table>,
    /// Each finished table, tagged with its node and port, as Arrow IPC.
    finished: Vec<Vec<u8>>,
}

fn failure(error: impl std::fmt::Display) -> DataFusionError {
    DataFusionError::Execution(error.to_string())
}

fn write_frame(writer: &mut impl Write, bytes: &[u8]) -> Result<()> {
    writer.write_all(&(bytes.len() as u64).to_le_bytes())?;
    writer.write_all(bytes)?;
    Ok(())
}

fn read_frame(reader: &mut impl Read) -> Result<Vec<u8>> {
    let mut length = [0u8; 8];
    reader.read_exact(&mut length)?;
    let mut bytes = vec![0u8; u64::from_le_bytes(length) as usize];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn table_bytes((schema, batches): &Table) -> Result<Vec<u8>> {
    let mut writer = StreamWriter::try_new(Vec::new(), schema)?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.finish()?;
    Ok(writer.into_inner()?)
}

fn read_table(bytes: &[u8]) -> Result<Table> {
    let reader = StreamReader::try_new(bytes, None)?;
    let schema = reader.schema();
    let batches = reader.collect::<std::result::Result<Vec<_>, _>>()?;
    Ok((schema, batches))
}

impl Server {
    fn connect(path: &PathBuf) -> Result<Self> {
        let stream = UnixStream::connect(path)
            .map_err(|error| failure(format!("cannot reach {}: {error}", path.display())))?;
        Ok(Server {
            reader: BufReader::new(stream.try_clone()?),
            writer: BufWriter::new(stream),
        })
    }

    /// Run one stage on the server's GPU over its inputs' rows.
    fn call(&mut self, stage: usize, inputs: &[Table]) -> Result<Reply> {
        let header = serde_json::json!({"stage": stage, "inputs": inputs.len()}).to_string();
        write_frame(&mut self.writer, header.as_bytes())?;
        for input in inputs {
            write_frame(&mut self.writer, &table_bytes(input)?)?;
        }
        self.writer.flush()?;
        let header: serde_json::Value =
            serde_json::from_slice(&read_frame(&mut self.reader)?).map_err(failure)?;
        if let Some(error) = header.get("error").and_then(|error| error.as_str()) {
            return Err(failure(format!(
                "stage {stage} failed on its GPU:\n{error}"
            )));
        }
        let count = |name: &str| header[name].as_u64().unwrap_or(0);
        let outputs = (0..count("outputs"))
            .map(|_| read_table(&read_frame(&mut self.reader)?))
            .collect::<Result<Vec<_>>>()?;
        let finished = (0..count("finished"))
            .map(|_| read_frame(&mut self.reader))
            .collect::<Result<Vec<_>>>()?;
        Ok(Reply { outputs, finished })
    }

    /// Tell the server the region is over.
    fn end(&mut self) -> Result<()> {
        write_frame(&mut self.writer, br#"{"end": true}"#)?;
        self.writer.flush()?;
        Ok(())
    }
}

/// The server end of a region's connection, on one GPU: it reads the
/// stages [`Server`] sends and writes back what each wrote.
pub struct StageListener {
    listener: UnixListener,
    connection: Option<Server>,
}

impl StageListener {
    /// Listen at `path` for the region's connection.
    ///
    /// # Errors
    ///
    /// The socket does not bind.
    pub fn bind(path: &Path) -> Result<Self> {
        Ok(StageListener {
            listener: UnixListener::bind(path)?,
            connection: None,
        })
    }

    fn connection(&mut self) -> Result<&mut Server> {
        if self.connection.is_none() {
            let (stream, _) = self.listener.accept()?;
            self.connection = Some(Server {
                reader: BufReader::new(stream.try_clone()?),
                writer: BufWriter::new(stream),
            });
        }
        Ok(self.connection.as_mut().expect("a connection"))
    }

    /// The next stage to run and its inputs' rows, or None once the
    /// region ended; the first call waits for the connection.
    ///
    /// # Errors
    ///
    /// The connection fails or a frame does not read.
    pub fn next_stage(&mut self) -> Result<Option<(usize, Vec<Table>)>> {
        let connection = self.connection()?;
        let header: serde_json::Value =
            serde_json::from_slice(&read_frame(&mut connection.reader)?).map_err(failure)?;
        let Some(stage) = header["stage"].as_u64() else {
            return Ok(None);
        };
        let inputs = (0..header["inputs"].as_u64().unwrap_or(0))
            .map(|_| read_table(&read_frame(&mut connection.reader)?))
            .collect::<Result<Vec<_>>>()?;
        Ok(Some((stage as usize, inputs)))
    }

    /// Send what a stage wrote: its outputs' rows and its finished rows.
    ///
    /// # Errors
    ///
    /// The connection fails.
    pub fn reply(&mut self, outputs: &[Table], finished: &[Table]) -> Result<()> {
        let connection = self.connection()?;
        let header = serde_json::json!({"outputs": outputs.len(), "finished": finished.len()});
        write_frame(&mut connection.writer, header.to_string().as_bytes())?;
        for table in outputs.iter().chain(finished) {
            write_frame(&mut connection.writer, &table_bytes(table)?)?;
        }
        connection.writer.flush()?;
        Ok(())
    }

    /// Send why a stage failed.
    ///
    /// # Errors
    ///
    /// The connection fails.
    pub fn fail(&mut self, error: &str) -> Result<()> {
        let connection = self.connection()?;
        let header = serde_json::json!({"error": error}).to_string();
        write_frame(&mut connection.writer, header.as_bytes())?;
        connection.writer.flush()?;
        Ok(())
    }
}

/// Hands a finished table, as Arrow IPC, to the region's rows.
pub type Report = Box<dyn Fn(Vec<u8>) -> Result<()> + Send + Sync>;

/// A region's stages running on its GPUs' servers.
struct Staged {
    stages: Vec<Stage>,
    servers: Vec<Arc<Mutex<Server>>>,
    /// Scan node id -> its alias and number of documents.
    scans: HashMap<String, (String, usize)>,
    /// Alias -> each document's key, for a table hashed by a key other
    /// than its ids.
    keys: HashMap<String, Int32Array>,
    report: Arc<Mutex<Report>>,
    task: Arc<TaskContext>,
    runs: Mutex<HashMap<(usize, usize), Once<Written>>>,
    inputs: Mutex<HashMap<(usize, usize), Once<Ready>>>,
}

impl Staged {
    fn partitions(&self) -> usize {
        self.servers.len()
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
        let server = Arc::clone(&self.servers[partition]);
        let reply = tokio::task::spawn_blocking(move || {
            server.lock().expect("the server lock").call(stage, &tables)
        })
        .await
        .map_err(failure)??;
        let report = Arc::clone(&self.report);
        let finished = reply.finished;
        tokio::task::spawn_blocking(move || {
            // one report at a time: the region's rows take one table at once
            let report = report.lock().expect("the report lock");
            finished.into_iter().try_for_each(&*report)
        })
        .await
        .map_err(failure)??;
        Ok(Arc::new(reply.outputs))
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
                let output = self.stages[from]
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
                let schema = written
                    .iter()
                    .map(|tables| &tables[output])
                    .find(|(_, batches)| !rows(batches).is_empty())
                    .unwrap_or(&written[0][output])
                    .0
                    .clone();
                let partitions = written
                    .iter()
                    .map(|tables| rows(&tables[output].1))
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

    /// Tell every server the region is over.
    fn end(&self) -> Result<()> {
        let mut found = Ok(());
        for server in &self.servers {
            let ended = server.lock().expect("the server lock").end();
            if found.is_ok() {
                found = ended;
            }
        }
        found
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
                    columns.push(take(keys, batch.column(index).as_ref(), None)?);
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

/// Run a region's stages on the servers at `servers`, one per GPU.
///
/// `graph` holds the region's compute nodes; `keys` gives, by alias,
/// each document's key for a table whose rows go to a GPU by a key other
/// than their id, such as a stop key. Each table a stage finishes goes
/// to `report`, one at a time. Every server is told the region ended,
/// whether or not it failed.
///
/// # Errors
///
/// The graph does not cut into stages, a server is unreachable, or a
/// stage fails.
pub fn run_stages(
    graph: &PhysicalGraph,
    keys: HashMap<String, Vec<i32>>,
    servers: &[PathBuf],
    report: Report,
) -> Result<()> {
    // each server waits for this connection, so a region that cannot
    // run still connects, then ends
    let servers = servers
        .iter()
        .map(|path| Ok(Arc::new(Mutex::new(Server::connect(path)?))))
        .collect::<Result<Vec<_>>>()?;
    let cut = match stages(graph) {
        Ok(cut) => cut,
        Err(error) => {
            for server in &servers {
                let _ = server.lock().expect("the server lock").end();
            }
            return Err(DataFusionError::Plan(error));
        }
    };
    let scans = graph
        .nodes
        .iter()
        .filter_map(|node| match node {
            PhysicalNode::Scan(scan) => Some((
                scan.id.clone(),
                (
                    scan.attributes.alias.clone(),
                    scan.attributes.num_documents as usize,
                ),
            )),
            _ => None,
        })
        .collect();
    let keys = keys
        .into_iter()
        .map(|(alias, values)| (alias, Int32Array::from(values)))
        .collect();
    let staged = Arc::new(Staged {
        stages: cut,
        servers,
        scans,
        keys,
        report: Arc::new(Mutex::new(report)),
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
    let ran = thread
        .join()
        .unwrap_or_else(|_| Err(failure("the stages' thread panicked")));
    let ended = staged.end();
    ran.and(ended)
}
