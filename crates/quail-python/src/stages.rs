//! Run a region's stages on several GPUs, one DataFusion partition per
//! GPU.
//!
//! [`quail_planner::stages`] cuts the region's graph into stages when
//! the region is planned; this module runs the stages the plan holds.
//! Partition p of every stage runs on the server of GPU p, a process
//! that holds every document and the GPU's KV; this module sends it the
//! stage's input rows and reads back the rows the stage writes. Each
//! table crosses in a file both processes map ([`crate::shared`]); the
//! Unix socket carries only the file's path. DataFusion moves the rows between stages: a
//! hashed input goes through a `RepartitionExec` on the key's ids, so a
//! document reaches the same GPU in every stage that reads it by its
//! key; a broadcast input goes through a `CoalescePartitionsExec`,
//! collected once and handed whole to every partition.
//!
//! While a stage runs, its server sends each table it finishes as soon
//! as it has it, and may ask the region a question, such as whether the
//! tests above the region still keep a row; the region answers before
//! the stage goes on.
//!
//! A stage's partition starts once all of its inputs have ended, and a
//! server runs one stage at a time; the stages form a DAG, so every
//! call a server receives can finish.

use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use datafusion::arrow::array::{Int32Array, RecordBatch};
use datafusion::arrow::compute::{TakeOptions, take};
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
use quail_planner::physical::{PhysicalGraph, PhysicalNode};
use quail_planner::stages::Stage;

/// The column a keyed table's rows are hashed by, which no partition
/// reads.
const KEY: &str = "quail.key";

pub use crate::shared::Table;
use crate::shared::{read_table, write_table};

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

/// One end of the connection between the region and a GPU's server.
struct Server {
    reader: BufReader<UnixStream>,
    writer: BufWriter<UnixStream>,
    /// The path each table this end writes starts with.
    shared: String,
    num_written: usize,
}

/// What the region answers a server's question with.
pub enum Answer {
    Json(serde_json::Value),
    Table(Table),
}

/// Takes a finished table, tagged with its node and port.
pub type Report = Box<dyn Fn(Table) -> Result<()> + Send + Sync>;

/// Answers a server's question.
pub type Ask = Box<dyn Fn(&serde_json::Value) -> Result<Answer> + Send + Sync>;

/// What a region does for its servers while its stages run.
pub struct Region {
    /// Called with one table at a time.
    pub report: Report,
    pub ask: Ask,
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
    let length = u64::from_le_bytes(length);
    // Do not allocate from an untrusted length before receiving its bytes.
    let mut bytes = Vec::new();
    reader.take(length).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != length {
        return Err(failure("a stage frame ended before its declared length"));
    }
    Ok(bytes)
}

fn header_count(header: &serde_json::Value, name: &str) -> Result<usize> {
    header
        .get(name)
        .and_then(serde_json::Value::as_u64)
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| failure(format!("a stage header has no valid {name} count")))
}

impl Server {
    /// An end of `stream`, the socket at `socket`, whose tables' files
    /// are named after the socket and `end`.
    fn new(stream: UnixStream, socket: &Path, end: &str) -> Result<Self> {
        Ok(Server {
            reader: BufReader::new(stream.try_clone()?),
            writer: BufWriter::new(stream),
            shared: format!("{}.{end}", socket.display()),
            num_written: 0,
        })
    }

    fn connect(path: &PathBuf) -> Result<Self> {
        let stream = UnixStream::connect(path)
            .map_err(|error| failure(format!("cannot reach {}: {error}", path.display())))?;
        Server::new(stream, path, "region")
    }

    /// Share a table with the other end: write it to a file and send
    /// the file's path.
    fn send_table(&mut self, table: &Table) -> Result<()> {
        let path = format!("{}.{}", self.shared, self.num_written);
        self.num_written += 1;
        write_table(Path::new(&path), table)?;
        write_frame(&mut self.writer, path.as_bytes())
    }

    /// Read a table the other end shared, over the file's mapped pages.
    fn receive_table(&mut self) -> Result<Table> {
        let path = String::from_utf8(read_frame(&mut self.reader)?).map_err(failure)?;
        read_table(Path::new(&path))
    }

    /// Run one stage on the server's GPU over its inputs' rows, and
    /// return the rows of its outputs.
    fn call(&mut self, stage: usize, inputs: &[Table], region: &Region) -> Result<Vec<Table>> {
        let header = serde_json::json!({"stage": stage, "inputs": inputs.len()}).to_string();
        write_frame(&mut self.writer, header.as_bytes())?;
        for input in inputs {
            self.send_table(input)?;
        }
        self.writer.flush()?;
        loop {
            let header: serde_json::Value =
                serde_json::from_slice(&read_frame(&mut self.reader)?).map_err(failure)?;
            if let Some(error) = header.get("error").and_then(|error| error.as_str()) {
                return Err(failure(format!(
                    "stage {stage} failed on its GPU:\n{error}"
                )));
            }
            if header.get("finished").is_some() {
                (region.report)(self.receive_table()?)?;
            } else if let Some(question) = header.get("ask") {
                self.answer((region.ask)(question)?)?;
            } else {
                return (0..header_count(&header, "outputs")?)
                    .map(|_| self.receive_table())
                    .collect();
            }
        }
    }

    fn answer(&mut self, answer: Answer) -> Result<()> {
        match answer {
            Answer::Json(value) => {
                let header = serde_json::json!({"answer": value}).to_string();
                write_frame(&mut self.writer, header.as_bytes())?;
            }
            Answer::Table(table) => {
                write_frame(&mut self.writer, br#"{"table": true}"#)?;
                self.send_table(&table)?;
            }
        }
        self.writer.flush()?;
        Ok(())
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
    socket: PathBuf,
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
            socket: path.to_path_buf(),
            connection: None,
        })
    }

    fn connection(&mut self) -> Result<&mut Server> {
        if self.connection.is_none() {
            let (stream, _) = self.listener.accept()?;
            self.connection = Some(Server::new(stream, &self.socket, "server")?);
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
        if header.get("end").and_then(serde_json::Value::as_bool) == Some(true) {
            return Ok(None);
        }
        let stage = header_count(&header, "stage")?;
        let inputs = (0..header_count(&header, "inputs")?)
            .map(|_| connection.receive_table())
            .collect::<Result<Vec<_>>>()?;
        Ok(Some((stage, inputs)))
    }

    /// Send the rows of a stage's outputs, which ends the stage.
    ///
    /// # Errors
    ///
    /// The connection fails.
    pub fn reply(&mut self, outputs: &[Table]) -> Result<()> {
        let connection = self.connection()?;
        let header = serde_json::json!({"outputs": outputs.len()});
        write_frame(&mut connection.writer, header.to_string().as_bytes())?;
        for table in outputs {
            connection.send_table(table)?;
        }
        connection.writer.flush()?;
        Ok(())
    }

    /// Send a table the running stage finished, tagged with its node
    /// and port.
    ///
    /// # Errors
    ///
    /// The connection fails.
    pub fn finished(&mut self, table: &Table) -> Result<()> {
        let connection = self.connection()?;
        write_frame(&mut connection.writer, br#"{"finished": true}"#)?;
        connection.send_table(table)?;
        connection.writer.flush()?;
        Ok(())
    }

    /// Ask the region a question while a stage runs, and wait for its
    /// answer.
    ///
    /// # Errors
    ///
    /// The connection fails, or the region ends it without an answer.
    pub fn ask(&mut self, question: &serde_json::Value) -> Result<Answer> {
        let connection = self.connection()?;
        let header = serde_json::json!({"ask": question}).to_string();
        write_frame(&mut connection.writer, header.as_bytes())?;
        connection.writer.flush()?;
        let mut header: serde_json::Value =
            serde_json::from_slice(&read_frame(&mut connection.reader)?).map_err(failure)?;
        if header.get("table").is_some() {
            return Ok(Answer::Table(connection.receive_table()?));
        }
        match header.get_mut("answer") {
            Some(answer) => Ok(Answer::Json(answer.take())),
            None => Err(failure("the region ended before it answered")),
        }
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

/// A region's stages running on its GPUs' servers.
struct Staged {
    stages: Vec<Stage>,
    servers: Vec<Arc<Mutex<Server>>>,
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
        let region = Arc::clone(&self.region);
        let outputs = tokio::task::spawn_blocking(move || {
            (server.lock().expect("the server lock")).call(stage, &tables, &region)
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

/// Run a region's stages, as the plan cut them, on the servers at
/// `servers`, one per GPU.
///
/// `graph` holds the region's compute nodes; `groupings` gives, by
/// alias, the groups [`grouped_keys`] reads for a table whose rows go to
/// a GPU by a key other than their id. Each table a stage finishes goes
/// to `region.report`, one at a time, and each question a server asks
/// to `region.ask`. Every server is told the region ended, whether or
/// not it failed.
///
/// # Errors
///
/// A grouping names no scan, a server is unreachable, or a stage fails.
pub fn run_stages(
    graph: &PhysicalGraph,
    cut: Vec<Stage>,
    groupings: HashMap<String, Vec<Vec<i64>>>,
    servers: &[PathBuf],
    region: Region,
) -> Result<()> {
    if !cut.is_empty() && servers.is_empty() {
        return Err(failure("a stage has no GPU servers"));
    }
    // each server waits for this connection, so a region that cannot
    // run still connects, then ends
    let mut connected = Vec::with_capacity(servers.len());
    let mut first_error = None;
    for path in servers {
        match Server::connect(path) {
            Ok(server) => connected.push(Arc::new(Mutex::new(server))),
            Err(error) => {
                // Do not strand the later listeners in accept(). Preserve
                // the first error, but connect and end every reachable peer.
                first_error.get_or_insert(error);
            }
        }
    }
    let servers = connected;
    if let Some(error) = first_error {
        for server in &servers {
            let _ = server.lock().expect("the server lock").end();
        }
        return Err(error);
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
        .collect::<Result<HashMap<_, _>>>();
    let (scans, keys) = match scans.and_then(|scans| {
        let num_documents = scans.values().cloned().collect();
        let keys = grouped_keys(groupings, &num_documents)?;
        Ok((scans, keys))
    }) {
        Ok(found) => found,
        Err(error) => {
            for server in &servers {
                let _ = server.lock().expect("the server lock").end();
            }
            return Err(error);
        }
    };
    let staged = Arc::new(Staged {
        stages: cut,
        servers,
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
    let ran = thread
        .join()
        .unwrap_or_else(|_| Err(failure("the stages' thread panicked")));
    let ended = staged.end();
    ran.and(ended)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_frames_return_errors() {
        let length = u64::MAX.to_le_bytes();
        assert!(read_frame(&mut length.as_slice()).is_err());
        assert!(read_frame(&mut &[0u8; 3][..]).is_err());
    }

    #[test]
    fn malformed_header_counts_return_errors() {
        for header in [
            serde_json::json!(null),
            serde_json::json!({}),
            serde_json::json!({"outputs": -1}),
            serde_json::json!({"outputs": "one"}),
        ] {
            assert!(matches!(
                header_count(&header, "outputs"),
                Err(DataFusionError::Execution(_))
            ));
        }
        assert_eq!(
            header_count(&serde_json::json!({"outputs": 0}), "outputs").unwrap(),
            0
        );
    }

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
