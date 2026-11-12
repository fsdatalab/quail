//! The GPU workers of a session: one process per GPU, which keeps its
//! model loaded between queries, and the connection to each.
//!
//! A worker is a Python process that runs the model. Quail starts it,
//! and the two talk over one Unix socket for the worker's life:
//!
//! - `describe` asks what Quail plans a configuration's regions with:
//!   the model's prompt layout, tokenizer, and planning settings. The
//!   models are defined in Python, so a worker answers.
//! - `boot` prepares the worker's GPU for a plan; the worker answers
//!   with how long each part of the boot took.
//! - `region` begins a region: its plan and settings as JSON, and its
//!   documents and columns as tables. Then come the region's stages
//!   ([`crate::stages`]), and `end`, which the worker answers with what
//!   it ran.
//!
//! A message is a JSON header. A table never goes through the socket:
//! its writer puts it in a file both processes map ([`crate::shared`])
//! and sends the file's path.

use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use datafusion::common::{DataFusionError, Result};
use serde_json::{Value, json};

use crate::shared::{Table, read_table, write_table};

/// How long a worker has to connect after it starts, or to exit after
/// it is told to.
const PATIENCE: Duration = Duration::from_secs(120);
const POLL: Duration = Duration::from_millis(10);

pub(crate) fn failure(error: impl std::fmt::Display) -> DataFusionError {
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
        return Err(failure("a frame ended before its declared length"));
    }
    Ok(bytes)
}

fn count(header: &Value, name: &str) -> Result<usize> {
    header
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| failure(format!("a header has no valid {name} count")))
}

/// What the region answers a worker's question with.
pub enum Answer {
    Json(Value),
    Table(Table),
}

/// Takes a finished table, tagged with its node and port.
pub type Report = Box<dyn Fn(Table) -> Result<()> + Send + Sync>;

/// Answers a worker's question.
pub type Ask = Box<dyn Fn(&Value) -> Result<Answer> + Send + Sync>;

/// What a region does for its workers while its stages run.
pub struct Region {
    /// Called with one table at a time.
    pub report: Report,
    pub ask: Ask,
}

/// What a worker ran of a region: its counters as JSON, and each
/// node's output tables, tagged with node and port.
pub type Ran = (Value, Vec<Table>);

/// One end of the socket between Quail and a worker.
pub struct Connection {
    reader: BufReader<UnixStream>,
    writer: BufWriter<UnixStream>,
    /// The path each table this end writes starts with.
    shared: String,
    num_written: usize,
    in_region: bool,
}

impl Connection {
    /// An end of `stream`, the socket at `socket`, whose tables' files
    /// are named after the socket and `end`.
    fn new(stream: UnixStream, socket: &Path, end: &str) -> Result<Self> {
        Ok(Connection {
            reader: BufReader::new(stream.try_clone()?),
            writer: BufWriter::new(stream),
            shared: format!("{}.{end}", socket.display()),
            num_written: 0,
            in_region: false,
        })
    }

    fn send(&mut self, header: &Value) -> Result<()> {
        write_frame(&mut self.writer, header.to_string().as_bytes())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(self.writer.flush()?)
    }

    /// The next header; a worker's error is the error.
    fn receive(&mut self) -> Result<Value> {
        let header: Value =
            serde_json::from_slice(&read_frame(&mut self.reader)?).map_err(failure)?;
        match header.get("error").and_then(Value::as_str) {
            Some(error) => Err(failure(format!("a GPU worker failed:\n{error}"))),
            None => Ok(header),
        }
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
    fn receive_table(&mut self, remove: bool) -> Result<Table> {
        let path = String::from_utf8(read_frame(&mut self.reader)?).map_err(failure)?;
        read_table(Path::new(&path), remove)
    }

    fn receive_tables(&mut self, count: usize) -> Result<Vec<Table>> {
        (0..count).map(|_| self.receive_table(true)).collect()
    }
}

/// What Quail asks of a worker.
impl Connection {
    /// What Quail plans the regions of `config` with.
    fn describe(&mut self, config: &Value) -> Result<Value> {
        self.send(&json!({"describe": config}))?;
        self.flush()?;
        let mut header = self.receive()?;
        Ok(header["described"].take())
    }

    /// Prepare the worker's GPU for a plan; returns its boot record.
    fn boot(&mut self, setup: &Value) -> Result<Value> {
        self.send(&json!({"boot": setup}))?;
        self.flush()?;
        let mut header = self.receive()?;
        Ok(header["booted"].take())
    }

    /// Begin a region: its description and the paths of its tables,
    /// which every worker of the region maps.
    fn begin(&mut self, region: &Value, tables: &[(String, String)]) -> Result<()> {
        self.send(&json!({"region": region, "tables": tables}))?;
        self.flush()?;
        self.receive()?;
        self.in_region = true;
        Ok(())
    }

    /// Run one stage on the worker's GPU over its inputs' rows, and
    /// return the rows of its outputs.
    pub(crate) fn call(
        &mut self,
        stage: usize,
        inputs: &[Table],
        region: &Region,
    ) -> Result<Vec<Table>> {
        self.send(&json!({"stage": stage, "inputs": inputs.len()}))?;
        for input in inputs {
            self.send_table(input)?;
        }
        self.flush()?;
        loop {
            let header = self.receive()?;
            if header.get("finished").is_some() {
                (region.report)(self.receive_table(true)?)?;
            } else if let Some(question) = header.get("ask") {
                match (region.ask)(question)? {
                    Answer::Json(value) => self.send(&json!({"answer": value}))?,
                    Answer::Table(table) => {
                        self.send(&json!({"table": true}))?;
                        self.send_table(&table)?;
                    }
                }
                self.flush()?;
            } else {
                return self.receive_tables(count(&header, "outputs")?);
            }
        }
    }

    /// End the region the worker is in, if any; returns what it ran.
    fn end(&mut self) -> Result<Option<Ran>> {
        if !std::mem::take(&mut self.in_region) {
            return Ok(None);
        }
        self.send(&json!({"end": true}))?;
        self.flush()?;
        let mut header = self.receive()?;
        let tables = self.receive_tables(count(&header, "tables")?)?;
        Ok(Some((header["ran"].take(), tables)))
    }
}

/// What a worker reads next from Quail.
pub enum Message {
    /// A configuration to describe.
    Describe(Value),
    Boot(Value),
    /// A region's description and its tables, by name.
    Region(Value, Vec<(String, Table)>),
    /// A stage's index and its inputs' rows.
    Stage(usize, Vec<Table>),
    /// The region ended.
    End,
    /// Quail is done with the worker.
    Shutdown,
}

/// The worker's end of its connection to Quail.
pub struct WorkerEnd {
    connection: Connection,
}

impl WorkerEnd {
    /// Connect to the socket Quail listens at.
    ///
    /// # Errors
    ///
    /// The socket is unreachable.
    pub fn connect(socket: &Path) -> Result<Self> {
        let stream = UnixStream::connect(socket)
            .map_err(|error| failure(format!("cannot reach {}: {error}", socket.display())))?;
        Ok(WorkerEnd {
            connection: Connection::new(stream, socket, "worker")?,
        })
    }

    /// The next message; a closed connection is a shutdown.
    ///
    /// # Errors
    ///
    /// A frame or a table does not read.
    pub fn read(&mut self) -> Result<Message> {
        let connection = &mut self.connection;
        let frame = match read_frame(&mut connection.reader) {
            Ok(frame) => frame,
            Err(DataFusionError::IoError(error)) if error.kind() == ErrorKind::UnexpectedEof => {
                return Ok(Message::Shutdown);
            }
            Err(error) => return Err(error),
        };
        let mut header: Value = serde_json::from_slice(&frame).map_err(failure)?;
        if header.get("describe").is_some() {
            return Ok(Message::Describe(header["describe"].take()));
        }
        if header.get("boot").is_some() {
            return Ok(Message::Boot(header["boot"].take()));
        }
        if header.get("region").is_some() {
            let named: Vec<(String, String)> =
                serde_json::from_value(header["tables"].take()).map_err(failure)?;
            let tables = named
                .into_iter()
                // the region's other workers map the same files
                .map(|(name, path)| Ok((name, read_table(Path::new(&path), false)?)))
                .collect::<Result<Vec<_>>>()?;
            return Ok(Message::Region(header["region"].take(), tables));
        }
        if header.get("stage").is_some() {
            let inputs = count(&header, "inputs")?;
            return Ok(Message::Stage(
                count(&header, "stage")?,
                connection.receive_tables(inputs)?,
            ));
        }
        if header.get("end").is_some() {
            return Ok(Message::End);
        }
        Ok(Message::Shutdown)
    }

    fn reply(&mut self, header: &Value, tables: &[Table]) -> Result<()> {
        self.connection.send(header)?;
        for table in tables {
            self.connection.send_table(table)?;
        }
        self.connection.flush()
    }

    /// Answer a describe with what Quail plans the configuration with.
    ///
    /// # Errors
    ///
    /// The connection fails.
    pub fn described(&mut self, description: &Value) -> Result<()> {
        self.reply(&json!({"described": description}), &[])
    }

    /// Answer a boot with its record.
    ///
    /// # Errors
    ///
    /// The connection fails.
    pub fn booted(&mut self, record: &Value) -> Result<()> {
        self.reply(&json!({"booted": record}), &[])
    }

    /// Say the region's stages can start.
    ///
    /// # Errors
    ///
    /// The connection fails.
    pub fn ready(&mut self) -> Result<()> {
        self.reply(&json!({"ready": true}), &[])
    }

    /// Send the rows of a stage's outputs, which ends the stage.
    ///
    /// # Errors
    ///
    /// The connection fails.
    pub fn outputs(&mut self, outputs: &[Table]) -> Result<()> {
        self.reply(&json!({"outputs": outputs.len()}), outputs)
    }

    /// Send a table the running stage finished, tagged with its node
    /// and port.
    ///
    /// # Errors
    ///
    /// The connection fails.
    pub fn finished(&mut self, table: &Table) -> Result<()> {
        self.reply(&json!({"finished": true}), std::slice::from_ref(table))
    }

    /// Ask the region a question while a stage runs, and wait for its
    /// answer.
    ///
    /// # Errors
    ///
    /// The connection fails, or the region ends it without an answer.
    pub fn ask(&mut self, question: &Value) -> Result<Answer> {
        self.reply(&json!({"ask": question}), &[])?;
        let mut header = self.connection.receive()?;
        if header.get("table").is_some() {
            return Ok(Answer::Table(self.connection.receive_table(true)?));
        }
        match header.get_mut("answer") {
            Some(answer) => Ok(Answer::Json(answer.take())),
            None => Err(failure("the region ended before it answered")),
        }
    }

    /// Answer the region's end with what the worker ran.
    ///
    /// # Errors
    ///
    /// The connection fails.
    pub fn ran(&mut self, ran: &Value, tables: &[Table]) -> Result<()> {
        self.reply(&json!({"ran": ran, "tables": tables.len()}), tables)
    }

    /// Say why the message the worker is answering failed.
    ///
    /// # Errors
    ///
    /// The connection fails.
    pub fn fail(&mut self, error: &str) -> Result<()> {
        self.reply(&json!({"error": error}), &[])
    }
}

/// A directory for a session's sockets and shared tables, removed with
/// its files when dropped. In `/dev/shm` a file is memory only.
struct Directory(PathBuf);

impl Directory {
    fn new() -> Result<Self> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let memory = Path::new("/dev/shm");
        let parent = if memory.is_dir() {
            memory.to_path_buf()
        } else {
            std::env::temp_dir()
        };
        let name = format!(
            "quail-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let path = parent.join(name);
        std::fs::create_dir(&path)?;
        Ok(Directory(path))
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A session's GPU workers and the connection to each.
pub struct Workers {
    // dropped after the connections and the processes
    directory: Directory,
    listeners: Vec<UnixListener>,
    paths: Vec<PathBuf>,
    processes: Arc<Mutex<Vec<Child>>>,
    /// The command the processes were started with, to start them again.
    command: Option<Vec<String>>,
    connections: Vec<Arc<Mutex<Connection>>>,
    num_regions: usize,
}

impl Workers {
    /// Listen for `gpus` workers, one socket each.
    ///
    /// # Errors
    ///
    /// The directory or a socket cannot be made.
    pub fn bind(gpus: usize) -> Result<Self> {
        let directory = Directory::new()?;
        let paths: Vec<PathBuf> = (0..gpus)
            .map(|gpu| directory.0.join(format!("gpu{gpu}.sock")))
            .collect();
        let listeners = paths
            .iter()
            .map(|path| {
                let listener = UnixListener::bind(path)?;
                listener.set_nonblocking(true)?;
                Ok(listener)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Workers {
            directory,
            listeners,
            paths,
            processes: Arc::new(Mutex::new(Vec::new())),
            command: None,
            connections: Vec::new(),
            num_regions: 0,
        })
    }

    /// Each worker's socket, by GPU.
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// Start one process per GPU: `command`, then the worker's socket
    /// and its GPU's index; the process sees that GPU alone.
    ///
    /// # Errors
    ///
    /// The command is empty or does not start.
    pub fn spawn(&mut self, command: &[String]) -> Result<()> {
        let (program, arguments) = command
            .split_first()
            .ok_or_else(|| failure("a worker needs a command"))?;
        for (gpu, path) in self.paths.iter().enumerate() {
            let process = Command::new(program)
                .args(arguments)
                .arg(path)
                .arg(gpu.to_string())
                .env("CUDA_VISIBLE_DEVICES", gpu.to_string())
                .spawn()
                .map_err(|error| failure(format!("cannot start {program}: {error}")))?;
            self.processes
                .lock()
                .expect("the processes lock")
                .push(process);
        }
        self.command = Some(command.to_vec());
        Ok(())
    }

    /// Wait for every worker to connect.
    ///
    /// # Errors
    ///
    /// A worker exits, or does not connect in time.
    pub fn connect(&mut self) -> Result<()> {
        let started = std::time::Instant::now();
        for (gpu, listener) in self.listeners.iter().enumerate() {
            let stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {}
                    Err(error) => return Err(error.into()),
                }
                if !alive(&mut self.processes.lock().expect("the processes lock")) {
                    return Err(failure("a GPU worker exited before it connected"));
                }
                if started.elapsed() > PATIENCE {
                    return Err(failure(format!("GPU worker {gpu} did not connect")));
                }
                std::thread::sleep(POLL);
            };
            stream.set_nonblocking(false)?;
            let connection = Connection::new(stream, &self.paths[gpu], "quail")?;
            self.connections.push(Arc::new(Mutex::new(connection)));
        }
        Ok(())
    }

    /// A handle that stops the worker processes from another thread,
    /// while a region holds the workers.
    pub fn interrupter(&self) -> Interrupter {
        Interrupter(Arc::clone(&self.processes))
    }

    /// Start the worker processes again, with the command they were
    /// started with, when one has exited; the models load again.
    ///
    /// # Errors
    ///
    /// The workers were not started by a command, or do not start.
    pub fn revive(&mut self) -> Result<()> {
        if self.alive() {
            return Ok(());
        }
        let command = (self.command.clone())
            .ok_or_else(|| failure("a GPU worker exited, and Quail did not start it"))?;
        self.connections.clear();
        for mut process in self.processes.lock().expect("the processes lock").drain(..) {
            let _ = process.kill();
            let _ = process.wait();
        }
        self.spawn(&command)?;
        self.connect()
    }

    /// Whether every worker process is running.
    pub fn alive(&mut self) -> bool {
        alive(&mut self.processes.lock().expect("the processes lock"))
    }

    /// What Quail plans the regions of `config` with, as the first
    /// worker describes it.
    ///
    /// # Errors
    ///
    /// There is no worker, the worker refuses the configuration, or
    /// its connection fails.
    pub fn describe(&self, config: &Value) -> Result<Value> {
        let worker = (self.connections.first()).ok_or_else(|| failure("there is no GPU worker"))?;
        worker.lock().expect("the worker lock").describe(config)
    }

    /// Prepare every worker's GPU for a plan, at the same time; returns
    /// each worker's boot record.
    ///
    /// # Errors
    ///
    /// A worker fails to boot or its connection fails.
    pub fn boot(&self, setup: &Value) -> Result<Vec<Value>> {
        std::thread::scope(|scope| {
            let booting: Vec<_> = (self.connections.iter())
                .map(|connection| {
                    scope.spawn(move || connection.lock().expect("the worker lock").boot(setup))
                })
                .collect();
            // every answer is read before an error is returned, so each
            // connection is ready for its next message
            let booted: Vec<Result<Value>> = booting
                .into_iter()
                .map(|thread| {
                    thread
                        .join()
                        .unwrap_or_else(|_| Err(failure("a boot panicked")))
                })
                .collect();
            booted.into_iter().collect()
        })
    }

    /// Run a region on the workers: begin it on each, run `stages` over
    /// the connections, and end it; returns what each worker ran.
    ///
    /// Every worker that began the region is told it ended, whether or
    /// not the region failed.
    ///
    /// # Errors
    ///
    /// A table does not write, a worker fails, or `stages` fails.
    pub fn run_region(
        &mut self,
        region: &Value,
        tables: &[(String, Table)],
        stages: impl FnOnce(&[Arc<Mutex<Connection>>]) -> Result<()>,
    ) -> Result<Vec<Ran>> {
        self.num_regions += 1;
        // each table is written once; every worker maps the same file
        let mut shared = Vec::with_capacity(tables.len());
        let mut ran = Ok(());
        for (index, (name, table)) in tables.iter().enumerate() {
            let path = (self.directory.0).join(format!("region{}.{index}", self.num_regions));
            ran = write_table(&path, table);
            shared.push((name.clone(), path.to_string_lossy().into_owned()));
            if ran.is_err() {
                break;
            }
        }
        for connection in &self.connections {
            if ran.is_ok() {
                ran = (connection.lock().expect("the worker lock")).begin(region, &shared);
            }
        }
        // the workers hold their mappings; the names can go
        for (_, path) in &shared {
            let _ = std::fs::remove_file(path);
        }
        if ran.is_ok() {
            ran = stages(&self.connections);
        }
        let mut found = Vec::with_capacity(self.connections.len());
        for connection in &self.connections {
            match connection.lock().expect("the worker lock").end() {
                Ok(Some(worker)) => found.push(worker),
                Ok(None) => {}
                Err(error) if ran.is_ok() => ran = Err(error),
                Err(_) => {}
            }
        }
        ran.map(|()| found)
    }
}

/// Stops a session's worker processes: the region they run fails, and
/// the next region starts them again.
#[derive(Debug, Clone)]
pub struct Interrupter(Arc<Mutex<Vec<Child>>>);

impl Interrupter {
    /// Kill every worker process.
    pub fn interrupt(&self) {
        for process in self.0.lock().expect("the processes lock").iter_mut() {
            let _ = process.kill();
        }
    }
}

fn alive(processes: &mut [Child]) -> bool {
    processes
        .iter_mut()
        .all(|process| matches!(process.try_wait(), Ok(None)))
}

impl Drop for Workers {
    /// Tell each worker to stop, wait a moment, then kill what is left.
    fn drop(&mut self) {
        for connection in self.connections.drain(..) {
            if let Ok(mut connection) = connection.lock() {
                let _ = connection.send(&json!({"shutdown": true}));
                let _ = connection.flush();
            }
        }
        let started = std::time::Instant::now();
        let running = |processes: &mut [Child]| {
            (processes.iter_mut()).any(|process| matches!(process.try_wait(), Ok(None)))
        };
        let mut processes = self.processes.lock().expect("the processes lock");
        while running(&mut processes) && started.elapsed() < Duration::from_secs(1) {
            std::thread::sleep(POLL);
        }
        for process in processes.iter_mut() {
            let _ = process.kill();
            let _ = process.wait();
        }
    }
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
        for header in [json!(null), json!({}), json!({"outputs": -1})] {
            assert!(count(&header, "outputs").is_err());
        }
        assert_eq!(count(&json!({"outputs": 0}), "outputs").unwrap(), 0);
    }

    #[test]
    fn a_worker_boots_runs_a_region_and_reports_what_it_ran() {
        let mut workers = Workers::bind(1).unwrap();
        let socket = workers.paths()[0].clone();
        let worker = std::thread::spawn(move || {
            let mut end = WorkerEnd::connect(&socket).unwrap();
            let mut seen = Vec::new();
            loop {
                match end.read().unwrap() {
                    Message::Describe(config) => {
                        end.described(&json!({"layout": config["model"]})).unwrap()
                    }
                    Message::Boot(setup) => end.booted(&json!({"model": setup["model"]})).unwrap(),
                    Message::Region(region, tables) => {
                        seen.push(format!("{} {}", region["name"], tables[0].0));
                        end.ready().unwrap();
                    }
                    Message::Stage(stage, inputs) => {
                        let Answer::Json(answer) = end.ask(&json!("more?")).unwrap() else {
                            panic!("a table");
                        };
                        seen.push(format!("stage {stage} {answer}"));
                        end.finished(&inputs[0]).unwrap();
                        end.outputs(&inputs).unwrap();
                    }
                    Message::End => end.ran(&json!({"stages": 1}), &[]).unwrap(),
                    Message::Shutdown => return seen,
                }
            }
        });
        workers.connect().unwrap();
        let described = workers.describe(&json!({"model": "m"})).unwrap();
        assert_eq!(described, json!({"layout": "m"}));
        let booted = workers.boot(&json!({"model": "m"})).unwrap();
        assert_eq!(booted, vec![json!({"model": "m"})]);

        use datafusion::arrow::array::{Int32Array, RecordBatch};
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let table: Table = (schema, vec![batch]);
        let reported = Arc::new(Mutex::new(0));
        let rows = Arc::clone(&reported);
        let region = Region {
            report: Box::new(move |(_, batches)| {
                *rows.lock().unwrap() += batches[0].num_rows();
                Ok(())
            }),
            ask: Box::new(|_| Ok(Answer::Json(json!("no")))),
        };
        let ran = workers
            .run_region(
                &json!({"name": "r"}),
                &[("documents:r".into(), table.clone())],
                |connections| {
                    let outputs = (connections[0].lock().unwrap()).call(
                        0,
                        std::slice::from_ref(&table),
                        &region,
                    )?;
                    assert_eq!(outputs[0].1, table.1);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(ran[0].0, json!({"stages": 1}));
        assert_eq!(*reported.lock().unwrap(), 3);
        let directory = workers.directory.0.clone();
        // only the socket is left: every shared table's file is gone
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
        drop(workers);
        assert_eq!(
            worker.join().unwrap(),
            vec!["\"r\" documents:r", "stage 0 \"no\""]
        );
        assert!(!directory.exists());
    }

    #[test]
    fn a_worker_that_exits_before_it_connects_is_an_error() {
        let mut workers = Workers::bind(1).unwrap();
        workers.spawn(&["true".to_string()]).unwrap();
        let error = workers.connect().unwrap_err();
        assert!(error.to_string().contains("exited before it connected"));
    }
}
