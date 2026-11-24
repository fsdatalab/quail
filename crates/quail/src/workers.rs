use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use datafusion::common::{DataFusionError, Result};
use serde_json::{Value, json};

use crate::shared::{Table, read_table, write_table};

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

pub enum Answer {
    Json(Value),
    Table(Table),
}

pub type Report = Box<dyn Fn(Table) -> Result<()> + Send + Sync>;

pub type Ask = Box<dyn Fn(&Value) -> Result<Answer> + Send + Sync>;

pub struct Region {
    pub report: Report,
    pub ask: Ask,
}

pub type Ran = (Value, Vec<Table>);

pub struct Connection {
    reader: BufReader<UnixStream>,
    writer: BufWriter<UnixStream>,
    shared: String,
    num_written: usize,
    in_region: bool,
    broken: bool,
}

impl Connection {
    fn new(stream: UnixStream, socket: &Path, end: &str) -> Result<Self> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let generation = NEXT.fetch_add(1, Ordering::Relaxed);
        Ok(Connection {
            reader: BufReader::new(stream.try_clone()?),
            writer: BufWriter::new(stream),
            shared: format!(
                "{}.{end}{}-{generation}",
                socket.display(),
                std::process::id()
            ),
            num_written: 0,
            in_region: false,
            broken: false,
        })
    }

    fn send(&mut self, header: &Value) -> Result<()> {
        write_frame(&mut self.writer, header.to_string().as_bytes())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(self.writer.flush()?)
    }

    fn receive(&mut self) -> Result<Value> {
        self.receive_reply()?
    }

    fn receive_reply(&mut self) -> Result<Result<Value>> {
        let header: Value =
            serde_json::from_slice(&read_frame(&mut self.reader)?).map_err(failure)?;
        Ok(match header.get("error").and_then(Value::as_str) {
            Some(error) => Err(failure(format!("a GPU worker failed:\n{error}"))),
            None => Ok(header),
        })
    }

    fn next_header(&mut self) -> Result<Option<Value>> {
        let frame = match read_frame(&mut self.reader) {
            Ok(frame) => frame,
            Err(DataFusionError::IoError(error)) if error.kind() == ErrorKind::UnexpectedEof => {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        serde_json::from_slice(&frame).map(Some).map_err(failure)
    }

    fn send_table(&mut self, table: &Table) -> Result<()> {
        let path = format!("{}.{}", self.shared, self.num_written);
        self.num_written += 1;
        write_table(Path::new(&path), table)?;
        write_frame(&mut self.writer, path.as_bytes())
    }

    fn receive_table(&mut self, remove: bool) -> Result<Table> {
        let frame = read_frame(&mut self.reader)?;
        if let Ok(header) = serde_json::from_slice::<Value>(&frame)
            && let Some(error) = header.get("error").and_then(Value::as_str)
        {
            return Err(failure(format!("a GPU worker failed:\n{error}")));
        }
        let path = String::from_utf8(frame).map_err(failure)?;
        read_table(Path::new(&path), remove)
    }

    fn receive_tables(&mut self, count: usize) -> Result<Vec<Table>> {
        (0..count).map(|_| self.receive_table(true)).collect()
    }
}

impl Connection {
    fn describe(&mut self, config: &Value) -> Result<Value> {
        self.send(&json!({"describe": config}))?;
        self.flush()?;
        let mut header = self.receive()?;
        Ok(header["described"].take())
    }

    fn boot(&mut self, setup: &Value) -> Result<Value> {
        self.send(&json!({"boot": setup}))?;
        self.flush()?;
        let mut header = self.receive()?;
        Ok(header["booted"].take())
    }

    fn begin(&mut self, region: &Value, tables: &[(String, String)]) -> Result<()> {
        self.send(&json!({"region": region, "tables": tables}))?;
        self.flush()?;
        self.receive()?;
        self.in_region = true;
        Ok(())
    }

    pub(crate) fn call(
        &mut self,
        stage: usize,
        inputs: &[Table],
        region: &Region,
    ) -> Result<Vec<Table>> {
        let called = self.exchange(stage, inputs, region);
        self.broken |= called.is_err();
        called?
    }

    fn exchange(
        &mut self,
        stage: usize,
        inputs: &[Table],
        region: &Region,
    ) -> Result<Result<Vec<Table>>> {
        self.send(&json!({"stage": stage, "inputs": inputs.len()}))?;
        for input in inputs {
            self.send_table(input)?;
        }
        self.flush()?;
        loop {
            let header = match self.receive_reply()? {
                Ok(header) => header,
                Err(failed) => return Ok(Err(failed)),
            };
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
                return Ok(Ok(self.receive_tables(count(&header, "outputs")?)?));
            }
        }
    }

    fn end(&mut self) -> Result<Option<Ran>> {
        if !std::mem::take(&mut self.in_region) || self.broken {
            return Ok(None);
        }
        self.send(&json!({"end": true}))?;
        self.flush()?;
        let mut header = self.receive()?;
        let tables = self.receive_tables(count(&header, "tables")?)?;
        Ok(Some((header["ran"].take(), tables)))
    }
}

pub enum Message {
    Describe(Value),
    Boot(Value),
    Region(Value, Vec<(String, Table)>),
    Stage(usize, Vec<Table>),
    End,
    Shutdown,
}

pub struct WorkerEnd {
    connection: Connection,
}

impl WorkerEnd {
    pub fn connect(socket: &Path) -> Result<Self> {
        let stream = UnixStream::connect(socket)
            .map_err(|error| failure(format!("cannot reach {}: {error}", socket.display())))?;
        Ok(WorkerEnd {
            connection: Connection::new(stream, socket, "worker")?,
        })
    }

    pub fn read(&mut self) -> Result<Message> {
        let connection = &mut self.connection;
        let Some(mut header) = connection.next_header()? else {
            return Ok(Message::Shutdown);
        };
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

    pub fn described(&mut self, description: &Value) -> Result<()> {
        self.reply(&json!({"described": description}), &[])
    }

    pub fn booted(&mut self, record: &Value) -> Result<()> {
        self.reply(&json!({"booted": record}), &[])
    }

    pub fn ready(&mut self) -> Result<()> {
        self.reply(&json!({"ready": true}), &[])
    }

    pub fn outputs(&mut self, outputs: &[Table]) -> Result<()> {
        self.reply(&json!({"outputs": outputs.len()}), outputs)
    }

    pub fn finished(&mut self, table: &Table) -> Result<()> {
        self.reply(&json!({"finished": true}), std::slice::from_ref(table))
    }

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

    pub fn ran(&mut self, ran: &Value, tables: &[Table]) -> Result<()> {
        self.reply(&json!({"ran": ran, "tables": tables.len()}), tables)
    }

    pub fn fail(&mut self, error: &str) -> Result<()> {
        self.reply(&json!({"error": error}), &[])
    }
}

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

pub struct Workers {
    directory: Directory,
    listeners: Vec<UnixListener>,
    paths: Vec<PathBuf>,
    processes: Arc<Mutex<Vec<Child>>>,
    interrupted: Arc<AtomicBool>,
    command: Option<Vec<String>>,
    connections: Vec<Arc<Mutex<Connection>>>,
    num_regions: usize,
}

impl Workers {
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
            interrupted: Arc::new(AtomicBool::new(false)),
            command: None,
            connections: Vec::new(),
            num_regions: 0,
        })
    }

    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    pub fn spawn(&mut self, command: &[String]) -> Result<()> {
        let (program, arguments) = command
            .split_first()
            .ok_or_else(|| failure("a worker needs a command"))?;
        let visible: Vec<String> = std::env::var("CUDA_VISIBLE_DEVICES")
            .map(|devices| {
                devices
                    .split(',')
                    .map(|device| device.trim().to_string())
                    .filter(|device| !device.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        for (gpu, path) in self.paths.iter().enumerate() {
            let device = visible.get(gpu).cloned().unwrap_or_else(|| gpu.to_string());
            let process = Command::new(program)
                .args(arguments)
                .arg(path)
                .arg(gpu.to_string())
                .env("CUDA_VISIBLE_DEVICES", device)
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

    pub fn interrupter(&self) -> Interrupter {
        Interrupter {
            processes: Arc::clone(&self.processes),
            interrupted: Arc::clone(&self.interrupted),
        }
    }

    pub fn check_interrupted(&self) -> Result<()> {
        match self.interrupted.load(Ordering::SeqCst) {
            true => Err(failure("the query was stopped")),
            false => Ok(()),
        }
    }

    pub fn revive(&mut self) -> Result<()> {
        let connected = self.connections.len() == self.listeners.len()
            && (self.connections.iter())
                .all(|connection| !connection.lock().expect("the worker lock").broken);
        if connected && self.alive() {
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

    pub fn alive(&mut self) -> bool {
        alive(&mut self.processes.lock().expect("the processes lock"))
    }

    pub fn describe(&self, config: &Value) -> Result<Value> {
        let worker = (self.connections.first()).ok_or_else(|| failure("there is no GPU worker"))?;
        worker.lock().expect("the worker lock").describe(config)
    }

    pub fn boot(&self, setup: &Value) -> Result<Vec<Value>> {
        std::thread::scope(|scope| {
            let booting: Vec<_> = (self.connections.iter())
                .map(|connection| {
                    scope.spawn(move || connection.lock().expect("the worker lock").boot(setup))
                })
                .collect();
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

    pub fn run_region(
        &mut self,
        region: &Value,
        tables: &[(String, Table)],
        stages: impl FnOnce(&[Arc<Mutex<Connection>>]) -> Result<()>,
    ) -> Result<Vec<Ran>> {
        self.num_regions += 1;
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

#[derive(Debug, Clone)]
pub struct Interrupter {
    processes: Arc<Mutex<Vec<Child>>>,
    interrupted: Arc<AtomicBool>,
}

impl Interrupter {
    pub fn interrupt(&self) {
        self.interrupted.store(true, Ordering::SeqCst);
        for process in self
            .processes
            .lock()
            .expect("the processes lock")
            .iter_mut()
        {
            let _ = process.kill();
        }
    }

    pub fn resume(&self) {
        self.interrupted.store(false, Ordering::SeqCst);
    }
}

fn alive(processes: &mut [Child]) -> bool {
    processes
        .iter_mut()
        .all(|process| matches!(process.try_wait(), Ok(None)))
}

impl Drop for Workers {
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
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
        drop(workers);
        assert_eq!(
            worker.join().unwrap(),
            vec!["\"r\" documents:r", "stage 0 \"no\""]
        );
        assert!(!directory.exists());
    }

    fn one_row_table() -> Table {
        use datafusion::arrow::array::{Int32Array, RecordBatch};
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        (schema, vec![batch])
    }

    fn lost_region(on_stage: fn(&mut WorkerEnd), ask: Ask) -> (Result<Vec<Ran>>, Workers) {
        let mut workers = Workers::bind(1).unwrap();
        let socket = workers.paths()[0].clone();
        std::thread::spawn(move || {
            let mut end = WorkerEnd::connect(&socket).unwrap();
            loop {
                match end.read() {
                    Ok(Message::Region(..)) => end.ready().unwrap(),
                    Ok(Message::Stage(..)) => on_stage(&mut end),
                    Ok(Message::Shutdown) | Err(_) => return,
                    Ok(_) => {}
                }
            }
        });
        workers.connect().unwrap();
        let region = Region {
            report: Box::new(|_| Ok(())),
            ask,
        };
        let ran = workers.run_region(&json!({}), &[], |connections| {
            (connections[0].lock().unwrap()).call(0, &[one_row_table()], &region)?;
            Ok(())
        });
        (ran, workers)
    }

    #[test]
    fn a_worker_that_fails_to_write_its_outputs_fails_the_region_and_is_restarted() {
        let (ran, mut workers) = lost_region(
            |end| {
                end.connection.send(&json!({"outputs": 1})).unwrap();
                end.fail("No space left on device").unwrap();
            },
            Box::new(|_| Ok(Answer::Json(json!(null)))),
        );
        assert!(ran.unwrap_err().to_string().contains("No space left"));
        assert!(workers.revive().is_err());
    }

    #[test]
    fn a_question_quail_cannot_answer_fails_the_region_and_restarts_the_worker() {
        let (ran, mut workers) = lost_region(
            |end| {
                let _ = end.ask(&json!("more?"));
            },
            Box::new(|_| Err(failure("the hint failed"))),
        );
        assert!(ran.unwrap_err().to_string().contains("the hint failed"));
        assert!(workers.revive().is_err());
    }

    #[test]
    fn workers_missing_a_connection_are_restarted() {
        let mut workers = Workers::bind(2).unwrap();
        let (stream, _) = UnixStream::pair().unwrap();
        let path = workers.paths()[0].clone();
        let connection = Connection::new(stream, &path, "quail").unwrap();
        workers.connections.push(Arc::new(Mutex::new(connection)));
        assert!(workers.revive().is_err());
    }

    #[test]
    fn each_connection_at_a_socket_names_its_own_table_files() {
        let path = PathBuf::from("/tmp/socket");
        let named = |_| {
            let (stream, _) = UnixStream::pair().unwrap();
            Connection::new(stream, &path, "worker").unwrap().shared
        };
        assert_ne!(named(0), named(1));
    }

    #[test]
    fn a_worker_that_exits_before_it_connects_is_an_error() {
        let mut workers = Workers::bind(1).unwrap();
        workers.spawn(&["true".to_string()]).unwrap();
        let error = workers.connect().unwrap_err();
        assert!(error.to_string().contains("exited before it connected"));
    }
}
