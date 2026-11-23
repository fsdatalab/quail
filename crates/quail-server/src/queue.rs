use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use arrow::array::RecordBatch;
use quail::shared::{Table, read_table, write_table};
use quail::{Observer, RegionEvent, Session};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const STOPPED: &str = "the server stopped before the query finished";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Submission {
    pub id: String,
    pub sql: String,
    #[serde(default)]
    pub dialect: Option<String>,
    pub state: String,
    pub error: Option<String>,
    pub timeout_s: Option<f64>,
    pub submitted_at: f64,
    pub started_at: Option<f64>,
    pub finished_at: Option<f64>,
    pub num_rows: Option<usize>,
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub plans: Vec<PlanShown>,
    #[serde(default)]
    pub ready: bool,
    #[serde(default)]
    pub num_answers: usize,
    #[serde(default)]
    pub tables: Vec<SavedTable>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanShown {
    pub text: String,
    pub estimated_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedTable {
    pub region: usize,
    pub node: String,
    pub port: String,
    pub num_rows: usize,
}

impl Submission {
    fn finished(&self) -> bool {
        matches!(self.state.as_str(), "succeeded" | "failed" | "cancelled")
    }
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |since| since.as_secs_f64())
}

#[derive(Default)]
struct State {
    submissions: BTreeMap<String, Submission>,
    waiting: VecDeque<String>,
    running: Option<(String, Option<String>)>,
    answers: BTreeMap<String, Arc<Vec<Value>>>,
    streaming: BTreeMap<String, Table>,
    closed: bool,
}

pub struct Queue {
    directory: PathBuf,
    session: Arc<Session>,
    state: Mutex<State>,
    changed: Condvar,
    num_submitted: AtomicUsize,
    runner: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Queue {
    pub fn open(directory: &Path, session: Arc<Session>) -> std::io::Result<Arc<Self>> {
        std::fs::create_dir_all(directory)?;
        let mut state = State::default();
        for entry in std::fs::read_dir(directory)? {
            let path = entry?.path().join("status.json");
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(mut found) = serde_json::from_str::<Submission>(&text) else {
                continue;
            };
            if !found.finished() {
                found.state = "failed".into();
                found.error = Some(STOPPED.into());
                found.finished_at = Some(now());
                save(directory, &mut found)?;
            }
            state.submissions.insert(found.id.clone(), found);
        }
        let queue = Arc::new(Queue {
            directory: directory.to_path_buf(),
            session,
            state: Mutex::new(state),
            changed: Condvar::new(),
            num_submitted: AtomicUsize::new(0),
            runner: Mutex::new(None),
        });
        let runner = Arc::clone(&queue);
        let thread = std::thread::spawn(move || runner.run());
        *queue.runner.lock().expect("the runner lock") = Some(thread);
        Ok(queue)
    }

    pub fn submit(
        &self,
        sql: &str,
        dialect: Option<String>,
        id: Option<String>,
        timeout_s: Option<f64>,
    ) -> Result<Submission, String> {
        let mut state = self.state.lock().expect("the queue lock");
        let id = id.unwrap_or_else(|| {
            let count = self.num_submitted.fetch_add(1, Ordering::Relaxed);
            format!("q{:.0}-{count}", now() * 1000.0)
        });
        if id.is_empty() || id.contains(['/', '\\', '.']) {
            return Err(format!("{id:?} is not a submission id"));
        }
        if let Some(found) = state.submissions.get(&id) {
            return match found.sql == sql && found.dialect == dialect {
                true => Ok(found.clone()),
                false => Err(format!("the id {id} names another query")),
            };
        }
        let found = Submission {
            id: id.clone(),
            sql: sql.to_string(),
            dialect,
            state: "queued".into(),
            error: None,
            timeout_s,
            submitted_at: now(),
            started_at: None,
            finished_at: None,
            num_rows: None,
            revision: 0,
            plans: Vec::new(),
            ready: false,
            num_answers: 0,
            tables: Vec::new(),
        };
        let mut found = found;
        save(&self.directory, &mut found).map_err(|error| error.to_string())?;
        state.submissions.insert(id.clone(), found.clone());
        state.waiting.push_back(id);
        self.changed.notify_all();
        Ok(found)
    }

    pub fn status(&self, id: &str) -> Option<Value> {
        let state = self.state.lock().expect("the queue lock");
        let found = state.submissions.get(id)?;
        let mut status = serde_json::to_value(found).ok()?;
        if found.state == "running" {
            status["progress"] = json!(self.session.progress());
        }
        Some(status)
    }

    pub fn status_after(&self, id: &str, after: u64, wait: Duration) -> Option<Value> {
        let deadline = std::time::Instant::now() + wait;
        let mut state = self.state.lock().expect("the queue lock");
        loop {
            let found = state.submissions.get(id)?;
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if found.revision > after || found.finished() || left.is_zero() {
                break;
            }
            state = (self.changed.wait_timeout(state, left))
                .expect("the queue lock")
                .0;
        }
        drop(state);
        self.status(id)
    }

    pub fn answers(&self, id: &str, after: usize, limit: usize) -> Result<Value, String> {
        let held = {
            let state = self.state.lock().expect("the queue lock");
            if !state.submissions.contains_key(id) {
                return Err(format!("no submission {id}"));
            }
            state.answers.get(id).cloned()
        };
        let events = match held {
            Some(events) => events,
            None => {
                let path = self.directory.join(id).join("answers.json");
                let saved = std::fs::read(path).ok();
                Arc::new(
                    saved
                        .and_then(|text| serde_json::from_slice(&text).ok())
                        .unwrap_or_default(),
                )
            }
        };
        let start = after.min(events.len());
        let end = start.saturating_add(limit).min(events.len());
        Ok(json!({"answers": &events[start..end], "next": end}))
    }

    pub fn report(&self, id: &str) -> Result<String, String> {
        self.succeeded(id)?;
        std::fs::read_to_string(self.directory.join(id).join("report.json"))
            .map_err(|error| format!("the submission {id} has no report: {error}"))
    }

    pub fn table(&self, id: &str, index: usize) -> Result<Table, String> {
        self.succeeded(id)?;
        let path = self.directory.join(id).join(format!("table-{index}.arrow"));
        read_table(&path, false).map_err(|error| error.to_string())
    }

    fn succeeded(&self, id: &str) -> Result<(), String> {
        let found = (self.status(id)).ok_or_else(|| format!("no submission {id}"))?;
        match found["state"] == "succeeded" {
            true => Ok(()),
            false => Err(format!(
                "the submission {id} is {}, and has no results",
                found["state"]
            )),
        }
    }

    fn observed(&self, id: &str, event: RegionEvent<'_>) {
        let mut state = self.state.lock().expect("the queue lock");
        let state = &mut *state;
        let Some(found) = state.submissions.get_mut(id) else {
            return;
        };
        match event {
            RegionEvent::Planned {
                explained,
                estimated_seconds,
            } => {
                found.plans.push(PlanShown {
                    text: explained.to_string(),
                    estimated_seconds,
                });
                let _ = save(&self.directory, found);
            }
            RegionEvent::Ready => {
                found.ready = true;
                let _ = save(&self.directory, found);
            }
            RegionEvent::Answers(answered) => {
                let events = state.answers.entry(id.to_string()).or_default();
                Arc::make_mut(events).push(answered.clone());
                found.num_answers = events.len();
                found.revision += 1;
            }
        }
        self.changed.notify_all();
    }

    fn save_regions(&self, id: &str) -> datafusion::common::Result<Vec<SavedTable>> {
        let home = self.directory.join(id);
        let mut reports = Vec::new();
        let mut saved = Vec::new();
        let reported = self
            .session
            .take_regions(|ran| self.session.region_report(&ran))?;
        for (region, (report, tables)) in reported.into_iter().enumerate() {
            reports.push(report);
            for table in tables {
                let metadata = table.0.metadata();
                let named = |key: &str| metadata.get(key).cloned().unwrap_or_default();
                let path = home.join(format!("table-{}.arrow", saved.len()));
                let _ = std::fs::remove_file(&path);
                write_table(&path, &table)?;
                saved.push(SavedTable {
                    region,
                    node: named("quail.node"),
                    port: named("quail.port"),
                    num_rows: table.1.iter().map(|batch| batch.num_rows()).sum(),
                });
            }
        }
        std::fs::write(
            home.join("report.json"),
            json!({"regions": reports}).to_string(),
        )?;
        let events = {
            let state = self.state.lock().expect("the queue lock");
            state.answers.get(id).cloned().unwrap_or_default()
        };
        std::fs::write(home.join("answers.json"), json!(*events).to_string())?;
        Ok(saved)
    }

    pub fn list(&self) -> Vec<Submission> {
        let state = self.state.lock().expect("the queue lock");
        let mut found: Vec<Submission> = state.submissions.values().cloned().collect();
        found.sort_by(|one, two| one.submitted_at.total_cmp(&two.submitted_at));
        found
    }

    pub fn cancel(&self, id: &str) -> Result<Submission, String> {
        let mut state = self.state.lock().expect("the queue lock");
        let unknown = || format!("no submission {id}");
        if state.submissions.get(id).ok_or_else(unknown)?.state == "queued" {
            state.waiting.retain(|waiting| waiting != id);
            let found = state.submissions.get_mut(id).ok_or_else(unknown)?;
            found.state = "cancelled".into();
            found.error = Some("cancelled before it ran".into());
            found.finished_at = Some(now());
            let _ = save(&self.directory, found);
        }
        self.stop(&mut state, id, "cancelled");
        state.submissions.get(id).cloned().ok_or_else(unknown)
    }

    fn stop(&self, state: &mut State, id: &str, reason: &str) {
        if let Some((running, stopped)) = &mut state.running
            && running == id
            && stopped.is_none()
        {
            *stopped = Some(reason.to_string());
            self.session.interrupt();
        }
    }

    pub fn rows(&self, id: &str) -> Result<Table, String> {
        let found = (self.status(id)).ok_or_else(|| format!("no submission {id}"))?;
        if found["state"] != "succeeded" {
            return Err(format!(
                "the submission {id} is {}, and has no rows",
                found["state"]
            ));
        }
        let path = self.directory.join(id).join("result.arrow");
        read_table(&path, false).map_err(|error| error.to_string())
    }

    pub fn rows_after(
        &self,
        id: &str,
        after: usize,
        wait: Duration,
    ) -> Result<(Option<arrow::datatypes::SchemaRef>, Vec<RecordBatch>, bool), String> {
        let deadline = std::time::Instant::now() + wait;
        let mut state = self.state.lock().expect("the queue lock");
        loop {
            let found = (state.submissions.get(id)).ok_or_else(|| format!("no submission {id}"))?;
            if let Some((schema, batches)) = state.streaming.get(id) {
                let rows = rows_past(batches, after);
                let left = deadline.saturating_duration_since(std::time::Instant::now());
                if !rows.is_empty() || left.is_zero() {
                    return Ok((Some(Arc::clone(schema)), rows, false));
                }
            } else if found.state == "succeeded" {
                break;
            } else if found.finished() {
                let error = found.error.clone().unwrap_or_default();
                return Err(format!("the submission {id} is {}: {error}", found.state));
            }
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                return Ok((None, Vec::new(), false));
            }
            state = (self.changed.wait_timeout(state, left))
                .expect("the queue lock")
                .0;
        }
        drop(state);
        let (schema, batches) = self.rows(id)?;
        Ok((Some(schema), rows_past(&batches, after), true))
    }

    fn run_rows(
        &self,
        id: &str,
        sql: &str,
        dialect: Option<&str>,
    ) -> datafusion::common::Result<Table> {
        let rows = self.session.sql_stream(sql, dialect)?;
        let mut state = self.state.lock().expect("the queue lock");
        state
            .streaming
            .insert(id.to_string(), (rows.schema(), Vec::new()));
        drop(state);
        self.changed.notify_all();
        for batch in rows {
            let batch = batch?;
            let mut state = self.state.lock().expect("the queue lock");
            if let Some((_, batches)) = state.streaming.get_mut(id) {
                batches.push(batch);
            }
            drop(state);
            self.changed.notify_all();
        }
        let state = self.state.lock().expect("the queue lock");
        (state.streaming.get(id).cloned()).ok_or_else(|| {
            datafusion::common::DataFusionError::Internal(format!("no rows of {id}"))
        })
    }

    pub fn close(&self) {
        {
            let mut state = self.state.lock().expect("the queue lock");
            state.closed = true;
            if let Some(id) = state.running.as_ref().map(|(id, _)| id.clone()) {
                self.stop(&mut state, &id, STOPPED);
            }
        }
        self.changed.notify_all();
        if let Some(thread) = self.runner.lock().expect("the runner lock").take() {
            let _ = thread.join();
        }
    }

    fn run(self: Arc<Self>) {
        loop {
            let found = {
                let mut state = self.state.lock().expect("the queue lock");
                let id = loop {
                    if state.closed {
                        return;
                    }
                    if let Some(id) = state.waiting.pop_front() {
                        break id;
                    }
                    state = self.changed.wait(state).expect("the queue lock");
                };
                state.running = Some((id.clone(), None));
                let found = state
                    .submissions
                    .get_mut(&id)
                    .expect("a waiting submission");
                found.state = "running".into();
                found.started_at = Some(now());
                let _ = save(&self.directory, found);
                found.clone()
            };
            let (done, finished) = std::sync::mpsc::channel::<()>();
            let watcher = found.timeout_s.map(|seconds| {
                let (queue, id) = (Arc::clone(&self), found.id.clone());
                std::thread::spawn(move || {
                    let limit = Duration::from_secs_f64(seconds.max(0.0));
                    if finished.recv_timeout(limit) == Err(RecvTimeoutError::Timeout) {
                        let mut state = queue.state.lock().expect("the queue lock");
                        let reason = format!("timed out after {seconds} s");
                        queue.stop(&mut state, &id, &reason);
                    }
                })
            });
            let observer: Observer = {
                let (queue, id) = (Arc::clone(&self), found.id.clone());
                Arc::new(move |event| queue.observed(&id, event))
            };
            self.session.observe(Some(observer), true);
            let ran = (self.run_rows(&found.id, &found.sql, found.dialect.as_deref())).and_then(
                |table| {
                    let rows: usize = table.1.iter().map(|batch| batch.num_rows()).sum();
                    let path = self.directory.join(&found.id).join("result.arrow");
                    let _ = std::fs::remove_file(&path);
                    write_table(&path, &table)?;
                    Ok((rows, self.save_regions(&found.id)?))
                },
            );
            self.session.observe(None, false);
            self.session.discard_regions();
            drop(done);
            if let Some(watcher) = watcher {
                let _ = watcher.join();
            }
            let mut state = self.state.lock().expect("the queue lock");
            let stopped = state.running.take().and_then(|(_, stopped)| stopped);
            state.streaming.remove(&found.id);
            self.session.resume();
            let found = state
                .submissions
                .get_mut(&found.id)
                .expect("the submission");
            found.finished_at = Some(now());
            match (ran, stopped) {
                (Ok((rows, tables)), _) => {
                    found.state = "succeeded".into();
                    found.num_rows = Some(rows);
                    found.tables = tables;
                }
                (Err(_), Some(reason)) if reason == "cancelled" => {
                    found.state = "cancelled".into();
                    found.error = Some("cancelled while it ran".into());
                }
                (Err(_), Some(reason)) => {
                    found.state = "failed".into();
                    found.error = Some(reason);
                }
                (Err(error), None) => {
                    found.state = "failed".into();
                    found.error = Some(error.to_string());
                }
            }
            let _ = save(&self.directory, found);
            self.changed.notify_all();
        }
    }
}

fn rows_past(batches: &[RecordBatch], after: usize) -> Vec<RecordBatch> {
    let mut skipped = 0;
    let mut rows = Vec::new();
    for batch in batches {
        let start = after.saturating_sub(skipped).min(batch.num_rows());
        skipped += batch.num_rows();
        if start < batch.num_rows() {
            rows.push(batch.slice(start, batch.num_rows() - start));
        }
    }
    rows
}

fn save(directory: &Path, found: &mut Submission) -> std::io::Result<()> {
    found.revision += 1;
    let home = directory.join(&found.id);
    std::fs::create_dir_all(&home)?;
    let written = home.join("status.json.new");
    std::fs::write(&written, serde_json::to_vec_pretty(found)?)?;
    std::fs::rename(written, home.join("status.json"))
}
