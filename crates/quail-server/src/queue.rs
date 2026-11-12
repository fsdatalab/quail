//! The server's submitted queries: a queue that runs one at a time and
//! keeps each query's status and rows on disk.
//!
//! Each submission has a directory under the server's data directory:
//! `status.json`, and once it succeeded `result.arrow` (its rows),
//! `report.json` (each model region's report), `table-<n>.arrow` (the
//! rows of each operator port, as `status.json` lists them), and
//! `answers.json` (the answer events of its run). A server that starts
//! on the same directory lists the earlier submissions, and marks one
//! that had not finished as failed.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use quail::shared::{Table, read_table, write_table};
use quail::{Observer, RegionEvent, Session};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// The message of a submission the server stopped before it finished.
const STOPPED: &str = "the server stopped before the query finished";

/// One submitted query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Submission {
    pub id: String,
    pub sql: String,
    /// The SQL dialect the query is read in; None is Snowflake.
    #[serde(default)]
    pub dialect: Option<String>,
    /// "queued", "running", "succeeded", "failed", or "cancelled".
    pub state: String,
    /// Why the query failed or was cancelled.
    pub error: Option<String>,
    /// The most seconds the query may run.
    pub timeout_s: Option<f64>,
    /// Seconds since 1970 at each step.
    pub submitted_at: f64,
    pub started_at: Option<f64>,
    pub finished_at: Option<f64>,
    /// How many rows the query returned.
    pub num_rows: Option<usize>,
    /// A number that grows each time this status changes.
    #[serde(default)]
    pub revision: u64,
    /// Each model region's plan, once it is planned.
    #[serde(default)]
    pub plans: Vec<PlanShown>,
    /// Whether the workers have had the model loaded for a region.
    #[serde(default)]
    pub ready: bool,
    /// How many answer events the run has given so far.
    #[serde(default)]
    pub num_answers: usize,
    /// The operator port tables saved once the query succeeded; a
    /// table's place in this list names it.
    #[serde(default)]
    pub tables: Vec<SavedTable>,
}

/// One model region's plan, as the status shows it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanShown {
    /// The plan as the planner's explain writes it.
    pub text: String,
    /// The seconds the planner expects the region to take.
    pub estimated_seconds: f64,
}

/// The rows of one operator port of a region that ran.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedTable {
    /// Which of the query's model regions, in run order.
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
    /// The running submission, and why it was stopped, once it was.
    running: Option<(String, Option<String>)>,
    /// The answer events of the submissions run since the server started.
    answers: BTreeMap<String, Arc<Vec<Value>>>,
    closed: bool,
}

/// The server's submissions and the thread that runs them.
pub struct Queue {
    directory: PathBuf,
    session: Arc<Session>,
    state: Mutex<State>,
    changed: Condvar,
    num_submitted: AtomicUsize,
    /// The thread that runs the submissions.
    runner: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Queue {
    /// Open the queue on `directory`, with the submissions an earlier
    /// server left there, and start running what is submitted.
    ///
    /// # Errors
    ///
    /// The directory cannot be made or read.
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

    /// Submit a query; returns its submission. `id` names it, so that
    /// submitting the same query under the same id twice runs it once.
    ///
    /// # Errors
    ///
    /// The id names another query, or the status does not save.
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

    /// A submission's status as JSON, with the running region's
    /// progress while it runs; None for an unknown id.
    pub fn status(&self, id: &str) -> Option<Value> {
        let state = self.state.lock().expect("the queue lock");
        let found = state.submissions.get(id)?;
        let mut status = serde_json::to_value(found).ok()?;
        if found.state == "running" {
            status["progress"] = json!(self.session.progress());
        }
        Some(status)
    }

    /// A submission's status once its revision is past `after`, or
    /// after `wait` at the latest; None for an unknown id.
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

    /// The answer events of a submission's run from the event numbered
    /// `after` on, `limit` at most: JSON with `answers` and `next`, the
    /// number of the event after the last one returned.
    ///
    /// # Errors
    ///
    /// The id is unknown.
    pub fn answers(&self, id: &str, after: usize, limit: usize) -> Result<Value, String> {
        let held = {
            let state = self.state.lock().expect("the queue lock");
            if !state.submissions.contains_key(id) {
                return Err(format!("no submission {id}"));
            }
            state.answers.get(id).cloned()
        };
        // an earlier server's run left its events in a file
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

    /// The report of each model region of a submission that succeeded,
    /// as JSON text: `{"regions": [...]}`.
    ///
    /// # Errors
    ///
    /// The id is unknown, or the submission has no report.
    pub fn report(&self, id: &str) -> Result<String, String> {
        self.succeeded(id)?;
        std::fs::read_to_string(self.directory.join(id).join("report.json"))
            .map_err(|error| format!("the submission {id} has no report: {error}"))
    }

    /// The rows of the operator port a submission's status lists at
    /// `index` of its `tables`.
    ///
    /// # Errors
    ///
    /// The id is unknown, or the submission has no such table.
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

    /// Record what a region of the running submission `id` did.
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
            // answers come often: they are saved when the query ends
            RegionEvent::Answers(answered) => {
                let events = state.answers.entry(id.to_string()).or_default();
                Arc::make_mut(events).push(answered.clone());
                found.num_answers = events.len();
                found.revision += 1;
            }
        }
        self.changed.notify_all();
    }

    /// Save the report and the operator port tables of each region the
    /// submission `id` ran, and its answer events; returns the tables.
    fn save_regions(&self, id: &str) -> datafusion::common::Result<Vec<SavedTable>> {
        let home = self.directory.join(id);
        let mut reports = Vec::new();
        let mut saved = Vec::new();
        for (region, ran) in self.session.take_regions().iter().enumerate() {
            let (report, tables) = self.session.region_report(ran)?;
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

    /// Every submission, oldest first.
    pub fn list(&self) -> Vec<Submission> {
        let state = self.state.lock().expect("the queue lock");
        let mut found: Vec<Submission> = state.submissions.values().cloned().collect();
        found.sort_by(|one, two| one.submitted_at.total_cmp(&two.submitted_at));
        found
    }

    /// Cancel a submission: one that waits does not run, and one that
    /// runs is stopped. A finished submission stays as it is.
    ///
    /// # Errors
    ///
    /// The id is unknown.
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

    /// Stop the running submission `id`, for `reason`.
    fn stop(&self, state: &mut State, id: &str, reason: &str) {
        if let Some((running, stopped)) = &mut state.running
            && running == id
            && stopped.is_none()
        {
            *stopped = Some(reason.to_string());
            self.session.interrupt();
        }
    }

    /// The rows of a submission that succeeded, read over its file's
    /// mapped pages.
    ///
    /// # Errors
    ///
    /// The id is unknown, or the submission has no rows.
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

    /// Stop running submissions, and wait for the thread that runs
    /// them: the running one is stopped, and fails.
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

    /// Run each submission in turn, until the queue closes.
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
            // a watcher stops the query at its timeout
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
            let ran = (self
                .session
                .sql_in_dialect(&found.sql, found.dialect.as_deref()))
            .and_then(|table| {
                let rows: usize = table.1.iter().map(|batch| batch.num_rows()).sum();
                let path = self.directory.join(&found.id).join("result.arrow");
                let _ = std::fs::remove_file(&path);
                write_table(&path, &table)?;
                Ok((rows, self.save_regions(&found.id)?))
            });
            self.session.observe(None, false);
            // a query that failed leaves its regions behind
            drop(self.session.take_regions());
            drop(done);
            if let Some(watcher) = watcher {
                let _ = watcher.join();
            }
            let mut state = self.state.lock().expect("the queue lock");
            let stopped = state.running.take().and_then(|(_, stopped)| stopped);
            // under the lock, so no stop for this query comes after
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

/// Write a submission's status, whole or not at all, as a new revision.
fn save(directory: &Path, found: &mut Submission) -> std::io::Result<()> {
    found.revision += 1;
    let home = directory.join(&found.id);
    std::fs::create_dir_all(&home)?;
    let written = home.join("status.json.new");
    std::fs::write(&written, serde_json::to_vec_pretty(found)?)?;
    std::fs::rename(written, home.join("status.json"))
}
