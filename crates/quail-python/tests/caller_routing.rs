//! CPU-only concurrent callers, including sessions sharing one executor.

use std::path::PathBuf;
use std::sync::{Arc, Once, mpsc};
use std::time::Duration;

use datafusion::arrow::array::{RecordBatch, StringArray};
use datafusion::common::Result;
use datafusion::datasource::MemTable;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use quail_python::{Executor, Session, native::_quail_native};

const ENGINE: &std::ffi::CStr = c"
import threading
from quail.specs import QWEN3_4B_FP8

class Engine:
    turn = tuple(QWEN3_4B_FP8.turn)
    layout = QWEN3_4B_FP8.prompt_layout
    rules = None

    def __init__(self):
        self.callers = {}
        self.entered = threading.Event()
        self.release = threading.Event()
        self.active = 0
        self.calls = 0
        self.fail = False

    def mark_caller(self, tag):
        # Foreign Rust threads may get a new Python thread state on each
        # attach, so threading.local does not survive between callbacks.
        self.callers[threading.get_ident()] = tag

    def run_region(self, payload, *_args):
        tag = self.callers[threading.get_ident()]
        assert f'q{tag}:' in payload, 'wrong query caller thread'
        self.active += 1
        try:
            assert self.active == 1, 'overlapping engine calls'
            self.calls += 1
            if self.calls == 1:
                self.entered.set()
                assert self.release.wait(5), 'caller was not released'
                if self.fail:
                    raise RuntimeError('first query sentinel')
        finally:
            self.active -= 1
        return None

engine = Engine()
";

fn engine() -> Py<PyAny> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    static INITIALIZE: Once = Once::new();
    INITIALIZE.call_once(|| {
        if std::env::var_os("VIRTUAL_ENV").is_none() {
            // SAFETY: before this test binary starts Python or any worker.
            unsafe { std::env::set_var("VIRTUAL_ENV", root.join(".venv")) };
        }
        pyo3::append_to_inittab!(_quail_native);
        quail_python::start_python().expect("Python starts");
    });
    Python::attach(|py| {
        PyModule::from_code(py, ENGINE, c"caller_engine.py", c"caller_engine")
            .unwrap()
            .getattr("engine")
            .unwrap()
            .unbind()
    })
}

fn session(executor: Arc<Executor>) -> Result<Arc<Session>> {
    let session = Session::with_executor(executor)?;
    let batch = RecordBatch::try_from_iter([
        ("id", Arc::new(StringArray::from(vec!["r0"])) as _),
        ("body", Arc::new(StringArray::from(vec!["a review"])) as _),
    ])?;
    session.register_table(
        "reviews",
        Arc::new(MemTable::try_new(batch.schema(), vec![vec![batch]])?),
    )?;
    Ok(Arc::new(session))
}

#[test]
fn queries_own_their_caller_route_until_success_or_failure() -> Result<()> {
    for shared_session in [true, false] {
        for fail_first in [false, true] {
            let engine = engine();
            Python::attach(|py| engine.bind(py).setattr("fail", fail_first).unwrap());
            let executor = Arc::new(Executor::with_engine(Python::attach(|py| {
                engine.clone_ref(py)
            }))?);
            let first = session(Arc::clone(&executor))?;
            let second = if shared_session {
                Arc::clone(&first)
            } else {
                session(executor)?
            };
            let (done, finished) = mpsc::channel();
            let (started, starting) = mpsc::channel();
            let run = |session: Arc<Session>, index: usize| {
                let engine = Python::attach(|py| engine.clone_ref(py));
                let done = done.clone();
                let started = started.clone();
                std::thread::spawn(move || {
                    Python::attach(|py| engine.call_method1(py, "mark_caller", (index,)).unwrap());
                    started.send(index).unwrap();
                    let query = format!(
                        "SELECT r.id FROM reviews r WHERE AI.IF(PROMPT('q{index}: {{0}}', r.body))"
                    );
                    let result = session.sql(&query);
                    done.send((index, result)).unwrap();
                })
            };
            let one = run(first, 0);
            assert_eq!(starting.recv_timeout(Duration::from_secs(10)).unwrap(), 0);
            let entered = Python::attach(|py| {
                engine
                    .bind(py)
                    .getattr("entered")
                    .unwrap()
                    .call_method1("wait", (10,))
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            });
            assert!(
                entered,
                "first query did not enter engine: {:?}",
                finished.try_recv()
            );
            let two = run(Arc::clone(&second), 1);
            assert_eq!(starting.recv_timeout(Duration::from_secs(10)).unwrap(), 1);
            // The first callback releases the GIL while blocked, allowing a
            // second query to expose the old shared-route race.
            std::thread::sleep(Duration::from_millis(100));
            Python::attach(|py| {
                engine
                    .bind(py)
                    .getattr("release")
                    .unwrap()
                    .call_method0("set")
                    .unwrap();
            });
            for _ in 0..2 {
                let (index, result) = finished.recv_timeout(Duration::from_secs(15)).unwrap();
                if index == 0 && fail_first {
                    assert!(
                        result
                            .unwrap_err()
                            .to_string()
                            .contains("first query sentinel")
                    );
                } else {
                    result?;
                }
            }
            one.join().unwrap();
            two.join().unwrap();
            // An error before any model region must release ownership too.
            assert!(second.sql("SELECT * FROM missing_table").is_err());
            Python::attach(|py| engine.call_method1(py, "mark_caller", (2,)).unwrap());
            second.sql("SELECT r.id FROM reviews r WHERE AI.IF(PROMPT('q2: {0}', r.body))")?;
            Python::attach(|py| {
                assert_eq!(
                    engine
                        .bind(py)
                        .getattr("calls")
                        .unwrap()
                        .extract::<usize>()
                        .unwrap(),
                    3
                );
            });
        }
    }
    Ok(())
}
