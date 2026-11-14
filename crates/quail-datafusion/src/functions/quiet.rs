//! Run a PDF reader so that its panics print nothing.
//!
//! `AI.PARSE_DOCUMENT` and `AI.PDF_PAGES` read PDFs with crates that
//! panic on some malformed files. Each function answers NULL for such a
//! file, so the panic message is noise.

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Once;

thread_local! {
    /// Whether this thread is reading a PDF, whose panics print nothing.
    static READING: Cell<bool> = const { Cell::new(false) };
}

/// Wrap the process's panic hook once, so a panic while this thread
/// reads a PDF prints nothing; every other panic reaches the hook as
/// before.
fn quiet_read_panics() {
    static QUIET: Once = Once::new();
    QUIET.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !READING.with(Cell::get) {
                previous(info);
            }
        }));
    });
}

/// What `read` returns, or None when `read` panics; the panic prints
/// nothing.
pub(crate) fn quietly<T>(read: impl FnOnce() -> T) -> Option<T> {
    quiet_read_panics();
    let outer = READING.with(|reading| reading.replace(true));
    let result = catch_unwind(AssertUnwindSafe(read));
    READING.with(|reading| reading.set(outer));
    result.ok()
}
