use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Once;

thread_local! {
    static READING: Cell<bool> = const { Cell::new(false) };
}

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

pub(crate) fn quietly<T>(read: impl FnOnce() -> T) -> Option<T> {
    quiet_read_panics();
    let outer = READING.with(|reading| reading.replace(true));
    let result = catch_unwind(AssertUnwindSafe(read));
    READING.with(|reading| reading.set(outer));
    result.ok()
}
