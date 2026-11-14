//! `quail-server --model <model> --device <device> [--gpus <n>]
//! [--backend <backend>] [--listen <address>] [--data <directory>]`
//!
//! With `QUAIL_SERVER_TOKEN` set, the server refuses every call whose
//! `authorization` header is not `Bearer <token>`.
//!
//! Starts a GPU worker per GPU and serves a Quail session over Arrow
//! Flight ([`quail_server::QuailService`]) at the address, by default
//! `127.0.0.1:50051`. Submitted queries and their rows are kept in the
//! data directory, by default `quail-server-data`. Ctrl-C or SIGTERM
//! stops the server and its workers.

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    quail_server::run(std::env::args().skip(1))
}
