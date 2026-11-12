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

use std::sync::Arc;

use arrow_flight::flight_service_server::FlightServiceServer;
use quail::{EngineConfig, Session};
use quail_server::QuailService;
use quail_server::queue::Queue;
use tonic::transport::Server;

const USAGE: &str = "usage: quail-server --model <model> --device <device> \
[--gpus <n>] [--backend <backend>] [--listen <address>] [--data <directory>]";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut options = std::collections::HashMap::new();
    let mut arguments = std::env::args().skip(1);
    while let Some(name) = arguments.next() {
        let name = name.strip_prefix("--").ok_or(USAGE)?.to_string();
        options.insert(name, arguments.next().ok_or(USAGE)?);
    }
    let mut take = |name: &str| options.remove(name);
    let mut config = EngineConfig::new(take("model").ok_or(USAGE)?, take("device").ok_or(USAGE)?);
    if let Some(gpus) = take("gpus") {
        config.gpus = gpus.parse()?;
    }
    if let Some(backend) = take("backend") {
        config.backend = backend;
    }
    let listen = take("listen").unwrap_or_else(|| "127.0.0.1:50051".to_string());
    let data = take("data").unwrap_or_else(|| "quail-server-data".to_string());
    if !options.is_empty() {
        return Err(USAGE.into());
    }
    let address = listen.parse()?;
    // a token is a secret, so it comes from the environment
    let token = std::env::var("QUAIL_SERVER_TOKEN")
        .ok()
        .filter(|token| !token.is_empty());
    // the session starts its workers before the server accepts a query
    let session = Arc::new(Session::new(&config)?);
    let queue = Queue::open(std::path::Path::new(&data), Arc::clone(&session))?;
    let service = QuailService::new(session, Arc::clone(&queue));
    println!(
        "Quail Server: {} on {}, at {listen}",
        config.model, config.device
    );
    tokio::runtime::Runtime::new()?.block_on(
        Server::builder()
            .add_service(FlightServiceServer::with_interceptor(
                service,
                move |request| quail_server::check_token(token.as_deref(), request),
            ))
            .serve_with_shutdown(address, stopped()),
    )?;
    queue.close();
    // the session's workers stop, and their sockets go, as it drops
    Ok(())
}

/// Resolves when the server is told to stop: Ctrl-C, or SIGTERM.
async fn stopped() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut terminate = signal(SignalKind::terminate()).expect("a SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = terminate.recv() => {}
    }
}
