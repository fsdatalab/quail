pub mod queue;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::RecordBatch;
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_server::FlightService;
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
};
use datafusion::common::DataFusionError;
use futures::stream::{self, BoxStream};
use futures::{StreamExt, TryStreamExt};
use quail::Session;
use tonic::{Request, Response, Status, Streaming};

pub struct QuailService {
    session: Arc<Session>,
    queue: Arc<queue::Queue>,
    num_uploads: AtomicUsize,
}

impl QuailService {
    pub fn new(session: Arc<Session>, queue: Arc<queue::Queue>) -> Self {
        QuailService {
            session,
            queue,
            num_uploads: AtomicUsize::new(0),
        }
    }

    async fn blocking<T: Send + 'static>(
        &self,
        call: impl FnOnce(&Session) -> datafusion::common::Result<T> + Send + 'static,
    ) -> Result<T, Status> {
        let session = Arc::clone(&self.session);
        tokio::task::spawn_blocking(move || call(&session))
            .await
            .map_err(|error| Status::internal(error.to_string()))?
            .map_err(query_status)
    }
}

impl QuailService {
    async fn streamed_submission(
        &self,
        id: String,
    ) -> Result<(arrow::datatypes::SchemaRef, Rows), Status> {
        if self.queue.status(&id).is_none() {
            return Err(Status::not_found(format!("no submission {id}")));
        }
        let queue = Arc::clone(&self.queue);
        let asked = id.clone();
        let schema = tokio::task::spawn_blocking(move || {
            loop {
                if let (Some(schema), _, _) = queue.rows_after(&asked, 0, STREAM_WAIT)? {
                    return Ok::<_, String>(schema);
                }
            }
        })
        .await
        .map_err(|error| Status::internal(error.to_string()))?
        .map_err(Status::aborted)?;
        let queue = Arc::clone(&self.queue);
        let rows = produced(move |send| {
            let mut sent = 0;
            loop {
                let (batches, done) = match queue.rows_after(&id, sent, STREAM_WAIT) {
                    Ok((_, batches, done)) => (batches, done),
                    Err(error) => {
                        send(Err(FlightError::ExternalError(error.into())));
                        return;
                    }
                };
                for batch in batches {
                    sent += batch.num_rows();
                    if !send(Ok(batch)) {
                        return;
                    }
                }
                if done {
                    return;
                }
            }
        });
        Ok((schema, rows))
    }
}

type Rows = BoxStream<'static, Result<RecordBatch, FlightError>>;

const STREAM_WAIT: std::time::Duration = std::time::Duration::from_secs(1);

fn produced(
    produce: impl FnOnce(&mut dyn FnMut(Result<RecordBatch, FlightError>) -> bool) + Send + 'static,
) -> Rows {
    let (mut sender, received) = futures::channel::mpsc::channel(2);
    tokio::task::spawn_blocking(move || {
        produce(&mut |batch| {
            futures::executor::block_on(futures::SinkExt::send(&mut sender, batch)).is_ok()
        });
    });
    received.boxed()
}

fn query_status(error: DataFusionError) -> Status {
    match error.find_root() {
        DataFusionError::Plan(_)
        | DataFusionError::SQL(_, _)
        | DataFusionError::SchemaError(_, _) => Status::invalid_argument(error.to_string()),
        _ => Status::internal(error.to_string()),
    }
}

pub fn check_token<T>(token: Option<&str>, request: Request<T>) -> Result<Request<T>, Status> {
    let Some(token) = token else {
        return Ok(request);
    };
    let sent = (request.metadata().get("authorization")).and_then(|sent| sent.to_str().ok());
    match sent.and_then(|sent| sent.strip_prefix("Bearer ")) == Some(token) {
        true => Ok(request),
        false => Err(Status::unauthenticated("this server needs its token")),
    }
}

const ACTIONS: [(&str, &str); 7] = [
    (
        "explain",
        "Return a query's plan; the body is JSON with sql, dialect",
    ),
    (
        "submit",
        "Queue a query; the body is JSON with sql, dialect, id, timeout_s",
    ),
    (
        "status",
        "Return a submission's status, by id or JSON with id, after, wait_s",
    ),
    (
        "answers",
        "Return a run's answer events; the body is JSON with id, after, limit",
    ),
    (
        "report",
        "Return each region's report of a submission, by id",
    ),
    ("cancel", "Stop a submission, by id"),
    ("list", "Return every submission"),
];

fn text(bytes: &[u8], what: &str) -> Result<String, Status> {
    String::from_utf8(bytes.to_vec())
        .map_err(|_| Status::invalid_argument(format!("{what} is not UTF-8 text")))
}

fn asked_query(body: &str) -> Result<(String, Option<String>), Status> {
    let asked: serde_json::Value =
        serde_json::from_str(body).map_err(|error| Status::invalid_argument(error.to_string()))?;
    let sql =
        (asked["sql"].as_str()).ok_or_else(|| Status::invalid_argument("a query needs sql"))?;
    let dialect = asked["dialect"].as_str().map(str::to_string);
    Ok((sql.to_string(), dialect))
}

#[tonic::async_trait]
impl FlightService for QuailService {
    type HandshakeStream = BoxStream<'static, Result<HandshakeResponse, Status>>;
    type ListFlightsStream = BoxStream<'static, Result<FlightInfo, Status>>;
    type DoGetStream = BoxStream<'static, Result<FlightData, Status>>;
    type DoPutStream = BoxStream<'static, Result<PutResult, Status>>;
    type DoActionStream = BoxStream<'static, Result<arrow_flight::Result, Status>>;
    type ListActionsStream = BoxStream<'static, Result<ActionType, Status>>;
    type DoExchangeStream = BoxStream<'static, Result<FlightData, Status>>;

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let query = text(&request.into_inner().ticket, "the ticket")?;
        let submitted = serde_json::from_str::<serde_json::Value>(&query)
            .ok()
            .and_then(|ticket| {
                let id = ticket["query_id"].as_str()?.to_string();
                let streams = ticket["stream"].as_bool() == Some(true);
                Some((id, ticket["table"].as_u64(), streams))
            });
        let (schema, batches) = match submitted {
            Some((id, _, true)) => self.streamed_submission(id).await?,
            Some((id, None, false)) => {
                let (schema, batches) = self.queue.rows(&id).map_err(Status::not_found)?;
                (schema, stream::iter(batches.into_iter().map(Ok)).boxed())
            }
            Some((id, Some(index), false)) => {
                let (schema, batches) = (self.queue)
                    .table(&id, index as usize)
                    .map_err(Status::not_found)?;
                (schema, stream::iter(batches.into_iter().map(Ok)).boxed())
            }
            None => {
                let turn = self.queue.turn().await;
                let rows = self
                    .blocking(move |session| session.sql_stream(&query, None))
                    .await?;
                let schema = rows.schema();
                let batches = produced(move |send| {
                    let _turn = turn;
                    for batch in rows {
                        let batch =
                            batch.map_err(|error| FlightError::ExternalError(Box::new(error)));
                        if !send(batch) {
                            return;
                        }
                    }
                });
                (schema, batches)
            }
        };
        let rows = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(batches)
            .map_err(Status::from);
        Ok(Response::new(rows.boxed()))
    }

    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        let mut uploaded = request.into_inner();
        let first = uploaded
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("the upload is empty"))?;
        let path = match &first.flight_descriptor {
            Some(descriptor) if !descriptor.path.is_empty() => descriptor.path.clone(),
            _ => {
                return Err(Status::invalid_argument(
                    "an upload's descriptor path names its table",
                ));
            }
        };
        let (name, id_column) = (path[0].clone(), path.get(1).cloned());
        let messages = stream::once(async { Ok(first) })
            .chain(uploaded.map_err(FlightError::from))
            .boxed();
        let mut decoded = FlightRecordBatchStream::new_from_flight_data(messages);
        let mut batches: Vec<RecordBatch> = Vec::new();
        while let Some(batch) = decoded.try_next().await.map_err(Status::from)? {
            batches.push(batch);
        }
        let schema = decoded
            .schema()
            .cloned()
            .ok_or_else(|| Status::invalid_argument("the upload has no schema"))?;
        if let Some(column) = &id_column
            && schema.index_of(column).is_err()
        {
            return Err(Status::invalid_argument(format!(
                "the table has no id column {column}"
            )));
        }
        let identity = format!(
            "upload:{name}:{}",
            self.num_uploads.fetch_add(1, Ordering::Relaxed)
        );
        let table = datafusion::datasource::MemTable::try_new(schema, vec![batches])
            .map_err(query_status)?;
        self.session
            .register_scanned_table(&name, Arc::new(table), id_column.as_deref(), &identity)
            .map_err(query_status)?;
        Ok(Response::new(
            stream::once(async { Ok(PutResult::default()) }).boxed(),
        ))
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        let body = text(&action.body, "the action's body")?;
        let json = |value: serde_json::Value| value.to_string();
        let answer = match action.r#type.as_str() {
            "explain" => {
                let (sql, dialect) = asked_query(&body)?;
                self.blocking(move |session| session.explain(&sql, dialect.as_deref()))
                    .await?
            }
            "submit" => {
                let (sql, dialect) = asked_query(&body)?;
                let asked: serde_json::Value = serde_json::from_str(&body)
                    .map_err(|error| Status::invalid_argument(error.to_string()))?;
                let id = asked["id"].as_str().map(str::to_string);
                let found = (self.queue)
                    .submit(&sql, dialect, id, asked["timeout_s"].as_f64())
                    .map_err(Status::invalid_argument)?;
                json(serde_json::json!(found))
            }
            "status" => {
                let asked = serde_json::from_str::<serde_json::Value>(&body)
                    .ok()
                    .filter(serde_json::Value::is_object);
                let found = match asked {
                    None => self.queue.status(&body),
                    Some(asked) => {
                        let queue = Arc::clone(&self.queue);
                        let wait = asked["wait_s"].as_f64().unwrap_or(0.0).clamp(0.0, 60.0);
                        tokio::task::spawn_blocking(move || {
                            queue.status_after(
                                asked["id"].as_str().unwrap_or_default(),
                                asked["after"].as_u64().unwrap_or(0),
                                std::time::Duration::from_secs_f64(wait),
                            )
                        })
                        .await
                        .map_err(|error| Status::internal(error.to_string()))?
                    }
                };
                json(found.ok_or_else(|| Status::not_found("no such submission"))?)
            }
            "answers" => {
                let asked: serde_json::Value = serde_json::from_str(&body)
                    .map_err(|error| Status::invalid_argument(error.to_string()))?;
                let found = self.queue.answers(
                    asked["id"].as_str().unwrap_or_default(),
                    asked["after"].as_u64().unwrap_or(0) as usize,
                    asked["limit"].as_u64().unwrap_or(1000) as usize,
                );
                json(found.map_err(Status::not_found)?)
            }
            "report" => self.queue.report(&body).map_err(Status::not_found)?,
            "cancel" => json(serde_json::json!(
                self.queue.cancel(&body).map_err(Status::not_found)?
            )),
            "list" => json(serde_json::json!(self.queue.list())),
            other => {
                return Err(Status::invalid_argument(format!(
                    "unknown action {other}; the server has {}",
                    ACTIONS.map(|(name, _)| name).join(", ")
                )));
            }
        };
        let found = arrow_flight::Result {
            body: answer.into_bytes().into(),
        };
        Ok(Response::new(stream::once(async { Ok(found) }).boxed()))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let actions = ACTIONS.map(|(name, description)| {
            Ok(ActionType {
                r#type: name.into(),
                description: description.into(),
            })
        });
        Ok(Response::new(stream::iter(actions).boxed()))
    }

    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        Err(Status::unimplemented("the server has no handshake"))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Err(Status::unimplemented("the server lists no flights"))
    }

    async fn get_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("run a query with DoGet"))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("run a query with DoGet"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("run a query with DoGet"))
    }

    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("the server has no exchange"))
    }
}

const MAX_UPLOAD_MESSAGE_BYTES: usize = 1 << 30;

pub const USAGE: &str = "usage: quail-server --model <model> --device <device> \
[--gpus <n>] [--backend <backend>] [--listen <address>] [--data <directory>]";

pub fn run(
    arguments: impl IntoIterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut options = std::collections::HashMap::new();
    let mut arguments = arguments.into_iter();
    while let Some(name) = arguments.next() {
        let name = name.strip_prefix("--").ok_or(USAGE)?.to_string();
        options.insert(name, arguments.next().ok_or(USAGE)?);
    }
    let mut take = |name: &str| options.remove(name);
    let mut config =
        quail::EngineConfig::new(take("model").ok_or(USAGE)?, take("device").ok_or(USAGE)?);
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
    let token = std::env::var("QUAIL_SERVER_TOKEN")
        .ok()
        .filter(|token| !token.is_empty());
    let session = Arc::new(Session::new(&config)?);
    let queue = queue::Queue::open(std::path::Path::new(&data), Arc::clone(&session))?;
    let service = QuailService::new(session, Arc::clone(&queue));
    println!(
        "Quail Server: {} on {}, at {listen}",
        config.model, config.device
    );
    quail::runtime()?.block_on(
        tonic::transport::Server::builder()
            .add_service(tonic::service::interceptor::InterceptedService::new(
                arrow_flight::flight_service_server::FlightServiceServer::new(service)
                    .max_decoding_message_size(MAX_UPLOAD_MESSAGE_BYTES),
                move |request| check_token(token.as_deref(), request),
            ))
            .serve_with_shutdown(address, stopped()),
    )?;
    queue.close();
    Ok(())
}

async fn stopped() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut terminate = signal(SignalKind::terminate()).expect("a SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = terminate.recv() => {}
    }
}
