//! Quail Server: a Quail [`Session`] over Arrow Flight.
//!
//! A client uploads tables, runs SQL, and reads rows as Arrow:
//!
//! - `DoPut` with the descriptor path `[table]` or `[table, id column]`
//!   registers the uploaded rows as a table. The id column's values are
//!   unique; Quail names each row a model region returns by it.
//! - `DoGet` with a ticket that is a SQL query runs it and returns its
//!   rows.
//! - `DoAction` `explain` with a SQL query returns DataFusion's plan,
//!   with each model region as one operator.
//!
//! One query runs at a time; another waits its turn. The server sends a
//! query's rows once the query has finished.

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

/// The Flight service of one session.
pub struct QuailService {
    session: Arc<Session>,
    num_uploads: AtomicUsize,
}

impl QuailService {
    /// A service that runs every query on `session`.
    pub fn new(session: Arc<Session>) -> Self {
        QuailService {
            session,
            num_uploads: AtomicUsize::new(0),
        }
    }

    /// Run `call` on the session, on a thread that may block.
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

/// A query's failure as a Flight status: the query's own fault is an
/// invalid argument, anything else an internal error.
fn query_status(error: DataFusionError) -> Status {
    match error.find_root() {
        DataFusionError::Plan(_)
        | DataFusionError::SQL(_, _)
        | DataFusionError::SchemaError(_, _) => Status::invalid_argument(error.to_string()),
        _ => Status::internal(error.to_string()),
    }
}

fn text(bytes: &[u8], what: &str) -> Result<String, Status> {
    String::from_utf8(bytes.to_vec())
        .map_err(|_| Status::invalid_argument(format!("{what} is not UTF-8 text")))
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
        let (schema, batches) = self
            .blocking(move |session| session.sql_in_dialect(&query, None))
            .await?;
        let rows = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(stream::iter(batches.into_iter().map(Ok)))
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
        // each upload is new rows, so a table tokenized before is not reused
        let identity = format!(
            "upload:{name}:{}",
            self.num_uploads.fetch_add(1, Ordering::Relaxed)
        );
        let rows = Arc::new(batches);
        let scanned = Arc::clone(&schema);
        let scan: quail::Scan = Arc::new(move |columns: Vec<String>| {
            let indices = columns
                .iter()
                .map(|column| scanned.index_of(column))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows
                .iter()
                .map(|batch| batch.project(&indices))
                .collect::<Result<Vec<_>, _>>()?)
        });
        self.session
            .register_scanned_table(&name, schema, scan, id_column.as_deref(), &identity)
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
        if action.r#type != "explain" {
            return Err(Status::invalid_argument(format!(
                "unknown action {}; the server has explain",
                action.r#type
            )));
        }
        let query = text(&action.body, "the query")?;
        let plan = self
            .blocking(move |session| session.explain(&query, None))
            .await?;
        let found = arrow_flight::Result {
            body: plan.into_bytes().into(),
        };
        Ok(Response::new(stream::once(async { Ok(found) }).boxed()))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let explain = ActionType {
            r#type: "explain".into(),
            description: "Return the plan of a SQL query".into(),
        };
        Ok(Response::new(stream::once(async { Ok(explain) }).boxed()))
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
