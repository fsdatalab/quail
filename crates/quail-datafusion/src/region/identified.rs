//! `IdentifiedTable`: a table whose rows have a content identity.

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::Session;
use datafusion::common::{Constraints, Result};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::logical_expr::{Expr, LogicalPlan, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;

/// A table whose rows are the same whenever it is scanned, named by its
/// content identity, such as a registered Python provider's.
///
/// A model region reads it by name; the runner reads its rows and keeps
/// its token store for later queries under the identity. DataFusion
/// scans it for the operators outside model regions.
pub struct IdentifiedTable {
    inner: Arc<dyn TableProvider>,
    identity: String,
}

impl IdentifiedTable {
    /// `inner`, whose rows `identity` names.
    pub fn new(inner: Arc<dyn TableProvider>, identity: impl Into<String>) -> Self {
        Self {
            inner,
            identity: identity.into(),
        }
    }

    /// The identity of the table's rows.
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The table it marks.
    pub fn inner(&self) -> &Arc<dyn TableProvider> {
        &self.inner
    }
}

impl fmt::Debug for IdentifiedTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IdentifiedTable")
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for IdentifiedTable {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    fn constraints(&self) -> Option<&Constraints> {
        self.inner.constraints()
    }

    fn table_type(&self) -> TableType {
        self.inner.table_type()
    }

    fn get_logical_plan(&self) -> Option<Cow<'_, LogicalPlan>> {
        self.inner.get_logical_plan()
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner.scan(state, projection, filters, limit).await
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        self.inner.supports_filters_pushdown(filters)
    }
}
